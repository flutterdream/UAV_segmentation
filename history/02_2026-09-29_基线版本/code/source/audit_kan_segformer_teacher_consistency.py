# -*- coding: utf-8 -*-
"""Read-only ms2 consistency audit between one KAN EMA and rebuilt SegFormer anchors."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

import strict_env as P
import seg_swin as S
import kan_swin as K


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--anchor-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-stage", required=True)
    parser.add_argument("--expected-iter", type=int, required=True)
    args = parser.parse_args()
    for value in (args.ckpt, args.anchor_dir, args.output):
        if os.path.basename(value) != value:
            raise ValueError("ckpt/anchor-dir/output must be plain names inside strict run root")

    checkpoint = P.assert_run_path(P.RUN_ROOT / args.ckpt, must_exist=True)
    anchor_dir = P.assert_run_path(P.RUN_ROOT / args.anchor_dir)
    output = P.assert_run_path(P.RUN_ROOT / args.output)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite: {output}")
    if not anchor_dir.is_dir():
        raise FileNotFoundError(anchor_dir)
    payload = P.load_checkpoint(checkpoint, map_location="cpu")
    metadata = payload["strict_repro"]
    if payload.get("iter") != args.expected_iter or metadata.get("stage") != args.expected_stage:
        raise RuntimeError("checkpoint stage/iter mismatch")
    if payload.get("ema") is None:
        raise RuntimeError("checkpoint has no EMA")

    test_files = sorted(S.TEST_DIR_PATH.glob("*.png")) if hasattr(S, "TEST_DIR_PATH") else sorted(Path(S.TEST_DIR).glob("*.png"))
    anchor_files = sorted(anchor_dir.glob("*.png"))
    if len(test_files) != 500 or {p.name for p in test_files} != {p.name for p in anchor_files}:
        raise RuntimeError("anchor/test1 file set mismatch")

    model, iteration = K.load_trained(use_ema=True, path=str(checkpoint))
    model.eval()
    confusion = np.zeros((8, 9), dtype=np.int64)
    teacher_pixels = np.zeros(9, dtype=np.int64)
    student_pixels = np.zeros(9, dtype=np.int64)
    identical_files = 0
    matching_pixels = 0
    total_pixels = 0
    for index, image_path in enumerate(test_files, start=1):
        image = np.asarray(Image.open(image_path).convert("RGB"))
        teacher = np.asarray(Image.open(anchor_dir / image_path.name))
        if teacher.shape != image.shape[:2] or teacher.ndim != 2:
            raise RuntimeError(f"invalid teacher anchor shape: {image_path.name}: {teacher.shape}")
        if teacher.min() < 1 or teacher.max() > 8:
            raise RuntimeError(f"teacher labels outside 1..8: {image_path.name}")
        student = S.predict_logits(model, image, tta="ms2", bias=None).argmax(0).numpy().astype(np.uint8)
        equal = student == teacher
        identical_files += int(bool(equal.all()))
        matching_pixels += int(equal.sum())
        total_pixels += int(equal.size)
        teacher_pixels += np.bincount(teacher.ravel(), minlength=9)[:9]
        student_pixels += np.bincount(student.ravel(), minlength=9)[:9]
        packed = (teacher.astype(np.int64) - 1) * 9 + student.astype(np.int64)
        confusion += np.bincount(packed.ravel(), minlength=72).reshape(8, 9)
        if index % 25 == 0:
            print(f"consistency={index}/500", flush=True)

    per_class = {}
    ious = []
    for teacher_label in range(1, 9):
        row = teacher_label - 1
        intersection = int(confusion[row, teacher_label])
        union = int(teacher_pixels[teacher_label] + student_pixels[teacher_label] - intersection)
        iou = intersection / union if union else float("nan")
        agreement = intersection / teacher_pixels[teacher_label] if teacher_pixels[teacher_label] else float("nan")
        if not math.isfinite(iou) or not math.isfinite(agreement):
            raise RuntimeError(f"non-finite class consistency for class {teacher_label}")
        ious.append(iou)
        per_class[str(teacher_label)] = {
            "teacher_pixels": int(teacher_pixels[teacher_label]),
            "student_pixels": int(student_pixels[teacher_label]),
            "matching_pixels": intersection,
            "teacher_conditioned_agreement": agreement,
            "iou_between_teacher_and_student": iou
        }
    result = {
        "status": "local_consistency_audit_only",
        "checkpoint": args.ckpt,
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_iter": iteration,
        "strict_repro": metadata,
        "anchor_dir": args.anchor_dir,
        "route": "ms2",
        "bias": False,
        "ema": True,
        "files": 500,
        "identical_files": identical_files,
        "matching_pixels": matching_pixels,
        "total_pixels": total_pixels,
        "pixel_agreement": matching_pixels / total_pixels,
        "miou_between_teacher_and_student": float(np.mean(ious)),
        "student_prediction_pixels": {str(i): int(student_pixels[i]) for i in range(9)},
        "student_prediction_fraction": {str(i): float(student_pixels[i] / total_pixels) for i in range(9)},
        "teacher_prediction_pixels": {str(i): int(teacher_pixels[i]) for i in range(1, 9)},
        "teacher_prediction_fraction": {str(i): float(teacher_pixels[i] / total_pixels) for i in range(1, 9)},
        "per_teacher_class": per_class,
        "confusion_teacher_rows_1_to_8_student_columns_0_to_8": confusion.tolist()
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print("CONSISTENCY_RESULT=" + json.dumps({
        "pixel_agreement": result["pixel_agreement"],
        "miou": result["miou_between_teacher_and_student"],
        "identical_files": identical_files
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
