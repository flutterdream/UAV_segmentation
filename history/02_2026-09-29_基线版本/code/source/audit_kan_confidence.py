# -*- coding: utf-8 -*-
"""Read-only confidence-quality audit for the selected KAN16k EMA and PL cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

import strict_env as P
import seg_swin as S
import kan_swin as K


CKPT_NAME = "ckpt_kan_from_v4_1_7439_16k.pth"
CKPT_SHA = "806039602f21cb6da342085f867727712a9c06f40c8765c031791c07a5882798"
CACHE_NAME = "pl_cache_kan_from_v4_1_7439_ms2"
TAU = 0.85
CLASS_LABELS = ["class0_void_or_ignore", *S.CLASS_NAMES]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite audit output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def class_rows(total: np.ndarray, retained: np.ndarray,
               correct: np.ndarray | None = None) -> dict:
    rows = {}
    for class_id, label in enumerate(CLASS_LABELS):
        row = {
            "class_id": class_id,
            "meaning": label,
            "pixels": int(total[class_id]),
            "retained_pixels": int(retained[class_id]),
            "coverage": ratio(int(retained[class_id]), int(total[class_id]))
        }
        if correct is not None:
            row["correct_retained_pixels"] = int(correct[class_id])
            row["retained_accuracy"] = ratio(int(correct[class_id]), int(retained[class_id]))
        rows[str(class_id)] = row
    return rows


def audit_test1_cache(output: Path, ckpt_hash: str) -> dict:
    cache = P.assert_run_path(P.RUN_ROOT / CACHE_NAME)
    if not cache.is_dir():
        raise FileNotFoundError(cache)
    manifest = json.loads((cache / "_CACHE_MANIFEST.json").read_text(encoding="utf-8"))
    source_report = json.loads((cache / "_CONFIDENCE_REPORT.json").read_text(encoding="utf-8"))
    expected_manifest = {
        "checkpoint": CKPT_NAME,
        "checkpoint_sha256": ckpt_hash,
        "route": "ms2",
        "source": "official_test1",
        "expected_images": 500
    }
    if manifest != expected_manifest:
        raise RuntimeError("test1 PL cache manifest does not match the selected KAN/ms2 lineage")
    files = sorted(cache.glob("*.npz"))
    if len(files) != 500:
        raise RuntimeError(f"expected 500 test1 cache files, got {len(files)}")

    total = np.zeros(9, dtype=np.int64)
    retained = np.zeros(9, dtype=np.int64)
    confidence_sum = np.zeros(9, dtype=np.float64)
    for index, path in enumerate(files, 1):
        with np.load(path) as item:
            pred = item["pred"].astype(np.int64, copy=False)
            conf = item["conf"].astype(np.float32)
        if pred.shape != conf.shape:
            raise RuntimeError(f"pred/conf shape mismatch: {path.name}")
        high = conf >= TAU
        total += np.bincount(pred.ravel(), minlength=9)[:9]
        retained += np.bincount(pred[high].ravel(), minlength=9)[:9]
        confidence_sum += np.bincount(pred.ravel(), weights=conf.ravel(), minlength=9)[:9]
        if index % 50 == 0:
            print(f"test1-cache {index}/500", flush=True)

    all_pixels = int(total.sum())
    retained_pixels = int(retained.sum())
    if all_pixels != int(source_report["total_pixels"]):
        raise RuntimeError("test1 cache pixel count differs from original confidence report")
    if retained_pixels != int(source_report["retained_pixels"]["0.85"]):
        raise RuntimeError("test1 tau=0.85 retained count differs from original confidence report")
    rows = class_rows(total, retained)
    for class_id, row in rows.items():
        index = int(class_id)
        row["share_of_all_pixels"] = ratio(int(total[index]), all_pixels)
        row["share_of_retained_pixels"] = ratio(int(retained[index]), retained_pixels)
        row["mean_confidence"] = ratio(float(confidence_sum[index]), int(total[index]))
    result = {
        "status": "read_only_local_audit",
        "scope": "existing test1 cache; all pixels of all 500 official images",
        "checkpoint": CKPT_NAME,
        "checkpoint_sha256": ckpt_hash,
        "route": "ms2",
        "tau": TAU,
        "source_confidence_report_sha256": sha256(cache / "_CONFIDENCE_REPORT.json"),
        "images": len(files),
        "all_pixels": all_pixels,
        "retained_pixels": retained_pixels,
        "all_pixel_coverage": retained_pixels / all_pixels,
        "by_predicted_class": rows,
        "class_note": "class0 is void/ignore; class1 is the scored Background class"
    }
    atomic_json(output, result)
    print(f"TEST1_AUDIT_WRITTEN={output}", flush=True)
    return result


def empty_accumulators() -> dict:
    return {
        "all_pixels": 0, "retained_all": 0, "correct_retained_all": 0,
        "valid_pixels": 0, "retained_valid": 0, "correct_retained_valid": 0,
        "true_total": np.zeros(9, dtype=np.int64),
        "true_retained": np.zeros(9, dtype=np.int64),
        "true_correct": np.zeros(9, dtype=np.int64),
        "pred_total": np.zeros(9, dtype=np.int64),
        "pred_retained": np.zeros(9, dtype=np.int64),
        "pred_correct": np.zeros(9, dtype=np.int64),
        "region": {
            "boundary": {"all": 0, "retained_all": 0, "correct_all": 0,
                         "valid": 0, "retained_valid": 0, "correct_valid": 0},
            "non_boundary": {"all": 0, "retained_all": 0, "correct_all": 0,
                             "valid": 0, "retained_valid": 0, "correct_valid": 0}
        }
    }


def add_validation_image(acc: dict, pred: np.ndarray, conf: np.ndarray,
                         mask: np.ndarray) -> None:
    high = conf >= TAU
    valid = (mask != 0) & (mask != S.IGNORE)
    correct = pred == mask
    acc["all_pixels"] += int(mask.size)
    acc["retained_all"] += int(high.sum())
    acc["correct_retained_all"] += int((high & correct).sum())
    acc["valid_pixels"] += int(valid.sum())
    acc["retained_valid"] += int((high & valid).sum())
    acc["correct_retained_valid"] += int((high & valid & correct).sum())

    mask_for_count = np.where((mask >= 0) & (mask <= 8), mask, 0).astype(np.int64)
    acc["true_total"] += np.bincount(mask_for_count.ravel(), minlength=9)[:9]
    acc["true_retained"] += np.bincount(mask_for_count[high].ravel(), minlength=9)[:9]
    acc["true_correct"] += np.bincount(mask_for_count[high & correct].ravel(), minlength=9)[:9]
    acc["pred_total"] += np.bincount(pred.ravel(), minlength=9)[:9]
    acc["pred_retained"] += np.bincount(pred[high].ravel(), minlength=9)[:9]
    acc["pred_correct"] += np.bincount(pred[high & correct].ravel(), minlength=9)[:9]

    kernel = np.ones((5, 5), np.uint8)
    boundary = cv2.dilate(mask, kernel) != cv2.erode(mask, kernel)
    for name, region_mask in (("boundary", boundary), ("non_boundary", ~boundary)):
        row = acc["region"][name]
        row["all"] += int(region_mask.sum())
        row["retained_all"] += int((region_mask & high).sum())
        row["correct_all"] += int((region_mask & high & correct).sum())
        row["valid"] += int((region_mask & valid).sum())
        row["retained_valid"] += int((region_mask & valid & high).sum())
        row["correct_valid"] += int((region_mask & valid & high & correct).sum())


def validation_result(acc: dict, n: int, ckpt_hash: str) -> dict:
    regions = {}
    for name, raw in acc["region"].items():
        regions[name] = {
            **raw,
            "all_pixel_coverage": ratio(raw["retained_all"], raw["all"]),
            "retained_all_pixel_accuracy": ratio(raw["correct_all"], raw["retained_all"]),
            "valid_label_coverage": ratio(raw["retained_valid"], raw["valid"]),
            "retained_valid_label_accuracy": ratio(raw["correct_valid"], raw["retained_valid"])
        }
    result = {
        "status": "read_only_local_audit",
        "checkpoint": CKPT_NAME,
        "checkpoint_sha256": ckpt_hash,
        "checkpoint_iter": 16000,
        "checkpoint_stage": "kan_from_v4_1_7439_16k",
        "route": "ms2",
        "tau": TAU,
        "split": f"seed123_fixed_validation_first{n}",
        "images": n,
        "decision_accuracy_scope": "retained pixels with ground-truth labels 1..8",
        "boundary_definition": "5x5 cv2.dilate(mask) != cv2.erode(mask), matching training boundary band",
        "overall": {
            "all_pixels": acc["all_pixels"],
            "retained_all_pixels": acc["retained_all"],
            "all_pixel_coverage": ratio(acc["retained_all"], acc["all_pixels"]),
            "correct_retained_all_pixels": acc["correct_retained_all"],
            "retained_all_pixel_accuracy": ratio(acc["correct_retained_all"], acc["retained_all"]),
            "valid_label_pixels_mask_nonzero": acc["valid_pixels"],
            "retained_valid_label_pixels": acc["retained_valid"],
            "valid_label_coverage": ratio(acc["retained_valid"], acc["valid_pixels"]),
            "correct_retained_valid_label_pixels": acc["correct_retained_valid"],
            "retained_valid_label_accuracy": ratio(acc["correct_retained_valid"], acc["retained_valid"])
        },
        "by_true_class": class_rows(acc["true_total"], acc["true_retained"], acc["true_correct"]),
        "by_predicted_class": class_rows(acc["pred_total"], acc["pred_retained"], acc["pred_correct"]),
        "boundary_vs_non_boundary": regions,
        "class_note": "class0 is void/ignore; class1 is the scored Background class"
    }
    return result


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-n", type=int, choices=[100, 500], default=500)
    args = parser.parse_args()
    P.verify_inputs()
    checkpoint = P.assert_run_path(P.RUN_ROOT / CKPT_NAME, must_exist=True)
    ckpt_hash = sha256(checkpoint)
    if ckpt_hash.lower() != CKPT_SHA:
        raise RuntimeError("selected KAN16k checkpoint hash changed")

    test1_output = P.assert_run_path(P.RUN_ROOT / "audit_kan16k_ms2_test1_predicted_class_n500.json")
    n100_output = P.assert_run_path(P.RUN_ROOT / "audit_kan16k_ms2_confidence_quality_n100.json")
    n500_output = P.assert_run_path(P.RUN_ROOT / "audit_kan16k_ms2_confidence_quality_n500.json")
    outputs = [test1_output, n100_output] + ([n500_output] if args.max_n == 500 else [])
    if any(path.exists() for path in outputs):
        raise RuntimeError("one or more audit outputs already exist; refusing overwrite")

    audit_test1_cache(test1_output, ckpt_hash)
    model, iteration = K.load_trained(use_ema=True, path=str(checkpoint))
    if iteration != 16000:
        raise RuntimeError("selected KAN checkpoint is not iter16000")
    payload = P.load_checkpoint(checkpoint, map_location="cpu")
    if payload["strict_repro"].get("stage") != "kan_from_v4_1_7439_16k":
        raise RuntimeError("selected KAN checkpoint stage mismatch")

    val_ids = json.loads((Path(S.ROOT) / "split.json").read_text(encoding="utf-8"))["val_ids"]
    acc = empty_accumulators()
    for index, image_id in enumerate(val_ids[:args.max_n], 1):
        image = np.array(Image.open(Path(S.IMG_DIR) / f"{image_id:04d}.png").convert("RGB"))
        mask = np.array(Image.open(Path(S.MSK_DIR) / f"{image_id:04d}.png"))
        logits = S.predict_logits(model, image, tta="ms2")
        probability = torch.softmax(logits, dim=0)
        confidence, prediction = probability.max(dim=0)
        add_validation_image(acc, prediction.numpy().astype(np.int64),
                             confidence.numpy().astype(np.float32), mask)
        if index % 10 == 0:
            print(f"validation-ms2 {index}/{args.max_n}", flush=True)
        if index == 100:
            result100 = validation_result(acc, 100, ckpt_hash)
            atomic_json(n100_output, result100)
            print(f"N100_AUDIT_WRITTEN={n100_output}", flush=True)
    if args.max_n == 500:
        result500 = validation_result(acc, 500, ckpt_hash)
        atomic_json(n500_output, result500)
        print(f"N500_AUDIT_WRITTEN={n500_output}", flush=True)
    print("CONFIDENCE_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
