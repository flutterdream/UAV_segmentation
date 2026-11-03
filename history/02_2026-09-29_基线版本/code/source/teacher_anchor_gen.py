# -*- coding: utf-8 -*-
"""Generate official-test1 hard anchors from this run's independent teacher."""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
from PIL import Image
import torch

import strict_env as P
import seg_swin as S


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="strict run root 内独立教师断点名")
    parser.add_argument("--out", default="teacher_anchors_ms2")
    args = parser.parse_args()
    if os.path.basename(args.ckpt) != args.ckpt or os.path.basename(args.out) != args.out:
        raise ValueError("--ckpt/--out must be plain names inside strict run root")
    checkpoint = str(P.assert_run_path(P.RUN_ROOT / args.ckpt, must_exist=True))
    payload = P.load_checkpoint(checkpoint)
    stage = str(payload["strict_repro"].get("stage", ""))
    if not stage.startswith("teacher_"):
        raise RuntimeError(f"anchor source is not registered as an independent teacher: {stage}")
    out_dir = str(P.assert_run_path(P.RUN_ROOT / args.out))
    os.makedirs(out_dir, exist_ok=True)
    expected = {
        "teacher_checkpoint": args.ckpt,
        "teacher_checkpoint_sha256": sha256(checkpoint),
        "teacher_stage": stage,
        "teacher_seed": payload["strict_repro"].get("seed"),
        "route": "ms2",
        "source": "official_test1",
        "expected_images": 500
    }
    manifest_path = os.path.join(out_dir, "_ANCHOR_MANIFEST.json")
    if os.path.exists(manifest_path):
        if json.load(open(manifest_path, encoding="utf-8")) != expected:
            raise RuntimeError("teacher-anchor provenance mismatch")
    else:
        with open(manifest_path, "w", encoding="utf-8") as stream:
            json.dump(expected, stream, ensure_ascii=False, indent=2)

    model, iters = S.load_trained(path=checkpoint)
    model.eval()
    confidence = []
    files = sorted(f for f in os.listdir(S.TEST_DIR) if f.endswith(".png"))
    for index, filename in enumerate(files):
        output_path = os.path.join(out_dir, filename)
        image = np.asarray(Image.open(os.path.join(S.TEST_DIR, filename)).convert("RGB"))
        logits = S.predict_logits(model, image, tta="ms2")
        probability = torch.softmax(logits, dim=0)
        conf, pred = probability.max(dim=0)
        if not os.path.exists(output_path):
            Image.fromarray(pred.numpy().astype(np.uint8), mode="L").save(output_path)
        confidence.append(conf[::4, ::4].half().numpy().reshape(-1))
        if (index + 1) % 25 == 0:
            print(f"anchor {index + 1}/{len(files)}", flush=True)
    sampled = np.concatenate(confidence).astype(np.float32)
    report = {
        "teacher_iters": iters,
        "sample_stride": 4,
        "quantiles": {str(q): float(np.quantile(sampled, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "coverage": {str(t): float((sampled >= t).mean()) for t in (0.8, 0.85, 0.9, 0.95)}
    }
    with open(os.path.join(out_dir, "_CONFIDENCE_REPORT.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

