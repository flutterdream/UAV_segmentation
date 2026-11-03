# -*- coding: utf-8 -*-
"""
P1.1: 用 16k 冠军 + ms2(8路TTA) 为 test1 生成 伪标签+逐像素置信度.
产物: pl_cache/<name>.npz {pred:uint8, conf:float16} (1024x1024), 可断点续跑.
用法: python pl_gen.py [--ckpt ckpt_kannoise_i16000.pth] [--cache pl_cache_noise]
"""
import argparse
import hashlib
import json
import os
import numpy as np
from PIL import Image
import torch

import seg_swin as S
import kan_swin as K
import strict_env as P

ROOT = os.path.dirname(os.path.abspath(__file__))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="ckpt_kan_swin_16k.pth")
    ap.add_argument("--cache", type=str, default="pl_cache")
    a = ap.parse_args()
    if os.path.basename(a.ckpt) != a.ckpt or os.path.basename(a.cache) != a.cache:
        raise ValueError("--ckpt and --cache must be plain names inside strict run root")
    ckpt_path = str(P.assert_run_path(os.path.join(P.RUN_ROOT, a.ckpt), must_exist=True))
    OUT = str(P.assert_run_path(os.path.join(P.RUN_ROOT, a.cache)))
    os.makedirs(OUT, exist_ok=True)
    manifest_path = os.path.join(OUT, "_CACHE_MANIFEST.json")
    expected_manifest = {"checkpoint": a.ckpt, "checkpoint_sha256": sha256(ckpt_path),
                         "route": "ms2", "source": "official_test1", "expected_images": 500}
    if os.path.exists(manifest_path):
        actual_manifest = json.load(open(manifest_path, encoding="utf-8"))
        if actual_manifest != expected_manifest:
            raise RuntimeError("pseudo-label cache provenance mismatch")
    else:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(expected_manifest, f, ensure_ascii=False, indent=2)
    checkpoint_payload = P.load_checkpoint(ckpt_path)
    model, iters = K.load_trained(use_ema=True, path=ckpt_path)
    model.eval()
    files = sorted(f for f in os.listdir(S.TEST_DIR) if f.endswith(".png"))
    if len(files) != 500:
        raise RuntimeError(f"official test1 must contain 500 PNG files, got {len(files)}")
    print(f"[pl] {os.path.basename(a.ckpt)} EMA iters={iters}, {len(files)} 张 test1, route=ms2")
    thresholds = (0.8, 0.85, 0.9, 0.95)
    retained = {t: 0 for t in thresholds}
    total_pixels = 0
    for n, fn in enumerate(files):
        fp = os.path.join(OUT, fn.replace(".png", ".npz"))
        if os.path.exists(fp):
            conf_np = np.load(fp)["conf"].astype(np.float32)
        else:
            img = np.array(Image.open(os.path.join(S.TEST_DIR, fn)).convert("RGB"))
            lg = S.predict_logits(model, img, tta="ms2")          # (9,H,W) cpu float32
            p = torch.softmax(lg, dim=0)
            conf, pred = p.max(dim=0)
            conf_np = conf.half().numpy()
            np.savez_compressed(fp, pred=pred.numpy().astype(np.uint8), conf=conf_np)
        total_pixels += int(conf_np.size)
        for threshold in thresholds:
            retained[threshold] += int(np.count_nonzero(conf_np.astype(np.float32) >= threshold))
        if (n + 1) % 25 == 0:
            print(f"  pl {n+1}/{len(files)}", flush=True)
    report = {
        "checkpoint": a.ckpt,
        "checkpoint_sha256": expected_manifest["checkpoint_sha256"],
        "checkpoint_stage": checkpoint_payload["strict_repro"]["stage"],
        "checkpoint_iter": checkpoint_payload["iter"],
        "route": "ms2",
        "official_test1_images": len(files),
        "coverage_scope": "all pixels of all 500 images",
        "total_pixels": total_pixels,
        "coverage": {str(t): retained[t] / total_pixels for t in thresholds},
        "retained_pixels": {str(t): retained[t] for t in thresholds},
    }
    report_path = os.path.join(OUT, "_CONFIDENCE_REPORT.json")
    temporary = report_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    os.replace(temporary, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("[pl done]")


if __name__ == "__main__":
    main()
