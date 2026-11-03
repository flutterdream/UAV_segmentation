# -*- coding: utf-8 -*-
"""
P1.1: 用 16k 冠军 + ms2(8路TTA) 为 test1 生成 伪标签+逐像素置信度.
产物: pl_cache/<name>.npz {pred:uint8, conf:float16} (1024x1024), 可断点续跑.
用法: python pl_gen.py [--ckpt ckpt_kannoise_i16000.pth] [--cache pl_cache_noise]
"""
import argparse
import os
import numpy as np
from PIL import Image
import torch

import seg_swin as S
import kan_swin as K

ROOT = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="ckpt_kan_swin_16k.pth")
    ap.add_argument("--cache", type=str, default="pl_cache")
    a = ap.parse_args()
    OUT = os.path.join(ROOT, a.cache)
    os.makedirs(OUT, exist_ok=True)
    model, iters = K.load_trained(use_ema=True, path=os.path.join(ROOT, a.ckpt))
    model.eval()
    files = sorted(os.listdir(S.TEST_DIR))
    print(f"[pl] {os.path.basename(a.ckpt)} EMA iters={iters}, {len(files)} 张 test1, route=ms2")
    conf_samples = []
    for n, fn in enumerate(files):
        fp = os.path.join(OUT, fn.replace(".png", ".npz"))
        if os.path.exists(fp):
            continue
        img = np.array(Image.open(os.path.join(S.TEST_DIR, fn)).convert("RGB"))
        lg = S.predict_logits(model, img, tta="ms2")          # (9,H,W) cpu float32
        p = torch.softmax(lg, dim=0)
        conf, pred = p.max(dim=0)
        np.savez_compressed(fp, pred=pred.numpy().astype(np.uint8),
                            conf=conf.half().numpy())
        if n < 30:  # 抽样估计置信度分布
            conf_samples.append(conf.half().numpy()[::4, ::4].ravel())
        if (n + 1) % 25 == 0:
            print(f"  pl {n+1}/{len(files)}", flush=True)
    if conf_samples:
        c = np.concatenate(conf_samples)
        print("[pl] 置信度分布: " + " ".join(f"{q}={np.quantile(c, q):.3f}" for q in
              (0.1, 0.25, 0.5, 0.75, 0.9)))
        for t in (0.8, 0.85, 0.9, 0.95):
            print(f"  τ={t}: 保留 {(c >= t).mean() * 100:.1f}% 像素")
    print("[pl done]")


if __name__ == "__main__":
    main()
