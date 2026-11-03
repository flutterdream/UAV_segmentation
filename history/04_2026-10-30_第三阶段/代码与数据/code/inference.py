# -*- coding: utf-8 -*-
"""推理脚本：单模型 + 8 路概率平均（TTA），输出 3594 张 PNG 并打包。

用法：
    python inference.py --ckpt ../checkpoints/best_model.pth \
                        --images /path/to/test_3/images \
                        --out ../predictions/png \
                        --zip ../predictions/predictions.zip

TTA 说明（与最终成绩完全一致，八路等权平均）：
    (1.0, 原图) (1.0, 水平翻转) (0.75, 原图) (0.75, 翻转)
    (1.25, 原图) (1.25, 翻转) (1.5, 原图) (0.5, 原图)
    每路先归一化 → fp16 前向 → logits 双线性还原到原尺寸 → softmax → 翻转回来 → 累加
    八路概率算术平均后取 argmax，像素值 1..8（0=忽略类不输出）
"""
import argparse
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.deepkanseg import DeepKANSegSwin, NUM_CLASSES

PASSES = [(1.0, False), (1.0, True), (0.75, False), (0.75, True),
          (1.25, False), (1.25, True), (1.5, False), (0.5, False)]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_model(ckpt, device):
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = payload["ema"] if isinstance(payload, dict) and "ema" in payload else payload
    model = DeepKANSegSwin(NUM_CLASSES)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval().requires_grad_(False)
    return model, payload


@torch.no_grad()
def probability_average(model, image, device):
    h, w = image.shape[:2]
    acc = None
    for scale, flip in PASSES:
        im = image
        if abs(scale - 1.0) > 1e-6:
            im = np.asarray(Image.fromarray(im).resize(
                (int(w * scale), int(h * scale)), Image.BILINEAR))
        if flip:
            im = im[:, ::-1]
        x = (im.astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        with torch.autocast("cuda", torch.float16, enabled=(device == "cuda")):
            logits = model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits = logits.float()
        if logits.shape[-1] != w:
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        p = logits.softmax(1)
        if flip:
            p = torch.flip(p, dims=(3,))
        acc = p if acc is None else acc + p
    averaged = acc / len(PASSES)
    if not torch.isfinite(averaged).all():
        raise RuntimeError("概率平均出现非有限值")
    return averaged.squeeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="模型权重 .pth")
    ap.add_argument("--images", required=True, help="测试集图像目录（PNG/JPG）")
    ap.add_argument("--out", required=True, help="输出 PNG 目录")
    ap.add_argument("--zip", default="", help="可选：打包为 zip")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, payload = load_model(a.ckpt, device)
    print(f"[载入] {a.ckpt}  iter={payload.get('iter') if isinstance(payload, dict) else '?'}  "
          f"val500={payload.get('val500_miou') if isinstance(payload, dict) else '?'}")

    src = Path(a.images)
    files = sorted([p for p in src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[推理] {len(files)} 张，输出到 {out}")

    for k, p in enumerate(files, 1):
        img = np.asarray(Image.open(p).convert("RGB"))
        prob = probability_average(model, img, device)
        pred = prob.argmax(0).cpu().numpy().astype(np.uint8)
        if pred.min() < 1 or pred.max() > 8:
            raise RuntimeError(f"类别越界：{p.name}")
        Image.fromarray(pred, mode="L").save(out / p.name)
        if k % 200 == 0:
            print(f"  {k}/{len(files)}")

    if a.zip:
        zp = Path(a.zip)
        zp.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
            for p in sorted(out.glob("*.png")):
                z.write(p, p.name)
        print(f"[完成] {zp}  {zp.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
