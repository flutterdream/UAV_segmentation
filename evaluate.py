# -*- coding: utf-8 -*-
"""在带标注的验证集上计算 mIoU，复现 README 中的指标。

与训练时使用的评测口径完全一致：1024×1024 原生分辨率、单次前向、argmax、
忽略类（像素值 0）不计入统计。

用法：
    python evaluate.py --ckpt weights/best_model.pth \
                       --images data/train/images \
                       --masks  data/train/masks \
                       --val-ids data/val_ids.json

    # 若希望用 8 路概率平均（多视图口径）评测，加 --tta
    python evaluate.py ... --tta

预期：`weights/best_model.pth` 在 `data/val_ids.json` 指定的 500 张上
      单次前向 mIoU ≈ 0.7668，8 路 TTA ≈ 0.7675。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.deepkanseg import DeepKANSegSwin, NUM_CLASSES

NB = 9
NAMES = ["背景", "建筑", "道路", "水体", "荒地", "森林", "农田", "车辆"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
PASSES = [(1.0, False), (1.0, True), (0.75, False), (0.75, True),
          (1.25, False), (1.25, True), (1.5, False), (0.5, False)]


def load_model(ckpt, device):
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "ema" not in payload:
        raise SystemExit("权重文件缺少 ema 参数，请确认是否为训练产出的 .pth")
    model = DeepKANSegSwin(NUM_CLASSES, enc_weights=None)  # 权重由 ckpt 提供，无需联网
    model.load_state_dict(payload["ema"], strict=True)
    return model.to(device).eval().requires_grad_(False), payload


@torch.no_grad()
def predict(model, img, device, tta):
    h, w = img.shape[:2]
    passes = PASSES if tta else [(1.0, False)]
    acc = None
    for scale, flip in passes:
        im = img
        if abs(scale - 1.0) > 1e-6:
            im = np.asarray(Image.fromarray(im).resize(
                (int(w * scale), int(h * scale)), Image.BILINEAR))
        if flip:
            im = im[:, ::-1]
        x = (im.astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        with torch.autocast("cuda", torch.bfloat16, enabled=(device == "cuda")):
            logits = model(x)
        logits = logits.float()
        if logits.shape[-1] != w:
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        p = logits.softmax(1)
        if flip:
            p = torch.flip(p, dims=(3,))
        acc = p if acc is None else acc + p
    return (acc / len(passes)).argmax(1)[0].cpu().numpy().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", required=True, help="验证集影像目录")
    ap.add_argument("--masks", required=True, help="验证集标签目录（与影像同名）")
    ap.add_argument("--val-ids", required=True, help="val_ids.json，含 500 个影像 ID")
    ap.add_argument("--limit", type=int, default=0, help="只评测前 N 张（调试用）")
    ap.add_argument("--tta", action="store_true", help="启用 8 路概率平均")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, payload = load_model(a.ckpt, device)
    print(f"[载入] {a.ckpt}  iter={payload.get('iter')}  "
          f"训练时记录 val500={payload.get('val500')}")

    ids = json.load(open(a.val_ids, encoding="utf-8"))["val_ids"]
    if a.limit:
        ids = ids[:a.limit]
    img_dir, msk_dir = Path(a.images), Path(a.masks)
    print(f"[评测] {len(ids)} 张  TTA={'开' if a.tta else '关'}  设备={device}")

    cm = np.zeros((NB, NB), np.int64)
    skipped = 0
    for k, i in enumerate(ids, 1):
        name = f"{i:04d}.png"
        ip, mp = img_dir / name, msk_dir / name
        if not ip.exists() or not mp.exists():
            skipped += 1
            continue
        img = np.asarray(Image.open(ip).convert("RGB"))
        gt = np.asarray(Image.open(mp))
        pred = np.clip(predict(model, img, device, a.tta), 1, 8)
        m = gt > 0
        cm += np.bincount(gt[m].astype(np.int16) * NB + pred[m].astype(np.int16),
                          minlength=NB * NB).reshape(NB, NB)
        if k % 100 == 0:
            print(f"  {k}/{len(ids)}")

    if skipped:
        print(f"[警告] {skipped} 张未找到，已跳过")
    inter = np.diag(cm).astype(float)
    union = cm.sum(0) + cm.sum(1) - np.diag(cm)
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    present = cm.sum(1) > 0

    print("\n类别      IoU      像素数")
    for c in range(NB):
        if present[c]:
            print(f"  {NAMES[c-1] if c else '忽略':6s}  {iou[c]:.6f}  {int(cm[c].sum())}")
    miou = float(np.nanmean(iou[present]))
    pixel_acc = float((inter[1:].sum()) / max(cm[1:].sum(), 1))
    print(f"\n  mIoU = {miou:.6f}")
    print(f"  OA   = {pixel_acc:.6f}")


if __name__ == "__main__":
    main()
