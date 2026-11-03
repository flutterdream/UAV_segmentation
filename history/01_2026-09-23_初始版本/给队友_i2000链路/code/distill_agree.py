# -*- coding: utf-8 -*-
"""蒸馏快照选拔: 各快照在 test1 上与队友教师标签的一致率(mIoU口径).
教师=榜67.33 -> 一致率越高, 外部指标越接近教师. 前50张plain(快), 仅用于横向比较."""
import os, sys
import numpy as np
from PIL import Image
import torch

import seg_swin as S
import kan_swin as K

ROOT = os.path.dirname(os.path.abspath(__file__))
TEACHER = os.path.join(ROOT, "teammate_best")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50


@torch.no_grad()
def agree(ckpt, files):
    model, it = K.load_trained(path=os.path.join(ROOT, ckpt))
    model.eval()
    inter = np.zeros(9, np.float64); union = np.zeros(9, np.float64)
    for fn in files:
        img = np.array(Image.open(os.path.join(S.TEST_DIR, fn)).convert("RGB"))
        pred = S.predict_logits(model, img).argmax(0).numpy()
        t = np.array(Image.open(os.path.join(TEACHER, fn)))
        for c in range(1, 9):
            p, tt = pred == c, t == c
            inter[c] += (p & tt).sum(); union[c] += (p | tt).sum()
        del img, pred
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return np.nanmean(iou), it


def main():
    files = sorted(os.listdir(S.TEST_DIR))[:N]
    for ck in sys.argv[2:]:
        m, it = agree(ck, files)
        print(f"{ck}: 教师一致率 mIoU={m:.4f} (n={N})")


if __name__ == "__main__":
    main()
