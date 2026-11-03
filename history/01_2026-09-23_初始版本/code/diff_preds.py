#!/usr/bin/env python
"""两个预测目录的逐像素差异分析（test2 版锚质检工具）
   audit_anchor.py 硬编码了 test1/500 张，本脚本自适应任意前缀/数量。

用法: python diff_preds.py <候选目录> <基线目录>
"""
import sys, os, glob
import numpy as np
from PIL import Image

CLASS_NAMES = ["Ignore", "Background", "Building", "Road", "Water",
               "Barren", "Vegetation", "Agricultural", "Vehicle"]


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    cand, base = sys.argv[1], sys.argv[2]
    fs = sorted(glob.glob(os.path.join(cand, "*.png")))
    if not fs:
        print("候选中没有 png:", cand)
        return

    tot = diff = n_img = changed_imgs = 0
    trans = {}
    for f in fs:
        n = os.path.basename(f)
        pb = os.path.join(base, n)
        if not os.path.exists(pb):
            continue
        A = np.array(Image.open(f))
        B = np.array(Image.open(pb))
        if A.shape != B.shape:
            continue
        d = A != B
        nd = int(d.sum())
        tot += A.size
        diff += nd
        n_img += 1
        if nd:
            changed_imgs += 1
            for y, x in zip(B[d], A[d]):
                k = (int(y), int(x))
                trans[k] = trans.get(k, 0) + 1

    print("=" * 62)
    print("候选: %s" % cand)
    print("基线: %s" % base)
    print("=" * 62)
    print("比对图像      : %d 张" % n_img)
    print("有差异的图像  : %d 张 (%.1f%%)" % (changed_imgs, 100.0 * changed_imgs / max(n_img, 1)))
    print("像素差异率    : %.4f%%   (%d / %d)" % (100.0 * diff / max(tot, 1), diff, tot))
    if trans:
        print()
        print("主要转移方向（基线 -> 候选, Top12）：")
        for (a, b), c in sorted(trans.items(), key=lambda kv: -kv[1])[:12]:
            na = CLASS_NAMES[a] if 0 <= a < len(CLASS_NAMES) else str(a)
            nb = CLASS_NAMES[b] if 0 <= b < len(CLASS_NAMES) else str(b)
            print("  %-14s -> %-14s %9d  %6.3f%%" % (na, nb, c, 100.0 * c / max(tot, 1)))


if __name__ == "__main__":
    main()
