"""
锚质检尺 —— 评估一个候选锚目录是否值得花 2 小时 GPU 去做 DACS 强锚回填

用法:
    python audit_anchor.py <候选锚目录> [基线锚目录]

做三件事:
  1. 合规检查: 500 张、命名 test1_1..500、1024²、uint8、像素值 ⊆ 1..8
  2. 打印类别占比
  3. 与基线逐像素对比: 一致率 / 差异率 / 差异的转移方向 (谁被换成了谁)

判断口径 (结合「锚质量≈学生上限」定律):
  * 差异率 < 0.1%   → 基本等同基线, 不值得重跑
  * 差异率 0.5~5%   → 有意义的变化, 值得花 GPU 验证 (前提: 候选确实更强)
  * 差异率 > 8%     → 太大, 可能是别的配方/别的域, 先确认来源
  ★ 关键前提: 候选锚必须**严格强于**基线 (拿模型方口径的外部指标), 否则融合/替换都是降级
"""
import os
import sys

import numpy as np
from PIL import Image

CLASS_NAMES = ["Ignore", "Background", "Building", "Road", "Water",
               "Barren", "Vegetation", "Agricultural", "Vehicle"]


def load_dir(d):
    files = sorted(os.listdir(d))
    pngs = [f for f in files if f.lower().endswith(".png")]
    problems = []
    if len(pngs) != 500:
        problems.append(f"PNG 数量={len(pngs)} (应为 500)")
    bad_non_png = [f for f in files if not f.lower().endswith(".png")]
    if bad_non_png:
        problems.append(f"存在非 PNG: {bad_non_png[:5]}")
    arrs, names = [], []
    for i in range(1, 501):
        p = os.path.join(d, f"test1_{i}.png")
        if not os.path.exists(p):
            problems.append(f"缺少 test1_{i}.png")
            continue
        im = Image.open(p)
        a = np.array(im)
        if a.shape != (1024, 1024):
            problems.append(f"test1_{i}.png 尺寸 {a.shape}")
        if a.dtype != np.uint8:
            problems.append(f"test1_{i}.png dtype {a.dtype}")
        v = np.unique(a)
        if v.min() < 1 or v.max() > 8:
            problems.append(f"test1_{i}.png 像素值越界 {v.min()}..{v.max()}")
        arrs.append(a.astype(np.uint8))
        names.append(f"test1_{i}.png")
    return arrs, names, problems


def dist(arrs):
    c = np.bincount(np.concatenate([a.ravel() for a in arrs]), minlength=10)[:10]
    tot = c[1:9].sum()
    return {CLASS_NAMES[i]: 100.0 * c[i] / tot for i in range(1, 9)}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cand = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else None

    print("=" * 70)
    print(f"候选锚: {cand}")
    print("=" * 70)
    A, names, probs = load_dir(cand)
    if probs:
        print("  ⚠️ 合规问题:")
        for p in probs[:12]:
            print("    -", p)
    else:
        print("  ✅ 合规: 500 张 / 1024² / uint8 / 像素值 ⊆ 1..8")
    if not A:
        print("  没有可用的图, 退出")
        sys.exit(1)

    dc = dist(A)
    print("\n  类别占比(%):")
    print("    " + "  ".join(f"{k}={v:.2f}" for k, v in dc.items()))

    if base:
        print("\n" + "=" * 70)
        print(f"基线锚: {base}")
        print("=" * 70)
        B, _, pb = load_dir(base)
        if pb:
            print(f"  ⚠️ 基线有 {len(pb)} 项问题(忽略)")
        n = min(len(A), len(B))
        diff_px = 0
        tot_px = 0
        trans = {}
        for i in range(n):
            a, b = A[i], B[i]
            m = a != b
            diff_px += int(m.sum())
            tot_px += a.size
            if m.any():
                pair = np.stack([b[m], a[m]], axis=1)
                uq, cnt = np.unique(pair, axis=0, return_counts=True)
                for (bf, af), c in zip(uq, cnt):
                    key = (CLASS_NAMES[bf], CLASS_NAMES[af])
                    trans[key] = trans.get(key, 0) + int(c)
        rate = 100.0 * diff_px / tot_px if tot_px else 0.0
        print(f"\n  逐像素差异率: {rate:.3f}%   (差异 {diff_px:,} / {tot_px:,} 像素)")
        print("  一致率:       {:.3f}%".format(100.0 - rate))
        db = dist(B)
        print("\n  占比对比 (候选 - 基线, pp):")
        for k in dc:
            d = dc[k] - db[k]
            flag = "  <<<" if abs(d) >= 0.5 else ""
            print(f"    {k:14s} {db[k]:6.2f} -> {dc[k]:6.2f}   {d:+.2f}{flag}")
        top = sorted(trans.items(), key=lambda kv: -kv[1])[:10]
        if top:
            print("\n  主要变化方向 (基线 -> 候选):")
            for (bf, af), c in top:
                print(f"    {bf:14s} -> {af:14s}  {c:>12,} px  ({100.0*c/tot_px:.3f}%)")
        print("\n  判断:")
        if rate < 0.1:
            print("    ❌ 与基线几乎相同 (<0.1%) → 不值得重跑")
        elif rate <= 8:
            print("    ✅ 差异在合理区间 → 若候选确实更强, 值得花 GPU 验证")
        else:
            print("    ⚠️ 差异过大 (>8%) → 先确认候选来源与配方")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
