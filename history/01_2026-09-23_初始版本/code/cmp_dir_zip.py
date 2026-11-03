# -*- coding: utf-8 -*-
"""对比「推理输出目录」与「参考 zip」，逐像素统计一致率"""
import sys, os, io, zipfile, numpy as np
from PIL import Image


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "_verify_infer/out"
    ref_zip = sys.argv[2] if len(sys.argv) > 2 else \
        r"D:\kimi的游乐园\oi\test2_ret_0922\output_test2_kan_b8.zip"

    z = zipfile.ZipFile(ref_zip)
    znames = set(n for n in z.namelist() if n.endswith(".png"))
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".png"))
    print("输出目录: %s  (%d 张)" % (out_dir, len(files)))
    print("参考 zip : %s  (%d 张)" % (ref_zip, len(znames)))
    print()

    same = 0
    diff_px = 0
    tot_px = 0
    worst = []
    missing = 0
    for fn in files:
        if fn not in znames:
            missing += 1
            print("  [缺] zip 中无 %s" % fn)
            continue
        a = np.array(Image.open(os.path.join(out_dir, fn)))
        b = np.array(Image.open(io.BytesIO(z.read(fn))))
        if a.shape != b.shape:
            print("  [形状不符] %s %s vs %s" % (fn, a.shape, b.shape))
            continue
        m = a != b
        d = int(m.sum())
        tot_px += a.size
        diff_px += d
        if d == 0:
            same += 1
        else:
            worst.append((fn, d))

    n_cmp = len(files) - missing
    print()
    print("=== 结果 ===")
    print("  参与对比      : %d 张" % n_cmp)
    print("  逐像素完全一致: %d 张 (%.1f%%)" % (same, 100.0 * same / max(n_cmp, 1)))
    if tot_px:
        print("  像素一致率    : %.6f%%" % (100.0 * (tot_px - diff_px) / tot_px))
    if worst:
        worst.sort(key=lambda x: -x[1])
        print("  有差异的图 Top5 (差异像素数):")
        for fn, d in worst[:5]:
            print("     %-18s %d  (%.4f%%)" % (fn, d, 100.0 * d / (1024 * 1024)))
    print()
    if n_cmp > 0 and same == n_cmp:
        print("  ★★ 结论: 完全一致 —— 推理链路可复现")
    elif tot_px and (tot_px - diff_px) / tot_px > 0.999:
        print("  ★ 结论: 一致率 >99.9% —— 可视为可复现（差异属浮点/算子级）")
    else:
        print("  ⚠ 结论: 存在明显差异，需排查")


if __name__ == "__main__":
    main()
