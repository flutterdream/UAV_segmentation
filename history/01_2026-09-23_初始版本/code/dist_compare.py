# -*- coding: utf-8 -*-
"""
对比多个输出目录的类别占比。
★类别id约定(重要): 0=Ignore, 1=Background, 2=Building, 3=Road, 4=Water,
                    5=Barren, 6=Vegetation, 7=Agricultural, 8=Vehicle
即 CLASS_NAMES[i] 对应 id = i+1 (输出PNG中像素值=该id, 从不出现0)
"""
import os, sys
import numpy as np
from PIL import Image

NAMES = ["Background", "Building", "Road", "Water", "Barren", "Veg", "Agri", "Vehicle"]
IDS = list(range(1, 9))  # id 1..8

DIRS = {
    "cw7(68.9709)":  r"D:\kimi的游乐园\oi\output_test1_kan_dacs_cw7",
    "cw17(68.9417)": r"D:\kimi的游乐园\oi\output_test1_kan_dacs_cw17"
}
if len(sys.argv) > 1:
    mapping = {
        "cw7": r"D:\kimi的游乐园\oi\output_test1_kan_dacs_cw7",
        "cw17": r"D:\kimi的游乐园\oi\output_test1_kan_dacs_cw17",
        "tm6889": r"D:\kimi的游乐园\oi\teammate_688921",
        "tm6908": r"D:\kimi的游乐园\oi\teammate_690801",
        "sl20s7": r"D:\kimi的游乐园\oi\output_test1_kan_dacs_sl20s7",
        "s17":    r"D:\kimi的游乐园\oi\output_test1_kan_dacs_sl20s17",
        "s17hf":  r"D:\kimi的游乐园\oi\output_test1_kan_dacs_sl20s17hf",
        "ci":     r"D:\kimi的游乐园\oi\output_test1_kan_dacs_sl20ci_s7"
    }
    DIRS = {}
    for a in sys.argv[1:]:
        if a in mapping:
            DIRS[a] = mapping[a]
else:
    DIRS = {
        "cw7(68.9709)":  r"D:\kimi的游乐园\oi\output_test1_kan_dacs_cw7",
        "cw17(68.9417)": r"D:\kimi的游乐园\oi\output_test1_kan_dacs_cw17",
        "TM68.89":       r"D:\kimi的游乐园\oi\teammate_688921",
        "TM69.08":       r"D:\kimi的游乐园\oi\teammate_690801"
    }


def dist(d):
    files = sorted([f for f in os.listdir(d) if f.lower().endswith(".png")])
    cnt = np.zeros(10, dtype=np.int64)
    for f in files:
        a = np.asarray(Image.open(os.path.join(d, f)))
        cnt += np.bincount(a.ravel(), minlength=10)[:10]
    # 返回 8 个类的占比(%)
    tot = cnt[IDS].sum()
    return cnt[IDS] / tot * 100.0, len(files), int(cnt[0])


res = {}
for k, d in DIRS.items():
    if not os.path.isdir(d):
        print(f"[跳过] {k} 不存在: {d}")
        continue
    pct, n, ig = dist(d)
    res[k] = pct
    print(f"[{k}] n={n}  Ignore像素={ig}")
    print("   " + "  ".join(f"{NAMES[i]}={pct[i]:.2f}" for i in range(8)))

keys = list(res.keys())
print("\n=== 类别占比 (%) ===")
print("类别".ljust(12) + "".join(k.ljust(16) for k in keys))
for i in range(8):
    print(NAMES[i].ljust(12) + "".join(f"{res[k][i]:<16.2f}" for k in keys))

if "TM69.08" in res:
    ref = res["TM69.08"]
    print("\n=== 相对 TM69.08 的占比差 (pp)  [正=我方多判, 负=我方少判] ===")
    others = [k for k in keys if k != "TM69.08"]
    print("类别".ljust(12) + "".join(k.ljust(16) for k in others))
    for i in range(8):
        print(NAMES[i].ljust(12) + "".join(f"{res[k][i]-ref[i]:<+16.2f}" for k in others))
    print("\n=== 队友自身进化 TM68.89 -> TM69.08 (pp) ===")
    if "TM68.89" in res:
        for i in range(8):
            print(f"  {NAMES[i]:<12}{res['TM69.08'][i]-res['TM68.89'][i]:+.2f}")
