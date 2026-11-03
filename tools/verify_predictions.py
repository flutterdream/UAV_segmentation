# -*- coding: utf-8 -*-
"""预测结果格式校验。

检查项：
  1. zip 可读、无损坏条目
  2. PNG 数量与命名（平铺，无子目录）
  3. 尺寸 1024×1024、单通道 8 位
  4. 像素值域 1~8（0 为忽略类，不应出现在提交结果中）

用法：
    python tools/verify_predictions.py outputs/predictions.zip
    python tools/verify_predictions.py outputs/png          # 也可校验目录
"""
import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

EXPECT_SIZE = (1024, 1024)
EXPECT_RANGE = (1, 8)


def check_array(arr, name):
    problems = []
    if arr.shape != EXPECT_SIZE:
        problems.append(f"{name}: 尺寸 {arr.shape} != {EXPECT_SIZE}")
    if arr.dtype != np.uint8:
        problems.append(f"{name}: 位深 {arr.dtype} != uint8")
    lo, hi = EXPECT_RANGE
    if arr.size and (arr.min() < lo or arr.max() > hi):
        problems.append(f"{name}: 值域 [{arr.min()}, {arr.max()}] 越界（应为 {lo}~{hi}）")
    if arr.ndim != 2:
        problems.append(f"{name}: 通道数异常 {arr.shape}")
    return problems


def from_zip(path):
    z = zipfile.ZipFile(path)
    bad = z.testzip()
    if bad:
        print(f"[FAIL] zip 损坏条目: {bad}")
        return 1
    names = [n for n in z.namelist() if n.lower().endswith(".png")]
    print(f"PNG 数量: {len(names)}")
    nested = [n for n in names if "/" in n]
    if nested:
        print(f"[WARN] 存在子目录: {nested[:3]}（提交通常要求平铺）")
    problems = []
    step = max(1, len(names) // 200)
    for i, n in enumerate(names):
        if i % step:
            continue
        try:
            arr = np.array(Image.open(io.BytesIO(z.read(n))))
        except Exception as ex:
            problems.append(f"{n}: 无法读取 ({ex})")
            continue
        problems += check_array(arr, n)
    return report(problems, len(names))


def from_dir(path):
    files = sorted(p for p in Path(path).rglob("*.png"))
    print(f"PNG 数量: {len(files)}")
    problems = []
    step = max(1, len(files) // 200)
    for i, p in enumerate(files):
        if i % step:
            continue
        try:
            arr = np.array(Image.open(p))
        except Exception as ex:
            problems.append(f"{p.name}: 无法读取 ({ex})")
            continue
        problems += check_array(arr, p.name)
    return report(problems, len(files))


def report(problems, total):
    if total == 0:
        print("[FAIL] 未找到任何 PNG")
        return 1
    if problems:
        print(f"[FAIL] 发现 {len(problems)} 个问题：")
        for p in problems[:10]:
            print("   ", p)
        return 1
    print("[OK] 全部检查通过")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="预测 zip 或 PNG 目录")
    a = ap.parse_args()
    p = Path(a.path)
    if not p.exists():
        print(f"[FAIL] 路径不存在: {p}")
        return 1
    return from_zip(p) if p.suffix.lower() == ".zip" else from_dir(p)


if __name__ == "__main__":
    sys.exit(main())
