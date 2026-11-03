# -*- coding: utf-8 -*-
"""
输出包自检 —— 输出前跑一条命令, 把所有会被平台拒绝的点全查一遍
════════════════════════════════════════════════════════════════
用法:
  python check_output.py                          # 检查所有 output_test1_kan_dacs_*.zip
  python check_output.py output_test1_kan_dacs_tau92s7.zip
  python check_output.py --all                    # 同默认, 全量扫描

检查项 (逐条对应平台可能的拒绝理由):
  1) zip 能否正常打开、有无损坏条目 (testzip)
  2) ★全部条目都是 .png —— 不含任何非 PNG (平台报过的错: "zip 中存在非 PNG 文件")
  3) 条目全为扁平命名, 不含目录前缀 (如 xxx/test1_1.png)
  4) 条目数 == 500, 且命名 test1_1.png ... test1_500.png 完整无缺
  5) 每张图 1024x1024 / uint8
  6) 像素值 ⊆ {1..8} (0=Ignore 与 9 都不该出现)
  7) 各类别占比 (便于和锚线对照)

★教训 (9/13): tau92s7 首次输出失败, 报"存在非 PNG 文件 ['output_test1_kan_dacs_cw2s7.zip']"。
  该字样是个 zip 文件名 → **不是包内部条目, 而是上传时把多个文件一起交了**。
  包本身 500 条全 PNG、结构与此前输出成功的包逐字一致。
  ⇒ **输出流程: 一次只传一个 zip 文件。**
"""
import glob
import os
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
# ★ 9/21 第二阶段适配：默认【自动从包内文件名推断】前缀与数量（test1=500 / test2=1300 通吃），
#   也可用环境变量强制覆盖：SEG_PREFIX=test2 SEG_EXPECT_N=1300
EXPECT_N = int(os.environ.get("SEG_EXPECT_N") or 0) or None
EXPECT_PREFIX = os.environ.get("SEG_PREFIX") or None
EXPECT_WH = (1024, 1024)


def _infer_prefix_count(pngs):
    """从包内 PNG 文件名推断 (前缀, 最大编号)。例: test2_1..test2_1300 → ('test2', 1300)"""
    import re
    pats = {}
    for n in pngs:
        m = re.match(r"^([A-Za-z0-9]+?)_(\d+)\.png$", n)
        if m:
            pats.setdefault(m.group(1), []).append(int(m.group(2)))
    if not pats:
        return None, None
    pref = max(pats, key=lambda k: len(pats[k]))       # 条目最多的前缀
    return pref, max(pats[pref])
CLASS_NAMES = ["Ignore", "Background", "Building", "Road",
               "Water", "Barren", "Vegetation", "Agricultural", "Vehicle"]


def check_one(zpath):
    name = os.path.basename(zpath)
    print(f"\n{'=' * 68}\n{name}\n{'=' * 68}")
    problems = []

    # 1) 打开 + 完整性
    try:
        zf = zipfile.ZipFile(zpath)
    except Exception as e:
        print(f"  [FAIL] 无法打开: {e}")
        return False
    bad = zf.testzip()
    if bad:
        problems.append(f"zip 内有损坏条目: {bad}")
    print(f"  条目总数      : {len(zf.namelist())}")
    print(f"  文件大小      : {os.path.getsize(zpath) / 1024 / 1024:.2f} MB")

    nl = zf.namelist()

    # 2) ★全部条目必须是 PNG
    nonpng = [n for n in nl if not n.lower().endswith(".png")]
    if nonpng:
        problems.append(f"存在非 PNG 条目 {len(nonpng)} 个: {nonpng[:5]}")
    print(f"  非 PNG 条目   : {len(nonpng)}  {'(OK)' if not nonpng else '★违规'}")

    # 3) 扁平命名
    nested = [n for n in nl if "/" in n or "\\" in n]
    if nested:
        problems.append(f"存在目录前缀 {len(nested)} 个: {nested[:3]}")
    print(f"  目录前缀条目  : {len(nested)}  {'(OK)' if not nested else '★违规'}")

    # 4) 数量 + 命名完整性（★ 自动推断前缀/数量，兼容 test1 与 test2）
    pngs = [n for n in nl if n.lower().endswith(".png")]
    pref, exp_n = EXPECT_PREFIX, EXPECT_N
    if pref is None or exp_n is None:
        ip, inn = _infer_prefix_count(pngs)
        pref = pref or ip
        exp_n = exp_n or inn
    print(f"  命名规则      : 前缀={pref}  期望数量={exp_n}")
    expect = set()
    if pref is None or exp_n is None:
        problems.append("无法从文件名推断前缀/数量，请用 SEG_PREFIX / SEG_EXPECT_N 指定")
    else:
        if len(nl) != exp_n:
            problems.append(f"条目数 {len(nl)} != {exp_n}")
        expect = {f"{pref}_{i}.png" for i in range(1, exp_n + 1)}
    missing = expect - set(pngs)
    extra = set(pngs) - expect
    if missing:
        problems.append(f"缺少 {len(missing)} 个应有文件, 例: {sorted(missing)[:3]}")
    if extra:
        problems.append(f"多出 {len(extra)} 个意外文件, 例: {sorted(extra)[:3]}")
    print(f"  命名完整性    : {'OK' if not missing and not extra else '★违规'}"
          f"  (缺{len(missing)} 多{len(extra)})")

    # 5)+6)+7) 逐张检查尺寸/类型/像素值, 并累计分布
    tot = np.zeros(10, dtype=np.int64)
    bad_wh, bad_dt, bad_val, unreadable = [], [], [], []
    for n in pngs:
        try:
            a = np.array(Image.open(zf.open(n)))
        except Exception:
            unreadable.append(n)
            continue
        if a.shape != EXPECT_WH:
            bad_wh.append((n, a.shape))
        if a.dtype != np.uint8:
            bad_dt.append((n, str(a.dtype)))
        u = np.unique(a)
        if u.min() < 0 or u.max() > 9 or 0 in u.tolist() or 9 in u.tolist():
            bad_val.append((n, u.tolist()))
        tot += np.bincount(a.ravel(), minlength=10)

    if unreadable:
        problems.append(f"{len(unreadable)} 张无法读取, 例: {unreadable[:3]}")
    if bad_wh:
        problems.append(f"{len(bad_wh)} 张尺寸异常, 例: {bad_wh[:3]}")
    if bad_dt:
        problems.append(f"{len(bad_dt)} 张 dtype 异常, 例: {bad_dt[:3]}")
    if bad_val:
        problems.append(f"{len(bad_val)} 张出现 0/9 像素值, 例: {bad_val[:3]}")
    print(f"  尺寸/类型     : {'OK' if not bad_wh and not bad_dt else '★违规'}"
          f"  (异常尺寸{len(bad_wh)} 异常类型{len(bad_dt)})")
    print(f"  像素值范围    : {'OK (全在 1..8)' if not bad_val else '★违规'}"
          f"  (含 0/9 的图 {len(bad_val)} 张)")

    if tot.sum() > 0:
        p = tot / tot.sum() * 100
        print("  类别占比(%):  " + "  ".join(
            f"{CLASS_NAMES[i]}={p[i]:.2f}" for i in range(1, 9)))

    if problems:
        print(f"\n  ★★ 结论: 不合规, 共 {len(problems)} 项问题")
        for x in problems:
            print(f"     - {x}")
        return False
    print("\n  ✅ 结论: 合规, 可输出 (记住: 一次只传这一个 zip)")
    return True


def main():
    args = [a for a in sys.argv[1:] if a != "--all"]
    if args:
        targets = [a if os.path.isabs(a) else os.path.join(ROOT, a) for a in args]
    else:
        targets = sorted(glob.glob(os.path.join(ROOT, "output_test1_kan_dacs_*.zip")))
    if not targets:
        print("未找到待检查的 zip")
        return 1
    ok = all(check_one(t) for t in targets)
    print(f"\n{'=' * 68}")
    print(f"总计 {len(targets)} 个包, 全部{'合规 ✅' if ok else '存在违规 ★★'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
