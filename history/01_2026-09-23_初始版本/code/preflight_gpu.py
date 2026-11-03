# -*- coding: utf-8 -*-
"""
起跑前显存闸门: 训练峰值约 4.42GB allocated / ~4.6GB reserved (bs2+crop640+grad_ckpt)。
若 nvidia-smi 报告的 free < 1500MiB, 驱动会做显存分页(thrash), 训练表现为
"GPU util 99% + 日志零推进 + free 极小", 实际 50 iter 要跑十几分钟 → 等于卡死。
本脚本用于在启动训练前拦下这种情况。

用法: python preflight_gpu.py [需要的free MiB, 默认 1500]
退出码: 0=可跑, 1=显存不足(不要起跑)
"""
import subprocess, sys

NEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1500

try:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
    total, used, free, util = [int(x.strip()) for x in out.split(",")]
except Exception as e:
    print(f"[preflight] 无法读取 GPU 状态: {e}")
    print("[preflight] 保守起见: 请手动确认显存后再起跑")
    sys.exit(1)

print(f"[preflight] 显存 total={total}MiB used={used}MiB free={free}MiB | GPU util={util}%")
print(f"[preflight] 训练峰值约 4.42GB(allocated), 本闸门要求 free >= {NEED}MiB")

if free >= NEED:
    print("[preflight] ✅ 可起跑")
    sys.exit(0)

print(f"[preflight] ❌ 显存不足 (free {free} < {NEED} MiB), 起跑会 thrash 卡死")
print("[preflight] 请先关闭占用显存的程序: Edge 多余标签页 / WPS / 腾讯视频 /")
print("[preflight] NVIDIA Overlay / 小组件面板 / 手机连接 / 正在运行的游戏")
print("[preflight] 若无法腾出, 只能把 crop 640->576 (或 bs2->1), 但会丧失与 cw7 的单变量可比性")
sys.exit(1)
