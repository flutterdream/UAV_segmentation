#!/bin/bash
# ══════════════════════════════════════════════════════════════
#  本地验证代码 —— 逐类 IoU 评估（在源域 val 集上）
#
#  ⚠️ 重要说明（评审须知）:
#     本项目实测发现「源域 val 分数」与「竞赛: val 从 0.7528 降到 0.7483，而09）。
#     原因是 val 属源域且已饱和，而。
#     ⇒ 本脚本用于回归检查（确认代码正确、模型没崩），
#       **不作为模型选择的唯一依据**。
#
#  用法: bash scripts/03_eval_val.sh [n]        # 默认评估前 120 张 val
# ══════════════════════════════════════════════════════════════
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(dirname "$HERE")"
cd "$PKG"

source configs/best.env

PY="${PY:-python}"
command -v "$PY" >/dev/null 2>&1 || { echo "!! 找不到 Python: $PY"; exit 1; }

N="${1:-120}"

export SEG_IMG_DIR="${DATA_DIR:-$PKG/data/train/images}"
export SEG_MSK_DIR="${MASK_DIR:-$PKG/data/train/masks}"
export PYTHONPATH="$PKG/code:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

CKPT="ckpt_kan_${TAG}.pth"
[ -f "$CKPT" ] || { echo "!! 缺少 $CKPT（请先训练，或指定其它 ckpt）"; exit 1; }

echo "[$(date '+%F %T')] 源域 val 逐类评估: n=$N  ckpt=$CKPT  TTA=$TTA"
"$PY" code/eval_perclass.py "$CKPT" "$N" "$TTA" 2>&1 | tee "output_eval_${TAG}.log"
