#!/bin/bash
# ══════════════════════════════════════════════════════════════
#  仅训练（不推理）—— 用于单独产出权重
#
#  用法: bash scripts/01_train.sh
#         DATA_DIR=... MASK_DIR=... ANCHOR_DIR=... bash scripts/01_train.sh
# ══════════════════════════════════════════════════════════════
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(dirname "$HERE")"
cd "$PKG"

source configs/best.env

PY="${PY:-python}"
command -v "$PY" >/dev/null 2>&1 || { echo "!! 找不到 Python: $PY"; exit 1; }

export SEG_IMG_DIR="${DATA_DIR:-$PKG/data/train/images}"
export SEG_MSK_DIR="${MASK_DIR:-$PKG/data/train/masks}"
ANCHOR_PATH="${ANCHOR_DIR:-$PKG/data/anchor}"
export PYTHONPATH="$PKG/code:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p output
echo "[$(date '+%F %T')] 训练开始: tag=$TAG crop=$CROP iters=$TARGET_ITERS"
"$PY" code/dacs_train.py --train \
    --tag "$TAG" \
    --warm "$WARM_CKPT" \
    --anchor "$ANCHOR_PATH" \
    --seed "$SEED" \
    --crop "$CROP" \
    --batch "$BATCH" \
    --target-iters "$TARGET_ITERS" \
    --max-seconds "$MAX_SECONDS" \
    --ce-weights "$CE_WEIGHTS" \
    --boost "$BOOST" \
    $SCALE_UP \
    2>&1 | tee "output/train_${TAG}.log"

rc=${PIPESTATUS[0]}
echo "[$(date '+%F %T')] 训练退出码=$rc"
exit "$rc"
