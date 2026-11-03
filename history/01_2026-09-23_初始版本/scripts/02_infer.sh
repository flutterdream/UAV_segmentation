#!/bin/bash
# ══════════════════════════════════════════════════════════════
#  仅推理（生成输出 zip）—— 用已有权重，零训练成本
#
#  用法: bash scripts/02_infer.sh [ckpt] [out_name] [tta]
#    默认: ckpt_kan_dacs_cw7_1024su_relaxbg.pth + ms2x
#    TEST_DIR=/path/to/images bash scripts/02_infer.sh
# ══════════════════════════════════════════════════════════════
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(dirname "$HERE")"
cd "$PKG"

source configs/best.env

PY="${PY:-python}"
command -v "$PY" >/dev/null 2>&1 || { echo "!! 找不到 Python: $PY"; exit 1; }

CKPT="${1:-ckpt_kan_${TAG}.pth}"
OUT="${2:-output_kan_${TAG}_${TTA}}"
TTA_ARG="${3:-$TTA}"

export SEG_TEST_DIR="${TEST_DIR:-$PKG/data/test/images}"
export PYTHONPATH="$PKG/code:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "[$(date '+%F %T')] 推理: ckpt=$CKPT  test=$SEG_TEST_DIR  tta=$TTA_ARG"
"$PY" code/dacs_train.py --infer --ckpt "$CKPT" --out "$OUT" --tta "$TTA_ARG" 2>&1 \
  | tee "output_infer_${OUT}.log"

rc=${PIPESTATUS[0]}
ZIP="${OUT}.zip"
if [ -f "$ZIP" ]; then
  echo "[$(date '+%F %T')] 自检:"
  "$PY" code/check_output.py "$ZIP"
  echo "[$(date '+%F %T')] ★ 完成: $ZIP"
else
  echo "[$(date '+%F %T')] !! 未生成 $ZIP (rc=$rc)"
  exit 1
fi
