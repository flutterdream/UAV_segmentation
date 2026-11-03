#!/bin/bash
# ══════════════════════════════════════════════════════════════════
#  04_bn_calib.sh —— 目标域 BN 统计校准 → 推理 → 打包 → 自检
#
#  机制：冻结全部学习参数，只把 41 个 BatchNorm 层的
#        running_mean / running_var 按 alpha 与目标域统计混合：
#           mu_new  = (1-a)*mu_old + a*mu_target
#           var_new = (1-a)*var_old + a*var_target + a*(1-a)*(mu_target-mu_old)^2
#        （最后一项是混合分布的方差项，不能漏）
#  来源：队内 2026-09-22 技术复盘「49 号 / S2-02」；test1 上 →（alpha=0.25）
#
#  用法:
#    bash scripts/04_bn_calib.sh [父ckpt] [TAG] [目标域图目录] [alpha] [tta]
#
#  例（用 test2 统计校准 relaxbg 母体）:
#    export SEG_TEST_DIR=/data/test2/images
#    export OUT_PREFIX=output_test2_kan_
#    bash scripts/04_bn_calib.sh weights/ckpt_kan_dacs_cw7_1024su_relaxbg.pth bnT2 \
#         "$SEG_TEST_DIR" 0.25 ms2x
# ══════════════════════════════════════════════════════════════════
set -eu
cd "$(dirname "$0")/.."

PY="${PY:-python}"
CKPT="${1:?用法: bash scripts/04_bn_calib.sh <父ckpt> <TAG> <目标域图目录> [alpha] [tta]}"
TAG="${2:?缺 TAG}"
IMGS="${3:?缺 目标域图目录}"
AL="${4:-0.25}"
TTA="${5:-ms2x}"

OUT="${OUT_PREFIX:-output_test1_kan_}${TAG}"
NEW="weights/ckpt_bn_${TAG}.pth"

say() { echo "[$(date '+%F %T')] $*"; }

say "════ BN 校准 ════"
say "  父模型 : $CKPT"
say "  输出   : $NEW"
say "  alpha  : $AL"
say "  目标域 : $IMGS ($(ls "$IMGS" 2>/dev/null | wc -l) 张)"

[ -f "$CKPT" ] || { say "!! 父 ckpt 不存在"; exit 2; }
[ -d "$IMGS" ] || { say "!! 图像目录不存在"; exit 2; }
command -v "$PY" >/dev/null || { say "!! 找不到解释器 $PY（可用 PY=... 覆盖）"; exit 2; }

say "── ① 收集目标域统计并混合 ──"
"$PY" -u code/bn_calib.py --ckpt "$CKPT" --out "$NEW" --imgs "$IMGS" --alpha "$AL"
[ -f "$NEW" ] || { say "!! 未生成 $NEW"; exit 3; }

say "── ② 推理（tta=$TTA）──"
"$PY" -u code/dacs_train.py --infer --ckpt "$NEW" --out "$OUT" --tta "$TTA"

say "── ③ 输出前自检 ──"
"$PY" -u code/check_output.py "${OUT}.zip"

say "★ 完成: ${OUT}.zip"
