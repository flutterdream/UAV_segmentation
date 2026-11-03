#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
#  阶段 A —— 从公开学术权重复现「基础模型」（N 线四步）
#
#  起点: 公开 ImageNet 预训练 Swin-B
#        smp.UPerNet(encoder_name="tu-swin_base_patch4_window7_224",
#                    encoder_weights="imagenet") 自动下载, 无需预置任何权重
#  终点: code/ckpt_kan_noise_dist_i4000.pth
#        = 阶段 B（scripts/00_reproduce_all.sh）的暖启动权重 WARM_CKPT
#
#  四步（N 线 = 在第 1 步加入「噪声剔除 + 暗化增广」的修复线）:
#    [A1] 基础模型 16k, crop512, --exclude-noise
#    [A2] 伪标签生成（对 test1 跑 ms2 推理 → pred+conf 缓存）
#    [A3] PL 自蒸馏微调, crop640, 4000it → 峰值 i1000
#    [A4] 队友蒸馏,      crop640, 4000it → 峰值 i4000   ★ 终点
#
#  ⚠️ 阶段 A 全程基于 test1；切到 test2 是阶段 B 的事
#  ⚠️ [A4] 需要队友教师标签（上游共享资产），见文末「合规说明」
#
#  用法:
#    bash scripts/00a_stage_a_base_model.sh        # 四步全跑
#    bash scripts/00a_stage_a_base_model.sh 1      # 只跑 [A1]
#    bash scripts/00a_stage_a_base_model.sh 3 4    # 跑 [A3][A4]
#
#  断点续训: 每段受 --max-seconds 限制自动退出；重跑同一条命令即从快照 resume。
#           小显存机器建议逐段跑（见 README 的显存预算表）。
# ══════════════════════════════════════════════════════════════════════════
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(dirname "$HERE")"
cd "$PKG"

# ---- 载入配置 ----
# shellcheck disable=SC1091
source configs/stage_a.env

PY="${PY:-python}"
command -v "$PY" >/dev/null 2>&1 || { echo "!! 找不到 Python 解释器: $PY（可用 PY=/path/to/python 覆盖）"; exit 1; }

# ---- 路径解析（与阶段 B 保持同名环境变量）----
export SEG_IMG_DIR="${DATA_DIR:-$PKG/data/train/images}"
export SEG_MSK_DIR="${MASK_DIR:-$PKG/data/train/masks}"
# ⚠️ 阶段 A 的「测试集」必须是 test1（伪标签与蒸馏都基于它）
export SEG_TEST_DIR="${TEST1_DIR:-${TEST_DIR:-$PKG/data/test1/images}}"
# 队友教师标签必须落在 code/ 下（kan_swin.py 里 ROOT=code/，路径硬编码）
TEACHER_SRC="${TEACHER_DIR:-}"
TEACHER_DST="$PKG/code/teammate_best"

export PYTHONPATH="$PKG/code:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

say() { echo "[$(date '+%F %T')] $*"; }
hr()  { echo "────────────────────────────────────────────────────────"; }

# ---- 选择要跑的步骤（默认全部）----
STEPS="${*:-1 2 3 4}"
want() { case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }

hr
say "阶段 A 启动 —— 从公开 ImageNet Swin-B 复现基础模型"
say "  训练图像 : $SEG_IMG_DIR"
say "  训练标签 : $SEG_MSK_DIR"
say "  测试图像 : $SEG_TEST_DIR   (应为 test1)"
say "  教师标签 : $TEACHER_DST"
say "  执行步骤 : $STEPS"
hr

# ── 前置检查 ────────────────────────────────────────────────
say "=== 前置检查 ==="
fail=0
for p in "$SEG_IMG_DIR" "$SEG_MSK_DIR" "$SEG_TEST_DIR"; do
  if [ -d "$p" ]; then echo "  OK   $p  ($(ls "$p" 2>/dev/null | wc -l) 张)"; else echo "  MISS $p"; fail=1; fi
done
for f in split.json train_mask_stats.npy; do
  if [ -e "$f" ]; then echo "  OK   $f"; else echo "  MISS $f"; fail=1; fi
done
# 队友教师标签：仅 [A4] 需要
if [ -d "$TEACHER_DST" ] && [ "$(ls "$TEACHER_DST" 2>/dev/null | wc -l)" -gt 0 ]; then
  echo "  OK   $TEACHER_DST  ($(ls "$TEACHER_DST" | wc -l) 张)"
elif [ -n "$TEACHER_SRC" ] && [ -d "$TEACHER_SRC" ]; then
  mkdir -p "$(dirname "$TEACHER_DST")"
  ln -sfn "$TEACHER_SRC" "$TEACHER_DST"
  echo "  OK   已链接 $TEACHER_SRC → $TEACHER_DST"
else
  if want 4; then
    echo "  WARN $TEACHER_DST 缺失 —— [A4] 队友蒸馏需要它；可在运行时用 TEACHER_DIR=<路径> 指定"
  else
    echo "  --   $TEACHER_DST 缺失（本次不跑 [A4]，无影响）"
  fi
fi
[ "$fail" = "1" ] && { say "!! 数据缺失, 终止"; exit 1; }

# ══════════════════════════════════════════════════════════
#  [A1] 基础模型：噪声剔除 + 暗化增广
# ══════════════════════════════════════════════════════════
if want 1; then
  hr
  say "=== [A1] 基础模型 16k, crop=$A1_CROP, $A1_EXCLUDE_NOISE ==="
  say "  预期: val 0.7630（含根因修复；未修复线为 0.7535）"
  "$PY" -u code/kan_swin.py --train $A1_EXCLUDE_NOISE \
      --crop "$A1_CROP" --target-iters "$A1_ITERS" --max-seconds "$A_MAX_SECONDS"
  [ -f code/ckpt_kannoise_i16000.pth ] || { say "!! 未产出 ckpt_kannoise_i16000.pth, 终止"; exit 1; }
  say "[A1] 完成 → code/ckpt_kannoise_i16000.pth"
fi

# ══════════════════════════════════════════════════════════
#  [A2] 伪标签生成
# ══════════════════════════════════════════════════════════
if want 2; then
  hr
  say "=== [A2] 伪标签生成 → $A2_PL_CACHE ==="
  "$PY" -u code/pl_gen.py --ckpt code/ckpt_kannoise_i16000.pth --cache "$A2_PL_CACHE"
  say "[A2] 完成"
fi

# ══════════════════════════════════════════════════════════
#  [A3] PL 自蒸馏微调
# ══════════════════════════════════════════════════════════
if want 3; then
  hr
  say "=== [A3] PL 自蒸馏微调 crop=$A3_CROP, ${A3_ITERS}it, τ=$A3_PL_TAU ==="
  say "  预期峰值: $A3_PEAK (一致率 0.6804)"
  "$PY" -u code/kan_swin.py --train --pl --pl-tau "$A3_PL_TAU" --pl-cache "$A2_PL_CACHE" \
      --warm-base code/"$A3_FROM" --ckpt-name "$A3_TO" \
      --crop "$A3_CROP" --target-iters "$A3_ITERS" \
      --restart-lr "$A_RESTART_LR" --max-seconds "$A_MAX_SECONDS"
  [ -f code/"$A3_PEAK" ] || { say "!! 未产出 $A3_PEAK, 终止"; exit 1; }
  say "[A3] 完成 → code/$A3_PEAK"
fi

# ══════════════════════════════════════════════════════════
#  [A4] 队友蒸馏（终点）
# ══════════════════════════════════════════════════════════
if want 4; then
  hr
  say "=== [A4] 队友蒸馏 crop=$A4_CROP, ${A4_ITERS}it ==="
  say "  预期峰值: $A4_PEAK (一致率 0.7586 单调升)"
  [ -d "$TEACHER_DST" ] || { say "!! 缺 $TEACHER_DST, 无法蒸馏; 用 TEACHER_DIR=<路径> 指定"; exit 1; }
  "$PY" -u code/kan_swin.py --train --distill \
      --warm-dist code/"$A3_PEAK" --ckpt-name "$A4_TO" \
      --crop "$A4_CROP" --target-iters "$A4_ITERS" \
      --restart-lr "$A_RESTART_LR" --max-seconds "$A_MAX_SECONDS"
  [ -f code/"$A4_PEAK" ] || { say "!! 未产出 $A4_PEAK, 终止"; exit 1; }
  say "[A4] 完成 → code/$A4_PEAK"
fi

hr
say "★ 阶段 A 完成"
say "  阶段 B 的暖启动权重: $PKG/code/${A4_PEAK}"
say "  下一步: WARM_CKPT=$PKG/code/${A4_PEAK} bash scripts/00_reproduce_all.sh stage-b"
hr

cat <<'NOTE'

── 合规说明（第三阶段交付须写明）─────────────────────────────
[A4] 的教师标签来自同队队友的模型预测（teammate_best），属共享上游资产。
按赛事要求，交付材料需补齐其模型、训练代码、权重来源与生成命令；
若赛事方另有口径，按赛事方口径处理。
本队最终输出为单模型，该教师标签仅用于训练期的知识蒸馏，不构成多模型集成。
──────────────────────────────────────────────────────────
NOTE
