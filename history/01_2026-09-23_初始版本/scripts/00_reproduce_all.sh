#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
#   无人机航拍语义分割 —— 一键完整复现
#
#  一条命令跑完整条链：环境自检 → 训练 → 推理 → 打包 → 输出前自检
#
#  用法:
#    bash scripts/00_reproduce_all.sh                 # 完整复现（阶段A + 阶段B）
#    bash scripts/00_reproduce_all.sh full            # 同上（显式）
#    bash scripts/00_reproduce_all.sh stage-a         # 只跑阶段A: 公开权重 → 基础模型(warm ckpt)
#    bash scripts/00_reproduce_all.sh stage-b         # 只跑阶段B: 从 WARM_CKPT 训练 → 推理 → 输出包
#    bash scripts/00_reproduce_all.sh infer-only      # 跳过训练, 直接用已有 ckpt 推理
#    bash scripts/00_reproduce_all.sh check           # 只做环境与数据自检
#
#  ── 两阶段说明 ──
#    阶段A (scripts/00a_stage_a_base_model.sh): 公开 ImageNet Swin-B → 基础模型
#    阶段B (本脚本):                            基础模型 → DACS 训练 → 推理 → 输出包
#    stage-b 模式下, WARM_CKPT 需由外部提供(挂载或 -e 传入)。
#
#  ── 数据路径覆盖（默认用镜像内 data/，也可挂载外部数据）──
#    DATA_DIR=/data     训练集 images 所在目录（内含 0000.png …）
#    MASK_DIR=/data     训练集 masks 所在目录
#    TEST_DIR=/test     待推理的测试集 images 所在目录
#    ANCHOR_DIR=/anchor 低置信像素回填用的锚定预测目录
#    例: docker run --gpus all \
#          -v /host/train/images:/data -v /host/train/masks:/mdata \
#          -v /host/test:/test -v /host/anchor:/anchor \
#          -v $(pwd)/output:/workspace/output aic-seg
#
#  ⚠️ 目标域图像与锚定预测必须【文件名一一对应】（同为 test1_1.png 或同为 test2_1.png）。
#     若不一致，锚回填会自动跳过（不影响训练，但会损失一部分伪标签质量）。
# ══════════════════════════════════════════════════════════════════════════
set -u

MODE="${1:-full}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(dirname "$HERE")"
cd "$PKG"

# ---- 载入最佳配方 ----
# shellcheck disable=SC1091
source configs/best.env

# ---- Python 解释器（容器内即 python；本地可用 PY 覆盖）----
PY="${PY:-python}"
command -v "$PY" >/dev/null 2>&1 || { echo "!! 找不到 Python 解释器: $PY（可用 PY=/path/to/python 覆盖）"; exit 1; }

# ---- 路径解析（环境变量优先，否则用包内默认）----
export SEG_IMG_DIR="${DATA_DIR:-$PKG/data/train/images}"
export SEG_MSK_DIR="${MASK_DIR:-$PKG/data/train/masks}"
export SEG_TEST_DIR="${TEST_DIR:-$PKG/data/test/images}"
# 锚目录：外部用 ANCHOR_DIR 传入（若未传，用包内默认）
ANCHOR_PATH="${ANCHOR_DIR:-$PKG/data/anchor}"

# 训练代码在 code/ 下，需把它加入模块搜索路径
export PYTHONPATH="$PKG/code:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

OUTDIR="output_kan_${TAG}"
# 推理输出前缀从测试集文件名自动推断（test1_N.png / test2_N.png）
PREFIX="$(ls "$SEG_TEST_DIR" 2>/dev/null | sed -n 's/^\([A-Za-z0-9]*\)_[0-9]*\.png$/\1/p' | head -1)"
[ -z "$PREFIX" ] && PREFIX="test1"

say() { echo "[$(date '+%F %T')] $*"; }
hr()  { echo "────────────────────────────────────────────────────────"; }

hr
say " 复现流程启动  mode=$MODE"
case "$MODE" in
  full)       say "  将执行: 阶段A(公开权重 → 基础模型) + 阶段B(DACS训练 → 推理 → 输出包)";;
  stage-a)    say "  将执行: 仅阶段A（公开权重 → 基础模型 / 暖启动权重）";;
  stage-b)    say "  将执行: 仅阶段B（需外部提供 WARM_CKPT）";;
  infer-only) say "  将执行: 仅推理（跳过训练）";;
  check)      say "  将执行: 仅环境与数据自检";;
  *)          say "  !! 未知模式: $MODE（可选 full|stage-a|stage-b|infer-only|check）"; exit 1;;
esac
say "  训练图像   : $SEG_IMG_DIR"
say "  训练标签   : $SEG_MSK_DIR"
say "  测试图像   : $SEG_TEST_DIR   (前缀=$PREFIX)"
say "  锚定目录   : $ANCHOR_PATH"
say "  输出       : $OUTDIR"
hr

# ══════════════════════════════════════════════════════════
#  1) 环境自检
# ══════════════════════════════════════════════════════════
say "=== [1/5] 环境自检 ==="
"$PY" - <<'PY' || { say "!! Python 环境不完整, 终止"; exit 1; }
import sys
mods = {}
try:
    import torch
    mods["torch"] = torch.__version__
    mods["cuda_available"] = torch.cuda.is_available()
    mods["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
except Exception as e:
    print(f"  MISS torch: {e}"); sys.exit(1)
for name, mod in [("timm", "timm"), ("smp", "segmentation_models_pytorch"),
                  ("numpy", "numpy"), ("PIL", "PIL"), ("cv2", "cv2")]:
    try:
        m = __import__(mod)
        mods[name] = getattr(m, "__version__", "?")
    except Exception as e:
        print(f"  MISS {name}: {e}"); sys.exit(1)
for k, v in mods.items():
    print(f"  OK   {k:<16} {v}")
if not mods.get("cuda_available"):
    print("  !! 警告: CUDA 不可用, 训练与推理将退化为 CPU(极慢)。请用 --gpus all 启动。")
PY
[ $? -ne 0 ] && exit 1

# ══════════════════════════════════════════════════════════
#  1.5) 阶段 A —— 从公开学术权重复现「基础模型」
#       （仅 full / stage-a 模式执行）
# ══════════════════════════════════════════════════════════
if [ "$MODE" = "full" ] || [ "$MODE" = "stage-a" ]; then
  hr
  say "=== [阶段 A] 从公开 ImageNet Swin-B 复现基础模型 ==="
  if ! bash "$PKG/scripts/00a_stage_a_base_model.sh"; then
    say "!! 阶段 A 失败, 终止"; exit 1
  fi
  A_CKPT="$PKG/code/ckpt_kan_noise_dist_i4000.pth"
  [ -f "$A_CKPT" ] || { say "!! 阶段 A 未产出 $A_CKPT, 终止"; exit 1; }
  export WARM_CKPT="$A_CKPT"
  say "  阶段 A 产物已接管 WARM_CKPT: $WARM_CKPT"
  if [ "$MODE" = "stage-a" ]; then
    hr; say "★ stage-a 模式完成（未进入阶段 B）"; hr; exit 0
  fi
fi

# ══════════════════════════════════════════════════════════
#  2) 数据自检
# ══════════════════════════════════════════════════════════
say "=== [2/5] 数据自检 ==="
fail=0
for p in "$SEG_IMG_DIR" "$SEG_MSK_DIR" "$SEG_TEST_DIR"; do
  if [ -d "$p" ]; then
    n=$(ls "$p" 2>/dev/null | wc -l)
    echo "  OK   $p  ($n 张)"
    [ "$n" -eq 0 ] && { echo "       ^^ 目录为空"; fail=1; }
  else
    echo "  MISS $p"; fail=1
  fi
done
for f in split.json train_mask_stats.npy; do
  if [ -e "$f" ]; then echo "  OK   $f"; else echo "  MISS $f"; fail=1; fi
done
# 暖启动权重: 属阶段B 前提。check 模式下仅告警, 不阻塞环境/数据自检
if [ -e "$WARM_CKPT" ]; then
  echo "  OK   $WARM_CKPT  (暖启动权重)"
elif [ "$MODE" = "check" ]; then
  echo "  WARN $WARM_CKPT 缺失（check 模式不阻塞；运行阶段B 前必须提供）"
else
  echo "  MISS $WARM_CKPT"; fail=1
fi
if [ -d "$ANCHOR_PATH" ]; then
  echo "  OK   $ANCHOR_PATH  ($(ls "$ANCHOR_PATH" | wc -l) 张)  ← 锚回填"
else
  echo "  WARN $ANCHOR_PATH 缺失 → 低置信像素将标记为 Ignore(仍可训练, 但伪标签质量下降)"
fi
[ "$fail" = "1" ] && { say "!! 数据缺失, 终止"; exit 1; }

# ══════════════════════════════════════════════════════════
#  3) 显存预检
# ══════════════════════════════════════════════════════════
say "=== [3/5] 显存预检 ==="
# 本配方(crop 1024 / bs 2 / grad-ckpt) 实测 peak ≈ 6.12GB
"$PY" code/preflight_gpu.py 6200 || {
  say "  提示: 若显存不足, 可改用 crop 640(需同时去掉 --scale-up), 但会损失全图视野增益"
  exit 1
}

if [ "$MODE" = "check" ]; then
  say "=== check 模式: 自检全部通过, 退出 ==="; exit 0
fi

# ══════════════════════════════════════════════════════════
#  4) 训练（可跳过）
# ══════════════════════════════════════════════════════════
CKPT="ckpt_kan_${TAG}.pth"
if [ "$MODE" = "infer-only" ]; then
  say "=== [4/5] 跳过训练（infer-only）==="
  [ -f "$CKPT" ] || { say "!! infer-only 模式需要 $CKPT, 但未找到"; exit 1; }
else
  say "=== [4/5] 训练（crop=$CROP, ${TARGET_ITERS} iters, 约 60 分钟）==="
  mkdir -p output
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
  [ "$rc" -ne 0 ] && { say "!! 训练退出码=$rc, 终止"; exit 1; }
  [ -f "$CKPT" ] || { say "!! 训练未产出 $CKPT, 终止"; exit 1; }
  say "训练完成 → $CKPT"
fi

# ══════════════════════════════════════════════════════════
#  5) 推理 + 打包 + 输出前自检
# ══════════════════════════════════════════════════════════
say "=== [5/5] 推理（TTA=$TTA）→ $OUTDIR ==="
"$PY" code/dacs_train.py --infer --ckpt "$CKPT" --out "$OUTDIR" --tta "$TTA" 2>&1 \
  | tee "output/infer_${TAG}.log"
rc=${PIPESTATUS[0]}
[ "$rc" -ne 0 ] && { say "!! 推理退出码=$rc, 终止"; exit 1; }

ZIP="${OUTDIR}.zip"
[ -f "$ZIP" ] || { say "!! 未生成 $ZIP, 终止"; exit 1; }

say "=== 输出前自检（文件数 / PNG / 尺寸 / 像素值 1..8）==="
"$PY" code/check_output.py "$ZIP"

# 归档到 output/
mkdir -p output
cp -f "$ZIP" "output/$ZIP" 2>/dev/null || true

hr
say "★ 全流程完成"
say "  输出包 : $PKG/$ZIP"
say "  副本   : $PKG/output/$ZIP"
say "  日志   : $PKG/output/train_${TAG}.log  /  output/infer_${TAG}.log"
hr
