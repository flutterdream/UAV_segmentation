#  无人机航拍语义分割 —— 可复现交付包

> **任务**：无人机航拍图像语义分割（8 类 + Ignore）
> **模型**：DeepKANSeg-Swin-B（Swin-B 编码器 + KAN 解码器，98.3M 参数）
> **方法**：DACS 域自适应在线自训练 + 锚定伪标签回填 + 类权重定向调制
> **起点**：公开 ImageNet 预训练 Swin-B（`encoder_weights="imagenet"`，学术公开权重）
> **当前成绩**：第二阶段 test2 ****（配方 = relaxbg + batch8 + ms2x TTA）
> **本包内容**：完整训练代码、验证代码、复现脚本、环境锁定、数据说明

---

★ **完整链路（公开学术权重 → ）见独立文档：
[`从公开权重复现_完整链路.md`](./从公开权重复现_完整链路.md)**

---

## 目录

- [一、代码说明](#一代码说明)
- [二、环境说明](#二环境说明)
- [三、操作说明](#三操作说明)
- [附录 A：方法原理](#附录-a方法原理)
- [附录 B：竞赛输出格式](#附录-b竞赛输出格式)
- [附录 C：常见问题](#附录-c常见问题)
- [★ 从公开权重复现完整链路（独立文档）](./从公开权重复现_完整链路.md)

---

# 一、代码说明

## 1.1 目录结构

```
output_pkg/
├── Dockerfile                    # 镜像构建（基于官方 PyTorch 2.6.0 + CUDA 12.4）
├── requirements.txt              # Python 依赖精确锁定
├── README.md                     # 本文档
├── code/                         # ★ 代码（自包含，无外部依赖）
│   ├── dacs_train.py             # 主入口：训练 / 推理 / 源域评估
│   ├── seg_swin.py               # 数据管线 + Swin 编码器 + 多尺度 TTA 推理
│   ├── kan_swin.py               # KAN 解码器 + 模型构建 / 权重加载
│   ├── check_output.py       # 输出文件格式校验（7 项）
│   ├── eval_perclass.py          # 逐类 IoU 评估（验证代码）
│   ├── preflight_gpu.py          # 起跑前显存闸门
│   ├── audit_anchor.py           # 锚定预测质量审计
│   └── dist_compare.py           # 类别分布对比（诊断域偏移）
├── configs/
│   └── best.env                  # 最佳配方参数（超参 / 类权重 / TTA）
├── scripts/
│   ├── 00_reproduce_all.sh       # ★★ 一键全流程（评审只需跑这一个）
│   ├── 01_train.sh               # 仅训练
│   ├── 02_infer.sh               # 仅推理出输出包
│   └── 03_eval_val.sh            # 源域 val 逐类评估（验证代码）
├── data/
│   ├── train/{images,masks}      # 训练集（图像 + 标签）
│   ├── test/images               # 待推理测试集
│   └── anchor/                   # 低置信像素回填用的锚定预测
├── split.json                    # 训练/验证划分（固定，与_stats.npy          # 各类像素频率统计（加权采样用）
└── weights/
    └── ckpt_kan_noise_dist_i4000.pth   # 暖启动权重
```

## 1.2 模块职责

| 模块 | 职责 | 关键点 |
|---|---|---|
| `dacs_train.py` | 训练主循环、教师打标、DACS 混合、推理与打包 | `--train` / `--infer` / `--eval-ms2` 三种模式 |
| `seg_swin.py` | 数据集类 `UAV`、强增广、`predict_logits` 多尺度 TTA | 含 `--scale-up` 尺度增广修正 |
| `kan_swin.py` | KAN 解码器、`build_model`、`load_trained` | Swin-B 骨干从 timm 加载 ImageNet 权重 |
| `check_output.py` | 7 项输出自检 | 数量 / 格式 / 尺寸 / 像素值域 |
| `eval_perclass.py` | 逐类 IoU + 混淆结构 | ⚠️ 见下方「关于 val 分数的说明」 |

## 1.3 数据流

```
                    ┌─────────────────────────────────┐
   源域有标签数据    │  UAV Dataset                    │
   (train/images,   │  强增广: 随机尺度 0.67~1.33      │──┐
    train/masks)    │  随机翻转 / 旋转 / 色彩抖动       │  │
                    └─────────────────────────────────┘  │
                                                          ▼
                    ┌─────────────────────────────────┐  ┌──────────────┐
   目标域无标签数据  │  TargetPool                     │  │  ClassMix    │
   (test/images)    │  随机图 + 随机 crop              │─▶│  伪标签类别的 │
                    └─────────────────────────────────┘  │  一半像素粘贴 │
                                  │                       └──────┬───────┘
                                  ▼                              │
                    ┌─────────────────────────────────┐          │
                    │  EMA 教师（fp16 推理）           │          │
                    │  置信度 τ=0.85 筛选              │          │
                    │  低置信 → 锚定预测回填            │          │
                    └──────────────┬──────────────────┘          │
                                   ▼                             ▼
                            ┌────────────────────────────────────────┐
                            │  学生模型（Swin-B + KAN 解码器）         │
                            │  损失 = 边界CE + Dice + 0.5·Lovász       │
                            │  类权重 w 定向调制（CE_WEIGHTS）          │
                            └────────────────┬───────────────────────┘
                                             ▼
                                       EMA 更新 (0.999)
                                             │
                                             └──▶ 反馈给教师
```

## 1.4 关于 val 分数的说明（⚠️ 评审须知）

本项目在实赛中观察到一个**重要现象**：

> **源域 val 分数与竞赛。**

实证：某次配置调整使 val mIoU 从 `0.7528` 降至 `0.7483`（**下降**），
而竞赛.9709`（**上升 0.2066**）。

**原因**：`split.json` 划分出的验证集属于**源域**，模型在其上已接近饱和；
而线自训练后的效果。两者优化的方向并不一致。

**因此**：
- `03_eval_val.sh` 的用途是**回归检查**（确认代码正确、模型未崩溃、逐类结构合理），
- **不应**将其作为模型选择的唯一依据。
- 本包提供的训练配置，是**以竞赛最优配方。

---

# 二、环境说明

## 2.1 基础镜像

```
pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
```

选择官方镜像的原因：CUDA 12.4 / cuDNN 9 / torch 2.6.0 三者版本已由官方对齐，
避免自行编译 torch 引入版本漂移。

## 2.2 Python 依赖（精确锁定）

| 包 | 版本 | 用途 |
|---|---|---|
| torch | 2.6.0+cu124 | 镜像自带 |
| timm | 1.0.28 | Swin-B 骨干 + ImageNet 预训练权重 |
| segmentation-models-pytorch | 0.5.0 | Dice / Lovász 损失函数 |
| numpy | 2.5.0 | 数值计算 |
| pillow | 12.2.0 | 图像读写 |
| opencv-python-headless | 4.10.0.84 | 边界权重图的形态学运算（无 GUI 依赖） |

## 2.3 硬件要求

| 场景 | 最低显存 | 说明 |
|---|---|---|
| 训练（crop 1024 / bs 2） | **≥ 8 GB** | 实测峰值 6.12 GB（allocated），留出冗余 |
| 训练（crop 640 / bs 2） | ≥ 5 GB | 实测峰值 3.93 GB |
| 推理（TTA=ms2x） | ≥ 4 GB | — |

> 若显存不足，请用 `crop 640` 并**同时移除** `--scale-up`
> （`--scale-up` 仅在 crop > 683 时才有意义）。但这会损失全图视野带来的增益。

## 2.4 网络依赖

Swin-B 的 ImageNet 预训练权重由 `timm` 从 HuggingFace 拉取，
镜像已设置 `HF_ENDPOINT=https://hf-mirror.com`。

**完全离线运行**：首次运行后缓存位于 `/workspace/.hf_cache`，
可将其预置进镜像（见 `Dockerfile` 中注释行 `COPY hf_cache/ ...`）。

## 2.5 复现性说明

- 所有随机种子固定：`--seed 7`（数据处理与 PyTorch 均设种子）
- `split.json` 中训练/验证划分**随包提供**，不重新生成 —— 保证与ckpt_kan_<TAG>.pth`，可多次复用于推理，避免重复训练成本

---

# 三、操作说明

## 3.1 构建镜像

```bash
cd output_pkg
docker build -t aic-seg .
```

> 镜像体积约 15 GB（含基础镜像、依赖与数据）。
> 若数据不进镜像，见 [3.4 挂载外部数据]。

## 3.2 一键完整复现（推荐）

```bash
docker run --gpus all -v $(pwd)/output:/workspace/output aic-seg
```

不带参数 = **完整链路（阶段 A + 阶段 B）**，依次执行：

| 阶段 | 步骤 | 动作 | 参考耗时 |
|---|---|---|---|
| A | A1 | 基础模型 16k（crop 512，噪声剔除 + 暗化增广） | 数小时（以实跑日志为准） |
| A | A2 | 伪标签生成（对 test1 跑 ms2） | ≈ 15 min |
| A | A3 | PL 自蒸馏微调（crop 640 / 4000 it） | ≈ 30 min |
| A | A4 | 队友蒸馏（crop 640 / 4000 it） | ≈ 30 min |
| B | 1/5 | 环境自检（torch / timm / smp / cv2 / CUDA） | < 1 min |
| B | 2/5 | 数据自检 + 显存预检（闸门 free ≥ 6200 MiB 起） | < 1 min |
| B | 3/5 | **训练**（crop 1024 / --scale-up / **batch 8** / 4000 it） | **≈ 3.5 h**（实测 12503 s） |
| B | 4/5 | **推理**（ms2x TTA）+ 打包 + 7 项自检 | ≈ 1 h |

> ⚠️ 阶段 A 的耗时受显存影响较大（crop 512 可用更大 batch）。
> 表中为参考值，**请以实际运行日志为准**；每段受 `--max-seconds` 限制，
> 超时退出后**重跑同一条命令即从快照续训**，不会重复计算。

**若只需复现阶段 B**（已有阶段 A 产出的暖启动权重），用 `stage-b` 可省去数小时：

```bash
docker run --gpus all \
  -e WARM_CKPT=/workspace/weights/ckpt_kan_noise_dist_i4000.pth \
  -v /host/weights:/workspace/weights \
  -v $(pwd)/output:/workspace/output \
  aic-seg bash scripts/00_reproduce_all.sh stage-b
```

**产出**：
```
output_kan_dacs_cw7_1024su_relaxbg_b8.zip
output/           # 日志与输出包副本
```

## 3.3 分步执行

```bash
# 只做自检（不训练不推理，~1 分钟）
docker run --gpus all aic-seg bash scripts/00_reproduce_all.sh check

# 只训练
docker run --gpus all -v $(pwd)/output:/workspace/output aic-seg bash scripts/01_train.sh

# 只推理（用已有权重，零训练成本）
docker run --gpus all -v $(pwd)/output:/workspace/output aic-seg bash scripts/02_infer.sh

# 源域 val 逐类评估（验证代码）
docker run --gpus all aic-seg bash scripts/03_eval_val.sh 120
```

## 3.4 挂载外部数据（不把数据打进镜像）

若希望镜像体积更小，或需要用**新数据**复现，可在运行时覆盖路径：

```bash
docker run --gpus all \
  -v /host/train/images:/data \
  -v /host/train/masks:/mdata \
  -v /host/test/images:/test \
  -v /host/anchor:/anchor \
  -e DATA_DIR=/data -e MASK_DIR=/mdata \
  -e TEST_DIR=/test -e ANCHOR_DIR=/anchor \
  -v $(pwd)/output:/workspace/output \
  aic-seg
```

支持的环境变量：

| 变量 | 含义 | 默认值 |
|---|---|---|
| `DATA_DIR` | 训练图像目录 | `data/train/images` |
| `MASK_DIR` | 训练标签目录 | `data/train/masks` |
| `TEST_DIR` | 待推理测试集目录 | `data/test/images` |
| `ANCHOR_DIR` | 锚定预测目录 | `data/anchor` |
| `PY` | Python 解释器路径 | `python` |

> ⚠️ **锚定预测与测试图像必须文件名一一对应**（同为 `test1_1.png` 或同为 `test2_1.png`）。
> 若不一致，锚回填会自动跳过 —— 训练仍可进行，但伪标签质量下降。

## 3.5 调整配方

配方参数集中在 `configs/best.env`，可直接编辑或用环境变量覆盖：

```bash
# 例：改用 640 视野训练（显存不足时）
docker run --gpus all -e CROP=640 -e SCALE_UP="" aic-seg bash scripts/01_train.sh
```

主要参数：

| 参数 | 当前值 | 说明 |
|---|---|---|
| `CROP` | 1024 | 训练裁剪尺寸（全图视野，本项目最大增益来源） |
| `BATCH` | 2 | 批大小 |
| `TARGET_ITERS` | 4000 | 训练迭代数 |
| `SEED` | 7 | 随机种子 |
| `CE_WEIGHTS` | `0.40,0.65,0.78,0.94,1.66,0.37,1.25,2.59` | 8 类 CE 权重（Bg→Vehicle） |
| `BOOST` | `1.10,1.15,1.10,1.00,1.50,1.00,1.30,1.50` | 8 类采样频率权重 |
| `TTA` | `ms2x` | 多尺度 TTA 档位 |

## 3.6 验证产出

推理完成后脚本会自动运行 7 项自检：

1. 文件数量正确
2. 全部为 PNG 格式
3. 文件命名扁平（无子目录）
4. 文件名前缀一致
5. 每张图尺寸符合预期
6. **像素值严格落在 1..8**（类别 id 约定）
7. zip 可正常解压

全部通过才会输出 `★ 完成` 并生成输出包。

---

# 附录 A：方法原理

## A.1 核心思路

竞赛的关键难点是**域偏移**：训练数据（源域）与测试数据（目标域）在
地表覆盖、光照、季节上存在差异。直接迁移会导致大量像素被误判为背景。

本方案采用 **DACS（Domain Adaptation via Cross-domain Mixed Sampling）**
的在线自训练框架，配合三项关键改进：

### ① ClassMix 跨域混合

从目标域伪标签中随机选取**一半类别**，将这些类别的像素（图像 + 标签）
粘贴到源域有标签样本上。这样学生模型在每个 batch 中同时见到：
- 源域的**正确标签**
- 目标域的**伪标签**

从而在保持源域判别能力的同时适应目标域分布。

### ② 锚定伪标签回填

纯自训练的伪标签会**自我强化错误**。本方案引入「锚定预测」：
对教师置信度低于阈值 τ 的像素，用锚定预测回填而不是丢弃。

> **关键发现**：锚的质量决定学生上限，但**仅在师生共享同一观测窗口时成立**。
> 一旦学生获得教师训练时**未观测过**的信息维度（如全图视野），
> 上限可被打破 —— 这是本项目的重要经验之一。

### ③ 类权重定向调制

两个可调旋钮作用于背景类：
- `CE_WEIGHTS[0]`：背景类的交叉熵权重
- `BOOST[0]`：背景类的采样频率权重

**实测发现**：在 640 视野下，压制背景是有效的（补偿视野不足导致的假背景）；
但在 1024 全图视野下，**该压制从「补偿」变为「负担」** ——
因为全局语境本身已把背景压到合理比例。

本项目扫描得到单调线：

| CE_Bg / boost_Bg |
|---|---|
| 0.60 / 1.50 |  |
| 0.50 / 1.30 |  |
| **0.40 / 1.10** | **** ← 当前最佳 |

> **推广规律**：换到新视野后，旧视野下调出来的每个旋钮都要重新审视。

## A.2 损失函数

```
L = (边界加权CE + Dice) + 0.5 × Lovász(half-resolution)
```

- **边界加权 CE**：用 5×5 形态学膨胀/腐蚀生成边界带，边界像素权重 ×3.0
- **Lovász**：直接优化 IoU 的凸代理，在半分辨率上计算以节省显存
- 三项互补：CE 逐像素、Dice 区域级、Lovász 直接对齐评测指标

## A.3 推理：多尺度 TTA

`ms2x` 档位对输入做 **9 路**推理并平均概率：

| 缩放倍率 | 1.00 | 0.75 | 1.25 | 1.50 | 0.50 | 1.75 |
|---|---|---|---|---|---|---|
| 是否含水平翻转 | ✓ | ✓ | ✓ | — | — | — |
| 路数 | 2 | 2 | 2 | 1 | 1 | 1 |

**实测**：仅**最克制**的档位为正增益（+0.0089）；
扰动翻倍的档位实测为负（−0.0473 / −0.0814），且**扰动越大分越低**。
故最终只保留 `ms2x`。

---

# 附录 B：竞赛输出格式

| 项 | 约定 |
|---|---|
| 打包 | zip |
| 内容 | 每张测试图对应一个 PNG，**扁平存放**（无子目录） |
| 命名 | 与输入图同名（如 `test1_1.png`） |
| 尺寸 | 与输入图一致 |
| **像素值** | **类别 id 严格取 1..8**（0 为 Ignore，不出现） |

**类别 id 约定**：

| id | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| 类别 | Ignore | Background | Building | Road | Water | Barren | Vegetation | Agricultural | Vehicle |

> ⚠️ 注意：`CE_WEIGHTS` 与 `BOOST` 数组长度为 **8**，
> 对应 **类别 1..8**（不含 Ignore）。顺序恒为：
> `Background, Building, Road, Water, Barren, Vegetation, Agricultural, Vehicle`。

---

# 附录 C：常见问题

**Q1：训练时报 `CUDA out of memory`**
A：显存不足。检查 `nvidia-smi` 的 free 值；脚本已内置闸门（要求 ≥ 6200 MiB）。
可选：改用 `CROP=640`（并移除 `--scale-up`），或用更大显存的卡。

**Q2：报 `MISS data/train/images`**
A：数据未打进镜像或挂载路径不对。检查 `DATA_DIR` 等环境变量，
或确认 `data/` 目录已随镜像构建。

**Q3：训练很慢，日志长时间无输出**
A：若 `nvidia-smi` 显示 GPU 利用率接近 100% 但日志不推进，
可能是**显存分页（thrash）**。本项目对此有专门防护：
- ckpt 一律先加载到 CPU
- 目标图像缓存上限 48 张
- 起跑前显存闸门
若仍发生，说明显存余量确实不够，需降 `CROP`。

**Q4：能否跳过训练直接复现推理？**
A：可以。`bash scripts/02_infer.sh` 会使用包内已有权重直接推理。
但若需要**完整复现训练流程**，请跑 `00_reproduce_all.sh`。

**Q5：`--scale-up` 是做什么的？**
A：`crop > 图边长 / 1.5` 时，随机尺度增广的上限会被图边长截断，
导致增广退化为「只放大、不缩小」。`--scale-up` 改为先放大原图再取整幅，
保住 0.67~1.33 倍的完整尺度分布。仅在 crop > 683 时有效。

**Q6：单次运行成本参考**
A：AutoDL RTX 4080S 约 ¥1.58/小时。
一轮 4000 iter 的 1024 训练约 58 分钟，加推理约 80 分钟 ⇒ **约 ¥2.1/轮**。
