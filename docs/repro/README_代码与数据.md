# AIC-2026 无人机低空航拍图像语义分割 — 扩展阶段代码与数据

**小组编号**：AIC-2026-59375991　**小组名称**：流水的天才　**队员**：周宇桐、刘俊希
**扩展阶段IoU，测试集 3594 张，取历史最好一次）

本包整理了完整训练代码与配套材料，包含
推理代码、配置文件、实际训练权重与最终预测结果，可直接按本文档复现推理。

---

## 一、目录结构

```
代码与数据/
├── code/
│   ├── train.py                     训练入口（四段解码器约束微调）
│   ├── inference.py                 推理入口（单模型 8 路概率平均）
│   ├── configs/
│   │   └── decoder_finetune.yaml    完整训练配置与四段复现命令
│   └── models/
│       └── deepkanseg.py            模型定义（自包含，无本地路径依赖）
├── checkpoints/
│   └── best_model.pth               实际训练并用于生成预测的权重（391.8 MB）
├── predictions/
│   └── predictions.zip              最终预测结果（3594 张 PNG，37.6 MB）
└── README.md                        本文件
```

---

## 二、环境配置

### 2.1 创建 Conda 环境

```bash
conda create -n aic2026 python=3.12.3 -y
conda activate aic2026

# PyTorch（CUDA 12.4）
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 其余依赖
pip install timm==1.0.30 \
            segmentation-models-pytorch==0.5.0 \
            numpy==2.1.3 \
            pillow==11.0.0 \
            opencv-python-headless==5.0.0.93
```

### 2.2 版本核对

实际训练与推理所用版本（已在本包代码上验证一致）：

| 组件 | 版本 |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.5.1+cu124 |
| torchvision | 0.20.1+cu124 |
| timm | 1.0.30 |
| segmentation-models-pytorch | 0.5.0 |
| numpy | 2.1.3 |
| Pillow | 11.0.0 |
| opencv-python-headless | 5.0.0.93 |
| 显卡 | Tesla V100-SXM2-32GB（推理显存约 6.8 GB；≥12 GB 的卡均可） |

### 2.3 公开预训练权重说明

公开预训练权重只注明名称与下载地址，不随包提供文件本体。
本方案仅使用以下学术公开权重，**未使用任何商业闭源模型或额外数据**：

| 用途 | 权重名称 | 来源 |
|---|---|---|
| 编码器初始化 | Swin-B `tu-swin_base_patch4_window7_224` | https://huggingface.co/timm/swin_base_patch4_window7_224.ms_in1k |
| 教师链语料聚类（仅前序阶段，本包推理不涉及） | DINOv2-small | https://huggingface.co/facebook/dinov2-small |
| 教师链编码器（仅前序阶段，本包推理不涉及） | SegFormer-B2 `mit_b2` | https://huggingface.co/smp-hub/mit_b2.imagenet |

> 编码器权重在 `models/deepkanseg.py` 中通过 `segmentation_models_pytorch.UPerNet(encoder_weights="imagenet")`
> 自动下载。若离线环境，请预先将上述权重放入 timm/HuggingFace 缓存目录。

---

## 三、推理（复现）

```bash
cd 代码与数据/code

python inference.py \
    --ckpt ../checkpoints/best_model.pth \
    --images /path/to/test_3/images \
    --out ../predictions/png \
    --zip ../predictions/predictions.zip
```

### 3.1 推理流程（与最终成绩完全一致）

1. 读取 1024×1024 原始测试图，RGB
2. **八路 TTA**，每路独立前向：

   | 序号 | 缩放 | 水平翻转 |
   |---|---|---|
   | 1 | 1.00 | 否 |
   | 2 | 1.00 | 是 |
   | 3 | 0.75 | 否 |
   | 4 | 0.75 | 是 |
   | 5 | 1.25 | 否 |
   | 6 | 1.25 | 是 |
   | 7 | 1.50 | 否 |
   | 8 | 0.50 | 否 |

3. 每路：ImageNet 归一化 → fp16 前向 → logits 双线性插值回 1024×1024 → softmax →
   若翻转则把概率在宽度维翻回 → 累加
4. 八路**等权算术平均**，`argmax` 取类别
5. 输出单通道 PNG，像素值 1–8（0 为忽略类，不输出），文件名与测试图一致

### 3.2 关键说明（容易出错的三点）

- **必须加载 `payload["ema"]`**。权重文件里同时存有 `model` 与 `ema` 两套完整参数，
  前者是训练结束时的即时权重，后者是滑动平均权重。输出结果用的是 `ema`，
  加载错**不会报错**，只会掉分。`inference.py` 已默认取 `ema`。
- **BN 统计已固化在权重中**，推理全程 `model.eval()`，不做任何统计重估。
- **无任何后处理**：不做连通域过滤、不做孔洞填充、不做类别偏置，`argmax` 后仅做值域检查。

---

## 四、训练

最终权重由四段连续微调得到，每段以"上一段权重"为父模型，并用父解码器的软预测作 KL 约束。

```bash
cd 代码与数据/code

# 需要的输入：官方训练集 6996 张（images / masks 同名成对）
DATA=/path/to/official_train
R0=/path/to/R0.pth      # 前序阶段最终权重

python train.py --images $DATA/images --masks $DATA/masks \
                --parent $R0   --out v2.pth        --steps 1600  --batch 4
python train.py --images $DATA/images --masks $DATA/masks \
                --parent v2.pth --out r2.pth        --steps 3200  --batch 4
python train.py --images $DATA/images --masks $DATA/masks \
                --parent r2.pth --out r3.pth        --steps 6400  --batch 4
python train.py --images $DATA/images --masks $DATA/masks \
                --parent r3.pth --out best_model.pth --steps 12800 --batch 4
```

### 4.1 训练设计（这是本方案的核心）

| 项 | 设置 |
|---|---|
| 冻结 | 编码器（86.74 M）、DFR（8.47 M）、全部 BatchNorm running 统计 |
| 可训练 | 仅解码器（2.67 M） |
| 损失 | `0.20 × L_sup + 1.0 × L_KL` |
| `L_sup` | 边界加权 CE + Dice + 0.5·Lovász |
| `L_KL` | 父解码器在 256² 上的软预测，通道 1–8 |
| 优化器 | AdamW，lr 1e-5 余弦退火至 1e-6，wd 0.01 |
| 精度 | bfloat16 自动混合，梯度裁剪 1.0 |
| EMA | 0.99（仅解码器，BN buffer 沿用父权重） |
| 增强 | 仅几何：水平翻转 0.5 / 垂直翻转 0.5 / 旋转 90° 0.25 |
| 数据顺序 | 固定种子打乱后**顺序消费**，保证任一步可精确对齐 |

完整超参与四段结果见 `code/configs/decoder_finetune.yaml`。

### 4.2 四段训练结果

| 阶段 | 累计 crop | 验证集 mIoU ||
|---|---|---|---|
| 父模型 R0 | — | 0.753938  |
| v2 | 6400 | 0.758314  |
| r2 | 12800 | 0.761884 | 未归档 |
| r3 | 25600 | 0.764587  |
| **sw（最终）** | **51200** | **0.766757** | — |

> 验证集为训练集 6996 张中固定划出的 500 张，不参与训练，仅用于监控收敛。

---

## 五、权重与预测的一致性

| 项 | 值 |
|---|---|
| 权重 `checkpoints/best_model.pth` | 391,818,938 B　SHA256 `873bb246c889012c9e873ce9c2df4a946a30e8cd116de383f8eb0967ced832b9` |
| 预测 `predictions/predictions.zip` | 37,569,220 B　SHA256 `6a337e66218f402474bfe32c11331a26fe938556ec6cc1e41e5a4a4a1330d974` |
| 包内文件 | 3594 张单通道 PNG，1024×1024，像素值 1–8，平铺无子目录 |

对应关系：`best_model.pth` 仅被 `inference.py` 使用过一次生成上述 zip，该 zip 即。
三者（权重 / 代码 / 预测）严格一致。

校验命令：

```bash
sha256sum checkpoints/best_model.pth predictions/predictions.zip
python - <<'PY'
import zipfile, numpy as np, io
from PIL import Image
z = zipfile.ZipFile("predictions/predictions.zip")
ns = [n for n in z.namelist() if n.endswith(".png")]
assert len(ns) == 3594 and all("/" not in n for n in ns)
a = np.array(Image.open(io.BytesIO(z.read(ns[0]))))
assert a.shape == (1024, 1024) and a.dtype == np.uint8 and 1 <= a.min() and a.max() <= 8
print("OK", len(ns))
PY
```

---

## 六、复现边界（如实说明）

1. **跨环境不保证逐字节一致**。不同 GPU 型号、驱动或 PyTorch 版本下，fp16 累加顺序与
   cuDNN 算法选择会变化，生成的 PNG 可能有极少数像素差异。判据应以"重新上传评测、
   分数接近"为准，而非文件哈希相同。
2. **权重哈希必须一致**。若 `best_model.pth` 的 SHA256 与上表不符，说明文件被改动或传输损坏，
   此时任何分数都不成立。
3. **训练可复现性**。四段训练使用了固定随机种子（42）与固定数据顺序，理论上可重训出等价权重；
   但跨环境的浮点差异会使重训权重与归档权重不完全相同（属正常现象），应以重新评测结果为准。
4. **本包只覆盖扩展阶段阶段**。父模型 R0 的完整训练链条（教师链、Swin/KAN 链、BN 统计混合）
   见前序交付包中的 `REPRODUCE_FROM_SCRATCH.md`；本包从 R0 之后开始。

---

## 七、合规声明

- **单一模型**：全程单模型推理，八路 TTA 为同一模型的多视图概率平均，**不构成模型集成**。
- **数据**：仅使用赛事官方提供的训练集与测试集，未引入任何外部数据。
- **预训练权重**：仅使用第二节列出的学术公开权重，未使用商业闭源模型。
- **复现材料**：训练代码、推理代码、配置、权重、预测结果齐备，可按本文档完成推理复现。
