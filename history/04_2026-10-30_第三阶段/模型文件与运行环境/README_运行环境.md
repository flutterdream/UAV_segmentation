# 模型文件与运行环境

本目录保存**第三阶段最终权重**、父权重、以及环境说明。本次输出**不使用 Docker**，
环境通过 Conda 创建（命令见下）。

## 一、本目录内容

| 文件 | 作用 | 是否推理必需 |
|---|---|---|
| `sw_s12800.pth` | **最终权重**checkpoints/best_model.pth` 同一文件） | **是** |
| `ckpt_kan_s2_i600_c1_alpha075_20261003.pth` | 父权重 R0，榜单 ，四段微调的起点 | 否，仅供链条追溯 |
| `MODEL_CARD.md` | 模型结构、推理规范、指标 | — |
| `SHA256.txt` | 两份权重的完整指纹 | — |
| `DECODER_CHAIN_SHA256.txt` | 四段微调链全部五份权重的指纹 | — |
| `environment/` | 依赖版本锁定清单，供核对 | — |

> 训练期使用过的 SegFormer / DINOv2 公开学术权重位于 `算法代码/public_pretrained`，
> **不属于最终输出模型**，仅作历史记录。

## 二、Conda 环境创建

```bash
conda create -n aic2026 python=3.12.3 -y
conda activate aic2026

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install timm==1.0.30 \
            segmentation-models-pytorch==0.5.0 \
            numpy==2.1.3 \
            pillow==11.0.0 \
            opencv-python-headless==5.0.0.93
```

版本核对（实际训练与推理所用）：

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
| 硬件 | Tesla V100-SXM2-32GB（推理显存约 6.8 GB，12 GB 以上的卡均可） |

## 三、推理

```bash
cd 代码与数据/code
python inference.py --ckpt ../checkpoints/best_model.pth \
                    --images /path/to/test_3/images \
                    --out ../predictions/png \
                    --zip ../predictions/predictions.zip
```

完整说明（TTA 定义、加载 `ema` 的注意事项、一致性校验）见 `代码与数据/README.md`。

## 四、权重指纹

```
873bb246c889012c9e873ce9c2df4a946a30e8cd116de383f8eb0967ced832b9  sw_s12800.pth                         391,818,938 B860f2dd71ef964f047d4dbeed47a3b724f75f8a0b59a78bcf46952d  ckpt_kan_s2_i600_c1_alpha075_20261003.pth  783,673,997 B