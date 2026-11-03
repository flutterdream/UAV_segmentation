# 算法代码说明（前序全链，历史与追溯用）

团队编号：AIC-2026-59375991　团队：流水的天才　队长：周宇桐

---

## 一、这个目录是什么

本目录保存**前序阶段**的完整训练与推理代码、配置，以及复现说明。
它产出的是父权重 R0（。

**扩展阶段的最终输出的代码与权重不在这里**，而在与本科目录并列的 `代码与数据/`：

| 需要什么 | 去哪里 |
|---|---|
| **复现 的推理** | `代码与数据/README.md`（主文档）+ `代码与数据/code/inference.py` |
| 四段解码器微调的训练代码 | `代码与数据/code/train.py` |
| 最终权重 | `代码与数据/checkpoints/best_model.pth` |
| 复现父权重 R0（） | 本目录 `REPRODUCE_FROM_SCRATCH.md` |

换句话说：**本目录 = 上游（教师链 → Swin/KAN 链 → BN 统计混合 → R0）**，
`代码与数据/` = 下游（R0 → 四段微调 →）。

## 二、目录内容

```
AIC-2026-59375991-算法代码/
├── code/                 教师、学生、伪标签、蒸馏、BN 统计、预测与验证代码
├── scripts/entrypoint.py verify / reproduce / infer / evaluate 统一入口
├── configs/              前序全链的完整参数
├── public_pretrained/    公开学术权重（Swin-B ImageNet、SegFormer-B2、DINOv2-small）
├── DATA_SETUP.md         官方数据目录与准备说明（数据集不随包提供）
├── REPRODUCE_FROM_SCRATCH.md   从公开权重到 R0 的完整链条
├── REPRODUCE_DECODER_CHAIN.md  R0 到 的四段微调（与代码与数据/ 重复提供一份）
└── 归档校验/             SHA-256 清单与只读完整性检查工具
```

## 三、运行环境

本包**不使用 Docker**。环境用 Conda 创建，完整命令见
`代码与数据/README.md` 第 2.1 节（Python 3.12.3 / torch 2.5.1+cu124 / timm 1.0.30 /
segmentation-models-pytorch 0.5.0）。

前序阶段曾用三套隔离环境（teacher / kan / stage47），其版本锁定清单保留在本目录
`public_pretrained/` 与 `配置` 中；**扩展阶段的推理与微调只需上述单一环境即可**。

## 四、公开权重说明

`public_pretrained/` 内为学术公开权重，公开预训练权重只注明名称与下载地址，不随包提供文件本体。
此处保留是为便于离线复现：

| 权重 | 用途 | 公开地址 |
|---|---|---|
| Swin-B `swin_base_patch4_window7_224.ms_in22k_ft_in1k` | 编码器初始化 | https://huggingface.co/timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k |
| SegFormer-B2 `mit_b2` | 教师链编码器 | https://huggingface.co/smp-hub/mit_b2.imagenet |
| DINOv2-small | 教师链域特征聚类 | https://huggingface.co/facebook/dinov2-small |

三者**均不参与扩展阶段最终推理**，最终推理只用 `代码与数据/checkpoints/best_model.pth`。

## 五、完整性检查

```bash
python 归档校验/VERIFY_SUBMISSION.py
```

该工具核对文件 SHA-256、测试图名一致性，以及结果 ZIP 的 PNG 数量、尺寸、标签范围与 CRC。

## 六、数据

仅使用赛事官方提供的训练集与测试集，数据集不随包提供，准备方式见 `DATA_SETUP.md`。
