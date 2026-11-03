# 权重说明

## 一、本仓库提供什么

| 文件 | 大小 | SHA256 |
|---|---|---|
| `best_model.pth` | 391,818,938 B | `873bb246c889012c9e873ce9c2df4a946a30e8cd116de383f8eb0967ced832b9` |

这是**最终模型**（四段约束微调链的最后一环）。加载它即可复现报告中的全部指标与预测结果。

## 二、完整权重链

最终模型不是一次训练出来的，而是从 R0 出发、经四段串联微调得到的。链条如下：

| 环节 | 文件名 | 大小 | 是否随仓库提供 |
|---|---|---|---|
| 起点（父权重） | `ckpt_kan_s2_i600_c1_alpha075_20261003.pth`（简称 R0） | 783.7 MB | ❌ **未提供** |
| 第 1 段 | `v2.pth` | 391.8 MB | ❌ 未提供 |
| 第 2 段 | `r2.pth` | 391.8 MB | ❌ 未提供 |
| 第 3 段 | `r3.pth` | 391.8 MB | ❌ 未提供 |
| 第 4 段（最终） | `best_model.pth`（即 sw_s12800） | 391.8 MB | ✅ **已提供** |

中间三段的权重可以按 `docs/repro/REPRODUCE_DECODER_CHAIN.md` 的命令重训得到，因此不必存档。

## 三、为什么 R0 没有提供

**存储配额限制。** 本仓库的大文件走 Git LFS，免费额度为 1 GB：

```
已用（best_model.pth + predictions.zip）   430 MB
R0 单个文件                                 784 MB
──────────────────────────────────────────────────
合计                                        1.21 GB  超出额度
```

因此无法把 R0 与本仓库放在一起。

## 四、没有 R0 时怎么复现

按目标分三种情况：

### 情况 A：只想得到报告的结果

**可以直接做。** `best_model.pth` 已包含全部参数，推理即可：

```bash
python inference.py --ckpt weights/best_model.pth \
                    --images /path/to/test/images \
                    --out ./pred_png --zip ./predictions.zip
```

验证指标：

```bash
python evaluate.py --ckpt weights/best_model.pth \
                   --images data/train/images --masks data/train/masks \
                   --val-ids data/val_ids.json
```

### 情况 B：想重跑四段微调链

**需要先得到 R0。** 两条途径：

1. 按 `docs/repro/REPRODUCE_FROM_SCRATCH.md` 从公开学术权重（SegFormer-B2、DINOv2-small、Swin-B）从头训练，
   该文档给出了完整的多环境 Docker 配方与每一步超参；
2. 如果你有历史存档，直接使用对应的 R0 权重，并按
   `docs/repro/REPRODUCE_DECODER_CHAIN.md` 提供的四条命令依次执行。

拿到 R0 后，四段微调本身是轻量的：只训练解码器（2.67 M 参数），
四段合计约 15 小时（单卡 V100，四段串行）。

### 情况 C：想完全从零训练

**可以做到，但成本较高。** 按 `docs/repro/REPRODUCE_FROM_SCRATCH.md` 走完整链路：
教师模型（SegFormer-B2 + DINOv2 域特征）→ 伪标签生成 → KAN 主链训练 → 检查点筛选 → BN 统计混合 → 四段微调。

需要准备：
- 官方数据集（训练 6996 张 1024×1024，含标注；测试集无标注）
- 公开预训练权重（Swin-B、SegFormer-B2、DINOv2-small，约 500 MB，需自行下载）
- 三个隔离的 Python 环境（teacher / kan / stage47，版本要求见该文档）

## 五、内容结构

`torch.load` 后得到一个字典，包含两套**完整**参数与元信息：

| 键 | 说明 |
|---|---|
| `ema` | **滑动平均权重，推理必须用这一套** |
| `model` | 训练结束时的即时权重，仅供追溯，不用于预测 |
| `iter` | 训练步数（本权重为 12800） |
| `parent` | 父权重文件名（本权重为 `r3.pth`） |
| `val500_miou` | 训练侧的验证集指标（0.766757） |

> ⚠️ 两套参数形状完全相同，**加载错不会报错，只会掉精度**。
> `inference.py` 已默认读取 `ema`；若自行加载请确认这一行：
>
> ```python
> state = payload["ema"] if "ema" in payload else payload
> ```

## 六、模型规模

| 模块 | 参数量 | 状态 |
|---|---|---|
| 编码器（Swin-B） | 86.74 M | 冻结 |
| DFR（KAN 精炼） | 8.47 M | 冻结 |
| 解码器（GLKAN） | 2.67 M | 可训练 |
| 合计 | 97.81 M | — |

## 七、加载示例

```python
import torch
from models.deepkanseg import DeepKANSegSwin, NUM_CLASSES

payload = torch.load("weights/best_model.pth", map_location="cpu", weights_only=False)
model = DeepKANSegSwin(NUM_CLASSES, enc_weights=None)   # 权重由 ckpt 提供，无需联网
model.load_state_dict(payload["ema"], strict=True)      # 严格加载，缺失/多余应为 0
model.eval()
```

## 八、预期校验

在 500 张固定验证集上，本权重的 mIoU 应为 **0.7668**（训练侧记录）。
在不同 GPU 上独立复测可能在 0.7642 – 0.7668 之间浮动，属正常范围
（详见 `docs/results/RESULTS.md` 第 1 节）。

若加载后明显低于此值，先检查是否误用了 `model` 键。
