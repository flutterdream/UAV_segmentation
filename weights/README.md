# 权重说明

## 文件

| 文件 | 大小 | SHA256 |
|---|---|---|
| `best_model.pth` | 391,818,938 B | `873bb246c889012c9e873ce9c2df4a946a30e8cd116de383f8eb0967ced832b9` |

## 内容结构

`torch.load` 后得到一个字典，包含两套**完整**参数与元信息：

| 键 | 说明 |
|---|---|
| `ema` | **滑动平均权重，推理必须用这一套** |
| `model` | 训练结束时的即时权重，仅供追溯，不用于预测 |
| `iter` | 训练步数（本权重为 12800） |
| `parent` | 父权重文件名（本权重为 `r3.pth`） |

> ⚠️ 两套参数形状完全相同，**加载错不会报错，只会掉精度**。
> `inference.py` 已默认读取 `ema`；若自行加载请确认这一行：
>
> ```python
> state = payload["ema"] if "ema" in payload else payload
> ```

## 模型规模

| 模块 | 参数量 | 状态 |
|---|---|---|
| 编码器（Swin-B） | 86.74 M | 冻结 |
| DFR（KAN 精炼） | 8.47 M | 冻结 |
| 解码器（GLKAN） | 2.67 M | 可训练 |
| 合计 | 97.81 M | — |

加载方式：

```python
import torch
from models.deepkanseg import DeepKANSegSwin, NUM_CLASSES

payload = torch.load("weights/best_model.pth", map_location="cpu", weights_only=False)
model = DeepKANSegSwin(NUM_CLASSES)
model.load_state_dict(payload["ema"], strict=True)   # 严格加载，缺失/多余应为 0
model.eval()
```

## 预期校验

在 500 张固定验证集上，本权重的 mIoU 应为 **0.766757**。
若加载后明显低于此值，先检查是否误用了 `model` 键。
