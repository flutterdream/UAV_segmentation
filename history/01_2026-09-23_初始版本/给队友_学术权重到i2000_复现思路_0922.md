# 从学术权重到 i2000 的完整复现思路

> **文档目的**：向队友交付「学术 Swin-B 权重 → `ckpt_kandist_i2000.pth`」这条早期链路的复现方案。
> **证据等级标注**：✅ 有据可查（代码/文件实证） · ⚠️ 逻辑推断（需实测确认） · ❌ 证据缺失
> 日期：2026-09-22

---

## 零、先说结论：哪些能交、哪些交不了

| 环节 | 产物 | 脚本 | 证据等级 |
|---|---|---|---|
| ① 学术权重 → 16k | `ckpt_kan_swin_16k.pth` | 推断为 `kan_swin.py` 训练主循环 | ⚠️ 推断 |
| ② 生成伪标签 | `pl_cache/*.npz` | ✅ `pl_gen.py` | ✅ 实证 |
| ③ 自蒸馏 → 2000it | `ckpt_kanpl_i2000.pth` | ❌ **未找到专属脚本** | ❌ 缺失 |
| ④ 教师蒸馏 → 2000it | `ckpt_kandist_i2000.pth` | ❌ **未找到专属脚本** | ❌ 缺失 |
| ⑤ 快照选拔 | （评估用） | ✅ `distill_agree.py` | ✅ 实证 |
| — 日志 | — | Aug 2~18 全空白 | ❌ 缺失 |

**⇒ 现有材料可直接交的只有 ②⑤ 两个工具 + 3 个 ckpt 本身。①③④ 需要按本文档的思路重建。**

---

## 一、环境（已验证，可直接用）

```
torch        2.6.0+cu124
timm         1.0.28
smp          0.5.0        （import 名 segmentation_models_pytorch）
numpy        2.5.0
Pillow       12.2.0
opencv       4.10.0
```

**⚠️ 已知陷阱**：包名是 `segmentation_models_pytorch`，代码里写作 `import segmentation_models_pytorch as smp`。
直接 `import smp` 会 `ModuleNotFoundError` —— 别被这个误导以为自己环境坏了。

---

## 二、数据与划分（✅ 实证）

- **文件**：`split.json`（2.9 KB）
- **训练数据**：`official/train/train/images`（6996 张）+ `masks`（6998 张）
- **测试集**：`official/test_1/images`（500 张，第一阶段）
- **类别**：0=Ignore, 1=Bg, 2=Building, 3=Road, 4=Water, 5=Barren, 6=Veg, 7=Agri, 8=Vehicle
- **输出格式**：PNG 单通道，像素值恒 1..8

**环境变量覆盖**（后期加的，向后兼容）：
```bash
export SEG_IMG_DIR=...   SEG_MSK_DIR=...   SEG_TEST_DIR=...
```

---

## 三、环节 ① 学术权重 → `ckpt_kan_swin_16k.pth`

### 3.1 架构（✅ 实证）

```python
# kan_swin.py
model = DeepKANSegSwin(NUM_CLASSES, b2_kan=b2_kan, enc=enc)   # enc 默认 "swin_b"
```

**Swin-B 主干 + DFR + KAN 解码器，98.3 M 参数。**

**⚠️ 排除项（重要）**：`seg_official.py` 与 `seg_train.py` 的 docstring 都写着
「SegFormer-B5」—— **它们不是这条链的脚本**，别照名字找错。

### 3.2 ckpt 结构（✅ 实证）

```python
torch.save(dict(
    model=model.state_dict(),
    opt=opt.state_dict(),
    sched=sched.state_dict(),
    iter=it,
    hist=hist,
    ema=ema,            # ← 推理走这个
), path)
```

**加载约定**：`load_trained(use_ema=True)` ⇒ 推理读 `sd["ema"]`；续训读 `sd["model"]+opt+sched+iter`。

**文件名规律**：`ckpt_kan_<tag>.pth`（最终） / `ckpt_kan_<tag>_i<N>.pth`（仅 `N % 1000 == 0` 时写）

### 3.3 重建命令（⚠️ 推断）

```bash
cd <workdir>
python -u kan_swin.py --train \
  --tag swin_16k \
  --target-iters 16000 \
  --max-seconds 9000 \
  --seed 42
```

**关键待确认项**：
1. 预训练权重从哪加载 —— 是 `timm.create_model("swin_base_patch4_window7_224", pretrained=True)` 还是本地 `.pth`？
2. 数据划分用的是 `split.json` 的哪一段
3. `--max-seconds` 需配合外部循环（脚本支持断点续训，每次跑满即存盘退出）

**成本估算**：16k it @ 640 crop ≈ **2~3 h**（4080S）

---

## 四、环节 ② 生成伪标签（✅ 实证，可直接跑）

### 4.1 脚本原文（`pl_gen.py`，2.1 KB）

```python
"""P1.1: 用 16k 冠军 + ms2(8路TTA) 为 test1 生成 伪标签+逐像素置信度.
   产物: pl_cache/<name>.npz {pred:uint8, conf:float16} (1024x1024), 可断点续跑.
   用法: python pl_gen.py [--ckpt ckpt_kannoise_i16000.pth] [--cache pl_cache_noise]"""

ap.add_argument("--ckpt",  type=str, default="ckpt_kan_swin_16k.pth")   # ← 默认正是 16k
ap.add_argument("--cache", type=str, default="pl_cache")
...
lg = S.predict_logits(model, img, tta="ms2")      # 8 路 TTA
p  = torch.softmax(lg, dim=0)
conf, pred = p.max(dim=0)
np.savez_compressed(fp, pred=..., conf=conf.half())
```

### 4.2 运行

```bash
python pl_gen.py --ckpt ckpt_kan_swin_16k.pth --cache pl_cache
```

**特性**：✅ **可断点续跑**（`if os.path.exists(fp): continue`）· ✅ 自带置信度分布统计（τ=0.8/0.85/0.9/0.95 各保留多少像素）

**成本**：500 张 × ms2 8 路 ≈ **20~30 min**

---

## 五、环节 ③④ 自蒸馏 / 教师蒸馏（❌ 脚本缺失，需重建）

### 5.1 已确认的事实

- `ckpt_kanpl_i2000.pth`（Aug 7）→ 同系列最终版 `ckpt_kan_pl.pth`（Aug 10）
- `ckpt_kandist_i2000.pth`（Aug 10）→ 同系列最终版 `ckpt_kan_distill.pth`（Aug 10）
- `distill_agree.py` **不是训练脚本**，是快照选拔工具：

```python
"""蒸馏快照选拔: 各快照在 test1 上与队友教师标签的一致率(mIoU口径).
   教师=榜67.33 -> 一致率越高,= os.path.join(ROOT, "teammate_best")
用法: python distill_agree.py [N] ckpt1 ckpt2 ...
```

### 5.2 重建思路（⚠️ 按已有接口反推）

**两段蒸馏的共同点**：都是「以 16k 为起点，用某种监督信号再训 2000 it，
每 1000 it 存一个快照，最后用 `distill_agree.py` 从快照里选最优」。

#### 环节 ③ 自蒸馏（伪标签监督）

**监督信号来源** = 环节 ② 产出的 `pl_cache/*.npz`
```
pred : uint8  (1024×1024)   伪标签
conf : float16               逐像素置信度（用于加权/阈值过滤）
```

**训练逻辑（推断）**：
1. 加载 `ckpt_kan_swin_16k.pth` 的 EMA 作为初始化
2. 对每张 test1 图读入 `pl_cache` 的 `pred` 与 `conf`
3. **损失 = 置信度加权的 CE**（弱置信像素降权），可选叠加 Dice
   - `conf` 的存在就是为了这个 —— 否则不必存它
   - 可能的过滤阈值：`conf >= τ`（τ 从 `pl_gen.py` 输出的分布里选）
4. 训 2000 it，存 `ckpt_kanpl_i2000.pth`

**为什么是 2000 it 而不是 4000**：伪标签有噪声，训久了会过拟合到自己的错误上（自我强化偏差）。
2000 it 是"够用就停"的保守量。

#### 环节 ④ 教师蒸馏

**监督信号来源** = `teammate_best/`（500 张 PNG，队友模型在 test1 上的预测）**：
1. 加载 `ckpt_kanpl_i2000.pth`（或 16k）作为初始化
2. 以 `teammate_best` 的 500 张作为**硬标签**，做监督训练
3. 或做**软蒸馏**（需教师的 logits，但 `teammate_best` 只有 argmax 后的 PNG ⇒ **只能是 hard label**）
4. 训 2000 it，存 `ckpt_kandist_i2000.pth`

**⚠️ 一个重要判断**：因为 `teammate_best` 是**单通道 PNG（1..8）**，没有概率分布
⇒ 环节 ④ 只能是「**硬标签蒸馏**」，不是真正的知识蒸馏（无 logits/T）。
这一点写进文档能帮队友避免误解。

### 5.3 数据划分的关键细节

`teammate_best` 的文件名是 `test1_1.png … test1_500.png`，与 `official/test_1/images` **一一对应**。
所以：
- **训练集** = test1 的 500 张（无标签，配教师预测当标签）
- **验证** = 用 `split.json` 里源域的那一段跑 `eval_perclass.py`

**⚠️ 这里存在一个合规性问题需要队友自行判断**：把 test1 的教师预测当训练标签，
等于在**测试集上做自训练** —— 竞赛规则是否允许，需按赛事方口径确认
（我们第一阶段/第二阶段都这么做了，但正式交付文档里应注明）。

---

## 六、环节 ⑤ 快照选拔（✅ 实证）

```bash
python distill_agree.py 50 ckpt_kanpl_i1000.pth ckpt_kanpl_i2000.pth ckpt_kandist_i1000.pth ckpt_kandist_i2000.pth
```

输出每个快照与 `teammate_best` 的逐类 IoU 均值，**一致率高 ⇒。
只取前 50 张（`N=50`）做横向比较，省时间。

**注意**：这只是**选拔**，不是提升 —— 它的价值在于从多个快照里挑出最优的那一个。

---

## 七、完整复现路线图（按依赖排序）

```
[公开 Swin-B 预训练权重]
        │  ① 监督训练 16k it  （kan_swin.py，⚠️ 需确认预训练加载方式）
        ▼
  ckpt_kan_swin_16k.pth            ← 队友要的 ckpt_kan_i16000.pth
        │  ② 伪标签生成（pl_gen.py，ms2 8路TTA，输出 pred+conf）
        ▼
  pl_cache/*.npz
        │  ③ 置信度加权自蒸馏 2000 it  （❌ 脚本缺失，按 §5.2 重建）
        ▼
  ckpt_kanpl_i2000.pth
        │  ④ 用 teammate_best 做硬标签蒸馏 2000 it（❌ 脚本缺失，按 §5.2 重建）
        ▼
  ckpt_kandist_i2000.pth           ← 队友的接续点
        │  → 队友已掌握：18 → 47 → 49 → 
```

---

## 八、交付清单现状

| 项 | 状态 |
|---|---|
| 3 个 ckpt | ✅ 全在（`swin_16k` / `kanpl_i2000` / `kandist_i2000`，均为 783 MB~1.57 GB） |
| `pl_gen.py` | ✅ 可直接交 |
| `distill_agree.py` | ✅ 可直接交 |
| `teammate_best/` 500 张 | ✅ 在（5.2 MB） |
| `split.json` | ✅ 在 |
| 环境清单 | ✅ 已验证 |
| **①③④ 训练命令** | ⚠️ **需重建**（本文档给思路） |
| **各阶段训练日志** | ❌ **缺失**（Aug 2~18 空白） |

---

## 九、给队友的建议

1. **先验证血脉**：拿 `ckpt_kandist_i2000.pth` 直接跑 `distill_agree.py`，
   看它与 `teammate_best` 的一致率是否显著高于随机 —— 若一致率很低，说明这条链
   **接不上**他手上的 warm，得重新评估。
2. **①③④ 用重建的脚本跑一遍**，既补上日志，也能验证脚本与 ckpt 确实对应
   （现在只是"时间戳看起来吻合"，没有硬证据）。
3. **命名统一**：`ckpt_kan_swin_16k.pth` ≡ 他说的 `ckpt_kan_i16000.pth`，交付时加一份对照表。

---

## 十、命名对照表

| 队友的称呼 | 我们的实际文件名 |
|---|---|
| `ckpt_kan_i16000.pth` | `ckpt_kan_swin_16k.pth` |
| `ckpt_kanpl_i2000.pth` | `ckpt_kanpl_i2000.pth`（同名） |
| `ckpt_kandist_i2000.pth` | `ckpt_kandist_i2000.pth`（同名） |
| （未提） | `ckpt_kan_pl.pth` / `ckpt_kan_distill.pth`（各自最终版，1.57 GB） |
| （未提） | `ckpt_kannoise_i16000.pth` / `ckpt_kan_noise_pl_i2000.pth` / `ckpt_kan_noise_dist_i4000.pth`（Aug 23 noise 平行链） |

**⚠️ 最后一行是风险点**：我们交给队友的 warm 是 **noise 版**（`ckpt_kan_noise_dist_i4000.pth`，Aug 23），
而 Aug 10 这条链严格来说**不一定**是它的祖先。**这个血脉关系尚未验证**，建议优先做 §9.1 的一致性检查。
