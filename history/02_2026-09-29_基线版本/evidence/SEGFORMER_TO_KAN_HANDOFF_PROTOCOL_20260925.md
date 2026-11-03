# 历史 SegFormer 教师 → KAN PL → SegFormer 蒸馏一次性续接协议

时间：2026-09-25 08:05（Asia/Hong_Kong）

## 性质与边界

本协议只为白天无人值守的一次性本地续接。它不修改、不终止、不恢复当前历史 SegFormer 教师队列，只每 300 秒读取一次队列状态。教师队列只有明确写出 `complete` 后才允许进入后续步骤；`failed`、`stopped`、`paused_by_user`、`blocked` 或未知状态均立即停止，不自行修复、改参或重跑。

本轮禁止 AutoDL、47、49、test2 推理、打包和输出。所有产物只能称为本地候选；test2 包未生成、未归档、无真实。

## 启动前只读事实

- 教师队列 PID：`8288`；当前 C5 子进程 PID：`37504`。
- 当前状态：`running`，当前 C5 日志已到 epoch 2 step 5750/6996，未发现异常退出。
- `resume_count=2`，且 `trajectory_equivalent_to_uninterrupted=false`。该事实永久保留，不得宣称随机轨迹完全复现。
- C5/C6 及身份核验仍由现有教师队列自然完成；本续接不启动第二个 GPU 训练进程。
- KAN 基座：`ckpt_kan_from_v4_1_7439_16k.pth`，SHA-256 `806039602f21cb6da342085f867727712a9c06f40c8765c031791c07a5882798`，只使用 EMA。
- test1 自伪标签缓存固定为 `pl_cache_kan_from_v4_1_7439_ms2`；必须在训练前逐项核验其 KAN 血统、EMA/ms2 路线与 500 张官方 test1 对应关系。
- tau=0.85 的本轮全像素覆盖率为 60.8063%，历史记录约 88.2%；统计口径是否相同尚未证实。本轮把它作为历史差异警告而非硬门槛，但固定 tau=0.85，禁止降阈值或扫描。

## 新教师选择与复制

只读取教师队列生成的 `historical_anchor_identity.json`：

1. C5/C6 任一与历史 500 张锚逐文件完全一致时选择完全一致者；若两者都完全一致，以历史链中更晚的 C6 作为确定性同分规则。
2. 若都不完全一致，选择全部像素一致率更高者，且必须至少 90%；同分时选择 C6。
3. 两者都低于 90% 时停止，不运行 C7/C8，不做后处理或类别/阈值调整。
4. D 盘历史锚不复制、不进入训练；只复制本次从公开权重与官方数据重新生成的 C5/C6 预测。

目标目录 `teacher_anchors_segformer_historical_rebuilt` 在复制前不存在。先完整验证源文件，再一次性创建目录；目录内只能有恰好 500 张、与 test1 同名、1024×1024、单通道、像素值 1–8 的 PNG。manifest 放在目录旁，记录逐文件 SHA-256、总摘要、教师 checkpoint 血统及 SHA-256、C5/C6 身份与历史一致率、推理参数。

## 固定训练顺序与守卫

严格顺序：`KAN 16k EMA → 固定 PL 4000 → PL-i2000 → SegFormer 硬标签 anchor 4000 → 蒸馏-i2000`。

PL 固定参数：tau0.85、crop640、batch2、seed42、4000 iter、restart LR 0.25、伪标签样本权重 3×，其余完全使用冻结 `kan_swin.py`。单进程连续运行，保留 i1000/i2000/i3000/i4000，但只选择 i2000。i2000 完整 n500 plain EMA 必须至少 0.7300，八个有效类别 IoU 必须有限、非零且不低于 0.05，否则停止。

蒸馏只能从本轮 PL-i2000 EMA 热启动，只使用本轮重建 SegFormer PNG，走 `--anchor` 而非旧 `--distill`。crop640、batch2、seed42、4000 iter、restart LR 0.25、教师样本权重 3×；同样固定选择 i2000。完成后执行完整 n500 plain EMA、对本轮教师的 test1 ms2 一致率、各类别 IoU 与预测占比检查；守卫同 PL。

## 时间规则

- 教师在 19:30 或之后才完成：只完成教师选择/锚核验，不启动 PL。
- PL 在 19:30 前启动：允许自然完成并评估。
- PL 完成及评估时已到 19:30：不启动蒸馏。
- 蒸馏在 19:30 前启动：允许自然完成，即使略超过 20:00。

## 冻结源码 SHA-256

- `kan_swin.py`: `c84d5c884a17fff40257b137b08ed08bc1729243a9884439b315b8132ccdb9af`
- `seg_swin.py`: `d33ff136e70ba1d2474372909e78d6af46f14562838bcaa6f33985d3ea648f0d`
- `strict_env.py`: `1675f72661f6663870293732851bae9c055d7edd6f19f7ee2744f4d1cfb101f9`
- `strict_eval.py`: `7f7da8b9534f54f824413a53ecd459b4acbdd90bce219ef271f93992940fbd54`
- `pl_gen.py`: `be95121991a4332b306e3d4ea08c99c0d4b5181c566ee1f2d9c712de6eab7e18`
- `daytime_segformer_pl_distill_handoff.py`: `6c03958672ec0ab22168fccc48de346ae60d8a9df00916f2f583808c1a4a6d95`
- `audit_kan_segformer_teacher_consistency.py`: `8356c3404b731e44f82f52b210d340bd32e9f388be66384b6b75ff6de1b915c7`

续接脚本和一致性审计均已通过 Python 语法编译检查。续接状态写入 `runs/strict/segformer_pl_distill_handoff_status.json`，总报告持续写入 `runs/strict/SEGFORMER_PL_DISTILL_STATUS_20260925.md`，不覆盖既有证据。
