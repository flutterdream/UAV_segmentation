# 从公开学术权重开始完整复现

> **适用范围声明**：本文件描述的是**前序阶段**（产出父权重 R0）。
> 其中提到的 Docker 方式**不适用于本次扩展阶段输出**；
> 扩展阶段的最终推理与微调请见 `docs/notes/04_推理说明.md`，环境用 Conda 创建（不使用镜像）。

---

## 输入和初始化

train、test1、test2原始数据不随预测包提供。请按 DATA_SETUP.md 准备官方数据；Docker模式将外部数据目录只读挂载到容器的 `/workspace/output/official_data`。算法配置与训练流程不变。

从空 work 目录执行 python scripts/entrypoint.py reproduce。允许输入为 official_data、public_pretrained、固定 split.json 和从官方标签统计出的 train_mask_stats.npy；程序先验证其指纹。代码中的中间状态全部写入 work，默认拒绝非空 work 和未声明续训。模型文件目录里的比赛权重、历史锚、缓存统计量都不参与从头训练。

安装推荐使用 DOCKER_RUN.md 的三环境 Docker：
- teacher：Python3.9.25，NumPy2.0.2，Pillow11.3.0，timm1.0.28，transformers4.57.6；
- kan：Python3.13.5，NumPy2.1.3，Pillow11.1.0，timm1.0.30；
- stage47：Python3.10.8，同KAN核心包版本；
- 三者 torch2.7.1+cu118、torchvision0.22.1+cu118。
本地执行时用KAN解释器启动，并设置 TEACHER_PYTHON 和 STAGE47_PYTHON 为对应解释器的绝对路径。

## 固定教师链

由公开 SegFormer-B2 与 DINOv2-small 生成：
1. 首先用原始audit_data.py读取全部6996张官方标签，在work/teacher/artifacts/data_audit生成image_stats.csv；再显式传入该CSV重新建立五折分组信息，在work/teacher/artifacts/splits生成folds.csv。所有教师训练阶段显式传入本轮folds.csv，不读取官方数据目录下的历史artifacts。DINO仅提取官方数据的域特征，64簇、40轮、blend0.7，采样权重截断0.35至3.0。
2. A：train-all，50epochs，crop512，batch2，梯度累积2，lr6e-5，seed2026。在epoch32保存后新进程恢复，仍保持50epoch调度周期。
3. B：从A开始8epochs，crop512，lr1e-5，EMA0.9998，目标域采样权重。
4. C1：6epochs，crop768，batch1，累积4，lr3e-6，FDA概率0.35、频带0.005至0.015；在epoch1保存后重启。
5. C3：4epochs，crop896，lr1.5e-6；C4：2epochs，实际crop928，lr5e-7。
6. C5：4epochs，crop1024，lr4e-7，梯度检查点；C6：4epochs，DACS，crop768，lr1e-6，动态教师EMA0.9995，目标损失0.5。
7. 新C6对test1预测，尺度0.75/1/1.25及水平翻转，概率平均后输出硬标签。不会读取历史教师预测。

教师的显式重启从本轮刚生成的last.pt读取；与历史一样恢复模型、优化器、调度器、scaler/EMA，然后在新进程seed2026重新播种。未保存历史RNG，因此不宣称逐位确定性。

## 固定Swin与KAN链

1. Swin-B UPerNet v4：公开Swin-B权重 → crop512，batch2，seed42，12000iter，lr6e-5。
2. v4.1：v4 EMA热启动，crop640，batch2，seed42，4000iter，新优化器lr3e-5；灰度和模糊各10%，禁用暗化和scale-up。在iter3400保存后重启，保持4000iter调度周期并恢复model/opt/sched/EMA。
3. v4/v4.1的随机裁剪下限固定为crop，符合最终0.7439父血统；不使用后来失败的0.75*crop分支。
4. KAN16k：从v4.1 EMA只初始化编码器，crop512，batch2，seed42，16000iter。
5. test1伪标签：KAN16k做历史MS2 logit平均后softmax，保存硬标签和float16置信度；训练阈值0.85。这一历史伪标签路线与最终概率平均推理不同。
6. PL：crop640、batch2、4000iter、restart-lr0.25；选择i2000。
7. C6硬锚蒸馏：从PL-i2000开始，crop640、batch2、4000iter、restart-lr0.25；选择蒸馏i2000。
8. Stage47：从新蒸馏i2000开始，crop1024、真实batch8、seed17、source_supervised0.20、tau0.85、EMA0.999；固定KAN教师在test1上做翻转一致性过滤，并要求与新C6硬锚相同。使用整图放大再裁剪。原1400iter早停括号脚本原样保留，LR余弦周期4000，保存600/800/1000/1200，固定选择i600。
9. 对i600 EMA的41个BatchNorm层，使用全部1300张test2原1024图单尺度eval前向统计输入均值/方差；混合alpha0.75，含组间均值差方差项。只改变82个running_mean/running_var，不改参数或num_batches_tracked。
10. 最终单模型八路概率平均：1.0/翻转、0.75/翻转、1.25/翻转、1.5、0.5。各路先将logits双线性还原1024，再softmax、翻回、平均，argmax生成PNG。

## 质量检查和产物

固定历史门槛：v4 n100>=0.735；v4.1 n500>=0.7224；KAN16k n500>=0.745；PL/蒸馏i2000>=0.7300；最终BN候选n500>=0.74；类别不得非有限或零IoU。失败即保留日志并停止，不调阈值、换种子或自动加载旧权重。

成功输出：
- work/runs/strict/ckpt_kan_stage47_i600_test2_alpha075_retrained.pth
- work/final_predictions/predictions.zip
- work/REPRODUCTION_COMPLETE.json
- work/orchestration_logs、work/teacher/logs 逐阶段日志和本轮权重哈希

n500只用于诊断；教师train-all包含这500张，因此全链n500不是严格独立验证结果。外部评测由隐藏标签平台产生。跨系统/GPU的训练数值会有差异，重训模型和ZIP哈希不必与归档逐字节一致，应重新评测，不能把历史成绩直接赋给重训产物。

## 硬件和时长

推荐单张32GB以上NVIDIA GPU，至少32GB内存和150GB空闲运行盘。历史真实batch8峰值18,591MiB；教师C5峰值约7.95GB。完整教师、学生与预测链粗估24至40小时，取决于GPU和I/O，仅供安排，不作为完成承诺。不能用batch缩小或梯度累积替代Stage47的真实batch8。
