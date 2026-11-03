# 代码与复现操作说明

团队编号：。团队：流水的天才。队长：周宇桐。

最终方案为Swin-B + DFR + GLKAN单模型，使用Stage47 i600 EMA、静态BN混合alpha0.75和八路概率平均。归档结果对应第二阶段成绩。

## 目录与入口

- code：教师、学生训练、伪标签、蒸馏、BN统计、预测与验证代码。
- scripts/entrypoint.py：verify、reproduce、infer、evaluate统一入口。
- configs：完整复现参数。
- public_pretrained：公开Swin-B、SegFormer-B2和DINOv2-small学术权重。
- DATA_SETUP.md：赛事方官方数据的目录与挂载说明；数据集不包含在输出包中。
- 归档校验：SHA-256清单及只读完整性检查工具。

先按DATA_SETUP.md准备赛事方提供的官方数据。环境构建及运行命令见DOCKER_RUN.md；完整训练阶段、参数、输入和硬件要求见REPRODUCE_FROM_SCRATCH.md。

## 从公开权重训练

使用reproduce命令，从空work目录开始；输入为赛事方提供的官方数据、本包公开权重和源码。教师、学生、中间锚预测、伪标签及最终BN统计均由本轮生成，不读取相邻模型目录中的比赛权重。教师训练的epoch32/epoch1及Swin v4.1的iter3400恢复均由复现脚本显式执行，恢复点来自本轮新生成的断点。

## 最终模型预测

使用infer命令，加载相邻“模型文件与运行环境”目录中的唯一KAN权重的ema字段。SegFormer与DINOv2仅参与训练，最终推理不加载它们。正式预测应覆盖test2全部1300张；--limit仅用于短程检查。

## 完整性检查

在本目录运行：`python 归档校验/VERIFY_OUTPUT.py`。该工具核对文件SHA-256、原始测试图名以及结果ZIP的PNG数量、尺寸、标签范围和CRC。

技术方案PDF和模型/结果目录与本目录并列，共四个顶层项。数据仅用于赛事评委复现，不公开传播。
