# 模型文件与运行环境

本目录保存真实最高对应的唯一KAN比赛权重、环境锁定清单和Dockerfile。
训练期SegFormer和DINOv2的公开学术权重位于相邻“算法代码/public_pretrained”，不属于最终测试模型。

Docker构建必须从团队根文件夹执行，以本目录docker/Dockerfile为构建文件；操作命令见相邻“算法代码/DOCKER_RUN.md”。不得直接在docker子目录构建，因为构建上下文需要相邻算法源码。

模型结构、最终推理状态与SHA-256见MODEL_CARD.md及SHA256.txt。完整训练入口只使用赛事方提供并按DATA_SETUP.md挂载的官方数据，以及相邻算法目录的公开学术权重；模型目录中的比赛权重用于最终预测与结果核对。
