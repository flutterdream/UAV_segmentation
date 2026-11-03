# Docker构建与操作

以下命令均从团队根文件夹执行。Linux x86_64，Docker BuildKit和NVIDIA Container Toolkit，驱动支持CUDA11.8。Dockerfile及环境版本锁定在“模型文件与运行环境”；源码在“算法代码”，官方数据由赛事方外部提供。

官方train、test1、test2数据不随输出包提供。按算法代码/DATA_SETUP.md组织外部数据目录，先设置：

```bash
export DATA_ROOT="/absolute/path/to/official_data"
```

下列运行命令从该外部目录只读挂载数据。构建镜像不需要数据集。

## 构建镜像

```bash
docker build --platform linux/amd64 \
  -f "-模型文件与运行环境/docker/Dockerfile" \
  -t aic-59375991: .
```

构建上下文为团队根目录，不是docker子目录。Dockerfile.dockerignore位于Dockerfile旁边，过滤外部数据、最终比赛权重、预测ZIP和运行产物，避免无关文件进入构建器。该命名方式依据Docker官方文档：https://docs.docker.com/build/concepts/context/#filename-and-location。

镜像包含teacher/kan/stage47三个解释器，仅打包代码和公开学术权重；官方数据与正式模型只读挂载。构建时需要访问公开镜像、Conda-forge、Anaconda官方main源及PyPI，完成后可离线运行。teacher保持Python 3.9.25，因conda-forge缺少该Linux版本，单独从Anaconda官方main源安装，不降级解释器；kan和stage47继续使用conda-forge。

## 输入验证

```bash
docker run --rm --gpus all --network none --ipc=host \
  -v "${DATA_ROOT}:/workspace/output/official_data:ro" \
  aic-59375991: verify
```

## 复核已输出最高分模型

```bash
mkdir -p "$PWD/-算法代码/work_inference"
docker run --rm --gpus all --network none --ipc=host \
  -v "${DATA_ROOT}:/workspace/output/official_data:ro" \
  -v "$PWD/-模型文件与运行环境:/workspace/models:ro" \
  -v "$PWD/-算法代码/work_inference:/workspace/output" \
  aic-59375991: infer \
  --checkpoint /workspace/models/ckpt_kan_s2_i600_c1_alpha075_20261003.pth \
  --output-dir /workspace/output
```

生成算法代码/work_inference/predictions.zip。固定单模型EMA、八路概率平均。--limit只用于冒烟，不能用于正式1300张结果。任务结果文档内保留的是原始最高分ZIP，不会被重算输出覆盖。

## 从公开权重完整训练

```bash
mkdir -p "$PWD/-算法代码/work"
docker run --rm --gpus all --network none --ipc=host \
  -v "${DATA_ROOT}:/workspace/output/official_data:ro" \
  -v "$PWD/-算法代码/work:/workspace/output/work" \
  aic-59375991: reproduce
```

work必须为空；此命令不挂载最终比赛模型。全部中间权重、教师锚、伪标签与BN统计量由本轮从公开权重和官方数据生成。入口明确重放历史教师A epoch32、C1 epoch1及v4.1 iter3400重启，恢复的断点只允许来自本次训练。质量闸门失败保留日志，不调阈值或加载旧权重。

## 验证代码

```bash
docker run --rm --gpus all --network none --ipc=host \
  -v "${DATA_ROOT}:/workspace/output/official_data:ro" \
  -v "$PWD/-模型文件与运行环境:/workspace/models:ro" \
  -v "$PWD/-算法代码/work_inference:/workspace/output" \
  aic-59375991: evaluate \
  --checkpoint /workspace/models/ckpt_kan_s2_i600_c1_alpha075_20261003.pth \
  --output /workspace/output/evaluation_n500.json
```

固定seed123的500张、plain EMA，参考mIoU0.7566282292。教师train-all包含这500张，因此它不是全链独立验证，不能当成真实。

## 环境与结果检查

镜像内包含teacher、kan、stage47三个独立解释器。环境版本以随包锁定文件及Dockerfile为准；构建时分别执行pip check。最终模型采用单EMA权重，固定八路概率平均。跨平台训练应以相同算法配置与官方评价结果核对，而非要求重训文件哈希完全相同。

完整性校验从“算法代码”目录执行：python 归档校验/VERIFY_OUTPUT.py。

## 下载缓存加速（可选，不改变版本）

Stage47环境复用教师环境中同版本的Linux通用NVIDIA依赖文件及发行元数据，不复制cp39的torch/torchvision扩展。仍安装cp310的PyTorch 2.7.1+cu118和torchvision 0.22.1+cu118，并执行pip check及GPU运行检查。

若已有官方Linux cp310 torch安装包，可放在单独缓存目录，并用BuildKit参数 `--build-context wheelcache=/absolute/path/to/cache` 加速；未提供该参数时仍从官方索引安装，不依赖作者电脑路径。缓存文件应命名为 `torch-2.7.1+cu118-cp310-cp310-manylinux_2_28_x86_64.whl`，使用前核对官方SHA-256：`89433c62b02ec802d4c0887c867d935887ae8f00d7cc549ecf1c2640d096bd4c`。

可选缓存还可包含官方 `triton-3.3.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`，SHA-256为 `b74db445b1c562844d3cfad6e9679c72e93fdfb1a90a24052b03bb5c49d1242e`；未提供时仍从官方索引获取。
