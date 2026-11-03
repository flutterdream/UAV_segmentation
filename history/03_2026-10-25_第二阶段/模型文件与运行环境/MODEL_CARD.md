# 唯一最终模型

Swin-B + DFR + GLKAN，9输出通道，0为忽略类别，最终PNG实测标签1至8。
唯一文件：ckpt_kan_s2_i600_c1_alpha075_20261003.pth。
推理固定使用ema字段；model字段是同一网络的训练状态，仅原样保留以保持历史文件指纹，不参与预测，也不进行双状态融合。
Stage47：crop1024，真实batch8，seed17，选i600，LR调度周期4000。
静态BN：test2全部1300张，单尺度，alpha0.75，41层82buffers，无梯度更新。
最终八路概率平均是同一只模型的多视角预测；不读取SegFormer/DINO或第二只KAN预测。
真实；plain EMA n500诊断0.7566282292。
SHA256：60c499b9d860f2dd71ef964f047d4dbeed47a3b724f75f8a0b59a78bcf46952d。
