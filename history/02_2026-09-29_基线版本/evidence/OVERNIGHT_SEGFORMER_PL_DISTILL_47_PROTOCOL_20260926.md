# 夜间一次性条件队列：C6锚 → PL → 蒸馏 → 47包

本轮只使用E盘严格复现链生成的KAN16k、ms2缓存、本轮C6预测、官方train/test1和源码。D盘历史锚只在此前完成的身份审计中读过；本轮不读取D盘任何权重、预测或标签。

顺序和门槛冻结为：固化本轮C6锚；KAN16k EMA → tau.85/crop640/batch2/seed42/4000/restart LR .25的PL；固定评估PL-i2000 n500；通过0.7300及类别守卫后，从PL-i2000 EMA进行相同crop/batch/seed/iters/restart LR的C6硬标签anchor训练；固定评估蒸馏-i2000；通过后统计与C6锚一致率，再准备47包并只做batch1/iter1非候选冒烟。

47正式配置锁定为crop1024、整图scale-up、真实batch8、seed17、4000 iter、restart LR .25、tau.85、teacher hflip、ignore-anchor-conflict、source_supervised .20、EMA .999、最终ms2。正式47不在本地启动，也不以梯度累积或小batch冒充。

任一阶段失败或门槛不通过即停止。i1000/i3000/i4000只保存，PL和蒸馏正式候选固定为i2000。不启动AutoDL、49、test2、打包或输出，不执行关机脚本。
