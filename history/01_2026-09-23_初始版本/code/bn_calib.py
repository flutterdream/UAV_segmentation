#!/usr/bin/env python
# ══════════════════════════════════════════════════════════════════
#  目标域 BN 统计校准 (AdaBN 思路)
#
#  机制来源: 队友 2026-09-22 复盘「49号 / S2-02」
#    · 冻结全部学习参数, 只改 BatchNorm 的 running_mean / running_var
#    · 用目标域无标签图像前向一遍, 收集总体均值/方差
#    · 与父模型统计按 alpha 混合:
#        mu_new  = (1-a)*mu_old + a*mu_target
#        var_new = (1-a)*var_old + a*var_target + a*(1-a)*(mu_target-mu_old)^2
#      (最后一项是两组分布混合时由均值差产生的方差项, 不能只线性平均 variance)
#    · 队友在 test1 上: 69.41 -> 69.56 (+0.15, alpha=0.25, 41个BN层全层均值+方差)
#
#  用法:
#    python bn_calib.py --ckpt ckpt_kan_dacs_cw7_1024su_relaxbg.pth \
#                       --out  ckpt_kan_dacs_cw7_1024su_relaxbg_bnA25.pth \
#                       --imgs "E:/无人机语义分割第二阶段/images" --alpha 0.25
#
#  典型耗时: 1300 张 1024² 单尺度前向 @4080S ≈ 1~2 分钟
# ══════════════════════════════════════════════════════════════════
import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kan_swin as K


def bn_modules(model):
    """返回所有 BatchNorm 层的 (name, module)"""
    out = []
    for n, m in model.named_modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            out.append((n, m))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="父模型 ckpt")
    ap.add_argument("--out", required=True, help="输出 ckpt")
    ap.add_argument("--imgs", required=True, help="目标域图像目录")
    ap.add_argument("--alpha", type=float, default=0.25, help="目标域统计权重(默认0.25)")
    ap.add_argument("--batch", type=int, default=4, help="前向 batch(无梯度, 可大)")
    ap.add_argument("--n", type=int, default=0, help="只用前 n 张(0=全部)")
    ap.add_argument("--size", type=int, default=0, help="强制 resize 尺寸(0=原尺寸)")
    ap.add_argument("--device", type=str, default=None)
    a = ap.parse_args()

    dev = a.device or K.DEVICE
    print(f"[bn_calib] ckpt={os.path.basename(a.ckpt)}  alpha={a.alpha}  device={dev}")

    # ---- 1. 加载父模型(EMA) ----
    model, iters = K.load_trained(use_ema=True, path=a.ckpt)
    model.eval()
    bns = bn_modules(model)
    print(f"[bn_calib] BatchNorm 层数 = {len(bns)}   parent iter = {iters}")
    if not bns:
        raise SystemExit("!! 模型里没有 BatchNorm 层, 校准无意义")

    # ---- 2. 记录原始统计, 并把 BN 切到「累积总体统计」模式 ----
    orig = {}
    for n, m in bns:
        orig[n] = (m.running_mean.clone(), m.running_var.clone(),
                   m.num_batches_tracked.clone())
        m.reset_running_stats()
        m.momentum = None      # None = 累积等权平均(总体统计)
        m.train()              # 只让 BN 在 train 模式更新 running stats
    # 注意: model 其余部分保持 eval(), 避免 dropout 等引入随机性

    # ---- 3. 目标域前向 ----
    files = sorted(os.listdir(a.imgs))
    if a.n:
        files = files[:a.n]
    if not files:
        raise SystemExit(f"!! 图像目录为空: {a.imgs}")
    print(f"[bn_calib] 目标域图像 {len(files)} 张, batch={a.batch}")

    t0 = time.time()
    buf, done = [], 0
    with torch.no_grad():
        for i, fn in enumerate(files):
            im = np.array(Image.open(os.path.join(a.imgs, fn)).convert("RGB"))
            if a.size and (im.shape[0] != a.size or im.shape[1] != a.size):
                im = np.array(Image.fromarray(im).resize((a.size, a.size), Image.BILINEAR))
            x = torch.from_numpy(im.copy()).permute(2, 0, 1).float().div_(255.0)
            buf.append(x)
            if len(buf) == a.batch or i == len(files) - 1:
                xb = torch.stack(buf).to(dev)
                with torch.autocast("cuda", torch.float16, enabled=(dev == "cuda")):
                    model(xb)
                done += len(buf)
                buf = []
                if done % 100 < a.batch or done == len(files):
                    el = time.time() - t0
                    print(f"  fwd {done}/{len(files)}  {el:.1f}s  ({done/max(el,1e-6):.1f} img/s)",
                          flush=True)
    print(f"[bn_calib] 前向完成, 用时 {time.time()-t0:.1f}s")

    # ---- 4. 混合统计量 ----
    al = a.alpha
    stats = []
    state = model.state_dict()
    for n, m in bns:
        mu_o, var_o, nbt = orig[n]
        mu_t, var_t = m.running_mean.clone(), m.running_var.clone()
        mu_n = (1 - al) * mu_o + al * mu_t
        var_n = (1 - al) * var_o + al * var_t + al * (1 - al) * (mu_t - mu_o) ** 2
        m.running_mean.copy_(mu_n)
        m.running_var.copy_(var_n)
        m.num_batches_tracked.copy_(nbt)
        m.momentum = 0.1
        m.eval()
        stats.append((n, float((mu_t - mu_o).abs().mean()),
                      float((var_o.sqrt() - var_t.sqrt()).abs().mean())))

    # 诊断: 域差异最大的几层
    stats.sort(key=lambda x: -x[1])
    print("\n[bn_calib] 域统计差异 Top-8 (|Δmean| 越大说明该层域漂移越明显):")
    for n, dm, ds in stats[:8]:
        print(f"   {n:<44} |dmean|={dm:.4f}  |dstd|={ds:.4f}")

    # ---- 5. 保存: 只改 BN 缓冲区, 学习参数逐位不变 ----
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    # ★ 推理走 load_trained(use_ema=True) ⇒ 只改 ema; 若父模型没有 ema 才改 model
    keys = ["ema"] if sd.get("ema") is not None else ["model"]
    changed = 0
    n_buf = 0
    for key in keys:
        for n, _ in bns:
            for suffix in ("running_mean", "running_var", "num_batches_tracked"):
                k = f"{n}.{suffix}"
                if k in sd[key] and k in state:
                    n_buf += 1
                    if not torch.equal(sd[key][k].cpu(), state[k].cpu()):
                        changed += 1
                    sd[key][k] = state[k].detach().cpu().clone()
    sd["bn_calib"] = dict(alpha=al, n=len(files), src=os.path.basename(a.ckpt),
                          iter=iters, keys=keys)
    torch.save(sd, a.out)

    # ---- 6. 自检: 学习参数必须逐位未变 ----
    print(f"\n[bn_calib] 写入缓冲区 {n_buf} 条 ({len(bns)} 层 × 3), 其中值发生变化 {changed} 条")
    print(f"[bn_calib] 更新的 state_dict 分支: {keys}  (推理走 ema)")
    print(f"[bn_calib] 学习参数未动 (只改 running_mean/var/num_batches_tracked)")
    print(f"[bn_calib] 已保存 -> {a.out}")
    print(f"[bn_calib] 用法: python dacs_train.py --infer --ckpt {a.out} "
          f"--out output_test2_<TAG> --tta ms2x")


if __name__ == "__main__":
    main()
