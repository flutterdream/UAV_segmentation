# -*- coding: utf-8 -*-
"""训练脚本：在已有主干权重之上做"冻结编码器与 DFR、只训解码器"的约束微调。

最终输出的权重（checkpoints/best_model.pth）由四个连续阶段训练而成，
每一阶段都以"上一阶段的权重"为父模型、并以它的解码器软预测作为 KL 约束参照：

    R0  (第二阶段链终点，基线 70.3835)
     → v2 (1600 步) 70.5353 → r2 (3200 步) → r3 (6400 步) 70.6951 → sw (12800 步) 70.7198

本脚本即这四段所用的同一份代码。示例：

    # 第 1 段：从第二阶段权重出发，6400 个 crop
    python train.py --images /data/train/images --masks /data/train/masks \
                    --parent R0.pth --out v2.pth --steps 1600 --batch 4

    # 第 4 段：从 r3 出发，51200 个 crop（最终输出的权重）
    python train.py --images /data/train/images --masks /data/train/masks \
                    --parent r3.pth --out best_model.pth --steps 12800 --batch 4

说明：编码器与 DFR 全程冻结，BatchNorm 的 running 统计也不更新（model.eval()）。
      这一设计决定了训练是"在既有特征上重训读出头"，因此四段都只需要数十分钟。
"""
import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.deepkanseg import DeepKANSegSwin, NUM_CLASSES

# 8 类（不含忽略类）的类别权重，按训练集逆频率设定
CLS_W = [0.40, 0.49, 0.65, 0.94, 1.66, 0.37, 0.90, 2.59]
NB = 9


def segmentation_loss(logits, labels, weight_map, dice, lovasz, class_weight):
    """边界加权 CE + Dice + 0.5·Lovász（与最终训练完全一致）"""
    ce = F.cross_entropy(logits, labels, weight=class_weight, ignore_index=0, reduction="none")
    down_logits = F.interpolate(logits, scale_factor=0.5, mode="bilinear", align_corners=False)
    down_labels = F.interpolate(labels[:, None].float(), scale_factor=0.5, mode="nearest")[:, 0].long()
    return ((ce * weight_map).sum() / weight_map.sum() + dice(logits, labels)
            + 0.5 * lovasz(down_logits, down_labels))


def kl_loss(student_logits, ref_logits, label, size=256):
    """对父解码器的软预测做 KL 约束，只在非忽略像素上平均"""
    sl = F.interpolate(student_logits, size=(size, size), mode="bilinear",
                       align_corners=False)[:, 1:9]
    rl = F.interpolate(ref_logits, size=(size, size), mode="bilinear",
                       align_corners=False)[:, 1:9]
    rp = rl.softmax(1)
    slp = F.log_softmax(sl, 1)
    kl = ((rp.clamp_min(1e-8).log() - slp) * rp).sum(1)
    m = F.interpolate((label > 0).float().unsqueeze(1), size=(size, size),
                      mode="nearest")[:, 0] > 0.5
    if m.sum() == 0:
        return torch.zeros((), device=student_logits.device)
    return kl[m].mean()


def frozen_features(model, x):
    """冻结部分（encoder + DFR）的前向，返回解码器所需的四级特征"""
    with torch.no_grad():
        fs = model.encoder(x)
        res1, res2, res3, res4 = fs[2], fs[3], fs[4], fs[5]
        b, _, h4, w4 = res4.shape
        z = model.dfr_in(res4).flatten(2).transpose(1, 2)
        for blk in model.dfr_blocks:
            z = blk(z, h4, w4)
        z = z.transpose(1, 2).reshape(b, -1, h4, w4)
        res4 = res4 + model.dfr_out(z)
    return res1, res2, res3, res4, x.size(-2), x.size(-1)


def geo_aug(img, msk, seed):
    """仅几何增强：水平翻转 0.5 / 垂直翻转 0.5 / 90° 旋转 0.25"""
    rng = random.Random(seed)
    if rng.random() < 0.5:
        img, msk = img[:, ::-1], msk[:, ::-1]
    if rng.random() < 0.5:
        img, msk = img[::-1], msk[::-1]
    if rng.random() < 0.25:
        k = rng.choice((1, 2, 3))
        img, msk = np.rot90(img, k).copy(), np.rot90(msk, k).copy()
    return np.ascontiguousarray(img), np.ascontiguousarray(msk)


def band_weight(msk):
    """边界带权重：5×5 膨胀/腐蚀之差处权重 3.0，其余 1.0"""
    import cv2
    band = cv2.dilate(msk, np.ones((5, 5), np.uint8)) != cv2.erode(msk, np.ones((5, 5), np.uint8))
    return np.where(band, 3.0, 1.0).astype(np.float32)


def norm(img):
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    return ((img.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="训练图像目录")
    ap.add_argument("--masks", required=True, help="训练标签目录（与图像同名）")
    ap.add_argument("--parent", required=True, help="父模型权重 .pth")
    ap.add_argument("--out", required=True, help="输出权重路径")
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--sup-w", type=float, default=0.20, dest="sup_w")
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(a.parent, map_location="cpu", weights_only=False)
    state = payload["ema"] if isinstance(payload, dict) and "ema" in payload else payload
    model = DeepKANSegSwin(NUM_CLASSES).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()                                  # BN 用运行统计、不更新
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.decoder.parameters():
        p.requires_grad_(True)

    ref = deepcopy(model.decoder).to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    files = sorted(p.name for p in Path(a.images).iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    random.Random(a.seed).shuffle(files)
    need = a.steps * a.batch
    while len(files) < need:
        files = files + files
    order = files[:need]
    print(f"[数据] {len(order)} 张（{a.steps} 步 × batch {a.batch}），父权重 {Path(a.parent).name}")

    dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=0)
    lovasz = smp.losses.LovaszLoss(mode="multiclass", ignore_index=0)
    cw = torch.tensor([1.0] + CLS_W, device=device)
    opt = torch.optim.AdamW([p for p in model.decoder.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=a.lr * 0.1)
    ema = {k: v.detach().clone() for k, v in model.decoder.state_dict().items()}

    for step in range(a.steps):
        seed = a.seed * 1000003 + step * a.batch
        xs, ys, ws = [], [], []
        for b in range(a.batch):
            fn = order[step * a.batch + b]
            img = np.asarray(Image.open(Path(a.images) / fn).convert("RGB"))
            msk = np.asarray(Image.open(Path(a.masks) / fn))
            img, msk = geo_aug(img, msk, seed + b)
            xs.append(norm(img))
            ys.append(torch.from_numpy(msk.astype(np.int64)))
            ws.append(torch.from_numpy(band_weight(msk)))
        x = torch.from_numpy(np.stack(xs)).float().to(device)
        y = torch.stack(ys).to(device)
        w = torch.stack(ws).to(device)

        feats = frozen_features(model, x)
        with torch.no_grad():
            ref_logits = ref(*feats)
        with torch.autocast("cuda", torch.bfloat16, enabled=(device == "cuda")):
            logits = model.decoder(*feats)
        logits = logits.float()
        sup = segmentation_loss(logits, y, w, dice, lovasz, cw)
        kl = kl_loss(logits, ref_logits.float(), y)
        loss = a.sup_w * sup + a.kl * kl

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.decoder.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for k, v in model.decoder.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(a.ema).add_(v, alpha=1 - a.ema)

        if (step + 1) % 100 == 0:
            print(f"  {step+1}/{a.steps} loss {float(loss):.4f} sup {float(sup):.4f} "
                  f"kl {float(kl):.4f} lr {sched.get_last_lr()[0]:.2e}")

    # 导出：解码器取 EMA 值，BN 缓冲一律沿用父权重（避免浮点漂移污染统计）
    out_state = {k: v.clone() for k, v in state.items()}
    for k, v in ema.items():
        if k.endswith((".running_mean", ".running_var", ".num_batches_tracked")):
            continue
        out_state["decoder." + k] = v
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": out_state, "ema": out_state, "iter": a.steps,
                "parent": Path(a.parent).name}, a.out)
    print(f"[完成] {a.out}")


if __name__ == "__main__":
    main()
