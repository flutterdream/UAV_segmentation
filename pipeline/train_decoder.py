# -*- coding: utf-8 -*-
"""
解码器约束微调（复刻队友 70.46 配方 + 我们的可控改进）

父权重 = 未校准 i600 EMA → 用 test_3(3594) 统计按 α=0.75 混合（等价队友的 mixed3594 父模型）
只训练 decoder（2.64M），冻结 encoder / DFR / 全部 BN running 统计
损失 = 0.20 * L_sup(边界加权CE + Dice + 0.5·Lovasz) + λ_kl * L_KL(父解码器软预测, 256², 8 类)

用法:
  python train_decoder.py --tag v1 --steps 800  --batch 1 --kl 1.0
  python train_decoder.py --tag v2 --steps 1600 --batch 4 --kl 1.0
"""
import argparse, json, math, os, random, sys, time
from copy import deepcopy
from pathlib import Path

ROOT = Path("/root/autodl-tmp/aic_semi")
os.environ.setdefault("AIC_SKIP_VERIFY", "1")
os.environ["HF_HOME"] = str(ROOT / "public_pretrained" / "huggingface")
os.environ["HF_HUB_CACHE"] = str(ROOT / "public_pretrained" / "huggingface" / "hub")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TORCH_HOME"] = str(ROOT / "work" / "runtime_cache" / "torch")
os.environ.update(STRICT_REPRO_ROOT=str(ROOT))
sys.path.insert(0, str(ROOT / "code" / "kan_pipeline"))
sys.path.insert(0, str(ROOT / "code"))

import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from PIL import Image

import strict_env as P
import seg_swin as S
import kan_swin as K
import stage47_i600 as T47

RAW = ROOT / "ckpt_kan_stage47_earlystop_bracket_i600.pth"
MOM = ROOT / "bn_moments_test3.json"
IMG = ROOT / "data" / "train" / "images"
MSK = ROOT / "data" / "train" / "masks"
TEST_IMG = ROOT / "official_data" / "test_3" / "images"
LABEL = ROOT / "labels"
OUT = ROOT / "p0"
CLS_W = [0.40, 0.49, 0.65, 0.94, 1.66, 0.37, 0.90, 2.59]
NB = 9


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_parent(alpha=0.75, kind="final"):
    if kind.startswith("/") or kind.endswith(".pth"):
        return Path(kind)                     # 直接指定权重路径（多轮微调用）
    if kind == "final":
        return ROOT / "final.pth"
    out = OUT / f"parent_mixed3594_a{int(alpha*100):03d}.pth"
    if out.exists():
        log(f"父权重已存在 {out.name}")
        return out
    d = torch.load(RAW, map_location="cpu", weights_only=False)
    ema = {k: v.clone() for k, v in d["ema"].items()}
    mom = json.load(open(MOM, encoding="utf-8"))
    n = 0
    for name, m in mom.items():
        km, kv = name + ".running_mean", name + ".running_var"
        if km not in ema:
            continue
        sm = ema[km].double(); sv = ema[kv].double()
        tm = torch.tensor(m["target_mean"], dtype=torch.float64)
        tv = torch.tensor(m["target_var"], dtype=torch.float64)
        mean = (1 - alpha) * sm + alpha * tm
        var = (1 - alpha) * sv + alpha * tv + alpha * (1 - alpha) * (tm - sm).square()
        if not torch.isfinite(mean).all() or (var <= 0).any():
            raise RuntimeError(f"混合非法 {name}")
        ema[km] = mean.float(); ema[kv] = var.float(); n += 1
    torch.save({"model": d["model"], "ema": ema, "iter": d["iter"],
                "strict_repro": {"stage": f"p0_parent_mixed3594_bn{int(alpha*100):03d}", "seed": 42},
                "alpha": alpha, "bn_layers": n}, out)
    log(f"父权重写入 {out.name}（改了 {n} 个 BN 层）")
    return out


def features(model, x):
    """冻结主干：encoder + DFR，返回解码器所需特征"""
    fs = model.encoder(x)
    r1, r2, r3, r4 = fs[2], fs[3], fs[4], fs[5]
    b, _, h4, w4 = r4.shape
    z = model.dfr_in(r4).flatten(2).transpose(1, 2)
    for blk in model.dfr_blocks:
        z = blk(z, h4, w4)
    z = z.transpose(1, 2).reshape(b, -1, h4, w4)
    r4 = r4 + model.dfr_out(z)
    h, w = x.shape[-2:]
    return r1, r2, r3, r4, h, w


def kl_loss(student_logits, ref_logits, label, size=256):
    sl = F.interpolate(student_logits, size=(size, size), mode="bilinear",
                       align_corners=False)[:, 1:9]
    rl = F.interpolate(ref_logits, size=(size, size), mode="bilinear",
                       align_corners=False)[:, 1:9]
    rp = rl.softmax(1)
    slp = F.log_softmax(sl, 1)
    kl = (rp.clamp_min(1e-8).log() - slp) * rp
    kl = kl.sum(1)
    m = F.interpolate((label > 0).float().unsqueeze(1), size=(size, size),
                      mode="nearest")[:, 0] > 0.5
    if m.sum() == 0:
        return torch.zeros((), device=student_logits.device)
    return kl[m].mean()


def geo_aug(img, msk, seed):
    """只用几何增强：hflip .5 / vflip .5 / rot90 .25"""
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
    import cv2
    band = cv2.dilate(msk, np.ones((5, 5), np.uint8)) != cv2.erode(msk, np.ones((5, 5), np.uint8))
    return np.where(band, 3.0, 1.0).astype(np.float32)


@torch.no_grad()
def eval_labels(model, dev):
    import importlib.util
    from PIL import Image
    spec = importlib.util.spec_from_file_location(
        "fp", ROOT / "code" / "kan_pipeline" / "final_pipeline.py")
    FP = importlib.util.module_from_spec(spec); spec.loader.exec_module(FP)
    was = model.training
    model.eval()
    cm = np.zeros((NB, NB), np.int64)
    for p in sorted(LABEL.glob("*.png")):
        img = np.asarray(Image.open(TEST_IMG / p.name).convert("RGB"))
        pr = FP.predict_probability_average(model, img, S, dev).argmax(0).numpy().astype(np.uint8)
        gt = np.asarray(Image.open(p))
        m = gt > 0
        cm += np.bincount(gt[m].astype(np.int16) * NB + pr[m].astype(np.int16),
                          minlength=NB * NB).reshape(NB, NB)
    if was:
        model.train()
    i = np.diag(cm).astype(float); u = cm.sum(0) + cm.sum(1) - np.diag(cm)
    iou = np.where(u > 0, i / np.maximum(u, 1), np.nan)
    present = cm.sum(1) > 0
    return float(np.nanmean(iou[present])), {str(k): round(float(iou[k]), 4) for k in range(1, NB)}


@torch.no_grad()
def eval_val500(model, dev, ids):
    cm = np.zeros((NB, NB), np.int64)
    model.eval()
    for i in ids:
        fn = f"{i:04d}.png"
        img = np.asarray(Image.open(IMG / fn).convert("RGB"))
        gt = np.asarray(Image.open(MSK / fn))
        x = S.norm(img).unsqueeze(0).to(dev)
        with torch.autocast("cuda", torch.float16, enabled=(dev == "cuda")):
            out = model(x)
        pr = out.float().argmax(1)[0].cpu().numpy().astype(np.uint8)
        m = gt > 0
        cm += np.bincount(gt[m].astype(np.int16) * NB + pr[m].astype(np.int16),
                          minlength=NB * NB).reshape(NB, NB)
    i = np.diag(cm).astype(float); u = cm.sum(0) + cm.sum(1) - np.diag(cm)
    iou = np.where(u > 0, i / np.maximum(u, 1), np.nan)
    present = cm.sum(1) > 0
    return float(np.nanmean(iou[present]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--sup-w", type=float, default=0.20)
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--parent", default="final", help="final / mixed / 或 .pth 路径")
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--snapshots", default="", help="逗号分隔的步数，在这些点存快照并评 val500")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(exist_ok=True)
    parent = build_parent(args.alpha, args.parent)
    payload = torch.load(parent, map_location="cpu", weights_only=False)

    model = K.DeepKANSegSwin(K.NUM_CLASSES).to(dev)
    model.load_state_dict(payload["ema"], strict=True)
    model.eval()                      # BN 用运行统计、不更新；dropout 关闭
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.decoder.parameters():
        p.requires_grad_(True)

    ref_decoder = deepcopy(model.decoder).to(dev).eval()
    for p in ref_decoder.parameters():
        p.requires_grad_(False)

    bn_before = {n: m.running_mean.clone() for n, m in model.named_modules()
                 if isinstance(m, torch.nn.BatchNorm2d)}
    enc_before = {n: p.detach().clone() for n, p in model.named_parameters()
                  if not n.startswith("decoder.")}

    split = json.load(open(ROOT / "code" / "kan_pipeline" / "split.json", encoding="utf-8"))
    val_ids = [int(v) for v in split["val_ids"]]
    ids = sorted(set(range(6996)) - set(val_ids))
    random.Random(42).shuffle(ids)
    need = args.steps * args.batch
    if need > len(ids):
        ids = ids * (need // len(ids) + 1)
    order = ids[:need]
    log(f"源图 {len(ids)} 张可用，本轮消费 {len(order)} 张（{args.steps} 步 × batch {args.batch}）")

    opt = torch.optim.AdamW([p for p in model.decoder.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.1)
    dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=0)
    lovasz = smp.losses.LovaszLoss(mode="multiclass", ignore_index=0)
    class_w = torch.tensor([1.0] + CLS_W, device=dev)
    ema = {k: v.detach().clone() for k, v in model.decoder.state_dict().items()}

    base_lab, base_pc = eval_labels(model, dev)
    log(f"父模型（mixed3594 α{args.alpha}）标注 mIoU = {base_lab:.4f}")

    snap_steps = {int(x) for x in args.snapshots.split(",") if x.strip()}
    if snap_steps:
        log(f"快照点 {sorted(snap_steps)}")
    it_log = open(OUT / f"{args.tag}_train.jsonl", "w", encoding="utf-8")
    t0 = time.time()
    running = []
    for step in range(args.steps):
        seed = 42 * 1000003 + step * args.batch
        xs, ys, ws = [], [], []
        for b in range(args.batch):
            fn = f"{order[step * args.batch + b]:04d}.png"
            img = np.asarray(Image.open(IMG / fn).convert("RGB"))
            msk = np.asarray(Image.open(MSK / fn))
            img, msk = geo_aug(img, msk, seed + b)
            xs.append(S.norm(img)); ys.append(torch.from_numpy(msk.astype(np.int64)))
            ws.append(torch.from_numpy(band_weight(msk)))
        x = torch.from_numpy(np.stack(xs)).to(dev)
        y = torch.stack(ys).to(dev)
        w = torch.stack(ws).to(dev)

        with torch.no_grad():
            feats = features(model, x)
            ref_logits = ref_decoder(*feats)
        with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
            logits = model.decoder(*feats)
        logits = logits.float()
        sup = T47.segmentation_loss(logits, y, w, dice, lovasz, class_w)
        kl = kl_loss(logits, ref_logits.float(), y)
        loss = args.sup_w * sup + args.kl * kl

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in model.decoder.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            for k, v in model.decoder.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(args.ema).add_(v, alpha=1 - args.ema)

        running.append(float(loss.detach()))
        it_log.write(json.dumps({"step": step, "loss": float(loss.detach()),
                                 "sup": float(sup.detach()), "kl": float(kl.detach()),
                                 "gnorm": float(gn), "lr": sched.get_last_lr()[0]}) + "\n")
        if (step + 1) % 50 == 0:
            log(f"  {step+1}/{args.steps} loss {np.mean(running[-50:]):.4f} "
                f"sup {float(sup):.4f} kl {float(kl):.4f} gn {float(gn):.2f} "
                f"{(time.time()-t0)/60:.0f} min")
        if (step + 1) in snap_steps:
            bak = {k: v.clone() for k, v in model.decoder.state_dict().items()}
            model.decoder.load_state_dict(ema)
            lm, lpc = eval_labels(model, dev)
            vm = eval_val500(model, dev, val_ids)
            st_state = {k: v.clone() for k, v in payload["ema"].items()}
            for k, v in ema.items():
                if k.endswith((".running_mean", ".running_var", ".num_batches_tracked")):
                    continue
                st_state["decoder." + k] = v
            sp = OUT / f"{args.tag}_s{step+1}.pth"
            torch.save({"model": st_state, "ema": st_state, "iter": step + 1,
                        "strict_repro": {"stage": f"p0_decoder_{args.tag}_s{step+1}", "seed": 42},
                        "label_miou": lm, "val500_miou": vm, "parent": parent.name}, sp)
            log(f"  ◆ 快照 step {step+1}  val500 {vm:.6f}  标注 {lm:.4f}  已存 {sp.name}")
            it_log.write(json.dumps({"step": step, "snapshot": step + 1,
                                     "val500": vm, "label": lm}) + "\n")
            it_log.flush()
            model.decoder.load_state_dict(bak)

        if args.eval_every and (step + 1) % args.eval_every == 0:
            bak = {k: v.clone() for k, v in model.decoder.state_dict().items()}
            model.decoder.load_state_dict(ema)
            m, pc = eval_labels(model, dev)
            model.decoder.load_state_dict(bak)
            log(f"  ★ step {step+1} 标注 mIoU {m:.4f}（父 {base_lab:.4f}）{pc}")
            it_log.write(json.dumps({"step": step, "eval_miou": m}) + "\n")
    it_log.close()

    # 导出：解码器 EMA + 其余父状态
    model.decoder.load_state_dict(ema)
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            if not torch.allclose(m.running_mean, bn_before[n], rtol=0, atol=0):
                log(f"  提示：{n} 有 {float((m.running_mean-bn_before[n]).abs().max()):.2e} 的 EMA 漂移，导出时已强制保留父值")
    for n, p in model.named_parameters():
        if not n.startswith("decoder.") and not torch.equal(p, enc_before[n]):
            raise RuntimeError(f"冻结参数被改动 {n}")
    lab_miou, lab_pc = eval_labels(model, dev)
    val_miou = eval_val500(model, dev, val_ids)
    state = {k: v.clone() for k, v in payload["ema"].items()}
    skipped = 0
    for k, v in ema.items():
        if k.endswith((".running_mean", ".running_var", ".num_batches_tracked")):
            skipped += 1
            continue          # BN buffer 保持父模型原值，避免 EMA 浮点漂移
        state["decoder." + k] = v
    out = OUT / f"{args.tag}.pth"
    torch.save({"model": state, "ema": state, "iter": args.steps,
                "strict_repro": {"stage": f"p0_decoder_{args.tag}", "seed": 42},
                "train_args": vars(args), "label_miou": lab_miou,
                "label_per_class": lab_pc, "val500_miou": val_miou,
                "parent": parent.name}, out)
    res = {"tag": args.tag, "parent_label_miou": base_lab, "label_miou": lab_miou,
           "label_per_class": lab_pc, "val500_miou": val_miou,
           "delta_label": lab_miou - base_lab,
           "minutes": round((time.time() - t0) / 60, 1),
           "peak_mib": int(torch.cuda.max_memory_allocated() / 2 ** 20), **vars(args)}
    json.dump(res, open(OUT / f"{args.tag}_result.json", "w"), indent=1, ensure_ascii=False)
    log("RESULT=" + json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
