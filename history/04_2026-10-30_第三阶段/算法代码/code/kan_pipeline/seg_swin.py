# -*- coding: utf-8 -*-
"""
v4 对照模型: UPerNet + Swin-Base (单模型, 竞赛合规, 仅官方数据+ImageNet公开权重)
- 与 seg_official.py v3.1 完全相同的获胜配方: 强增强+边界加权CE+Dice+0.5*Lovász+EMA+温和采样
- --crop 640 为 v4.1: 从 512 权重 EMA 热启动 + 新增轻量增强(灰度/模糊)作为新信号, lr 3e-5
- 断点续训: 每 200 iter 存档, 每次调用跑 --max-seconds 秒
用法:
  python seg_swin.py --train --target-iters 12000                 # v4 从零
  python seg_swin.py --train --crop 640 --target-iters 4000       # v4.1 热启动
  python seg_swin.py --eval / --eval-tta / --eval-ms / --calibrate / --infer [--crop 640]
"""
import argparse, json, os, random, time
import numpy as np
from PIL import Image

import strict_env as P
P.configure_process()

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_ROOT = str(P.RUN_ROOT)
IMG_DIR = str(P.TRAIN_IMAGES)
MSK_DIR = str(P.TRAIN_MASKS)
TEST_DIR = str(P.TEST1_IMAGES)
TEST2_DIR = str(P.TEST2_IMAGES)
LOG = os.path.join(ARTIFACT_ROOT, "log_swin_upernet.json")
NUM_CLASSES, IGNORE = 9, 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Background","Building","Road","Water","Barren","Vegetation","Agricultural","Vehicle"]
CE_WEIGHTS = [0.40, 0.49, 0.65, 0.94, 1.66, 0.37, 0.90, 2.59]  # 类 1-8, 与 v3.1 相同


def ckpt_path(crop=512, barren=False, name=None):
    if name:
        if os.path.basename(name) != name or not name.endswith(".pth"):
            raise ValueError(f"checkpoint name must be a plain .pth filename: {name}")
        return str(P.assert_run_path(os.path.join(ARTIFACT_ROOT, name)))
    base = "ckpt_swin_upernet.pth" if crop == 512 else f"ckpt_swin_upernet_{crop}.pth"
    if barren:
        base = base.replace(".pth", "_barren.pth")
    return str(P.assert_run_path(os.path.join(ARTIFACT_ROOT, base)))


def bias_path(crop=512, barren=False):
    base = "bias_swin.npy" if crop == 512 else f"bias_swin_{crop}.npy"
    if barren:
        base = base.replace(".npy", "_barren.npy")
    return str(P.assert_run_path(os.path.join(ARTIFACT_ROOT, base)))


class UAV(Dataset):
    def __init__(self, ids, train, crop=512, epoch_mult=1, extra=False,
                 img_dir=None, msk_dir=None, name_fmt="{k:04d}.png",
                 darken=False, full_image_scale_up=False,
                 frozen_7439_scale_floor=False):
        self.ids, self.train, self.crop = ids, train, crop
        self.epoch_mult = epoch_mult
        self.extra = extra  # v4.1 新信号: 随机灰度/高斯模糊
        self.darken = darken  # 后加的 v5 暗化；严格 v4.1 必须关闭
        self.full_image_scale_up = full_image_scale_up  # 仅后续47的独立整图处理开关
        # 最终69.83链明确选择了审计前的0.7439 v4.1父权重；该父链的
        # 尺度下限曾被实现为crop。归档入口用显式开关复现这一事实，
        # 不能把它误写成历史0.75x公式。
        self.frozen_7439_scale_floor = frozen_7439_scale_floor
        self.img_dir = img_dir or IMG_DIR
        self.msk_dir = msk_dir or MSK_DIR
        self.name_fmt = name_fmt

    def __len__(self):
        return len(self.ids) * self.epoch_mult if self.train else len(self.ids)

    def __getitem__(self, i):
        k = self.ids[i % len(self.ids)]
        img = np.array(Image.open(os.path.join(self.img_dir, self.name_fmt.format(k=k))).convert("RGB"))
        msk = np.array(Image.open(os.path.join(self.msk_dir, self.name_fmt.format(k=k))))
        if self.train:
            h, w = msk.shape; c = self.crop
            if self.full_image_scale_up:
                raise RuntimeError("47 full-image scale-up must use its dedicated stage implementation")
            min_s = c if self.frozen_7439_scale_floor else int(c * 0.75)
            s = min(random.randint(min_s, int(c * 1.5)), h)
            y = random.randint(0, h - s); x = random.randint(0, w - s)
            img, msk = img[y:y+s, x:x+s], msk[y:y+s, x:x+s]
            if s != c:
                img = np.array(Image.fromarray(img).resize((c, c), Image.BILINEAR))
                msk = np.array(Image.fromarray(msk).resize((c, c), Image.NEAREST))
            if random.random() < 0.5: img, msk = img[:, ::-1], msk[:, ::-1]
            if random.random() < 0.5: img, msk = img[::-1], msk[::-1]
            if random.random() < 0.25:
                r = random.randint(1, 3)
                img, msk = np.rot90(img, r).copy(), np.rot90(msk, r).copy()
            if random.random() < 0.8:  # 色彩抖动
                im = Image.fromarray(img)
                from PIL import ImageEnhance
                im = ImageEnhance.Brightness(im).enhance(random.uniform(0.8, 1.2))
                im = ImageEnhance.Contrast(im).enhance(random.uniform(0.8, 1.2))
                im = ImageEnhance.Color(im).enhance(random.uniform(0.85, 1.15))
                img = np.array(im)
            if self.extra:  # v4.1 新增轻量增强 (新信号, 防无信息续训过拟合)
                if random.random() < 0.1:  # 灰度化
                    img = np.array(Image.fromarray(img).convert("L").convert("RGB"))
                if random.random() < 0.1:  # 高斯模糊
                    from PIL import ImageFilter
                    img = np.array(Image.fromarray(img).filter(
                        ImageFilter.GaussianBlur(random.uniform(0.3, 1.0))))
            if self.darken:  # v5 后加暗化，与历史严格 v4.1 分离
                if random.random() < 0.18:
                    g = random.uniform(1.4, 2.4)
                    img = np.clip((img.astype(np.float32) / 255.0) ** g * 255.0,
                                  0, 255).astype(np.uint8)
                    if random.random() < 0.5:
                        img = np.clip(img.astype(np.float32) * random.uniform(0.55, 0.9),
                                      0, 255).astype(np.uint8)
        x = norm(img)
        y = torch.from_numpy(msk.copy()).long()
        if self.train:  # 边界带权重图 (v3.1 配方)
            import cv2
            band = cv2.dilate(msk, np.ones((5, 5), np.uint8)) != cv2.erode(msk, np.ones((5, 5), np.uint8))
            wmap = torch.from_numpy(np.where(band, 3.0, 1.0).astype(np.float32))
            return x, y, wmap
        return x, y


def norm(img):
    x = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (x - mean) / std


def build_model(weights="imagenet"):
    # timm resolves this registry name from the one pinned offline HF snapshot.
    # verify_inputs() proves that the snapshot is the sole approved public weight.
    P.verify_inputs()
    return smp.UPerNet(encoder_name="tu-swin_base_patch4_window7_224",
                       encoder_weights=weights, in_channels=3, classes=NUM_CLASSES,
                       strict_img_size=False).to(DEVICE)


@torch.no_grad()
def predict_logits(model, img, tta=False, bias=None):
    """img: HxWx3 uint8 -> logits (C,H,W) cpu. bias: (C,) 逐类 logit 偏置."""
    if tta == "ms":
        passes = [(1.0, False), (1.0, True), (0.75, False), (1.25, False)]
    elif tta == "ms2":
        passes = [(1.0, False), (1.0, True), (0.75, False), (0.75, True),
                  (1.25, False), (1.25, True), (1.5, False), (0.5, False)]
    elif tta:
        passes = [(1.0, False), (1.0, True)]
    else:
        passes = [(1.0, False)]
    H, W = img.shape[:2]
    acc = None
    for scale, flip in passes:
        im = img[:, ::-1] if flip else img
        if scale != 1.0:
            im = np.array(Image.fromarray(im).resize(
                (int(W * scale), int(H * scale)), Image.BILINEAR))
        x = norm(im).unsqueeze(0).to(DEVICE)
        with torch.autocast("cuda", torch.float16, enabled=DEVICE == "cuda"):
            out = model(x)
            if isinstance(out, (tuple, list)): out = out[0]
        lg = out.float()
        if scale != 1.0:
            lg = F.interpolate(lg, size=(H, W), mode="bilinear", align_corners=False)
        lg = lg.cpu()
        if flip:
            lg = torch.flip(lg, dims=[3])
        acc = lg if acc is None else acc + lg
    out = acc / len(passes)
    if bias is not None:
        out = out + bias.view(-1, 1, 1)
    return out.squeeze(0)


@torch.no_grad()
def evaluate(model, val_ids, tta=False, bias=None):
    model.eval()
    inter = np.zeros(NUM_CLASSES); union = np.zeros(NUM_CLASSES)
    for n, k in enumerate(val_ids):
        img = np.array(Image.open(os.path.join(IMG_DIR, f"{k:04d}.png")).convert("RGB"))
        msk = np.array(Image.open(os.path.join(MSK_DIR, f"{k:04d}.png")))
        pred = predict_logits(model, img, tta, bias).argmax(0).numpy()
        valid = msk != IGNORE
        for c in range(1, NUM_CLASSES):
            p, t = (pred == c) & valid, (msk == c) & valid
            inter[c] += (p & t).sum(); union[c] += (p | t).sum()
        if (n + 1) % 100 == 0:
            print(f"  eval {n+1}/{len(val_ids)}", flush=True)
    iou = [inter[c] / union[c] if union[c] > 0 else float("nan") for c in range(1, NUM_CLASSES)]
    return float(np.nanmean(iou)), iou


def train(a):
    if a.strict_v4_1:
        if (a.crop != 640 or a.seed != 42 or a.target_iters != 4000 or a.barren or
                a.darken or a.full_image_scale_up):
            raise RuntimeError("strict v4.1 requires crop640/seed42/4000, no barren/darken/47-scale-up")
        if not a.warm_from or not a.strict_parent_stage:
            raise RuntimeError("strict v4.1 requires explicit parent checkpoint and parent stage")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    val_ids = json.load(open(os.path.join(ROOT, "split.json"), encoding="utf-8"))["val_ids"]
    all_ids = list(range(6996))
    random.Random(123).shuffle(all_ids)
    if val_ids != all_ids[:500]:
        raise RuntimeError("fixed split does not match seed123 shuffled first 500 IDs")
    train_ids = all_ids[500:]  # 历史顺序：保留 seed123 打乱后的剩余序列，禁止排序

    ckpt = ckpt_path(a.crop, a.barren, a.ckpt_name)
    if os.path.exists(ckpt) != a.resume:
        raise RuntimeError("Existing checkpoint requires explicit --resume; fresh training requires a new path")
    model = build_model()
    if a.crop >= 640:  # 梯度检查点: 2.86GB@bs2x640, 以时间换显存
        _inner = getattr(model.encoder, "model", None)
        if _inner is not None and hasattr(_inner, "set_grad_checkpointing"):
            _inner.set_grad_checkpointing(True)
            print("[grad-ckpt] 已启用 (crop>=640)")
    dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=IGNORE)
    cew = list(CE_WEIGHTS)
    if a.barren:  # v4.2: 荒地定向 (最大混淆对 Barren->Background 22.7%)
        cew[4] = 2.2  # Barren CE 权重 1.66 -> 2.2
    w = torch.tensor([1.0] + cew, dtype=torch.float32).to(DEVICE)
    lovasz = smp.losses.LovaszLoss(mode="multiclass", ignore_index=IGNORE)
    warm = (a.crop != 512 or a.barren) and not os.path.exists(ckpt)
    warm_from = None
    if warm and a.warm_from:
        warm_from = a.warm_from
    elif warm:  # barren->同crop普通权重; 否则优先最近的低分辨率权重
        cands = [(a.crop, False)] if a.barren else []
        cands += [(cc, False) for cc in (640, 512) if cc != a.crop]
        for cc, bb in cands:
            if os.path.exists(ckpt_path(cc, bb)):
                warm_from = (cc, bb)
                break
    lr = 3e-5 if warm else 6e-5
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.target_iters, eta_min=3e-6)

    start, hist, ema = 0, [], None
    resumed_parents = None
    if os.path.exists(ckpt):
        sd = P.load_checkpoint(ckpt, map_location=DEVICE)
        model.load_state_dict(sd["model"]); opt.load_state_dict(sd["opt"])
        sched.load_state_dict(sd["sched"]); start = sd["iter"]; hist = sd.get("hist", [])
        ema = sd.get("ema")
        resumed_parents = list(sd["strict_repro"].get("parents", []))
        print(f"[resume] from iter {start}")
    elif warm and warm_from is not None:
        warm_path = (ckpt_path(name=warm_from) if isinstance(warm_from, str)
                     else ckpt_path(*warm_from))
        sd = P.load_checkpoint(warm_path, map_location=DEVICE)
        if a.strict_v4_1:
            meta = sd.get("strict_repro", {})
            if (sd.get("iter") != 12000 or meta.get("stage") != a.strict_parent_stage or
                    meta.get("seed") != 42 or meta.get("parents") != ["public_swin_b"]):
                raise RuntimeError("strict v4.1 parent does not match the declared v4 EMA lineage")
        model.load_state_dict(sd["ema"] if sd.get("ema") is not None else sd["model"])
        print(f"[warm] {warm_from} EMA iter={sd['iter']} -> crop{a.crop}"
              f"{' barren-ft' if a.barren else ''} (fresh optimizer, lr={lr})")

    # 温和采样 (与 v3.1 相同); v4.2 荒地定向加强
    stats = np.load(os.path.join(ROOT, "train_mask_stats.npy"))
    boost = np.array([0, 1.2, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.5], dtype=np.float32)
    if a.barren:
        boost[5] = 2.5  # 含荒地图片采样权重上调
    wi = (stats[train_ids] * boost).sum(axis=1)
    wi = wi / wi.mean()
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.from_numpy(wi).double(), num_samples=len(train_ids) * 4, replacement=True)
    print(f"[swin{a.crop}] 采样权重: mean=1.0, max={wi.max():.2f}, min={wi.min():.2f}")
    dl = DataLoader(UAV(train_ids, True, a.crop, extra=(a.crop != 512),
                        darken=a.darken, full_image_scale_up=a.full_image_scale_up,
                        frozen_7439_scale_floor=a.frozen_7439_scale_floor),
                    batch_size=2, sampler=sampler,  # bs1 会触发 PSP+BN 单样本错误, 保持 bs2
                    num_workers=3, persistent_workers=True, drop_last=True)

    if ema is None:
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print("[swin] EMA 已初始化 (decay=0.999)")
    ema_pairs = []
    msd = model.state_dict()
    for k, et in ema.items():
        if et.dtype.is_floating_point and k in msd:
            ema_pairs.append((et, msd[k]))
    ema_list = [p[0] for p in ema_pairs]
    src_list = [p[1] for p in ema_pairs]
    print(f"[EMA] 绑定 {len(ema_pairs)} 个张量")

    parent_names = resumed_parents or (["public_swin_b"] if warm_from is None else [
        warm_from if isinstance(warm_from, str) else os.path.basename(ckpt_path(*warm_from))])

    def save_ck(it, hist):
        metadata = P.strict_metadata(a.stage, a.seed, parent_names)
        metadata["random_scale_floor"] = (
            "crop_frozen_7439_parent_lineage"
            if a.frozen_7439_scale_floor else "0.75_crop_historical_formula"
        )
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), iter=it, hist=hist, ema=ema,
                        strict_repro=metadata), ckpt)

    it = start; t0 = time.time(); model.train(); done = False
    opt.zero_grad()
    accum = 1  # bs2 固定; 640 bs1 不可行 (PSP+BN), 640 bs2 又超显存, 故上限 576
    it = start; micro = 0; t0 = time.time(); model.train(); done = False
    opt.zero_grad()
    while it < a.target_iters and not done:
        for x, y, wmap in dl:
            if it >= a.target_iters or (a.stop_after_iter is not None and it >= a.stop_after_iter) or time.time() - t0 > a.max_seconds:
                done = True; break
            x, y, wmap = x.to(DEVICE), y.to(DEVICE), wmap.to(DEVICE)
            with torch.autocast("cuda", torch.float16, enabled=DEVICE == "cuda"):
                out = model(x)
                ce_none = F.cross_entropy(out, y, weight=w, ignore_index=IGNORE, reduction="none")
                out_ds = F.interpolate(out, scale_factor=0.5, mode="bilinear", align_corners=False)
                y_ds = F.interpolate(y.unsqueeze(1).float(), scale_factor=0.5,
                                     mode="nearest").squeeze(1).long()
                loss = ((ce_none * wmap).sum() / wmap.sum() + dice(out, y)
                        + 0.5 * lovasz(out_ds, y_ds)) / accum
            loss.backward(); micro += 1
            if micro % accum != 0:
                continue
            opt.step(); opt.zero_grad(); sched.step()
            with torch.no_grad():
                torch._foreach_mul_(ema_list, 0.999)
                torch._foreach_add_(ema_list, src_list, alpha=0.001)
            hist.append(round(loss.item() * accum, 4)); it += 1
            if it % 50 == 0:
                print(f"iter {it}/{a.target_iters} loss={np.mean(hist[-50:]):.4f} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
            if it % 200 == 0:
                save_ck(it, hist)
    save_ck(it, hist)
    print(f"[saved] iter={it}, peak={torch.cuda.max_memory_allocated()/2**30:.2f}GB")
    print("[done]" if it >= a.target_iters else f"[partial] {it}/{a.target_iters}")


def load_trained(use_ema=True, crop=512, barren=False, path=None):
    model = build_model(weights=None)
    selected = str(P.assert_run_path(path, must_exist=True)) if path else ckpt_path(crop, barren)
    sd = P.load_checkpoint(selected, map_location=DEVICE)
    if use_ema and sd.get("ema") is not None:
        model.load_state_dict(sd["ema"])
        print(f"[load] {os.path.basename(selected)} EMA iter={sd['iter']}")
    else:
        model.load_state_dict(sd["model"])
        print(f"[load] {os.path.basename(selected)} iter={sd['iter']}")
    return model, sd["iter"]


def load_bias(crop=512, barren=False):
    bf = bias_path(crop, barren)
    if os.path.exists(bf):
        b = torch.from_numpy(np.load(bf).astype(np.float32))
        print(f"[bias:{os.path.basename(bf)}] {np.round(b.numpy(), 3).tolist()}")
        return b
    return None


def run_eval(tta, a, use_ema=True):
    model, iters = load_trained(use_ema, a.crop, a.barren)
    val_ids = json.load(open(os.path.join(ROOT, "split.json")))["val_ids"]
    bias = load_bias(a.crop, a.barren)
    miou, ious = evaluate(model, val_ids[:a.n], tta, bias)
    res = dict(iters=iters, tta=tta, n=a.n, ema=use_ema, bias=(bias is not None),
               val_mIoU=round(miou, 4),
               val_IoU={n: (None if np.isnan(v) else round(v, 4))
                        for n, v in zip(CLASS_NAMES, ious)})
    tag = "swin" + ("" if a.crop == 512 else f"_{a.crop}") + ("_barren" if a.barren else "")
    key = tag + "_" + ("plain" if not tta else (f"ms_n{a.n}" if tta == "ms" else f"tta_n{a.n}"))
    if use_ema:
        key += "_ema"
    if bias is not None:
        key += "_bias"
    log = json.load(open(LOG)) if os.path.exists(LOG) else {}
    log[key] = res
    json.dump(log, open(LOG, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))


@torch.no_grad()
def calibrate(a):
    model, iters = load_trained(use_ema=not a.raw, crop=a.crop, barren=a.barren); model.eval()
    n = min(a.n, 100)
    val_ids = json.load(open(os.path.join(ROOT, "split.json")))["val_ids"][:n]
    cache_ms, cache_tta, cache_plain = [], [], []
    for i, k in enumerate(val_ids):
        img = np.array(Image.open(os.path.join(IMG_DIR, f"{k:04d}.png")).convert("RGB"))
        msk = np.array(Image.open(os.path.join(MSK_DIR, f"{k:04d}.png")))
        cache_ms.append((predict_logits(model, img, tta="ms").half(), msk))
        cache_tta.append((predict_logits(model, img, tta=True).half(), msk))
        cache_plain.append((predict_logits(model, img, tta=False).half(), msk))
        if (i + 1) % 25 == 0:
            print(f"  cached {i+1}/{len(val_ids)}", flush=True)

    def score(b, cache):
        conf = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
        bt = torch.from_numpy(b).view(-1, 1, 1)
        for lg, msk in cache:
            pred = (lg[:, ::2, ::2].float() + bt).argmax(0).numpy().ravel()
            t = msk[::2, ::2].ravel()
            m = t != IGNORE
            conf += np.bincount(NUM_CLASSES * t[m] + pred[m],
                                minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
        inter = np.diag(conf).astype(np.float64)
        union = conf.sum(1) + conf.sum(0) - inter
        iou = [inter[c] / union[c] if union[c] > 0 else float("nan") for c in range(1, NUM_CLASSES)]
        return float(np.nanmean(iou)), iou

    bias = np.zeros(NUM_CLASSES, np.float32)
    base_ms, _ = score(bias, cache_ms)
    base_tta, _ = score(bias, cache_tta)
    base_plain, _ = score(bias, cache_plain)
    routes = {"ms": (base_ms, cache_ms), "tta": (base_tta, cache_tta),
              "plain": (base_plain, cache_plain)}
    route = max(routes, key=lambda r: routes[r][0])
    best, cache = routes[route]
    print(f"[calib] iters={iters} n={n} base: ms={base_ms:.4f} tta={base_tta:.4f} "
          f"plain={base_plain:.4f} -> 校准 {route} 路", flush=True)
    for rd in range(3):
        improved = False
        for c in range(1, NUM_CLASSES):
            for d in (0.05, -0.05):
                b2 = bias.copy(); b2[c] += d
                m2, _ = score(b2, cache)
                if m2 > best + 1e-5:
                    bias = b2; best = m2; improved = True
                    print(f"[calib] rd{rd} cls{c}{d:+.2f} -> {m2:.4f}", flush=True)
        if not improved:
            break
    np.save(bias_path(a.crop, a.barren), bias)
    _, ious = score(bias, cache)
    print(f"[calib] final={best:.4f} route={route} bias={np.round(bias, 3).tolist()}")
    print("[calib] per-class IoU:",
          {n_: (None if np.isnan(v) else round(v, 4)) for n_, v in zip(CLASS_NAMES, ious)})


@torch.no_grad()
def infer(a):
    model, iters = load_trained(crop=a.crop, barren=a.barren); model.eval()
    bias = load_bias(a.crop, a.barren)
    tag = "swin" + ("" if a.crop == 512 else f"_{a.crop}") + ("_barren" if a.barren else "")
    out_dir = os.path.join(ARTIFACT_ROOT, f"output_test1_{tag}" + ("_cal" if bias is not None else ""))
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(os.listdir(TEST_DIR))
    for n, fn in enumerate(files):
        out_path = os.path.join(out_dir, fn)
        if os.path.exists(out_path):
            continue
        img = np.array(Image.open(os.path.join(TEST_DIR, fn)).convert("RGB"))
        pred = predict_logits(model, img, tta="ms", bias=bias).argmax(0).numpy().astype(np.uint8)
        Image.fromarray(pred, mode="L").save(out_path)
        if (n + 1) % 25 == 0:
            print(f"infer {n+1}/{len(files)}", flush=True)
    for fn in files[:5] + files[-5:]:
        im = Image.open(os.path.join(out_dir, fn))
        assert im.mode == "L" and im.size == (1024, 1024), (fn, im.mode, im.size)
        assert set(np.unique(im)) <= set(range(9)), fn
    import shutil
    shutil.make_archive(out_dir, "zip", out_dir)
    print(f"[infer done] {len(files)} 张, iters={iters}, tta=ms, zip={out_dir}.zip 已生成并通过格式自检")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Explicitly restore a same-run checkpoint")
    ap.add_argument("--raw", action="store_true", help="评估/校准时不使用EMA权重")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--eval-tta", action="store_true")
    ap.add_argument("--eval-ms", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--target-iters", type=int, default=12000)
    ap.add_argument("--stop-after-iter", type=int, default=None,
                    help="Controlled historical interruption; scheduler horizon is unchanged")
    ap.add_argument("--max-seconds", type=int, default=240)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--barren", action="store_true", help="v4.2: 荒地定向微调 (CE权重+过采样)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage", type=str, default="main_v4")
    ap.add_argument("--ckpt-name", type=str, default=None)
    ap.add_argument("--warm-from", type=str, default=None,
                    help="strict run root 内的显式父断点文件名")
    ap.add_argument("--strict-v4-1", action="store_true", dest="strict_v4_1")
    ap.add_argument("--strict-parent-stage", type=str, default=None,
                    help="严格v4.1唯一父断点的stage")
    ap.add_argument("--darken", action="store_true",
                    help="后加的v5暗化；严格v4.1禁止")
    ap.add_argument("--full-image-scale-up", action="store_true", dest="full_image_scale_up",
                    help="47阶段专用整图scale-up；v4/v4.1/KAN禁止")
    ap.add_argument("--frozen-7439-scale-floor", action="store_true",
                    dest="frozen_7439_scale_floor",
                    help="只用于复现最终69.83链所选0.7439 v4/v4.1父血统的crop尺度下限")
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    if a.train: train(a)
    elif a.calibrate: calibrate(a)
    elif a.eval: run_eval(False, a, use_ema=not a.raw)
    elif a.eval_tta: run_eval(True, a, use_ema=not a.raw)
    elif a.eval_ms: run_eval("ms", a, use_ema=not a.raw)
    elif a.infer: infer(a)
