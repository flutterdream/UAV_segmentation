# -*- coding: utf-8 -*-
"""
dacs_train.py — DACS 在线自训练 (Domain Adaptation via Cross-domain Mixed Sampling)

队友 C6 配方核心 (66.63 -> 67.33 的关键): EMA teacher 在线打标 + 置信度阈值 + ClassMix。
与离线 PL/Distill 的本质区别: 教师不是静态标签而是学生自己的 EMA, 随训练自我改进,
不受"蒸馏上限=教师分"的瓶颈约束。

机制 (每步):
  1. 有标签批: 训练集 640 crop ×2 (沿用主线数据管线/加权采样)
  2. 目标批: test1 随机 640 crop ×2
  3. EMA 教师 fp16 在线推理目标 crop -> 伪标签
  4. 不确定像素处理: softmax 置信度 < τ 的像素 -> 默认 Ignore;
     --team-fill 开启时用队友 67.33 线预测回填(成熟模型辅助标注 unclear 区域)
  5. ClassMix: 从伪标签出现的类中随机选一半, 把这些类的像素(图+标签)粘贴到有标签 crop 上
  6. 学生在混合图上训练 (主线损失配方), EMA 更新, 教师=EMA 同步

暖启动: ckpt_kandist_i2000 EMA (当前最佳 65.29)
选拔: 快照每 1000 步; val 守卫(>=0.72) + 队友一致率(横向比较, 基线 0.6688)
用法:
  python dacs_train.py --train --target-iters 4000 --max-seconds 200
  python dacs_train.py --agree 50 ckpt_kan_dacs_i1000.pth ckpt_kan_dacs_i2000.pth ...
  python dacs_train.py --eval-ms2 --ckpt ckpt_kan_dacs_i2000.pth --n 100
  python dacs_train.py --infer --ckpt ckpt_kan_dacs_i2000.pth
"""
import argparse, json, os, random, time
import numpy as np
from PIL import Image

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ===== 加速: cudnn 卷积算法自动寻优 + TF32 (不改训练语义) =====
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True          # crop 固定 640, 算法寻优收益稳定
    torch.backends.cuda.matmul.allow_tf32 = True   # fp32 matmul 走 TF32
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

import seg_swin as S
import kan_swin as K

ROOT = os.path.dirname(os.path.abspath(__file__))
TEAM = os.path.join(ROOT, "teammate_best")
NUM_CLASSES, IGNORE = 9, 0
DEVICE = S.DEVICE
TAU = 0.85
CONFLICT_IGNORE = False  # 队友68.89规则: 低置信 & 教师-锚冲突 -> 保持Ignore不回填
SOURCE_W = 0.0  # ★(9/14) source_supervised 支路: 未混合源域监督损失权重
                #   队友 69.08 核心 = DACS 主损失 + 0.20 × L_source(未混合官方源域监督)
                #   0.0 = 关闭(默认, 与旧行为逐位一致); --source-loss-weight 覆盖
TEACHER_HFLIP = False  # ★(9/16) 队友 69.08 配方里的 teacher-hflip:
                       #   教师打标时做水平翻转 TTA(原图 + 翻转后平均概率), 提升伪标签质量。
                       #   默认关(与旧行为逐位一致); --teacher-hflip 开启
DEFAULT_WARM = os.path.join(ROOT, "ckpt_kandist_i2000.pth")  # 第一轮起点(蒸馏65.29)


# ================== 目标域 (test1) 采样池 ==================
class TargetPool:
    """test1 图像池: 随机图 + 随机 crop 区域; 锚定预测(队友或自我)按同区域裁剪"""

    def __init__(self, anchor_dir):
        self.files = sorted(os.listdir(S.TEST_DIR))
        self.anchor_dir = anchor_dir
        self.anchor = {}
        self._imgs = {}  # 图像缓存(9/13 由300降到48: 300张≈900MB, 会把24GB内存压到71%+
                         #   引发 Windows内存压缩 / WDDM把显存页换出 → GPU thrash, 得不偿失)

    def _img(self, fn):
        if fn not in self._imgs:
            if len(self._imgs) > 48:
                self._imgs.pop(next(iter(self._imgs)))
            self._imgs[fn] = np.array(
                Image.open(os.path.join(S.TEST_DIR, fn)).convert("RGB"))
        return self._imgs[fn]

    def _anchor_pred(self, fn):
        if fn not in self.anchor:
            fp = os.path.join(self.anchor_dir, fn)
            self.anchor[fn] = (np.array(Image.open(fp))
                               if os.path.exists(fp) else None)
        return self.anchor[fn]

    def sample(self, crop):
        """返回 (target_crop_img, fn, y, x)"""
        fn = random.choice(self.files)
        img = self._img(fn)
        H, W = img.shape[:2]
        c = min(crop, H, W)
        y = random.randint(0, H - c)
        x = random.randint(0, W - c)
        return img[y:y + c, x:x + c].copy(), fn, y, x, c

    def anchor_crop(self, fn, y, x, c):
        t = self._anchor_pred(fn)
        return None if t is None else t[y:y + c, x:x + c]


# ================== 教师打标 + 不确定像素处理 ==================
@torch.no_grad()
def teacher_label(teacher, timg_np, tau=TAU, team_crop=None):
    """EMA 教师 fp16 推理 -> (伪标签HxW, 有效像素占比, 回填占比)
    低置信像素: 有队友预测则回填(team_crop), 否则 Ignore;
    conflict_ignore: 低置信且教师硬标签与锚不一致的像素保持 Ignore(不回填)
    TEACHER_HFLIP: 教师打标做水平翻转 TTA(原图 + 翻转平均概率), 提升伪标签质量"""
    x = S.norm(timg_np).unsqueeze(0).to(DEVICE)
    with torch.autocast("cuda", torch.float16, enabled=DEVICE == "cuda"):
        out = teacher(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
    prob = F.softmax(out.float(), dim=1)[0]          # (9,h,w)
    if TEACHER_HFLIP:
        # ★(9/16) teacher-hflip: 翻转再推理, 概率翻回后与原图各占一半 = 2 路 TTA
        with torch.autocast("cuda", torch.float16, enabled=DEVICE == "cuda"):
            out_f = teacher(torch.flip(x, dims=[3]))
            if isinstance(out_f, (tuple, list)):
                out_f = out_f[0]
        prob = (prob + torch.flip(F.softmax(out_f.float(), dim=1)[0], dims=[2])) * 0.5
        del out_f
    conf, pl = prob.max(0)
    pl = pl.cpu().numpy()
    conf = conf.cpu().numpy()
    lowc = conf < tau
    fill_ratio = 0.0
    if lowc.any():
        pl_raw = pl.copy()          # 置Ignore前的教师原始标签(冲突判定用)
        pl[lowc] = IGNORE
        if team_crop is not None:
            tc = team_crop
            if CONFLICT_IGNORE:
                # 68.89: 仅回填教师==锚的一致像素; 冲突像素(最不可靠)保持Ignore不学
                fill = lowc & (tc != IGNORE) & (pl_raw == tc)
            else:
                fill = lowc & (tc != IGNORE)
            pl[fill] = tc[fill]
            fill_ratio = fill.sum() / max(pl.size, 1)
    valid = (pl != IGNORE).mean()
    return pl.astype(np.uint8), valid, fill_ratio


# ================== ClassMix ==================
def classmix(img, lab, timg, tpl, min_px=50):
    """把目标域伪标签中约一半类的像素粘贴到源图 (DACS 原版: 类数减半随机).
    返回混合后的 (img, lab)"""
    present = [c for c in range(1, NUM_CLASSES)
               if (tpl == c).sum() >= min_px]
    if not present:
        return img, lab
    k = max(1, len(present) // 2)
    chosen = set(random.sample(present, k))
    mask = np.isin(tpl, list(chosen))
    if not mask.any():
        return img, lab
    img = img.copy()
    lab = lab.copy()
    img[mask] = timg[mask]
    lab[mask] = tpl[mask]
    return img, lab


def boundary_wmap(lab):
    """混合标签的边界带权重 (与主线 UAV 数据集一致: 5x5 形态学边界 x3.0)"""
    import cv2
    band = cv2.dilate(lab, np.ones((5, 5), np.uint8)) != \
        cv2.erode(lab, np.ones((5, 5), np.uint8))
    return np.where(band, 3.0, 1.0).astype(np.float32)


# ================== 教师 (EMA 的 fp16 镜像) ==================
def build_teacher(state_dict):
    """从 EMA state dict 建教师 (fp32 参数 + autocast fp16 推理, 与主线 predict_logits 同模式)"""
    teacher = K.DeepKANSegSwin(NUM_CLASSES)
    teacher.load_state_dict(state_dict)
    teacher = teacher.to(DEVICE).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


@torch.no_grad()
def sync_teacher(teacher, ema):
    """把 EMA 权重(含 BN 运行统计)整体同步进教师 (按名 copy, 严格匹配)"""
    teacher.load_state_dict(ema)


def _fsync(path):
    """强制把文件内容刷到磁盘.
    torch.save + os.replace 若无 fsync, 断电/系统睡眠时数据可能未落盘,
    导致 ckpt zip 结构不完整(central directory 缺失)而无法读取."""
    try:
        fd = os.open(path, os.O_RDWR)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception as e:
        print(f"[fsync] 跳过 {os.path.basename(path)} ({e})")


# ================== 训练 ==================
def train(a):
    _sd = getattr(a, "seed", 42)
    random.seed(_sd); np.random.seed(_sd); torch.manual_seed(_sd)
    all_ids = list(range(6996))
    random.Random(123).shuffle(all_ids)  # val/train 划分固定
    val_ids, train_ids = all_ids[:500], all_ids[500:]

    model = K.build_model()
    crop = a.crop
    tag = a.tag
    CKPT = os.path.join(ROOT, f"ckpt_kan_{tag}.pth")
    WARM = a.warm or DEFAULT_WARM

    # 暖启动 / 断点续训
    start, hist, ema = 0, [], None
    # ★显存优化(9/13): ckpt 一律先在 CPU 加载, 再 load_state_dict 拷入模型。
    #   原实现用 map_location=DEVICE 会在显存里多驻留一整份 1.5GB 权重,
    #   6GB 卡(桌面程序已占1.3GB)上会直接触发驱动分页 thrash。
    if os.path.exists(CKPT):
        sd = torch.load(CKPT, map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model"])
        start = sd["iter"]; hist = sd.get("hist", [])
        ema = sd.get("ema")
        if ema is not None:
            ema = {k: v.to(DEVICE) for k, v in ema.items()}
        print(f"[resume] from iter {start}")
    else:
        sd = torch.load(WARM, map_location="cpu", weights_only=False)
        model.load_state_dict(sd.get("ema") or sd["model"])
        print(f"[warm] {os.path.basename(WARM)} EMA iter={sd['iter']} (起点)")
        sd = None

    if crop >= 640:
        _inner = getattr(model.encoder, "model", None)
        if _inner is not None and hasattr(_inner, "set_grad_checkpointing"):
            _inner.set_grad_checkpointing(True)
            print("[grad-ckpt] 已启用")

    import segmentation_models_pytorch as smp
    dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=IGNORE)
    lovasz = smp.losses.LovaszLoss(mode="multiclass", ignore_index=IGNORE)
    # 类权重可由命令行覆盖 —— 针对性提升不同语义类型
    ce_w = list(S.CE_WEIGHTS)
    if getattr(a, "ce_weights", None):
        ce_w = [float(x) for x in a.ce_weights.split(",")]
        print(f"[ce-weights] 覆盖为 {ce_w}", flush=True)
    w = torch.tensor([1.0] + ce_w, dtype=torch.float32).to(DEVICE)

    enc_params = list(model.encoder.parameters())
    new_params = [p for m_ in (model.dfr_in, model.dfr_blocks, model.dfr_out, model.decoder)
                  for p in m_.parameters()]
    lr_enc, lr_new = 3e-5, 1.5e-4
    if a.restart_lr > 0:
        lr_enc *= a.restart_lr; lr_new *= a.restart_lr
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": lr_enc},
        {"params": new_params, "lr": lr_new}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.target_iters, eta_min=3e-6)

    if start > 0 and sd is not None:
        # ★复用上面已加载到 CPU 的 sd, 不再二次读盘 + 二次上显存(省约 1.6GB 显存尖峰)
        opt.load_state_dict(sd["opt"]); sched.load_state_dict(sd["sched"])
        del sd

    # 有标签数据管线 (与主线一致)
    stats = np.load(os.path.join(ROOT, "train_mask_stats.npy"))
    boost = np.array([0, 1.2, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.5], dtype=np.float32)
    if getattr(a, "boost", None):
        boost = np.array([0] + [float(x) for x in a.boost.split(",")], dtype=np.float32)
        print(f"[boost] 覆盖为 {boost[1:].tolist()}", flush=True)
    wi = (stats[train_ids] * boost).sum(axis=1)
    wi = wi / wi.mean()
    msk_dir = None
    if getattr(a, "clean_mask", False):
        msk_dir = os.path.join(ROOT, "official", "train", "train", "masks_clean")
        print(f"[clean-mask] 源域标签改用 {msk_dir}")
    ds = S.UAV(train_ids, True, crop, extra=True, msk_dir=msk_dir)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.from_numpy(wi).double(), num_samples=len(train_ids) * 4, replacement=True)
    dl = DataLoader(ds, batch_size=a.batch, sampler=sampler,
                    num_workers=3, persistent_workers=True, drop_last=True)

    # EMA (0.999), 教师 = EMA fp16 镜像
    if ema is None:
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema_pairs = []
    msd = model.state_dict()
    for k, et in ema.items():
        if et.dtype.is_floating_point and k in msd:
            ema_pairs.append((et, msd[k]))
    ema_list = [p[0] for p in ema_pairs]
    src_list = [p[1] for p in ema_pairs]

    teacher = build_teacher(ema)
    sync_teacher(teacher, ema)
    anchor_dir = a.anchor or TEAM
    pool = TargetPool(anchor_dir)
    team_fill = a.team_fill and os.path.isdir(anchor_dir)
    print(f"[dacs] τ={TAU} fill={'on' if team_fill else 'off'}(锚={os.path.basename(anchor_dir)}) "
          f"crop={crop} target=test1(500) lr×{a.restart_lr or 1} "
          f"source_loss_λ={SOURCE_W} hflip={'on' if TEACHER_HFLIP else 'off'} "
          f"scale_up={'on' if S.SCALE_UP else 'off'}")

    def save_ck(it, hist):
        tmp = CKPT + ".tmp"
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), iter=it, hist=hist, ema=ema), tmp)
        _fsync(tmp)                      # 强制落盘, 防断电/睡眠导致 ckpt 损坏
        os.replace(tmp, CKPT)
        if it % 1000 == 0 and it > 0:
            snap = os.path.join(ROOT, f"ckpt_kan_{tag}_i{it}.pth")
            torch.save(dict(model=model.state_dict(), iter=it, ema=ema), snap)
            _fsync(snap)

    it = start; t0 = time.time(); model.train(); done = False
    stat_v, stat_f, stat_n = [], [], []
    hist_src = []  # source_supervised 支路损失(仅监控用)
    opt.zero_grad()
    while it < a.target_iters and not done:
        for x_l, y_l, _ in dl:  # wmap 用混合后标签重算, 弃用原始
            if it >= a.target_iters or time.time() - t0 > a.max_seconds:
                done = True; break
            # --- 对 batch 内每个样本独立做 DACS 混合 ---
            xs, ys, ws = [], [], []
            wsrc = []  # source_supervised: 未混合源标签的边界权重(仅 SOURCE_W>0 时填充)
            for bi in range(x_l.size(0)):
                img = (x_l[bi].permute(1, 2, 0).numpy()
                       * np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
                       + np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3))
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
                lab = y_l[bi].numpy().astype(np.uint8)
                timg, fn, ty, tx, tc = pool.sample(crop)
                timg_r = np.array(Image.fromarray(timg).resize(
                    (crop, crop), Image.BILINEAR)) if tc != crop else timg
                tcrop = pool.anchor_crop(fn, ty, tx, tc) if team_fill else None
                if tcrop is not None and tc != crop:
                    tcrop = np.array(Image.fromarray(tcrop).resize(
                        (crop, crop), Image.NEAREST))
                tpl, valid, fill = teacher_label(teacher, timg_r, TAU, tcrop)
                img_m, lab_m = classmix(img, lab, timg_r, tpl)
                stat_v.append(valid); stat_f.append(fill); stat_n.append(1)
                xs.append(S.norm(img_m))
                ys.append(torch.from_numpy(lab_m.copy()).long())
                ws.append(torch.from_numpy(boundary_wmap(lab_m)))
                if SOURCE_W > 0:
                    # lab 此刻仍是未混合的原始源标签(本循环开头从 y_l[bi] 取, classmix 只改 img_m/lab_m)
                    wsrc.append(torch.from_numpy(boundary_wmap(lab)))
            x = torch.stack(xs).to(DEVICE)
            y = torch.stack(ys).to(DEVICE)
            wmap = torch.stack(ws).to(DEVICE)

            opt.zero_grad()
            # ★(9/14) source_supervised 支路 (队友 69.08 的核心机制):
            #   DACS 主损失 + λ·L_source, L_source = 未混合官方源图上的 CE(边界加权)+Dice+0.5Lovász。
            #   ★显存安全: 先反传该支路并释放其计算图, 再构建主图 → 两张图不共存, 峰值不叠加
            #     (守 6GB thrash 铁律)。两支路各自 backward 但只做一次 opt.step()。
            #   λ(SOURCE_W)=0 时整段跳过 → 与原实现逐位一致, 保证单变量对照干净。
            if SOURCE_W > 0:
                x_src, y_src = x_l.to(DEVICE), y_l.to(DEVICE)
                wmap_src = torch.stack(wsrc).to(DEVICE)
                with torch.autocast("cuda", torch.float16, enabled=DEVICE == "cuda"):
                    out_src = model(x_src)
                    ce_src = F.cross_entropy(out_src, y_src, weight=w,
                                             ignore_index=IGNORE, reduction="none")
                    out_src_ds = F.interpolate(out_src, scale_factor=0.5,
                                               mode="bilinear", align_corners=False)
                    y_src_ds = F.interpolate(y_src.unsqueeze(1).float(), scale_factor=0.5,
                                             mode="nearest").squeeze(1).long()
                    loss_src = ((ce_src * wmap_src).sum() / wmap_src.sum() + dice(out_src, y_src)
                                + 0.5 * lovasz(out_src_ds, y_src_ds))
                (SOURCE_W * loss_src).backward()
                hist_src.append(round(float(loss_src.item()), 4))
                del out_src, ce_src, out_src_ds, y_src_ds, loss_src

            with torch.autocast("cuda", torch.float16, enabled=DEVICE == "cuda"):
                out = model(x)
                ce_none = F.cross_entropy(out, y, weight=w, ignore_index=IGNORE, reduction="none")
                out_ds = F.interpolate(out, scale_factor=0.5, mode="bilinear", align_corners=False)
                y_ds = F.interpolate(y.unsqueeze(1).float(), scale_factor=0.5,
                                     mode="nearest").squeeze(1).long()
                loss = ((ce_none * wmap).sum() / wmap.sum() + dice(out, y)
                        + 0.5 * lovasz(out_ds, y_ds))
            loss.backward(); opt.step(); sched.step()
            with torch.no_grad():
                torch._foreach_mul_(ema_list, 0.999)
                torch._foreach_add_(ema_list, src_list, alpha=0.001)
            sync_teacher(teacher, ema)  # 教师=EMA, 每步同步
            hist.append(round(loss.item(), 4)); it += 1
            if it % 50 == 0:
                v = np.mean(stat_v[-100:]) if stat_v else 0
                f = np.mean(stat_f[-100:]) if stat_f else 0
                src_s = f" src={np.mean(hist_src[-50:]):.4f}" if hist_src else ""
                print(f"iter {it}/{a.target_iters} loss={np.mean(hist[-50:]):.4f}{src_s} "
                      f"教师有效px={v:.1%} 队友回填={f:.1%} "
                      f"elapsed={time.time()-t0:.0f}s "
                      f"peak={torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)
            if it % 200 == 0:
                save_ck(it, hist)
    save_ck(it, hist)
    print(f"[saved] iter={it}, peak={torch.cuda.max_memory_allocated()/2**30:.2f}GB")
    print("[done]" if it >= a.target_iters else f"[partial] {it}/{a.target_iters}")


# ================== 评估 / 一致率 / 推理 ==================
@torch.no_grad()
def agree(ckpt, files):
    model, _it = K.load_trained(path=os.path.join(ROOT, ckpt))
    model.eval()
    inter = np.zeros(9, np.float64); union = np.zeros(9, np.float64)
    for fn in files:
        img = np.array(Image.open(os.path.join(S.TEST_DIR, fn)).convert("RGB"))
        pred = S.predict_logits(model, img).argmax(0).numpy()
        t = np.array(Image.open(os.path.join(TEAM, fn)))
        for c in range(1, 9):
            p, tt = pred == c, t == c
            inter[c] += (p & tt).sum(); union[c] += (p | tt).sum()
        del img, pred
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return float(np.nanmean(iou))


def run_eval(a, tta="ms2"):
    model, iters = K.load_trained(path=a.ckpt)
    val_ids = json.load(open(os.path.join(ROOT, "split.json")))["val_ids"][:a.n]
    miou, ious = S.evaluate(model, val_ids, tta)
    print(f"[val] iters={iters} tta={tta} n={a.n} mIoU={miou:.4f}")
    for n_, v in zip(S.CLASS_NAMES, ious):
        print(f"  {n_:<14}{v:.4f}" if not np.isnan(v) else f"  {n_:<14}N/A")


def infer(a):
    model, iters = K.load_trained(path=a.ckpt)
    model.eval()
    out_dir = os.path.join(ROOT, a.out or "output_test1_kan_dacs_ms2")
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(os.listdir(S.TEST_DIR))
    for n_, fn in enumerate(files):
        out_path = os.path.join(out_dir, fn)
        if os.path.exists(out_path):
            continue
        img = np.array(Image.open(os.path.join(S.TEST_DIR, fn)).convert("RGB"))
        pred = S.predict_logits(model, img, tta=a.tta or "ms2").argmax(0).numpy().astype(np.uint8)
        Image.fromarray(pred, mode="L").save(out_path)
        if (n_ + 1) % 25 == 0:
            print(f"infer {n_+1}/{len(files)}", flush=True)
    for fn in files[:5] + files[-5:]:
        im = Image.open(os.path.join(out_dir, fn))
        assert im.mode == "L" and im.size == (1024, 1024), (fn, im.mode, im.size)
        assert set(np.unique(im)) <= set(range(9)), fn
    import shutil
    shutil.make_archive(out_dir, "zip", out_dir)
    print(f"[infer done] {len(files)}张 iters={iters} -> {out_dir}.zip 已通过格式自检")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--agree", type=int, default=None,
                    help="队友一致率选拔: --agree N ckpt1 ckpt2 ...")
    ap.add_argument("--eval-ms2", action="store_true")
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--out", type=str, default=None,
                    help="推理输出目录名(默认 output_test1_kan_dacs_ms2)")
    ap.add_argument("--batch", type=int, default=2,
                    help="训练 batch size(默认 2, 与历史行为一致)")
    ap.add_argument("--target-iters", type=int, default=4000)
    ap.add_argument("--restart-lr", type=float, default=0.25)
    ap.add_argument("--max-seconds", type=int, default=200)
    ap.add_argument("--crop", type=int, default=640)
    ap.add_argument("--seed", type=int, default=42, help="随机种子(默认42)")
    ap.add_argument("--tau", type=float, default=None,
                    help="覆盖教师置信度阈值 TAU(默认0.85)")
    ap.add_argument("--tta", type=str, default=None,
                    help="TTA 模式 (ms/ms2/ms2b/ms2c/ms2d/ms2x; 默认 ms2)。"
                         "9/18 起同时作用于 --eval-ms2(源域val) 与 --infer(生成输出)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warm", type=str, default=None,
                    help="暖启动权重路径(默认 ckpt_kandist_i2000.pth)")
    ap.add_argument("--tag", type=str, default="dacs",
                    help="检查点/快照命名前缀(默认 dacs; 第二轮用 dacs2)")
    ap.add_argument("--anchor", type=str, default=None,
                    help="低置信像素回填的锚定目录(默认 teammate_best; 自我锚定可指 output_test1_kan_dacs_ms2)")
    ap.add_argument("--clean-mask", action="store_true",
                    help="源域用 masks_clean/ 清洗后的标签(relabel.py 生成)")
    ap.add_argument("--team-fill", action="store_true", default=True,
                    help="低置信像素用队友67.33线预测回填(默认开)")
    ap.add_argument("--no-team-fill", dest="team_fill", action="store_false")
    ap.add_argument("--conflict-ignore", action="store_true",
                    help="68.89规则: 低置信且教师硬标签与锚冲突的像素保持Ignore(不回填), "
                         "只回填教师==锚的一致像素")
    ap.add_argument("--ce-weights", type=str, default=None,
                    help="覆盖 CE_WEIGHTS, 逗号分隔8值(Bg,Building,Road,Water,Barren,Vegetation,Agri,Vehicle)")
    ap.add_argument("--source-loss-weight", type=float, default=0.0,
                    help="★(9/14) source_supervised: 未混合源域监督损失权重 λ。"
                         "0=关闭(默认, 等价旧行为); 队友 69.08 配方用 0.20")
    ap.add_argument("--teacher-hflip", action="store_true",
                    help="★(9/16) 队友 69.08 配方项: 教师打标做水平翻转 TTA"
                         "(原图 + 翻转后平均概率), 提升伪标签质量")
    ap.add_argument("--scale-up", action="store_true",
                    help="★(9/17) 尺度增广不被图边长截断: s>图边长时先放大原图再取整幅, "
                         "保住 0.67~1.33 倍完整尺度分布。crop>683 时才有效(如 crop 1024)")
    ap.add_argument("--boost", type=str, default=None,
                    help="覆盖采样boost, 逗号分隔8值(同顺序); 提高弱类出现频率")
    ap.add_argument("ckpts", nargs="*", help="--agree 模式下的权重列表")
    a = ap.parse_args()
    # ★★修正(9/13): 原实现用 `import dacs_train as _self` 去改 TAU / CONFLICT_IGNORE,
    #   但脚本是以 __main__ 运行的, `import dacs_train` 会**再加载一份模块副本**,
    #   改的是副本的全局 → 真正在跑的 __main__ 里开关**完全没生效**!
    #   铁证: n5b 日志打印 "[tau] 覆盖 TAU=0.93" 紧接着又打印 "[dacs] τ=0.85";
    #         最小实验也复现(import副本0.93 / __main__实际0.85)。
    #   影响: n5(τ0.90) 与 n5b(τ0.93) 其实是重复基线; ci7 的 --conflict-ignore 也没生效。
    #   → **τ 与 conflict-ignore 至今从未被真正测试过**。
    #   修法: 此处就是模块级作用域, 直接赋值即可(__main__ 的命名空间即全局命名空间)。
    if a.tau is not None:
        TAU = a.tau
        print(f"[tau] 覆盖 TAU={TAU} [已生效]", flush=True)
    if a.conflict_ignore:
        CONFLICT_IGNORE = True
        print("[conflict-ignore] 低置信教师-锚冲突像素不回填(保持Ignore), 仅回填一致像素 [已生效]",
              flush=True)
    if a.source_loss_weight:
        SOURCE_W = a.source_loss_weight
        print(f"[source-loss] λ={SOURCE_W} (未混合源域监督支路已启用) [已生效]", flush=True)
    if a.teacher_hflip:
        TEACHER_HFLIP = True
        print("[teacher-hflip] 教师打标启用水平翻转 TTA(原图+翻转平均) [已生效]", flush=True)
    if a.scale_up:
        S.SCALE_UP = True
        print("[scale-up] 尺度增广不被图边长截断(先放大原图再取整幅) [已生效]", flush=True)
    if a.train:
        train(a)
    elif a.agree is not None:
        files = sorted(os.listdir(S.TEST_DIR))[:a.agree]
        for ck in a.ckpts:
            if not ck.endswith(".pth"):
                continue
            m = agree(ck, files)
            print(f"{ck}: 队友一致率 mIoU={m:.4f} (n={a.agree})")
    elif a.eval_ms2:
        run_eval(a, tta=a.tta or "ms2")
    elif a.infer:
        infer(a)
