# 通用 per-class IoU 评估 (无 shell 管道依赖, 输出直写 stdout)
# 用法: python eval_perclass.py <ckpt> [n_val=120] [tta=ms]
import os, sys, json
import numpy as np, torch
from PIL import Image
import kan_swin as K, seg_swin as S

ckpt = sys.argv[1] if len(sys.argv) > 1 else "ckpt_kan_dacs_a69s7.pth"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 120
tta = sys.argv[3] if len(sys.argv) > 3 else "ms"

model, iters = K.load_trained(path=ckpt)
model.eval()
print(f"[load] {os.path.basename(ckpt)} EMA iter={iters} tta={tta} n={n}", flush=True)

val_ids = json.load(open(os.path.join(K.ROOT, "split.json")))["val_ids"][:n]
bias = K.load_bias()
conf = np.zeros((K.NUM_CLASSES, K.NUM_CLASSES), np.int64)
for i, k in enumerate(val_ids):
    img = np.array(Image.open(os.path.join(S.IMG_DIR, f"{k:04d}.png")).convert("RGB"))
    msk = np.array(Image.open(os.path.join(S.MSK_DIR, f"{k:04d}.png")))
    pred = S.predict_logits(model, img, tta=tta, bias=bias).argmax(0).numpy().astype(np.uint8)
    t = msk.ravel(); p = pred.ravel(); m = t != K.IGNORE
    conf += np.bincount(K.NUM_CLASSES * t[m] + p[m],
                        minlength=K.NUM_CLASSES ** 2).reshape(K.NUM_CLASSES, K.NUM_CLASSES)
    if (i + 1) % 40 == 0:
        torch.cuda.empty_cache(); print(f"  {i+1}/{n}", flush=True)

names = ["", "Background", "Building", "Road", "Water", "Barren", "Vegetation", "Agri", "Vehicle"]
inter = np.diag(conf).astype(np.float64)
union = conf.sum(1) + conf.sum(0) - inter
print("\n=== per-class IoU ===", flush=True)
ious = []
for c in range(1, 9):
    iou = inter[c] / union[c] if union[c] > 0 else float("nan")
    ious.append(iou)
    frac = conf[c].sum() / conf[1:].sum() * 100
    print(f"  {names[c]:<12} IoU={iou:.4f} 真值占比={frac:5.2f}% 贡献={iou*frac/100:.4f}", flush=True)
print(f"  mIoU = {np.nanmean(ious):.4f}", flush=True)

print("\n=== 漏检结构 (真值类主要误判去向, >4%) ===", flush=True)
for c in range(1, 9):
    tot = conf[c].sum()
    if tot == 0: continue
    row = conf[c].copy(); row[c] = 0
    for t in np.argsort(row)[::-1][:3]:
        if row[t] / tot > 0.04:
            print(f"  {names[c]} 漏判为 {names[t]}: {row[t]/tot*100:.1f}%", flush=True)

print("\n=== 当前类权重配置 ===", flush=True)
boost = [1.2, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.5]
for nm, w, b in zip(names[1:], S.CE_WEIGHTS, boost):
    print(f"  {nm:<12} CE={w:.2f} boost={b:.2f}", flush=True)
print("DONE", flush=True)
