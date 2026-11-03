# -*- coding: utf-8 -*-
"""P0 微调模型的输出推理：8 路 TTA → 3594 张 PNG → zip，并报告类别分布。"""
import argparse, json, os, sys, time, zipfile
from pathlib import Path

ROOT = Path("/root/autodl-tmp/aic_semi")
os.environ.setdefault("_SKIP_VERIFY", "1")
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
from PIL import Image
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "fp", ROOT / "code" / "kan_pipeline" / "final_pipeline.py")
FP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FP)
import strict_env as P
import seg_swin as S
import kan_swin as K

IMG = ROOT / "official_data" / "test_3" / "images"
CLS = ["背景", "建筑", "道路", "水体", "荒地", "森林", "农田", "车辆"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="p0")
    ap.add_argument("--state", default="ema", choices=["ema", "model"])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    payload = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = K.DeepKANSegSwin(K.NUM_CLASSES)
    model.load_state_dict(payload[a.state], strict=True)
    model = model.to(dev).eval().requires_grad_(False)
    log(f"载入 {a.ckpt} state={a.state} iter={payload.get('iter')}")

    out = ROOT / "experiments" / a.tag / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p.name for p in IMG.glob("*.png"))
    cnt = np.zeros(9, np.int64)
    t0 = time.time()
    for i, fn in enumerate(files, 1):
        img = np.asarray(Image.open(IMG / fn).convert("RGB"))
        prob = FP.predict_probability_average(model, img, S, dev)
        pr = prob.argmax(0).numpy().astype(np.uint8)
        if pr.min() < 1 or pr.max() > 8:
            raise RuntimeError(f"标签越界 {fn}")
        Image.fromarray(pr).save(out / fn)
        cnt += np.bincount(pr.ravel(), minlength=9)
        if i % 200 == 0:
            el = (time.time() - t0) / 60
            log(f"  {i}/{len(files)}  {el:.0f} min  剩约 {(len(files)-i)*el/i:.0f} min")
    zp = ROOT / f"predictions_{a.tag}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        for fn in files:
            z.write(out / fn, fn)
    v = cnt[1:] / cnt[1:].sum() * 100
    train = [28.14, 18.31, 8.92, 6.25, 1.83, 28.65, 7.12, 0.78]
    log("类别分布：" + "  ".join(f"{CLS[i]} {v[i]:.2f}" for i in range(8)))
    log(f"L1 先验距离 {np.abs(v-np.array(train)).sum():.2f}")
    json.dump({"tag": a.tag, "dist": v.tolist(), "zip": str(zp),
               "size_mb": round(zp.stat().st_size / 1e6, 1)},
              open(ROOT / "p0" / f"{a.tag}_dist.json", "w"), indent=1)
    log(f"完成 {zp}")


if __name__ == "__main__":
    main()
