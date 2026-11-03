# -*- coding: utf-8 -*-
"""
第三阶段多轮实验总控：BN 统计一次收集 → 多 alpha 混合 → 逐轮推理 → 对比选优 → 打包

设计依据（来自数据挖掘）：
  · test_3 = test_1(500) + test_2(1300) + 新图(1794)，新图与 test_2 同域（亮度 94.3 vs 94.5）
  · train 比测试集亮约 6 个灰度级 ⇒ BN 校准是唯一验证有效的推理期杠杆
  · 第二阶段最终版是 alpha=0.75 / BN 池=test_2(1300)；现在池可以扩到 test_3(3594)

关键优化：
  BN 统计只收集一次（3594 张，约 15 min），之后每个 alpha 只是对
  target_mean/var 做一次加权混合（秒级），不必为每个 alpha 重跑统计。

用法：
  python run_experiments.py --stage all      # 全流程
  python run_experiments.py --stage smoke    # 每轮只跑 4 张，验证管线
  python run_experiments.py --stage collect  # 只收 BN 统计
"""
import os, sys, json, time, shutil, argparse, zipfile, traceback
from pathlib import Path

ROOT = Path("/root/autodl-tmp/aic_semi")
os.environ.setdefault("_SKIP_VERIFY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
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

CKPT = ROOT / "final.pth"
EXP_DIR = ROOT / "experiments"
TEST_IMAGES = ROOT / "official_data" / "test_3" / "images"
N_TEST = 3594

# (轮次名, BN 池来源, alpha, 说明)
EXPS = [
    ("R0_base_a075_p1300", "shipped", 0.75, "基线：沿用第二阶段权重（池=test2 1300）"),
    ("R1_a075_p3594",      "test_3",  0.75, "BN 池扩到 test_3 全量 3594"),
    ("R2_a055_p3594",      "test_3",  0.55, "alpha 下探"),
    ("R3_a090_p3594",      "test_3",  0.90, "alpha 上探"),
    ("R4_a100_p3594",      "test_3",  1.00, "纯目标域统计"),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- BN 统计
def collect_moments(model, files, device):
    """收集目标域 BN 统计（不改动模型参数）"""
    bns = {n: m for n, m in model.named_modules()
           if isinstance(m, torch.nn.BatchNorm2d)}
    log(f"  BN 层数 {len(bns)}")
    buckets = {n: {"count": 0} for n in bns}
    hooks = [m.register_forward_pre_hook(FP.moments_hook(buckets[n]))
             for n, m in bns.items()]
    try:
        with torch.inference_mode():
            for i, path in enumerate(files, 1):
                image = np.asarray(Image.open(path).convert("RGB"))
                tensor = S.norm(image).unsqueeze(0).to(device)
                with torch.autocast("cuda", torch.float16,
                                    enabled=(device == "cuda")):
                    out = model(tensor)
                if not torch.isfinite(out).all():
                    raise RuntimeError(f"non-finite: {path.name}")
                if i % 200 == 0:
                    log(f"    BN 统计 {i}/{len(files)}")
                del tensor, out
    finally:
        for h in hooks:
            h.remove()
    out = {}
    for n, b in buckets.items():
        if b["count"] <= 1:
            raise RuntimeError(f"BN {n} 统计不足")
        mean = b["mean"].float()
        var = (b["m2"] / (b["count"] - 1)).float()
        out[n] = {"target_mean": mean, "target_var": var,
                  "count": int(b["count"])}
    return out


def blend(orig_state, moments, alpha):
    """用 moments 与原始统计按 alpha 混合，返回新的 state_dict（只改 82 个 buffer）"""
    new = {k: (v.clone() if torch.is_tensor(v) else v)
           for k, v in orig_state.items()}
    for name, m in moments.items():
        sm = orig_state[name + ".running_mean"].float()
        sv = orig_state[name + ".running_var"].float()
        tm, tv = m["target_mean"], m["target_var"]
        mean = (1 - alpha) * sm + alpha * tm
        var = (1 - alpha) * sv + alpha * tv + alpha * (1 - alpha) * (tm - sm) ** 2
        if not torch.isfinite(mean).all() or (var <= 0).any():
            raise RuntimeError(f"混合后统计非法: {name}")
        new[name + ".running_mean"] = mean
        new[name + ".running_var"] = var
    return new


# ---------------------------------------------------------------- 推理
def infer_all(model, files, out_dir, device, limit=None):
    out_dir = Path(out_dir)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    sel = files[:limit] if limit else files
    accs = []
    t0 = time.time()
    for i, path in enumerate(sel, 1):
        image = np.asarray(Image.open(path).convert("RGB"))
        prob = FP.predict_probability_average(model, image, S, device)
        pred = prob.argmax(0).numpy().astype(np.uint8)
        if pred.min() < 1 or pred.max() > 8:
            raise RuntimeError(f"标签越界: {path.name}")
        Image.fromarray(pred).save(pred_dir / path.name)
        if i % 200 == 0:
            el = (time.time() - t0) / 60
            log(f"    推理 {i}/{len(sel)}  已用 {el:.1f} min  预计 {(len(sel)-i)*el/i/60:.1f} h")
    return pred_dir, time.time() - t0


def class_dist(pred_dir):
    """预测的类别分布（抽样统计，省时间）"""
    files = sorted(Path(pred_dir).glob("*.png"))
    step = max(1, len(files) // 200)
    cnt = np.zeros(9, dtype=np.int64)
    for f in files[::step]:
        a = np.asarray(Image.open(f))
        cnt += np.bincount(a.ravel(), minlength=9)
    return (cnt / cnt.sum()).tolist()


def diff_rate(d1, d2):
    """两轮预测的像素差异率（抽样）"""
    f1 = sorted(Path(d1).glob("*.png"))
    step = max(1, len(f1) // 120)
    tot = same = 0
    for f in f1[::step]:
        g = Path(d2) / f.name
        if not g.exists():
            continue
        a = np.asarray(Image.open(f)); b = np.asarray(Image.open(g))
        tot += a.size; same += int((a == b).sum())
    return 1 - same / tot if tot else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["collect", "smoke", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--budget-h", type=float, default=8.0,
                    help="总时长上限（小时），超预算则跳过剩余轮次直接打包")
    args = ap.parse_args()

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(TEST_IMAGES.glob("*.png"))
    log(f"测试图 {len(files)} 张")
    if len(files) != N_TEST:
        raise RuntimeError(f"期望 {N_TEST} 张，实得 {len(files)}")

    device = K.DEVICE if hasattr(K, "DEVICE") else (
        "cuda" if torch.cuda.is_available() else "cpu")
    log(f"设备 {device}")

    # ---- 载入最终权重
    log("载入 final.pth")
    payload = torch.load(CKPT, map_location="cpu", weights_only=False)
    orig_state = payload["ema"]
    model = K.DeepKANSegSwin(K.NUM_CLASSES)
    model.load_state_dict(orig_state, strict=True)
    model = model.to(device).eval().requires_grad_(False)
    log(f"  参数量 {sum(p.numel() for p in model.parameters()):,}")

    moments_path = EXP_DIR / "bn_moments_test3.json"
    moments = None

    if args.stage in ("collect", "all"):
        if moments_path.exists():
            log("复用已有 BN 统计")
            raw = json.load(open(moments_path, encoding="utf-8"))
            moments = {k: {"target_mean": torch.tensor(v["target_mean"]),
                           "target_var": torch.tensor(v["target_var"]),
                           "count": v["count"]} for k, v in raw.items()}
        else:
            log("收集 test_3(3594) 的 BN 统计 ...")
            t0 = time.time()
            moments = collect_moments(model, files, device)
            log(f"  完成，用时 {(time.time()-t0)/60:.1f} min")
            json.dump({k: {"target_mean": v["target_mean"].tolist(),
                           "target_var": v["target_var"].tolist(),
                           "count": v["count"]} for k, v in moments.items()},
                      open(moments_path, "w", encoding="utf-8"))

    if args.stage == "collect":
        log("仅收集阶段，结束")
        return

    limit = args.limit or (4 if args.stage == "smoke" else None)
    t_start = time.time()

    # ---- 逐轮实验
    summary = []
    for name, pool, alpha, desc in EXPS:
        log(f"===== {name}  (池={pool}, alpha={alpha}) =====")
        out_dir = EXP_DIR / name
        if (out_dir / "DONE").exists():
            log("  已完成，跳过")
            summary.append(json.load(open(out_dir / "report.json", encoding="utf-8")))
            continue
        used = (time.time() - t_start) / 3600
        if limit is None and used > args.budget_h:
            log(f"  超预算（已用 {used:.1f}h > {args.budget_h}h），跳过")
            continue
        try:
            t0 = time.time()
            if pool == "shipped":
                state = orig_state
            else:
                if moments is None:
                    raise RuntimeError("缺少 BN 统计")
                state = blend(orig_state, moments, alpha)
            model.load_state_dict(state, strict=True)
            pred_dir, sec = infer_all(model, files, out_dir, device, limit)
            rep = {"name": name, "pool": pool, "alpha": alpha, "desc": desc,
                   "n": len(list(pred_dir.glob('*.png'))),
                   "seconds": round(sec, 1),
                   "class_dist": class_dist(pred_dir)}
            json.dump(rep, open(out_dir / "report.json", "w"), indent=1)
            (out_dir / "DONE").touch()
            summary.append(rep)
            log(f"  完成 {rep['n']} 张，{sec/60:.1f} min")
        except Exception:
            log(f"  ✗ 失败：{traceback.format_exc()[-400:]}")

    # ---- 对比
    log("===== 轮次对比 =====")
    base = EXP_DIR / EXPS[0][0] / "predictions"
    for r in summary:
        d = EXP_DIR / r["name"] / "predictions"
        dr = diff_rate(base, d) if d.exists() and base.exists() and r["name"] != EXPS[0][0] else 0.0
        r["diff_vs_base"] = round(dr, 5) if dr == dr else None
        dist = " ".join(f"{x:.3f}" for x in r["class_dist"][1:])
        log(f"  {r['name']:<22} n={r['n']:<5} {r['seconds']/60:>6.1f}min  "
            f"差异={r.get('diff_vs_base')}  分布[{dist}]")
    json.dump(summary, open(EXP_DIR / "SUMMARY.json", "w"), ensure_ascii=False, indent=1)

    # ---- 打包（选 top2：基线 + 最优候选）
    if limit is None:
        log("===== 打包 =====")
        picks = [EXPS[0][0]]
        cands = [r for r in summary if r["name"] != EXPS[0][0] and r["n"] == N_TEST]
        if cands:
            # 选与基线差异适中、且已完成全量的第一个候选（R1 优先）
            order = {n: i for i, (n, *_rest) in enumerate(EXPS)}
            cands.sort(key=lambda r: order[r["name"]])
            picks.append(cands[0]["name"])
        for name in picks:
            d = EXP_DIR / name / "predictions"
            zp = EXP_DIR / f"output_{name}.zip"
            with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
                for f in sorted(d.glob("*.png")):
                    z.write(f, arcname=f.name)
            log(f"  {zp.name}  {len(list(d.glob('*.png')))} 张  "
                f"{zp.stat().st_size/1e6:.1f} MB")
        log("打包完成（top2：基线 + 首个候选）")
    log("ALL DONE")


if __name__ == "__main__":
    main()
