# -*- coding: utf-8 -*-
"""预算扫描：从 r3 继续训练，在 3200/6400/9600/12800 处各评 val500，取最优点推理。

目的：不再假设"更多一定更好"，一次跑完看到曲线拐点。
"""
import json, os, subprocess, time
from pathlib import Path

PY = "/root/miniconda3/bin/python"
S = Path("/root/autodl-tmp/aic_semi/p0_scripts")
ROOT = Path("/root/autodl-tmp/aic_semi")
OUT = ROOT / "p0"
STATUS = OUT / "SWEEP_STATUS.json"
SNAPS = [3200, 6400, 9600, 12800]
FLOOR_VAL = 0.764587      # r3 的 val500
FLOOR_LABEL = 0.68        # 漂移体温计下限


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def schedule_shutdown(minutes=15, why=""):
    """留出回传窗口后自动关机；与其他任务并行时由外部 watchdog 统一关机。"""
    if os.environ.get("NO_SHUTDOWN") == "1":
        log("NO_SHUTDOWN=1，跳过关机（交由 watchdog 统一处理）")
        return
    try:
        r = subprocess.run(["shutdown", "-h", f"+{minutes}"], capture_output=True, text=True)
        if r.returncode == 0:
            log(f"已计划 {minutes} 分钟后关机（{why}）——请在此之前回传产物")
        else:
            log(f"关机命令返回 {r.returncode}: {(r.stderr or '').strip()[:120]}")
    except Exception as ex:
        log(f"关机命令异常: {type(ex).__name__}")


def st(**kw):
    cur = json.load(open(STATUS)) if STATUS.exists() else {}
    cur.update(kw); cur["ts"] = time.strftime("%H:%M:%S")
    json.dump(cur, open(STATUS, "w"), ensure_ascii=False, indent=1)


def main():
    t0 = time.time()
    log(f"起点 P0_r3：val500 {FLOOR_VAL:.6f}")
    st(stage="train", snapshots=SNAPS)

    cmd = [PY, str(S / "train_decoder.py"), "--tag", "sw",
           "--parent", str(OUT / "r3.pth"),
           "--steps", "12800", "--batch", "4", "--kl", "1.0",
           "--snapshots", ",".join(map(str, SNAPS))]
    vlog = open(OUT / "sw.log", "w")
    subprocess.run(cmd, cwd=str(S), stdout=vlog, stderr=subprocess.STDOUT)
    vlog.close()

    # 从训练日志里挑最优点
    pts = []
    for line in open(OUT / "sw_train.jsonl", encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "snapshot" in d:
            pts.append(d)
    if not pts:
        st(stage="error", msg="没有快照")
        return
    for d in pts:
        log(f"  快照 {d['snapshot']:>5}  val500 {d['val500']:.6f}  标注 {d['label']:.4f}")

    ok = [d for d in pts if d["val500"] > FLOOR_VAL and d["label"] >= FLOOR_LABEL]
    if not ok:
        best = max(pts, key=lambda d: d["val500"])
        log(f"没有快照同时满足双闸门（最好 {best['snapshot']} val500 {best['val500']:.6f}）；"
            f"若 val500 未超过 r3 则不推理")
        if best["val500"] <= FLOOR_VAL:
            st(stage="skip", points=pts)
            return
        ok = [best]
    best = max(ok, key=lambda d: d["val500"])
    log(f"选中最优快照 step {best['snapshot']}  val500 {best['val500']:.6f}"
        f"（比 r3 {best['val500']-FLOOR_VAL:+.6f}）")
    st(stage="selected", best=best, points=pts)

    ck = OUT / f"sw_s{best['snapshot']}.pth"
    cmd2 = [PY, str(S / "infer_p0.py"), "--ckpt", str(ck), "--tag", f"P0_sw{best['snapshot']}"]
    ilog = open(OUT / "infer_sw.log", "w")
    subprocess.run(cmd2, cwd=str(S), stdout=ilog, stderr=subprocess.STDOUT)
    ilog.close()
    st(stage="done", zip=str(ROOT / f"output_P0_sw{best['snapshot']}.zip"),
       minutes=round((time.time() - t0) / 60, 1))


if __name__ == "__main__":
    why = "正常结束"
    try:
        main()
    except Exception as ex:
        why = f"异常 {type(ex).__name__}"
        st(stage="crash", err=repr(ex)[:500])
        raise
    finally:
        try:
            (OUT / "SWEEP_DONE").write_text(f"{time.strftime('%H:%M:%S')} {why}\n", encoding="utf-8")
        except Exception:
            pass
        schedule_shutdown(15, why)
