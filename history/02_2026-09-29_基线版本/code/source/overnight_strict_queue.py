# -*- coding: utf-8 -*-
"""One-shot, single-GPU overnight queue for the sealed historical-scale chain."""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(r"E:\复现")
SOURCE = ROOT / "workspace" / "source"
RUN = ROOT / "runs" / "strict"
LOGS = RUN / "logs"
STATUS_PATH = RUN / "overnight_queue_status.json"
MASTER_LOG = LOGS / "overnight_queue.log"
PREFLIGHT_PATH = RUN / "overnight_historical_scale_preflight.json"
PUBLIC_SHA = "c5d8c5ec47726ea2f4fd2b60954972ffe9d765f2c166812ffc2b1d61eef02cb8"


class QueueStop(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def log(message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with MASTER_LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def ensure_absent(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise QueueStop("refusing to overwrite existing overnight artifacts: " + "; ".join(existing))


def update_status(state: dict) -> None:
    state["updated_at"] = now()
    atomic_json(STATUS_PATH, state)


def verify_preflight(*, include_preserved: bool) -> dict:
    if not PREFLIGHT_PATH.is_file():
        raise QueueStop(f"missing frozen preflight: {PREFLIGHT_PATH}")
    frozen = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    if frozen.get("public_weight_sha256") != PUBLIC_SHA:
        raise QueueStop("preflight public-weight hash declaration mismatch")
    for filename, expected in frozen.get("source_sha256", {}).items():
        actual = sha256(SOURCE / filename)
        if actual.upper() != expected.upper():
            raise QueueStop(f"source changed after preflight: {filename}")
    if include_preserved:
        for filename, expected in frozen.get("preserved_diagnostic_artifacts", {}).items():
            path = RUN / filename
            if not path.is_file() or sha256(path).upper() != expected.upper():
                raise QueueStop(f"preserved diagnostic artifact changed: {filename}")
    return frozen


def record_checkpoint(state: dict, key: str, filename: str, stage: str,
                      seed: int, iteration: int) -> None:
    path = RUN / filename
    if not path.is_file():
        raise QueueStop(f"missing expected checkpoint after {key}: {path}")
    state["stages"][key]["checkpoint"] = {
        "filename": filename,
        "sha256": sha256(path),
        "stage": stage,
        "seed": seed,
        "iteration": iteration
    }
    update_status(state)


def run_job(state: dict, key: str, args: list[str], *, parent: dict | None = None) -> None:
    verify_preflight(include_preserved=False)
    stdout = LOGS / f"overnight_{key}.stdout.log"
    stderr = LOGS / f"overnight_{key}.stderr.log"
    ensure_absent([stdout, stderr])
    item = {
        "status": "running",
        "started_at": now(),
        "command": [sys.executable, *args],
        "stdout": str(stdout),
        "stderr": str(stderr),
        "parent": parent
    }
    state["stages"][key] = item
    state["current_stage"] = key
    update_status(state)
    log(f"START {key}: {' '.join(args)}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        completed = subprocess.run([sys.executable, *args], cwd=SOURCE, env=env,
                                   stdout=out, stderr=err, check=False)
    item["exit_code"] = completed.returncode
    item["finished_at"] = now()
    item["status"] = "complete" if completed.returncode == 0 else "failed"
    update_status(state)
    log(f"END {key}: exit={completed.returncode}")
    if completed.returncode != 0:
        raise QueueStop(f"{key} exited with code {completed.returncode}; see {stderr}")


def run_eval(state: dict, key: str, model: str, checkpoint: str, n: int,
             output: str, stage: str, iteration: int) -> dict:
    run_job(state, key, [
        "strict_eval.py", "--model", model, "--ckpt", checkpoint,
        "--n", str(n), "--output", output,
        "--expected-stage", stage, "--expected-iter", str(iteration),
    ], parent={"checkpoint": checkpoint, "sha256": sha256(RUN / checkpoint)})
    result = json.loads((RUN / output).read_text(encoding="utf-8"))
    state["stages"][key]["evaluation"] = result
    update_status(state)
    return result


def valid_classes(result: dict, baseline: dict | None = None) -> tuple[bool, str]:
    values = result["val_IoU"]
    for name, value in values.items():
        if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
            return False, f"{name} has non-finite/zero IoU"
        if float(value) < 0.05:
            return False, f"{name} IoU below preregistered collapse floor 0.05"
    if baseline is not None:
        for name, value in values.items():
            drop = float(baseline["val_IoU"][name]) - float(value)
            if drop > 0.20:
                return False, f"{name} absolute IoU drop {drop:.4f} exceeds collapse limit 0.20"
    return True, "all class IoUs finite, nonzero, and non-collapsed"


def gate(condition: bool, reason: str) -> None:
    if not condition:
        raise QueueStop(reason)
    log("GATE PASS: " + reason)


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    if STATUS_PATH.exists() or MASTER_LOG.exists():
        raise RuntimeError("overnight queue was already launched; refusing a second launch")
    frozen_preflight = verify_preflight(include_preserved=True)

    artifacts = [
        "ckpt_swin_v4_historical_scale.pth",
        "eval_swin_v4_historical_scale_plain_ema_n100.json",
        "eval_swin_v4_historical_scale_plain_ema_n500.json",
        "ckpt_swin_v4_1_historical_scale.pth",
        "eval_swin_v4_1_historical_scale_plain_ema_n100.json",
        "eval_swin_v4_1_historical_scale_plain_ema_n500.json",
        "ckpt_kan_historical_scale_16k.pth",
        "eval_kan_historical_scale_16k_plain_ema_n500.json",
        "ckpt_kan_historical_scale_pl.pth",
        "ckpt_kan_historical_scale_pl_i1000.pth",
        "ckpt_kan_historical_scale_pl_i2000.pth",
        "ckpt_kan_historical_scale_pl_i3000.pth",
        "ckpt_kan_historical_scale_pl_i4000.pth",
        "eval_kan_historical_scale_pl_i2000_plain_ema_n500.json",
        "ckpt_teacher_v4_seed314159.pth",
        "eval_teacher_v4_seed314159_plain_ema_n500.json",
    ]
    ensure_absent([RUN / name for name in artifacts] + [RUN / "pl_cache_night_kan16k_ms2"])
    started = datetime.now().astimezone()
    # This queue is an overnight run: before noon, "morning" is today; after noon it is tomorrow.
    teacher_deadline = started.replace(hour=5, minute=30, second=0, microsecond=0)
    if started.hour >= 12:
        teacher_deadline += timedelta(days=1)
    state = {
        "status": "running",
        "started_at": started.isoformat(),
        "current_stage": "preflight",
        "policy": {
            "single_gpu_serial": True,
            "no_autodl": True,
            "no_test2": True,
            "no_output": True,
            "stop_on_failure_or_gate": True
        },
        "public_weight_sha256": PUBLIC_SHA,
        "preflight": str(PREFLIGHT_PATH),
        "source_sha256": frozen_preflight["source_sha256"],
        "teacher_start_deadline": teacher_deadline.isoformat(),
        "stages": {}
    }
    update_status(state)
    log("OVERNIGHT QUEUE START")

    try:
        # 1) Historical-scale v4 from the sole public academic weight.
        v4_ckpt = "ckpt_swin_v4_historical_scale.pth"
        run_job(state, "v4_train", [
            "seg_swin.py", "--train", "--crop", "512", "--target-iters", "12000",
            "--max-seconds", "86400", "--seed", "42", "--stage", "night_v4_historical_scale",
            "--ckpt-name", v4_ckpt,
        ], parent={"public_weight_sha256": PUBLIC_SHA})
        record_checkpoint(state, "v4_train", v4_ckpt,
                          "night_v4_historical_scale", 42, 12000)
        v4_100 = run_eval(state, "v4_eval_n100", "swin", v4_ckpt, 100,
                          "eval_swin_v4_historical_scale_plain_ema_n100.json",
                          "night_v4_historical_scale", 12000)
        v4_500 = run_eval(state, "v4_eval_n500", "swin", v4_ckpt, 500,
                          "eval_swin_v4_historical_scale_plain_ema_n500.json",
                          "night_v4_historical_scale", 12000)
        gate(v4_100["val_mIoU"] >= 0.735,
             f"v4 n100 mIoU {v4_100['val_mIoU']:.4f} >= 0.735")
        valid, reason = valid_classes(v4_500)
        gate(valid, "v4: " + reason)

        # 2) Historical-scale v4.1 from this queue's v4 EMA only.
        v41_ckpt = "ckpt_swin_v4_1_historical_scale.pth"
        run_job(state, "v4_1_train", [
            "seg_swin.py", "--train", "--strict-v4-1", "--crop", "640",
            "--target-iters", "4000", "--max-seconds", "86400", "--seed", "42",
            "--stage", "night_v4_1_historical_scale", "--ckpt-name", v41_ckpt,
            "--warm-from", v4_ckpt, "--strict-parent-stage", "night_v4_historical_scale",
        ], parent={"checkpoint": v4_ckpt, "sha256": sha256(RUN / v4_ckpt)})
        record_checkpoint(state, "v4_1_train", v41_ckpt,
                          "night_v4_1_historical_scale", 42, 4000)
        v41_100 = run_eval(state, "v4_1_eval_n100", "swin", v41_ckpt, 100,
                           "eval_swin_v4_1_historical_scale_plain_ema_n100.json",
                           "night_v4_1_historical_scale", 4000)
        v41_500 = run_eval(state, "v4_1_eval_n500", "swin", v41_ckpt, 500,
                           "eval_swin_v4_1_historical_scale_plain_ema_n500.json",
                           "night_v4_1_historical_scale", 4000)
        gate(v41_500["val_mIoU"] >= v4_500["val_mIoU"] - 0.005,
             f"v4.1 n500 {v41_500['val_mIoU']:.4f} >= v4 n500 {v4_500['val_mIoU']:.4f} - 0.005")
        valid, reason = valid_classes(v41_500, v4_500)
        gate(valid, "v4.1: " + reason)

        # 3) KAN 16k with only the new v4.1 encoder; DFR/decoder initialize fresh.
        kan_ckpt = "ckpt_kan_historical_scale_16k.pth"
        run_job(state, "kan16k_train", [
            "kan_swin.py", "--train", "--crop", "512", "--batch-size", "2",
            "--num-workers", "3", "--target-iters", "16000", "--max-seconds", "86400",
            "--seed", "42", "--stage", "night_kan_historical_scale_16k",
            "--ckpt-name", kan_ckpt, "--encoder-warm", v41_ckpt,
            "--encoder-parent-stage", "night_v4_1_historical_scale",
        ], parent={"checkpoint": v41_ckpt, "sha256": sha256(RUN / v41_ckpt)})
        record_checkpoint(state, "kan16k_train", kan_ckpt,
                          "night_kan_historical_scale_16k", 42, 16000)
        kan_500 = run_eval(state, "kan16k_eval_n500", "kan", kan_ckpt, 500,
                           "eval_kan_historical_scale_16k_plain_ema_n500.json",
                           "night_kan_historical_scale_16k", 16000)
        gate(kan_500["val_mIoU"] >= 0.745,
             f"KAN16k n500 mIoU {kan_500['val_mIoU']:.4f} >= 0.745")
        valid, reason = valid_classes(kan_500)
        gate(valid, "KAN16k: " + reason)

        # 4) All-500, all-pixel test1 ms2 pseudo-labels and exact coverage.
        cache_name = "pl_cache_night_kan16k_ms2"
        run_job(state, "pl_generate", ["pl_gen.py", "--ckpt", kan_ckpt, "--cache", cache_name],
                parent={"checkpoint": kan_ckpt, "sha256": sha256(RUN / kan_ckpt)})
        coverage_report = json.loads((RUN / cache_name / "_CONFIDENCE_REPORT.json").read_text(encoding="utf-8"))
        coverage = float(coverage_report["coverage"]["0.85"])
        state["stages"]["pl_generate"]["coverage_report"] = coverage_report
        update_status(state)
        gate(coverage >= 0.80, f"tau=0.85 all-500 coverage {coverage:.4%} >= 80%")

        # 5) Fixed PL recipe; evaluate i2000 only for selection gate.
        pl_ckpt = "ckpt_kan_historical_scale_pl.pth"
        run_job(state, "pl_train", [
            "kan_swin.py", "--train", "--pl", "--crop", "640", "--batch-size", "2",
            "--num-workers", "3", "--target-iters", "4000", "--max-seconds", "86400",
            "--seed", "42", "--stage", "night_kan_historical_scale_pl",
            "--ckpt-name", pl_ckpt, "--warm-base", kan_ckpt,
            "--warm-parent-stage", "night_kan_historical_scale_16k", "--warm-parent-iter", "16000",
            "--pl-cache", cache_name, "--pl-tag", "night_historical_scale",
            "--pl-tau", "0.85", "--restart-lr", "0.25",
        ], parent={"checkpoint": kan_ckpt, "sha256": sha256(RUN / kan_ckpt),
                   "pseudo_label_cache": cache_name})
        record_checkpoint(state, "pl_train", pl_ckpt,
                          "night_kan_historical_scale_pl", 42, 4000)
        snapshot_names = [f"ckpt_kan_historical_scale_pl_i{iteration}.pth"
                          for iteration in (1000, 2000, 3000, 4000)]
        missing_snapshots = [name for name in snapshot_names if not (RUN / name).is_file()]
        if missing_snapshots:
            raise QueueStop("PL snapshots missing: " + ", ".join(missing_snapshots))
        state["stages"]["pl_train"]["snapshots"] = [
            {"filename": name, "sha256": sha256(RUN / name)} for name in snapshot_names
        ]
        update_status(state)
        pl_i2000 = "ckpt_kan_historical_scale_pl_i2000.pth"
        pl_500 = run_eval(state, "pl_i2000_eval_n500", "kan", pl_i2000, 500,
                          "eval_kan_historical_scale_pl_i2000_plain_ema_n500.json",
                          "night_kan_historical_scale_pl", 2000)
        gate(pl_500["val_mIoU"] >= kan_500["val_mIoU"] - 0.015,
             f"PL i2000 n500 {pl_500['val_mIoU']:.4f} >= KAN warm {kan_500['val_mIoU']:.4f} - 0.015")
        valid, reason = valid_classes(pl_500, kan_500)
        gate(valid, "PL i2000: " + reason)

        # Optional independent teacher: public weight only, never a main-chain parent.
        if datetime.now().astimezone() < teacher_deadline:
            teacher_ckpt = "ckpt_teacher_v4_seed314159.pth"
            run_job(state, "teacher_v4_train", [
                "seg_swin.py", "--train", "--crop", "512", "--target-iters", "12000",
                "--max-seconds", "86400", "--seed", "314159",
                "--stage", "independent_teacher_v4_seed314159", "--ckpt-name", teacher_ckpt,
            ], parent={"public_weight_sha256": PUBLIC_SHA, "independent_of_main_chain": True})
            record_checkpoint(state, "teacher_v4_train", teacher_ckpt,
                              "independent_teacher_v4_seed314159", 314159, 12000)
            teacher_500 = run_eval(state, "teacher_v4_eval_n500", "swin", teacher_ckpt, 500,
                                   "eval_teacher_v4_seed314159_plain_ema_n500.json",
                                   "independent_teacher_v4_seed314159", 12000)
            gate(teacher_500["val_mIoU"] >= 0.740,
                 f"independent teacher n500 {teacher_500['val_mIoU']:.4f} >= 0.740")
            valid, reason = valid_classes(teacher_500)
            gate(valid, "independent teacher: " + reason)
            state["teacher_anchor_generation"] = "not_started_by_policy"
        else:
            state["teacher_stage"] = "skipped_after_05:30_deadline"

        state["status"] = "complete"
        state["current_stage"] = None
        state["finished_at"] = now()
        update_status(state)
        log("OVERNIGHT QUEUE COMPLETE")
    except QueueStop as error:
        state["status"] = "stopped"
        state["stop_reason"] = str(error)
        state["finished_at"] = now()
        update_status(state)
        log("OVERNIGHT QUEUE STOPPED: " + str(error))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
