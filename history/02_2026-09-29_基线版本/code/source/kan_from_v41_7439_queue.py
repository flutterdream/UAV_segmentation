# -*- coding: utf-8 -*-
"""One-shot local queue: selected v4.1 EMA -> KAN16k -> coverage -> fixed PL."""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(r"E:\复现")
SOURCE = ROOT / "workspace" / "source"
RUN = ROOT / "runs" / "strict"
LOGS = RUN / "logs"
STATUS_PATH = RUN / "kan_from_v4_1_7439_queue_status.json"
MASTER_LOG = LOGS / "kan_from_v4_1_7439_queue.log"
PREFLIGHT_PATH = RUN / "kan_from_v4_1_7439_preflight.json"
PARENT = "ckpt_swin_upernet_v4_1_strict.pth"
PARENT_SHA = "abface51bfd457617fadc3e65607eef80e3cb2c3dde730b3c9876c1140a887c4"


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


def update_status(state: dict) -> None:
    state["updated_at"] = now()
    atomic_json(STATUS_PATH, state)


def verify_preflight(*, include_preserved: bool) -> dict:
    if not PREFLIGHT_PATH.is_file():
        raise QueueStop(f"missing frozen preflight: {PREFLIGHT_PATH}")
    frozen = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    for filename, expected in frozen["source_sha256"].items():
        if sha256(SOURCE / filename).lower() != expected.lower():
            raise QueueStop(f"source changed after preflight: {filename}")
    if sha256(RUN / PARENT).lower() != PARENT_SHA:
        raise QueueStop("selected v4.1 parent hash changed")
    if include_preserved:
        for filename, expected in frozen["preserved_artifacts"].items():
            path = RUN / filename
            if not path.is_file() or sha256(path).lower() != expected.lower():
                raise QueueStop(f"preserved artifact changed: {filename}")
    return frozen


def ensure_absent(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise QueueStop("refusing to overwrite: " + "; ".join(existing))


def run_job(state: dict, key: str, args: list[str], parent: dict) -> None:
    verify_preflight(include_preserved=False)
    stdout = LOGS / f"kan7439_{key}.stdout.log"
    stderr = LOGS / f"kan7439_{key}.stderr.log"
    ensure_absent([stdout, stderr])
    item = {
        "status": "running", "started_at": now(),
        "command": [sys.executable, *args],
        "stdout": str(stdout), "stderr": str(stderr), "parent": parent
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


def record_checkpoint(state: dict, key: str, filename: str,
                      stage: str, seed: int, iteration: int) -> None:
    path = RUN / filename
    if not path.is_file():
        raise QueueStop(f"missing expected checkpoint: {path}")
    state["stages"][key]["checkpoint"] = {
        "filename": filename, "sha256": sha256(path), "stage": stage,
        "seed": seed, "iteration": iteration
    }
    update_status(state)


def run_eval(state: dict, key: str, checkpoint: str, output: str,
             stage: str, iteration: int) -> dict:
    parent = {"checkpoint": checkpoint, "sha256": sha256(RUN / checkpoint)}
    run_job(state, key, [
        "strict_eval.py", "--model", "kan", "--ckpt", checkpoint,
        "--n", "500", "--output", output,
        "--expected-stage", stage, "--expected-iter", str(iteration),
    ], parent)
    result = json.loads((RUN / output).read_text(encoding="utf-8"))
    state["stages"][key]["evaluation"] = result
    update_status(state)
    return result


def valid_classes(result: dict, baseline: dict | None = None) -> tuple[bool, str]:
    for name, value in result["val_IoU"].items():
        if value is None or not math.isfinite(float(value)) or float(value) <= 0:
            return False, f"{name} has non-finite/zero IoU"
        if float(value) < 0.05:
            return False, f"{name} IoU {value} is below collapse floor 0.05"
        if baseline is not None:
            drop = float(baseline["val_IoU"][name]) - float(value)
            if drop > 0.20:
                return False, f"{name} IoU drop {drop:.4f} exceeds 0.20"
    return True, "all class IoUs finite, nonzero, and non-collapsed"


def require(condition: bool, failure_reason: str, pass_message: str) -> None:
    if not condition:
        raise QueueStop(failure_reason)
    log("GATE PASS: " + pass_message)


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    if STATUS_PATH.exists() or MASTER_LOG.exists():
        raise RuntimeError("KAN 0.7439 queue was already launched; refusing a second launch")
    frozen = verify_preflight(include_preserved=True)

    kan_ckpt = "ckpt_kan_from_v4_1_7439_16k.pth"
    kan_eval = "eval_kan_from_v4_1_7439_16k_plain_ema_n500.json"
    cache_name = "pl_cache_kan_from_v4_1_7439_ms2"
    pl_ckpt = "ckpt_kan_from_v4_1_7439_pl.pth"
    snapshots = [f"ckpt_kan_from_v4_1_7439_pl_i{i}.pth" for i in (1000, 2000, 3000, 4000)]
    pl_eval = "eval_kan_from_v4_1_7439_pl_i2000_plain_ema_n500.json"
    ensure_absent([RUN / name for name in [kan_ckpt, kan_eval, pl_ckpt, pl_eval, *snapshots]]
                  + [RUN / cache_name])

    state = {
        "status": "running", "started_at": now(), "current_stage": "preflight",
        "selected_parent": {"filename": PARENT, "sha256": PARENT_SHA,
                            "plain_ema_n500_mIoU": 0.7439,
                            "stage": "main_v4_1_strict", "iter": 4000, "seed": 42},
        "source_sha256": frozen["source_sha256"],
        "policy": {"single_gpu_serial": True, "no_v4_retrain": True,
                   "no_autodl": True, "no_test2": True, "no_output": True,
                   "stop_on_failure_or_gate": True},
        "stages": {}
    }
    update_status(state)
    log("KAN FROM V4.1 0.7439 QUEUE START")

    try:
        run_job(state, "kan16k_train", [
            "kan_swin.py", "--train", "--crop", "512", "--batch-size", "2",
            "--num-workers", "3", "--target-iters", "16000", "--max-seconds", "86400",
            "--seed", "42", "--stage", "kan_from_v4_1_7439_16k",
            "--ckpt-name", kan_ckpt, "--encoder-warm", PARENT,
            "--encoder-parent-stage", "main_v4_1_strict",
        ], {"checkpoint": PARENT, "sha256": PARENT_SHA, "ema_only_encoder": True})
        record_checkpoint(state, "kan16k_train", kan_ckpt,
                          "kan_from_v4_1_7439_16k", 42, 16000)

        kan_500 = run_eval(state, "kan16k_eval_n500", kan_ckpt, kan_eval,
                           "kan_from_v4_1_7439_16k", 16000)
        require(kan_500["val_mIoU"] >= 0.745,
                f"KAN16k n500 mIoU {kan_500['val_mIoU']:.4f} is below 0.745; stop before pseudo-labeling",
                f"KAN16k n500 mIoU {kan_500['val_mIoU']:.4f} >= 0.745")
        valid, reason = valid_classes(kan_500)
        require(valid, "KAN16k class-collapse gate failed: " + reason, "KAN16k: " + reason)

        run_job(state, "pl_generate", [
            "pl_gen.py", "--ckpt", kan_ckpt, "--cache", cache_name,
        ], {"checkpoint": kan_ckpt, "sha256": sha256(RUN / kan_ckpt),
            "route": "ms2", "scope": "all pixels of all 500 official test1 images"})
        report = json.loads((RUN / cache_name / "_CONFIDENCE_REPORT.json").read_text(encoding="utf-8"))
        coverage = float(report["coverage"]["0.85"])
        state["stages"]["pl_generate"]["coverage_report"] = report
        update_status(state)
        require(coverage >= 0.80,
                f"tau=0.85 all-500 coverage {coverage:.4%} is below 80%; stop before PL",
                f"tau=0.85 all-500 coverage {coverage:.4%} >= 80%")

        run_job(state, "pl_train", [
            "kan_swin.py", "--train", "--pl", "--crop", "640", "--batch-size", "2",
            "--num-workers", "3", "--target-iters", "4000", "--max-seconds", "86400",
            "--seed", "42", "--stage", "kan_from_v4_1_7439_pl",
            "--ckpt-name", pl_ckpt, "--warm-base", kan_ckpt,
            "--warm-parent-stage", "kan_from_v4_1_7439_16k", "--warm-parent-iter", "16000",
            "--pl-cache", cache_name, "--pl-tag", "from_v4_1_7439",
            "--pl-tau", "0.85", "--restart-lr", "0.25",
        ], {"checkpoint": kan_ckpt, "sha256": sha256(RUN / kan_ckpt),
            "pseudo_label_cache": cache_name, "tau": 0.85})
        record_checkpoint(state, "pl_train", pl_ckpt, "kan_from_v4_1_7439_pl", 42, 4000)
        missing = [name for name in snapshots if not (RUN / name).is_file()]
        if missing:
            raise QueueStop("missing PL snapshots: " + ", ".join(missing))
        state["stages"]["pl_train"]["snapshots"] = [
            {"filename": name, "sha256": sha256(RUN / name)} for name in snapshots
        ]
        update_status(state)

        pl_500 = run_eval(state, "pl_i2000_eval_n500", snapshots[1], pl_eval,
                          "kan_from_v4_1_7439_pl", 2000)
        require(pl_500["val_mIoU"] >= kan_500["val_mIoU"] - 0.015,
                f"PL i2000 n500 {pl_500['val_mIoU']:.4f} dropped more than 0.015 from KAN {kan_500['val_mIoU']:.4f}",
                f"PL i2000 n500 {pl_500['val_mIoU']:.4f} is within 0.015 of KAN {kan_500['val_mIoU']:.4f}")
        valid, reason = valid_classes(pl_500, kan_500)
        require(valid, "PL i2000 class-collapse gate failed: " + reason, "PL i2000: " + reason)

        state["status"] = "complete"
        state["current_stage"] = None
        state["finished_at"] = now()
        update_status(state)
        log("KAN FROM V4.1 0.7439 QUEUE COMPLETE")
    except QueueStop as error:
        state["status"] = "stopped"
        state["stop_reason"] = str(error)
        state["finished_at"] = now()
        update_status(state)
        log("KAN FROM V4.1 0.7439 QUEUE STOPPED: " + str(error))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
