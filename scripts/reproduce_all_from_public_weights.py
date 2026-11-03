"""Clean-start replay of the winning Stage47 i600 + test2 BN alpha075 lineage.

Default behavior always starts from public academic weights. It never searches
system model caches, other drives, old experiments, or milestone checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(stage: str, command: list[str], log_dir: Path, env: dict[str, str], expected: Path | None = None) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / f"{stage}.stdout.log"
    err = log_dir / f"{stage}.stderr.log"
    meta = log_dir / f"{stage}.json"
    if out.exists() or err.exists() or meta.exists() or (expected and expected.exists()):
        raise FileExistsError(f"refusing to resume or overwrite stage {stage}")
    started = time.time()
    with out.open("wb") as stdout, err.open("wb") as stderr:
        result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, check=False)
    record = {
        "stage": stage,
        "command": command,
        "return_code": result.returncode,
        "duration_seconds": time.time() - started,
        "resumed": "--resume" in command,
        "expected": str(expected) if expected else None,
    }
    if expected and expected.is_file():
        record.update({"output_sha256": sha256(expected), "output_bytes": expected.stat().st_size})
    meta.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"{stage} failed; see {err}")
    if expected is not None and not expected.exists():
        raise RuntimeError(f"{stage} did not create {expected}")


def require_eval(path: Path, minimum_miou: float) -> None:
    result = json.loads(path.read_text(encoding="utf-8"))
    score = float(result["val_mIoU"])
    per_class = result["val_IoU"]
    if score < minimum_miou:
        raise RuntimeError(f"quality gate failed: {path.name}: {score} < {minimum_miou}")
    if not math.isfinite(score) or any(value is None or not math.isfinite(float(value)) or float(value) <= 0.0 for value in per_class.values()):
        raise RuntimeError(f"quality gate failed: non-finite/zero class IoU in {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--teacher-only", action="store_true")
    args = parser.parse_args()
    root = args.archive_root.resolve()
    code = root / "code"
    kan = code / "kan_pipeline"
    teacher = code / "segformer_teacher"
    official = root / "official_data"
    work = root / "work"
    strict = work / "runs" / "strict"
    logs = work / "orchestration_logs"
    if work.exists() and any(work.iterdir()):
        raise FileExistsError("work/ already exists; default reproduction is clean-start only and will not resume")
    strict.mkdir(parents=True)

    env = os.environ.copy()
    env.update({
        "STRICT_REPRO_ROOT": str(root),
        "STRICT_RUN_ROOT": str(strict),
        "STRICT_RUNTIME_ROOT": str(work / "runtime_cache"),
        "HF_HOME": str(root / "public_pretrained" / "huggingface"),
        "HF_HUB_CACHE": str(root / "public_pretrained" / "huggingface" / "hub"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TORCH_HOME": str(work / "runtime_cache" / "torch"),
        "XDG_CACHE_HOME": str(work / "runtime_cache" / "xdg"),
        "CUDA_CACHE_PATH": str(work / "runtime_cache" / "cuda"),
        "PYTHONPATH": str(kan) + os.pathsep + str(teacher),
    })
    py = sys.executable
    teacher_py = os.environ.get("TEACHER_PYTHON", py)
    stage47_py = os.environ.get("STAGE47_PYTHON", py)

    run(
        "verify_inputs",
        [py, str(code / "prepare_inputs.py"), "--root", str(root), "--verify-only", "--rederive-check"],
        logs,
        env,
    )
    run("teacher_chain", [teacher_py, str(teacher / "scripts" / "run_portable_teacher_chain.py"), "--archive-root", str(root)], logs, env, work / "teacher" / "teacher_chain_complete.json")
    if args.teacher_only:
        return

    def train(stage: str, command: list[str], expected_name: str) -> Path:
        expected = strict / expected_name
        run(stage, [py, "-u"] + command, logs, env, expected)
        return expected

    v4 = train("swin_v4_12k", [str(kan / "seg_swin.py"), "--train", "--crop", "512", "--target-iters", "12000", "--seed", "42", "--stage", "main_v4_corrected_once", "--ckpt-name", "ckpt_swin_upernet_v4_corrected_once.pth", "--frozen-7439-scale-floor", "--max-seconds", "2147483647"], "ckpt_swin_upernet_v4_corrected_once.pth")
    run("swin_v4_eval_n100", [py, str(kan / "strict_eval.py"), "--model", "swin", "--ckpt", v4.name, "--n", "100", "--output", "eval_swin_v4_n100.json", "--expected-stage", "main_v4_corrected_once", "--expected-iter", "12000"], logs, env, strict / "eval_swin_v4_n100.json")
    run("swin_v4_eval_n500", [py, str(kan / "strict_eval.py"), "--model", "swin", "--ckpt", v4.name, "--n", "500", "--output", "eval_swin_v4_n500.json", "--expected-stage", "main_v4_corrected_once", "--expected-iter", "12000"], logs, env, strict / "eval_swin_v4_n500.json")
    require_eval(strict / "eval_swin_v4_n100.json", 0.735)
    v41_args = [str(kan / "seg_swin.py"), "--train", "--crop", "640", "--target-iters", "4000", "--seed", "42", "--stage", "main_v4_1_strict", "--ckpt-name", "ckpt_swin_upernet_v4_1_strict.pth", "--warm-from", v4.name, "--strict-v4-1", "--strict-parent-stage", "main_v4_corrected_once", "--frozen-7439-scale-floor", "--max-seconds", "2147483647"]
    v41 = train("swin_v4_1_to3400", v41_args + ["--stop-after-iter", "3400"], "ckpt_swin_upernet_v4_1_strict.pth")
    # Historical user-authorized restart at i3400 is reproduced using only the
    # checkpoint just generated in this clean run, never a shipped checkpoint.
    run("swin_v4_1_resume3400", [py, "-u"] + v41_args + ["--resume"], logs, env)
    run("swin_v4_1_eval_n100", [py, str(kan / "strict_eval.py"), "--model", "swin", "--ckpt", v41.name, "--n", "100", "--output", "eval_swin_v4_1_n100.json", "--expected-stage", "main_v4_1_strict", "--expected-iter", "4000"], logs, env, strict / "eval_swin_v4_1_n100.json")
    run("swin_v4_1_eval_n500", [py, str(kan / "strict_eval.py"), "--model", "swin", "--ckpt", v41.name, "--n", "500", "--output", "eval_swin_v4_1_n500.json", "--expected-stage", "main_v4_1_strict", "--expected-iter", "4000"], logs, env, strict / "eval_swin_v4_1_n500.json")
    require_eval(strict / "eval_swin_v4_1_n500.json", 0.7224)
    base = train("kan_base_16k", [str(kan / "kan_swin.py"), "--train", "--crop", "512", "--batch-size", "2", "--target-iters", "16000", "--seed", "42", "--stage", "kan_from_v4_1_7439_16k", "--ckpt-name", "ckpt_kan_from_v4_1_7439_16k.pth", "--encoder-warm", v41.name, "--encoder-parent-stage", "main_v4_1_strict", "--max-seconds", "2147483647"], "ckpt_kan_from_v4_1_7439_16k.pth")
    run("kan_base_eval_n500", [py, str(kan / "strict_eval.py"), "--model", "kan", "--ckpt", base.name, "--n", "500", "--output", "eval_kan_base_n500.json", "--expected-stage", "kan_from_v4_1_7439_16k", "--expected-iter", "16000"], logs, env, strict / "eval_kan_base_n500.json")
    require_eval(strict / "eval_kan_base_n500.json", 0.745)
    run("kan_test1_ms2_pseudo", [py, str(kan / "pl_gen.py"), "--ckpt", base.name, "--cache", "pl_cache_kan_from_v4_1_7439_ms2"], logs, env, strict / "pl_cache_kan_from_v4_1_7439_ms2" / "_CACHE_MANIFEST.json")

    pl = train("kan_pl_4k", [str(kan / "kan_swin.py"), "--train", "--pl", "--pl-tau", "0.85", "--pl-cache", "pl_cache_kan_from_v4_1_7439_ms2", "--pl-tag", "historical_rebuilt", "--crop", "640", "--batch-size", "2", "--target-iters", "4000", "--restart-lr", "0.25", "--seed", "42", "--stage", "kan_historical_pl_rebuilt", "--ckpt-name", "ckpt_kan_historical_pl_rebuilt.pth", "--warm-base", base.name, "--warm-parent-stage", "kan_from_v4_1_7439_16k", "--warm-parent-iter", "16000", "--max-seconds", "2147483647"], "ckpt_kan_historical_pl_rebuilt.pth")
    pl_i2000 = strict / "ckpt_kan_historical_pl_rebuilt_i2000.pth"
    run("kan_pl_i2000_eval_n500", [py, str(kan / "strict_eval.py"), "--model", "kan", "--ckpt", pl_i2000.name, "--n", "500", "--output", "eval_kan_pl_i2000_n500.json", "--expected-stage", "kan_historical_pl_rebuilt", "--expected-iter", "2000"], logs, env, strict / "eval_kan_pl_i2000_n500.json")
    require_eval(strict / "eval_kan_pl_i2000_n500.json", 0.7300)

    teacher_anchors = work / "teacher" / "teacher_anchors_segformer_historical_rebuilt"
    anchor_dest = strict / "teacher_anchors_segformer_historical_rebuilt"
    shutil.copytree(teacher_anchors, anchor_dest)
    distill = train("kan_segformer_distill_4k", [str(kan / "kan_swin.py"), "--train", "--anchor", "--anchor-dir", anchor_dest.name, "--crop", "640", "--batch-size", "2", "--target-iters", "4000", "--restart-lr", "0.25", "--seed", "42", "--stage", "kan_segformer_distill_rebuilt", "--ckpt-name", "ckpt_kan_segformer_distill_rebuilt.pth", "--warm-base", pl_i2000.name, "--warm-parent-stage", "kan_historical_pl_rebuilt", "--warm-parent-iter", "2000", "--max-seconds", "2147483647"], "ckpt_kan_segformer_distill_rebuilt.pth")
    distill_i2000 = strict / "ckpt_kan_segformer_distill_rebuilt_i2000.pth"
    run("kan_distill_i2000_eval_n500", [py, str(kan / "strict_eval.py"), "--model", "kan", "--ckpt", distill_i2000.name, "--n", "500", "--output", "eval_kan_distill_i2000_n500.json", "--expected-stage", "kan_segformer_distill_rebuilt", "--expected-iter", "2000"], logs, env, strict / "eval_kan_distill_i2000_n500.json")
    require_eval(strict / "eval_kan_distill_i2000_n500.json", 0.7300)

    stage47 = work / "stage47"
    inputs = stage47 / "inputs"
    inputs.mkdir(parents=True)
    shutil.copy2(distill_i2000, inputs / distill_i2000.name)
    shutil.copy2(kan / "train_mask_stats.npy", inputs / "train_mask_stats.npy")
    shutil.copytree(anchor_dest, inputs / anchor_dest.name)
    # Replay the exact historical bracket trainer unchanged. Its scheduler
    # horizon is 4000, execution length is 1400, and the selected snapshot is
    # i600. Later updates cannot alter the already-written i600 snapshot.
    run("kan_stage47_1400_select_i600", [stage47_py, "-u", str(kan / "stage47_i600.py"), "--package-root", str(stage47), "--data-root", str(official), "--config", str(root / "configs" / "stage47_formal.json")], logs, env, stage47 / "runs" / "ckpt_kan_stage47_earlystop_bracket_i1400.pth")
    parent = stage47 / "runs" / "ckpt_kan_stage47_earlystop_bracket_i600.pth"
    final = strict / "ckpt_kan_stage47_i600_test2_alpha075_retrained.pth"
    run("test2_bn_alpha075", [py, "-u", str(kan / "final_pipeline.py"), "--archive-root", str(root), "calibrate", "--parent", str(parent), "--output", str(final)], logs, env, final)
    output = work / "final_predictions"
    run("test2_eight_probability", [py, "-u", str(kan / "final_pipeline.py"), "--archive-root", str(root), "infer", "--checkpoint", str(final), "--output-dir", str(output), "--retrained"], logs, env, output / "predictions.zip")

    summary = {
        "status": "complete_unsubmitted",
        "final_checkpoint": str(final),
        "final_checkpoint_sha256": sha256(final),
        "submission_zip": str(output / "predictions.zip"),
        "recipe_reference_real_score": 70.65,
        "retraining_real_score": None,
        "submitted": False,
        "historical_inputs_used": False,
    }
    (work / "REPRODUCTION_COMPLETE.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
