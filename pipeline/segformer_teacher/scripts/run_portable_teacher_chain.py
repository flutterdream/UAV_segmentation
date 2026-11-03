"""Portable, fail-closed rebuild of the frozen SegFormer-B2 teacher chain.

Inputs are restricted to official_data and public_pretrained under the archive.
No historical prediction or checkpoint path is accepted or inspected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SEGFORMER_SHA256 = "187ca07bea003a5717c63d04ea90b07f33cd033c0ebf44b4b89fce5070d6c8f3"
DINO_SHA256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(stage: str, command: list[str], logs: Path, expected: Path | None = None) -> None:
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{stage}.stdout.log"
    stderr_path = logs / f"{stage}.stderr.log"
    if stdout_path.exists() or stderr_path.exists() or (expected and expected.exists()):
        raise FileExistsError(f"refusing to overwrite stage evidence: {stage}")
    started = time.time()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{stage} failed with exit code {completed.returncode}; see {stderr_path}")
    if expected is not None and not expected.exists():
        raise RuntimeError(f"{stage} did not create {expected}")
    metadata = {
        "stage": stage,
        "command": command,
        "duration_seconds": time.time() - started,
        "resumed": "--resume" in command,
        "output": str(expected) if expected else None,
        "output_sha256": sha256(expected) if expected and expected.is_file() else None,
    }
    (logs / f"{stage}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Generate official-mask statistics and folds without feature extraction or training")
    parser.add_argument("--work-root", type=Path,
                        help="Optional empty external workspace for non-candidate startup checks")
    args = parser.parse_args()
    archive = args.archive_root.resolve()
    project = Path(__file__).resolve().parents[1]
    data = archive / "official_data"
    public = archive / "public_pretrained" / "segformer_teacher_historical"
    segformer = public / "segformer_b2_ade_de01bae"
    dino = public / "dinov2_small_ed25f3a"
    work = args.work_root.resolve() if args.work_root else archive / "work" / "teacher"
    if work.exists() and any(work.iterdir()):
        raise FileExistsError("Teacher workspace must be empty; never resume implicitly")
    outputs = work / "outputs"
    artifacts = work / "artifacts"
    logs = work / "logs"
    stats = artifacts / "data_audit" / "image_stats.csv"
    split = artifacts / "splits" / "folds.csv"
    features = artifacts / "domain_features" / "domain_features.npz"
    weights = artifacts / "domain_features" / "train_weights.csv"
    test1 = data / "test_1" / "images"

    for directory, count in ((data / "train" / "images", 6996), (data / "train" / "masks", 6996), (test1, 500)):
        if len(list(directory.glob("*.png"))) != count:
            raise RuntimeError(f"official data count mismatch: {directory}")
    if sha256(segformer / "pytorch_model.bin") != SEGFORMER_SHA256:
        raise RuntimeError("public SegFormer-B2 SHA-256 mismatch")
    if sha256(dino / "model.safetensors") != DINO_SHA256:
        raise RuntimeError("public DINOv2-small SHA-256 mismatch")

    env = os.environ.copy()
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(archive / "work" / "hf_cache"),
        "HUGGINGFACE_HUB_CACHE": str(archive / "work" / "hf_cache" / "hub"),
    })
    py = sys.executable

    def execute(stage: str, command: list[str], expected: Path | None = None) -> None:
        old = os.environ.copy()
        os.environ.update(env)
        try:
            run(stage, command, logs, expected)
        finally:
            os.environ.clear(); os.environ.update(old)

    execute("audit_official_data", [py, str(project / "scripts" / "audit_data.py"), "--root", str(data), "--output-dir", str(artifacts / "data_audit")], stats)
    execute("make_splits", [py, str(project / "scripts" / "make_splits.py"), "--root", str(data), "--stats-csv", str(stats), "--output-dir", str(artifacts / "splits"), "--folds", "5", "--groups", "100", "--seed", "2026"], split)
    if args.prepare_only:
        (work / "PREPARE_ONLY_COMPLETE.json").write_text(json.dumps({
            "status": "preparation_complete_no_training",
            "image_stats_csv": str(stats), "image_stats_sha256": sha256(stats),
            "folds_csv": str(split), "folds_sha256": sha256(split),
            "training_started": False, "candidate": False,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    execute("extract_domain_features", [py, str(project / "scripts" / "extract_domain_features.py"), "--root", str(data), "--model-dir", str(dino), "--batch-size", "32", "--workers", "2", "--output", str(features)], features)
    execute("build_target_domain_weights", [py, str(project / "scripts" / "build_target_domain_weights.py"), "--root", str(data), "--features", str(features), "--clusters", "64", "--iterations", "40", "--blend", "0.7", "--min-weight", "0.35", "--max-weight", "3.0", "--seed", "2026", "--output", str(weights)], weights)

    common = ["--model", "segformer_b2", "--train-all", "--workers", "2", "--weight-decay", "0.01", "--rare-crop-probability", "0.75", "--strong-augmentation", "--lovasz-weight", "0.3", "--class-weights", "1,1,1,1,1.5,1,1,1", "--seed", "2026"]

    common += ["--split-csv", str(split)]

    def train(stage: str, script: str, out_name: str, arguments: list[str]) -> Path:
        out = outputs / out_name
        final = out / "final.pt"
        prefix = [py, "-u", str(project / script), "--root", str(data), "--output-dir", str(out)]
        restart_epoch = {"all_enhanced": 32, "c1_fda768": 1}.get(stage)
        if restart_epoch is not None:
            execute(stage + "_before_restart", prefix + arguments + ["--stop-after-epoch", str(restart_epoch)], out / "last.pt")
            # Only same-run generated checkpoints are resumed. Match the
            # historical restart argument list, without initial weight loading.
            resumed_args = []
            index = 0
            while index < len(arguments):
                if arguments[index] in ("--pretrained-path", "--initial-checkpoint"):
                    index += 2
                else:
                    resumed_args.append(arguments[index]); index += 1
            execute(stage + "_historical_restart", prefix + resumed_args + ["--resume", str(out / "last.pt")], final)
        else:
            execute(stage, prefix + arguments, final)
        return final

    a = train("all_enhanced", "train.py", "segformer_b2_all_enhanced", common + ["--pretrained-path", str(segformer), "--epochs", "50", "--crop-size", "512", "--batch-size", "2", "--gradient-accumulation", "2", "--learning-rate", "6e-5", "--log-interval", "100"])
    b = train("target_weighted", "train.py", "segformer_b2_target_weighted", common + ["--initial-checkpoint", str(a), "--epochs", "8", "--crop-size", "512", "--batch-size", "2", "--gradient-accumulation", "2", "--learning-rate", "1e-5", "--ema-decay", "0.9998", "--sample-weights-csv", str(weights), "--log-interval", "100"])
    c1 = train("c1_fda768", "train.py", "segformer_b2_c1_fda768", common + ["--initial-checkpoint", str(b), "--epochs", "6", "--crop-size", "768", "--batch-size", "1", "--gradient-accumulation", "4", "--learning-rate", "3e-6", "--ema-decay", "0.9998", "--fda-target-dir", str(test1), "--fda-probability", "0.35", "--fda-beta-min", "0.005", "--fda-beta-max", "0.015", "--sample-weights-csv", str(weights), "--log-interval", "100"])
    c3 = train("c3_context896", "train.py", "segformer_b2_c3_context896", common + ["--initial-checkpoint", str(c1), "--epochs", "4", "--crop-size", "896", "--batch-size", "1", "--gradient-accumulation", "4", "--learning-rate", "1.5e-6", "--ema-decay", "0.9997", "--fda-target-dir", str(test1), "--fda-probability", "0.35", "--fda-beta-min", "0.005", "--fda-beta-max", "0.015", "--sample-weights-csv", str(weights), "--log-interval", "100"])
    c4 = train("c4_context928", "train.py", "segformer_b2_c4_context928", common + ["--initial-checkpoint", str(c3), "--epochs", "2", "--crop-size", "928", "--batch-size", "1", "--gradient-accumulation", "4", "--learning-rate", "5e-7", "--min-learning-rate", "5e-8", "--ema-decay", "0.9997", "--fda-target-dir", str(test1), "--fda-probability", "0.35", "--fda-beta-min", "0.005", "--fda-beta-max", "0.015", "--sample-weights-csv", str(weights), "--log-interval", "50"])
    c5 = train("c5_full1024", "train.py", "segformer_b2_c5_full1024", common + ["--initial-checkpoint", str(c4), "--epochs", "4", "--crop-size", "1024", "--batch-size", "1", "--gradient-accumulation", "4", "--gradient-checkpointing", "--learning-rate", "4e-7", "--min-learning-rate", "4e-8", "--ema-decay", "0.9997", "--fda-target-dir", str(test1), "--fda-probability", "0.35", "--fda-beta-min", "0.005", "--fda-beta-max", "0.015", "--sample-weights-csv", str(weights), "--log-interval", "50", "--save-every", "1"])
    c6 = train("c6_dacs768", "train_uda.py", "segformer_b2_c6_dacs768", ["--initial-checkpoint", str(c5), "--split-csv", str(split), "--source-fold", "-1", "--target-dir", str(test1), "--sample-weights-csv", str(weights), "--epochs", "4", "--crop-size", "768", "--batch-size", "1", "--gradient-accumulation", "4", "--workers", "2", "--learning-rate", "1e-6", "--min-learning-rate", "1e-7", "--weight-decay", "0.01", "--teacher-decay", "0.9995", "--target-loss-weight", "0.5", "--target-ramp-epochs", "1", "--lovasz-weight", "0.3", "--class-weights", "1,1,1,1,1.5,1,1,1", "--pseudo-thresholds", "0.994,0.90,0.991,0.90,0.994,0.90,0.973,0.949", "--rare-crop-probability", "0.75", "--seed", "2026", "--log-interval", "50"])

    anchors = work / "teacher_anchors_segformer_historical_rebuilt"
    execute("c6_test1_probability_average", [py, "-u", str(project / "scripts" / "infer_probability_average.py"), "--root", str(project), "--checkpoint", str(c6), "--input-dir", str(test1), "--output-dir", str(anchors), "--tile-size", "1024", "--overlap", "0", "--scales", "0.75,1.0,1.25", "--hflip"])
    validation = work / "c6_test1_validation.json"
    execute("c6_test1_validate", [py, str(project / "scripts" / "validate_anchor_predictions.py"), "--prediction-dir", str(anchors), "--test-dir", str(test1), "--output", str(validation)], validation)
    (work / "teacher_chain_complete.json").write_text(json.dumps({"status": "complete", "c6_checkpoint": str(c6), "c6_sha256": sha256(c6), "anchors": str(anchors), "historical_inputs_used": False}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
