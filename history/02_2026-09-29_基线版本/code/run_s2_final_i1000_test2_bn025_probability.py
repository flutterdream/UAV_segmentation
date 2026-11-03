# -*- coding: utf-8 -*-
"""Preregistered i600 candidate: Stage47 early stop -> test2 BN025 -> MS2 probability average."""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import zipfile

import numpy as np
from PIL import Image
import torch


ROOT = Path(r"E:\复现")
SCRIPTS = ROOT / "scripts"
PACKAGE = ROOT / "autodl_47_segformer_rebuilt"
SOURCE = PACKAGE / "source"
STRICT = ROOT / "runs" / "strict"
PARENT = (STRICT / "stage47_earlystop_remote_20260927" /
          "autodl_47_earlystop_bracket" / "runs" /
          "ckpt_kan_stage47_earlystop_bracket_i600.pth")
PARENT_SHA256 = "4c3e9ae1e230f82f23cdbdcebed593ed1dadd8ba3dd513b22acdfc2ef3bd4da5"
BASELINE_CKPT = STRICT / "ckpt_kan_s2_final_i1000_test2_bn025.pth"
BASELINE_CKPT_SHA256 = "76092b5a6ed92688a03ca6b0ddbbc42c1935a83ac1079dc2268771f7db1758ae"
BASELINE_DIR = STRICT / "s2_final_i1000_test2_bn025_probability"
BASELINE_ZIP = BASELINE_DIR / "S2_final_i1000_test2_bn025_ms2_probability.zip"
BASELINE_ZIP_SHA256 = "9b709f80043b57f80d1d9d11a76e2795a59962240b197b174615f35e90eeef27"
TEST2 = ROOT / "test_2" / "images"
OUTPUT = STRICT / "s2_stage47_earlystop_i600_test2_bn025_probability"
CHECKPOINT = STRICT / "ckpt_kan_s2_stage47_earlystop_i600_test2_bn025.pth"
ALPHA = 0.25
N500_FLOOR = 0.7551544710120403
CLASS_NAMES = ["Background", "Building", "Road", "Water", "Barren",
               "Vegetation", "Agricultural", "Vehicle"]

os.environ["STAGE47_PACKAGE_ROOT"] = str(PACKAGE)
os.environ["STRICT_REPRO_ROOT"] = str(ROOT)
os.environ["PYTHONPATH"] = str(SOURCE)
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SOURCE))

import run_stage49_rebuilt_and_test2 as B  # noqa: E402
import run_stage49_test2_ms2_probability as P  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def manifest(files: list[Path], destination: Path) -> str:
    entries = []
    for index, path in enumerate(files, 1):
        with Image.open(path) as image:
            if image.size != (1024, 1024):
                raise RuntimeError(f"invalid test2 image size: {path.name}: {image.size}")
        entries.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
        if index % 100 == 0:
            print(f"manifest {index}/{len(files)}", flush=True)
    write_json(destination, {"root": str(TEST2), "count": len(files), "files": entries})
    return sha256(destination)


def assert_saved_checkpoint(parent_payload, candidate_payload, original, calibrated, audit) -> None:
    meta = candidate_payload.get("strict_repro", {})
    if (candidate_payload.get("iter") != 600 or
            meta.get("stage") != "kan_s2_stage47_earlystop_i600_test2_bn025" or
            candidate_payload.get("alpha") != ALPHA or
            candidate_payload.get("statistics_image_count") != 1300 or
            candidate_payload.get("parent_sha256") != PARENT_SHA256):
        raise RuntimeError("candidate metadata mismatch")
    if set(candidate_payload["ema"]) != set(original):
        raise RuntimeError("candidate EMA key set mismatch")
    for name, value in calibrated.items():
        if not torch.equal(value, candidate_payload["ema"][name].cpu()):
            raise RuntimeError(f"candidate EMA mismatch: {name}")
    for name, value in parent_payload["model"].items():
        if not torch.equal(value.cpu(), candidate_payload["model"][name].cpu()):
            raise RuntimeError(f"model field changed: {name}")
    if (audit["changed_buffer_count"] != 82 or audit["parameter_changes"] or
            audit["non_bn_buffer_changes"] or audit["num_batches_tracked_changes"]):
        raise RuntimeError("BN-only integrity gate failed")


def evaluate_n500(model) -> dict:
    split = json.loads((PACKAGE / "inputs" / "split.json").read_text(encoding="utf-8"))
    val_ids = split["val_ids"]
    ids = list(range(6996))
    random.Random(123).shuffle(ids)
    if len(val_ids) != 500 or val_ids != ids[:500]:
        raise RuntimeError("fixed validation split mismatch")
    started = time.monotonic()
    miou, ious = B.S.evaluate(model, val_ids, tta=False, bias=None)
    values = [float(v) for v in ious]
    result = {"route": "plain", "validation_count": 500, "miou": float(miou),
              "class_iou": dict(zip(CLASS_NAMES, values)),
              "duration_seconds": time.monotonic() - started,
              "result_prediction": False}
    write_json(OUTPUT / "n500_plain_ema.json", result)
    if (not math.isfinite(miou) or miou < N500_FLOOR or
            any((not math.isfinite(v)) or v <= 0.0 for v in values)):
        raise RuntimeError(f"n500 gate failed: {result}")
    return result


def infer_compare_and_package(model, files: list[Path]) -> dict:
    prediction_dir = OUTPUT / "predictions"
    prediction_dir.mkdir()
    baseline_dir = BASELINE_DIR / "predictions"
    if len(list(baseline_dir.glob("*.png"))) != 1300:
        raise RuntimeError("baseline 69.88 predictions are incomplete")
    started = time.monotonic()
    class_counts = np.zeros(9, dtype=np.int64)
    baseline_counts = np.zeros(9, dtype=np.int64)
    total_disagreement = 0
    total_pixels = 0
    per_image = []
    records = []
    for index, path in enumerate(files, 1):
        image = np.asarray(Image.open(path).convert("RGB"))
        prediction = P.predict_probability_average(model, image).argmax(0).numpy().astype(np.uint8)
        baseline = np.asarray(Image.open(baseline_dir / path.name))
        labels = np.unique(prediction)
        if prediction.shape != (1024, 1024) or labels.min() < 1 or labels.max() > 8:
            raise RuntimeError(f"invalid prediction {path.name}: {labels.tolist()}")
        disagreement = int(np.count_nonzero(prediction != baseline))
        pixels = prediction.size
        total_disagreement += disagreement
        total_pixels += pixels
        per_image.append({"name": path.name, "fraction": disagreement / pixels})
        class_counts += np.bincount(prediction.ravel(), minlength=9)
        baseline_counts += np.bincount(baseline.ravel(), minlength=9)
        output = prediction_dir / path.name
        temporary = output.with_suffix(".tmp.png")
        Image.fromarray(prediction, mode="L").save(temporary)
        os.replace(temporary, output)
        records.append({"name": path.name, "bytes": output.stat().st_size,
                        "sha256": sha256(output)})
        if index % 25 == 0 or index == len(files):
            print(f"test2 candidate {index}/{len(files)}", flush=True)
            write_json(OUTPUT / "status.json", {"phase": "inference", "completed": index,
                       "total": len(files), "submitted": False,
                       "elapsed_seconds": time.monotonic() - started})

    overall = total_disagreement / total_pixels
    fractions = np.asarray([x["fraction"] for x in per_image])
    candidate_fraction = class_counts[1:9] / class_counts[1:9].sum()
    baseline_fraction = baseline_counts[1:9] / baseline_counts[1:9].sum()
    delta = candidate_fraction - baseline_fraction
    comparison = {
        "baseline": "submitted_70.30_i1000_test2_bn025_probability_average",
        "overall_pixel_disagreement": float(overall),
        "per_image_median": float(np.median(fractions)),
        "per_image_p90": float(np.quantile(fractions, 0.90)),
        "per_image_p95": float(np.quantile(fractions, 0.95)),
        "per_image_p99": float(np.quantile(fractions, 0.99)),
        "per_image_max": float(fractions.max()),
        "candidate_class_fraction": dict(zip(CLASS_NAMES, candidate_fraction.tolist())),
        "baseline_class_fraction": dict(zip(CLASS_NAMES, baseline_fraction.tolist())),
        "class_fraction_delta": dict(zip(CLASS_NAMES, delta.tolist())),
        "per_image": per_image
    }
    write_json(OUTPUT / "prediction_comparison.json", comparison)
    if not (0.001 <= overall <= 0.05):
        raise RuntimeError(f"overall disagreement gate failed: {overall}")
    if np.abs(delta).max() > 0.02:
        raise RuntimeError(f"class-distribution gate failed: {delta}")

    determinism = []
    for idx in (0, len(files) // 2, len(files) - 1):
        path = files[idx]
        repeat = P.predict_probability_average(
            model, np.asarray(Image.open(path).convert("RGB"))).argmax(0).numpy().astype(np.uint8)
        equal = bool(np.array_equal(repeat, np.asarray(Image.open(prediction_dir / path.name))))
        determinism.append({"name": path.name, "pixel_equal": equal})
        if not equal:
            raise RuntimeError(f"determinism gate failed: {path.name}")

    prediction_manifest = OUTPUT / "predictions_manifest.json"
    write_json(prediction_manifest, {"count": 1300, "route": "eight_pass_probability_average",
               "passes": P.PASSES, "files": records, "comparison": comparison,
               "determinism": determinism})
    archive = OUTPUT / "S2_stage47_earlystop_i600_test2_bn025_ms2_probability.zip"
    names = {p.name for p in files}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in files:
            handle.write(prediction_dir / path.name, arcname=path.name)
    with zipfile.ZipFile(archive) as handle:
        if (len(handle.namelist()) != 1300 or set(handle.namelist()) != names or
                any("/" in n or "\\" in n for n in handle.namelist()) or
                handle.testzip() is not None):
            raise RuntimeError("ZIP gate failed")
    return {"archive": str(archive), "archive_sha256": sha256(archive),
            "archive_bytes": archive.stat().st_size,
            "prediction_manifest_sha256": sha256(prediction_manifest),
            "comparison": comparison, "determinism": determinism,
            "duration_seconds": time.monotonic() - started}


def main() -> None:
    if OUTPUT.exists() or CHECKPOINT.exists():
        raise FileExistsError("candidate output already exists; never overwrite")
    OUTPUT.mkdir(parents=True)
    shutil.copy2(Path(__file__).resolve(), OUTPUT / Path(__file__).name)
    shutil.copy2(ROOT / "docs" / "S2_STAGE47_EARLYSTOP_I600_BN025_PREREG_20260927.md",
                 OUTPUT / "PREREGISTRATION.md")
    started = time.monotonic()
    torch.set_num_threads(8)
    torch.manual_seed(17)
    np.random.seed(17)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if sha256(PARENT) != PARENT_SHA256:
        raise RuntimeError("Stage47 parent hash mismatch")
    if sha256(BASELINE_CKPT) != BASELINE_CKPT_SHA256 or sha256(BASELINE_ZIP) != BASELINE_ZIP_SHA256:
        raise RuntimeError("submitted 70.30 baseline integrity mismatch")
    files = sorted(TEST2.glob("*.png"))
    if len(files) != 1300:
        raise RuntimeError(f"test2 count mismatch: {len(files)}")
    test2_manifest_sha = manifest(files, OUTPUT / "test2_input_manifest.json")
    parent_payload = torch.load(PARENT, map_location="cpu", weights_only=False)
    parent_meta = parent_payload.get("strict_repro", {})
    if (parent_payload.get("iter") != 600 or
            parent_meta.get("stage") != "kan_stage47_earlystop_bracket" or
            parent_payload.get("ema") is None):
        raise RuntimeError("Stage47 i1000 parent lineage mismatch")
    model = B.K.DeepKANSegSwin(B.K.NUM_CLASSES)
    model.load_state_dict(parent_payload["ema"], strict=True)
    model = model.to(B.K.DEVICE).eval().requires_grad_(False)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    bn_started = time.monotonic()
    original, calibrated, moments, audit = B.collect_and_blend(model, files, "test2 BN statistics")
    bn_seconds = time.monotonic() - bn_started
    write_json(OUTPUT / "bn_moments.json", moments)
    write_json(OUTPUT / "bn_integrity_audit.json", audit)
    strict_meta = dict(parent_payload.get("strict_repro", {}))
    strict_meta.update({"stage": "kan_s2_stage47_earlystop_i600_test2_bn025", "seed": 17,
                        "parents": [PARENT.name]})
    payload = {"model": parent_payload["model"], "ema": calibrated, "iter": 600,
               "strict_repro": strict_meta, "method": "test2_population_bn_moments_blend025",
               "parent_checkpoint": PARENT.name, "parent_sha256": PARENT_SHA256,
               "alpha": ALPHA, "statistics_pool": str(TEST2),
               "statistics_image_count": 1300,
               "statistics_manifest_sha256": test2_manifest_sha,
               "changed_buffers": audit["changed_buffers"],
               "stage47_config": parent_payload.get("stage47_config"),
               "stage49_config": {"gradient_updates": False, "scale": 1.0,
                   "flip": False, "augmentation": False, "crop": False,
                   "variance_between_group_term": True}}
    temporary = CHECKPOINT.with_suffix(".pth.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, CHECKPOINT)
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    assert_saved_checkpoint(parent_payload, saved, original, calibrated, audit)
    checkpoint_sha = sha256(CHECKPOINT)
    evaluation = evaluate_n500(model)
    result = infer_compare_and_package(model, files)
    final = {"status": "complete_unsubmitted", "classification": "local candidate",
             "single_variable": "Stage47 parent: submitted i1000 trajectory -> preregistered new i600 early stop; BN025 and inference fixed",
             "alpha": ALPHA, "parent_sha256": PARENT_SHA256,
             "checkpoint": str(CHECKPOINT), "checkpoint_sha256": checkpoint_sha,
             "test2_input_manifest_sha256": test2_manifest_sha,
             "bn_changed_buffers": 82, "bn_seconds": bn_seconds,
             "peak_cuda_allocated_mib": int(torch.cuda.max_memory_allocated() / 2**20),
             "validation": evaluation, "package": result,
             "submitted": False, "real_result_score": None,
             "total_duration_seconds": time.monotonic() - started}
    write_json(OUTPUT / "FINAL_REPORT.json", final)
    write_json(OUTPUT / "status.json", {"phase": "complete", **final})
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        main()
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
