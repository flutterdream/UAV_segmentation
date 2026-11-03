# -*- coding: utf-8 -*-
"""Strict Stage 49 BN-statistics adaptation and single test2 MS2 package."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np
from PIL import Image
import torch


ROOT = Path(r"E:\复现")
PACKAGE = ROOT / "autodl_47_segformer_rebuilt"
SOURCE = PACKAGE / "source"
PARENT = (ROOT / "runs" / "strict" / "stage47_remote_20260926" / "artifacts" /
          "autodl_47_segformer_rebuilt" / "runs" /
          "ckpt_kan_stage47_segformer_rebuilt_i4000.pth")
PARENT_SHA256 = "67dcf9b0da69195592881dd026e689ea7fee2b23fda588f76d4891315675991b"
PARENT_STAGE = "kan_stage47_segformer_rebuilt"
OUTPUT = ROOT / "runs" / "strict" / "stage2_rebuilt_49_test2_ms2"
SMOKE_OUTPUT = ROOT / "runs" / "strict" / "stage49_rebuilt_smoke2"
STAGE49_CHECKPOINT = ROOT / "runs" / "strict" / "ckpt_kan_stage49_segformer_rebuilt_bn025.pth"
TEST1 = ROOT / "test_1" / "images"
TEST2 = ROOT / "test_2" / "images"
ALPHA = 0.25
CLASS_NAMES = ["Background", "Building", "Road", "Water", "Barren",
               "Vegetation", "Agricultural", "Vehicle"]

os.environ["STAGE47_PACKAGE_ROOT"] = str(PACKAGE)
os.environ["STRICT_REPRO_ROOT"] = str(ROOT)
os.environ["PYTHONPATH"] = str(SOURCE)
sys.path.insert(0, str(SOURCE))

import kan_swin as K  # noqa: E402
import seg_swin as S  # noqa: E402


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


def file_manifest(files: list[Path], root: Path, output: Path, label: str) -> tuple[dict, str]:
    entries = []
    for index, path in enumerate(files, 1):
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
        entries.append({
            "name": path.name,
            "relative_path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "width": width,
            "height": height,
            "mode": mode
        })
        if index % 100 == 0:
            print(f"manifest {label} {index}/{len(files)}", flush=True)
    value = {"label": label, "root": str(root), "count": len(files), "files": entries}
    write_json(output, value)
    return value, sha256(output)


def moments_hook(bucket: dict):
    def hook(_module, inputs):
        tensor = inputs[0].detach().float()
        variance, mean = torch.var_mean(tensor, dim=(0, 2, 3), unbiased=False)
        mean = mean.double().cpu()
        variance = variance.double().cpu()
        count = int(tensor.shape[0] * tensor.shape[2] * tensor.shape[3])
        if bucket["count"] == 0:
            bucket.update(count=count, mean=mean, m2=variance * count)
        else:
            previous = bucket["count"]
            delta = mean - bucket["mean"]
            total = previous + count
            bucket["mean"] += delta * (count / total)
            bucket["m2"] += variance * count + delta.square() * previous * count / total
            bucket["count"] = total
    return hook


def load_parent_model():
    if not PARENT.is_file():
        raise FileNotFoundError(PARENT)
    actual = sha256(PARENT)
    if actual != PARENT_SHA256:
        raise RuntimeError(f"parent SHA-256 mismatch: {actual}")
    payload = torch.load(PARENT, map_location="cpu", weights_only=False)
    metadata = payload.get("strict_repro", {})
    if payload.get("iter") != 4000 or metadata.get("stage") != PARENT_STAGE:
        raise RuntimeError(f"parent lineage mismatch: iter={payload.get('iter')} meta={metadata}")
    if payload.get("ema") is None:
        raise RuntimeError("parent EMA is missing")
    model = K.DeepKANSegSwin(K.NUM_CLASSES)
    model.load_state_dict(payload["ema"], strict=True)
    model = model.to(K.DEVICE).eval().requires_grad_(False)
    return payload, model


def collect_and_blend(model, files: list[Path], progress_label: str):
    original = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    parameter_names = set(dict(model.named_parameters()))
    # Some timm buffers (for example relative_position_index) are explicitly
    # non-persistent and therefore absent from state_dict/checkpoints.  The
    # Stage 47 vs Stage 49 audit is restricted to actually saved tensors.
    all_buffers = set(dict(model.named_buffers())) & set(original)
    bns = {name: module for name, module in model.named_modules()
           if isinstance(module, torch.nn.BatchNorm2d)}
    if len(bns) != 41:
        raise RuntimeError(f"expected 41 BatchNorm2d layers, got {len(bns)}")
    allowed = {name + suffix for name in bns for suffix in (".running_mean", ".running_var")}
    tracked = {name + ".num_batches_tracked" for name in bns}
    if len(allowed) != 82:
        raise RuntimeError("expected exactly 82 allowed BN running-stat buffers")
    buckets = {name: {"count": 0} for name in bns}
    hooks = [module.register_forward_pre_hook(moments_hook(buckets[name]))
             for name, module in bns.items()]
    try:
        with torch.inference_mode():
            for index, path in enumerate(files, 1):
                image = np.asarray(Image.open(path).convert("RGB"))
                if image.shape != (1024, 1024, 3):
                    raise RuntimeError(f"invalid calibration image shape: {path.name}: {image.shape}")
                tensor = S.norm(image).unsqueeze(0).to(K.DEVICE)
                with torch.autocast("cuda", torch.float16, enabled=K.DEVICE == "cuda"):
                    output = model(tensor)
                if not torch.isfinite(output).all():
                    raise FloatingPointError(f"non-finite calibration output: {path.name}")
                del tensor, output
                if index % 25 == 0 or index == len(files):
                    print(f"{progress_label} {index}/{len(files)}", flush=True)
    finally:
        for hook in hooks:
            hook.remove()

    after_collection = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    collection_changes = [name for name in original if not torch.equal(original[name], after_collection[name])]
    if collection_changes:
        raise RuntimeError(f"statistics collection mutated model: {collection_changes[:20]}")

    report = {}
    for name, module in bns.items():
        bucket = buckets[name]
        if bucket["count"] <= 1:
            raise RuntimeError(f"insufficient target samples for {name}")
        target_mean = bucket["mean"].float()
        target_var = (bucket["m2"] / (bucket["count"] - 1)).float()
        source_mean = original[name + ".running_mean"]
        source_var = original[name + ".running_var"]
        mixed_mean = (1.0 - ALPHA) * source_mean + ALPHA * target_mean
        mixed_var = ((1.0 - ALPHA) * source_var + ALPHA * target_var +
                     ALPHA * (1.0 - ALPHA) * (target_mean - source_mean).square())
        for label, value in (("target_mean", target_mean), ("target_var", target_var),
                             ("mixed_mean", mixed_mean), ("mixed_var", mixed_var)):
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"non-finite {label} in {name}")
        if (target_var < 0).any() or (mixed_var < 0).any():
            raise FloatingPointError(f"negative variance in {name}")
        with torch.no_grad():
            module.running_mean.copy_(mixed_mean)
            module.running_var.copy_(mixed_var)
        report[name] = {
            "sample_count": bucket["count"],
            "source_mean": source_mean.tolist(),
            "source_var": source_var.tolist(),
            "target_mean": target_mean.tolist(),
            "target_var": target_var.tolist(),
            "mixed_mean": mixed_mean.tolist(),
            "mixed_var": mixed_var.tolist(),
            "mean_abs_change": float((mixed_mean - source_mean).abs().mean()),
            "var_abs_change": float((mixed_var - source_var).abs().mean()),
            "normalized_mean_shift": float(((target_mean - source_mean).abs() /
                                             (source_var + 1e-5).sqrt()).mean()),
            "target_source_variance_ratio_median": float((target_var / (source_var + 1e-5)).median())
        }

    calibrated = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    changed = [name for name in original if not torch.equal(original[name], calibrated[name])]
    if set(changed) != allowed or len(changed) != 82:
        raise RuntimeError(f"changed buffer set mismatch: got={len(changed)} missing={sorted(allowed-set(changed))} extra={sorted(set(changed)-allowed)}")
    parameter_changes = [name for name in parameter_names if not torch.equal(original[name], calibrated[name])]
    non_bn_buffers = sorted(all_buffers - allowed - tracked)
    non_bn_changes = [name for name in non_bn_buffers if not torch.equal(original[name], calibrated[name])]
    tracked_changes = [name for name in tracked if not torch.equal(original[name], calibrated[name])]
    if parameter_changes or non_bn_changes or tracked_changes:
        raise RuntimeError({"parameter_changes": parameter_changes,
                            "non_bn_changes": non_bn_changes,
                            "num_batches_tracked_changes": tracked_changes})
    audit = {
        "bn_layer_count": len(bns),
        "changed_buffer_count": len(changed),
        "changed_buffers": sorted(changed),
        "parameter_count": len(parameter_names),
        "parameter_changes": parameter_changes,
        "non_bn_buffer_changes": non_bn_changes,
        "num_batches_tracked_changes": tracked_changes,
        "all_statistics_finite": True
    }
    return original, calibrated, report, audit


def validate_saved_stage49(parent_payload, stage49_payload, original_ema, calibrated_ema):
    if stage49_payload.get("iter") != 4000:
        raise RuntimeError("saved Stage 49 iteration mismatch")
    metadata = stage49_payload.get("strict_repro", {})
    if metadata.get("stage") != "kan_stage49_segformer_rebuilt_bn025":
        raise RuntimeError("saved Stage 49 stage mismatch")
    if stage49_payload.get("parent_sha256") != PARENT_SHA256:
        raise RuntimeError("saved Stage 49 parent SHA mismatch")
    if stage49_payload.get("alpha") != ALPHA or stage49_payload.get("statistics_image_count") != 500:
        raise RuntimeError("saved Stage 49 calibration metadata mismatch")
    for name, value in parent_payload["model"].items():
        if not torch.equal(value.cpu(), stage49_payload["model"][name].cpu()):
            raise RuntimeError(f"saved model field changed from Stage 47: {name}")
    for name, value in calibrated_ema.items():
        if not torch.equal(value, stage49_payload["ema"][name].cpu()):
            raise RuntimeError(f"saved EMA mismatch: {name}")
    if set(stage49_payload["ema"]) != set(original_ema):
        raise RuntimeError("saved EMA key set mismatch")


def evaluate_plain(model, val_ids: list[int], output: Path):
    started = time.monotonic()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    miou, ious = S.evaluate(model, val_ids, tta=False, bias=None)
    values = [float(value) for value in ious]
    if not math.isfinite(miou) or any(not math.isfinite(value) for value in values):
        raise RuntimeError("non-finite validation metric")
    if any(value <= 0.0 for value in values):
        raise RuntimeError(f"zero-IoU class in collapse guard: {values}")
    report = {
        "checkpoint": str(STAGE49_CHECKPOINT),
        "route": "plain",
        "bias": None,
        "validation_count": len(val_ids),
        "miou": miou,
        "class_iou": dict(zip(CLASS_NAMES, values)),
        "duration_seconds": time.monotonic() - started,
        "peak_cuda_allocated_mib": int(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
        "collapse_guard_passed": True,
        "result_prediction": False
    }
    write_json(output, report)
    return report


def infer_test2(model, files: list[Path], output_root: Path, source_hashes: dict):
    prediction_dir = output_root / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    class_counts = np.zeros(9, dtype=np.int64)
    png_records = []
    for index, path in enumerate(files, 1):
        image = np.asarray(Image.open(path).convert("RGB"))
        if image.shape != (1024, 1024, 3):
            raise RuntimeError(f"invalid test2 image shape: {path.name}: {image.shape}")
        prediction = S.predict_logits(model, image, tta="ms2", bias=None).argmax(0).numpy().astype(np.uint8)
        values = np.unique(prediction)
        if prediction.shape != (1024, 1024) or values.min() < 1 or values.max() > 8:
            raise RuntimeError(f"illegal prediction labels for {path.name}: shape={prediction.shape} values={values.tolist()}")
        destination = prediction_dir / path.name
        temporary = destination.with_suffix(".tmp.png")
        Image.fromarray(prediction, mode="L").save(temporary)
        os.replace(temporary, destination)
        class_counts += np.bincount(prediction.ravel(), minlength=9)
        png_records.append({"name": path.name, "bytes": destination.stat().st_size,
                            "sha256": sha256(destination)})
        if index % 25 == 0 or index == len(files):
            print(f"test2 ms2 {index}/{len(files)}", flush=True)

    determinism = []
    for index in (0, len(files) // 2, len(files) - 1):
        path = files[index]
        image = np.asarray(Image.open(path).convert("RGB"))
        repeated = S.predict_logits(model, image, tta="ms2", bias=None).argmax(0).numpy().astype(np.uint8)
        saved = np.asarray(Image.open(prediction_dir / path.name))
        equal = bool(np.array_equal(repeated, saved))
        determinism.append({"name": path.name, "pixel_equal": equal})
        if not equal:
            raise RuntimeError(f"determinism check failed: {path.name}")

    expected_names = {path.name for path in files}
    actual_names = {path.name for path in prediction_dir.glob("*.png")}
    if actual_names != expected_names or len(actual_names) != 1300:
        raise RuntimeError("prediction filename/count mismatch")

    total = int(class_counts.sum())
    distribution = {str(index): {"pixels": int(class_counts[index]),
                                 "fraction": float(class_counts[index] / total)}
                    for index in range(1, 9)}
    png_manifest = {
        "count": len(png_records),
        "prediction_dir": str(prediction_dir),
        "files": png_records,
        "class_distribution": distribution,
        "determinism": determinism,
        "stage49_checkpoint_sha256": sha256(STAGE49_CHECKPOINT),
        **source_hashes
    }
    png_manifest_path = output_root / "predictions_manifest.json"
    write_json(png_manifest_path, png_manifest)

    archive = output_root / "S2_rebuilt_49_test2_ms2.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in files:
            handle.write(prediction_dir / path.name, arcname=path.name)
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        if len(names) != 1300 or set(names) != expected_names or any("/" in name or "\\" in name for name in names):
            raise RuntimeError("ZIP root layout/count mismatch")
        if handle.testzip() is not None:
            raise RuntimeError("ZIP CRC validation failed")
        for info in handle.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".png"):
                raise RuntimeError(f"illegal ZIP entry: {info.filename}")

    report = {
        "status": "complete_unsubmitted",
        "prediction_count": 1300,
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "predictions_manifest": str(png_manifest_path),
        "predictions_manifest_sha256": sha256(png_manifest_path),
        "class_distribution": distribution,
        "determinism": determinism,
        "duration_seconds": time.monotonic() - started,
        "peak_cuda_allocated_mib": int(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
        "stage49": "local candidate",
        "test2_package": "generated_and_validated",
        "submitted": False,
        "real_result_score": None,
        "historical_69_96_is_reference_only": True
    }
    write_json(output_root / "test2_package_report.json", report)
    return report


def smoke() -> None:
    if SMOKE_OUTPUT.exists():
        raise FileExistsError(SMOKE_OUTPUT)
    SMOKE_OUTPUT.mkdir(parents=True)
    payload, model = load_parent_model()
    files = sorted(TEST1.glob("*.png"))[:2]
    _, _, moments, audit = collect_and_blend(model, files, "smoke statistics")
    test2_file = sorted(TEST2.glob("*.png"))[0]
    prediction = S.predict_logits(model, np.asarray(Image.open(test2_file).convert("RGB")), "ms2", None).argmax(0).numpy()
    if prediction.shape != (1024, 1024) or not np.isfinite(prediction).all():
        raise RuntimeError("smoke MS2 output invalid")
    report = {
        "status": "SMOKE_OK",
        "candidate": False,
        "parent_sha256": PARENT_SHA256,
        "calibration_images": [path.name for path in files],
        "bn_audit": audit,
        "moment_layer_count": len(moments),
        "ms2_image": test2_file.name,
        "ms2_unique_labels": np.unique(prediction).tolist()
    }
    write_json(SMOKE_OUTPUT / "SMOKE_REPORT.json", report)
    print("SMOKE_OK", flush=True)


def formal() -> None:
    if OUTPUT.exists() or STAGE49_CHECKPOINT.exists():
        raise FileExistsError("formal Stage 49/test2 output already exists")
    OUTPUT.mkdir(parents=True)
    shutil.copy2(Path(__file__).resolve(), OUTPUT / "run_stage49_rebuilt_and_test2.py")
    status_path = OUTPUT / "status.json"
    started = time.monotonic()

    def status(phase: str, **extra):
        value = {"phase": phase, "elapsed_seconds": time.monotonic() - started,
                 "stage49": "local candidate", "test2_package": "not_generated",
                 "submitted": False, "real_result_score": None}
        value.update(extra)
        write_json(status_path, value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    status("preflight")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    test1_files = sorted(TEST1.glob("*.png"))
    test2_files = sorted(TEST2.glob("*.png"))
    if len(test1_files) != 500 or len(test2_files) != 1300:
        raise RuntimeError(f"official input count mismatch: test1={len(test1_files)} test2={len(test2_files)}")
    for path in test1_files + test2_files:
        with Image.open(path) as image:
            if image.size != (1024, 1024):
                raise RuntimeError(f"invalid official image size: {path}")

    _, test1_manifest_sha = file_manifest(test1_files, TEST1, OUTPUT / "test1_statistics_manifest.json", "official_test1_bn_statistics")
    _, test2_manifest_sha = file_manifest(test2_files, TEST2, OUTPUT / "test2_input_manifest.json", "official_test2_ms2_inference")
    payload, model = load_parent_model()
    status("collecting_test1_bn_moments", bn_layers=41, statistics_images=500)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    bn_started = time.monotonic()
    original_ema, calibrated_ema, moments, audit = collect_and_blend(model, test1_files, "test1 BN statistics")
    bn_seconds = time.monotonic() - bn_started
    bn_peak = int(torch.cuda.max_memory_allocated() / 2**20)
    write_json(OUTPUT / "stage49_bn_moments.json", moments)
    write_json(OUTPUT / "stage49_integrity_audit.json", audit)

    strict_metadata = dict(payload.get("strict_repro", {}))
    strict_metadata.update({"stage": "kan_stage49_segformer_rebuilt_bn025",
                            "seed": 17, "parents": [PARENT.name]})
    stage49_payload = {
        "model": payload["model"],
        "ema": calibrated_ema,
        "iter": 4000,
        "strict_repro": strict_metadata,
        "method": "test1_population_bn_moments_blend025",
        "parent_checkpoint": PARENT.name,
        "parent_sha256": PARENT_SHA256,
        "alpha": ALPHA,
        "statistics_pool": str(TEST1),
        "statistics_image_count": 500,
        "statistics_manifest": "test1_statistics_manifest.json",
        "statistics_manifest_sha256": test1_manifest_sha,
        "changed_buffers": audit["changed_buffers"],
        "stage47_config": payload.get("stage47_config"),
        "stage49_config": {"gradient_updates": False, "scale": 1.0, "flip": False,
                           "augmentation": False, "crop": False,
                           "variance_between_group_term": True}
    }
    temporary = STAGE49_CHECKPOINT.with_suffix(".pth.tmp")
    torch.save(stage49_payload, temporary)
    os.replace(temporary, STAGE49_CHECKPOINT)
    checkpoint_sha = sha256(STAGE49_CHECKPOINT)
    saved = torch.load(STAGE49_CHECKPOINT, map_location="cpu", weights_only=False)
    validate_saved_stage49(payload, saved, original_ema, calibrated_ema)
    status("stage49_saved_validated", checkpoint=str(STAGE49_CHECKPOINT),
           checkpoint_sha256=checkpoint_sha, changed_buffers=82,
           bn_seconds=bn_seconds, bn_peak_cuda_allocated_mib=bn_peak)

    split = json.loads((PACKAGE / "inputs" / "split.json").read_text(encoding="utf-8"))
    val_ids = split["val_ids"]
    all_ids = list(range(6996))
    import random
    random.Random(123).shuffle(all_ids)
    if len(val_ids) != 500 or val_ids != all_ids[:500]:
        raise RuntimeError("fixed validation split mismatch")
    evaluation = evaluate_plain(model, val_ids, OUTPUT / "stage49_plain_ema_n500.json")
    status("stage49_guard_passed", checkpoint_sha256=checkpoint_sha,
           validation_miou=evaluation["miou"], validation_class_iou=evaluation["class_iou"])

    source_hashes = {
        "inference_source_sha256": sha256(OUTPUT / "run_stage49_rebuilt_and_test2.py"),
        "seg_swin_source_sha256": sha256(SOURCE / "seg_swin.py"),
        "kan_swin_source_sha256": sha256(SOURCE / "kan_swin.py"),
        "test2_input_manifest_sha256": test2_manifest_sha
    }
    result = infer_test2(model, test2_files, OUTPUT, source_hashes)
    status("complete", checkpoint_sha256=checkpoint_sha,
           validation_miou=evaluation["miou"], test2_package="generated_and_validated",
           package=result["archive"], package_sha256=result["archive_sha256"],
           inference_seconds=result["duration_seconds"],
           inference_peak_cuda_allocated_mib=result["peak_cuda_allocated_mib"])

    final = {
        "stage49": "local candidate",
        "stage49_checkpoint": str(STAGE49_CHECKPOINT),
        "stage49_checkpoint_sha256": checkpoint_sha,
        "parent_checkpoint": str(PARENT),
        "parent_sha256": PARENT_SHA256,
        "bn_statistics_images": 500,
        "bn_changed_buffers": 82,
        "bn_duration_seconds": bn_seconds,
        "bn_peak_cuda_allocated_mib": bn_peak,
        "validation": evaluation,
        "test2": result,
        "test2_package": "generated_and_validated",
        "submitted": False,
        "real_result_score": None,
        "historical_69_96_is_reference_only": True,
        "total_duration_seconds": time.monotonic() - started,
        **source_hashes
    }
    write_json(OUTPUT / "FINAL_REPORT.json", final)
    print("STAGE49_TEST2_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(17)
    np.random.seed(17)
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        smoke() if args.smoke else formal()
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
