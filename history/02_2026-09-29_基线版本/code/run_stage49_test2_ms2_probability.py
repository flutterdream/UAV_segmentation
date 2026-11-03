# -*- coding: utf-8 -*-
"""Corrected eight-route probability-average test2 inference for strict Stage 49."""
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
import torch.nn.functional as F


ROOT = Path(r"E:\复现")
PACKAGE = ROOT / "autodl_47_segformer_rebuilt"
SOURCE = PACKAGE / "source"
STAGE49 = ROOT / "runs" / "strict" / "ckpt_kan_stage49_segformer_rebuilt_bn025.pth"
STAGE49_SHA256 = "5654185c2ee517dd02e16a00acc7f68230fc45bb7e7d7a8e53bebdbe2d4036ba"
OUTPUT = ROOT / "runs" / "strict" / "stage2_rebuilt_49_test2_ms2_probability"
SMOKE_OUTPUT = ROOT / "runs" / "strict" / "stage49_test2_probability_smoke"
TEST2 = ROOT / "test_2" / "images"
PASSES = [(1.0, False), (1.0, True), (0.75, False), (0.75, True),
          (1.25, False), (1.25, True), (1.5, False), (0.5, False)]

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


@torch.no_grad()
def predict_probability_average(model, image: np.ndarray) -> torch.Tensor:
    height, width = image.shape[:2]
    accumulator = None
    for scale, flip in PASSES:
        transformed = image[:, ::-1] if flip else image
        if scale != 1.0:
            transformed = np.asarray(Image.fromarray(transformed).resize(
                (int(width * scale), int(height * scale)), Image.BILINEAR))
        tensor = S.norm(transformed).unsqueeze(0).to(K.DEVICE)
        with torch.autocast("cuda", torch.float16, enabled=K.DEVICE == "cuda"):
            logits = model(tensor)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
        logits = logits.float()
        if scale != 1.0:
            logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        probabilities = logits.softmax(dim=1).cpu()
        if flip:
            probabilities = torch.flip(probabilities, dims=(3,))
        accumulator = probabilities if accumulator is None else accumulator + probabilities
        del tensor, logits, probabilities
    averaged = accumulator / len(PASSES)
    if not torch.isfinite(averaged).all():
        raise FloatingPointError("non-finite probability average")
    normalization_error = float((averaged.sum(1) - 1.0).abs().max())
    if normalization_error > 1e-5:
        raise RuntimeError(f"probability normalization error: {normalization_error}")
    return averaged.squeeze(0)


def load_model():
    if sha256(STAGE49) != STAGE49_SHA256:
        raise RuntimeError("Stage 49 SHA-256 mismatch")
    payload = torch.load(STAGE49, map_location="cpu", weights_only=False)
    metadata = payload.get("strict_repro", {})
    if (payload.get("iter") != 4000 or
            metadata.get("stage") != "kan_stage49_segformer_rebuilt_bn025" or
            payload.get("alpha") != 0.25 or payload.get("statistics_image_count") != 500 or
            payload.get("ema") is None):
        raise RuntimeError("Stage 49 metadata/EMA mismatch")
    model = K.DeepKANSegSwin(K.NUM_CLASSES)
    model.load_state_dict(payload["ema"], strict=True)
    return model.to(K.DEVICE).eval().requires_grad_(False)


def make_test2_manifest(files: list[Path], destination: Path) -> str:
    entries = []
    for index, path in enumerate(files, 1):
        with Image.open(path) as image:
            if image.size != (1024, 1024):
                raise RuntimeError(f"invalid test2 image size: {path.name}: {image.size}")
        entries.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
        if index % 100 == 0:
            print(f"test2 manifest {index}/{len(files)}", flush=True)
    write_json(destination, {"root": str(TEST2), "count": len(files), "files": entries})
    return sha256(destination)


def smoke():
    if SMOKE_OUTPUT.exists():
        raise FileExistsError(SMOKE_OUTPUT)
    SMOKE_OUTPUT.mkdir(parents=True)
    model = load_model()
    path = sorted(TEST2.glob("*.png"))[0]
    image = np.asarray(Image.open(path).convert("RGB"))
    first = predict_probability_average(model, image).argmax(0).numpy().astype(np.uint8)
    second = predict_probability_average(model, image).argmax(0).numpy().astype(np.uint8)
    labels = np.unique(first)
    if not np.array_equal(first, second) or labels.min() < 1 or labels.max() > 8:
        raise RuntimeError("probability-average smoke failed")
    diagnostic = ROOT / "runs" / "strict" / "stage2_rebuilt_49_test2_ms2" / "predictions" / path.name
    disagreement = None
    if diagnostic.is_file():
        disagreement = float(np.mean(first != np.asarray(Image.open(diagnostic))))
    write_json(SMOKE_OUTPUT / "SMOKE_REPORT.json", {
        "status": "SMOKE_OK", "candidate": False, "image": path.name,
        "route": "eight_pass_probability_average", "passes": PASSES,
        "labels": labels.tolist(), "repeat_pixel_equal": True,
        "logit_average_diagnostic_disagreement": disagreement
    })
    print("PROBABILITY_AVERAGE_SMOKE_OK", flush=True)


def formal():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    shutil.copy2(Path(__file__).resolve(), OUTPUT / Path(__file__).name)
    files = sorted(TEST2.glob("*.png"))
    if len(files) != 1300:
        raise RuntimeError(f"test2 count mismatch: {len(files)}")
    manifest_path = OUTPUT / "test2_input_manifest.json"
    test2_manifest_sha = make_test2_manifest(files, manifest_path)
    model = load_model()
    prediction_dir = OUTPUT / "predictions"
    prediction_dir.mkdir()
    status_path = OUTPUT / "status.json"
    started = time.monotonic()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    class_counts = np.zeros(9, dtype=np.int64)
    records = []
    for index, path in enumerate(files, 1):
        image = np.asarray(Image.open(path).convert("RGB"))
        prediction = predict_probability_average(model, image).argmax(0).numpy().astype(np.uint8)
        labels = np.unique(prediction)
        if prediction.shape != (1024, 1024) or labels.min() < 1 or labels.max() > 8:
            raise RuntimeError(f"invalid prediction: {path.name}: {labels.tolist()}")
        output = prediction_dir / path.name
        temporary = output.with_suffix(".tmp.png")
        Image.fromarray(prediction, mode="L").save(temporary)
        os.replace(temporary, output)
        class_counts += np.bincount(prediction.ravel(), minlength=9)
        records.append({"name": path.name, "bytes": output.stat().st_size, "sha256": sha256(output)})
        if index % 25 == 0 or index == len(files):
            print(f"test2 probability-ms2 {index}/{len(files)}", flush=True)
            write_json(status_path, {"phase": "inference", "completed": index, "total": len(files),
                                     "submitted": False, "elapsed_seconds": time.monotonic() - started})

    determinism = []
    for index in (0, len(files) // 2, len(files) - 1):
        path = files[index]
        repeated = predict_probability_average(model, np.asarray(Image.open(path).convert("RGB"))).argmax(0).numpy().astype(np.uint8)
        equal = bool(np.array_equal(repeated, np.asarray(Image.open(prediction_dir / path.name))))
        determinism.append({"name": path.name, "pixel_equal": equal})
        if not equal:
            raise RuntimeError(f"determinism failure: {path.name}")

    names = {path.name for path in files}
    actual = {path.name for path in prediction_dir.glob("*.png")}
    if actual != names or len(actual) != 1300:
        raise RuntimeError("prediction count/name mismatch")
    total = int(class_counts.sum())
    distribution = {str(label): {"pixels": int(class_counts[label]),
                                 "fraction": float(class_counts[label] / total)}
                    for label in range(1, 9)}
    script_hash = sha256(OUTPUT / Path(__file__).name)
    prediction_manifest = OUTPUT / "predictions_manifest.json"
    write_json(prediction_manifest, {
        "count": 1300, "route": "eight_pass_probability_average", "passes": PASSES,
        "files": records, "class_distribution": distribution, "determinism": determinism,
        "stage49_checkpoint_sha256": STAGE49_SHA256,
        "inference_source_sha256": script_hash,
        "seg_swin_source_sha256": sha256(SOURCE / "seg_swin.py"),
        "kan_swin_source_sha256": sha256(SOURCE / "kan_swin.py"),
        "test2_input_manifest_sha256": test2_manifest_sha
    })

    archive = OUTPUT / "S2_rebuilt_49_test2_ms2_probability.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in files:
            handle.write(prediction_dir / path.name, arcname=path.name)
    with zipfile.ZipFile(archive) as handle:
        archive_names = handle.namelist()
        if (len(archive_names) != 1300 or set(archive_names) != names or
                any("/" in name or "\\" in name for name in archive_names) or
                handle.testzip() is not None):
            raise RuntimeError("ZIP validation failed")

    report = {
        "status": "complete_unsubmitted",
        "stage49": "local candidate",
        "stage49_checkpoint": str(STAGE49),
        "stage49_checkpoint_sha256": STAGE49_SHA256,
        "route": "eight_pass_probability_average",
        "prediction_count": 1300,
        "class_distribution": distribution,
        "determinism": determinism,
        "test2_input_manifest_sha256": test2_manifest_sha,
        "predictions_manifest_sha256": sha256(prediction_manifest),
        "inference_source_sha256": script_hash,
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "duration_seconds": time.monotonic() - started,
        "peak_cuda_allocated_mib": int(torch.cuda.max_memory_allocated() / 2**20),
        "test2_package": "generated_and_validated",
        "submitted": False,
        "real_result_score": None,
        "historical_69_96_is_reference_only": True,
        "supersedes_logit_average_diagnostic": str(ROOT / "runs" / "strict" / "stage2_rebuilt_49_test2_ms2")
    }
    write_json(OUTPUT / "FINAL_REPORT.json", report)
    write_json(status_path, {"phase": "complete", **report})
    print("PROBABILITY_AVERAGE_TEST2_COMPLETE", flush=True)


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
