# -*- coding: utf-8 -*-
"""Portable sealed environment used only inside the stage-47 bundle."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path

PACKAGE_ROOT = Path(os.environ["STAGE47_PACKAGE_ROOT"]).resolve()
REPRO_ROOT = Path(os.environ["STRICT_REPRO_ROOT"]).resolve()
SOURCE_ROOT = (PACKAGE_ROOT / "source").resolve()
RUN_ROOT = (PACKAGE_ROOT / "runs").resolve()
RUNTIME_ROOT = (PACKAGE_ROOT / "runtime_cache").resolve()
PUBLIC_HF_ROOT = Path(os.environ.get("HF_HOME", str(REPRO_ROOT / "public_pretrained" / "huggingface"))).resolve()
PUBLIC_WEIGHT_SHA256 = "c5d8c5ec47726ea2f4fd2b60954972ffe9d765f2c166812ffc2b1d61eef02cb8"
STRICT_MARKER = "sealed-reproduction-20260923-v1"
TRAIN_IMAGES = (REPRO_ROOT / "train" / "images").resolve()
TRAIN_MASKS = (REPRO_ROOT / "train" / "masks").resolve()
TEST1_IMAGES = (REPRO_ROOT / "test_1" / "images").resolve()
TEST2_IMAGES = (REPRO_ROOT / "test_2" / "images").resolve()

def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""): value.update(chunk)
    return value.hexdigest()

def configure_process():
    RUN_ROOT.mkdir(parents=True, exist_ok=True); RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ.update({"HF_HOME": str(PUBLIC_HF_ROOT), "HF_HUB_CACHE": str(PUBLIC_HF_ROOT / "hub"),
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "TORCH_HOME": str(RUNTIME_ROOT / "torch"), "XDG_CACHE_HOME": str(RUNTIME_ROOT / "xdg"),
                       "CUDA_CACHE_PATH": str(RUNTIME_ROOT / "cuda")})

def verify_inputs():
    expected = {TRAIN_IMAGES: 6996, TRAIN_MASKS: 6996, TEST1_IMAGES: 500}
    for directory, count in expected.items():
        if len(list(directory.glob("*.png"))) != count: raise RuntimeError(f"official input count mismatch: {directory}")
    return json.loads((PACKAGE_ROOT / "inputs" / "INPUT_MANIFEST.json").read_text(encoding="utf-8"))

def assert_run_path(path, must_exist=False):
    resolved = Path(path).resolve()
    try: resolved.relative_to(RUN_ROOT)
    except ValueError: raise RuntimeError(f"artifact outside stage47 run root: {resolved}")
    if must_exist and not resolved.is_file(): raise FileNotFoundError(resolved)
    return resolved

def strict_metadata(stage, seed, parents=None):
    manifest_hash = json.loads((PACKAGE_ROOT / "inputs" / "lineage.json").read_text(encoding="utf-8"))["input_manifest_sha256"]
    return {"marker": STRICT_MARKER, "stage": stage, "seed": int(seed), "parents": list(parents or []),
            "public_weight_sha256": PUBLIC_WEIGHT_SHA256, "input_manifest_sha256": manifest_hash}

def load_checkpoint(path, map_location="cpu"):
    import torch
    approved = assert_run_path(path, must_exist=True)
    payload = torch.load(approved, map_location=map_location, weights_only=False)
    if payload.get("strict_repro", {}).get("marker") != STRICT_MARKER: raise RuntimeError("invalid strict checkpoint")
    return payload

configure_process()
