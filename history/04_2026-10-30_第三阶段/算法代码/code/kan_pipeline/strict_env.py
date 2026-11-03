# -*- coding: utf-8 -*-
"""Hard safety boundary for the 2026-09-23 sealed reproduction.

This module must be imported before timm / Hugging Face code.  It pins every
cache and every model/data path to the archive root, verifies the sole public
weight, and rejects checkpoints that were not produced by this strict run.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ARCHIVE_DATA_ROOT = Path(__file__).resolve().parents[2]
REPRO_ROOT = Path(os.environ.get("STRICT_REPRO_ROOT", ARCHIVE_DATA_ROOT)).resolve()
SOURCE_ROOT = Path(__file__).resolve().parent
RUN_ROOT = Path(os.environ.get("STRICT_RUN_ROOT", REPRO_ROOT / "work" / "runs" / "strict")).resolve()
RUNTIME_ROOT = Path(os.environ.get("STRICT_RUNTIME_ROOT", REPRO_ROOT / "work" / "runtime_cache")).resolve()
PUBLIC_HF_ROOT = (REPRO_ROOT / "public_pretrained" / "huggingface").resolve()
PUBLIC_WEIGHT = (
    PUBLIC_HF_ROOT
    / "hub"
    / "models--timm--swin_base_patch4_window7_224.ms_in22k_ft_in1k"
    / "snapshots"
    / "a6a1eb2321b4f556fa0fa243fb777d47679f13c9"
    / "model.safetensors"
).resolve()
PUBLIC_WEIGHT_SHA256 = "c5d8c5ec47726ea2f4fd2b60954972ffe9d765f2c166812ffc2b1d61eef02cb8"
STRICT_MARKER = "sealed-reproduction-20260923-v1"

OFFICIAL_DATA_ROOT = (REPRO_ROOT / "official_data").resolve()
TRAIN_IMAGES = (OFFICIAL_DATA_ROOT / "train" / "images").resolve()
TRAIN_MASKS = (OFFICIAL_DATA_ROOT / "train" / "masks").resolve()
TEST1_IMAGES = (OFFICIAL_DATA_ROOT / "test_1" / "images").resolve()
TEST2_IMAGES = (OFFICIAL_DATA_ROOT / "test_2" / "images").resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def configure_process() -> None:
    """Force all runtime/model caches inside the portable archive workspace."""
    for directory in (RUN_ROOT, RUNTIME_ROOT, PUBLIC_HF_ROOT):
        directory.mkdir(parents=True, exist_ok=True)
    forced = {
        "STRICT_REPRO_ROOT": str(REPRO_ROOT),
        "HF_HOME": str(PUBLIC_HF_ROOT),
        "HF_HUB_CACHE": str(PUBLIC_HF_ROOT / "hub"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TORCH_HOME": str(RUNTIME_ROOT / "torch"),
        "XDG_CACHE_HOME": str(RUNTIME_ROOT / "xdg"),
        "CUDA_CACHE_PATH": str(RUNTIME_ROOT / "cuda")
    }
    for key, value in forced.items():
        os.environ[key] = value


def verify_inputs() -> dict[str, Any]:
    """Verify official counts, public hash, and derived-artifact hashes."""
    expected = {
        TRAIN_IMAGES: 6996,
        TRAIN_MASKS: 6996,
        TEST1_IMAGES: 500,
        TEST2_IMAGES: 1300
    }
    for directory, wanted in expected.items():
        actual = sum(1 for _ in directory.glob("*.png"))
        if actual != wanted:
            raise RuntimeError(f"official input count mismatch: {directory}: {actual} != {wanted}")
    if _sha256(PUBLIC_WEIGHT) != PUBLIC_WEIGHT_SHA256:
        raise RuntimeError("approved public Swin-B weight hash mismatch")
    manifest_path = REPRO_ROOT / "INPUT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["public_pretrained"]["sha256"] != PUBLIC_WEIGHT_SHA256:
        raise RuntimeError("INPUT_MANIFEST public hash mismatch")
    split = SOURCE_ROOT / "split.json"
    stats = SOURCE_ROOT / "train_mask_stats.npy"
    if _sha256(split) != manifest["derived"]["split_json_sha256"]:
        raise RuntimeError("split.json hash mismatch")
    if _sha256(stats) != manifest["derived"]["train_mask_stats_sha256"]:
        raise RuntimeError("train_mask_stats.npy hash mismatch")
    return manifest


def assert_run_path(path: str | os.PathLike[str], *, must_exist: bool = False) -> Path:
    resolved = Path(path).resolve()
    if not _inside(resolved, RUN_ROOT):
        raise RuntimeError(f"checkpoint/artifact path is outside strict run root: {resolved}")
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def strict_metadata(stage: str, seed: int, parents: list[str] | None = None) -> dict[str, Any]:
    return {
        "marker": STRICT_MARKER,
        "stage": stage,
        "seed": int(seed),
        "parents": list(parents or ["public_swin_b"]),
        "public_weight_sha256": PUBLIC_WEIGHT_SHA256,
        "input_manifest_sha256": _sha256(REPRO_ROOT / "INPUT_MANIFEST.json")
    }


def validate_checkpoint_payload(payload: dict[str, Any], path: str | os.PathLike[str]) -> None:
    meta = payload.get("strict_repro")
    if not isinstance(meta, dict) or meta.get("marker") != STRICT_MARKER:
        raise RuntimeError(f"checkpoint is not from this sealed reproduction: {path}")
    if meta.get("public_weight_sha256") != PUBLIC_WEIGHT_SHA256:
        raise RuntimeError(f"checkpoint public-weight lineage mismatch: {path}")
    current_manifest = _sha256(REPRO_ROOT / "INPUT_MANIFEST.json")
    if meta.get("input_manifest_sha256") != current_manifest:
        raise RuntimeError(f"checkpoint input-manifest lineage mismatch: {path}")


def load_checkpoint(path: str | os.PathLike[str], *, map_location: Any = "cpu") -> dict[str, Any]:
    import torch

    approved = assert_run_path(path, must_exist=True)
    payload = torch.load(approved, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid checkpoint payload: {approved}")
    validate_checkpoint_payload(payload, approved)
    return payload


configure_process()
