"""Prepare the clean, auditable KAN restart workspace on E:\\复现.

This script derives every small data artifact from the official dataset.  It
does not copy or load any competition-trained checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image


PUBLIC_WEIGHT_SHA256 = "c5d8c5ec47726ea2f4fd2b60954972ffe9d765f2c166812ffc2b1d61eef02cb8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"E:\复现"))
    args = parser.parse_args()

    root = args.root.resolve()
    upstream = root / "workspace" / "source"
    image_dir = root / "train" / "images"
    mask_dir = root / "train" / "masks"
    test1_dir = root / "test_1" / "images"
    test2_dir = root / "test_2" / "images"
    public_weight = (
        root
        / "public_pretrained"
        / "huggingface"
        / "hub"
        / "models--timm--swin_base_patch4_window7_224.ms_in22k_ft_in1k"
        / "snapshots"
        / "a6a1eb2321b4f556fa0fa243fb777d47679f13c9"
        / "model.safetensors"
    )

    expected = {
        image_dir: 6996,
        mask_dir: 6996,
        test1_dir: 500,
        test2_dir: 1300
    }
    counts: dict[str, int] = {}
    for directory, wanted in expected.items():
        files = sorted(directory.glob("*.png"))
        counts[str(directory)] = len(files)
        if len(files) != wanted:
            raise RuntimeError(f"{directory}: expected {wanted} PNG files, found {len(files)}")

    public_hash = sha256(public_weight)
    if public_hash != PUBLIC_WEIGHT_SHA256:
        raise RuntimeError(f"public weight hash mismatch: {public_hash}")

    stats = np.zeros((6996, 9), dtype=np.float32)
    for index in range(6996):
        path = mask_dir / f"{index:04d}.png"
        mask = np.asarray(Image.open(path), dtype=np.uint8)
        if mask.shape != (1024, 1024) or mask.max() > 8:
            raise RuntimeError(f"invalid official mask: {path}")
        counts_i = np.bincount(mask.reshape(-1), minlength=9).astype(np.float64)
        if counts_i[1:].sum() <= 0:
            raise RuntimeError(f"mask has no labelled pixels: {path}")
        # Historical statistics use all pixels (including class-0 ignore)
        # as denominator. The training sampler later weights class 0 by 0.
        stats[index] = (counts_i / mask.size).astype(np.float32)
    np.save(upstream / "train_mask_stats.npy", stats)

    all_ids = list(range(6996))
    random.Random(123).shuffle(all_ids)
    split = {"val_ids": all_ids[:500]}
    save_json(upstream / "split.json", split)

    manifest = {
        "policy": "No competition-trained checkpoint is present in the clean upstream workspace.",
        "official_file_counts": counts,
        "split_seed": 123,
        "train_seed": 42,
        "public_pretrained": {
            "model": "timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k",
            "revision": "a6a1eb2321b4f556fa0fa243fb777d47679f13c9",
            "sha256": public_hash,
            "source_url": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth"
        },
        "derived": {
            "split_json_sha256": sha256(upstream / "split.json"),
            "train_mask_stats_sha256": sha256(upstream / "train_mask_stats.npy")
        },
        "strict_reproduction": (
            "Only official data and the academic public encoder weight are used. "
            "Historical teammate predictions are archived outside the clean training workspace and must not be read."
        )
    }
    save_json(root / "INPUT_MANIFEST.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
