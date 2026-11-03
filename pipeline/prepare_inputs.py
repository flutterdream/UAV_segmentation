"""Prepare the clean, auditable KAN restart workspace from this archive.

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
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify the archived derived artifacts without modifying the archive",
    )
    parser.add_argument("--rederive-check", action="store_true",
                        help="Recompute sampling statistics from all official masks without rewriting lineage files")
    args = parser.parse_args()
    if args.rederive_check and not args.verify_only:
        parser.error("--rederive-check requires --verify-only")

    root = args.root.resolve()
    upstream = root / "code" / "kan_pipeline"
    data_root = root / "official_data"
    image_dir = data_root / "train" / "images"
    mask_dir = data_root / "train" / "masks"
    test1_dir = data_root / "test_1" / "images"
    test2_dir = data_root / "test_2" / "images"
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
        counts[directory.relative_to(root).as_posix()] = len(files)
        if len(files) != wanted:
            raise RuntimeError(f"{directory}: expected {wanted} PNG files, found {len(files)}")

    public_hash = sha256(public_weight)
    if public_hash != PUBLIC_WEIGHT_SHA256:
        raise RuntimeError(f"public weight hash mismatch: {public_hash}")

    if args.verify_only:
        manifest_path = root / "INPUT_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Preserve the historical manifest bytes used in checkpoint lineage.
        # Its Windows path strings are labels only, never filesystem inputs.
        recorded_counts = {"official_data/" + "/".join(key.replace("\\", "/").split("/")[-2:]): value
                           for key, value in manifest.get("official_file_counts", {}).items()}
        if recorded_counts != counts:
            raise RuntimeError("official file counts do not match INPUT_MANIFEST.json")
        if manifest.get("public_pretrained", {}).get("sha256") != public_hash:
            raise RuntimeError("public weight does not match INPUT_MANIFEST.json")
        split_path = upstream / "split.json"
        stats_path = upstream / "train_mask_stats.npy"
        derived = manifest.get("derived", {})
        if sha256(split_path) != derived.get("split_json_sha256"):
            raise RuntimeError("archived split.json hash mismatch")
        if sha256(stats_path) != derived.get("train_mask_stats_sha256"):
            raise RuntimeError("archived train_mask_stats.npy hash mismatch")
        all_ids = list(range(6996))
        random.Random(123).shuffle(all_ids)
        split = json.loads(split_path.read_text(encoding="utf-8"))
        if split.get("val_ids") != all_ids[:500]:
            raise RuntimeError("split.json is not the fixed seed123 validation split")
        if args.rederive_check:
            recomputed = np.zeros((6996, 9), dtype=np.float32)
            for index in range(6996):
                path = mask_dir / f"{index:04d}.png"
                mask = np.asarray(Image.open(path), dtype=np.uint8)
                if mask.shape != (1024, 1024) or mask.max() > 8:
                    raise RuntimeError(f"invalid official mask: {path}")
                histogram = np.bincount(mask.reshape(-1), minlength=9).astype(np.float64)
                if histogram[1:].sum() <= 0:
                    raise RuntimeError(f"mask has no labelled pixels: {path}")
                recomputed[index] = (histogram / mask.size).astype(np.float32)
                if (index + 1) % 1000 == 0:
                    print(f"rederive masks {index+1}/6996", flush=True)
            if not np.array_equal(recomputed, np.load(stats_path)):
                raise RuntimeError("sampling statistics differ from official-mask recomputation")
        print(json.dumps({"status": "verified", "counts": counts,
                          "official_mask_statistics_rederived": args.rederive_check,
                          "fixed_split_rederived": True}, ensure_ascii=False, indent=2))
        return

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
