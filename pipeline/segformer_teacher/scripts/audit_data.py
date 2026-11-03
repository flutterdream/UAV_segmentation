import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_NAMES = [
    "Ignore",
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Vegetation",
    "Agricultural",
    "Vehicle",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def image_metadata(paths):
    sizes = Counter()
    modes = Counter()
    for path in paths:
        with Image.open(path) as image:
            sizes[f"{image.width}x{image.height}"] += 1
            modes[image.mode] += 1
    return {"sizes": dict(sizes), "modes": dict(modes)}


def main():
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir or root / "artifacts" / "data_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_images = sorted((root / "train" / "images").glob("*.png"))
    train_masks = sorted((root / "train" / "masks").glob("*.png"))
    test_images = sorted((root / "test_1" / "images").glob("*.png"))
    image_names = {path.name for path in train_images}
    mask_names = {path.name for path in train_masks}
    if image_names != mask_names:
        raise RuntimeError(
            f"Train pairing mismatch: missing masks={sorted(image_names-mask_names)[:10]}, "
            f"missing images={sorted(mask_names-image_names)[:10]}"
        )

    total_pixels = np.zeros(9, dtype=np.int64)
    image_presence = np.zeros(9, dtype=np.int64)
    rows = []
    for index, mask_path in enumerate(train_masks, start=1):
        with Image.open(mask_path) as mask:
            array = np.asarray(mask.convert("L"), dtype=np.uint8)
        if array.shape != (1024, 1024):
            raise RuntimeError(f"Unexpected mask size: {mask_path} {array.shape}")
        values = np.unique(array)
        if np.any(values > 8):
            raise RuntimeError(f"Unexpected label IDs in {mask_path}: {values.tolist()}")
        counts = np.bincount(array.reshape(-1), minlength=9).astype(np.int64)
        total_pixels += counts
        image_presence += counts > 0
        row = {"filename": mask_path.name}
        row.update({f"pixels_{class_id}": int(counts[class_id]) for class_id in range(9)})
        row["dominant_class"] = int(np.argmax(counts[1:]) + 1)
        rows.append(row)
        if index % 500 == 0 or index == len(train_masks):
            print(f"processed_masks={index}/{len(train_masks)}", flush=True)

    fieldnames = ["filename"] + [f"pixels_{i}" for i in range(9)] + ["dominant_class"]
    with open(output_dir / "image_stats.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid_pixels = int(total_pixels[1:].sum())
    summary = {
        "train_images": len(train_images),
        "train_masks": len(train_masks),
        "test1_images": len(test_images),
        "paired_train_names": image_names == mask_names,
        "train_image_metadata": image_metadata(train_images),
        "train_mask_metadata": image_metadata(train_masks),
        "test1_image_metadata": image_metadata(test_images),
        "classes": {
            str(index): {
                "name": name,
                "pixel_count": int(total_pixels[index]),
                "pixel_ratio_valid": (
                    None if index == 0 else float(total_pixels[index] / valid_pixels)
                ),
                "images_present": int(image_presence[index]),
                "image_presence_ratio": float(image_presence[index] / len(train_masks)),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

