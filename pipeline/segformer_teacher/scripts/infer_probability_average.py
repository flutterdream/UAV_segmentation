"""Strict C5/C6 inference adapter requested by the 2026-09-24 protocol.

The frozen historical infer.py averages logits.  This adapter deliberately
keeps that file untouched and averages per-view softmax probabilities instead.
"""

import argparse
from pathlib import Path
import sys

# This script lives in <project>/scripts.  When Python executes it by absolute
# path, only the scripts directory is added to sys.path, so the sibling
# contest_seg package is otherwise invisible.  This changes import resolution
# only; inference logic and parameters remain untouched.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from contest_seg.inference import normalize_image
from contest_seg.labels import train_to_raw
from contest_seg.model import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--scales", default="0.75,1.0,1.25")
    parser.add_argument("--hflip", action="store_true")
    return parser.parse_args()


def resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def positions(length, tile, overlap):
    if length <= tile:
        return [0]
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile size")
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


@torch.no_grad()
def probability_tta(model, tensor, scales, hflip):
    original_size = tensor.shape[-2:]
    probability_sum = None
    count = 0
    for scale in scales:
        scaled_size = tuple(max(32, round(value * scale)) for value in original_size)
        scaled = (
            tensor
            if scaled_size == original_size
            else F.interpolate(tensor, size=scaled_size, mode="bilinear", align_corners=False)
        )
        for mirrored in ((False, True) if hflip else (False,)):
            augmented = torch.flip(scaled, dims=(-1,)) if mirrored else scaled
            logits = model(augmented)
            if mirrored:
                logits = torch.flip(logits, dims=(-1,))
            if logits.shape[-2:] != original_size:
                logits = F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)
            probabilities = logits.softmax(dim=1)
            probability_sum = probabilities if probability_sum is None else probability_sum.add_(probabilities)
            count += 1
    return probability_sum / count


@torch.no_grad()
def tiled_predict(model, image, device, tile_size, overlap, scales, hflip):
    width, height = image.size
    probability_sum = torch.zeros((8, height, width), dtype=torch.float32, device=device)
    counts = torch.zeros((1, height, width), dtype=torch.float32, device=device)
    for top in positions(height, tile_size, overlap):
        for left in positions(width, tile_size, overlap):
            tile = image.crop((left, top, left + tile_size, top + tile_size))
            tensor = normalize_image(np.asarray(tile)).unsqueeze(0).to(device)
            probabilities = probability_tta(model, tensor, scales, hflip)[0]
            probability_sum[:, top : top + tile_size, left : left + tile_size] += probabilities
            counts[:, top : top + tile_size, left : left + tile_size] += 1
    return (probability_sum / counts).argmax(dim=0).cpu().numpy().astype(np.uint8)


def main():
    args = parse_args()
    root = args.root.resolve()
    checkpoint_path = resolve(root, args.checkpoint).resolve()
    input_dir = resolve(root, args.input_dir).resolve()
    output_dir = resolve(root, args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    scales = tuple(float(value) for value in args.scales.split(","))
    if not scales or any(value <= 0 for value in scales):
        raise ValueError("scales must be positive")
    paths = sorted(input_dir.glob("*.png"))
    if len(paths) != 500:
        raise RuntimeError(f"expected 500 test1 PNG files, found {len(paths)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_name"], pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    for index, path in enumerate(paths, start=1):
        image = Image.open(path).convert("RGB")
        prediction = tiled_predict(
            model, image, device, args.tile_size, args.overlap, scales, args.hflip
        )
        raw_prediction = train_to_raw(prediction).astype(np.uint8)
        Image.fromarray(raw_prediction, mode="L").save(output_dir / path.name)
        if index % 25 == 0 or index == len(paths):
            print(f"predicted={index}/{len(paths)}", flush=True)


if __name__ == "__main__":
    main()
