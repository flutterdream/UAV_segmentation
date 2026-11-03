import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from contest_seg.inference import normalize_image
from contest_seg.labels import train_to_raw
from contest_seg.model import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/baseline/best.pt"))
    parser.add_argument("--input-dir", type=Path, default=Path("test_1/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument(
        "--tta",
        choices=("none", "hflip", "flips", "d4", "hybrid_d4"),
        default="none",
        help="Fixed-weight geometric test-time augmentation.",
    )
    parser.add_argument(
        "--hybrid-d4-classes",
        default="0,1,2,5,6,7",
        help="Class channels blended from D4 when --tta=hybrid_d4.",
    )
    parser.add_argument(
        "--hybrid-d4-alpha",
        type=float,
        default=1.0,
        help="D4-minus-hflip blend strength when --tta=hybrid_d4.",
    )
    parser.add_argument(
        "--scales",
        default="1.0",
        help="Comma-separated fixed inference scales, for example 0.75,1.0,1.25.",
    )
    parser.add_argument(
        "--edge-refine-scale",
        type=float,
        default=1.0,
        help="Scale EdgeRefine residual logits; ignored by other models.",
    )
    parser.add_argument(
        "--edge-gate-threshold",
        type=float,
        default=None,
        help="Only apply EdgeRefine where the predicted boundary gate reaches this value.",
    )
    return parser.parse_args()


def positions(length, tile, overlap):
    if length <= tile:
        return [0]
    stride = tile - overlap
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


def tta_transforms(mode):
    if mode == "none":
        return [(0, False)]
    if mode == "hflip":
        return [(0, False), (0, True)]
    if mode == "flips":
        return [(0, False), (0, True), (2, False), (2, True)]
    if mode in ("d4", "hybrid_d4"):
        return [(rotation, mirrored) for rotation in range(4) for mirrored in (False, True)]
    raise ValueError(f"Unknown TTA mode: {mode}")


@torch.no_grad()
def tta_logits(
    model,
    tensor,
    mode,
    scales=(1.0,),
    hybrid_d4_classes=(0, 1, 2, 5, 6, 7),
    hybrid_d4_alpha=1.0,
):
    logits_sum = None
    baseline_sum = None
    transforms = tta_transforms(mode)
    original_size = tensor.shape[-2:]
    predictions = 0
    baseline_predictions = 0
    for scale in scales:
        if scale <= 0:
            raise ValueError(f"TTA scale must be positive, got {scale}")
        scaled_size = tuple(max(32, round(value * scale)) for value in original_size)
        scaled = (
            tensor
            if scaled_size == original_size
            else F.interpolate(
                tensor, size=scaled_size, mode="bilinear", align_corners=False
            )
        )
        for rotation, mirrored in transforms:
            augmented = torch.rot90(scaled, rotation, dims=(-2, -1))
            if mirrored:
                augmented = torch.flip(augmented, dims=(-1,))
            logits = model(augmented)
            if mirrored:
                logits = torch.flip(logits, dims=(-1,))
            logits = torch.rot90(logits, -rotation, dims=(-2, -1))
            if logits.shape[-2:] != original_size:
                logits = F.interpolate(
                    logits, size=original_size, mode="bilinear", align_corners=False
                )
            logits_sum = logits if logits_sum is None else logits_sum + logits
            predictions += 1
            if mode == "hybrid_d4" and rotation == 0:
                baseline_sum = (
                    logits.clone()
                    if baseline_sum is None
                    else baseline_sum.add_(logits)
                )
                baseline_predictions += 1
    averaged = logits_sum / predictions
    if mode != "hybrid_d4":
        return averaged
    baseline = baseline_sum / baseline_predictions
    hybrid = baseline.clone()
    hybrid[:, hybrid_d4_classes] += hybrid_d4_alpha * (
        averaged[:, hybrid_d4_classes] - baseline[:, hybrid_d4_classes]
    )
    return hybrid


@torch.no_grad()
def tiled_predict(
    model,
    image,
    device,
    tile_size,
    overlap,
    tta="none",
    scales=(1.0,),
    hybrid_d4_classes=(0, 1, 2, 5, 6, 7),
    hybrid_d4_alpha=1.0,
):
    width, height = image.size
    logits_sum = torch.zeros((8, height, width), dtype=torch.float32, device=device)
    counts = torch.zeros((1, height, width), dtype=torch.float32, device=device)
    for top in positions(height, tile_size, overlap):
        for left in positions(width, tile_size, overlap):
            tile = image.crop((left, top, left + tile_size, top + tile_size))
            tensor = normalize_image(np.asarray(tile)).unsqueeze(0).to(device)
            logits = tta_logits(
                model,
                tensor,
                tta,
                scales,
                hybrid_d4_classes,
                hybrid_d4_alpha,
            )[0]
            logits_sum[:, top : top + tile_size, left : left + tile_size] += logits
            counts[:, top : top + tile_size, left : left + tile_size] += 1
    return (logits_sum / counts).argmax(dim=0).cpu().numpy().astype(np.uint8)


def main():
    args = parse_args()
    root = args.root.resolve()
    checkpoint_path = root / args.checkpoint
    input_dir = root / args.input_dir
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_name"], pretrained=False)
    model.load_state_dict(checkpoint["model"])
    if hasattr(model, "refine_scale"):
        if args.edge_refine_scale < 0:
            raise ValueError("--edge-refine-scale must be non-negative")
        if args.edge_gate_threshold is not None and not (
            0.0 <= args.edge_gate_threshold <= 1.0
        ):
            raise ValueError("--edge-gate-threshold must be between 0 and 1")
        model.refine_scale = args.edge_refine_scale
        model.refine_gate_threshold = args.edge_gate_threshold
    model.to(device).eval()
    scales = tuple(float(value) for value in args.scales.split(","))
    if not scales or any(value <= 0 for value in scales):
        raise ValueError("--scales must contain positive comma-separated numbers")
    hybrid_d4_classes = tuple(
        int(value) for value in args.hybrid_d4_classes.split(",") if value.strip()
    )
    if not hybrid_d4_classes or any(
        value < 0 or value >= 8 for value in hybrid_d4_classes
    ):
        raise ValueError("--hybrid-d4-classes must contain class IDs between 0 and 7")
    if not 0.0 <= args.hybrid_d4_alpha <= 1.0:
        raise ValueError("--hybrid-d4-alpha must be between 0 and 1")
    paths = sorted(input_dir.glob("*.png"))
    for index, path in enumerate(paths, start=1):
        image = Image.open(path).convert("RGB")
        prediction = tiled_predict(
            model,
            image,
            device,
            args.tile_size,
            args.overlap,
            args.tta,
            scales,
            hybrid_d4_classes,
            args.hybrid_d4_alpha,
        )
        raw_prediction = train_to_raw(prediction).astype(np.uint8)
        Image.fromarray(raw_prediction).save(output_dir / path.name)
        if index % 50 == 0 or index == len(paths):
            print(f"predicted={index}/{len(paths)}", flush=True)


if __name__ == "__main__":
    main()
