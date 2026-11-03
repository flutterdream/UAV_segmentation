# -*- coding: utf-8 -*-
"""Frozen stage-47 trainer for the rebuilt SegFormer-anchor lineage.

Formal mode is locked to crop1024, true batch8, seed17, 4000 iterations,
restart_lr=.25, tau=.85, teacher hflip, anchor-conflict ignore,
source_supervised=.20 and EMA=.999.  Smoke mode changes only batch/iterations
and writes to a non-candidate smoke directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import segmentation_models_pytorch as smp

import strict_env as P
import seg_swin as S
import kan_swin as K


FORMAL = {
    "stage": "kan_stage47_segformer_rebuilt",
    "crop": 1024,
    "batch_size": 8,
    "seed": 17,
    "target_iters": 4000,
    "restart_lr": 0.25,
    "tau": 0.85,
    "teacher_hflip": True,
    "ignore_anchor_conflict": True,
    "source_supervised": 0.20,
    "ema_decay": 0.999,
    "full_image_scale_up": True,
    "scale_min": 0.75,
    "scale_max": 1.5,
    "final_inference": "ms2",
    "true_batch_required": True,
    "gradient_accumulation": 1
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FullImageScaleDataset(Dataset):
    def __init__(self, ids, image_dir: Path, mask_dir: Path, name_format: str, crop: int):
        self.ids = list(ids)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.name_format = name_format
        self.crop = crop

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        key = self.ids[index % len(self.ids)]
        name = self.name_format.format(k=key)
        image = np.asarray(Image.open(self.image_dir / name).convert("RGB"))
        mask = np.asarray(Image.open(self.mask_dir / name))
        if image.shape[:2] != mask.shape or mask.ndim != 2:
            raise RuntimeError(f"image/mask mismatch: {name}")
        crop = self.crop
        sampled = random.randint(int(crop * FORMAL["scale_min"]), int(crop * FORMAL["scale_max"]))
        height, width = mask.shape
        if sampled > min(height, width):
            # Dedicated stage-47 scale-up: enlarge the whole image first, then
            # take a true crop1024.  Earlier v4/v4.1/KAN stages cap at source size.
            ratio = sampled / min(height, width)
            resized = (max(crop, round(width * ratio)), max(crop, round(height * ratio)))
            image = np.asarray(Image.fromarray(image).resize(resized, Image.BILINEAR))
            mask = np.asarray(Image.fromarray(mask).resize(resized, Image.NEAREST))
            height, width = mask.shape
            top = random.randint(0, height - crop)
            left = random.randint(0, width - crop)
            image, mask = image[top:top + crop, left:left + crop], mask[top:top + crop, left:left + crop]
        else:
            top = random.randint(0, height - sampled)
            left = random.randint(0, width - sampled)
            image, mask = image[top:top + sampled, left:left + sampled], mask[top:top + sampled, left:left + sampled]
            if sampled != crop:
                image = np.asarray(Image.fromarray(image).resize((crop, crop), Image.BILINEAR))
                mask = np.asarray(Image.fromarray(mask).resize((crop, crop), Image.NEAREST))
        if random.random() < 0.5:
            image, mask = image[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:
            image, mask = image[::-1], mask[::-1]
        if random.random() < 0.25:
            turns = random.randint(1, 3)
            image, mask = np.rot90(image, turns).copy(), np.rot90(mask, turns).copy()
        if random.random() < 0.8:
            pil = Image.fromarray(image)
            pil = ImageEnhance.Brightness(pil).enhance(random.uniform(0.8, 1.2))
            pil = ImageEnhance.Contrast(pil).enhance(random.uniform(0.8, 1.2))
            pil = ImageEnhance.Color(pil).enhance(random.uniform(0.85, 1.15))
            image = np.asarray(pil)
        if random.random() < 0.1:
            image = np.asarray(Image.fromarray(image).convert("L").convert("RGB"))
        if random.random() < 0.1:
            image = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.0))))
        band = cv2.dilate(mask, np.ones((5, 5), np.uint8)) != cv2.erode(mask, np.ones((5, 5), np.uint8))
        return (
            S.norm(image),
            torch.from_numpy(mask.copy()).long(),
            torch.from_numpy(np.where(band, 3.0, 1.0).astype(np.float32)),
        )


def segmentation_loss(logits, labels, weight_map, dice, lovasz, class_weight):
    ce = F.cross_entropy(logits, labels, weight=class_weight, ignore_index=0, reduction="none")
    down_logits = F.interpolate(logits, scale_factor=0.5, mode="bilinear", align_corners=False)
    down_labels = F.interpolate(labels[:, None].float(), scale_factor=0.5, mode="nearest")[:, 0].long()
    return ((ce * weight_map).sum() / weight_map.sum() + dice(logits, labels)
            + 0.5 * lovasz(down_logits, down_labels))


@torch.no_grad()
def teacher_consensus(teacher, images, anchors):
    first = teacher(images).float().softmax(1)
    flipped = torch.flip(teacher(torch.flip(images, dims=(-1,))).float().softmax(1), dims=(-1,))
    probabilities = (first + flipped) * 0.5
    confidence, prediction = probabilities.max(1)
    keep = confidence.ge(FORMAL["tau"])
    if FORMAL["ignore_anchor_conflict"]:
        keep &= prediction.eq(anchors)
    labels = torch.where(keep, anchors, torch.zeros_like(anchors))
    return labels, int(keep.sum()), int(keep.numel())


def validate_formal_config(config: dict) -> None:
    for key, expected in FORMAL.items():
        if config.get(key) != expected:
            raise RuntimeError(f"formal config mismatch for {key}: {config.get(key)!r} != {expected!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    package = args.package_root.resolve()
    data = args.data_root.resolve()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_formal_config(config)
    batch_size = 1 if args.smoke else FORMAL["batch_size"]
    target_iters = 1 if args.smoke else FORMAL["target_iters"]
    output_dir = package / ("smoke" if args.smoke else "runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / ("stage47_smoke_iter1.pth" if args.smoke else "ckpt_kan_stage47_segformer_rebuilt.pth")
    if output.exists():
        raise FileExistsError(output)
    parent = package / "inputs" / "ckpt_kan_segformer_distill_rebuilt_i2000.pth"
    anchors = package / "inputs" / "teacher_anchors_segformer_historical_rebuilt"
    if not parent.is_file() or len(list(anchors.glob("*.png"))) != 500:
        raise RuntimeError("stage47 parent or 500 rebuilt C6 anchors are missing")
    payload = torch.load(parent, map_location="cpu", weights_only=False)
    metadata = payload.get("strict_repro", {})
    if payload.get("iter") != 2000 or metadata.get("stage") != "kan_segformer_distill_rebuilt" or payload.get("ema") is None:
        raise RuntimeError("stage47 parent lineage/EMA mismatch")
    random.seed(FORMAL["seed"]); np.random.seed(FORMAL["seed"]); torch.manual_seed(FORMAL["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = K.DeepKANSegSwin(K.NUM_CLASSES).to(device)
    teacher = K.DeepKANSegSwin(K.NUM_CLASSES).to(device)
    student.load_state_dict(payload["ema"]); teacher.load_state_dict(payload["ema"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    inner = getattr(student.encoder, "model", None)
    if inner is not None and hasattr(inner, "set_grad_checkpointing"):
        inner.set_grad_checkpointing(True)
    train_ids = list(range(6996)); random.Random(123).shuffle(train_ids); train_ids = train_ids[500:]
    target_ids = [path.stem for path in sorted(anchors.glob("*.png"))]
    source_ds = FullImageScaleDataset(train_ids, data / "train" / "images", data / "train" / "masks", "{k:04d}.png", 1024)
    target_ds = FullImageScaleDataset(target_ids, data / "test_1" / "images", anchors, "{k}.png", 1024)
    stats = np.load(package / "inputs" / "train_mask_stats.npy")
    boost = np.array([0, 1.2, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.5], dtype=np.float32)
    weights = (stats[train_ids] * boost).sum(1); weights /= weights.mean()
    source_sampler = WeightedRandomSampler(torch.from_numpy(weights).double(), len(train_ids) * 4, replacement=True)
    generator = torch.Generator().manual_seed(FORMAL["seed"])
    source_loader = DataLoader(source_ds, batch_size=batch_size, sampler=source_sampler, num_workers=0 if args.smoke else 4, drop_last=True)
    target_loader = DataLoader(target_ds, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0 if args.smoke else 4, drop_last=True)
    encoder_parameters = list(student.encoder.parameters())
    new_parameters = [parameter for module in (student.dfr_in, student.dfr_blocks, student.dfr_out, student.decoder) for parameter in module.parameters()]
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": 3e-5 * FORMAL["restart_lr"]},
        {"params": new_parameters, "lr": 1.5e-4 * FORMAL["restart_lr"]},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=target_iters, eta_min=3e-6)
    dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=0)
    lovasz = smp.losses.LovaszLoss(mode="multiclass", ignore_index=0)
    class_weight = torch.tensor([1.0] + list(S.CE_WEIGHTS), device=device)
    ema = {name: value.detach().clone() for name, value in student.state_dict().items()}
    ema_pairs = [(ema[name], value) for name, value in student.state_dict().items() if name in ema and value.dtype.is_floating_point]
    target_iterator, source_iterator = iter(target_loader), iter(source_loader)
    history, retained, total = [], 0, 0
    started = time.time(); student.train()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    for iteration in range(1, target_iters + 1):
        try: target_batch = next(target_iterator)
        except StopIteration: target_iterator = iter(target_loader); target_batch = next(target_iterator)
        try: source_batch = next(source_iterator)
        except StopIteration: source_iterator = iter(source_loader); source_batch = next(source_iterator)
        tx, anchor, tw = (value.to(device) for value in target_batch)
        sx, sy, sw = (value.to(device) for value in source_batch)
        filtered, kept, count = teacher_consensus(teacher, tx, anchor)
        retained += kept; total += count
        if kept == 0:
            raise RuntimeError("stage47 retained zero target pixels")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            target_loss = segmentation_loss(student(tx), filtered, tw, dice, lovasz, class_weight)
        target_loss.backward()
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            source_loss = segmentation_loss(student(sx), sy, sw, dice, lovasz, class_weight)
            weighted_source = FORMAL["source_supervised"] * source_loss
        weighted_source.backward()
        optimizer.step(); scheduler.step()
        with torch.no_grad():
            for ema_value, model_value in ema_pairs:
                ema_value.mul_(FORMAL["ema_decay"]).add_(model_value, alpha=1.0 - FORMAL["ema_decay"])
        loss_value = float(target_loss.detach() + weighted_source.detach())
        if not np.isfinite(loss_value): raise RuntimeError("non-finite stage47 loss")
        history.append(loss_value)
        if iteration % 10 == 0 or iteration == target_iters:
            print(f"iter={iteration}/{target_iters} loss={np.mean(history[-10:]):.5f} retained={retained/total:.4%}", flush=True)
        if not args.smoke and iteration % 1000 == 0:
            snapshot = output_dir / f"ckpt_kan_stage47_segformer_rebuilt_i{iteration}.pth"
            torch.save({"model": student.state_dict(), "ema": ema, "iter": iteration,
                        "strict_repro": P.strict_metadata(FORMAL["stage"], FORMAL["seed"], [parent.name]),
                        "stage47_config": FORMAL, "candidate": True}, snapshot)
    result = {"model": student.state_dict(), "ema": ema, "iter": target_iters,
              "strict_repro": P.strict_metadata(FORMAL["stage"] + ("_smoke" if args.smoke else ""), FORMAL["seed"], [parent.name]),
              "stage47_config": FORMAL, "candidate": not args.smoke,
              "smoke_overrides": ({"batch_size": 1, "target_iters": 1} if args.smoke else None),
              "retained_fraction": retained / total, "duration_seconds": time.time() - started,
              "peak_cuda_allocated_mib": (int(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None)}
    temporary = output.with_suffix(output.suffix + ".tmp"); torch.save(result, temporary); os.replace(temporary, output)
    print("STAGE47_RESULT=" + json.dumps({"output": str(output), "candidate": not args.smoke,
          "retained_fraction": result["retained_fraction"], "peak_mib": result["peak_cuda_allocated_mib"]}), flush=True)


if __name__ == "__main__":
    main()
