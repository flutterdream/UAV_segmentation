import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from contest_seg.data import SegmentationDataset, read_fold_rows
from contest_seg.inference import normalize_image
from contest_seg.labels import CLASS_NAMES, IGNORE_TRAIN_ID
from contest_seg.losses import CrossEntropyLovaszLoss
from contest_seg.model import build_model


class TargetPairDataset(Dataset):
    """Return aligned weak/strong views of an unlabeled target crop."""

    def __init__(self, image_dir, crop_size, allowed_filenames=None):
        allowed = set(allowed_filenames) if allowed_filenames is not None else None
        self.paths = sorted(
            path
            for path in Path(image_dir).iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
            and (allowed is None or path.name in allowed)
        )
        if not self.paths:
            raise ValueError(f"No target images found in {image_dir}")
        self.crop_size = crop_size

    def __len__(self):
        return len(self.paths)

    def _shared_geometry(self, image):
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        rotations = random.randrange(4)
        if rotations:
            image = image.rotate(90 * rotations)
        width, height = image.size
        if width < self.crop_size or height < self.crop_size:
            scale = max(self.crop_size / width, self.crop_size / height)
            image = image.resize(
                (round(width * scale), round(height * scale)),
                Image.Resampling.BILINEAR,
            )
            width, height = image.size
        left = random.randint(0, width - self.crop_size)
        top = random.randint(0, height - self.crop_size)
        return image.crop(
            (left, top, left + self.crop_size, top + self.crop_size)
        )

    @staticmethod
    def _strong_photometric(image):
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.6, 1.4))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.6, 1.4))
        image = ImageEnhance.Color(image).enhance(random.uniform(0.6, 1.4))
        gamma = random.uniform(0.7, 1.4)
        image = image.point(lambda value: 255 * ((value / 255) ** gamma))
        if random.random() < 0.2:
            image = ImageOps.autocontrast(image)
        if random.random() < 0.35:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 1.5)))
        elif random.random() < 0.25:
            image = ImageEnhance.Sharpness(image).enhance(random.uniform(1.2, 2.2))
        return image

    def __getitem__(self, index):
        path = self.paths[index % len(self.paths)]
        shared = self._shared_geometry(Image.open(path).convert("RGB"))
        weak = normalize_image(np.asarray(shared)).float()
        strong_image = self._strong_photometric(shared.copy())
        strong_array = np.asarray(strong_image, dtype=np.float32) / 255.0
        if random.random() < 0.35:
            noise = np.random.normal(
                0.0, random.uniform(0.005, 0.03), strong_array.shape
            )
            strong_array = np.clip(strong_array + noise, 0.0, 1.0)
        strong = normalize_image((strong_array * 255.0).astype(np.uint8)).float()
        if random.random() < 0.25:
            cutout_h = random.randint(self.crop_size // 16, self.crop_size // 5)
            cutout_w = random.randint(self.crop_size // 16, self.crop_size // 5)
            top = random.randint(0, self.crop_size - cutout_h)
            left = random.randint(0, self.crop_size - cutout_w)
            strong[:, top : top + cutout_h, left : left + cutout_w] = 0.0
        return weak, strong, path.name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=Path("outputs/segformer_b2_c5_full1024/final.pt"),
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/c6_uda"))
    parser.add_argument("--split-csv", type=Path, default=Path("artifacts/splits/folds.csv"))
    parser.add_argument(
        "--source-fold",
        type=int,
        default=-1,
        help="Exclude this fold from labeled source training; -1 uses every source image.",
    )
    parser.add_argument("--target-dir", type=Path, default=Path("test_1/images"))
    parser.add_argument(
        "--target-split-csv",
        type=Path,
        default=None,
        help="Optional split CSV used to restrict unlabeled target images.",
    )
    parser.add_argument("--target-fold", type=int, default=None)
    parser.add_argument("--sample-weights-csv", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--crop-size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--min-learning-rate", type=float, default=1e-7)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--teacher-decay", type=float, default=0.9995)
    parser.add_argument("--teacher-hflip", action="store_true")
    parser.add_argument(
        "--distribution-prior",
        default=None,
        help="Optional comma-separated target class prior for online distribution alignment.",
    )
    parser.add_argument(
        "--distribution-alignment-strength", type=float, default=0.0
    )
    parser.add_argument(
        "--distribution-alignment-momentum", type=float, default=0.99
    )
    parser.add_argument("--edge-refine-scale", type=float, default=1.0)
    parser.add_argument("--edge-gate-threshold", type=float, default=None)
    parser.add_argument("--target-loss-weight", type=float, default=0.5)
    parser.add_argument("--target-ramp-epochs", type=float, default=1.0)
    parser.add_argument("--lovasz-weight", type=float, default=0.3)
    parser.add_argument(
        "--class-weights", default="1,1,1,1,1.5,1,1,1"
    )
    parser.add_argument(
        "--target-class-weights", default="1,1,1,1,1,1,1,1"
    )
    parser.add_argument(
        "--pseudo-thresholds",
        default="0.994,0.90,0.991,0.90,0.994,0.90,0.973,0.949",
    )
    parser.add_argument("--rare-crop-probability", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--max-train-batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def classmix_mask(target):
    masks = []
    for sample in target:
        classes = torch.unique(sample)
        classes = classes[classes != IGNORE_TRAIN_ID]
        if classes.numel() == 0:
            masks.append(torch.zeros_like(sample, dtype=torch.bool))
            continue
        count = max(1, (classes.numel() + 1) // 2)
        selected = classes[torch.randperm(classes.numel(), device=classes.device)[:count]]
        masks.append(torch.isin(sample, selected))
    return torch.stack(masks)


def weighted_pseudo_ce(logits, target, valid, class_weights=None):
    loss_map = F.cross_entropy(
        logits,
        target,
        weight=class_weights,
        ignore_index=IGNORE_TRAIN_ID,
        reduction="none",
    )
    weights = valid.to(loss_map.dtype)
    return (loss_map * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def update_teacher(teacher, student, decay):
    for teacher_parameter, student_parameter in zip(
        teacher.parameters(), student.parameters()
    ):
        teacher_parameter.mul_(decay).add_(student_parameter.detach(), alpha=1.0 - decay)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        teacher_buffer.copy_(student_buffer)


def next_target_batch(iterator, loader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def compact_checkpoint(model, model_name, epoch, args, report):
    return {
        "model": model.state_dict(),
        "model_name": model_name,
        "epoch": epoch,
        "metrics": report,
        "args": vars(args),
        "teacher_ema": True,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_weights = [float(value) for value in args.class_weights.split(",")]
    target_class_weights = [
        float(value) for value in args.target_class_weights.split(",")
    ]
    thresholds = [float(value) for value in args.pseudo_thresholds.split(",")]
    if (
        len(class_weights) != 8
        or len(target_class_weights) != 8
        or len(thresholds) != 8
    ):
        raise ValueError("Class weights and pseudo thresholds must each have 8 values")
    threshold_tensor = torch.tensor(thresholds, device=device)
    target_class_weight_tensor = torch.tensor(
        target_class_weights, dtype=torch.float32, device=device
    )
    distribution_prior = None
    target_probability_ema = None
    if args.distribution_prior is not None:
        prior_values = [float(value) for value in args.distribution_prior.split(",")]
        if len(prior_values) != 8 or any(value <= 0 for value in prior_values):
            raise ValueError("--distribution-prior must contain 8 positive values")
        if not 0.0 <= args.distribution_alignment_strength <= 1.0:
            raise ValueError("--distribution-alignment-strength must be in [0, 1]")
        if not 0.0 <= args.distribution_alignment_momentum < 1.0:
            raise ValueError("--distribution-alignment-momentum must be in [0, 1)")
        distribution_prior = torch.tensor(
            prior_values, dtype=torch.float32, device=device
        )
        distribution_prior /= distribution_prior.sum()
        target_probability_ema = distribution_prior.clone()

    rows = read_fold_rows(root / args.split_csv, args.source_fold, training=True)
    source_dataset = SegmentationDataset(
        root,
        rows,
        crop_size=args.crop_size,
        training=True,
        rare_crop_probability=args.rare_crop_probability,
        strong_augmentation=True,
    )
    sampler = None
    if args.sample_weights_csv is not None:
        with open(root / args.sample_weights_csv, newline="", encoding="utf-8") as stream:
            lookup = {
                row["filename"]: float(row["weight"]) for row in csv.DictReader(stream)
            }
        sampler = WeightedRandomSampler(
            torch.tensor([lookup[row["filename"]] for row in rows], dtype=torch.double),
            num_samples=len(rows),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    source_loader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    target_filenames = None
    if args.target_split_csv is not None:
        if args.target_fold is None:
            raise ValueError("--target-fold is required with --target-split-csv")
        target_rows = read_fold_rows(
            root / args.target_split_csv, args.target_fold, training=False
        )
        target_filenames = [row["filename"] for row in target_rows]
    target_loader = DataLoader(
        TargetPairDataset(
            root / args.target_dir,
            args.crop_size,
            allowed_filenames=target_filenames,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )

    initial_path = root / args.initial_checkpoint
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    model_name = initial["model_name"]
    student = build_model(model_name, pretrained=False)
    student.load_state_dict(initial["model"])
    teacher = copy.deepcopy(student).eval()
    if args.gradient_checkpointing:
        if not hasattr(student, "gradient_checkpointing_enable"):
            raise ValueError(
                f"Model {model_name} does not support --gradient-checkpointing"
            )
        student.gradient_checkpointing_enable()
        print(json.dumps({"gradient_checkpointing": True}), flush=True)
    student.to(device)
    teacher.to(device)
    teacher.requires_grad_(False)
    for candidate in (student, teacher):
        if hasattr(candidate, "refine_scale"):
            candidate.refine_scale = args.edge_refine_scale
            candidate.refine_gate_threshold = args.edge_gate_threshold

    source_criterion = CrossEntropyLovaszLoss(
        lovasz_weight=args.lovasz_weight, class_weights=class_weights
    ).to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.min_learning_rate
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 1
    history = []
    if args.resume is not None:
        resume_path = root / args.resume
        resumed = torch.load(resume_path, map_location="cpu", weights_only=False)
        student.load_state_dict(resumed["student"])
        teacher.load_state_dict(resumed["teacher"])
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        scaler.load_state_dict(resumed["scaler"])
        if target_probability_ema is not None and "target_probability_ema" in resumed:
            target_probability_ema.copy_(
                resumed["target_probability_ema"].to(device=device)
            )
        start_epoch = int(resumed["epoch"]) + 1
        history_path = output_dir / "history.json"
        if history_path.exists():
            with open(history_path, encoding="utf-8") as stream:
                history = json.load(stream)
        print(
            json.dumps({"resumed_from": str(resume_path), "start_epoch": start_epoch}),
            flush=True,
        )
    else:
        print(json.dumps({"initialized_from": str(initial_path)}), flush=True)

    optimizer.zero_grad(set_to_none=True)
    global_step = (start_epoch - 1) * len(source_loader)
    total_ramp_steps = max(1, round(args.target_ramp_epochs * len(source_loader)))
    target_iterator = iter(target_loader)

    for epoch in range(start_epoch, args.epochs + 1):
        student.train()
        teacher.eval()
        source_running = 0.0
        mixed_running = 0.0
        predicted_counts = torch.zeros(8, dtype=torch.long, device=device)
        retained_counts = torch.zeros(8, dtype=torch.long, device=device)
        epoch_started = time.perf_counter()
        steps = 0
        for steps, (source_image, source_target, _) in enumerate(source_loader, start=1):
            (target_weak, target_strong, _), target_iterator = next_target_batch(
                target_iterator, target_loader
            )
            source_image = source_image.to(device, non_blocking=True)
            source_target = source_target.to(device, non_blocking=True)
            target_weak = target_weak.to(device, non_blocking=True)
            target_strong = target_strong.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                source_logits = student(source_image)
                source_loss = source_criterion(source_logits, source_target)
            scaler.scale(source_loss / args.gradient_accumulation).backward()

            with torch.no_grad(), torch.amp.autocast(
                "cuda", enabled=device.type == "cuda"
            ):
                teacher_logits = teacher(target_weak)
                if args.teacher_hflip:
                    flipped_logits = teacher(torch.flip(target_weak, dims=(-1,)))
                    teacher_logits = 0.5 * (
                        teacher_logits
                        + torch.flip(flipped_logits, dims=(-1,))
                    )
                probabilities = torch.softmax(teacher_logits.float(), dim=1)
                if target_probability_ema is not None:
                    batch_probability = probabilities.mean(dim=(0, 2, 3))
                    target_probability_ema.mul_(
                        args.distribution_alignment_momentum
                    ).add_(
                        batch_probability,
                        alpha=1.0 - args.distribution_alignment_momentum,
                    )
                    ratio = (distribution_prior / target_probability_ema.clamp_min(1e-6))
                    ratio = ratio.pow(args.distribution_alignment_strength).clamp_(
                        0.5, 2.0
                    )
                    probabilities = probabilities * ratio.view(1, -1, 1, 1)
                    probabilities /= probabilities.sum(dim=1, keepdim=True)
                confidence, pseudo_target = probabilities.max(dim=1)
                valid_pseudo = confidence >= threshold_tensor[pseudo_target]

            predicted_counts += torch.bincount(
                pseudo_target.flatten(), minlength=8
            )
            retained_counts += torch.bincount(
                pseudo_target[valid_pseudo].flatten(), minlength=8
            )
            mix_source = classmix_mask(source_target)
            mixed_image = torch.where(
                mix_source.unsqueeze(1), source_image, target_strong
            )
            mixed_target = torch.where(mix_source, source_target, pseudo_target)
            mixed_valid = torch.where(
                mix_source,
                source_target != IGNORE_TRAIN_ID,
                valid_pseudo,
            )
            mixed_target = mixed_target.masked_fill(~mixed_valid, IGNORE_TRAIN_ID)
            ramp = min(1.0, (global_step + 1) / total_ramp_steps)
            target_weight = args.target_loss_weight * ramp
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                mixed_logits = student(mixed_image)
                mixed_loss = weighted_pseudo_ce(
                    mixed_logits,
                    mixed_target,
                    mixed_valid,
                    target_class_weight_tensor,
                )
            scaler.scale(
                target_weight * mixed_loss / args.gradient_accumulation
            ).backward()

            if steps % args.gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_teacher(teacher, student, args.teacher_decay)
            global_step += 1
            source_running += float(source_loss.detach())
            mixed_running += float(mixed_loss.detach())

            if steps % args.log_interval == 0:
                elapsed = time.perf_counter() - epoch_started
                coverage = retained_counts.sum().float() / predicted_counts.sum().clamp_min(1)
                payload = {
                    "epoch": epoch,
                    "step": steps,
                    "steps_total": len(source_loader),
                    "source_loss": source_running / steps,
                    "mixed_loss": mixed_running / steps,
                    "pseudo_coverage": float(coverage),
                    "target_weight": target_weight,
                    "seconds_per_step": elapsed / steps,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                if device.type == "cuda":
                    payload["cuda_memory_gb"] = (
                        torch.cuda.max_memory_allocated() / 1024**3
                    )
                print(json.dumps(payload), flush=True)
            if args.max_train_batches and steps >= args.max_train_batches:
                break

        if steps % args.gradient_accumulation:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            update_teacher(teacher, student, args.teacher_decay)
        scheduler.step()

        predicted_cpu = predicted_counts.cpu().tolist()
        retained_cpu = retained_counts.cpu().tolist()
        report = {
            "epoch": epoch,
            "source_loss": source_running / max(steps, 1),
            "mixed_loss": mixed_running / max(steps, 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "pseudo_coverage": sum(retained_cpu) / max(sum(predicted_cpu), 1),
            "target_probability_ema": (
                target_probability_ema.cpu().tolist()
                if target_probability_ema is not None
                else None
            ),
            "classes": {
                str(class_id): {
                    "name": CLASS_NAMES[class_id],
                    "threshold": thresholds[class_id],
                    "predicted_pixels": predicted_cpu[class_id],
                    "retained_pixels": retained_cpu[class_id],
                    "retained_ratio": retained_cpu[class_id]
                    / max(predicted_cpu[class_id], 1),
                }
                for class_id in range(8)
            },
        }
        history.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        with open(output_dir / "history.json", "w", encoding="utf-8") as stream:
            json.dump(history, stream, ensure_ascii=False, indent=2)
        torch.save(
            {
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "model_name": model_name,
                "epoch": epoch,
                "args": vars(args),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "report": report,
                "target_probability_ema": (
                    target_probability_ema.cpu()
                    if target_probability_ema is not None
                    else None
                ),
            },
            output_dir / "last.pt",
        )
        torch.save(
            compact_checkpoint(teacher, model_name, epoch, args, report),
            output_dir / f"epoch_{epoch:02d}.pt",
        )

    if start_epoch <= args.epochs:
        torch.save(
            compact_checkpoint(teacher, model_name, args.epochs, args, history[-1]),
            output_dir / "final.pt",
        )


if __name__ == "__main__":
    main()
