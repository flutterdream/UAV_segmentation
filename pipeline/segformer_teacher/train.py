import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from contest_seg.data import SegmentationDataset, read_fold_rows
from contest_seg.losses import CrossEntropyLovaszLoss, EnhancedSegmentationLoss
from contest_seg.metrics import SegmentationMetrics
from contest_seg.model import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--model", default="simple_unet")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--pretrained-path", type=Path, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--train-all", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--val-tile-size", type=int, default=1024)
    parser.add_argument("--val-overlap", type=int, default=128)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument(
        "--baseline-miou",
        type=float,
        default=None,
        help="Known validation baseline that a fine-tune must beat.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many validated epochs without improving the baseline/best; 0 disables.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum mIoU gain required to reset early-stop patience.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help="Keep a supported pretrained backbone frozen for the first N epochs.",
    )
    parser.add_argument(
        "--unfreeze-backbone-last-blocks",
        type=int,
        default=0,
        help="After the freeze phase, train only the final N backbone blocks.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=None,
        help="Optional lower learning rate for parameters under model.backbone.",
    )
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--rare-crop-probability", type=float, default=0.0)
    parser.add_argument("--strong-augmentation", action="store_true")
    parser.add_argument("--fda-target-dir", type=Path, default=None)
    parser.add_argument("--fda-probability", type=float, default=0.0)
    parser.add_argument("--fda-beta-min", type=float, default=0.005)
    parser.add_argument("--fda-beta-max", type=float, default=0.02)
    parser.add_argument("--fda-match-csv", type=Path, default=None)
    parser.add_argument("--lovasz-weight", type=float, default=0.0)
    parser.add_argument(
        "--loss",
        choices=("ce_lovasz", "d2ls", "enhanced"),
        default="ce_lovasz",
    )
    parser.add_argument("--ce-weight", type=float, default=0.35)
    parser.add_argument("--dice-weight", type=float, default=0.20)
    parser.add_argument("--focal-weight", type=float, default=0.10)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.10)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--boundary-dilation", type=int, default=3)
    parser.add_argument("--aux-loss-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--class-weights",
        default=None,
        help="Comma-separated weights for the 8 train classes.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save a compact inference checkpoint every N epochs; 0 disables it.",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--stop-after-epoch", type=int, default=None,
                        help="Stop after saving this epoch; retain the full --epochs LR horizon")
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--pseudo-image-dir", type=Path, default=None)
    parser.add_argument("--pseudo-mask-dir", type=Path, default=None)
    parser.add_argument("--pseudo-copies", type=int, default=0)
    parser.add_argument("--sample-weights-csv", type=Path, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unpack_training_output(
    output, criterion, targets, args, auxiliary_criterion=None
):
    if not isinstance(output, (tuple, list)):
        return output, criterion(output, targets)

    logits = output[0]
    loss = criterion(logits, targets)
    auxiliary_logits = [
        value
        for value in output[1:]
        if torch.is_tensor(value) and value.ndim == 4
    ]
    contrastive_terms = [
        value
        for value in output[1:]
        if torch.is_tensor(value) and value.ndim == 0
    ]
    for auxiliary in auxiliary_logits:
        aux_loss = (auxiliary_criterion or criterion)(auxiliary, targets)
        loss = loss + args.aux_loss_weight * aux_loss
    for contrastive in contrastive_terms:
        loss = loss + args.contrastive_loss_weight * contrastive
    return logits, loss


class ModelEMA:
    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        model_state = model.state_dict()
        for name, ema_value in self.module.state_dict().items():
            model_value = model_state[name].detach()
            if ema_value.is_floating_point():
                ema_value.mul_(self.decay).add_(
                    model_value.to(dtype=ema_value.dtype), alpha=1.0 - self.decay
                )
            else:
                ema_value.copy_(model_value)


def tile_positions(length, tile, overlap):
    if length <= tile:
        return [0]
    stride = tile - overlap
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


@torch.no_grad()
def tiled_logits(model, image, tile_size, overlap):
    _, _, height, width = image.shape
    logits_sum = torch.zeros((1, 8, height, width), device=image.device)
    counts = torch.zeros((1, 1, height, width), device=image.device)
    for top in tile_positions(height, tile_size, overlap):
        for left in tile_positions(width, tile_size, overlap):
            tile = image[:, :, top : top + tile_size, left : left + tile_size]
            logits = model(tile)
            logits_sum[:, :, top : top + tile_size, left : left + tile_size] += logits
            counts[:, :, top : top + tile_size, left : left + tile_size] += 1
    return logits_sum / counts


@torch.no_grad()
def validate(model, loader, criterion, device, tile_size, overlap, max_batches=None):
    model.eval()
    metrics = SegmentationMetrics()
    total_loss = 0.0
    batches = 0
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = tiled_logits(model, images, tile_size, overlap)
        total_loss += criterion(logits, targets).item()
        metrics.update(logits, targets)
        batches += 1
        if max_batches and batches >= max_batches:
            break
    result = metrics.compute()
    result["loss"] = total_loss / max(batches, 1)
    return result


def main():
    args = parse_args()
    set_seed(args.seed)
    root = args.root.resolve()
    split_csv = args.split_csv or root / "artifacts" / "splits" / "folds.csv"
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.train_all:
        train_rows = read_fold_rows(split_csv, -1, training=True)
        val_rows = []
    else:
        train_rows = read_fold_rows(split_csv, args.fold, training=True)
        val_rows = read_fold_rows(split_csv, args.fold, training=False)
    if args.smoke_test:
        train_rows = train_rows[:8]
        val_rows = val_rows[:4]
        args.epochs = 1
        args.crop_size = min(args.crop_size, 256)
        args.batch_size = min(args.batch_size, 2)
        args.workers = 0
    real_train_dataset = SegmentationDataset(
        root,
        train_rows,
        crop_size=args.crop_size,
        training=True,
        samples_per_epoch=args.samples_per_epoch,
        rare_crop_probability=args.rare_crop_probability,
        strong_augmentation=args.strong_augmentation,
        target_image_dir=args.fda_target_dir,
        fda_probability=args.fda_probability,
        fda_beta_min=args.fda_beta_min,
        fda_beta_max=args.fda_beta_max,
        fda_match_csv=args.fda_match_csv,
    )
    train_dataset = real_train_dataset
    if args.pseudo_copies > 0:
        if args.pseudo_image_dir is None or args.pseudo_mask_dir is None:
            raise ValueError("Pseudo-label training requires both pseudo directories")
        pseudo_image_dir = root / args.pseudo_image_dir
        pseudo_mask_dir = root / args.pseudo_mask_dir
        pseudo_rows = [
            {"filename": path.name}
            for path in sorted(pseudo_mask_dir.glob("*.png"))
            if np.count_nonzero(np.asarray(Image.open(path).convert("L")))
            >= 0.05 * 1024**2
        ]
        if not pseudo_rows:
            raise ValueError(f"No pseudo masks found in {pseudo_mask_dir}")
        missing_images = [
            row["filename"]
            for row in pseudo_rows
            if not (pseudo_image_dir / row["filename"]).exists()
        ]
        if missing_images:
            raise ValueError(f"Missing pseudo images: {missing_images[:10]}")
        pseudo_dataset = SegmentationDataset(
            root,
            pseudo_rows,
            crop_size=args.crop_size,
            training=True,
            rare_crop_probability=args.rare_crop_probability,
            strong_augmentation=args.strong_augmentation,
            image_dir=args.pseudo_image_dir,
            mask_dir=args.pseudo_mask_dir,
            require_valid_crop=True,
        )
        train_dataset = ConcatDataset(
            [real_train_dataset] + [pseudo_dataset] * args.pseudo_copies
        )
    val_dataset = (
        SegmentationDataset(root, val_rows, crop_size=None, training=False)
        if val_rows
        else None
    )
    sampler = None
    if args.sample_weights_csv is not None:
        if args.pseudo_copies > 0:
            raise ValueError("Weighted source sampling cannot be mixed with pseudo copies")
        weights_path = (
            args.sample_weights_csv
            if args.sample_weights_csv.is_absolute()
            else root / args.sample_weights_csv
        )
        with open(weights_path, newline="", encoding="utf-8") as stream:
            weight_lookup = {
                row["filename"]: float(row["weight"]) for row in csv.DictReader(stream)
            }
        missing_weights = [
            row["filename"] for row in train_rows if row["filename"] not in weight_lookup
        ]
        if missing_weights:
            raise ValueError(f"Missing sample weights: {missing_weights[:10]}")
        sample_weights = torch.as_tensor(
            [weight_lookup[row["filename"]] for row in train_rows], dtype=torch.double
        )
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(real_train_dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=not args.smoke_test,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        if val_dataset is not None
        else None
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_source = args.pretrained
    if (
        args.pretrained_path is not None
        and args.resume is None
        and args.initial_checkpoint is None
    ):
        pretrained_source = (root / args.pretrained_path).resolve()
    model = build_model(args.model, pretrained=pretrained_source).to(device)
    phased_backbone = (
        args.freeze_backbone_epochs > 0
        or args.unfreeze_backbone_last_blocks > 0
    )
    if phased_backbone:
        if not hasattr(model, "set_backbone_trainable"):
            raise ValueError(
                f"Model {args.model} does not support phased backbone training"
            )
        initial_blocks = 0 if args.freeze_backbone_epochs > 0 else args.unfreeze_backbone_last_blocks
        model.set_backbone_trainable(initial_blocks)
        print(
            json.dumps(
                {
                    "backbone_trainable_last_blocks": initial_blocks,
                    "freeze_backbone_epochs": args.freeze_backbone_epochs,
                }
            ),
            flush=True,
        )
    if args.gradient_checkpointing:
        if not hasattr(model, "gradient_checkpointing_enable"):
            raise ValueError(
                f"Model {args.model} does not support --gradient-checkpointing"
            )
        model.gradient_checkpointing_enable()
        print(json.dumps({"gradient_checkpointing": True}), flush=True)
    if args.initial_checkpoint is not None:
        initial_path = (
            args.initial_checkpoint
            if args.initial_checkpoint.is_absolute()
            else root / args.initial_checkpoint
        )
        initial = torch.load(initial_path, map_location="cpu", weights_only=False)
        model.load_state_dict(initial["model"])
        print(
            json.dumps({"initialized_from": str(initial_path)}, ensure_ascii=False),
            flush=True,
        )
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    class_weights = None
    if args.class_weights:
        class_weights = [float(value) for value in args.class_weights.split(",")]
        if len(class_weights) != 8:
            raise ValueError("--class-weights must contain exactly 8 numbers")
    auxiliary_criterion = None
    if args.loss == "d2ls":
        if class_weights is not None:
            raise ValueError("The official D2LS loss does not support --class-weights")
        from third_party.D2LS.network.losses.dice import DiceLoss
        from third_party.D2LS.network.losses.joint_loss import JointLoss
        from third_party.D2LS.network.losses.soft_ce import SoftCrossEntropyLoss

        auxiliary_criterion = SoftCrossEntropyLoss(
            smooth_factor=0.05, ignore_index=255
        ).to(device)
        criterion = JointLoss(
            auxiliary_criterion,
            DiceLoss(smooth=0.05, ignore_index=255),
            1.0,
            1.0,
        ).to(device)
    elif args.loss == "enhanced":
        criterion = EnhancedSegmentationLoss(
            ce_weight=args.ce_weight,
            lovasz_weight=args.lovasz_weight,
            dice_weight=args.dice_weight,
            focal_weight=args.focal_weight,
            boundary_weight=args.boundary_loss_weight,
            focal_gamma=args.focal_gamma,
            boundary_dilation=args.boundary_dilation,
            class_weights=class_weights,
        ).to(device)
    else:
        criterion = CrossEntropyLovaszLoss(
            lovasz_weight=args.lovasz_weight, class_weights=class_weights
        ).to(device)
    validation_criterion = CrossEntropyLovaszLoss(lovasz_weight=0.0).to(device)
    if args.backbone_learning_rate is None:
        optimizer_parameters = model.parameters()
    else:
        backbone_parameters = []
        task_parameters = []
        for name, parameter in model.named_parameters():
            is_backbone = (
                name.startswith("backbone.")
                or name.startswith("model.segformer.")
                or name.startswith("base.model.segformer.")
            )
            if is_backbone:
                backbone_parameters.append(parameter)
            else:
                task_parameters.append(parameter)
        if not backbone_parameters:
            raise ValueError(
                "--backbone-learning-rate was set, but model.backbone has no parameters"
            )
        optimizer_parameters = [
            {"params": task_parameters, "lr": args.learning_rate},
            {"params": backbone_parameters, "lr": args.backbone_learning_rate},
        ]
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.min_learning_rate
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be non-negative")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early-stop-min-delta must be non-negative")
    best_miou = float(args.baseline_miou) if args.baseline_miou is not None else -1.0
    epochs_without_improvement = 0
    history = []
    start_epoch = 1
    if args.resume is not None:
        resume_path = args.resume if args.resume.is_absolute() else root / args.resume
        resumed = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        scaler.load_state_dict(resumed["scaler"])
        if ema is not None:
            if "model_ema" in resumed:
                ema.module.load_state_dict(resumed["model_ema"])
            else:
                ema.module.load_state_dict(resumed["model"])
        start_epoch = int(resumed["epoch"]) + 1
        best_miou = float(resumed.get("best_miou", -1.0))
        epochs_without_improvement = int(
            resumed.get("epochs_without_improvement", 0)
        )
        history_path = output_dir / "history.json"
        if history_path.exists():
            with open(history_path, encoding="utf-8") as stream:
                history = json.load(stream)
        print(
            json.dumps(
                {"resumed_from": str(resume_path), "start_epoch": start_epoch},
                ensure_ascii=False,
            ),
            flush=True,
        )
    for epoch in range(start_epoch, args.epochs + 1):
        if phased_backbone:
            trainable_blocks = (
                0
                if epoch <= args.freeze_backbone_epochs
                else args.unfreeze_backbone_last_blocks
            )
            model.set_backbone_trainable(trainable_blocks)
            if epoch == start_epoch or epoch == args.freeze_backbone_epochs + 1:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "backbone_trainable_last_blocks": trainable_blocks,
                            "trainable_parameters": sum(
                                parameter.numel()
                                for parameter in model.parameters()
                                if parameter.requires_grad
                            ),
                        }
                    ),
                    flush=True,
                )
        model.train()
        running_loss = 0.0
        epoch_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for step, (images, targets, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                output = model(images)
                logits, loss = unpack_training_output(
                    output, criterion, targets, args, auxiliary_criterion
                )
                loss = loss / args.gradient_accumulation
            scaler.scale(loss).backward()
            if step % args.gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * args.gradient_accumulation
            if step % args.log_interval == 0:
                elapsed = time.perf_counter() - epoch_started
                payload = {
                    "epoch": epoch,
                    "step": step,
                    "steps_total": len(train_loader),
                    "train_loss_running": running_loss / step,
                    "seconds_per_step": elapsed / step,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                if device.type == "cuda":
                    payload["cuda_memory_gb"] = (
                        torch.cuda.max_memory_allocated() / 1024**3
                    )
                print(json.dumps(payload, ensure_ascii=False), flush=True)
            if args.smoke_test and step >= 2:
                break
            if args.max_train_batches and step >= args.max_train_batches:
                break
        if step % args.gradient_accumulation:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        should_validate = (
            val_loader is not None
            and (
                args.smoke_test
                or epoch % args.validate_every == 0
                or epoch == args.epochs
            )
        )
        if should_validate:
            evaluation_model = ema.module if ema is not None else model
            result = validate(
                evaluation_model,
                val_loader,
                validation_criterion,
                device,
                min(args.val_tile_size, 256) if args.smoke_test else args.val_tile_size,
                args.val_overlap,
                max_batches=2 if args.smoke_test else args.max_val_batches,
            )
        else:
            result = {"miou": None, "per_class_iou": None, "loss": None}
        result.update(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(step, 1),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        history.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        previous_best = best_miou
        is_best = result["miou"] is not None and result["miou"] > best_miou
        if is_best:
            best_miou = result["miou"]
        if result["miou"] is not None:
            if result["miou"] > previous_best + args.early_stop_min_delta:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        should_early_stop = (
            args.early_stop_patience > 0
            and result["miou"] is not None
            and epochs_without_improvement >= args.early_stop_patience
        )
        checkpoint = {
            "model": model.state_dict(),
            "model_name": args.model,
            "epoch": epoch,
            "metrics": result,
            "args": vars(args),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_miou": best_miou,
            "epochs_without_improvement": epochs_without_improvement,
        }
        if ema is not None:
            checkpoint["model_ema"] = ema.module.state_dict()
        torch.save(checkpoint, output_dir / "last.pt")
        if args.stop_after_epoch is not None and epoch == args.stop_after_epoch:
            print(json.dumps({"controlled_historical_stop": epoch}), flush=True)
            break
        if is_best:
            best_checkpoint = dict(checkpoint)
            if ema is not None:
                best_checkpoint["model"] = checkpoint["model_ema"]
            torch.save(best_checkpoint, output_dir / "best.pt")
        if args.save_every > 0 and epoch % args.save_every == 0:
            epoch_checkpoint = {
                "model": (
                    ema.module.state_dict() if ema is not None else model.state_dict()
                ),
                "model_name": args.model,
                "epoch": epoch,
                "metrics": result,
                "args": vars(args),
                "ema_decay": args.ema_decay,
            }
            torch.save(epoch_checkpoint, output_dir / f"epoch_{epoch:02d}.pt")
        if epoch == args.epochs or should_early_stop:
            final_checkpoint = {
                "model": (
                    ema.module.state_dict() if ema is not None else model.state_dict()
                ),
                "model_name": args.model,
                "epoch": epoch,
                "metrics": result,
                "args": vars(args),
                "ema_decay": args.ema_decay,
            }
            torch.save(final_checkpoint, output_dir / "final.pt")
        if should_early_stop:
            print(
                json.dumps(
                    {
                        "early_stopped": True,
                        "epoch": epoch,
                        "best_miou": best_miou,
                        "epochs_without_improvement": epochs_without_improvement,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            break
    with open(output_dir / "history.json", "w", encoding="utf-8") as stream:
        json.dump(history, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
