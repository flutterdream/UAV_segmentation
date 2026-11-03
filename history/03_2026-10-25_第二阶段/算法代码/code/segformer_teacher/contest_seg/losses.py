import torch
from torch import nn
from torch.nn import functional as F

from .labels import IGNORE_TRAIN_ID


def _lovasz_grad(sorted_ground_truth):
    count = len(sorted_ground_truth)
    intersection = sorted_ground_truth.sum() - sorted_ground_truth.float().cumsum(0)
    union = (
        sorted_ground_truth.sum()
        + (1.0 - sorted_ground_truth).float().cumsum(0)
    )
    gradient = 1.0 - intersection / union
    if count > 1:
        gradient[1:count] = gradient[1:count] - gradient[:-1].clone()
    return gradient


def lovasz_softmax(logits, target, ignore_index=IGNORE_TRAIN_ID):
    probabilities = F.softmax(logits, dim=1)
    probabilities = probabilities.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
    target = target.reshape(-1)
    valid = target != ignore_index
    probabilities = probabilities[valid]
    target = target[valid]
    if target.numel() == 0:
        return logits.sum() * 0.0
    losses = []
    for class_id in range(logits.shape[1]):
        foreground = (target == class_id).float()
        if foreground.sum() == 0:
            continue
        errors = (foreground - probabilities[:, class_id]).abs()
        errors_sorted, permutation = torch.sort(errors, descending=True)
        foreground_sorted = foreground[permutation]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(foreground_sorted)))
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def soft_dice_loss(logits, target, ignore_index=IGNORE_TRAIN_ID, smooth=1.0):
    """Class-balanced soft Dice averaged over classes present in the batch."""
    valid = target != ignore_index
    if not torch.any(valid):
        return logits.sum() * 0.0
    safe_target = target.masked_fill(~valid, 0)
    one_hot = F.one_hot(safe_target, num_classes=logits.shape[1])
    one_hot = one_hot.permute(0, 3, 1, 2).to(dtype=logits.dtype)
    valid_mask = valid[:, None].to(dtype=logits.dtype)
    one_hot = one_hot * valid_mask
    probabilities = F.softmax(logits, dim=1) * valid_mask
    spatial_dims = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dim=spatial_dims)
    denominator = probabilities.sum(dim=spatial_dims) + one_hot.sum(dim=spatial_dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    present = one_hot.sum(dim=spatial_dims) > 0
    return (1.0 - dice[present]).mean()


def focal_cross_entropy(
    logits,
    target,
    gamma=2.0,
    class_weights=None,
    ignore_index=IGNORE_TRAIN_ID,
):
    valid = target != ignore_index
    if not torch.any(valid):
        return logits.sum() * 0.0
    loss_map = F.cross_entropy(
        logits,
        target,
        weight=class_weights,
        ignore_index=ignore_index,
        reduction="none",
    )
    safe_target = target.masked_fill(~valid, 0)
    true_probability = F.softmax(logits, dim=1).gather(
        1, safe_target[:, None]
    ).squeeze(1)
    modulation = (1.0 - true_probability).pow(gamma)
    return (loss_map[valid] * modulation[valid]).mean()


def target_boundary_mask(target, ignore_index=IGNORE_TRAIN_ID, dilation=3):
    """Mark both sides of valid semantic boundaries, optionally dilated."""
    valid = target != ignore_index
    boundary = torch.zeros_like(valid)
    vertical = valid[:, 1:] & valid[:, :-1] & (target[:, 1:] != target[:, :-1])
    horizontal = valid[:, :, 1:] & valid[:, :, :-1] & (
        target[:, :, 1:] != target[:, :, :-1]
    )
    boundary[:, 1:] |= vertical
    boundary[:, :-1] |= vertical
    boundary[:, :, 1:] |= horizontal
    boundary[:, :, :-1] |= horizontal
    if dilation > 1:
        if dilation % 2 == 0:
            raise ValueError("Boundary dilation must be odd")
        boundary = F.max_pool2d(
            boundary[:, None].float(),
            kernel_size=dilation,
            stride=1,
            padding=dilation // 2,
        ).squeeze(1) > 0
    return boundary & valid


class EnhancedSegmentationLoss(nn.Module):
    """IoU-, class-, hard-pixel-, and boundary-aware segmentation objective."""

    def __init__(
        self,
        ce_weight=0.35,
        lovasz_weight=0.25,
        dice_weight=0.20,
        focal_weight=0.10,
        boundary_weight=0.10,
        focal_gamma=2.0,
        boundary_dilation=3,
        class_weights=None,
    ):
        super().__init__()
        weights = (ce_weight, lovasz_weight, dice_weight, focal_weight, boundary_weight)
        if any(value < 0 for value in weights) or sum(weights) <= 0:
            raise ValueError("Enhanced loss weights must be non-negative and not all zero")
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.focal_gamma = focal_gamma
        self.boundary_dilation = boundary_dilation
        self.normalizer = sum(weights)
        self.register_buffer(
            "class_weights",
            None if class_weights is None else torch.as_tensor(class_weights).float(),
        )

    def forward(self, logits, target):
        valid = target != IGNORE_TRAIN_ID
        if not torch.any(valid):
            return logits.sum() * 0.0
        total = logits.sum() * 0.0
        ce_map = None
        if self.ce_weight > 0 or self.boundary_weight > 0:
            ce_map = F.cross_entropy(
                logits,
                target,
                weight=self.class_weights,
                ignore_index=IGNORE_TRAIN_ID,
                reduction="none",
            )
        if self.ce_weight > 0:
            total = total + self.ce_weight * ce_map[valid].mean()
        if self.lovasz_weight > 0:
            total = total + self.lovasz_weight * lovasz_softmax(logits, target)
        if self.dice_weight > 0:
            total = total + self.dice_weight * soft_dice_loss(logits, target)
        if self.focal_weight > 0:
            focal = focal_cross_entropy(
                logits,
                target,
                gamma=self.focal_gamma,
                class_weights=self.class_weights,
            )
            total = total + self.focal_weight * focal
        if self.boundary_weight > 0:
            boundary = target_boundary_mask(
                target, dilation=self.boundary_dilation
            )
            boundary_ce = (
                ce_map[boundary].mean()
                if torch.any(boundary)
                else logits.sum() * 0.0
            )
            total = total + self.boundary_weight * boundary_ce
        return total / self.normalizer


class CrossEntropyLovaszLoss(nn.Module):
    def __init__(self, lovasz_weight=0.3, class_weights=None):
        super().__init__()
        self.lovasz_weight = lovasz_weight
        self.register_buffer(
            "class_weights",
            None if class_weights is None else torch.as_tensor(class_weights).float(),
        )

    def forward(self, logits, target):
        if not torch.any(target != IGNORE_TRAIN_ID):
            return logits.sum() * 0.0
        cross_entropy = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            ignore_index=IGNORE_TRAIN_ID,
        )
        if self.lovasz_weight <= 0:
            return cross_entropy
        lovasz = lovasz_softmax(logits, target)
        return (1.0 - self.lovasz_weight) * cross_entropy + self.lovasz_weight * lovasz
