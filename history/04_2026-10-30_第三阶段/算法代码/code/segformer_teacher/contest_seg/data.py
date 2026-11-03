import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import Dataset

from .labels import raw_to_train


def fourier_domain_adapt(source, target, beta):
    """Transfer low-frequency target appearance while preserving source geometry."""
    if source.shape != target.shape:
        raise ValueError(
            f"FDA source and target shapes must match: {source.shape} != {target.shape}"
        )
    if not 0.0 < beta < 0.5:
        raise ValueError(f"FDA beta must be between 0 and 0.5, got {beta}")

    source_fft = np.fft.fft2(source, axes=(0, 1))
    target_fft = np.fft.fft2(target, axes=(0, 1))
    source_amplitude = np.fft.fftshift(np.abs(source_fft), axes=(0, 1))
    target_amplitude = np.fft.fftshift(np.abs(target_fft), axes=(0, 1))
    source_phase = np.angle(source_fft)

    height, width = source.shape[:2]
    radius = max(1, int(round(min(height, width) * beta)))
    center_y, center_x = height // 2, width // 2
    y_slice = slice(center_y - radius, center_y + radius + 1)
    x_slice = slice(center_x - radius, center_x + radius + 1)
    source_amplitude[y_slice, x_slice] = target_amplitude[y_slice, x_slice]

    adapted_amplitude = np.fft.ifftshift(source_amplitude, axes=(0, 1))
    adapted_fft = adapted_amplitude * np.exp(1j * source_phase)
    adapted = np.real(np.fft.ifft2(adapted_fft, axes=(0, 1)))
    return np.clip(adapted, 0.0, 1.0).astype(np.float32)


def read_fold_rows(split_csv, fold, training):
    with open(split_csv, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if training:
        return [row for row in rows if int(row["fold"]) != fold]
    return [row for row in rows if int(row["fold"]) == fold]


class SegmentationDataset(Dataset):
    def __init__(
        self,
        root,
        rows,
        crop_size=512,
        training=False,
        samples_per_epoch=None,
        rare_crop_probability=0.0,
        strong_augmentation=False,
        image_dir="train/images",
        mask_dir="train/masks",
        require_valid_crop=False,
        target_image_dir=None,
        fda_probability=0.0,
        fda_beta_min=0.005,
        fda_beta_max=0.02,
        fda_match_csv=None,
    ):
        self.root = Path(root)
        self.rows = rows
        self.crop_size = crop_size
        self.training = training
        self.samples_per_epoch = samples_per_epoch
        self.rare_crop_probability = rare_crop_probability
        self.strong_augmentation = strong_augmentation
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.require_valid_crop = require_valid_crop
        self.target_image_dir = (
            Path(target_image_dir) if target_image_dir is not None else None
        )
        self.fda_probability = fda_probability
        self.fda_beta_min = fda_beta_min
        self.fda_beta_max = fda_beta_max
        self.fda_match_csv = Path(fda_match_csv) if fda_match_csv else None
        self.target_image_paths = []
        self.target_image_lookup = {}
        self.fda_matches = {}
        if self.fda_probability > 0.0:
            if self.target_image_dir is None:
                raise ValueError("FDA requires target_image_dir")
            if not 0.0 <= self.fda_probability <= 1.0:
                raise ValueError("FDA probability must be between 0 and 1")
            if not 0.0 < self.fda_beta_min <= self.fda_beta_max < 0.5:
                raise ValueError("Invalid FDA beta range")
            target_root = self.root / self.target_image_dir
            self.target_image_paths = sorted(
                path
                for path in target_root.iterdir()
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
            )
            if not self.target_image_paths:
                raise ValueError(f"No FDA target images found in {target_root}")
            self.target_image_lookup = {
                path.name: path for path in self.target_image_paths
            }
            if self.fda_match_csv is not None:
                match_path = (
                    self.fda_match_csv
                    if self.fda_match_csv.is_absolute()
                    else self.root / self.fda_match_csv
                )
                with open(match_path, newline="", encoding="utf-8") as stream:
                    for row in csv.DictReader(stream):
                        targets = [
                            row[key]
                            for key in sorted(row)
                            if key.startswith("target_") and row[key]
                        ]
                        missing = [
                            target
                            for target in targets
                            if target not in self.target_image_lookup
                        ]
                        if missing:
                            raise ValueError(
                                f"FDA match targets are missing from {target_root}: {missing[:5]}"
                            )
                        self.fda_matches[row["filename"]] = targets
                missing_sources = [
                    row["filename"]
                    for row in self.rows
                    if row["filename"] not in self.fda_matches
                ]
                if missing_sources:
                    raise ValueError(
                        f"FDA matches missing source images: {missing_sources[:5]}"
                    )

    def __len__(self):
        return self.samples_per_epoch or len(self.rows)

    def _augment(self, image, mask):
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        rotations = random.randrange(4)
        if rotations:
            image = image.rotate(90 * rotations)
            mask = mask.rotate(90 * rotations)
        if self.strong_augmentation:
            scale = random.uniform(0.75, 1.5)
            scaled_size = (
                max(self.crop_size, round(image.width * scale)),
                max(self.crop_size, round(image.height * scale)),
            )
            image = image.resize(scaled_size, Image.Resampling.BILINEAR)
            mask = mask.resize(scaled_size, Image.Resampling.NEAREST)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.65, 1.35))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.65, 1.35))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.65, 1.35))
            gamma = random.uniform(0.75, 1.35)
            image = image.point(lambda value: 255 * ((value / 255) ** gamma))
            if random.random() < 0.15:
                image = ImageOps.autocontrast(image)
            if random.random() < 0.2:
                image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 1.4)))
            elif random.random() < 0.2:
                image = ImageEnhance.Sharpness(image).enhance(random.uniform(1.2, 2.0))
        else:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.8, 1.2))
        return image, mask

    def _crop(self, image, mask):
        if self.crop_size is None:
            return image, mask
        width, height = image.size
        size = min(self.crop_size, width, height)
        if self.training:
            left = None
            top = None
            if random.random() < self.rare_crop_probability:
                mask_array = np.asarray(mask, dtype=np.uint8)
                present = [
                    class_id
                    for class_id in (8, 5, 7, 4)
                    if np.any(mask_array == class_id)
                ]
                if present:
                    priority = {5: 4, 7: 2, 8: 2, 4: 1}
                    weights = [priority[class_id] for class_id in present]
                    target_class = random.choices(present, weights=weights, k=1)[0]
                    ys, xs = np.where(mask_array == target_class)
                    selected = random.randrange(len(xs))
                    center_x, center_y = int(xs[selected]), int(ys[selected])
                    left = min(max(center_x - random.randint(0, size), 0), width - size)
                    top = min(max(center_y - random.randint(0, size), 0), height - size)
            if left is None:
                attempts = 10 if self.require_valid_crop else 1
                for _ in range(attempts):
                    left = random.randint(0, width - size)
                    top = random.randint(0, height - size)
                    if not self.require_valid_crop:
                        break
                    candidate = np.asarray(
                        mask.crop((left, top, left + size, top + size)),
                        dtype=np.uint8,
                    )
                    if np.any(candidate > 0):
                        break
        else:
            left = (width - size) // 2
            top = (height - size) // 2
        box = (left, top, left + size, top + size)
        return image.crop(box), mask.crop(box)

    def _load_fda_target(self, size, source_name):
        candidates = self.fda_matches.get(source_name)
        if candidates:
            ranks = list(range(len(candidates), 0, -1))
            target_name = random.choices(candidates, weights=ranks, k=1)[0]
            target_path = self.target_image_lookup[target_name]
        else:
            target_path = random.choice(self.target_image_paths)
        target = Image.open(target_path).convert("RGB")
        target_width, target_height = target.size
        width, height = size
        if target_width >= width and target_height >= height:
            left = random.randint(0, target_width - width)
            top = random.randint(0, target_height - height)
            target = target.crop((left, top, left + width, top + height))
        else:
            target = target.resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(target, dtype=np.float32) / 255.0

    def __getitem__(self, index):
        # A shortened epoch must sample the full source pool, not only its prefix.
        if self.training and self.samples_per_epoch is not None:
            index = random.randrange(len(self.rows))
        row = self.rows[index % len(self.rows)]
        name = row["filename"]
        image = Image.open(self.root / self.image_dir / name).convert("RGB")
        mask = Image.open(self.root / self.mask_dir / name).convert("L")
        if self.training:
            image, mask = self._augment(image, mask)
        image, mask = self._crop(image, mask)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        if (
            self.training
            and self.fda_probability > 0.0
            and random.random() < self.fda_probability
        ):
            target_array = self._load_fda_target(image.size, name)
            beta = random.uniform(self.fda_beta_min, self.fda_beta_max)
            image_array = fourier_domain_adapt(image_array, target_array, beta)
        if self.training and self.strong_augmentation and random.random() < 0.2:
            noise = np.random.normal(0.0, random.uniform(0.005, 0.025), image_array.shape)
            image_array = np.clip(image_array + noise, 0.0, 1.0)
        image_array = (image_array - np.array([0.485, 0.456, 0.406])) / np.array(
            [0.229, 0.224, 0.225]
        )
        mask_array = raw_to_train(np.asarray(mask, dtype=np.uint8))
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask_array.astype(np.int64))
        return image_tensor, mask_tensor, name
