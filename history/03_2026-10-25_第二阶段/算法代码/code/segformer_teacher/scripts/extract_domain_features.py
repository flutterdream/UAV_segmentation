import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import Dinov2Model


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ImageFeatureDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB").resize(
            (256, 256), Image.Resampling.BICUBIC
        )
        image = image.crop((16, 16, 240, 240))
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - MEAN) / STD
        return torch.from_numpy(array.transpose(2, 0, 1)).float(), path.name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--model-dir", type=Path, default=Path("pretrained/dinov2-small")
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/domain_features/dinov2_s.npz")
    )
    return parser.parse_args()


def extract(model, loader, device, label):
    features = []
    names = []
    processed = 0
    with torch.no_grad():
        for images, batch_names in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                hidden = model(pixel_values=images).last_hidden_state
                cls_feature = hidden[:, 0]
                patch_feature = hidden[:, 1:].mean(dim=1)
                feature = torch.cat([cls_feature, patch_feature], dim=1)
                feature = torch.nn.functional.normalize(feature, dim=1)
            features.append(feature.float().cpu().numpy())
            names.extend(batch_names)
            processed += len(batch_names)
            if processed % 512 < len(batch_names) or processed == len(loader.dataset):
                print(f"{label}_features={processed}/{len(loader.dataset)}", flush=True)
    return np.concatenate(features), np.asarray(names)


def main():
    args = parse_args()
    root = args.root.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Dinov2Model.from_pretrained(root / args.model_dir).to(device).eval()
    train_paths = sorted((root / "train/images").glob("*.png"))
    test_paths = sorted((root / "test_1/images").glob("*.png"))
    train_loader = DataLoader(
        ImageFeatureDataset(train_paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        ImageFeatureDataset(test_paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    train_features, train_names = extract(model, train_loader, device, "train")
    test_features, test_names = extract(model, test_loader, device, "test")
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        train_features=train_features,
        train_names=train_names,
        test_features=test_features,
        test_names=test_names,
    )
    print(f"saved={output} feature_dim={train_features.shape[1]}")


if __name__ == "__main__":
    main()
