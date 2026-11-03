import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--stats-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--groups", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_stats(path):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    counts = np.array(
        [[float(row[f"pixels_{index}"]) for index in range(1, 9)] for row in rows],
        dtype=np.float64,
    )
    counts /= np.maximum(counts.sum(axis=1, keepdims=True), 1)
    return rows, counts


def visual_features(root, rows):
    features = []
    for index, row in enumerate(rows, start=1):
        path = root / "train" / "images" / row["filename"]
        with Image.open(path) as image:
            small = image.convert("RGB").resize((8, 8), Image.Resampling.BILINEAR)
            array = np.asarray(small, dtype=np.float32) / 255.0
        mean = array.mean(axis=(0, 1))
        std = array.std(axis=(0, 1))
        features.append(np.concatenate([array.reshape(-1), mean, std]))
        if index % 1000 == 0 or index == len(rows):
            print(f"embedded_images={index}/{len(rows)}", flush=True)
    return np.stack(features)


def kmeans(features, groups, seed, iterations=30):
    rng = np.random.default_rng(seed)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    normalized = (features - mean) / np.maximum(std, 1e-5)
    centers = normalized[rng.choice(len(normalized), groups, replace=False)].copy()
    labels = np.zeros(len(normalized), dtype=np.int32)
    for iteration in range(iterations):
        distances = (
            np.sum(normalized * normalized, axis=1, keepdims=True)
            - 2 * normalized @ centers.T
            + np.sum(centers * centers, axis=1)
        )
        new_labels = distances.argmin(axis=1)
        if iteration and np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for group in range(groups):
            members = normalized[labels == group]
            if len(members):
                centers[group] = members.mean(axis=0)
            else:
                centers[group] = normalized[rng.integers(len(normalized))]
        print(f"kmeans_iteration={iteration + 1}", flush=True)
    return labels


def assign_groups_to_folds(group_labels, class_ratios, folds):
    unique_groups = np.unique(group_labels)
    group_sizes = np.array([(group_labels == group).sum() for group in unique_groups])
    group_classes = np.array(
        [class_ratios[group_labels == group].sum(axis=0) for group in unique_groups]
    )
    target_size = len(group_labels) / folds
    target_classes = class_ratios.sum(axis=0) / folds
    fold_sizes = np.zeros(folds, dtype=np.float64)
    fold_classes = np.zeros((folds, 8), dtype=np.float64)
    group_to_fold = {}
    rarity = np.max(
        group_classes / np.maximum(class_ratios.sum(axis=0, keepdims=True), 1e-9), axis=1
    )
    order = np.lexsort((-rarity, -group_sizes))
    for position in order:
        group = unique_groups[position]
        size = group_sizes[position]
        classes = group_classes[position]
        best_fold = None
        best_cost = None
        for fold in range(folds):
            next_sizes = fold_sizes.copy()
            next_classes = fold_classes.copy()
            next_sizes[fold] += size
            next_classes[fold] += classes
            size_cost = np.mean(((next_sizes - target_size) / target_size) ** 2)
            class_cost = np.mean(
                ((next_classes - target_classes) / np.maximum(target_classes, 1e-9)) ** 2
            )
            cost = 0.35 * size_cost + 0.65 * class_cost
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_fold = fold
        group_to_fold[int(group)] = int(best_fold)
        fold_sizes[best_fold] += size
        fold_classes[best_fold] += classes
    folds_by_image = np.array([group_to_fold[int(group)] for group in group_labels])
    return folds_by_image, fold_sizes, fold_classes


def main():
    args = parse_args()
    root = args.root.resolve()
    stats_csv = args.stats_csv or root / "artifacts" / "data_audit" / "image_stats.csv"
    output_dir = args.output_dir or root / "artifacts" / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, class_ratios = load_stats(stats_csv)
    visual = visual_features(root, rows)
    features = np.concatenate([visual, class_ratios * 3.0], axis=1)
    groups = kmeans(features, args.groups, args.seed)
    folds, fold_sizes, fold_classes = assign_groups_to_folds(
        groups, class_ratios, args.folds
    )
    with open(output_dir / "folds.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["filename", "fold", "pseudo_group", "dominant_class"]
        )
        writer.writeheader()
        for row, fold, group in zip(rows, folds, groups):
            writer.writerow(
                {
                    "filename": row["filename"],
                    "fold": int(fold),
                    "pseudo_group": int(group),
                    "dominant_class": row["dominant_class"]
                }
            )
    report = {
        "seed": args.seed,
        "folds": args.folds,
        "pseudo_groups": args.groups,
        "fold_sizes": fold_sizes.astype(int).tolist(),
        "fold_class_share_sums": fold_classes.round(4).tolist(),
        "group_leakage": bool(
            any(len(set(folds[groups == group])) > 1 for group in np.unique(groups))
        )
    }
    with open(output_dir / "report.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

