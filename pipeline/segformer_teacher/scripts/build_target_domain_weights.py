import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/domain_features/dinov2_s.npz")
    )
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--blend", type=float, default=0.7)
    parser.add_argument("--min-weight", type=float, default=0.35)
    parser.add_argument("--max-weight", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/domain_features/train_weights.csv")
    )
    return parser.parse_args()


def spherical_kmeans(features, clusters, iterations, seed):
    rng = np.random.default_rng(seed)
    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    centers = features[rng.choice(len(features), clusters, replace=False)].copy()
    labels = np.zeros(len(features), dtype=np.int32)
    for iteration in range(iterations):
        new_labels = (features @ centers.T).argmax(axis=1)
        if iteration and np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster in range(clusters):
            members = features[labels == cluster]
            if len(members):
                center = members.mean(axis=0)
                centers[cluster] = center / max(np.linalg.norm(center), 1e-8)
            else:
                centers[cluster] = features[rng.integers(len(features))]
        print(f"domain_kmeans_iteration={iteration + 1}", flush=True)
    return labels


def main():
    args = parse_args()
    root = args.root.resolve()
    data = np.load(root / args.features)
    train_features = data["train_features"].astype(np.float32)
    test_features = data["test_features"].astype(np.float32)
    train_names = data["train_names"].astype(str)
    combined = np.concatenate([train_features, test_features], axis=0)
    labels = spherical_kmeans(combined, args.clusters, args.iterations, args.seed)
    train_labels = labels[: len(train_features)]
    test_labels = labels[len(train_features) :]
    train_counts = np.bincount(train_labels, minlength=args.clusters).astype(np.float64)
    test_counts = np.bincount(test_labels, minlength=args.clusters).astype(np.float64)
    train_density = (train_counts + 5.0) / (train_counts.sum() + 5.0 * args.clusters)
    test_density = (test_counts + 1.0) / (test_counts.sum() + args.clusters)
    density_ratio = test_density / train_density
    cluster_weights = 1.0 + args.blend * (density_ratio - 1.0)
    cluster_weights = np.clip(cluster_weights, args.min_weight, args.max_weight)
    weights = cluster_weights[train_labels]
    weights /= weights.mean()
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["filename", "weight", "cluster", "density_ratio"]
        )
        writer.writeheader()
        for name, weight, cluster in zip(train_names, weights, train_labels):
            writer.writerow(
                {
                    "filename": name,
                    "weight": f"{weight:.8f}",
                    "cluster": int(cluster),
                    "density_ratio": f"{density_ratio[cluster]:.8f}"
                }
            )
    effective_size = float(weights.sum() ** 2 / np.square(weights).sum())
    report = {
        "clusters": args.clusters,
        "blend": args.blend,
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
        "effective_sample_size": effective_size,
        "train_images": len(train_features),
        "test_images": len(test_features),
        "cluster_details": [
            {
                "cluster": cluster,
                "train_count": int(train_counts[cluster]),
                "test_count": int(test_counts[cluster]),
                "density_ratio": float(density_ratio[cluster]),
                "weight": float(cluster_weights[cluster])
            }
            for cluster in range(args.clusters)
        ]
    }
    with open(output.parent / "domain_weight_report.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
