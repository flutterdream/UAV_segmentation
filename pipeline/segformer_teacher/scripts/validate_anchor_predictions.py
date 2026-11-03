import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = sorted(args.prediction_dir.glob("*.png"))
    tests = sorted(args.test_dir.glob("*.png"))
    prediction_names = {path.name for path in predictions}
    test_names = {path.name for path in tests}
    if len(predictions) != 500 or prediction_names != test_names:
        raise RuntimeError(
            f"file-set mismatch: predictions={len(predictions)} tests={len(tests)} "
            f"missing={sorted(test_names-prediction_names)[:5]} extra={sorted(prediction_names-test_names)[:5]}"
        )

    counts = np.zeros(9, dtype=np.int64)
    file_hashes = {}
    dimensions = {}
    for path in predictions:
        array = np.asarray(Image.open(path))
        if array.ndim != 2:
            raise RuntimeError(f"not single-channel: {path}")
        values = np.unique(array)
        if values.size == 0 or values.min() < 1 or values.max() > 8:
            raise RuntimeError(f"labels outside 1..8 in {path}: {values.tolist()}")
        counts += np.bincount(array.ravel(), minlength=9)[:9]
        dimensions[path.name] = [int(array.shape[1]), int(array.shape[0])]
        file_hashes[path.name] = sha256(path)

    payload = {
        "status": "valid",
        "prediction_dir": str(args.prediction_dir.resolve()),
        "test_dir": str(args.test_dir.resolve()),
        "files": len(predictions),
        "allowed_labels": [1, 2, 3, 4, 5, 6, 7, 8],
        "class_pixels": {str(index): int(counts[index]) for index in range(1, 9)},
        "dimensions": dimensions,
        "sha256": file_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("status", "files", "class_pixels")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
