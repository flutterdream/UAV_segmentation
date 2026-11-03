# -*- coding: utf-8 -*-
"""Plain-EMA evaluator that writes one immutable JSON result per checkpoint/split."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os

import numpy as np

import strict_env as P
import seg_swin as S


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["swin", "kan"], required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n", type=int, choices=[100, 500], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-stage", required=True)
    parser.add_argument("--expected-iter", type=int, required=True)
    args = parser.parse_args()
    for value in (args.ckpt, args.output):
        if os.path.basename(value) != value:
            raise ValueError("checkpoint/output must be plain names inside strict run root")
    checkpoint = str(P.assert_run_path(P.RUN_ROOT / args.ckpt, must_exist=True))
    output = str(P.assert_run_path(P.RUN_ROOT / args.output))
    if os.path.exists(output):
        raise RuntimeError(f"refusing to overwrite evaluation result: {output}")
    payload = P.load_checkpoint(checkpoint)
    meta = payload["strict_repro"]
    if payload.get("iter") != args.expected_iter or meta.get("stage") != args.expected_stage:
        raise RuntimeError("checkpoint stage/iter does not match evaluation declaration")

    if args.model == "swin":
        model, iters = S.load_trained(use_ema=True, path=checkpoint)
    else:
        import kan_swin as K
        model, iters = K.load_trained(use_ema=True, path=checkpoint)
    val_ids = json.load(open(os.path.join(S.ROOT, "split.json"), encoding="utf-8"))["val_ids"]
    score, ious = S.evaluate(model, val_ids[:args.n], tta=False, bias=None)
    per_class = {
        name: (None if np.isnan(value) else round(float(value), 4))
        for name, value in zip(S.CLASS_NAMES, ious)
    }
    if not math.isfinite(score) or any(value is None for value in per_class.values()):
        raise RuntimeError("non-finite evaluation result")
    result = {
        "status": "local_evaluation_only",
        "model_type": args.model,
        "checkpoint": args.ckpt,
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_iter": iters,
        "strict_repro": meta,
        "split": f"seed123_fixed_validation_first{args.n}",
        "tta": False,
        "ema": True,
        "bias": False,
        "n": args.n,
        "val_mIoU": round(float(score), 4),
        "val_IoU": per_class
    }
    temporary = output + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, output)
    print("STRICT_EVAL_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

