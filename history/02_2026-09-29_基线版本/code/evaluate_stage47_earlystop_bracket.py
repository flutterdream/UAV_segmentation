"""Audit and evaluate the five preregistered Stage47 early-stop checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


ROOT = Path(r"E:\复现")
PACKAGE = ROOT / "autodl_47_segformer_rebuilt"
SOURCE = PACKAGE / "source"
SCRIPTS = ROOT / "scripts"
REMOTE = (ROOT / "runs" / "strict" / "stage47_earlystop_remote_20260927" /
          "autodl_47_earlystop_bracket" / "runs")
OUTPUT = ROOT / "runs" / "strict" / "stage47_earlystop_eval_20260927"
PROVEN = (ROOT / "runs" / "strict" / "stage47_remote_20260926" / "artifacts" /
          "autodl_47_segformer_rebuilt" / "runs" /
          "ckpt_kan_stage47_segformer_rebuilt_i1000.pth")
PROVEN_SHA256 = "2508d7ed8c0af494505ce3ae11de4e396f42f365c1702db8e473acefcfe01f81"
ITERS = (600, 800, 1000, 1200, 1400)
CLASS_NAMES = ["Background", "Building", "Road", "Water", "Barren",
               "Vegetation", "Agricultural", "Vehicle"]

os.environ["STAGE47_PACKAGE_ROOT"] = str(PACKAGE)
os.environ["STRICT_REPRO_ROOT"] = str(ROOT)
os.environ["PYTHONPATH"] = str(SOURCE)
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SOURCE))

import run_stage49_rebuilt_and_test2 as B  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def checkpoint_path(iteration: int) -> Path:
    return REMOTE / f"ckpt_kan_stage47_earlystop_bracket_i{iteration}.pth"


def audit_payload(path: Path, iteration: int) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("strict_repro", {})
    config = payload.get("stage47_config", {})
    checks = {
        "iter": payload.get("iter") == iteration,
        "candidate": payload.get("candidate") is True,
        "stage": metadata.get("stage") == "kan_stage47_earlystop_bracket",
        "seed": metadata.get("seed") == 17,
        "parent": metadata.get("parents") == ["ckpt_kan_segformer_distill_rebuilt_i2000.pth"],
        "execution_iters": config.get("execution_iters") == 1400,
        "scheduler_horizon": config.get("scheduler_horizon") == 4000,
        "snapshot_iters": config.get("snapshot_iters") == [600, 800, 1000, 1200],
        "crop": config.get("crop") == 1024,
        "batch_size": config.get("batch_size") == 8,
        "source_supervised": config.get("source_supervised") == 0.20,
        "ema_decay": config.get("ema_decay") == 0.999,
        "has_ema": isinstance(payload.get("ema"), dict)
    }
    if not all(checks.values()):
        raise RuntimeError(f"lineage audit failed for i{iteration}: {checks}")
    return {"path": str(path), "sha256": sha256(path), "checks": checks}


def compare_i1000() -> dict:
    if sha256(PROVEN) != PROVEN_SHA256:
        raise RuntimeError("result-proven i1000 hash mismatch")
    old = torch.load(PROVEN, map_location="cpu", weights_only=False)["ema"]
    new = torch.load(checkpoint_path(1000), map_location="cpu", weights_only=False)["ema"]
    if set(old) != set(new):
        raise RuntimeError("i1000 EMA key mismatch")
    exact_tensors = 0
    float_tensors = 0
    max_abs = 0.0
    weighted_abs_sum = 0.0
    element_count = 0
    for name in old:
        left, right = old[name], new[name]
        if torch.equal(left, right):
            exact_tensors += 1
        if left.dtype.is_floating_point:
            float_tensors += 1
            difference = (left.float() - right.float()).abs()
            max_abs = max(max_abs, float(difference.max()))
            weighted_abs_sum += float(difference.sum())
            element_count += difference.numel()
    return {
        "proven_path": str(PROVEN),
        "proven_sha256": PROVEN_SHA256,
        "new_i1000_sha256": sha256(checkpoint_path(1000)),
        "tensor_count": len(old),
        "exact_tensor_count": exact_tensors,
        "floating_tensor_count": float_tensors,
        "max_abs_difference": max_abs,
        "mean_abs_difference": weighted_abs_sum / element_count,
        "role": "reproduction control only; the 70.30 checkpoint is immutable"
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audits = {str(iteration): audit_payload(checkpoint_path(iteration), iteration) for iteration in ITERS}
    i1000_control = compare_i1000()
    write_json(OUTPUT / "lineage_audit.json", {"checkpoints": audits, "i1000_control": i1000_control})

    ids = list(range(6996))
    random.Random(123).shuffle(ids)
    val_ids = ids[:500]
    results = []
    for iteration in ITERS:
        result_path = OUTPUT / f"eval_stage47_earlystop_i{iteration}_plain_ema_n500.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            results.append(result)
            print(f"REUSE i{iteration}: mIoU={result['miou']:.6f}", flush=True)
            continue
        started = time.monotonic()
        payload = torch.load(checkpoint_path(iteration), map_location="cpu", weights_only=False)
        model = B.K.DeepKANSegSwin(B.K.NUM_CLASSES)
        model.load_state_dict(payload["ema"], strict=True)
        model = model.to(B.K.DEVICE).eval().requires_grad_(False)
        miou, ious = B.S.evaluate(model, val_ids, tta=False, bias=None)
        values = [float(value) for value in ious]
        result = {
            "iteration": iteration,
            "route": "plain_ema",
            "validation_count": 500,
            "miou": float(miou),
            "class_iou": dict(zip(CLASS_NAMES, values)),
            "duration_seconds": time.monotonic() - started,
            "checkpoint_sha256": audits[str(iteration)]["sha256"],
            "finite_nonzero_all_classes": all(math.isfinite(value) and value > 0 for value in values),
            "result_prediction": False
        }
        write_json(result_path, result)
        results.append(result)
        print(f"EVAL i{iteration}: mIoU={miou:.6f}", flush=True)
        del model, payload
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    eligible = [result for result in results if result["finite_nonzero_all_classes"]]
    best = max(eligible, key=lambda result: (result["miou"], -result["iteration"]))
    proven_eval = json.loads((ROOT / "runs" / "strict" /
        "eval_kan_stage47_rebuilt_i1000_plain_ema_n500.json").read_text(encoding="utf-8"))
    # Older strict evaluations used val_IoU; newer ones use class_iou.
    reference_classes = proven_eval.get("class_iou", proven_eval.get("val_IoU"))
    if reference_classes is None:
        raise KeyError("reference evaluation has neither class_iou nor val_IoU")
    best_min_class_delta = min(
        best["class_iou"][name] - reference_classes[name] for name in CLASS_NAMES
    )
    gate = {
        "best_iteration": best["iteration"],
        "best_miou": best["miou"],
        "required_miou": 0.7526,
        "minimum_class_delta_vs_proven_i1000": best_min_class_delta,
        "required_minimum_class_delta": -0.01,
        "passes": best["miou"] >= 0.7526 and best_min_class_delta >= -0.01,
        "next_step_if_pass": "only the single winner may receive test2 BN0.25 and MS2 probability inference",
        "submitted": False,
        "real_result_score": None
    }
    summary = {"status": "complete", "results": results, "selection_gate": gate,
               "i1000_reproduction_control": i1000_control}
    write_json(OUTPUT / "FINAL_EVALUATION_SUMMARY.json", summary)
    print("FINAL=" + json.dumps(gate, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
