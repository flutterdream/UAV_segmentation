# -*- coding: utf-8 -*-
"""Create the immutable stage-47 AutoDL bundle and run a one-iteration local smoke."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

REPRO = Path(r"E:\复现")
SOURCE = REPRO / "workspace" / "source"
STRICT = REPRO / "runs" / "strict"
PYTHON = Path(r"D:\Anaconda3\python.exe")

def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""): value.update(chunk)
    return value.hexdigest()

def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text, encoding="utf-8", newline="\n")

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--package", type=Path, required=True); args = parser.parse_args()
    package = args.package.resolve()
    if package.exists(): raise FileExistsError(package)
    parent = STRICT / "ckpt_kan_segformer_distill_rebuilt_i2000.pth"
    anchors = STRICT / "teacher_anchors_segformer_historical_rebuilt"
    anchor_manifest = STRICT / "teacher_anchors_segformer_historical_rebuilt_MANIFEST.json"
    if not parent.is_file() or not anchor_manifest.is_file() or len(list(anchors.glob("*.png"))) != 500:
        raise RuntimeError("approved distill parent or rebuilt C6 anchors missing")
    import torch
    payload = torch.load(parent, map_location="cpu", weights_only=False)
    if payload.get("iter") != 2000 or payload.get("strict_repro", {}).get("stage") != "kan_segformer_distill_rebuilt" or payload.get("ema") is None:
        raise RuntimeError("distill parent lineage mismatch")
    for name in ("source", "inputs", "configs", "scripts", "env", "docs", "logs", "runs", "smoke"):
        (package / name).mkdir(parents=True, exist_ok=False)
    shutil.copy2(SOURCE / "stage47_train_template.py", package / "source" / "train_stage47.py")
    shutil.copy2(SOURCE / "seg_swin.py", package / "source" / "seg_swin.py")
    kan_text = (SOURCE / "kan_swin.py").read_text(encoding="utf-8")
    old = 'encoder_weights="imagenet", in_channels=3, classes=num_classes,'
    if old not in kan_text: raise RuntimeError("cannot make portable KAN constructor")
    write(package / "source" / "kan_swin.py", kan_text.replace(old, 'encoder_weights=None, in_channels=3, classes=num_classes,', 1))
    shutil.copy2(SOURCE / "strict_env_autodl47_template.py", package / "source" / "strict_env.py")
    shutil.copy2(parent, package / "inputs" / parent.name)
    shutil.copytree(anchors, package / "inputs" / anchors.name)
    shutil.copy2(anchor_manifest, package / "inputs" / anchor_manifest.name)
    shutil.copy2(REPRO / "INPUT_MANIFEST.json", package / "inputs" / "INPUT_MANIFEST.json")
    shutil.copy2(SOURCE / "train_mask_stats.npy", package / "inputs" / "train_mask_stats.npy")
    shutil.copy2(SOURCE / "split.json", package / "inputs" / "split.json")
    lineage = {"parent": parent.name, "parent_sha256": sha256(parent), "parent_iter": 2000,
               "parent_stage": "kan_segformer_distill_rebuilt", "parent_ema": True,
               "anchor_manifest_sha256": sha256(anchor_manifest),
               "input_manifest_sha256": payload["strict_repro"]["input_manifest_sha256"]}
    write(package / "inputs" / "lineage.json", json.dumps(lineage, ensure_ascii=False, indent=2))
    config = {"stage": "kan_stage47_segformer_rebuilt", "crop": 1024, "batch_size": 8, "seed": 17,
              "target_iters": 4000, "restart_lr": 0.25, "tau": 0.85, "teacher_hflip": True,
              "ignore_anchor_conflict": True, "source_supervised": 0.20, "ema_decay": 0.999,
              "full_image_scale_up": True, "scale_min": 0.75, "scale_max": 1.5,
              "final_inference": "ms2", "true_batch_required": True, "gradient_accumulation": 1}
    write(package / "configs" / "stage47_formal.json", json.dumps(config, ensure_ascii=False, indent=2))
    shutil.copy2(REPRO / "requirements.strict.lock.txt", package / "env" / "requirements.strict.lock.txt")
    write(package / "scripts" / "run_stage47.sh", """#!/usr/bin/env bash
set -euo pipefail
PKG="$(cd "$(dirname "$0")/.." && pwd)"
export STAGE47_PACKAGE_ROOT="$PKG"
export STRICT_REPRO_ROOT="${STRICT_REPRO_ROOT:?set STRICT_REPRO_ROOT to uploaded official-data root}"
export PYTHONPATH="$PKG/source"
python -u "$PKG/source/train_stage47.py" --package-root "$PKG" --data-root "$STRICT_REPRO_ROOT" --config "$PKG/configs/stage47_formal.json" 2>&1 | tee "$PKG/logs/stage47_formal.log"
""")
    write(package / "scripts" / "shutdown_after_success.sh", """#!/usr/bin/env bash
set -euo pipefail
# Run manually only after downloading and verifying every item in DOWNLOAD_CHECKLIST.md.
test -f "$(dirname "$0")/../runs/ckpt_kan_stage47_segformer_rebuilt.pth"
sudo shutdown -h now
""")
    write(package / "docs" / "DATA_LAYOUT.md", """# Data layout

Set `STRICT_REPRO_ROOT` to a directory containing only the official `train/images`, `train/masks`, `test_1/images` (and later `test_2/images`) trees plus the approved public cache if needed. The stage-47 bundle already contains the newly trained distill-i2000 parent and rebuilt C6 hard anchors. Do not add historical weights, predictions, pseudo-labels or D-drive artifacts.
""")
    write(package / "docs" / "RESOURCE_AND_COST.md", """# Resource estimate

- Formal shape: crop1024, true batch8, no gradient accumulation substitute.
- Expected peak VRAM: about 19–20 GiB; use a 32 GiB GPU instance.
- Expected formal training time: approximately 1.5–3 hours, to be revised from the first 100 cloud iterations.
- AutoDL's public price page checked on 2026-09-26 lists RTX 5090 32GB at CNY 2.78/hour (member discount may differ): https://www.autodl.com/home
- At 1.5–3 training hours, the compute estimate is CNY 4.17–8.34; reserve CNY 5.56–11.12 for 2–4 total instance hours including setup/download. Confirm the live instance price and availability with the user before starting because they change.
""")
    write(package / "DOWNLOAD_CHECKLIST.md", """# Download checklist

After formal stage47 completes, download and verify:

- `runs/ckpt_kan_stage47_segformer_rebuilt.pth`
- `runs/ckpt_kan_stage47_segformer_rebuilt_i1000.pth`
- `runs/ckpt_kan_stage47_segformer_rebuilt_i2000.pth`
- `runs/ckpt_kan_stage47_segformer_rebuilt_i3000.pth`
- `runs/ckpt_kan_stage47_segformer_rebuilt_i4000.pth`
- `logs/stage47_formal.log`
- the post-run SHA-256 manifest generated before any optional shutdown

Do not run 49 or test2 on the cloud unless separately authorized.
""")
    environment = os.environ.copy(); environment.update({"STAGE47_PACKAGE_ROOT": str(package), "STRICT_REPRO_ROOT": str(REPRO), "PYTHONPATH": str(package / "source")})
    smoke_stdout = package / "logs" / "stage47_smoke.stdout.log"; smoke_stderr = package / "logs" / "stage47_smoke.stderr.log"
    started = time.time()
    with smoke_stdout.open("w", encoding="utf-8") as out, smoke_stderr.open("w", encoding="utf-8") as err:
        process = subprocess.run([str(PYTHON), "-u", str(package / "source" / "train_stage47.py"),
             "--package-root", str(package), "--data-root", str(REPRO), "--config", str(package / "configs" / "stage47_formal.json"), "--smoke"],
             cwd=str(package), env=environment, stdout=out, stderr=err)
    if process.returncode != 0: raise RuntimeError(f"stage47 smoke failed; see {smoke_stderr}")
    smoke_ckpt = package / "smoke" / "stage47_smoke_iter1.pth"
    smoke = torch.load(smoke_ckpt, map_location="cpu", weights_only=False)
    if smoke.get("candidate") is not False or smoke.get("iter") != 1: raise RuntimeError("invalid smoke checkpoint")
    smoke_report = {"status": "passed", "candidate": False, "iterations": 1, "batch_size": 1,
                    "formal_config_unchanged": True, "duration_seconds": time.time() - started,
                    "peak_cuda_allocated_mib": smoke.get("peak_cuda_allocated_mib"),
                    "retained_fraction": smoke.get("retained_fraction"), "checkpoint_sha256": sha256(smoke_ckpt)}
    write(package / "smoke" / "SMOKE_REPORT.json", json.dumps(smoke_report, ensure_ascii=False, indent=2))
    files = {}
    total = 0
    for path in sorted(p for p in package.rglob("*") if p.is_file() and p.name != "MANIFEST_SHA256.json"):
        relative = path.relative_to(package).as_posix(); files[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size}; total += path.stat().st_size
    manifest = {"status": "prepared_local_only", "formal_stage47_not_started": True, "files": files,
                "total_bytes_excluding_manifest": total, "smoke": smoke_report,
                "boundaries": {"autodl_started": False, "stage47_formal_started": False, "stage49_started": False,
                               "test2_inference": False, "package_output": False}}
    write(package / "MANIFEST_SHA256.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print("AUTODL47_PACKAGE_RESULT=" + json.dumps({"path": str(package), "bytes": total, "files": len(files), "smoke": smoke_report}, ensure_ascii=False), flush=True)

if __name__ == "__main__": main()
