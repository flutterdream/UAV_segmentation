# -*- coding: utf-8 -*-
"""One-shot daytime handoff: wait for rebuilt SegFormer, then fixed PL -> fixed anchor training."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime

import numpy as np
from PIL import Image


REPRO = Path(r"E:\复现")
SOURCE = REPRO / "workspace" / "source"
STRICT = REPRO / "runs" / "strict"
STRICT_LOGS = STRICT / "logs"
TEACHER_RUN = REPRO / "runs" / "segformer_teacher_historical"
TEACHER_STATUS = TEACHER_RUN / "queue_status.json"
IDENTITY_REPORT = TEACHER_RUN / "audits" / "historical_anchor_identity.json"
PYTHON = Path(r"D:\Anaconda3\python.exe")

STATUS = STRICT / "segformer_pl_distill_handoff_status.json"
REPORT = STRICT / "SEGFORMER_PL_DISTILL_STATUS_20260925.md"
MASTER_LOG = STRICT_LOGS / "segformer_pl_distill_handoff.log"

BASE_NAME = "ckpt_kan_from_v4_1_7439_16k.pth"
BASE_PATH = STRICT / BASE_NAME
BASE_SHA = "806039602f21cb6da342085f867727712a9c06f40c8765c031791c07a5882798"
CACHE_NAME = "pl_cache_kan_from_v4_1_7439_ms2"
CACHE_DIR = STRICT / CACHE_NAME
ANCHOR_NAME = "teacher_anchors_segformer_historical_rebuilt"
ANCHOR_DIR = STRICT / ANCHOR_NAME
ANCHOR_MANIFEST = STRICT / f"{ANCHOR_NAME}_MANIFEST.json"

PL_MAIN = "ckpt_kan_historical_pl_rebuilt.pth"
PL_SNAPS = [f"ckpt_kan_historical_pl_rebuilt_i{i}.pth" for i in (1000, 2000, 3000, 4000)]
PL_EVAL = "eval_kan_historical_pl_rebuilt_i2000_plain_ema_n500.json"
PL_MASK_DIR = STRICT / "pl_masks_historical_rebuilt_t85"

DISTILL_MAIN = "ckpt_kan_segformer_distill_rebuilt.pth"
DISTILL_SNAPS = [f"ckpt_kan_segformer_distill_rebuilt_i{i}.pth" for i in (1000, 2000, 3000, 4000)]
DISTILL_EVAL = "eval_kan_segformer_distill_rebuilt_i2000_plain_ema_n500.json"
DISTILL_CONSISTENCY = "audit_kan_segformer_distill_rebuilt_i2000_teacher_consistency.json"

EXPECTED_KAN_SOURCE_SHA = "c84d5c884a17fff40257b137b08ed08bc1729243a9884439b315b8132ccdb9af"
EXPECTED_DEPENDENCY_SHA = {
    "kan_swin.py": EXPECTED_KAN_SOURCE_SHA,
    "seg_swin.py": "d33ff136e70ba1d2474372909e78d6af46f14562838bcaa6f33985d3ea648f0d",
    "strict_env.py": "1675f72661f6663870293732851bae9c055d7edd6f19f7ee2744f4d1cfb101f9",
    "strict_eval.py": "7f7da8b9534f54f824413a53ecd459b4acbdd90bce219ef271f93992940fbd54",
    "pl_gen.py": "be95121991a4332b306e3d4ea08c99c0d4b5181c566ee1f2d9c712de6eab7e18",
    "audit_kan_segformer_teacher_consistency.py": "8356c3404b731e44f82f52b210d340bd32e9f388be66384b6b75ff6de1b915c7"
}
INITIAL_TEACHER_RESUME_COUNT = 2
POLL_SECONDS = 300
DISTILL_START_DEADLINE_HOUR = 19
DISTILL_START_DEADLINE_MINUTE = 30


class StopFlow(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def log(message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with MASTER_LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def deadline() -> datetime:
    started = datetime.fromisoformat(STATE["started_at"])
    return started.replace(
        hour=DISTILL_START_DEADLINE_HOUR,
        minute=DISTILL_START_DEADLINE_MINUTE,
        second=0,
        microsecond=0,
    )


def render_report(state: dict) -> None:
    teacher = state.get("teacher", {})
    selected = state.get("selected_teacher", {})
    pl = state.get("pl", {})
    distill = state.get("distill", {})
    lines = [
        "# SegFormer教师 → KAN PL → SegFormer蒸馏 总状态",
        "",
        f"- 更新时间：`{state.get('updated_at')}`",
        f"- 流程状态：`{state.get('status')}`",
        f"- 当前阶段：`{state.get('current_stage')}`",
        f"- 停止/失败原因：`{state.get('reason')}`",
        "",
        "## 历史SegFormer教师",
        "",
        f"- 教师队列状态：`{teacher.get('queue_status')}`",
        f"- C5完成：`{teacher.get('c5_complete')}`；checkpoint SHA-256：`{teacher.get('c5_checkpoint_sha256')}`",
        f"- C6完成：`{teacher.get('c6_complete')}`；checkpoint SHA-256：`{teacher.get('c6_checkpoint_sha256')}`",
        f"- C5时长秒/峰值显存MiB：`{teacher.get('c5_duration_seconds')}` / `{teacher.get('c5_peak_gpu_memory_mib')}`",
        f"- C6时长秒/峰值显存MiB：`{teacher.get('c6_duration_seconds')}` / `{teacher.get('c6_peak_gpu_memory_mib')}`",
        f"- 本次续接等待期间发生新中断：`{teacher.get('new_interruption')}`",
        f"- 总resume次数：`{teacher.get('resume_count')}`；随机轨迹等价于不中断：`{teacher.get('trajectory_equivalent_to_uninterrupted')}`",
        f"- C5与历史锚像素一致率：`{teacher.get('c5_historical_pixel_agreement')}`",
        f"- C6与历史锚像素一致率：`{teacher.get('c6_historical_pixel_agreement')}`",
        "",
        "## 选择的新教师锚",
        "",
        f"- 选择：`{selected.get('identity')}`；理由：{selected.get('reason')}",
        f"- 新锚目录：`{selected.get('anchor_dir')}`",
        f"- manifest：`{selected.get('manifest')}`",
        f"- 锚文件数：`{selected.get('files')}`；像素范围：`{selected.get('labels')}`",
        "",
        "## 固定PL",
        "",
        f"- tau=0.85当前覆盖率：`{state.get('cache_audit', {}).get('coverage_tau_085')}`；历史记录约：`0.882`",
        f"- 启动：`{pl.get('started')}`；完成：`{pl.get('complete')}`；状态：`{pl.get('status')}`",
        f"- i2000 n500 plain EMA：`{pl.get('i2000_n500_miou')}`",
        f"- i2000各类别IoU：`{pl.get('i2000_per_class_iou')}`",
        f"- 主权重/快照：`{pl.get('weights')}`",
        f"- 时长秒：`{pl.get('duration_seconds')}`；峰值显存MiB：`{pl.get('peak_gpu_memory_mib')}`",
        "",
        "## SegFormer硬标签蒸馏",
        "",
        f"- 启动：`{distill.get('started')}`；完成：`{distill.get('complete')}`；状态：`{distill.get('status')}`",
        f"- i2000 n500 plain EMA：`{distill.get('i2000_n500_miou')}`",
        f"- 与本轮教师像素一致率：`{distill.get('teacher_pixel_agreement')}`",
        f"- i2000各类别IoU：`{distill.get('i2000_per_class_iou')}`",
        f"- i2000预测占比（test1 ms2）：`{distill.get('student_prediction_fraction')}`",
        f"- 主权重/快照：`{distill.get('weights')}`",
        f"- 时长秒：`{distill.get('duration_seconds')}`；峰值显存MiB：`{distill.get('peak_gpu_memory_mib')}`",
        "",
        "## 明确边界",
        "",
        "- 当前分类：本地候选。",
        "- test2包：未生成。",
        "- 已输出：否。",
        "- 本轮真实外部指标：无。",
        "- 47阶段：未启动；是否租用32GB AutoDL等待用户回来后决定。",
        "",
        "注意：历史教师链发生过两次用户主动暂停/续训，`trajectory_equivalent_to_uninterrupted=false`，不得宣称随机轨迹完全复现。",
    ]
    temporary = REPORT.with_suffix(REPORT.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(REPORT)


def update_status() -> None:
    STATE["updated_at"] = now()
    atomic_json(STATUS, STATE)
    render_report(STATE)


def gpu_memory_mib() -> int | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return max(int(line.strip()) for line in output.splitlines() if line.strip())
    except Exception:
        return None


def ensure_absent(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise StopFlow("refusing to overwrite existing artifacts: " + "; ".join(existing))


def run_job(key: str, args: list[str], section: str) -> dict:
    stdout = STRICT_LOGS / f"segformer_handoff_{key}.stdout.log"
    stderr = STRICT_LOGS / f"segformer_handoff_{key}.stderr.log"
    metadata_path = STRICT / f"segformer_handoff_{key}.json"
    ensure_absent([stdout, stderr, metadata_path])
    command = [str(PYTHON), "-u", *args]
    started = now()
    started_clock = time.time()
    baseline = gpu_memory_mib()
    peak = baseline
    item = {
        "status": "running",
        "started_at": started,
        "command": command,
        "cwd": str(SOURCE),
        "stdout": str(stdout),
        "stderr": str(stderr),
        "gpu_memory_baseline_mib": baseline
    }
    STATE[section].update({"status": key, "started": True})
    STATE["current_stage"] = key
    update_status()
    atomic_json(metadata_path, item)
    log(f"START {key}: {' '.join(args)}")
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        process = subprocess.Popen(
            command,
            cwd=str(SOURCE),
            env=environment,
            stdout=out,
            stderr=err,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        item["pid"] = process.pid
        atomic_json(metadata_path, item)
        while process.poll() is None:
            current = gpu_memory_mib()
            if current is not None:
                peak = current if peak is None else max(peak, current)
            time.sleep(5)
        code = process.returncode
    item.update(
        {
            "status": "complete" if code == 0 else "failed",
            "exit_code": code,
            "finished_at": now(),
            "duration_seconds": time.time() - started_clock,
            "gpu_memory_peak_mib": peak,
            "gpu_memory_peak_delta_mib": (
                peak - baseline if peak is not None and baseline is not None else None
            )
        }
    )
    atomic_json(metadata_path, item)
    log(f"END {key}: exit={code}")
    if code != 0:
        raise StopFlow(f"{key} failed with exit code {code}; see {stderr}")
    return item


def validate_checkpoint(name: str, stage: str, iteration: int) -> dict:
    import strict_env as P

    path = STRICT / name
    payload = P.load_checkpoint(path, map_location="cpu")
    metadata = payload["strict_repro"]
    if payload.get("iter") != iteration or metadata.get("stage") != stage:
        raise StopFlow(f"checkpoint stage/iter mismatch: {name}")
    if payload.get("ema") is None:
        raise StopFlow(f"checkpoint has no EMA: {name}")
    return {
        "filename": name,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "iter": payload.get("iter"),
        "strict_repro": metadata
    }


def valid_eval(result: dict) -> tuple[bool, str]:
    score = result.get("val_mIoU")
    if score is None or not math.isfinite(float(score)) or float(score) < 0.7300:
        return False, f"n500 mIoU {score} is below 0.7300 or non-finite"
    for name, value in result.get("val_IoU", {}).items():
        if value is None or not math.isfinite(float(value)) or float(value) <= 0:
            return False, f"{name} has non-finite/zero IoU"
        if float(value) < 0.05:
            return False, f"{name} IoU {value} is below collapse floor 0.05"
    return True, "n500 >= 0.7300 and all class IoUs are finite/nonzero/non-collapsed"


def wait_for_teacher() -> dict:
    log(f"Waiting for teacher queue with {POLL_SECONDS}s polling; current C5 is not touched")
    while True:
        if not TEACHER_STATUS.is_file():
            raise StopFlow(f"teacher status file missing: {TEACHER_STATUS}")
        queue = json.loads(TEACHER_STATUS.read_text(encoding="utf-8"))
        status = str(queue.get("status"))
        STATE["teacher"]["queue_status"] = status
        STATE["teacher"]["resume_count"] = queue.get("resume_count")
        STATE["teacher"]["trajectory_equivalent_to_uninterrupted"] = queue.get(
            "trajectory_equivalent_to_uninterrupted"
        )
        update_status()
        if status == "complete":
            return queue
        if status in {"failed", "stopped", "paused_by_user", "blocked"}:
            raise StopFlow(f"teacher queue ended without completion: status={status}")
        if status != "running":
            raise StopFlow(f"unexpected teacher queue status: {status}")
        time.sleep(POLL_SECONDS)


def select_and_copy_teacher(queue: dict) -> None:
    if not IDENTITY_REPORT.is_file():
        raise StopFlow(f"teacher completed but identity report is missing: {IDENTITY_REPORT}")
    identity = json.loads(IDENTITY_REPORT.read_text(encoding="utf-8"))
    comparisons = identity["comparisons"]
    c5_comparison = comparisons["historical_vs_c5"]
    c6_comparison = comparisons["historical_vs_c6"]
    c5_agreement = float(c5_comparison["pixel_agreement"])
    c6_agreement = float(c6_comparison["pixel_agreement"])
    c5_exact = int(c5_comparison["identical_png_pixels_files"]) == 500
    c6_exact = int(c6_comparison["identical_png_pixels_files"]) == 500

    c5_meta_path = TEACHER_RUN / "metadata" / "c5_full1024.json"
    c6_meta_path = TEACHER_RUN / "metadata" / "c6_dacs768.json"
    c5_meta = json.loads(c5_meta_path.read_text(encoding="utf-8"))
    c6_meta = json.loads(c6_meta_path.read_text(encoding="utf-8"))
    if c5_meta.get("status") != "complete" or c6_meta.get("status") != "complete":
        raise StopFlow("teacher queue says complete but C5/C6 metadata is incomplete")
    c5_ckpt = Path(c5_meta["output_path"])
    c6_ckpt = Path(c6_meta["output_path"])
    if sha256(c5_ckpt) != c5_meta["output_sha256"] or sha256(c6_ckpt) != c6_meta["output_sha256"]:
        raise StopFlow("C5/C6 checkpoint hash mismatch")

    STATE["teacher"].update(
        {
            "queue_status": "complete",
            "c5_complete": True,
            "c6_complete": True,
            "c5_checkpoint": str(c5_ckpt),
            "c6_checkpoint": str(c6_ckpt),
            "c5_checkpoint_sha256": c5_meta["output_sha256"],
            "c6_checkpoint_sha256": c6_meta["output_sha256"],
            "c5_duration_seconds": c5_meta.get("duration_seconds"),
            "c6_duration_seconds": c6_meta.get("duration_seconds"),
            "c5_peak_gpu_memory_mib": c5_meta.get("gpu_memory_peak_mib"),
            "c6_peak_gpu_memory_mib": c6_meta.get("gpu_memory_peak_mib"),
            "c5_historical_pixel_agreement": c5_agreement,
            "c6_historical_pixel_agreement": c6_agreement,
            "c5_historical_exact_files": c5_comparison["identical_png_pixels_files"],
            "c6_historical_exact_files": c6_comparison["identical_png_pixels_files"],
            "resume_count": queue.get("resume_count"),
            "new_interruption": int(queue.get("resume_count", 0)) > INITIAL_TEACHER_RESUME_COUNT,
            "trajectory_equivalent_to_uninterrupted": False
        }
    )

    if c5_exact or c6_exact:
        if c6_exact:
            selected_id, reason = "C6", "C6逐文件完全一致（如C5也一致，按历史链后阶段C6优先）"
        else:
            selected_id, reason = "C5", "仅C5逐文件完全一致"
    else:
        best = max(c5_agreement, c6_agreement)
        if best < 0.90:
            raise StopFlow(
                f"C5/C6 historical pixel agreement below 90%: C5={c5_agreement:.6%}, C6={c6_agreement:.6%}"
            )
        if c6_agreement >= c5_agreement:
            selected_id, reason = "C6", "两者均非完全一致；C6全部像素一致率较高且达到90%"
        else:
            selected_id, reason = "C5", "两者均非完全一致；C5全部像素一致率较高且达到90%"

    source_dir = TEACHER_RUN / ("c5_test1_anchor_candidate" if selected_id == "C5" else "c6_test1_anchor_candidate")
    checkpoint = c5_ckpt if selected_id == "C5" else c6_ckpt
    checkpoint_sha = c5_meta["output_sha256"] if selected_id == "C5" else c6_meta["output_sha256"]
    selected_comparison = c5_comparison if selected_id == "C5" else c6_comparison
    test_files = sorted((REPRO / "test_1" / "images").glob("*.png"))
    source_files = sorted(source_dir.glob("*.png"))
    if len(test_files) != 500 or {p.name for p in test_files} != {p.name for p in source_files}:
        raise StopFlow("selected teacher predictions do not match all 500 official test1 filenames")
    # Validate every rebuilt prediction before creating the destination. A bad
    # source must leave no partial training-label directory behind.
    source_hashes = {}
    class_pixels = np.zeros(9, dtype=np.int64)
    for source_file in source_files:
        with Image.open(source_file) as image:
            array = np.asarray(image)
        if array.ndim != 2 or array.shape != (1024, 1024):
            raise StopFlow(f"invalid anchor dimensions/channels: {source_file.name}: {array.shape}")
        values = np.unique(array)
        if values.size == 0 or values.min() < 1 or values.max() > 8:
            raise StopFlow(f"anchor labels outside 1..8: {source_file.name}: {values.tolist()}")
        source_hashes[source_file.name] = sha256(source_file)
        class_pixels += np.bincount(array.ravel(), minlength=9)[:9]

    ANCHOR_DIR.mkdir(parents=True, exist_ok=False)
    file_hashes = {}
    for source_file in source_files:
        destination = ANCHOR_DIR / source_file.name
        shutil.copy2(source_file, destination)
        file_hashes[source_file.name] = sha256(destination)
        if file_hashes[source_file.name] != source_hashes[source_file.name]:
            raise StopFlow(f"anchor changed during copy: {source_file.name}")
    if len(list(ANCHOR_DIR.iterdir())) != 500:
        raise StopFlow("anchor directory must contain exactly 500 PNG files and no manifest")
    digest = hashlib.sha256()
    for name in sorted(file_hashes):
        digest.update(f"{name}\0{file_hashes[name]}\n".encode("utf-8"))
    manifest = {
        "status": "local_rebuilt_teacher_anchor_candidate",
        "created_at": now(),
        "anchor_dir": str(ANCHOR_DIR),
        "manifest_location_note": "manifest is adjacent to the anchor directory so frozen kan_swin.py sees exactly 500 PNG files",
        "source_prediction_dir": str(source_dir),
        "selected_identity": selected_id,
        "selection_reason": reason,
        "teacher_checkpoint": str(checkpoint),
        "teacher_checkpoint_sha256": checkpoint_sha,
        "historical_anchor_role": "read-only identity reference; no historical file was copied",
        "historical_pixel_agreement": float(selected_comparison["pixel_agreement"]),
        "historical_identical_files": int(selected_comparison["identical_png_pixels_files"]),
        "c5_historical_pixel_agreement": c5_agreement,
        "c6_historical_pixel_agreement": c6_agreement,
        "teacher_trajectory_equivalent_to_uninterrupted": False,
        "teacher_resume_count": queue.get("resume_count"),
        "inference": {
            "tile_size": 1024,
            "overlap": 0,
            "scales": [0.75, 1.0, 1.25],
            "horizontal_flip": True,
            "aggregation": "per-view softmax probability average"
        },
        "files": 500,
        "dimensions": [1024, 1024],
        "channels": 1,
        "allowed_labels": list(range(1, 9)),
        "class_pixels": {str(i): int(class_pixels[i]) for i in range(1, 9)},
        "file_list_digest_sha256": digest.hexdigest(),
        "file_sha256": file_hashes
    }
    atomic_json(ANCHOR_MANIFEST, manifest)
    STATE["selected_teacher"] = {
        "identity": selected_id,
        "reason": reason,
        "anchor_dir": str(ANCHOR_DIR),
        "manifest": str(ANCHOR_MANIFEST),
        "files": 500,
        "labels": "1-8",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "historical_pixel_agreement": float(selected_comparison["pixel_agreement"])
    }
    update_status()


def audit_base_and_cache() -> None:
    for filename, expected in EXPECTED_DEPENDENCY_SHA.items():
        if sha256(SOURCE / filename).lower() != expected:
            raise StopFlow(f"frozen dependency hash changed: {filename}")
    if sha256(BASE_PATH).lower() != BASE_SHA:
        raise StopFlow("KAN16k base hash mismatch")
    import strict_env as P

    payload = P.load_checkpoint(BASE_PATH, map_location="cpu")
    metadata = payload["strict_repro"]
    if (
        payload.get("iter") != 16000
        or metadata.get("stage") != "kan_from_v4_1_7439_16k"
        or metadata.get("seed") != 42
        or payload.get("ema") is None
    ):
        raise StopFlow("KAN16k iter/stage/seed/EMA mismatch")
    manifest = json.loads((CACHE_DIR / "_CACHE_MANIFEST.json").read_text(encoding="utf-8"))
    confidence = json.loads((CACHE_DIR / "_CONFIDENCE_REPORT.json").read_text(encoding="utf-8"))
    expected_manifest = {
        "checkpoint": BASE_NAME,
        "checkpoint_sha256": BASE_SHA,
        "route": "ms2",
        "source": "official_test1",
        "expected_images": 500
    }
    if manifest != expected_manifest:
        raise StopFlow("pseudo-label cache manifest mismatch")
    if (
        confidence.get("checkpoint_sha256") != BASE_SHA
        or confidence.get("checkpoint_stage") != "kan_from_v4_1_7439_16k"
        or confidence.get("checkpoint_iter") != 16000
        or confidence.get("route") != "ms2"
        or confidence.get("official_test1_images") != 500
    ):
        raise StopFlow("pseudo-label confidence report lineage mismatch")
    test_names = {path.stem for path in (REPRO / "test_1" / "images").glob("*.png")}
    cache_files = sorted(CACHE_DIR.glob("*.npz"))
    if len(cache_files) != 500 or {path.stem for path in cache_files} != test_names:
        raise StopFlow("pseudo-label cache does not correspond to all 500 official test1 images")
    for path in cache_files:
        with np.load(path, allow_pickle=False) as values:
            if set(values.files) != {"pred", "conf"}:
                raise StopFlow(f"invalid cache keys: {path.name}")
            prediction, confidence_map = values["pred"], values["conf"]
            if prediction.shape != (1024, 1024) or confidence_map.shape != (1024, 1024):
                raise StopFlow(f"invalid cache shape: {path.name}")
            if prediction.dtype != np.uint8 or not np.isfinite(confidence_map).all():
                raise StopFlow(f"invalid cache dtype/values: {path.name}")
            if confidence_map.min() < 0 or confidence_map.max() > 1:
                raise StopFlow(f"invalid confidence range: {path.name}")
    generator_source = (SOURCE / "pl_gen.py").read_text(encoding="utf-8")
    if "use_ema=True" not in generator_source or 'tta="ms2"' not in generator_source:
        raise StopFlow("frozen pseudo-label generator no longer proves EMA/ms2 route")
    STATE["cache_audit"] = {
        "status": "passed",
        "checkpoint": BASE_NAME,
        "checkpoint_sha256": BASE_SHA,
        "checkpoint_stage": metadata.get("stage"),
        "checkpoint_iter": payload.get("iter"),
        "checkpoint_has_ema": True,
        "route": "ms2",
        "official_test1_files": 500,
        "cache_files": 500,
        "pl_gen_sha256": sha256(SOURCE / "pl_gen.py"),
        "coverage_tau_085": float(confidence["coverage"]["0.85"]),
        "historical_coverage_reference": 0.882,
        "warning": "current 60.8063% and historical about 88.2% use an unproven-identical statistic; threshold remains 0.85"
    }
    update_status()


def run_pl() -> None:
    item = run_job(
        "pl_train",
        [
            "kan_swin.py", "--train", "--pl", "--crop", "640", "--batch-size", "2",
            "--num-workers", "3", "--target-iters", "4000", "--max-seconds", "86400",
            "--seed", "42", "--stage", "kan_historical_pl_rebuilt",
            "--ckpt-name", PL_MAIN, "--warm-base", BASE_NAME,
            "--warm-parent-stage", "kan_from_v4_1_7439_16k", "--warm-parent-iter", "16000",
            "--pl-cache", CACHE_NAME, "--pl-tag", "historical_rebuilt",
            "--pl-tau", "0.85", "--restart-lr", "0.25",
        ],
        "pl",
    )
    weights = [validate_checkpoint(PL_MAIN, "kan_historical_pl_rebuilt", 4000)]
    for iteration, name in zip((1000, 2000, 3000, 4000), PL_SNAPS):
        weights.append(validate_checkpoint(name, "kan_historical_pl_rebuilt", iteration))
    STATE["pl"].update(
        {
            "complete": True,
            "status": "training_complete",
            "duration_seconds": item["duration_seconds"],
            "peak_gpu_memory_mib": item["gpu_memory_peak_mib"],
            "weights": weights,
            "historical_selection": PL_SNAPS[1]
        }
    )
    update_status()

    run_job(
        "pl_i2000_eval_n500",
        [
            "strict_eval.py", "--model", "kan", "--ckpt", PL_SNAPS[1],
            "--n", "500", "--output", PL_EVAL,
            "--expected-stage", "kan_historical_pl_rebuilt", "--expected-iter", "2000",
        ],
        "pl",
    )
    result = json.loads((STRICT / PL_EVAL).read_text(encoding="utf-8"))
    valid, reason = valid_eval(result)
    STATE["pl"].update(
        {
            "status": "gate_passed" if valid else "gate_failed",
            "i2000_n500_miou": result.get("val_mIoU"),
            "i2000_per_class_iou": result.get("val_IoU"),
            "gate_reason": reason
        }
    )
    update_status()
    if not valid:
        raise StopFlow("PL i2000 gate failed: " + reason)


def run_distill() -> None:
    item = run_job(
        "segformer_distill_train",
        [
            "kan_swin.py", "--train", "--anchor", "--crop", "640", "--batch-size", "2",
            "--num-workers", "3", "--target-iters", "4000", "--max-seconds", "86400",
            "--seed", "42", "--stage", "kan_segformer_distill_rebuilt",
            "--ckpt-name", DISTILL_MAIN, "--warm-base", PL_SNAPS[1],
            "--warm-parent-stage", "kan_historical_pl_rebuilt", "--warm-parent-iter", "2000",
            "--anchor-dir", ANCHOR_NAME, "--restart-lr", "0.25",
        ],
        "distill",
    )
    weights = [validate_checkpoint(DISTILL_MAIN, "kan_segformer_distill_rebuilt", 4000)]
    for iteration, name in zip((1000, 2000, 3000, 4000), DISTILL_SNAPS):
        weights.append(validate_checkpoint(name, "kan_segformer_distill_rebuilt", iteration))
    STATE["distill"].update(
        {
            "complete": True,
            "status": "training_complete",
            "duration_seconds": item["duration_seconds"],
            "peak_gpu_memory_mib": item["gpu_memory_peak_mib"],
            "weights": weights,
            "historical_selection": DISTILL_SNAPS[1]
        }
    )
    update_status()

    run_job(
        "segformer_distill_i2000_eval_n500",
        [
            "strict_eval.py", "--model", "kan", "--ckpt", DISTILL_SNAPS[1],
            "--n", "500", "--output", DISTILL_EVAL,
            "--expected-stage", "kan_segformer_distill_rebuilt", "--expected-iter", "2000",
        ],
        "distill",
    )
    result = json.loads((STRICT / DISTILL_EVAL).read_text(encoding="utf-8"))
    valid, reason = valid_eval(result)
    STATE["distill"].update(
        {
            "status": "eval_gate_passed" if valid else "eval_gate_failed",
            "i2000_n500_miou": result.get("val_mIoU"),
            "i2000_per_class_iou": result.get("val_IoU"),
            "gate_reason": reason
        }
    )
    update_status()
    if not valid:
        raise StopFlow("SegFormer distill i2000 gate failed: " + reason)

    run_job(
        "segformer_distill_i2000_teacher_consistency",
        [
            "audit_kan_segformer_teacher_consistency.py",
            "--ckpt", DISTILL_SNAPS[1], "--anchor-dir", ANCHOR_NAME,
            "--output", DISTILL_CONSISTENCY,
            "--expected-stage", "kan_segformer_distill_rebuilt", "--expected-iter", "2000",
        ],
        "distill",
    )
    consistency = json.loads((STRICT / DISTILL_CONSISTENCY).read_text(encoding="utf-8"))
    STATE["distill"].update(
        {
            "status": "complete",
            "teacher_pixel_agreement": consistency.get("pixel_agreement"),
            "teacher_miou": consistency.get("miou_between_teacher_and_student"),
            "student_prediction_fraction": consistency.get("student_prediction_fraction"),
            "teacher_consistency_report": DISTILL_CONSISTENCY
        }
    )
    update_status()


def preflight() -> None:
    STRICT_LOGS.mkdir(parents=True, exist_ok=True)
    if STATUS.exists() or REPORT.exists() or MASTER_LOG.exists():
        raise StopFlow("one-shot handoff was already launched; refusing a second launch")
    if not TEACHER_STATUS.is_file():
        raise StopFlow("teacher status file is missing")
    queue = json.loads(TEACHER_STATUS.read_text(encoding="utf-8"))
    if queue.get("status") != "running":
        raise StopFlow(f"teacher queue is not currently running: {queue.get('status')}")
    if int(queue.get("resume_count", -1)) != INITIAL_TEACHER_RESUME_COUNT:
        raise StopFlow("teacher resume count changed before handoff launch")
    ensure_absent(
        [
            ANCHOR_DIR, ANCHOR_MANIFEST, PL_MASK_DIR,
            *(STRICT / name for name in [PL_MAIN, PL_EVAL, *PL_SNAPS, DISTILL_MAIN, DISTILL_EVAL, DISTILL_CONSISTENCY, *DISTILL_SNAPS]),
        ]
    )
    if sha256(BASE_PATH).lower() != BASE_SHA:
        raise StopFlow("KAN16k base hash mismatch during preflight")
    for filename, expected in EXPECTED_DEPENDENCY_SHA.items():
        if sha256(SOURCE / filename).lower() != expected:
            raise StopFlow(f"dependency hash mismatch during preflight: {filename}")


def main() -> None:
    preflight()
    STATE.update(
        {
            "status": "running",
            "current_stage": "waiting_for_teacher",
            "reason": None,
            "teacher_poll_seconds": POLL_SECONDS,
            "distill_start_deadline": deadline().isoformat()
        }
    )
    update_status()
    try:
        queue = wait_for_teacher()
        STATE["current_stage"] = "selecting_rebuilt_teacher"
        update_status()
        select_and_copy_teacher(queue)
        audit_base_and_cache()
        if datetime.now().astimezone() >= deadline():
            STATE.update(
                {
                    "status": "complete_teacher_only_after_deadline",
                    "current_stage": None,
                    "reason": "teacher completed at/after 19:30; PL was not started"
                }
            )
            update_status()
            return
        run_pl()
        if datetime.now().astimezone() >= deadline():
            STATE.update(
                {
                    "status": "complete_pl_only_deadline",
                    "current_stage": None,
                    "reason": "PL completed/evaluated at or after 19:30; distillation was not started"
                }
            )
            update_status()
            return
        run_distill()
        STATE.update({"status": "complete", "current_stage": None, "reason": None, "finished_at": now()})
        update_status()
    except StopFlow as error:
        STATE.update(
            {
                "status": "stopped",
                "current_stage": None,
                "reason": str(error),
                "finished_at": now()
            }
        )
        update_status()
        log("STOP: " + str(error))
    except Exception as error:
        STATE.update(
            {
                "status": "failed",
                "current_stage": None,
                "reason": repr(error),
                "traceback": traceback.format_exc(),
                "finished_at": now()
            }
        )
        update_status()
        log("FAILED: " + repr(error))
        raise


STATE = {
    "protocol": "segformer_rebuilt_then_historical_pl_then_segformer_anchor_20260925",
    "started_at": now(),
    "updated_at": None,
    "status": "initializing",
    "current_stage": None,
    "reason": None,
    "teacher": {
        "queue_status": None,
        "initial_resume_count": INITIAL_TEACHER_RESUME_COUNT,
        "trajectory_equivalent_to_uninterrupted": False
    },
    "selected_teacher": {},
    "cache_audit": {},
    "pl": {"started": False, "complete": False, "status": "not_started"},
    "distill": {"started": False, "complete": False, "status": "not_started"},
    "boundaries": {
        "local_candidate": True,
        "test2_package_generated": False,
        "submitted": False,
        "real_result_score": None,
        "autodl_started": False,
        "stage47_started": False
    }
}


if __name__ == "__main__":
    main()
