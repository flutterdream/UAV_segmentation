# -*- coding: utf-8 -*-
"""Resume the conditional queue after PL completed but its log decode check failed."""
from __future__ import annotations
import json, math, traceback
from pathlib import Path

import overnight_segformer_pl_distill_47_queue as q

OLD_STATUS = q.STRICT / "overnight_segformer_pl_distill_47_status.json"
RESUME_STATUS = q.STRICT / "overnight_segformer_pl_distill_47_resume1_status.json"
RESUME_LOG = q.LOGS / "overnight_segformer_pl_distill_47_resume1.log"
AMENDMENT = q.REPRO / "docs" / "OVERNIGHT_QUEUE_PL_LOG_DECODE_AMENDMENT_20260926.md"


def preflight_and_restore_state():
    if RESUME_STATUS.exists() or RESUME_LOG.exists():
        raise q.StopQueue("resume1 was already launched")
    old = json.loads(OLD_STATUS.read_text(encoding="utf-8"))
    if old.get("status") != "failed" or "UnicodeDecodeError" not in str(old.get("reason")):
        raise q.StopQueue("old queue did not stop at the known log-decode failure")
    if old.get("pl", {}).get("complete") is not False or old.get("distill", {}).get("complete") is not False:
        raise q.StopQueue("old state is not the expected post-PL/pre-eval boundary")
    metadata = json.loads((q.STRICT / "overnight47_pl_train.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata.get("return_code") != 0 or metadata.get("resumed") is not False:
        raise q.StopQueue("PL job metadata does not prove a clean uninterrupted completion")
    stdout = (q.LOGS / "overnight47_pl_train.stdout.log").read_bytes()
    if b"[resume]" in stdout or b"iter 4000/4000" not in stdout or b"[done]" not in stdout:
        raise q.StopQueue("PL log does not prove clean 0->4000 completion")
    if (q.LOGS / "overnight47_pl_train.stderr.log").stat().st_size != 0:
        raise q.StopQueue("PL stderr is not empty")
    if (q.STRICT / q.PL_EVAL).exists() or (q.STRICT / q.DIST_MAIN).exists() or q.PACKAGE.exists():
        raise q.StopQueue("downstream artifacts already exist; refusing ambiguous resume")
    q.STATUS = RESUME_STATUS
    q.MASTER_LOG = RESUME_LOG
    q.STATE = old
    q.STATE.update({
        "status": "running",
        "current_stage": "validate_completed_pl_checkpoints",
        "reason": None,
        "resumed_queue_wrapper": True,
        "pl_training_repeated": False,
        "resume_amendment": str(AMENDMENT),
        "original_failed_status_preserved": str(OLD_STATUS)
    })
    q.STATE.pop("traceback", None)
    q.STATE.pop("finished_at", None)
    q.update()
    return metadata


def validate_pl_and_evaluate(metadata):
    weights = [q.checkpoint_info(q.PL_MAIN, "kan_historical_pl_rebuilt", 4000)]
    weights += [q.checkpoint_info(name, "kan_historical_pl_rebuilt", iteration)
                for name, iteration in zip(q.PL_SNAPS, (1000, 2000, 3000, 4000))]
    q.STATE["pl"].update({
        "complete": True,
        "status": "training_complete_validated",
        "weights": weights,
        "duration_seconds": metadata["duration_seconds"],
        "peak_gpu_memory_mib": metadata["peak_gpu_memory_mib"],
        "training_interrupted": False,
        "training_resumed": False
    })
    q.update()
    q.run_job(
        "pl_i2000_eval_n500",
        ["strict_eval.py", "--model", "kan", "--ckpt", q.PL_SNAPS[1],
         "--n", "500", "--output", q.PL_EVAL,
         "--expected-stage", "kan_historical_pl_rebuilt", "--expected-iter", "2000"],
        "pl",
    )
    result = json.loads((q.STRICT / q.PL_EVAL).read_text(encoding="utf-8"))
    passed, reason = q.eval_gate(result)
    q.STATE["pl"].update({
        "status": "gate_passed" if passed else "gate_failed",
        "i2000_n500_miou": result.get("val_mIoU"),
        "i2000_per_class_iou": result.get("val_IoU"),
        "gate_reason": reason
    })
    q.update()
    if not passed:
        raise q.StopQueue("PL gate failed: " + reason)


def main():
    try:
        metadata = preflight_and_restore_state()
        validate_pl_and_evaluate(metadata)
        q.run_distill()
        q.prepare_package()
        q.STATE.update({"status": "complete", "current_stage": None, "reason": None, "finished_at": q.now()})
        q.update(); q.log("COMPLETE after log-decode amendment; stopped before AutoDL/formal47")
    except q.StopQueue as error:
        if q.STATUS != RESUME_STATUS:
            q.STATUS = RESUME_STATUS; q.MASTER_LOG = RESUME_LOG
        q.STATE.update({"status": "stopped", "current_stage": None, "reason": str(error), "finished_at": q.now()})
        q.update(); q.log("STOP: " + str(error))
    except Exception as error:
        if q.STATUS != RESUME_STATUS:
            q.STATUS = RESUME_STATUS; q.MASTER_LOG = RESUME_LOG
        q.STATE.update({"status": "failed", "current_stage": None, "reason": repr(error),
                        "traceback": traceback.format_exc(), "finished_at": q.now()})
        q.update(); q.log("FAILED: " + repr(error)); raise


if __name__ == "__main__":
    main()
