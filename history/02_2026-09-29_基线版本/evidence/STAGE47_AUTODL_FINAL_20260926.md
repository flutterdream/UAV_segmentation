# Stage 47 AutoDL Final Report — 2026-09-26

## Outcome

- Status: completed and validated
- Stage: `kan_stage47_segformer_rebuilt`
- Iterations: 4000, continuous single-process run
- Seed: 17
- Parent: `ckpt_kan_segformer_distill_rebuilt_i2000.pth` EMA
- Parent SHA-256: `b92138265b2538ae33e47ba65058b8b1713aa4efbf41745ee98d36928674a24b`
- Retained target-pixel fraction: 0.6670779488682747
- Training duration: 17,453.10 seconds (4 h 50 min 53 s)
- Peak CUDA allocation: 18,591 MiB
- Resume: no

The frozen formal configuration was crop1024, true batch8, gradient accumulation 1, seed17, 4000 iterations, restart LR 0.25, tau0.85, teacher horizontal flip, ignore-anchor-conflict, source-supervised loss weight 0.20, EMA0.999, and full-image scale-up.

## Download and integrity

- Local return archive: `E:\复现\runs\strict\stage47_remote_20260926\stage47_return_20260926.tar`
- Archive bytes: 3,918,827,520
- Archive SHA-256: `79e941141591108b3c5e721bb30468e062559bca55165535dfde7b0237a89a9d`
- Download status: complete
- Local archive SHA-256 verification: passed
- Remote post-run metadata validation: passed
- Extracted artifact directory: `E:\复现\runs\strict\stage47_remote_20260926\artifacts`

## Checkpoint SHA-256

- i1000: `2508d7ed8c0af494505ce3ae11de4e396f42f365c1702db8e473acefcfe01f81`
- i2000: `3b2a07785b3f86af692835d10e3f790b30645d6e6d917b393bc9be7b3a814096`
- i3000: `e55e2a28f03137af4a5facb8b405c1e7d009dd0519aa8a795a39af6cfeaa79e6`
- i4000: `67dcf9b0da69195592881dd026e689ea7fee2b23fda588f76d4891315675991b`
- final main checkpoint: `01f592a97dbec9a0fd87cb94437b50f9e61a7b4f5f4a843fb353be90f156f359`

## Shutdown and classification

- The original GPU instance was shut down after verified download.
- A subsequently opened no-GPU instance was also sent a shutdown request after confirming the local artifacts were already complete.
- Current classification: local Stage 47 candidate
- Stage 49: not started
- test2 inference/package: not generated
- Submitted: no
- Real result score from this run: none
