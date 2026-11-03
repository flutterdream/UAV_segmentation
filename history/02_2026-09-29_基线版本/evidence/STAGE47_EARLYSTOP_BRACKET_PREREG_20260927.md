# Stage47 early-stop bracket preregistration

## Objective

Search only the stopping point around the result-proven Stage47 i1000 model.  No architecture, loss, data, teacher, anchor, augmentation, batch size, learning rate, source supervision, EMA, BN calibration or inference change is allowed.

## Frozen lineage and training

- Inputs must come only from the strict `E:\复现` lineage.
- Parent: rebuilt SegFormer-C6 distilled KAN i2000 used by the completed historical Stage47 run.
- Crop 1024, true batch 8, seed 17, restart LR 0.25, tau 0.85.
- Teacher horizontal flip, anchor-conflict ignore, source supervision 0.20, EMA 0.999.
- The cosine scheduler horizon remains exactly 4000 iterations.
- One uninterrupted run executes from iteration 0 through 1400.
- Fixed checkpoints: i600, i800, i1000, i1200 and final i1400.
- No resume and no adaptive stopping based on intermediate validation.

## Selection

All five checkpoints are evaluated locally on the same fixed seed123 n500 split with plain EMA inference.  Only checkpoints with finite, nonzero IoUs for all eight classes are eligible.  Select exactly one checkpoint by highest n500 mIoU; ties within 0.0005 prefer the earlier checkpoint.

The result-proven i1000 anchor has pre-BN n500 `0.7506` and post-test2-BN0.25 n500 `0.7551544710`.  Test2 BN0.25 and eight-pass MS2 inference are run only for the single selected checkpoint, and only if its pre-BN n500 is at least `0.7526` with no class IoU drop greater than 0.01 against the proven i1000 pre-BN evaluation.

No output ZIP is generated for the other checkpoints.  Local validation is a selection guard, not a result prediction.

## Resource estimate

- Required GPU memory: approximately 18.6 GiB; 24 GiB or 32 GiB GPU is sufficient.
- Expected cloud training time: 1.5-2.0 hours.
- Expected evaluation and transfer: 0.5-0.75 hour.
- At RMB 1.78/hour, expected total cost is approximately RMB 3-5.

Cloud shutdown is allowed only after the i1400 checkpoint and all fixed snapshots are verified and downloaded.  Current  weight, predictions and ZIP are immutable and must not be modified.
