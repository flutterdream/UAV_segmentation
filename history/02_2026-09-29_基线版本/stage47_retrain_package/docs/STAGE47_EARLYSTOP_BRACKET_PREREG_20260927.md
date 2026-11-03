# Stage47 early-stop bracket preregistration

Only the Stage47 stopping point is searched.  Architecture, parent, data, C6 anchors, augmentations, batch size, seed, learning rates, loss, tau, source supervision, EMA and inference remain frozen.

- Strict E-drive reconstructed lineage only.
- Crop 1024, true batch 8, seed 17.
- Original cosine schedule horizon: 4000 iterations.
- One uninterrupted run: iteration 0 through 1400.
- Fixed checkpoints: i600, i800, i1000, i1200, i1400.
- No resume and no adaptive validation during training.
- All five checkpoints receive the same fixed seed123 n500 plain-EMA evaluation locally.
- Only one winning checkpoint may proceed to test2 BN0.25 and eight-pass MS2 inference.
- The  checkpoint, predictions and ZIP are immutable.

Expected peak GPU memory is approximately 18.6 GiB.  Use one 24 GiB or 32 GiB GPU; 48 GiB is unnecessary.
