# Resource estimate

- Formal shape: crop1024, true batch8, no gradient accumulation substitute.
- Expected peak VRAM: about 19–20 GiB; use a 32 GiB GPU instance.
- The optimizer and scheduler retain the original 4000-iteration horizon, but this run executes only 1400 iterations.
- Expected training time: approximately 1.5–2.0 hours.
- The user's last observed live price was CNY 1.78/hour.
- Expected training cost is approximately CNY 2.67–3.56; reserve CNY 3–5 including upload, verification and download.
