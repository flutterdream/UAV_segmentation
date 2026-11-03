# Data layout

Set `STRICT_REPRO_ROOT` to a directory containing only the official `train/images`, `train/masks`, `test_1/images` (and later `test_2/images`) trees plus the approved public cache if needed. The stage-47 bundle already contains the newly trained distill-i2000 parent and rebuilt C6 hard anchors. Do not add historical weights, predictions, pseudo-labels or D-drive artifacts.
