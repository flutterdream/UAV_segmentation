#!/usr/bin/env bash
set -euo pipefail
# Run manually only after downloading and verifying every item in DOWNLOAD_CHECKLIST.md.
test -f "$(dirname "$0")/../runs/ckpt_kan_stage47_earlystop_bracket_i1400.pth"
sudo shutdown -h now
