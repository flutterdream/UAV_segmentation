#!/usr/bin/env bash
set -euo pipefail
PKG="$(cd "$(dirname "$0")/.." && pwd)"
export STAGE47_PACKAGE_ROOT="$PKG"
export STRICT_REPRO_ROOT="${STRICT_REPRO_ROOT:?set STRICT_REPRO_ROOT to uploaded official-data root}"
export PYTHONPATH="$PKG/source"
python -u "$PKG/source/train_stage47.py" --package-root "$PKG" --data-root "$STRICT_REPRO_ROOT" --config "$PKG/configs/stage47_formal.json" 2>&1 | tee "$PKG/logs/stage47_earlystop_bracket.log"
