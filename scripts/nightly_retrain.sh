#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

unset PYTHONHOME
unset PYTHONPATH

/Users/kwangjinpark/miniconda3/bin/python3 -m pipeline.auto_submit --nightly-retrain
