#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "[part8] running reliability analysis workflow..."
# Optional overrides:
#   COUNTDOWN_LIMIT=8 COUNTDOWN_REPEATS=2 bash examples/countdown/run_part8_pipeline.sh
uv run python examples/countdown/run_part8_analysis.py
echo "[part8] done"
