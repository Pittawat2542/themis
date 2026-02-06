#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "[part7] running operations workflow..."
# Optional overrides:
#   COUNTDOWN_LIMIT=12 bash examples/countdown/run_part7_pipeline.sh
uv run python examples/countdown/run_part7_ops.py
echo "[part7] done"
