#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "[part5] running baseline..."
uv run python examples/countdown/run_countdown_baseline.py

echo "[part5] running candidate..."
uv run python examples/countdown/run_countdown_candidate.py

echo "[part5] verifying reproducibility manifest..."
uv run python examples/countdown/verify_reproducibility.py

echo "[part5] running quality gate..."
uv run python examples/countdown/gate_candidate.py

echo "[part5] building research bundle..."
uv run python examples/countdown/build_research_bundle.py

echo "[part5] done"
