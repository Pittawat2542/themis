#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "[part9] building manifest index..."
uv run python examples/countdown/build_manifest_index.py

echo "[part9] diffing manifests..."
uv run python examples/countdown/diff_manifests.py

echo "[part9] enforcing reproducibility gate..."
uv run python examples/countdown/gate_reproducibility.py

echo "[part9] done"
