#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to run staged Python checks."
  echo "Install uv: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

changed_files=()
while IFS= read -r file; do
  changed_files+=("$file")
done < <(git diff --cached --name-only --diff-filter=ACMRT -- "*.py")
if [[ "${#changed_files[@]}" -eq 0 ]]; then
  exit 0
fi

echo "Running Ruff checks on ${#changed_files[@]} staged Python file(s)..."
uv run ruff format --check "${changed_files[@]}"
uv run ruff check "${changed_files[@]}"
