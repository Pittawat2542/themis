#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
  echo "pre-commit: uv is required to check staged Python files" >&2
  exit 1
fi

staged_python=()
while IFS= read -r path; do
  staged_python+=("$path")
done < <(git diff --cached --name-only --diff-filter=ACMR -- '*.py')

if [[ "${#staged_python[@]}" -eq 0 ]]; then
  exit 0
fi

tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/themis-pre-commit.XXXXXX")"
cleanup() {
  rm -rf "$tmp_root"
}
trap cleanup EXIT

checkout_paths=()
while IFS= read -r path; do
  checkout_paths+=("$path")
done < <(git ls-files -- '*.py')
if git ls-files --error-unmatch pyproject.toml >/dev/null 2>&1; then
  checkout_paths+=("pyproject.toml")
fi
if git ls-files --error-unmatch uv.lock >/dev/null 2>&1; then
  checkout_paths+=("uv.lock")
fi
git checkout-index --prefix="$tmp_root/" -- "${checkout_paths[@]}"

staged_snapshot_paths=()
for path in "${staged_python[@]}"; do
  staged_snapshot_paths+=("$tmp_root/$path")
done

uv run --extra dev ruff check --config "$repo_root/pyproject.toml" --force-exclude -- "${staged_snapshot_paths[@]}"
uv run python -m py_compile "${staged_snapshot_paths[@]}"
MYPYPATH="$tmp_root${MYPYPATH:+:$MYPYPATH}" \
  uv run --extra dev --project "$repo_root" mypy "${staged_snapshot_paths[@]}"
