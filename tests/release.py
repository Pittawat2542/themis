from __future__ import annotations

from pathlib import Path

import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT_VERSION = tomllib.loads(
    (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
)["project"]["version"]
CURRENT_TAG = f"v{CURRENT_VERSION}"
CURRENT_DIST_BASENAME = f"themis_eval-{CURRENT_VERSION}"
CURRENT_WHEEL = f"{CURRENT_DIST_BASENAME}-py3-none-any.whl"
CURRENT_SDIST = f"{CURRENT_DIST_BASENAME}.tar.gz"
CURRENT_DIST_INFO = f"{CURRENT_DIST_BASENAME}.dist-info"
