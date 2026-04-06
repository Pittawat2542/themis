from __future__ import annotations

from themis import __version__
from tests.release import CURRENT_DIST_BASENAME, CURRENT_TAG, CURRENT_VERSION


def test_release_constants_track_current_package_version() -> None:
    assert CURRENT_VERSION == __version__
    assert CURRENT_TAG == f"v{CURRENT_VERSION}"
    assert CURRENT_DIST_BASENAME == f"themis_eval-{CURRENT_VERSION}"
