from __future__ import annotations

import themis

from tests.constants import EXPECTED_ROOT_EXPORTS


def test_root_exports_only_benchmark_first_surface() -> None:
    assert set(themis.__all__) == EXPECTED_ROOT_EXPORTS
    assert not hasattr(themis, "ExperimentSpec")
    assert not hasattr(themis, "TaskSpec")
    assert not hasattr(themis, "PromptTemplateSpec")
    assert not hasattr(themis, "ItemSamplingSpec")
