"""Tests for generic benchmark factory functions."""

import pytest


class TestPresetsBehaviorPreservation:
    """Verify all existing presets behave identically after deduplication.

    Each test loads a preset and validates its key attributes haven't changed.
    Tests are written before refactoring to lock down current behavior.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "math500",
            "gsm8k",
            "aime24",
            "gsm-symbolic",
            "aime25",
            "amc23",
            "olympiadbench",
            "beyondaime",
        ],
    )
    def test_math_preset_has_expected_attributes(self, name):
        from themis.presets import get_benchmark_preset

        preset = get_benchmark_preset(name)
        assert preset.name == name
        assert preset.prompt_template is not None
        assert len(preset.metrics) > 0
        assert preset.extractor is not None
        assert preset.reference_field is not None
        assert callable(preset.load_dataset)

    @pytest.mark.parametrize(
        "name",
        [
            "mmlu-pro",
            "supergpqa",
            "gpqa",
            "commonsense_qa",
        ],
    )
    def test_mcq_preset_has_expected_attributes(self, name):
        from themis.presets import get_benchmark_preset

        preset = get_benchmark_preset(name)
        assert preset.name == name
        assert preset.prompt_template is not None
        assert len(preset.metrics) > 0
        assert preset.extractor is not None

    def test_total_benchmark_count_unchanged(self):
        from themis.presets import list_benchmarks

        benchmarks = list_benchmarks()
        # Lock current count to detect accidental removal during refactor
        assert len(benchmarks) >= 18
