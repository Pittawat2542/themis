from __future__ import annotations

from pathlib import Path


def test_countdown_tutorial_run_ids_are_consistent():
    text = Path("docs/guides/countdown-tutorial.md").read_text(encoding="utf-8")

    assert "countdown-baseline-spec" in text
    assert "countdown-variant-spec" in text
    assert "countdown-part2-baseline" not in text
    assert "countdown-part2-variant" not in text


def test_countdown_tutorial_has_part_map_and_preflight_sections():
    text = Path("docs/guides/countdown-tutorial.md").read_text(encoding="utf-8")

    assert "| 2 | Session + resume + compare |" in text
    assert "| 3 | Benchmark + export + serve |" in text
    assert "| 4 | Extractor + attempts + CLI ops |" in text
    assert "## Part Requirements" in text
    assert "## Preflight Checks" in text
