from __future__ import annotations

import inspect
from pathlib import Path

from themis import api as themis_api


def test_evaluate_docs_match_current_signature():
    docs_path = Path("docs/api/evaluate.md")
    text = docs_path.read_text(encoding="utf-8")
    signature = inspect.signature(themis_api.evaluate)

    assert "distributed" not in signature.parameters
    assert "**`distributed`**" not in text
    assert "distributed:" not in text

    assert "max_records_in_memory" in signature.parameters
    assert "**`max_records_in_memory`**" in text


def test_evaluate_docs_list_supported_extra_options():
    docs_path = Path("docs/api/evaluate.md")
    text = docs_path.read_text(encoding="utf-8")

    assert "`top_p`" in text
    for option in themis_api._PROVIDER_OPTION_KEYS:
        assert f"`{option}`" in text
