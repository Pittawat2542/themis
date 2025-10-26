import pytest

from themis.datasets import math500


def test_load_math500_rejects_unknown_source():
    with pytest.raises(ValueError, match="Unsupported source 'unknown'"):
        math500.load_math500(source="unknown")


def test_load_math500_requires_data_dir_for_local():
    with pytest.raises(ValueError, match="data_dir must be provided"):
        math500.load_math500(source="local", data_dir=None)
