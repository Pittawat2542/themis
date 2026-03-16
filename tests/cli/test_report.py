from __future__ import annotations

import pytest

from themis.cli.report import _load_project_spec
from themis.errors import SpecValidationError
from themis.types.enums import ErrorCode


def test_load_project_spec_wraps_json_decode_errors(tmp_path) -> None:
    project_path = tmp_path / "project.json"
    project_path.write_text('{"project_name": ')

    with pytest.raises(SpecValidationError) as exc_info:
        _load_project_spec(str(project_path))

    assert exc_info.value.code == ErrorCode.SCHEMA_MISMATCH
    assert "Failed to parse project config project.json" in str(exc_info.value)


def test_load_project_spec_wraps_unsupported_extension(tmp_path) -> None:
    project_path = tmp_path / "project.yaml"
    project_path.write_text("project_name: demo\n")

    with pytest.raises(SpecValidationError) as exc_info:
        _load_project_spec(str(project_path))

    assert exc_info.value.code == ErrorCode.SCHEMA_MISMATCH
    assert "Project files must use .toml or .json." in str(exc_info.value)
