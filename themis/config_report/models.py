"""Structured models for nested configuration reports."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from themis.types.json_types import JSONValueType


class ConfigReportParameter(BaseModel):
    """One reported parameter attached to a config node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    value: JSONValueType
    type_repr: str
    default: JSONValueType | None = None
    has_default: bool = False
    source_file: str | None = None
    source_line: int | None = None
    doc: str | None = None
    inline_comment: str | None = None
    declared_in: str | None = None
    default_visibility: bool | None = Field(default=None, exclude=True)


class ConfigReportNode(BaseModel):
    """One node in the collected config hierarchy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    path: str
    depth: int
    parent_path: str | None = None
    class_name: str
    qualified_class_name: str | None = None
    module_name: str | None = None
    declared_in: str | None = None
    source_file: str | None = None
    source_line: int | None = None
    docstring: str | None = None
    parameters: list[ConfigReportParameter] = Field(default_factory=list)
    children: list["ConfigReportNode"] = Field(default_factory=list)


class ConfigReportHeader(BaseModel):
    """Top-level report header metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    generated_at: str
    git_commit: str | None = None
    project_name: str | None = None
    entrypoint: str | None = None
    root_type: str
    verbosity: str = "full"


class ConfigReportDocument(BaseModel):
    """Final nested configuration report document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    header: ConfigReportHeader
    root: ConfigReportNode
