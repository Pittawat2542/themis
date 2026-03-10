"""Foundational spec models shared across projects, tasks, models, and datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, ValidationInfo, field_validator, model_validator

from themis.specs.base import SpecBase
from themis.types.json_types import JSONDict

if TYPE_CHECKING:
    from themis.specs.experiment import InferenceParamsSpec


class ModelSpec(SpecBase):
    """Configures one inference-engine target and its provider-specific extras."""

    model_id: str = Field(
        ..., description="The unique name/ID of the model (e.g., 'gpt-4')."
    )
    provider: str = Field(
        ..., description="The provider adapter to route to (e.g., 'openai')."
    )
    extras: JSONDict = Field(
        default_factory=dict, description="Provider-specific initialization arguments."
    )


def _default_judge_params() -> InferenceParamsSpec:
    from themis.specs.experiment import InferenceParamsSpec

    return InferenceParamsSpec()


class JudgeInferenceSpec(SpecBase):
    """Optional judge-model configuration used by judge-backed metrics."""

    model: ModelSpec = Field(
        ..., description="The model configuration to power the judge."
    )
    params: InferenceParamsSpec = Field(
        default_factory=_default_judge_params,
        description="Sampling parameters for the judge model.",
    )
    extras: JSONDict = Field(
        default_factory=dict, description="Metric-specific overrides or config."
    )


class RenameFieldTransform(SpecBase):
    """Copies a source field into a normalized destination field."""

    kind: Literal["rename"] = "rename"
    field: str = Field(..., description="Destination field name.")
    source_field: str = Field(..., description="Input payload field to copy from.")
    config: JSONDict = Field(
        default_factory=dict, description="Optional transform metadata."
    )


class JinjaTransform(SpecBase):
    """Renders a template into a normalized destination field."""

    kind: Literal["jinja"] = "jinja"
    field: str = Field(..., description="Destination field name.")
    template: str = Field(
        ..., description="Template used to derive the destination field value."
    )
    config: JSONDict = Field(
        default_factory=dict, description="Optional transform metadata."
    )


class PythonTransform(SpecBase):
    """References a Python transform implementation via structured config."""

    kind: Literal["python"] = "python"
    field: str = Field(..., description="Destination field name.")
    config: JSONDict = Field(
        default_factory=dict, description="Python transform configuration."
    )


TransformSpec = Annotated[
    RenameFieldTransform | JinjaTransform | PythonTransform,
    Field(discriminator="kind"),
]


class DatasetSpec(SpecBase):
    """Declarative dataset source description passed to a dataset loader."""

    source: str = Field(
        default="huggingface",
        description="Dataset adapter type (huggingface, local, memory).",
    )
    dataset_id: str | None = Field(
        default=None, description="Remote ID or local file path."
    )
    data_dir: str | None = Field(
        default=None, description="Local directory containing the data."
    )
    split: str = Field(default="test", description="Dataset split to evaluate.")
    revision: str | None = Field(
        default=None, description="Git commit or tag for version pinning."
    )
    transforms: list[TransformSpec] = Field(
        default_factory=list, description="Optional dataset normalization transforms."
    )

    @model_validator(mode="after")
    def _validate_semantic(self) -> DatasetSpec:
        if self.source == "huggingface" and not self.dataset_id:
            raise ValueError("DatasetSpec source='huggingface' requires a dataset_id.")
        if self.source == "local" and not (self.data_dir or self.dataset_id):
            raise ValueError(
                "DatasetSpec source='local' requires data_dir or dataset_id path."
            )
        return self


class ExtractorRefSpec(SpecBase):
    """References one extractor plus optional extractor-specific config."""

    id: str = Field(..., description="Registered extractor plugin ID.")
    config: JSONDict = Field(
        default_factory=dict,
        description="Extractor-specific structured configuration.",
    )


class ExtractorChainSpec(SpecBase):
    """Ordered extractor fallback chain."""

    extractors: list[ExtractorRefSpec] = Field(
        default_factory=list,
        description="Extractor plugin refs in fallback order.",
    )

    @field_validator("extractors", mode="before")
    @classmethod
    def _coerce_extractors(cls, value: object, info: ValidationInfo) -> object:
        del info
        if not isinstance(value, list):
            return value
        coerced: list[ExtractorRefSpec | object] = []
        for item in value:
            if isinstance(item, str):
                coerced.append(ExtractorRefSpec(id=item))
            else:
                coerced.append(item)
        return coerced


class TaskSpec(SpecBase):
    """Defines the dataset plus default extraction and scoring policy for a task."""

    task_id: str = Field(
        ..., description="Unique human-readable identifier for this task configuration."
    )
    dataset: DatasetSpec = Field(..., description="The underlying data source.")
    default_extractor_chain: ExtractorChainSpec | None = Field(
        default=None,
        description="Ordered extractor fallback chain applied before metric execution.",
    )
    default_metrics: list[str] = Field(
        default_factory=list,
        description="Metric plugin IDs to evaluate against the extracted output.",
    )

    @model_validator(mode="after")
    def _validate_semantic(self) -> TaskSpec:
        if not self.default_metrics:
            raise ValueError(
                f"TaskSpec '{self.task_id}' must define at least one metric."
            )
        return self
