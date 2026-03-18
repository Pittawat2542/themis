"""Foundational spec models shared across projects, tasks, models, and datasets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import (
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from themis.benchmark.query import DatasetQuerySpec
from themis.specs.base import SpecBase
from themis.types.enums import DatasetSource
from themis.types.json_types import JSONDict
from themis.types.json_validation import validate_json_dict

if TYPE_CHECKING:
    from themis.specs.experiment import InferenceParamsSpec


def _default_dataset_query() -> DatasetQuerySpec:
    return DatasetQuerySpec()


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
    """Optional judge-model configuration used by judge-backed metrics.

    Separate metrics can carry separate judge specs, which is how one candidate
    can be scored by multiple judge prompts or judge models in the same run.
    """

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
    """Declarative dataset source description passed to a dataset loader.

    Dataset identity is part of deterministic planning. Use `revision` when the
    upstream dataset source supports version pinning.
    """

    source: DatasetSource = Field(
        default=DatasetSource.HUGGINGFACE,
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

    @field_validator("source", mode="before")
    @classmethod
    def _coerce_source(cls, value: DatasetSource | str) -> DatasetSource | str:
        if isinstance(value, str):
            return DatasetSource(value)
        return value

    @model_validator(mode="after")
    def _validate_semantic(self) -> DatasetSpec:
        if self.source == DatasetSource.HUGGINGFACE and not self.dataset_id:
            raise ValueError("DatasetSpec source='huggingface' requires a dataset_id.")
        if self.source == DatasetSource.LOCAL and not (
            self.data_dir or self.dataset_id
        ):
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


class GenerationSpec(SpecBase):
    """Marker that a task participates in generation-stage execution."""


class OutputTransformSpec(SpecBase):
    """One named output-transformation stage backed by an extractor chain.

    Transform overlays let you re-extract or normalize existing generated text
    without changing the original generation identity.
    """

    name: str = Field(
        ..., description="Stable transform label referenced by evaluations."
    )
    extractor_chain: ExtractorChainSpec = Field(
        ..., description="Extractor chain used to normalize generated output."
    )


class EvaluationSpec(SpecBase):
    """One named scoring pass that optionally depends on a named output transform.

    The `metrics` list is ordered and can include more than one metric, including
    multiple judge-backed metrics over the same candidate set.
    """

    name: str = Field(..., description="Human-readable evaluation label.")
    transform: str | None = Field(
        default=None,
        description="Optional name of the output transform this evaluation consumes.",
    )
    metrics: list[str] = Field(
        default_factory=list,
        description="Metric plugin IDs to execute for this evaluation.",
    )

    @model_validator(mode="after")
    def _validate_semantic(self) -> EvaluationSpec:
        if not self.metrics:
            raise ValueError("EvaluationSpec must define at least one metric.")
        return self


class TaskSpec(SpecBase):
    """Defines dataset identity plus generation, transform, and evaluation stages.

    Tasks are stage-oriented: generation is optional, transforms normalize stored
    candidates, and evaluations score either the raw or transformed outputs.
    """

    task_id: str = Field(
        ..., description="Unique human-readable identifier for this task configuration."
    )
    dataset: DatasetSpec = Field(..., description="The underlying data source.")
    dataset_query: DatasetQuerySpec | JSONDict | None = Field(
        default_factory=_default_dataset_query,
        description="Query pushed down to dataset providers for this slice.",
    )
    generation: GenerationSpec | None = Field(
        default=None,
        description="Whether this task participates in generation-stage execution.",
    )
    output_transforms: list[OutputTransformSpec] = Field(
        default_factory=list,
        description="Named output transformations available after generation.",
    )
    evaluations: list[EvaluationSpec] = Field(
        default_factory=list,
        description="Named evaluation passes available for transformed outputs.",
    )
    benchmark_id: str | None = Field(
        default=None,
        description="Owning benchmark identifier when compiled from BenchmarkSpec.",
    )
    slice_id: str | None = Field(
        default=None,
        description="Owning slice identifier when compiled from BenchmarkSpec.",
    )
    dimensions: dict[str, str] = Field(
        default_factory=dict,
        description="Semantic benchmark dimensions attached to this slice.",
    )
    allowed_prompt_template_ids: list[str] | None = Field(
        default=None,
        description="Optional allow-list of prompt variant IDs for this slice.",
    )
    prompt_family_filters: list[str] | None = Field(
        default=None,
        description="Optional prompt-family selector preserved from BenchmarkSpec.",
    )

    @field_validator("dataset_query", mode="before")
    @classmethod
    def _coerce_dataset_query(cls, value: object) -> object:
        if value is None:
            return _default_dataset_query()
        if isinstance(value, DatasetQuerySpec):
            return value
        try:
            from themis.specs.experiment import ItemSamplingSpec

            if isinstance(value, ItemSamplingSpec):
                payload = dict(value.model_dump(mode="json"))
                try:
                    return DatasetQuerySpec.model_validate(payload)
                except ValidationError:
                    return validate_json_dict(payload, label="TaskSpec.dataset_query")
        except ImportError:
            pass
        if not isinstance(value, Mapping):
            return value

        try:
            return DatasetQuerySpec.model_validate(dict(value))
        except Exception:
            return validate_json_dict(dict(value), label="TaskSpec.dataset_query")

    @model_validator(mode="after")
    def _validate_semantic(self) -> TaskSpec:
        if (
            self.generation is None
            and not self.output_transforms
            and not self.evaluations
        ):
            raise ValueError(
                f"TaskSpec '{self.task_id}' must define at least one stage."
            )
        transform_names = [transform.name for transform in self.output_transforms]
        if len(transform_names) != len(set(transform_names)):
            raise ValueError(
                f"TaskSpec '{self.task_id}' has duplicate output transform name."
            )
        known_transforms = set(transform_names)
        for evaluation in self.evaluations:
            if (
                evaluation.transform is not None
                and evaluation.transform not in known_transforms
            ):
                raise ValueError(
                    f"TaskSpec '{self.task_id}' references unknown output transform "
                    f"'{evaluation.transform}'."
                )
        return self
