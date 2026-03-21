"""Benchmark-first public authoring specs."""

from __future__ import annotations

from pydantic import Field, ValidationInfo, field_validator, model_validator

from themis.benchmark.query import DatasetQuerySpec
from themis.specs.base import SpecBase
from themis.specs.experiment import (
    InferenceGridSpec,
    PromptMessage,
    PromptTurnSpec,
)
from themis.specs.foundational import (
    DatasetSpec,
    ExtractorRefSpec,
    GenerationSpec,
    ModelSpec,
)
from themis.types.json_types import JSONDict


class PromptVariantSpec(SpecBase):
    """Structured prompt variant scoped to one family or benchmark workflow."""

    id: str = Field(..., min_length=1)
    family: str | None = Field(default=None)
    messages: list[PromptMessage] = Field(..., min_length=1)
    follow_up_turns: list[PromptTurnSpec] = Field(default_factory=list)
    variables: JSONDict = Field(
        default_factory=dict,
        description="Static prompt-scoped variables exposed to prompt rendering.",
    )


class ParseSpec(SpecBase):
    """Named parse pipeline backed by one extractor chain."""

    name: str = Field(..., min_length=1)
    extractors: list[ExtractorRefSpec] = Field(default_factory=list)

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


class ScoreSpec(SpecBase):
    """Named scoring pass over raw or parsed candidate outputs."""

    name: str = Field(..., min_length=1)
    parse: str | None = Field(default=None)
    metrics: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_semantic(self) -> "ScoreSpec":
        if not self.metrics:
            raise ValueError("ScoreSpec must define at least one metric.")
        return self


class DatasetSliceSpec(SpecBase):
    """Public dataset-provider contract for one benchmark slice scan request."""

    benchmark_id: str | None = Field(default=None)
    slice_id: str = Field(..., min_length=1)
    dataset: DatasetSpec = Field(...)
    dimensions: dict[str, str] = Field(default_factory=dict)


class SliceSpec(SpecBase):
    """One benchmark slice with dataset identity, queries, prompts, and scoring."""

    slice_id: str = Field(..., min_length=1)
    dataset: DatasetSpec = Field(...)
    dataset_query: DatasetQuerySpec = Field(default_factory=DatasetQuerySpec)
    dimensions: dict[str, str] = Field(default_factory=dict)
    prompt_variant_ids: list[str] = Field(default_factory=list)
    prompt_families: list[str] = Field(default_factory=list)
    generation: GenerationSpec | None = Field(default=None)
    parses: list[ParseSpec] = Field(default_factory=list)
    scores: list[ScoreSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_semantic(self) -> "SliceSpec":
        if self.generation is None and not self.parses and not self.scores:
            raise ValueError(
                f"SliceSpec '{self.slice_id}' must define at least one stage."
            )
        parse_names = [parse.name for parse in self.parses]
        parse_name_set = set(parse_names)
        if len(parse_names) != len(set(parse_names)):
            raise ValueError(f"SliceSpec '{self.slice_id}' has duplicate parse name.")
        score_names = [score.name for score in self.scores]
        if len(score_names) != len(set(score_names)):
            raise ValueError(f"SliceSpec '{self.slice_id}' has duplicate score name.")
        for score in self.scores:
            if score.parse is not None and score.parse not in parse_name_set:
                raise ValueError(
                    f"SliceSpec '{self.slice_id}' references unknown parse "
                    f"'{score.parse}' in score '{score.name}'."
                )
        return self


class BenchmarkSpec(SpecBase):
    """Top-level benchmark configuration compiled into an execution plan."""

    benchmark_id: str = Field(..., min_length=1)
    models: list[ModelSpec] = Field(..., min_length=1)
    slices: list[SliceSpec] = Field(..., min_length=1)
    prompt_variants: list[PromptVariantSpec] = Field(..., min_length=1)
    inference_grid: InferenceGridSpec = Field(...)
    num_samples: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _validate_semantic(self) -> "BenchmarkSpec":
        slice_ids = [slice_spec.slice_id for slice_spec in self.slices]
        if len(slice_ids) != len(set(slice_ids)):
            raise ValueError(
                f"BenchmarkSpec '{self.benchmark_id}' has duplicate slice_id."
            )
        prompt_variant_ids = [
            prompt_variant.id for prompt_variant in self.prompt_variants
        ]
        if len(prompt_variant_ids) != len(set(prompt_variant_ids)):
            raise ValueError(
                f"BenchmarkSpec '{self.benchmark_id}' has duplicate prompt variant id."
            )
        valid_prompt_variant_ids = set(prompt_variant_ids)
        for slice_spec in self.slices:
            missing_prompt_variant_ids = sorted(
                set(slice_spec.prompt_variant_ids) - valid_prompt_variant_ids
            )
            if missing_prompt_variant_ids:
                missing_joined = ", ".join(missing_prompt_variant_ids)
                raise ValueError(
                    f"SliceSpec '{slice_spec.slice_id}' references unknown prompt "
                    f"variant id(s): {missing_joined}."
                )
        return self
