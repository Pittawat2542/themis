"""Benchmark-first public authoring specs."""

from __future__ import annotations

from typing import Any

from pydantic import Field, ValidationInfo, field_validator, model_validator

from themis.benchmark.query import DatasetQuerySpec
from themis.prompting import render_follow_up_turns, render_prompt_messages
from themis.specs.base import SpecBase
from themis.specs.experiment import (
    InferenceGridSpec,
    InferenceParamsSpec,
    PromptMessage,
    PromptTurnSpec,
)
from themis.types.enums import DatasetSource, PromptRole
from themis.specs.foundational import (
    DatasetSpec,
    ExtractorRefSpec,
    GenerationSpec,
    McpServerSpec,
    ModelSpec,
    ToolSpec,
    _validate_unique_mcp_server_ids,
    _validate_unique_tool_ids,
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
        description=(
            "Static prompt-scoped variables injected into the ``{prompt.*}`` "
            "rendering namespace. These are defined once per variant and do not "
            "change per item. For per-item dynamic values, use ``{item.<field>}`` "
            "in message content — those are resolved from the dataset item payload "
            "at render time. Example: ``variables={'tone': 'concise'}`` combined "
            "with a message template ``'Respond in a {prompt.tone} style: {item.question}'``."
        ),
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
    """Public dataset-provider contract for one benchmark slice scan request.

    This is the read-only view that ``DatasetProvider.scan()`` receives.  It
    carries only the fields needed for a provider to decide *which* items to
    return — dataset identity, benchmark context, and semantic dimensions.

    It is **not** the authoring spec.  If you are writing a ``BenchmarkSpec``
    you should use :class:`SliceSpec` instead, which includes prompt variant
    selection, parse pipelines, score passes, and dataset query controls.
    """

    benchmark_id: str | None = Field(default=None)
    slice_id: str = Field(..., min_length=1)
    dataset: DatasetSpec = Field(...)
    dimensions: dict[str, str] = Field(default_factory=dict)


class SliceSpec(SpecBase):
    """One benchmark slice with dataset identity, queries, prompts, and scoring.

    This is the **full authoring spec** used inside :class:`BenchmarkSpec`.  It
    owns the complete pipeline definition for one slice: which dataset to use,
    how to query/sample it, which prompt variants to run, which tools to pass,
    and how to parse and score outputs.

    The narrower :class:`DatasetSliceSpec` is what your ``DatasetProvider.scan()``
    implementation receives — it contains only the fields needed to resolve items,
    not the full pipeline definition.
    """

    slice_id: str = Field(..., min_length=1)
    dataset: DatasetSpec = Field(...)
    dataset_query: DatasetQuerySpec = Field(default_factory=DatasetQuerySpec)
    dimensions: dict[str, str] = Field(default_factory=dict)
    prompt_variant_ids: list[str] = Field(default_factory=list)
    prompt_families: list[str] = Field(default_factory=list)
    tool_ids: list[str] = Field(default_factory=list)
    mcp_server_ids: list[str] = Field(default_factory=list)
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
        if len(self.tool_ids) != len(set(self.tool_ids)):
            raise ValueError(f"SliceSpec '{self.slice_id}' has duplicate tool id.")
        if len(self.mcp_server_ids) != len(set(self.mcp_server_ids)):
            raise ValueError(
                f"SliceSpec '{self.slice_id}' has duplicate MCP server id."
            )
        return self


class BenchmarkSpec(SpecBase):
    """Top-level benchmark configuration compiled into an execution plan."""

    benchmark_id: str = Field(..., min_length=1)
    models: list[ModelSpec] = Field(..., min_length=1)
    slices: list[SliceSpec] = Field(..., min_length=1)
    prompt_variants: list[PromptVariantSpec] = Field(..., min_length=1)
    tools: list[ToolSpec] = Field(default_factory=list)
    mcp_servers: list[McpServerSpec] = Field(default_factory=list)
    inference_grid: InferenceGridSpec = Field(...)
    num_samples: int = Field(default=1, ge=1)

    @classmethod
    def simple(
        cls,
        benchmark_id: str,
        *,
        model_id: str,
        dataset_source: DatasetSource,
        dataset_id: str,
        prompt: str,
        metric: str,
        provider: str = "openai",
        slice_id: str | None = None,
    ) -> "BenchmarkSpec":
        """Construct a minimal single-model, single-slice benchmark for quick exploration.

        This factory reduces the boilerplate for common "run one model against one
        dataset with one prompt" workflows.  It produces a fully valid
        :class:`BenchmarkSpec` that can be passed directly to
        ``Orchestrator.run_benchmark()``.

        For more control (multiple models, prompt variants, inference parameter
        sweeps, custom parse pipelines) build a :class:`BenchmarkSpec` directly.

        Args:
            benchmark_id: Stable benchmark identifier.
            model_id: The model to evaluate (e.g. ``"gpt-4o"``).
            dataset_source: Dataset source enum value (e.g. ``DatasetSource.HUGGINGFACE``).
            dataset_id: Dataset identifier string passed to the dataset provider.
            prompt: User-turn prompt template. Use ``{item.<field>}`` placeholders for
                    dataset item values (e.g. ``"Solve: {item.question}"``).
            metric: Registered metric name used to score outputs (e.g. ``"exact_match"``).
            provider: Inference engine provider name. Defaults to ``"openai"``.
            slice_id: Override the slice ID. Defaults to ``benchmark_id``.

        Returns:
            A :class:`BenchmarkSpec` with one model, one slice, one prompt variant,
            and one scoring pass.

        Example::

            benchmark = BenchmarkSpec.simple(
                benchmark_id="gsm8k-quick",
                model_id="gpt-4o",
                dataset_source=DatasetSource.HUGGINGFACE,
                dataset_id="openai/gsm8k",
                prompt="Solve step by step: {item.question}",
                metric="exact_match",
            )
            result = orchestrator.run_benchmark(benchmark)
        """
        resolved_slice_id = slice_id or benchmark_id
        variant_id = f"{benchmark_id}-default"
        return cls(
            benchmark_id=benchmark_id,
            models=[ModelSpec(model_id=model_id, provider=provider)],
            slices=[
                SliceSpec(
                    slice_id=resolved_slice_id,
                    dataset=DatasetSpec(source=dataset_source, dataset_id=dataset_id),
                    prompt_variant_ids=[variant_id],
                    generation=GenerationSpec(),
                    scores=[ScoreSpec(name="default", metrics=[metric])],
                )
            ],
            prompt_variants=[
                PromptVariantSpec(
                    id=variant_id,
                    messages=[PromptMessage(role=PromptRole.USER, content=prompt)],
                )
            ],
            inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
        )

    def preview(
        self,
        item: dict[str, object],
        *,
        prompt_variant_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Dry-run render all prompt variants against a sample item dict.

        Returns one entry per matching prompt variant showing exactly what
        messages (and follow-up turns) the model would receive for ``item``,
        without executing inference or touching storage.

        This is useful for:

        - Verifying prompt templates look correct before a full run
        - Generating human-readable documentation of prompt variants
        - Debugging ``{item.*}`` and ``{prompt.*}`` rendering placeholders

        Args:
            item: A sample dataset item payload dict (e.g. ``{"question": "2+2"}``).
            prompt_variant_ids: Optional allow-list of variant IDs to preview.
                When ``None``, all variants defined in the benchmark are rendered.

        Returns:
            A list of dicts, one per rendered variant, each with keys:
            ``prompt_variant_id``, ``messages``, and ``follow_up_turns``.
        """
        variants = [
            v
            for v in self.prompt_variants
            if prompt_variant_ids is None or v.id in prompt_variant_ids
        ]
        results: list[dict[str, Any]] = []
        for variant in variants:
            namespaces: dict[str, Any] = {
                "item": item,
                "prompt": {
                    "family": variant.family,
                    "variables": variant.variables,
                    **{
                        k: v
                        for k, v in variant.variables.items()
                        if k not in ("family", "variables")
                    },
                },
            }
            rendered_messages = render_prompt_messages(variant.messages, namespaces)
            rendered_turns = render_follow_up_turns(variant.follow_up_turns, namespaces)
            results.append(
                {
                    "prompt_variant_id": variant.id,
                    "messages": rendered_messages,
                    "follow_up_turns": rendered_turns,
                }
            )
        return results

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
        _validate_unique_tool_ids(self.tools, owner_label="BenchmarkSpec")
        _validate_unique_mcp_server_ids(self.mcp_servers, owner_label="BenchmarkSpec")
        return self
