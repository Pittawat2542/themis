"""Quick benchmark runner for zero-friction Themis smoke evaluations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Annotated, Any, Literal

from cyclopts import App, Parameter
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from themis import (
    BenchmarkSpec,
    DatasetQuerySpec,
    ExecutionPolicySpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ModelSpec,
    Orchestrator,
    ParseSpec,
    ProjectSpec,
    PromptMessage,
    PromptVariantSpec,
    ScoreSpec,
    SliceSpec,
)
from themis.starter_catalog import (
    StarterDatasetProvider,
    build_builtin_benchmark_project,
    build_starter_registry,
    load_huggingface_rows as _load_huggingface_rows,
    load_local_rows as _load_local_rows,
)
from themis.specs.experiment import SqliteBlobStorageSpec
from themis.specs.foundational import DatasetSpec, ExtractorRefSpec, GenerationSpec
from themis.types.enums import CompressionCodec, DatasetSource, PromptRole
from themis.types.json_types import JSONDict

load_huggingface_rows = _load_huggingface_rows
load_local_rows = _load_local_rows

OUTPUT_FORMATS = {"table", "json"}
METRIC_EXTRACTORS: dict[str, str | None] = {
    "exact_match": None,
    "normalized_exact_match": "normalized_text",
    "choice_accuracy": "choice_letter",
    "numeric_exact_match": "first_number",
}


@dataclass(frozen=True, slots=True)
class QuickEvalConfig:
    mode: Literal["inline", "file", "huggingface", "benchmark"]
    model: str
    provider: str
    metric: str
    prompt: str
    benchmark: str | None
    max_tokens: int
    temperature: float
    top_p: float | None
    seed: int | None
    storage_root: Path
    preview: bool
    estimate_only: bool
    format: Literal["table", "json"]


def build_app() -> App:
    """Build the quick-eval Cyclopts app."""

    app = App(
        name="quick-eval",
        help="Run a tiny benchmark directly from CLI flags.",
    )

    @app.command(name="inline")
    def inline(
        model: str,
        input: Annotated[list[str], Parameter(allow_repeating=True)] = [],
        expected: Annotated[list[str], Parameter(allow_repeating=True)] = [],
        provider: str = "openai",
        metric: str = "exact_match",
        prompt: str = "{item.input}",
        max_tokens: int = 8192,
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int | None = None,
        storage_root: str | None = None,
        preview: bool = False,
        estimate_only: bool = False,
        format: str = "table",
    ) -> int:
        config = _build_config(
            mode="inline",
            model=model,
            provider=provider,
            metric=metric,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            storage_root=storage_root,
            preview=preview,
            estimate_only=estimate_only,
            format=format,
        )
        rows = _build_inline_rows(input, expected)
        return _run_quick_eval(
            config,
            dataset_spec=DatasetSpec(source=DatasetSource.MEMORY),
            dataset_provider=QuickEvalDatasetProvider(
                memory_rows=rows,
            ),
            sample_rows=rows,
        )

    @app.command(name="file")
    def file(
        model: str,
        file: str,
        input_field: str,
        expected_field: str,
        item_id_field: str | None = None,
        provider: str = "openai",
        metric: str = "exact_match",
        prompt: str = "{item.input}",
        max_tokens: int = 8192,
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int | None = None,
        storage_root: str | None = None,
        preview: bool = False,
        estimate_only: bool = False,
        format: str = "table",
    ) -> int:
        config = _build_config(
            mode="file",
            model=model,
            provider=provider,
            metric=metric,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            storage_root=storage_root,
            preview=preview,
            estimate_only=estimate_only,
            format=format,
        )
        file_path = Path(file)
        provider_instance = QuickEvalDatasetProvider(
            input_field=input_field,
            expected_field=expected_field,
            item_id_field=item_id_field,
        )
        sample_rows = provider_instance.normalize_rows(load_local_rows(file_path))
        return _run_quick_eval(
            config,
            dataset_spec=DatasetSpec(
                source=DatasetSource.LOCAL,
                dataset_id=str(file_path),
            ),
            dataset_provider=provider_instance,
            sample_rows=sample_rows,
        )

    @app.command(name="huggingface")
    def huggingface(
        model: str,
        dataset_id: str,
        split: str,
        input_field: str,
        expected_field: str,
        subset: int | None = None,
        revision: str | None = None,
        item_id_field: str | None = None,
        provider: str = "openai",
        metric: str = "exact_match",
        prompt: str = "{item.input}",
        max_tokens: int = 8192,
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int | None = None,
        storage_root: str | None = None,
        preview: bool = False,
        estimate_only: bool = False,
        format: str = "table",
    ) -> int:
        config = _build_config(
            mode="huggingface",
            model=model,
            provider=provider,
            metric=metric,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            storage_root=storage_root,
            preview=preview,
            estimate_only=estimate_only,
            format=format,
        )
        provider_instance = QuickEvalDatasetProvider(
            input_field=input_field,
            expected_field=expected_field,
            item_id_field=item_id_field,
            huggingface_loader=load_huggingface_rows,
        )
        try:
            sample_rows = provider_instance.normalize_rows(
                load_huggingface_rows(dataset_id, split, revision)
            )
        except Exception as exc:
            return _emit_quick_eval_error(exc)
        return _run_quick_eval(
            config,
            dataset_spec=DatasetSpec(
                source=DatasetSource.HUGGINGFACE,
                dataset_id=dataset_id,
                split=split,
                revision=revision,
            ),
            dataset_provider=provider_instance,
            sample_rows=sample_rows,
            subset=subset,
        )

    @app.command(name="benchmark")
    def benchmark(
        benchmark: str,
        model: str,
        provider: str = "openai",
        subset: int | None = None,
        revision: str | None = None,
        judge_model: str | None = None,
        judge_provider: str | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int | None = None,
        storage_root: str | None = None,
        preview: bool = False,
        estimate_only: bool = False,
        format: str = "table",
    ) -> int:
        config = _build_config(
            mode="benchmark",
            model=model,
            provider=provider,
            metric="builtin",
            prompt=f"builtin:{benchmark}",
            benchmark=benchmark,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            storage_root=storage_root,
            preview=preview,
            estimate_only=estimate_only,
            format=format,
        )
        return _run_builtin_benchmark(
            config,
            benchmark_id=benchmark,
            subset=subset,
            dataset_revision=revision,
            judge_model_id=judge_model,
            judge_provider=judge_provider,
        )

    return app


class QuickEvalDatasetProvider(StarterDatasetProvider):
    """Starter dataset provider that maps user fields onto `input` and `expected`."""

    def __init__(
        self,
        *,
        memory_rows: list[dict[str, object]] | None = None,
        input_field: str | None = None,
        expected_field: str | None = None,
        item_id_field: str | None = None,
        huggingface_loader=None,
    ) -> None:
        super().__init__(
            memory_rows=memory_rows,
            huggingface_loader=huggingface_loader or load_huggingface_rows,
            local_loader=load_local_rows,
        )
        self.input_field = input_field
        self.expected_field = expected_field
        self.item_id_field = item_id_field

    def scan(self, slice_spec, query):
        rows = super().scan(slice_spec, query)
        return self.normalize_rows(list(rows))

    def normalize_rows(
        self, rows: Sequence[dict[str, object]]
    ) -> list[dict[str, object]]:
        normalized: list[dict[str, object]] = []
        for index, row in enumerate(rows, start=1):
            payload = dict(row)
            if self.input_field is not None:
                if self.input_field not in payload:
                    raise ValueError(
                        f"Dataset rows must include input field '{self.input_field}'."
                    )
                payload["input"] = payload[self.input_field]
            else:
                payload.setdefault("input", payload.get("input", ""))
            if self.expected_field is not None:
                if self.expected_field not in payload:
                    raise ValueError(
                        "Dataset rows must include expected field "
                        f"'{self.expected_field}'."
                    )
                payload["expected"] = payload[self.expected_field]
            else:
                payload.setdefault("expected", payload.get("expected", ""))
            if self.item_id_field is not None:
                if self.item_id_field not in payload:
                    raise ValueError(
                        f"Dataset rows must include item id field '{self.item_id_field}'."
                    )
                payload["item_id"] = payload[self.item_id_field]
            payload.setdefault("item_id", f"item-{index}")
            normalized.append(payload)
        return normalized


def _run_quick_eval(
    config: QuickEvalConfig,
    *,
    dataset_spec: DatasetSpec,
    dataset_provider: QuickEvalDatasetProvider,
    sample_rows: list[dict[str, object]],
    subset: int | None = None,
) -> int:
    try:
        benchmark = _build_benchmark(config, dataset_spec=dataset_spec, subset=subset)
        if not sample_rows:
            raise ValueError("Quick eval requires at least one dataset row.")
        registry = build_starter_registry(config.provider)
        project = _build_project(config)
        orchestrator = Orchestrator.from_project_spec(
            project,
            registry=registry,
            dataset_provider=dataset_provider,
        )
        payload: dict[str, Any] = {
            "mode": config.mode,
            "model": config.model,
            "provider": config.provider,
            "metric": config.metric,
            "prompt": config.prompt,
            "storage_root": str(config.storage_root),
        }
        if config.preview:
            preview = benchmark.preview(sample_rows[0])
            payload["preview"] = preview
            _emit_quick_eval_output(payload, format=config.format)
            return 0
        if config.estimate_only:
            estimate = orchestrator.estimate(benchmark)
            payload["estimate"] = estimate.model_dump(mode="json")
            _emit_quick_eval_output(payload, format=config.format)
            return 0
        result = orchestrator.run_benchmark(benchmark)
        rows = result.aggregate(
            group_by=["model_id", "slice_id", "metric_id", "prompt_variant_id"]
        )
        payload["rows"] = rows
        payload["sqlite_db"] = str(config.storage_root / "themis.sqlite3")
        _emit_quick_eval_output(payload, format=config.format)
        return 0
    except Exception as exc:
        return _emit_quick_eval_error(exc)


def _run_builtin_benchmark(
    config: QuickEvalConfig,
    *,
    benchmark_id: str,
    subset: int | None,
    dataset_revision: str | None,
    judge_model_id: str | None,
    judge_provider: str | None,
) -> int:
    try:
        (
            project,
            benchmark,
            registry,
            dataset_provider,
            definition,
        ) = build_builtin_benchmark_project(
            benchmark_id=benchmark_id,
            model_id=config.model,
            provider=config.provider,
            storage_root=config.storage_root,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            seed=config.seed,
            subset=subset,
            dataset_revision=dataset_revision,
            judge_model_id=judge_model_id,
            judge_provider=judge_provider,
        )
        payload: dict[str, Any] = {
            "mode": config.mode,
            "benchmark": definition.benchmark_id,
            "model": config.model,
            "provider": config.provider,
            "metric": definition.metric_id,
            "storage_root": str(config.storage_root),
        }
        if config.preview:
            payload["preview"] = definition.render_preview(
                model_id=config.model,
                provider=config.provider,
                judge_model_id=judge_model_id,
                judge_provider=judge_provider,
            )
            _emit_quick_eval_output(payload, format=config.format)
            return 0

        orchestrator = Orchestrator.from_project_spec(
            project,
            registry=registry,
            dataset_provider=dataset_provider,
        )
        if config.estimate_only:
            estimate = orchestrator.estimate(benchmark)
            payload["estimate"] = estimate.model_dump(mode="json")
            _emit_quick_eval_output(payload, format=config.format)
            return 0

        result = orchestrator.run_benchmark(benchmark)
        setattr(result, "_builtin_scan_stats", dataset_provider.last_scan_stats())
        payload["rows"] = result.aggregate(
            group_by=["model_id", "slice_id", "metric_id", "prompt_variant_id"]
        )
        payload["summary"] = definition.summarize_result(result)
        payload["sqlite_db"] = str(config.storage_root / "themis.sqlite3")
        _emit_quick_eval_output(payload, format=config.format)
        return 0
    except Exception as exc:
        return _emit_quick_eval_error(exc)


def _build_benchmark(
    config: QuickEvalConfig,
    *,
    dataset_spec: DatasetSpec,
    subset: int | None = None,
) -> BenchmarkSpec:
    parse_extractor = METRIC_EXTRACTORS.get(config.metric)
    if config.metric not in METRIC_EXTRACTORS:
        raise ValueError(
            "Unsupported quick-eval metric. Choose one of: "
            + ", ".join(sorted(METRIC_EXTRACTORS))
        )
    benchmark_id = _slugify(f"{config.mode}-{config.model}-{config.metric}")
    prompt_variant_id = f"{benchmark_id}-default"
    parses = []
    score = ScoreSpec(name="default", metrics=[config.metric])
    if parse_extractor is not None:
        parses = [
            ParseSpec(
                name="parsed",
                extractors=[ExtractorRefSpec(id=parse_extractor)],
            )
        ]
        score = ScoreSpec(name="default", parse="parsed", metrics=[config.metric])
    query = (
        DatasetQuerySpec.subset(subset, seed=config.seed)
        if subset is not None
        else DatasetQuerySpec()
    )
    return BenchmarkSpec(
        benchmark_id=benchmark_id,
        models=[
            ModelSpec(
                model_id=config.model,
                provider=config.provider,
                extras=_provider_model_extras(config.provider),
            )
        ],
        slices=[
            SliceSpec(
                slice_id="quick-eval",
                dataset=dataset_spec,
                dataset_query=query,
                prompt_variant_ids=[prompt_variant_id],
                generation=GenerationSpec(),
                parses=parses,
                scores=[score],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id=prompt_variant_id,
                family="quick-eval",
                messages=[
                    PromptMessage(role=PromptRole.USER, content=config.prompt),
                ],
            )
        ],
        inference_grid=InferenceGridSpec(
            params=[
                InferenceParamsSpec(
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    seed=config.seed,
                )
            ]
        ),
    )


def _build_project(config: QuickEvalConfig) -> ProjectSpec:
    return ProjectSpec(
        project_name=f"quick-eval-{config.mode}",
        researcher_id="themis-cli",
        global_seed=config.seed or 7,
        storage=SqliteBlobStorageSpec(
            root_dir=str(config.storage_root),
            compression=CompressionCodec.NONE,
        ),
        execution_policy=ExecutionPolicySpec(),
    )


def _build_config(
    *,
    mode: Literal["inline", "file", "huggingface", "benchmark"],
    model: str,
    provider: str,
    metric: str,
    prompt: str,
    benchmark: str | None = None,
    max_tokens: int,
    temperature: float,
    top_p: float | None,
    seed: int | None,
    storage_root: str | None,
    preview: bool,
    estimate_only: bool,
    format: str,
) -> QuickEvalConfig:
    normalized_provider = _normalize_provider_name(provider)
    if format not in OUTPUT_FORMATS:
        raise ValueError("--format must be one of: table, json.")
    resolved_format: Literal["table", "json"] = "json" if format == "json" else "table"
    root = (
        Path(storage_root)
        if storage_root is not None
        else Path(".cache/themis/quick-eval") / _slugify(f"{mode}-{model}-{metric}")
    )
    return QuickEvalConfig(
        mode=mode,
        model=model,
        provider=normalized_provider,
        metric=metric,
        prompt=prompt,
        benchmark=benchmark,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        storage_root=root,
        preview=preview,
        estimate_only=estimate_only,
        format=resolved_format,
    )


def _build_inline_rows(
    inputs: Sequence[str], expected_values: Sequence[str]
) -> list[dict[str, object]]:
    if not inputs:
        raise ValueError("Inline quick eval requires at least one --input value.")
    if len(inputs) != len(expected_values):
        raise ValueError("--input and --expected must have the same number of values.")
    return [
        {"item_id": f"item-{index}", "input": prompt_input, "expected": answer}
        for index, (prompt_input, answer) in enumerate(
            zip(inputs, expected_values, strict=True),
            start=1,
        )
    ]


def _provider_model_extras(provider: str) -> JSONDict:
    if provider == "openai_compatible":
        return {
            "base_url": "http://127.0.0.1:8000/v1",
            "timeout_seconds": 60.0,
        }
    return {}


def _emit_quick_eval_output(
    payload: dict[str, Any], *, format: Literal["table", "json"]
) -> None:
    if format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    console = Console()
    header_lines = [f"mode={payload['mode']}"]
    if "benchmark" in payload:
        header_lines.append(f"benchmark={payload['benchmark']}")
    header_lines.extend(
        [
            f"model={payload['model']}",
            f"provider={payload['provider']}",
            f"metric={payload['metric']}",
            f"storage={payload['storage_root']}",
        ]
    )
    console.print(
        Panel.fit(
            "\n".join(header_lines),
            title="Quick Eval",
        )
    )
    if "preview" in payload:
        preview_table = Table(title="Preview")
        preview_table.add_column("Prompt Variant")
        preview_table.add_column("Messages")
        for preview in payload["preview"]:
            messages = "\n".join(
                f"[{message['role']}] {message['content']}"
                for message in preview["messages"]
            )
            preview_table.add_row(preview["prompt_variant_id"], messages)
        console.print(preview_table)
        return
    if "estimate" in payload:
        estimate = payload["estimate"]
        estimate_table = Table(title="Estimate")
        estimate_table.add_column("Trial Count")
        estimate_table.add_column("Total Work Items")
        estimate_table.add_column("Estimated Tokens")
        estimate_table.add_row(
            str(estimate["trial_count"]),
            str(estimate["total_work_items"]),
            str(estimate["estimated_total_tokens"]),
        )
        console.print(estimate_table)
        return
    results_table = Table(title="Results")
    results_table.add_column("Model")
    results_table.add_column("Slice")
    results_table.add_column("Metric")
    results_table.add_column("Prompt Variant")
    results_table.add_column("Mean")
    results_table.add_column("Count")
    for row in payload.get("rows", []):
        results_table.add_row(
            str(row.get("model_id", "")),
            str(row.get("slice_id", "")),
            str(row.get("metric_id", "")),
            str(row.get("prompt_variant_id", "")),
            str(row.get("mean", "")),
            str(row.get("count", "")),
        )
    console.print(results_table)
    if "sqlite_db" in payload:
        console.print(f"SQLite DB: {payload['sqlite_db']}")
    if "summary" in payload:
        summary_table = Table(title="Benchmark Summary")
        summary_table.add_column("Field")
        summary_table.add_column("Value")
        for key, value in payload["summary"].items():
            summary_table.add_row(str(key), str(value))
        console.print(summary_table)


def _emit_quick_eval_error(exc: Exception) -> int:
    Console(stderr=True, markup=False).print(str(exc))
    return 1


def _normalize_provider_name(provider: str) -> str:
    return provider.replace("-", "_")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "quick-eval"
