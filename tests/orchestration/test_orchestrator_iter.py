"""Tests for Orchestrator.run_benchmark_iter() streaming generator."""

from __future__ import annotations

from pathlib import Path


from themis import (
    BenchmarkSpec,
    ExecutionPolicySpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ModelSpec,
    Orchestrator,
    PluginRegistry,
    ProjectSpec,
    PromptMessage,
    PromptVariantSpec,
    ScoreSpec,
    SliceSpec,
    StorageSpec,
)
from themis.contracts.protocols import InferenceResult
from themis.records.evaluation import MetricScore
from themis.records.inference import InferenceRecord
from themis.records.trial import TrialRecord
from themis.specs.foundational import DatasetSpec, GenerationSpec
from themis.types.enums import CompressionCodec, DatasetSource, PromptRole


class _DatasetProvider:
    def scan(self, slice_spec, query):
        del query
        return [
            {"item_id": "item-1", "question": "2 + 2", "answer": "4"},
            {"item_id": "item-2", "question": "3 + 3", "answer": "6"},
        ]


class _Engine:
    def infer(self, trial, context, runtime):
        del runtime
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inf_{context['item_id']}",
                raw_text=str(context["answer"]),
            )
        )


class _ExactMatch:
    def score(self, trial, candidate, context):
        del trial
        actual = candidate.inference.raw_text if candidate.inference else ""
        return MetricScore(
            metric_id="exact_match",
            value=float(actual == context["answer"]),
        )


def _make_project(tmp_path: Path) -> ProjectSpec:
    return ProjectSpec(
        project_name="iter-test",
        researcher_id="tests",
        global_seed=42,
        storage=StorageSpec(
            root_dir=str(tmp_path / "store"),
            compression=CompressionCodec.NONE,
        ),
        execution_policy=ExecutionPolicySpec(),
    )


def _make_benchmark() -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark_id="iter-bench",
        models=[ModelSpec(model_id="demo", provider="demo")],
        slices=[
            SliceSpec(
                slice_id="arithmetic",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                prompt_variant_ids=["qa-default"],
                generation=GenerationSpec(),
                scores=[ScoreSpec(name="default", metrics=["exact_match"])],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="qa-default",
                messages=[
                    PromptMessage(role=PromptRole.USER, content="{item.question}")
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
    )


def _make_orchestrator(tmp_path: Path) -> Orchestrator:
    registry = PluginRegistry()
    registry.register_inference_engine("demo", _Engine())
    registry.register_metric("exact_match", _ExactMatch())
    return Orchestrator.from_project_spec(
        _make_project(tmp_path),
        registry=registry,
        dataset_provider=_DatasetProvider(),
    )


class TestRunBenchmarkIter:
    def test_returns_iterator(self, tmp_path: Path) -> None:
        orchestrator = _make_orchestrator(tmp_path)
        benchmark = _make_benchmark()
        result = orchestrator.run_benchmark_iter(benchmark)
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_yields_trial_records(self, tmp_path: Path) -> None:
        orchestrator = _make_orchestrator(tmp_path)
        benchmark = _make_benchmark()
        records = list(orchestrator.run_benchmark_iter(benchmark))
        assert all(isinstance(r, TrialRecord) for r in records)

    def test_yields_one_record_per_item(self, tmp_path: Path) -> None:
        orchestrator = _make_orchestrator(tmp_path)
        benchmark = _make_benchmark()
        records = list(orchestrator.run_benchmark_iter(benchmark))
        # 2 items × 1 model × 1 prompt variant = 2 trials
        assert len(records) == 2

    def test_trial_records_have_inference(self, tmp_path: Path) -> None:
        orchestrator = _make_orchestrator(tmp_path)
        benchmark = _make_benchmark()
        records = list(orchestrator.run_benchmark_iter(benchmark))
        for record in records:
            candidates = record.candidates
            assert candidates
            assert any(c.inference is not None for c in candidates)

    def test_trial_records_have_scores(self, tmp_path: Path) -> None:
        orchestrator = _make_orchestrator(tmp_path)
        benchmark = _make_benchmark()
        records = list(orchestrator.run_benchmark_iter(benchmark))
        for record in records:
            for candidate in record.candidates:
                assert candidate.evaluation is not None

    def test_records_arrive_before_all_complete(self, tmp_path: Path) -> None:
        """Verify the generator yields lazily — first record available before iterating all."""
        orchestrator = _make_orchestrator(tmp_path)
        benchmark = _make_benchmark()
        gen = orchestrator.run_benchmark_iter(benchmark)
        first = next(gen)
        assert isinstance(first, TrialRecord)
        # consume the rest
        rest = list(gen)
        assert len(rest) == 1  # 2 total - 1 already consumed
