from __future__ import annotations

from collections.abc import Sequence
import inspect
from types import SimpleNamespace
from typing import cast

import pytest

import themis.catalog as catalog
from themis import BenchmarkDefinition, DatasetQuerySpec
from themis.benchmark.specs import DatasetSliceSpec
from themis.catalog.datasets import common as dataset_common
from themis.catalog.datasets.common import (
    _normalize_healthbench_rows,
    _normalize_imo_answerbench_rows,
    _normalize_math_short_answer_rows,
)
from themis.catalog.runtime.metrics.common import MathEquivalenceMetric
from themis.errors import ThemisError
from themis.catalog.runtime.common import _build_judge_spec
from themis.specs.foundational import (
    DatasetSpec,
    JinjaTransform,
    PythonTransform,
    RenameFieldTransform,
)
from themis.types.enums import DatasetSource, ErrorCode
from themis.types.events import ScoreRow
from themis.types.json_types import JSONDict


class _StubProjectionRepo:
    def __init__(self, score_rows: list[ScoreRow]) -> None:
        self._score_rows = list(score_rows)

    def iter_candidate_scores(
        self,
        *,
        trial_hashes: list[str] | None = None,
        metric_id: str | None = None,
        evaluation_hash: str | None = None,
    ):
        del evaluation_hash
        allowed_hashes = set(trial_hashes or [])
        for row in self._score_rows:
            if allowed_hashes and row.trial_hash not in allowed_hashes:
                continue
            if metric_id is not None and row.metric_id != metric_id:
                continue
            yield row


class _StubResult:
    def __init__(
        self,
        score_rows: list[ScoreRow],
        *,
        scan_stats: dict[str, object] | None = None,
    ) -> None:
        self.projection_repo = _StubProjectionRepo(score_rows)
        self.trial_hashes = sorted({row.trial_hash for row in score_rows})
        self.active_evaluation_hash = None
        self._builtin_scan_stats = dict(scan_stats or {})


def _row_mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return {str(key): item for key, item in value.items()}


def _json_mapping(value: object) -> JSONDict:
    assert isinstance(value, dict)
    return cast(JSONDict, value)


def _first_preview_message(preview: Sequence[JSONDict]) -> dict[str, object]:
    messages = preview[0].get("messages")
    assert isinstance(messages, list)
    assert messages
    return _row_mapping(messages[0])


def test_public_catalog_lists_requested_benchmarks() -> None:
    assert set(catalog.list_catalog_benchmarks()) == {
        "aime_2025",
        "aime_2026",
        "apex_2025",
        "beyond_aime",
        "encyclo_k",
        "healthbench",
        "hle",
        "hmmt_feb_2025",
        "hmmt_nov_2025",
        "imo_answerbench",
        "lpfqa",
        "mmlu_pro",
        "simpleqa_verified",
        "supergpqa",
    }


def test_builtin_runtime_defaults_use_8192_generator_tokens() -> None:
    definition = catalog.get_catalog_benchmark("mmlu_pro")

    config = definition.build_runtime_config(
        model_id="demo-model",
        provider="demo",
    )

    assert config.max_tokens == 8192


def test_catalog_definitions_use_generic_benchmark_definition_and_metadata() -> None:
    definition = catalog.get_catalog_benchmark("mmlu_pro")

    assert isinstance(definition, BenchmarkDefinition)
    assert definition.family == "catalog"
    assert definition.primary_metric_id == "choice_accuracy"
    assert definition.metadata["dataset_id"] == "TIGER-Lab/MMLU-Pro"
    assert definition.metadata["split"] == "test"


def test_openai_compatible_benchmarks_use_env_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://127.0.0.1:1234/v1")
    definition = catalog.get_catalog_benchmark("aime_2026")

    benchmark = definition.build_benchmark(
        model_id="demo-model",
        provider="openai_compatible",
    )

    assert benchmark.models[0].extras["base_url"] == "http://127.0.0.1:1234/v1"


def test_builtin_judge_spec_defaults_use_8192_tokens() -> None:
    judge_spec = _build_judge_spec(
        model_id="judge-model",
        provider="demo",
    )

    assert judge_spec.params.max_tokens == 8192


@pytest.mark.parametrize(
    ("benchmark_id", "dataset_id", "split"),
    [
        ("mmlu_pro", "TIGER-Lab/MMLU-Pro", "test"),
        ("supergpqa", "m-a-p/SuperGPQA", "train"),
        ("encyclo_k", "m-a-p/Encyclo-K", "test"),
    ],
)
def test_mcq_benchmark_builders_use_expected_dataset_defaults(
    benchmark_id: str,
    dataset_id: str,
    split: str,
) -> None:
    definition = catalog.get_catalog_benchmark(benchmark_id)

    benchmark = definition.build_benchmark(model_id="demo-model", provider="demo")

    assert benchmark.benchmark_id == benchmark_id
    assert benchmark.slices[0].dataset.dataset_id == dataset_id
    assert benchmark.slices[0].dataset.split == split
    assert benchmark.slices[0].parses[0].extractors[0].id == "choice_letter"
    assert benchmark.slices[0].scores[0].metrics == ["choice_accuracy"]


@pytest.mark.parametrize(
    ("benchmark_id", "dataset_id", "split"),
    [
        ("aime_2026", "MathArena/aime_2026", "train"),
        ("aime_2025", "MathArena/aime_2025", "train"),
        ("hmmt_feb_2025", "MathArena/hmmt_feb_2025", "train"),
        ("hmmt_nov_2025", "MathArena/hmmt_nov_2025", "train"),
        ("apex_2025", "MathArena/apex_2025", "train"),
        ("beyond_aime", "ByteDance-Seed/BeyondAIME", "test"),
        ("imo_answerbench", "Hwilner/imo-answerbench", "train"),
    ],
)
def test_math_benchmark_builders_use_expected_dataset_defaults(
    benchmark_id: str,
    dataset_id: str,
    split: str,
) -> None:
    definition = catalog.get_catalog_benchmark(benchmark_id)

    benchmark = definition.build_benchmark(model_id="demo-model", provider="demo")

    assert benchmark.benchmark_id == benchmark_id
    assert benchmark.slices[0].dataset.dataset_id == dataset_id
    assert benchmark.slices[0].dataset.split == split
    assert benchmark.slices[0].parses[0].extractors[0].id == "math_answer"
    assert benchmark.slices[0].scores[0].metrics == ["math_equivalence"]


@pytest.mark.parametrize(
    "benchmark_id",
    ["simpleqa_verified", "healthbench", "lpfqa", "hle"],
)
def test_judge_backed_benchmark_builders_require_explicit_judge_config(
    benchmark_id: str,
) -> None:
    definition = catalog.get_catalog_benchmark(benchmark_id)

    with pytest.raises(ValueError, match="judge"):
        definition.build_benchmark(model_id="demo-model", provider="demo")


def test_render_preview_uses_dataset_native_messages_for_healthbench() -> None:
    definition = catalog.get_catalog_benchmark("healthbench")

    preview = definition.render_preview(
        model_id="demo-model",
        provider="demo",
        judge_model_id="judge-model",
        judge_provider="demo",
    )

    message = _first_preview_message(preview)

    assert message["role"] == "user"
    assert "postpartum depression" in str(message["content"])


def test_healthbench_row_normalizer_populates_prompt_text_for_runtime_rendering() -> (
    None
):
    normalized = _normalize_healthbench_rows(
        [
            {
                "item_id": "hb-1",
                "prompt_id": "hb-1",
                "prompt": [
                    {"role": "system", "content": "Stay concise."},
                    {"role": "user", "content": "How should I treat a burn?"},
                ],
                "rubrics": [],
            }
        ],
        DatasetSpec(source=DatasetSource.HUGGINGFACE, dataset_id="openai/healthbench"),
    )

    assert normalized.rows[0]["prompt_text"] == (
        "system: Stay concise.\n\nuser: How should I treat a burn?"
    )


def test_render_preview_formats_mcq_prompt_from_fixture_sample() -> None:
    definition = catalog.get_catalog_benchmark("mmlu_pro")

    preview = definition.render_preview(model_id="demo-model", provider="demo")
    message = _first_preview_message(preview)

    assert "Question:" in str(message["content"])
    assert "Return the best option letter only." in str(message["content"])


def test_render_preview_formats_hle_prompt_without_template_errors() -> None:
    definition = catalog.get_catalog_benchmark("hle")

    preview = definition.render_preview(
        model_id="demo-model",
        provider="demo",
        judge_model_id="judge-model",
        judge_provider="demo",
    )

    content = str(_first_preview_message(preview)["content"])
    assert "Explanation:" in content
    assert "Answer:" in content
    assert "Confidence:" in content


def test_render_preview_formats_math_prompt_with_boxed_answer_instruction() -> None:
    definition = catalog.get_catalog_benchmark("aime_2026")

    preview = definition.render_preview(model_id="demo-model", provider="demo")

    content = str(_first_preview_message(preview)["content"])
    assert "Problem:" in content
    assert "\\boxed{" in content


def test_starter_dataset_provider_applies_supported_dataset_transforms() -> None:
    provider = catalog.CatalogDatasetProvider(
        memory_rows=[{"question": "2 + 2", "answer": "4"}]
    )
    slice_spec = DatasetSliceSpec(
        benchmark_id="transform-demo",
        slice_id="qa",
        dataset=DatasetSpec(
            source=DatasetSource.MEMORY,
            transforms=[
                RenameFieldTransform(
                    field="rendered_question",
                    source_field="question",
                ),
                JinjaTransform(
                    field="prompt_text",
                    template="Solve: {rendered_question}",
                ),
            ],
        ),
    )

    rows = list(provider.scan(slice_spec, DatasetQuerySpec()))
    first_row = _row_mapping(rows[0])

    assert first_row["rendered_question"] == "2 + 2"
    assert first_row["prompt_text"] == "Solve: 2 + 2"


def test_starter_dataset_provider_rejects_python_dataset_transforms() -> None:
    provider = catalog.CatalogDatasetProvider(
        memory_rows=[{"question": "2 + 2", "answer": "4"}]
    )
    slice_spec = DatasetSliceSpec(
        benchmark_id="transform-demo",
        slice_id="qa",
        dataset=DatasetSpec(
            source=DatasetSource.MEMORY,
            transforms=[
                PythonTransform(
                    field="normalized",
                    config={"callable": "demo.normalize"},
                )
            ],
        ),
    )

    with pytest.raises(ValueError, match="python"):
        list(provider.scan(slice_spec, DatasetQuerySpec()))


def test_math_short_answer_row_normalizer_maps_matharena_fields() -> None:
    normalized = _normalize_math_short_answer_rows(
        [
            {
                "problem_idx": 4,
                "problem": "Find x.",
                "answer": 17,
                "problem_type": ["Algebra"],
                "source": "fixture",
            }
        ],
        DatasetSpec(source=DatasetSource.HUGGINGFACE, dataset_id="MathArena/aime_2026"),
    )

    row = _row_mapping(normalized.rows[0])
    metadata = _row_mapping(row["metadata"])

    assert row["item_id"] == "4"
    assert row["problem"] == "Find x."
    assert row["answer"] == "17"
    assert metadata["problem_idx"] == "4"
    assert metadata["problem_type"] == "Algebra"
    assert metadata["source"] == "fixture"


def test_math_short_answer_row_normalizer_maps_imo_answerbench_fields() -> None:
    normalized = _normalize_imo_answerbench_rows(
        [
            {
                "Problem ID": "imo-1",
                "Problem": "Prove something.",
                "Short Answer": "\\frac{3}{2}",
                "Category": "Geometry",
                "Subcategory": "3d_geometry",
                "Source": "Sharygin 2008",
            }
        ],
        DatasetSpec(
            source=DatasetSource.HUGGINGFACE,
            dataset_id="Hwilner/imo-answerbench",
        ),
    )

    row = _row_mapping(normalized.rows[0])
    metadata = _row_mapping(row["metadata"])

    assert row["item_id"] == "imo-1"
    assert row["problem"] == "Prove something."
    assert row["answer"] == "\\frac{3}{2}"
    assert metadata["category"] == "Geometry"
    assert metadata["subcategory"] == "3d_geometry"
    assert metadata["source"] == "Sharygin 2008"


def test_build_catalog_registry_registers_multiple_providers() -> None:
    registry = catalog.build_catalog_registry(["demo", "openai"])

    assert registry.has_inference_engine("demo")
    assert registry.has_inference_engine("openai")
    assert registry.has_metric("choice_accuracy")
    assert registry.has_metric("math_equivalence")
    assert inspect.isclass(registry.get_metric_registration("choice_accuracy").factory)
    assert inspect.isclass(registry.get_inference_engine_registration("demo").factory)


@pytest.mark.parametrize(
    ("benchmark_id", "metric_type", "metric_module"),
    [
        (
            "simpleqa_verified",
            "SimpleQAVerifiedJudgeMetric",
            "themis.catalog.benchmarks.simpleqa_verified.metric",
        ),
        (
            "healthbench",
            "HealthBenchRubricMetric",
            "themis.catalog.benchmarks.healthbench.metric",
        ),
        (
            "lpfqa",
            "LPFQAJudgeMetric",
            "themis.catalog.benchmarks.lpfqa.metric",
        ),
        (
            "hle",
            "HLEJudgeMetric",
            "themis.catalog.benchmarks.hle.metric",
        ),
    ],
)
def test_judge_backed_benchmarks_use_judge_modules_grouped_by_type(
    benchmark_id: str,
    metric_type: str,
    metric_module: str,
) -> None:
    definition = catalog.get_catalog_benchmark(benchmark_id)
    registry = catalog.build_catalog_registry("demo")

    definition.register_required_components(
        registry,
        judge_model_id="judge-model",
        judge_provider="demo",
    )

    assert definition.primary_metric_id is not None
    registration = registry.get_metric_registration(definition.primary_metric_id)
    metric = registry.get_metric(definition.primary_metric_id)

    assert callable(registration.factory)
    assert type(metric).__name__ == metric_type
    assert type(metric).__module__ == metric_module


@pytest.mark.parametrize(
    ("benchmark_id", "provider_type", "provider_module"),
    [
        (
            "mmlu_pro",
            "BuiltinMMLUProDatasetProvider",
            "themis.catalog.benchmarks.mmlu_pro",
        ),
        (
            "supergpqa",
            "BuiltinSuperGPQADatasetProvider",
            "themis.catalog.benchmarks.supergpqa",
        ),
        (
            "encyclo_k",
            "BuiltinEncycloKDatasetProvider",
            "themis.catalog.benchmarks.encyclo_k",
        ),
        (
            "simpleqa_verified",
            "BuiltinSimpleQAVerifiedDatasetProvider",
            "themis.catalog.benchmarks.simpleqa_verified.dataset",
        ),
        (
            "healthbench",
            "BuiltinHealthBenchDatasetProvider",
            "themis.catalog.benchmarks.healthbench.dataset",
        ),
        (
            "lpfqa",
            "BuiltinLPFQADatasetProvider",
            "themis.catalog.benchmarks.lpfqa.dataset",
        ),
        (
            "hle",
            "BuiltinHLEDatasetProvider",
            "themis.catalog.benchmarks.hle.dataset",
        ),
        (
            "aime_2026",
            "BuiltinMathArenaDatasetProvider",
            "themis.catalog.benchmarks.aime_2026",
        ),
        (
            "aime_2025",
            "BuiltinMathArenaDatasetProvider",
            "themis.catalog.benchmarks.aime_2025",
        ),
        (
            "hmmt_feb_2025",
            "BuiltinMathArenaDatasetProvider",
            "themis.catalog.benchmarks.hmmt_feb_2025",
        ),
        (
            "hmmt_nov_2025",
            "BuiltinMathArenaDatasetProvider",
            "themis.catalog.benchmarks.hmmt_nov_2025",
        ),
        (
            "apex_2025",
            "BuiltinMathArenaDatasetProvider",
            "themis.catalog.benchmarks.apex_2025",
        ),
        (
            "beyond_aime",
            "BuiltinBeyondAIMEDatasetProvider",
            "themis.catalog.benchmarks.beyond_aime",
        ),
        (
            "imo_answerbench",
            "BuiltinIMOAnswerBenchDatasetProvider",
            "themis.catalog.benchmarks.imo_answerbench",
        ),
    ],
)
def test_builtin_benchmarks_use_benchmark_specific_dataset_providers(
    benchmark_id: str,
    provider_type: str,
    provider_module: str,
) -> None:
    definition = catalog.get_catalog_benchmark(benchmark_id)

    provider = definition.build_dataset_provider()

    assert type(provider).__name__ == provider_type
    assert type(provider).__module__ == provider_module


@pytest.mark.parametrize(
    "benchmark_id",
    ["simpleqa_verified", "healthbench", "lpfqa", "hle"],
)
def test_specialized_dataset_providers_inline_their_own_implementation(
    benchmark_id: str,
) -> None:
    definition = catalog.get_catalog_benchmark(benchmark_id)

    provider = definition.build_dataset_provider()
    direct_base_name = type(provider).__bases__[0].__name__

    assert direct_base_name == "BuiltinDatasetProvider"


def test_simpleqa_summary_uses_f1_and_attempted_math_from_metric_details() -> None:
    definition = catalog.get_catalog_benchmark("simpleqa_verified")
    rows = [
        ScoreRow(
            trial_hash="trial-1",
            candidate_id="cand-1",
            metric_id="simpleqa_verified_score",
            score=1.0,
            details={"grade": "CORRECT", "attempted": True},
        ),
        ScoreRow(
            trial_hash="trial-2",
            candidate_id="cand-2",
            metric_id="simpleqa_verified_score",
            score=0.0,
            details={"grade": "INCORRECT", "attempted": True},
        ),
        ScoreRow(
            trial_hash="trial-3",
            candidate_id="cand-3",
            metric_id="simpleqa_verified_score",
            score=0.0,
            details={"grade": "NOT_ATTEMPTED", "attempted": False},
        ),
    ]

    summary = definition.summarize_result(_StubResult(rows))

    assert summary["count"] == 3
    assert summary["correct_rate"] == pytest.approx(1 / 3)
    assert summary["attempted_rate"] == pytest.approx(2 / 3)
    assert summary["accuracy_given_attempted"] == pytest.approx(0.5)
    assert summary["f1"] == pytest.approx(0.4)


def test_healthbench_summary_reports_mean_score_and_tag_breakdowns() -> None:
    definition = catalog.get_catalog_benchmark("healthbench")
    rows = [
        ScoreRow(
            trial_hash="trial-1",
            candidate_id="cand-1",
            metric_id="healthbench_score",
            score=1.0,
            details={"example_tags": ["theme:communication", "axis:safety"]},
        ),
        ScoreRow(
            trial_hash="trial-2",
            candidate_id="cand-2",
            metric_id="healthbench_score",
            score=0.5,
            details={"example_tags": ["theme:communication"]},
        ),
    ]

    summary = definition.summarize_result(_StubResult(rows))
    tag_means = _json_mapping(summary["tag_means"])

    assert summary["count"] == 2
    assert summary["mean_overall_score"] == pytest.approx(0.75)
    assert tag_means["theme:communication"] == pytest.approx(0.75)
    assert tag_means["axis:safety"] == pytest.approx(1.0)


def test_hle_summary_reports_accuracy_ci_calibration_and_skipped_images() -> None:
    definition = catalog.get_catalog_benchmark("hle")
    rows = [
        ScoreRow(
            trial_hash="trial-1",
            candidate_id="cand-1",
            metric_id="hle_accuracy",
            score=1.0,
            details={"correct": True, "confidence": 100},
        ),
        ScoreRow(
            trial_hash="trial-2",
            candidate_id="cand-2",
            metric_id="hle_accuracy",
            score=0.0,
            details={"correct": False, "confidence": 20},
        ),
    ]

    summary = definition.summarize_result(
        _StubResult(rows, scan_stats={"skipped_image_count": 4})
    )
    confidence_interval_half_width = summary["confidence_interval_half_width"]

    assert summary["count"] == 2
    assert summary["accuracy"] == pytest.approx(0.5)
    assert isinstance(confidence_interval_half_width, int | float)
    assert confidence_interval_half_width > 0.0
    assert summary["calibration_error"] == pytest.approx(0.3)
    assert summary["skipped_image_count"] == 4


def test_inspect_huggingface_dataset_uses_loader_hooks_for_schema_and_samples() -> None:
    summary = catalog.inspect_huggingface_dataset(
        "demo/qa",
        split="test",
        metadata_loader=lambda dataset_id, revision: {
            "dataset_id": dataset_id,
            "gated": False,
            "splits": ["train", "test"],
            "modalities": ["text"],
        },
        row_loader=lambda dataset_id, split, revision: [
            {"question": "2 + 2", "answer": "4"},
            {"question": "3 + 3", "answer": "6"},
        ],
    )
    fields = _json_mapping(summary["fields"])
    samples = summary["samples"]

    assert summary["dataset_id"] == "demo/qa"
    assert summary["splits"] == ["train", "test"]
    assert summary["modalities"] == ["text"]
    assert summary["row_count"] == 2
    assert fields["question"] == "str"
    assert isinstance(samples, list)
    assert len(samples) == 2
    assert summary["suggested_prompt_field"] == "question"
    assert summary["suggested_answer_field"] == "answer"
    assert summary["suggested_item_id_field"] is None
    assert summary["suggested_metadata_keys"] == []


def test_inspect_huggingface_dataset_emits_math_wiring_hints() -> None:
    summary = catalog.inspect_huggingface_dataset(
        "MathArena/aime_2026",
        split="train",
        metadata_loader=lambda dataset_id, revision: {
            "dataset_id": dataset_id,
            "gated": False,
            "splits": ["train"],
            "modalities": ["text"],
        },
        row_loader=lambda dataset_id, split, revision: [
            {
                "problem_idx": 1,
                "problem": "Find x.",
                "answer": "42",
                "problem_type": ["Algebra"],
            }
        ],
    )

    assert summary["suggested_prompt_field"] == "problem"
    assert summary["suggested_answer_field"] == "answer"
    assert summary["suggested_item_id_field"] == "problem_idx"
    assert summary["suggested_metadata_keys"] == ["problem_type"]


def test_math_equivalence_metric_scores_equivalent_answers(monkeypatch) -> None:
    metric = MathEquivalenceMetric()

    fake_math_verify = SimpleNamespace(
        parse=lambda value: f"parsed:{value}",
        verify=lambda gold, answer: gold == "parsed:0.5" and answer == "parsed:1/2",
    )
    monkeypatch.setattr(
        "themis.catalog.runtime.metrics.common.import_optional",
        lambda module_name, *, extra: fake_math_verify,
    )
    candidate = SimpleNamespace(
        best_extraction=lambda: SimpleNamespace(success=True, parsed_answer="1/2")
    )

    score = metric.score(None, candidate, {"expected": "0.5"})

    assert score.metric_id == "math_equivalence"
    assert score.value == 1.0
    assert score.details["candidate_answer"] == "1/2"
    assert score.details["gold_answer"] == "0.5"


def test_math_equivalence_metric_returns_install_hint_when_optional_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric = MathEquivalenceMetric()

    def _raise_missing_optional(module_name: str, *, extra: str):
        raise ThemisError(
            code=ErrorCode.MISSING_OPTIONAL_DEPENDENCY,
            message=f'Install it with `uv add "themis-eval[{extra}]"`.',
        )

    monkeypatch.setattr(
        "themis.catalog.runtime.metrics.common.import_optional",
        _raise_missing_optional,
    )
    candidate = SimpleNamespace(
        best_extraction=lambda: SimpleNamespace(success=True, parsed_answer="1/2")
    )

    score = metric.score(None, candidate, {"expected": "0.5"})

    assert score.value == 0.0
    assert score.error == 'Install it with `uv add "themis-eval[math]"`.'


def test_catalog_preview_rows_are_available_for_all_builtin_benchmarks() -> None:
    for benchmark_id in catalog.list_catalog_benchmarks():
        definition = catalog.get_catalog_benchmark(benchmark_id)
        assert definition.preview_rows_loader is not None
        rows = definition.preview_rows_loader(definition)
        assert rows
        assert isinstance(rows[0], dict)


def test_load_huggingface_rows_disables_image_decoding_for_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeImageFeature:
        def __init__(self, *, decode: bool = True) -> None:
            self.decode = decode

    class _FakeDataset:
        def __init__(self) -> None:
            self.features = {
                "question": "text",
                "image_preview": _FakeImageFeature(),
                "rationale_image": _FakeImageFeature(),
            }
            self.cast_calls: list[tuple[str, bool]] = []

        def cast_column(self, name: str, feature: object):
            assert isinstance(feature, _FakeImageFeature)
            self.cast_calls.append((name, feature.decode))
            self.features[name] = feature
            return self

        def __iter__(self):
            yield {"question": "What is 2 + 2?", "answer": "4"}

    fake_dataset = _FakeDataset()

    class _FakeDatasetsModule:
        Image = _FakeImageFeature

        @staticmethod
        def load_dataset(dataset_id: str, *, split: str, revision: str | None = None):
            assert dataset_id == "cais/hle"
            assert split == "test"
            assert revision is None
            return fake_dataset

    monkeypatch.setattr(
        dataset_common,
        "import_optional",
        lambda module_name, *, extra: _FakeDatasetsModule,
    )

    rows = catalog.load_huggingface_rows("cais/hle", "test")

    assert rows == [{"item_id": "item-1", "question": "What is 2 + 2?", "answer": "4"}]
    assert fake_dataset.cast_calls == [
        ("image_preview", False),
        ("rationale_image", False),
    ]


def test_load_huggingface_rows_retries_healthbench_with_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DatasetGenerationError(Exception):
        pass

    class _FakeStreamingDataset:
        features = {"prompt": "list", "rubrics": "list"}

        def __iter__(self):
            yield {
                "prompt_id": "hb-1",
                "prompt": [{"role": "user", "content": "Help me."}],
                "rubrics": [{"criterion": "safe", "points": 1, "tags": []}],
            }

    class _FakeDatasetsModule:
        DatasetGenerationError = _DatasetGenerationError

        @staticmethod
        def load_dataset(
            dataset_id: str,
            *,
            split: str,
            revision: str | None = None,
            streaming: bool = False,
        ):
            assert dataset_id == "openai/healthbench"
            assert split == "test"
            assert revision is None
            if streaming:
                return _FakeStreamingDataset()
            raise _DatasetGenerationError("broken non-streaming cast")

    monkeypatch.setattr(
        dataset_common,
        "import_optional",
        lambda module_name, *, extra: _FakeDatasetsModule,
    )

    rows = catalog.load_huggingface_rows("openai/healthbench", "test")

    assert rows == [
        {
            "item_id": "item-1",
            "prompt_id": "hb-1",
            "prompt": [{"role": "user", "content": "Help me."}],
            "rubrics": [{"criterion": "safe", "points": 1, "tags": []}],
        }
    ]
