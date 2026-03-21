from __future__ import annotations

from pathlib import Path

import pytest

import themis.starter_catalog as starter_catalog
from themis import DatasetQuerySpec
from themis.benchmark.specs import DatasetSliceSpec
from themis.specs.foundational import DatasetSpec
from themis.types.enums import DatasetSource
from themis.types.events import ScoreRow


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


def test_public_catalog_lists_requested_benchmarks() -> None:
    assert set(starter_catalog.list_builtin_benchmarks()) == {
        "encyclo_k",
        "healthbench",
        "hle",
        "lpfqa",
        "mmlu_pro",
        "simpleqa_verified",
        "supergpqa",
    }


def test_builtin_runtime_defaults_use_8192_generator_tokens() -> None:
    definition = starter_catalog.get_builtin_benchmark("mmlu_pro")

    config = definition.build_runtime_config(
        model_id="demo-model",
        provider="demo",
    )

    assert config.max_tokens == 8192


def test_builtin_judge_spec_defaults_use_8192_tokens() -> None:
    judge_spec = starter_catalog._build_judge_spec(
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
    definition = starter_catalog.get_builtin_benchmark(benchmark_id)

    benchmark = definition.build_benchmark(model_id="demo-model", provider="demo")

    assert benchmark.benchmark_id == benchmark_id
    assert benchmark.slices[0].dataset.dataset_id == dataset_id
    assert benchmark.slices[0].dataset.split == split
    assert benchmark.slices[0].parses[0].extractors[0].id == "choice_letter"
    assert benchmark.slices[0].scores[0].metrics == ["choice_accuracy"]


@pytest.mark.parametrize(
    "benchmark_id",
    ["simpleqa_verified", "healthbench", "lpfqa", "hle"],
)
def test_judge_backed_benchmark_builders_require_explicit_judge_config(
    benchmark_id: str,
) -> None:
    definition = starter_catalog.get_builtin_benchmark(benchmark_id)

    with pytest.raises(ValueError, match="judge"):
        definition.build_benchmark(model_id="demo-model", provider="demo")


def test_render_preview_uses_dataset_native_messages_for_healthbench() -> None:
    definition = starter_catalog.get_builtin_benchmark("healthbench")

    preview = definition.render_preview(
        model_id="demo-model",
        provider="demo",
        judge_model_id="judge-model",
        judge_provider="demo",
    )

    assert preview[0]["messages"][0]["role"] == "user"
    assert "postpartum depression" in preview[0]["messages"][0]["content"]


def test_healthbench_row_normalizer_populates_prompt_text_for_runtime_rendering() -> (
    None
):
    normalized = starter_catalog._normalize_healthbench_rows(
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
    definition = starter_catalog.get_builtin_benchmark("mmlu_pro")

    preview = definition.render_preview(model_id="demo-model", provider="demo")

    assert "Question:" in preview[0]["messages"][0]["content"]
    assert "Return the best option letter only." in preview[0]["messages"][0]["content"]


def test_render_preview_formats_hle_prompt_without_template_errors() -> None:
    definition = starter_catalog.get_builtin_benchmark("hle")

    preview = definition.render_preview(
        model_id="demo-model",
        provider="demo",
        judge_model_id="judge-model",
        judge_provider="demo",
    )

    content = preview[0]["messages"][0]["content"]
    assert "Explanation:" in content
    assert "Answer:" in content
    assert "Confidence:" in content


def test_starter_dataset_provider_applies_supported_dataset_transforms() -> None:
    provider = starter_catalog.StarterDatasetProvider(
        memory_rows=[{"question": "2 + 2", "answer": "4"}]
    )
    slice_spec = DatasetSliceSpec(
        benchmark_id="transform-demo",
        slice_id="qa",
        dataset=DatasetSpec(
            source=DatasetSource.MEMORY,
            transforms=[
                {
                    "kind": "rename",
                    "field": "rendered_question",
                    "source_field": "question",
                },
                {
                    "kind": "jinja",
                    "field": "prompt_text",
                    "template": "Solve: {rendered_question}",
                },
            ],
        ),
    )

    rows = list(provider.scan(slice_spec, DatasetQuerySpec()))

    assert rows[0]["rendered_question"] == "2 + 2"
    assert rows[0]["prompt_text"] == "Solve: 2 + 2"


def test_starter_dataset_provider_rejects_python_dataset_transforms() -> None:
    provider = starter_catalog.StarterDatasetProvider(
        memory_rows=[{"question": "2 + 2", "answer": "4"}]
    )
    slice_spec = DatasetSliceSpec(
        benchmark_id="transform-demo",
        slice_id="qa",
        dataset=DatasetSpec(
            source=DatasetSource.MEMORY,
            transforms=[
                {
                    "kind": "python",
                    "field": "normalized",
                    "config": {"callable": "demo.normalize"},
                }
            ],
        ),
    )

    with pytest.raises(ValueError, match="python"):
        list(provider.scan(slice_spec, DatasetQuerySpec()))


def test_build_starter_registry_registers_multiple_providers() -> None:
    registry = starter_catalog.build_starter_registry(["demo", "openai"])

    assert registry.has_inference_engine("demo")
    assert registry.has_inference_engine("openai")
    assert registry.has_metric("choice_accuracy")


@pytest.mark.parametrize(
    ("benchmark_id", "provider_type", "provider_module"),
    [
        (
            "mmlu_pro",
            "BuiltinMMLUProDatasetProvider",
            "themis._starter_catalog.datasets.mmlu_pro",
        ),
        (
            "supergpqa",
            "BuiltinSuperGPQADatasetProvider",
            "themis._starter_catalog.datasets.supergpqa",
        ),
        (
            "encyclo_k",
            "BuiltinEncycloKDatasetProvider",
            "themis._starter_catalog.datasets.encyclo_k",
        ),
        (
            "simpleqa_verified",
            "BuiltinSimpleQAVerifiedDatasetProvider",
            "themis._starter_catalog.datasets.simpleqa_verified",
        ),
        (
            "healthbench",
            "BuiltinHealthBenchDatasetProvider",
            "themis._starter_catalog.datasets.healthbench",
        ),
        (
            "lpfqa",
            "BuiltinLPFQADatasetProvider",
            "themis._starter_catalog.datasets.lpfqa",
        ),
        (
            "hle",
            "BuiltinHLEDatasetProvider",
            "themis._starter_catalog.datasets.hle",
        ),
    ],
)
def test_builtin_benchmarks_use_benchmark_specific_dataset_providers(
    benchmark_id: str,
    provider_type: str,
    provider_module: str,
) -> None:
    definition = starter_catalog.get_builtin_benchmark(benchmark_id)

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
    definition = starter_catalog.get_builtin_benchmark(benchmark_id)

    provider = definition.build_dataset_provider()
    direct_base_name = type(provider).__bases__[0].__name__

    assert direct_base_name == "BuiltinDatasetProvider"


def test_simpleqa_summary_uses_f1_and_attempted_math_from_metric_details() -> None:
    definition = starter_catalog.get_builtin_benchmark("simpleqa_verified")
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
    definition = starter_catalog.get_builtin_benchmark("healthbench")
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

    assert summary["count"] == 2
    assert summary["mean_overall_score"] == pytest.approx(0.75)
    assert summary["tag_means"]["theme:communication"] == pytest.approx(0.75)
    assert summary["tag_means"]["axis:safety"] == pytest.approx(1.0)


def test_hle_summary_reports_accuracy_ci_calibration_and_skipped_images() -> None:
    definition = starter_catalog.get_builtin_benchmark("hle")
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

    assert summary["count"] == 2
    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["confidence_interval_half_width"] > 0.0
    assert summary["calibration_error"] == pytest.approx(0.3)
    assert summary["skipped_image_count"] == 4


def test_inspect_huggingface_dataset_uses_loader_hooks_for_schema_and_samples() -> None:
    summary = starter_catalog.inspect_huggingface_dataset(
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

    assert summary["dataset_id"] == "demo/qa"
    assert summary["splits"] == ["train", "test"]
    assert summary["modalities"] == ["text"]
    assert summary["row_count"] == 2
    assert summary["fields"]["question"] == "str"
    assert len(summary["samples"]) == 2


def test_catalog_fixtures_are_available_for_all_builtin_benchmarks() -> None:
    fixture_root = Path(starter_catalog.__file__).resolve().parent / "starter_fixtures"

    for benchmark_id in starter_catalog.list_builtin_benchmarks():
        assert (fixture_root / f"{benchmark_id}.json").exists()


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
        starter_catalog,
        "import_optional",
        lambda module_name, *, extra: _FakeDatasetsModule,
    )

    rows = starter_catalog.load_huggingface_rows("cais/hle", "test")

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
        starter_catalog,
        "import_optional",
        lambda module_name, *, extra: _FakeDatasetsModule,
    )

    rows = starter_catalog.load_huggingface_rows("openai/healthbench", "test")

    assert rows == [
        {
            "item_id": "item-1",
            "prompt_id": "hb-1",
            "prompt": [{"role": "user", "content": "Help me."}],
            "rubrics": [{"criterion": "safe", "points": 1, "tags": []}],
        }
    ]
