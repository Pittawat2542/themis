from __future__ import annotations

from typing import Sequence

from themis.core import entities as core_entities
from themis.evaluation import extractors, metrics, pipeline as evaluation_pipeline
from themis.evaluation.reports import EvaluationReport
from themis.experiment.orchestrator import ExperimentOrchestrator
from themis.generation.plan import GenerationPlan
from themis.generation.runner import GenerationRunner
from themis.interfaces import StatelessTaskExecutor


class DummyCacheManager:
    def __init__(self):
        self.start_called = False
        self.complete_calls: list[str] = []
        self.fail_calls: list[dict[str, str]] = []
        self.cached_dataset_calls: list[dict[str, object]] = []
        self.load_cached_records_calls: list[str] = []
        self.load_cached_evaluations_calls: list[dict[str, object]] = []
        self.saved_generation_records: list[dict[str, object]] = []
        self.saved_evaluation_records: list[dict[str, object]] = []

    @property
    def has_storage(self) -> bool:
        return True

    def run_metadata_exists(self, run_id: str) -> bool:
        return False

    def start_run(
        self,
        run_id: str,
        *,
        experiment_id: str = "default",
        config: dict | None = None,
    ) -> None:
        self.start_called = True

    def cache_dataset(self, run_id: str, dataset: Sequence[dict[str, object]]) -> None:
        self.cached_dataset_calls.append(
            {"run_id": run_id, "sample_count": len(dataset)}
        )

    def load_cached_records(self, run_id: str):
        self.load_cached_records_calls.append(run_id)
        return {}

    def load_cached_evaluations(
        self, run_id: str, evaluation_config: dict | None = None
    ):
        self.load_cached_evaluations_calls.append(
            {"run_id": run_id, "evaluation_config": dict(evaluation_config or {})}
        )
        return {}

    def save_generation_record(self, run_id: str, record, cache_key: str) -> None:
        self.saved_generation_records.append(
            {
                "run_id": run_id,
                "cache_key": cache_key,
                "sample_id": record.task.metadata.get("dataset_id"),
            }
        )

    def save_evaluation_record(
        self,
        run_id: str,
        generation_record,
        evaluation_record,
        evaluation_config: dict | None = None,
    ) -> None:
        self.saved_evaluation_records.append(
            {
                "run_id": run_id,
                "sample_id": generation_record.task.metadata.get("dataset_id"),
                "evaluation_config": dict(evaluation_config or {}),
            }
        )

    def get_run_path(self, run_id: str) -> str | None:
        return None

    def complete_run(self, run_id: str) -> None:
        self.complete_calls.append(run_id)

    def fail_run(self, run_id: str, error_message: str) -> None:
        self.fail_calls.append({"run_id": run_id, "error_message": error_message})


class FakeProvider(StatelessTaskExecutor):
    def execute(
        self, task: core_entities.GenerationTask
    ) -> core_entities.GenerationRecord:
        return core_entities.GenerationRecord(
            task=task,
            output=core_entities.ModelOutput(text="ok"),
            error=None,
        )


def _make_plan() -> GenerationPlan:
    template = core_entities.PromptSpec(name="t", template="Q")
    prompt = core_entities.PromptRender(spec=template, text="Q")
    model = core_entities.ModelSpec(identifier="model-x", provider="fake")
    sampling = core_entities.SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=8)
    return GenerationPlan(
        templates=[
            type(
                "Template",
                (),
                {"render_prompt": lambda self, ctx: prompt, "metadata": {}},
            )()
        ],
        models=[model],
        sampling_parameters=[sampling],
        dataset_id_field="id",
        reference_field=None,
    )


def test_orchestrator_uses_cache_manager_api():
    plan = _make_plan()
    runner = GenerationRunner(executor=FakeProvider())
    eval_pipeline = evaluation_pipeline.EvaluationPipeline(
        extractor=extractors.IdentityExtractor(),
        metrics=[metrics.ResponseLength()],
    )
    cache_manager = DummyCacheManager()

    orchestrator = ExperimentOrchestrator(
        generation_plan=plan,
        generation_runner=runner,
        evaluation_pipeline=eval_pipeline,
        cache_manager=cache_manager,
    )

    report = orchestrator.run(
        dataset=[{"id": "1", "question": "2+2"}],
        run_id="orchestrator-test",
        resume=False,
        run_manifest={
            "schema_version": "1",
            "model": {
                "identifier": "model-x",
                "provider": "fake",
                "provider_options": {},
            },
            "sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 8},
            "num_samples": 1,
            "evaluation": {
                "metrics": [
                    "themis.evaluation.metrics.response_length.ResponseLength:ResponseLength"
                ],
                "extractor": "themis.evaluation.extractors.IdentityExtractor",
            },
            "seeds": {"sampling_seed": None},
            "package_versions": {"themis-eval": "1.0.0"},
            "git_commit_hash": "abc123",
        },
    )

    assert cache_manager.start_called is True
    assert cache_manager.cached_dataset_calls == [
        {"run_id": "orchestrator-test", "sample_count": 1}
    ]
    assert cache_manager.load_cached_records_calls == []
    assert cache_manager.load_cached_evaluations_calls == []
    assert len(cache_manager.saved_generation_records) == 1
    assert cache_manager.saved_generation_records[0]["sample_id"] == "1"
    assert len(cache_manager.saved_evaluation_records) == 1
    eval_config = cache_manager.saved_evaluation_records[0]["evaluation_config"]
    assert "metrics" in eval_config
    assert "extractor" in eval_config
    assert report.metadata["total_samples"] == 1
    assert cache_manager.complete_calls == ["orchestrator-test"]
    assert cache_manager.fail_calls == []


def test_orchestrator_prefers_evaluation_fingerprint():
    class FingerprintPipeline:
        def evaluation_fingerprint(self) -> dict:
            return {"fingerprint": "value"}

        def evaluate(self, records):
            return EvaluationReport(metrics={}, failures=[], records=[], slices={})

    plan = _make_plan()
    runner = GenerationRunner(executor=FakeProvider())
    cache_manager = DummyCacheManager()

    orchestrator = ExperimentOrchestrator(
        generation_plan=plan,
        generation_runner=runner,
        evaluation_pipeline=FingerprintPipeline(),
        cache_manager=cache_manager,
    )

    config = orchestrator._build_evaluation_config()
    assert config == {"fingerprint": "value"}
