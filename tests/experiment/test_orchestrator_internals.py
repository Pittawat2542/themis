from __future__ import annotations

from typing import Sequence

from themis.core import entities as core_entities
from themis.evaluation import extractors, metrics, pipeline as evaluation_pipeline
from themis.evaluation.reports import EvaluationReport
from themis.experiment.orchestrator import ExperimentOrchestrator
from themis.generation.plan import GenerationPlan
from themis.generation.runner import GenerationRunner
from themis.interfaces import ModelProvider


class DummyCacheManager:
    def __init__(self):
        self.start_called = False

    @property
    def has_storage(self) -> bool:
        return True

    def run_metadata_exists(self, run_id: str) -> bool:
        return False

    def start_run(self, run_id: str, *, experiment_id: str = "default") -> None:
        self.start_called = True

    def cache_dataset(self, run_id: str, dataset: Sequence[dict[str, object]]) -> None:
        pass

    def load_cached_records(self, run_id: str):
        return {}

    def load_cached_evaluations(self, run_id: str, evaluation_config: dict | None = None):
        return {}

    def save_generation_record(self, run_id: str, record, cache_key: str) -> None:
        pass

    def save_evaluation_record(self, run_id: str, generation_record, evaluation_record, evaluation_config: dict | None = None) -> None:
        pass

    def get_run_path(self, run_id: str) -> str | None:
        return None


class FakeProvider(ModelProvider):
    def generate(self, task: core_entities.GenerationTask) -> core_entities.GenerationRecord:
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
    runner = GenerationRunner(provider=FakeProvider())
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
    )

    assert cache_manager.start_called is True
    assert report.metadata["total_samples"] == 1


def test_orchestrator_prefers_evaluation_fingerprint():
    class FingerprintPipeline:
        def evaluation_fingerprint(self) -> dict:
            return {"fingerprint": "value"}

        def evaluate(self, records):
            return EvaluationReport(metrics={}, failures=[], records=[], slices={})

    plan = _make_plan()
    runner = GenerationRunner(provider=FakeProvider())
    cache_manager = DummyCacheManager()

    orchestrator = ExperimentOrchestrator(
        generation_plan=plan,
        generation_runner=runner,
        evaluation_pipeline=FingerprintPipeline(),
        cache_manager=cache_manager,
    )

    config = orchestrator._build_evaluation_config()
    assert config == {"fingerprint": "value"}
