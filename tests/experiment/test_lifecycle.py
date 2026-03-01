import pytest
from unittest.mock import MagicMock

# Import orchestrator first to avoid dormant circular import in themis.config
from themis.experiment.lifecycle import RunLifecycle
from themis.experiment.cache_manager import CacheManager
from themis.experiment.integration_manager import IntegrationManager
from themis.generation.plan import GenerationPlan
from themis.evaluation.pipeline import EvaluationPipeline


@pytest.fixture
def mock_cache():
    cache = MagicMock(spec=CacheManager)
    cache.has_storage = True
    cache.run_metadata_exists.return_value = False
    return cache


@pytest.fixture
def lifecycle(mock_cache):
    plan = MagicMock(spec=GenerationPlan)
    plan.models = []
    plan.sampling_parameters = []

    eval_pipe = MagicMock(spec=EvaluationPipeline)
    eval_pipe.evaluation_fingerprint.return_value = [
        ("metrics", ["accuracy"]),
        ("extractor", "default"),
    ]

    integrations = MagicMock(spec=IntegrationManager)

    return RunLifecycle(
        plan=plan, evaluation=eval_pipe, cache=mock_cache, integrations=integrations
    )


def test_lifecycle_fail_run_delegates_to_cache(lifecycle, mock_cache):
    lifecycle.fail_run("test-run", "Something broke")
    mock_cache.fail_run.assert_called_once_with("test-run", "Something broke")


def test_lifecycle_initialize_starts_run(lifecycle, mock_cache):
    dataset = [{"q": "1"}]
    lifecycle.initialize_run_resources(
        dataset=dataset,
        run_id="run-123",
        max_samples=1,
        resume=False,
        run_manifest=None,
    )
    mock_cache.start_run.assert_called_once()
    assert mock_cache.start_run.call_args[0][0] == "run-123"


def test_lifecycle_finalize_completes_run(lifecycle, mock_cache):
    from themis.core.entities import ExperimentReport
    from themis.experiment.cost import CostBreakdown
    from themis.experiment.context import _RetentionBuffer

    # Mock json saving so it doesn't do disk IO
    lifecycle._save_report_json = MagicMock()

    report = lifecycle.finalize_experiment_run(
        run_identifier="run-123",
        selected_dataset=[{"q": "1"}],
        manifest_payload={},
        run_manifest_hash="hash",
        generation_results=MagicMock(
            spec=_RetentionBuffer, dropped=0, to_list=lambda: []
        ),
        evaluation_report=MagicMock(records=[], failures=[], metadata={}),
        failures=[],
        cached_eval_records=MagicMock(
            spec=_RetentionBuffer, dropped=0, to_list=lambda: []
        ),
        new_eval_records=MagicMock(
            spec=_RetentionBuffer, dropped=0, to_list=lambda: []
        ),
        successful_generations_total=1,
        failed_generations_total=0,
        evaluation_record_failures_total=0,
        cost_breakdown=CostBreakdown(0.0, 0.0, 0.0, {}),
        cache_results=True,
    )

    assert isinstance(report, ExperimentReport)
    mock_cache.complete_run.assert_called_once_with("run-123")
    lifecycle._save_report_json.assert_called_once()
