import pytest

from themis.orchestration.executor import TrialExecutor
from themis.specs.experiment import ExecutionPolicySpec
from themis.specs.experiment import InferenceParamsSpec, PromptTemplateSpec, TrialSpec
from themis.specs.foundational import DatasetSpec, ModelSpec, TaskSpec
from themis.errors.exceptions import OrchestrationAbortedError
from themis.records.error import ErrorRecord
from themis.records.trial import TrialRecord
from themis.types.enums import ErrorCode, ErrorWhere, RecordStatus
from themis.types.events import TrialEventType


class MockTrialRunner:
    def __init__(self):
        self.ran_trials = []

    def run_trial(self, trial_spec, context, runtime):
        self.ran_trials.append(trial_spec.trial_id)
        return TrialRecord(
            spec_hash=trial_spec.compute_hash(short=True),
            status=RecordStatus.OK,
            candidates=[],
        )


class MockProjectionRepo:
    def __init__(self, completed_hashes):
        self.completed_hashes = completed_hashes
        self.saved_records = []

    def has_trial(self, spec_hash, eval_revision="latest"):
        return spec_hash in self.completed_hashes

    def save_trial_record(self, record, *, eval_revision="latest"):
        self.saved_records.append((record.spec_hash, eval_revision))


class MockProjectionHandler:
    def __init__(self):
        self.completed_trials = []

    def on_trial_completed(self, trial_hash, eval_revision="latest"):
        self.completed_trials.append((trial_hash, eval_revision))
        return None


class MockResumeEventRepo:
    def __init__(self, completed_revisions=None, terminal_statuses=None):
        self.completed_revisions = completed_revisions or set()
        self.terminal_statuses = terminal_statuses or {}

    def has_projection_for_revision(self, trial_hash, eval_revision):
        return (trial_hash, eval_revision) in self.completed_revisions

    def latest_terminal_event_type(self, trial_hash):
        return self.terminal_statuses.get(trial_hash)


class RecordingEventRepo(MockResumeEventRepo):
    def __init__(self):
        super().__init__()
        self.events = []

    def append_event(self, event):
        self.events.append(event)

    def last_event_index(self, trial_hash, candidate_id=None):
        matching = [
            event.event_seq
            for event in self.events
            if event.trial_hash == trial_hash and event.candidate_id == candidate_id
        ]
        return max(matching) if matching else None


def test_trial_executor_skips_completed_trials():
    # Setup trials
    completed_trial = TrialSpec(
        trial_id="t_done",
        model=ModelSpec(model_id="m1", provider="p"),
        task=TaskSpec(
            task_id="t1", dataset=DatasetSpec(source="memory"), default_metrics=["em"]
        ),
        item_id="1",
        prompt=PromptTemplateSpec(messages=[]),
        params=InferenceParamsSpec(),
    )
    new_trial = TrialSpec(
        trial_id="t_new",
        model=ModelSpec(model_id="m1", provider="p"),
        task=TaskSpec(
            task_id="t1", dataset=DatasetSpec(source="memory"), default_metrics=["em"]
        ),
        item_id="2",
        prompt=PromptTemplateSpec(messages=[]),
        params=InferenceParamsSpec(),
    )

    completed_hash = completed_trial.compute_hash(short=True)

    runner = MockTrialRunner()
    repo = MockProjectionRepo(completed_hashes={completed_hash})
    executor = TrialExecutor(runner=runner, projection_repo=repo)

    executor.execute_trials(
        [completed_trial, new_trial], dataset_context={}, runtime_context={}
    )

    assert runner.ran_trials == ["t_new"]


def test_trial_executor_routes_terminal_trials_through_projection_handler():
    trial = TrialSpec(
        trial_id="t_new",
        model=ModelSpec(model_id="m1", provider="p"),
        task=TaskSpec(
            task_id="t1", dataset=DatasetSpec(source="memory"), default_metrics=["em"]
        ),
        item_id="2",
        prompt=PromptTemplateSpec(messages=[]),
        params=InferenceParamsSpec(),
    )

    runner = MockTrialRunner()
    repo = MockProjectionRepo(completed_hashes=set())
    projection_handler = MockProjectionHandler()
    executor = TrialExecutor(
        runner=runner,
        projection_repo=repo,
        projection_handler=projection_handler,
    )

    executor.execute_trials(
        [trial], dataset_context={}, runtime_context={}, resume=False
    )

    assert projection_handler.completed_trials == [(trial.spec_hash, "latest")]


def test_trial_executor_skips_trials_using_event_log_for_matching_eval_revision():
    trial = TrialSpec(
        trial_id="t_done",
        model=ModelSpec(model_id="m1", provider="p"),
        task=TaskSpec(
            task_id="t1", dataset=DatasetSpec(source="memory"), default_metrics=["em"]
        ),
        item_id="1",
        prompt=PromptTemplateSpec(messages=[]),
        params=InferenceParamsSpec(),
    )

    runner = MockTrialRunner()
    repo = MockProjectionRepo(completed_hashes=set())
    event_repo = MockResumeEventRepo(
        completed_revisions={(trial.spec_hash, "latest")},
        terminal_statuses={trial.spec_hash: TrialEventType.TRIAL_COMPLETED},
    )
    executor = TrialExecutor(runner=runner, projection_repo=repo, event_repo=event_repo)

    executor.execute_trials(
        [trial],
        dataset_context={},
        runtime_context={},
        resume=True,
        eval_revision="latest",
    )

    assert runner.ran_trials == []


def test_trial_executor_does_not_skip_trials_for_other_revisions_or_failed_terminal_events():
    completed_trial = TrialSpec(
        trial_id="t_done",
        model=ModelSpec(model_id="m1", provider="p"),
        task=TaskSpec(
            task_id="t1", dataset=DatasetSpec(source="memory"), default_metrics=["em"]
        ),
        item_id="1",
        prompt=PromptTemplateSpec(messages=[]),
        params=InferenceParamsSpec(),
    )
    failed_trial = TrialSpec(
        trial_id="t_failed",
        model=ModelSpec(model_id="m1", provider="p"),
        task=TaskSpec(
            task_id="t1", dataset=DatasetSpec(source="memory"), default_metrics=["em"]
        ),
        item_id="2",
        prompt=PromptTemplateSpec(messages=[]),
        params=InferenceParamsSpec(),
    )

    runner = MockTrialRunner()
    repo = MockProjectionRepo(completed_hashes=set())
    event_repo = MockResumeEventRepo(
        completed_revisions={(completed_trial.spec_hash, "older")},
        terminal_statuses={
            completed_trial.spec_hash: TrialEventType.TRIAL_COMPLETED,
            failed_trial.spec_hash: TrialEventType.TRIAL_FAILED,
        },
    )
    executor = TrialExecutor(runner=runner, projection_repo=repo, event_repo=event_repo)

    executor.execute_trials(
        [completed_trial, failed_trial],
        dataset_context={},
        runtime_context={},
        resume=True,
        eval_revision="latest",
    )

    assert runner.ran_trials == ["t_done", "t_failed"]


class MockErrorTrialRunner:
    def __init__(self, records):
        self.records = list(records)
        self.ran_trials = []

    def run_trial(self, trial_spec, context, runtime):
        self.ran_trials.append(trial_spec.trial_id)
        return self.records.pop(0)


def test_trial_executor_trips_circuit_breaker_on_repeated_fingerprints():
    trials = [
        TrialSpec(
            trial_id="t_1",
            model=ModelSpec(model_id="m1", provider="p"),
            task=TaskSpec(
                task_id="t1",
                dataset=DatasetSpec(source="memory"),
                default_metrics=["em"],
            ),
            item_id="1",
            prompt=PromptTemplateSpec(messages=[]),
            params=InferenceParamsSpec(),
        ),
        TrialSpec(
            trial_id="t_2",
            model=ModelSpec(model_id="m1", provider="p"),
            task=TaskSpec(
                task_id="t1",
                dataset=DatasetSpec(source="memory"),
                default_metrics=["em"],
            ),
            item_id="2",
            prompt=PromptTemplateSpec(messages=[]),
            params=InferenceParamsSpec(),
        ),
        TrialSpec(
            trial_id="t_3",
            model=ModelSpec(model_id="m1", provider="p"),
            task=TaskSpec(
                task_id="t1",
                dataset=DatasetSpec(source="memory"),
                default_metrics=["em"],
            ),
            item_id="3",
            prompt=PromptTemplateSpec(messages=[]),
            params=InferenceParamsSpec(),
        ),
    ]
    shared_error = ErrorRecord(
        code=ErrorCode.PROVIDER_TIMEOUT,
        message="upstream timeout",
        retryable=True,
        where=ErrorWhere.INFERENCE,
    )
    failing_record = TrialRecord(
        spec_hash="trial_fail_1",
        status=RecordStatus.ERROR,
        error=shared_error,
        candidates=[],
    )
    runner = MockErrorTrialRunner([failing_record, failing_record, failing_record])
    repo = MockProjectionRepo(completed_hashes=set())
    executor = TrialExecutor(
        runner=runner,
        projection_repo=repo,
        execution_policy=ExecutionPolicySpec(circuit_breaker_threshold=2),
    )

    with pytest.raises(OrchestrationAbortedError, match="Circuit breaker triggered"):
        executor.execute_trials(
            trials, dataset_context={}, runtime_context={}, resume=False
        )

    assert runner.ran_trials == ["t_1", "t_2"]


class RaisingTrialRunner:
    def run_trial(self, trial_spec, context, runtime):
        raise RuntimeError("projection blew up")


def test_trial_executor_appends_terminal_trial_failed_event_for_uncaught_exceptions():
    trial = TrialSpec(
        trial_id="t_fail",
        model=ModelSpec(model_id="m1", provider="provider_x"),
        task=TaskSpec(
            task_id="t1", dataset=DatasetSpec(source="memory"), default_metrics=["em"]
        ),
        item_id="1",
        prompt=PromptTemplateSpec(messages=[]),
        params=InferenceParamsSpec(),
    )
    event_repo = RecordingEventRepo()
    executor = TrialExecutor(
        runner=RaisingTrialRunner(),
        projection_repo=MockProjectionRepo(completed_hashes=set()),
        event_repo=event_repo,
    )

    with pytest.raises(RuntimeError, match="projection blew up"):
        executor.execute_trials(
            [trial], dataset_context={}, runtime_context={}, resume=False
        )

    assert len(event_repo.events) == 1
    event = event_repo.events[0]
    assert event.event_type == TrialEventType.TRIAL_FAILED
    assert event.status == RecordStatus.ERROR
    assert event.error is not None
    assert event.error.details["provider"] == "provider_x"
    assert event.error.details["model_id"] == "m1"
