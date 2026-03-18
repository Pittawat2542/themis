from __future__ import annotations
from themis.errors import InferenceError
from themis.types.enums import PromptRole

from themis.orchestration.trial_runner import TrialRunner, candidate_hash_for_index
from themis.records.conversation import Conversation, MessageEvent, MessagePayload
from themis.records.evaluation import MetricScore
from themis.records.inference import InferenceRecord
from themis.registry.plugin_registry import PluginRegistry
from themis.specs.experiment import InferenceParamsSpec, PromptTemplateSpec, TrialSpec
from themis.specs.foundational import (
    DatasetSpec,
    EvaluationSpec,
    GenerationSpec,
    ModelSpec,
    TaskSpec,
)
from themis.storage.event_repo import SqliteEventRepository
from themis.storage.sqlite_schema import DatabaseManager
from themis.types.enums import ErrorCode, RecordStatus, DatasetSource
from themis.types.events import TrialEvent, TrialEventType, TimelineStage


class ResumeAwareInferenceEngine:
    def __init__(self) -> None:
        self.seen_resume = None

    def infer(self, trial, context, runtime):
        self.seen_resume = runtime.get("resume")
        return InferenceRecord(
            spec_hash="resume_inf_hash",
            raw_text="continued answer",
            conversation=Conversation(
                events=[
                    MessageEvent(
                        role=PromptRole.USER,
                        payload=MessagePayload(content="Turn 0"),
                        event_index=0,
                    ),
                    MessageEvent(
                        role=PromptRole.ASSISTANT,
                        payload=MessagePayload(content="Turn 1"),
                        event_index=1,
                    ),
                    MessageEvent(
                        role=PromptRole.ASSISTANT,
                        payload=MessagePayload(content="Turn 2"),
                        event_index=2,
                    ),
                ]
            ),
        )


class PassthroughMetric:
    def score(self, trial, candidate, context):
        return MetricScore(metric_id="em", value=1.0)


class ResumeRetryingInferenceEngine:
    def __init__(self) -> None:
        self.calls = 0
        self.seen_resume = None

    def infer(self, trial, context, runtime):
        del trial, context
        self.calls += 1
        self.seen_resume = runtime.get("resume")
        raise InferenceError(
            code=ErrorCode.PROVIDER_TIMEOUT,
            message="provider timeout",
            details={"calls": self.calls},
        )


def test_trial_runner_resumes_conversation_from_last_persisted_event(tmp_path):
    manager = DatabaseManager(f"sqlite:///{tmp_path}/resume.db")
    manager.initialize()
    event_repo = SqliteEventRepository(manager)
    registry = PluginRegistry()
    engine = ResumeAwareInferenceEngine()
    registry.register_inference_engine("openai", engine)
    registry.register_metric("em", PassthroughMetric())
    runner = TrialRunner(registry, event_repo=event_repo, parallel_candidates=1)

    trial = TrialSpec(
        trial_id="resume_trial",
        model=ModelSpec(model_id="gpt-4o-mini", provider="openai"),
        task=TaskSpec(
            task_id="math",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
            evaluations=[EvaluationSpec(name="score", metrics=["em"])],
        ),
        item_id="item-1",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
        candidate_count=1,
    )
    event_repo.save_spec(trial)
    candidate_id = candidate_hash_for_index(trial, 0)

    seed_events = [
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=1,
            event_id=f"{trial.spec_hash}:1",
            event_type=TrialEventType.TRIAL_STARTED,
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=2,
            event_id=f"{trial.spec_hash}:2",
            event_type=TrialEventType.ITEM_LOADED,
            stage=TimelineStage.ITEM_LOAD,
            status=RecordStatus.OK,
            metadata={"item_id": trial.item_id, "dataset_source": "memory"},
            payload={"question": "6 * 7"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=3,
            event_id=f"{trial.spec_hash}:3",
            event_type=TrialEventType.PROMPT_RENDERED,
            stage=TimelineStage.PROMPT_RENDER,
            status=RecordStatus.OK,
            metadata={"prompt_template_id": "baseline"},
            payload={"messages": []},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=4,
            event_id=f"{trial.spec_hash}:4",
            event_type=TrialEventType.CANDIDATE_STARTED,
            candidate_id=candidate_id,
            payload={"sample_index": 0},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=5,
            event_id=f"{trial.spec_hash}:5",
            event_type=TrialEventType.PROMPT_RENDERED,
            candidate_id=candidate_id,
            stage=TimelineStage.PROMPT_RENDER,
            status=RecordStatus.OK,
            metadata={"prompt_template_id": "baseline"},
            payload={"messages": []},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=6,
            event_id=f"{trial.spec_hash}:6",
            event_type=TrialEventType.CONVERSATION_EVENT,
            candidate_id=candidate_id,
            payload=MessageEvent(
                role=PromptRole.USER,
                payload=MessagePayload(content="Turn 0"),
                event_index=0,
            ).model_dump(mode="json"),
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=7,
            event_id=f"{trial.spec_hash}:7",
            event_type=TrialEventType.CONVERSATION_EVENT,
            candidate_id=candidate_id,
            payload=MessageEvent(
                role=PromptRole.ASSISTANT,
                payload=MessagePayload(content="Turn 1"),
                event_index=1,
            ).model_dump(mode="json"),
        ),
    ]
    for event in seed_events:
        event_repo.append_event(event)

    runner.run_trial(trial, {"question": "6 * 7"}, {})

    assert engine.seen_resume is not None
    assert engine.seen_resume["candidate_id"] == candidate_id
    assert engine.seen_resume["last_event_index"] == 1
    assert [
        event.payload.content for event in engine.seen_resume["conversation"].events
    ] == [
        "Turn 0",
        "Turn 1",
    ]

    events = event_repo.get_events(trial.spec_hash)
    candidate_events = [event for event in events if event.candidate_id == candidate_id]
    conversation_event_indices = [
        event.payload["event_index"]
        for event in candidate_events
        if event.event_type == TrialEventType.CONVERSATION_EVENT
    ]
    assert conversation_event_indices == [0, 1, 2]
    assert [event.event_type for event in events].count("trial_started") == 1
    assert [event.event_type for event in candidate_events].count(
        "candidate_started"
    ) == 1


def test_trial_runner_preserves_resume_attempt_counter(tmp_path):
    manager = DatabaseManager(f"sqlite:///{tmp_path}/resume_attempts.db")
    manager.initialize()
    event_repo = SqliteEventRepository(manager)
    registry = PluginRegistry()
    engine = ResumeRetryingInferenceEngine()
    registry.register_inference_engine("openai", engine)
    runner = TrialRunner(
        registry,
        event_repo=event_repo,
        parallel_candidates=1,
        max_retries=3,
        retryable_error_codes=[ErrorCode.PROVIDER_TIMEOUT.value],
    )

    trial = TrialSpec(
        trial_id="resume_attempt_trial",
        model=ModelSpec(model_id="gpt-4o-mini", provider="openai"),
        task=TaskSpec(
            task_id="math",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
        ),
        item_id="item-1",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
        candidate_count=1,
    )
    event_repo.save_spec(trial)
    candidate_id = candidate_hash_for_index(trial, 0)

    seed_events = [
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=1,
            event_id=f"{trial.spec_hash}:1",
            event_type=TrialEventType.TRIAL_STARTED,
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=2,
            event_id=f"{trial.spec_hash}:2",
            event_type=TrialEventType.ITEM_LOADED,
            stage=TimelineStage.ITEM_LOAD,
            status=RecordStatus.OK,
            metadata={"item_id": trial.item_id, "dataset_source": "memory"},
            payload={"question": "6 * 7"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=3,
            event_id=f"{trial.spec_hash}:3",
            event_type=TrialEventType.CANDIDATE_STARTED,
            candidate_id=candidate_id,
            payload={"sample_index": 0},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=4,
            event_id=f"{trial.spec_hash}:4",
            event_type=TrialEventType.PROMPT_RENDERED,
            candidate_id=candidate_id,
            stage=TimelineStage.PROMPT_RENDER,
            status=RecordStatus.OK,
            metadata={"prompt_template_id": "baseline"},
            payload={"messages": []},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=5,
            event_id=f"{trial.spec_hash}:5",
            event_type=TrialEventType.CONVERSATION_EVENT,
            candidate_id=candidate_id,
            payload=MessageEvent(
                role=PromptRole.USER,
                payload=MessagePayload(content="Turn 0"),
                event_index=0,
            ).model_dump(mode="json"),
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=6,
            event_id=f"{trial.spec_hash}:6",
            event_type=TrialEventType.TRIAL_RETRY,
            candidate_id=candidate_id,
            stage=TimelineStage.INFERENCE,
            metadata={"attempt": 1, "cand_index": 0},
            payload={"attempt": 1, "cand_index": 0},
        ),
    ]
    for event in seed_events:
        event_repo.append_event(event)

    record = runner.run_trial(trial, {"question": "6 * 7"}, {})
    retry_events = [
        event
        for event in event_repo.get_events(trial.spec_hash)
        if event.candidate_id == candidate_id
        and event.event_type == TrialEventType.TRIAL_RETRY
    ]

    assert engine.seen_resume is not None
    assert engine.seen_resume["attempt"] == 1
    assert record.status == RecordStatus.ERROR
    assert record.candidates[0].error is not None
    assert record.candidates[0].error.details["attempt"] == 3
    assert [event.metadata.attempt for event in retry_events] == [1, 2]
    assert retry_events[-1].payload["attempt"] == 2
