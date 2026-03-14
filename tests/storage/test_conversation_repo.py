from pydantic import TypeAdapter

from themis.records.conversation import MessageEvent, MessagePayload
from themis.specs.experiment import InferenceParamsSpec, PromptTemplateSpec, TrialSpec
from themis.specs.foundational import DatasetSpec, GenerationSpec, ModelSpec, TaskSpec
from themis.storage.event_repo import SqliteEventRepository
from themis.storage.sqlite_schema import DatabaseManager
from themis.types.events import TrialEvent


def test_sqlite_event_repo_conversation_events(tmp_path):
    manager = DatabaseManager(f"sqlite:///{tmp_path}/conv_test.db")
    manager.initialize()

    repo = SqliteEventRepository(manager)

    # Must save a spec to satisfy the foreign key constraints on trial_hash
    spec = TrialSpec(
        trial_id="trial_conversation",
        model=ModelSpec(model_id="gpt-4", provider="openai"),
        task=TaskSpec(
            task_id="task",
            dataset=DatasetSpec(source="memory"),
            generation=GenerationSpec(),
        ),
        item_id="item-1",
        prompt=PromptTemplateSpec(messages=[]),
        params=InferenceParamsSpec(),
    )
    repo.save_spec(spec)
    trial_hash = spec.spec_hash

    user_msg = MessageEvent(
        role="user",
        payload=MessagePayload(content="Hello world"),
        event_index=0,
    )
    asst_msg = MessageEvent(
        role="assistant",
        payload=MessagePayload(content="Hi there!"),
        event_index=1,
    )

    # Save events
    repo.append_event(
        TrialEvent(
            trial_hash=trial_hash,
            event_seq=1,
            event_id="evt_1",
            event_type="conversation_event",
            candidate_id="candidate_1",
            payload=user_msg.model_dump(mode="json"),
        )
    )
    repo.append_event(
        TrialEvent(
            trial_hash=trial_hash,
            event_seq=2,
            event_id="evt_2",
            event_type="conversation_event",
            candidate_id="candidate_1",
            payload=asst_msg.model_dump(mode="json"),
        )
    )

    # Retrieve events
    events = repo.get_events(trial_hash, candidate_id="candidate_1")
    assert len(events) == 2
    assert repo.last_event_index(trial_hash, candidate_id="candidate_1") == 2

    msg1 = TypeAdapter(MessageEvent).validate_python(events[0].payload)
    assert msg1.role == "user"
    assert isinstance(msg1.payload, MessagePayload)
    assert msg1.payload.content == "Hello world"

    msg2 = TypeAdapter(MessageEvent).validate_python(events[1].payload)
    assert msg2.role == "assistant"
    assert isinstance(msg2.payload, MessagePayload)
    assert msg2.payload.content == "Hi there!"
