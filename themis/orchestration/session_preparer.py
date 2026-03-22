"""Preparation boundary for trial execution sessions."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from themis.contracts.protocols import DatasetContext, TrialEventRepository
from themis.prompting import render_follow_up_turns, render_prompt_messages
from themis.orchestration.runner_state import (
    TrialExecutionSession,
    artifact_ref,
    dataset_payload,
    json_value,
)
from themis.orchestration.task_resolution import resolve_task_stages
from themis.records.provenance import ProvenanceRecord
from themis.specs.experiment import (
    PromptMessage,
    PromptTurnSpec,
    RuntimeContext,
    TrialSpec,
)
from themis.storage.artifact_store import ArtifactStore
from themis.types.enums import RecordStatus
from themis.types.events import (
    ArtifactRole,
    ItemLoadedEventMetadata,
    PromptRenderedEventMetadata,
    TimelineStage,
    TrialEventType,
)
from themis.types.json_validation import validate_json_dict


class TrialSessionPreparer:
    """Builds shared trial execution sessions and emits initial lifecycle events."""

    def __init__(
        self,
        *,
        event_repo: TrialEventRepository,
        artifact_store: ArtifactStore | None = None,
        store_item_payloads: bool = True,
        append_session_event: Callable[..., None],
    ) -> None:
        self.event_repo = event_repo
        self.artifact_store = artifact_store
        self.store_item_payloads = store_item_payloads
        self.append_session_event = append_session_event

    def prepare_trial_session(
        self,
        trial: TrialSpec,
        prepared_trial: TrialSpec,
        dataset_context: DatasetContext,
        base_runtime: RuntimeContext,
        provenance: ProvenanceRecord,
    ) -> TrialExecutionSession:
        """Build one trial session and emit initial trial-scoped events if needed."""
        self.event_repo.save_spec(trial)
        existing_trial_events = self.event_repo.get_events(trial.spec_hash)

        prompt_payload = json_value(
            {
                "messages": [
                    message.model_dump(mode="json")
                    for message in prepared_trial.prompt.messages
                ],
                "follow_up_turns": [
                    turn.model_dump(mode="json")
                    for turn in prepared_trial.prompt.follow_up_turns
                ],
                "tools": [
                    tool.model_dump(mode="json") for tool in prepared_trial.tools
                ],
                "mcp_servers": [
                    server.model_dump(mode="json")
                    for server in prepared_trial.mcp_servers
                ],
            },
            label="rendered prompt",
        )
        item_payload = json_value(
            dataset_payload(dataset_context),
            label="dataset payload",
        )
        dataset_metadata = getattr(dataset_context, "metadata", {})
        prompt_artifact = artifact_ref(
            prompt_payload,
            role=ArtifactRole.RENDERED_PROMPT,
            label="rendered_prompt",
            artifact_store=self.artifact_store,
        )
        session = TrialExecutionSession(
            trial=trial,
            prepared_trial=prepared_trial,
            dataset_context=dataset_context,
            base_runtime=base_runtime,
            provenance=provenance,
            resolved_stages=resolve_task_stages(trial.task),
            prompt_payload=prompt_payload,
            prompt_artifact=prompt_artifact,
            item_payload=item_payload,
            dataset_metadata=dataset_metadata
            if isinstance(dataset_metadata, Mapping)
            else {},
            event_seq=self.event_repo.last_event_index(trial.spec_hash) or 0,
        )

        if existing_trial_events:
            return session

        self.append_session_event(
            session,
            TrialEventType.TRIAL_STARTED,
            payload={"trial_id": trial.trial_id},
        )
        item_artifact = (
            artifact_ref(
                item_payload,
                role=ArtifactRole.ITEM_PAYLOAD,
                label="item_payload",
                artifact_store=self.artifact_store,
            )
            if self.store_item_payloads
            else None
        )
        tags: dict[str, str] = {}
        if isinstance(dataset_metadata, Mapping):
            tags.update(
                {str(key): str(value) for key, value in dataset_metadata.items()}
            )
        tags.update(base_runtime.run_labels)
        prompt_metadata = PromptRenderedEventMetadata(
            prompt_template_id=prepared_trial.prompt.id,
            rendered_prompt_hash=prompt_artifact[1],
            input_field_map=sorted(dataset_context.keys()),
        )
        self.append_session_event(
            session,
            TrialEventType.ITEM_LOADED,
            stage=TimelineStage.ITEM_LOAD,
            status=RecordStatus.OK,
            metadata=ItemLoadedEventMetadata(
                item_id=trial.item_id,
                dataset_source=str(trial.task.dataset.source),
                dataset_revision=trial.task.dataset.revision,
                tags=validate_json_dict(tags, label="item tags") if tags else None,
                item_payload_hash=item_artifact[1]
                if item_artifact is not None
                else None,
            ),
            payload=item_payload if self.store_item_payloads else None,
            artifact_refs=[item_artifact[0]] if item_artifact is not None else [],
        )
        self.append_session_event(
            session,
            TrialEventType.PROMPT_RENDERED,
            stage=TimelineStage.PROMPT_RENDER,
            status=RecordStatus.OK,
            metadata=prompt_metadata,
            payload=prompt_payload,
            artifact_refs=[prompt_artifact[0]],
        )
        return session


def prepare_benchmark_prompt(
    trial: TrialSpec,
    dataset_context: DatasetContext,
    runtime: RuntimeContext,
) -> TrialSpec:
    """Render benchmark-native prompt namespaces into one execution-ready trial."""

    if trial.task.benchmark_id is None:
        return trial

    item_namespace = dataset_payload(dataset_context)
    item_id = getattr(dataset_context, "item_id", None)
    if item_id is not None:
        item_namespace.setdefault("item_id", str(item_id))
    item_metadata = getattr(dataset_context, "metadata", None)
    if isinstance(item_metadata, Mapping):
        item_namespace.setdefault(
            "metadata",
            {str(key): value for key, value in item_metadata.items()},
        )
    render_namespace = {
        "item": item_namespace,
        "slice": {
            "benchmark_id": trial.task.benchmark_id,
            "slice_id": trial.task.slice_id or trial.task.task_id,
            "dimensions": dict(trial.task.dimensions),
        },
        "prompt": {
            "id": trial.prompt.id,
            "family": trial.prompt.family,
            "variables": dict(trial.prompt.variables),
        },
        "runtime": runtime.model_dump(mode="json"),
    }
    rendered_messages = render_prompt_messages(
        trial.prompt.messages,
        render_namespace,
        strict=True,
    )
    rendered_follow_up_turns = render_follow_up_turns(
        trial.prompt.follow_up_turns,
        render_namespace,
        strict=True,
    )
    return trial.model_copy(
        update={
            "prompt": trial.prompt.model_copy(
                update={
                    "messages": [
                        PromptMessage.model_validate(message)
                        for message in rendered_messages
                    ],
                    "follow_up_turns": [
                        PromptTurnSpec.model_validate(turn)
                        for turn in rendered_follow_up_turns
                    ],
                }
            )
        }
    )
