"""Import externally generated candidates into the generation overlay."""

from __future__ import annotations

from collections.abc import Sequence

from themis.contracts.protocols import (
    ProjectionHandler,
    ProjectionRepository,
    TrialEventRepository,
)
from themis.orchestration.run_manifest import RunManifest
from themis.records.trial import TrialRecord
from themis.types.enums import RecordStatus
from themis.types.events import (
    EmptyEventMetadata,
    EvaluationCompletedEventMetadata,
    TimelineStage,
    TrialEvent,
    TrialEventMetadata,
    TrialEventType,
)


class CandidateImporter:
    """Write imported generation candidates into the event log and projections."""

    def __init__(
        self,
        event_repo: TrialEventRepository,
        projection_repo: ProjectionRepository,
        projection_handler: ProjectionHandler,
    ) -> None:
        self.event_repo = event_repo
        self.projection_repo = projection_repo
        self.projection_handler = projection_handler

    def import_candidates(self, trial_records: Sequence[TrialRecord]) -> list[str]:
        """Persist imported trial records as generation-stage events."""
        imported_trial_hashes: list[str] = []
        for trial_record in trial_records:
            if trial_record.trial_spec is None:
                raise ValueError("Imported trial records must include trial_spec.")
            trial = trial_record.trial_spec
            self.event_repo.save_spec(trial)
            event_seq = self.event_repo.last_event_index(trial.spec_hash) or 0

            def append_event(
                event_type: TrialEventType,
                *,
                candidate_id: str | None = None,
                stage=None,
                status: RecordStatus | None = None,
                metadata: TrialEventMetadata | None = None,
                payload=None,
                error=None,
            ) -> None:
                nonlocal event_seq
                event_seq += 1
                self.event_repo.append_event(
                    TrialEvent(
                        trial_hash=trial.spec_hash,
                        event_seq=event_seq,
                        event_id=f"{trial.spec_hash}:{event_seq}",
                        candidate_id=candidate_id,
                        event_type=event_type,
                        stage=stage,
                        status=status,
                        metadata=metadata or EmptyEventMetadata(),
                        payload=payload,
                        error=error,
                    )
                )

            append_event(
                TrialEventType.TRIAL_STARTED, payload={"trial_id": trial.trial_id}
            )
            append_event(
                TrialEventType.PROMPT_RENDERED,
                stage=TimelineStage.PROMPT_RENDER,
                status=RecordStatus.OK,
                payload={
                    "messages": [
                        message.model_dump(mode="json")
                        for message in trial.prompt.messages
                    ],
                    "follow_up_turns": [
                        turn.model_dump(mode="json")
                        for turn in trial.prompt.follow_up_turns
                    ],
                    "tools": [tool.model_dump(mode="json") for tool in trial.tools],
                },
            )
            for candidate in sorted(
                trial_record.candidates, key=lambda item: item.sample_index
            ):
                candidate_id = candidate.candidate_id or candidate.spec_hash
                append_event(
                    TrialEventType.CANDIDATE_STARTED,
                    candidate_id=candidate_id,
                    payload={"sample_index": candidate.sample_index},
                )
                if candidate.inference is not None:
                    append_event(
                        TrialEventType.INFERENCE_COMPLETED,
                        candidate_id=candidate_id,
                        stage=TimelineStage.INFERENCE,
                        status=RecordStatus.OK,
                        payload=candidate.inference.model_dump(mode="json"),
                    )
                if candidate.conversation is not None:
                    for conversation_event in candidate.conversation.events:
                        append_event(
                            TrialEventType.CONVERSATION_EVENT,
                            candidate_id=candidate_id,
                            payload=conversation_event.model_dump(mode="json"),
                        )
                append_event(
                    TrialEventType.CANDIDATE_COMPLETED,
                    candidate_id=candidate_id,
                    status=candidate.status,
                    payload={"status": candidate.status.value},
                )
            append_event(
                TrialEventType.TRIAL_COMPLETED,
                status=trial_record.status,
                payload={"status": trial_record.status.value},
                error=trial_record.error,
            )
            self.projection_handler.on_trial_completed(trial.spec_hash)
            imported_trial_hashes.append(trial.spec_hash)
        return imported_trial_hashes


class StageResultImporter:
    """Append imported evaluation-stage events against existing generated candidates."""

    def __init__(
        self,
        event_repo: TrialEventRepository,
        projection_handler: ProjectionHandler,
    ) -> None:
        self.event_repo = event_repo
        self.projection_handler = projection_handler

    def import_evaluation_results(
        self,
        manifest: RunManifest,
        trial_records: Sequence[TrialRecord],
    ) -> list[str]:
        """Persist imported evaluation payloads as evaluation-stage events."""
        evaluation_items = {
            (item.trial_hash, item.candidate_id): item
            for item in manifest.work_items
            if item.stage == TimelineStage.EVALUATION
            and item.evaluation_hash is not None
        }
        imported_trial_hashes: list[str] = []
        touched_overlays: dict[str, set[str]] = {}

        for trial_record in trial_records:
            if trial_record.trial_spec is None:
                raise ValueError("Imported evaluation results must include trial_spec.")
            trial = trial_record.trial_spec
            self.event_repo.save_spec(trial)
            event_seq = self.event_repo.last_event_index(trial.spec_hash) or 0

            def append_event(
                event_type: TrialEventType,
                *,
                candidate_id: str | None = None,
                stage=None,
                status: RecordStatus | None = None,
                metadata: TrialEventMetadata | None = None,
                payload=None,
                error=None,
            ) -> None:
                nonlocal event_seq
                event_seq += 1
                self.event_repo.append_event(
                    TrialEvent(
                        trial_hash=trial.spec_hash,
                        event_seq=event_seq,
                        event_id=f"{trial.spec_hash}:{event_seq}",
                        candidate_id=candidate_id,
                        event_type=event_type,
                        stage=stage,
                        status=status,
                        metadata=metadata or EmptyEventMetadata(),
                        payload=payload,
                        error=error,
                    )
                )

            for candidate in sorted(
                trial_record.candidates, key=lambda item: item.sample_index
            ):
                candidate_id = candidate.candidate_id or candidate.spec_hash
                work_item = evaluation_items.get((trial.spec_hash, candidate_id))
                if work_item is None:
                    raise ValueError(
                        "Imported evaluation result does not match an exported work item: "
                        f"{trial.spec_hash}/{candidate_id}"
                    )
                if (
                    candidate.evaluation is None
                    or not candidate.evaluation.metric_scores
                ):
                    raise ValueError(
                        "Imported evaluation results must include evaluation payloads with metric scores."
                    )
                first_score = candidate.evaluation.metric_scores[0]
                append_event(
                    TrialEventType.EVALUATION_COMPLETED,
                    candidate_id=candidate_id,
                    stage=TimelineStage.EVALUATION,
                    status=RecordStatus.OK,
                    metadata=EvaluationCompletedEventMetadata(
                        transform_hash=work_item.transform_hash,
                        evaluation_hash=work_item.evaluation_hash,
                        metric_id=first_score.metric_id,
                        score=first_score.value,
                        judge_call_count=0,
                    ),
                    payload=candidate.evaluation.model_dump(mode="json"),
                )
                touched_overlays.setdefault(trial.spec_hash, set()).add(
                    work_item.evaluation_hash or ""
                )

            imported_trial_hashes.append(trial.spec_hash)

        for trial_hash, evaluation_hashes in touched_overlays.items():
            for evaluation_hash in sorted(
                hash_ for hash_ in evaluation_hashes if hash_
            ):
                self.projection_handler.on_trial_completed(
                    trial_hash,
                    evaluation_hash=evaluation_hash,
                )

        return imported_trial_hashes
