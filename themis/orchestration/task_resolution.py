"""Resolve task-local transform and evaluation stages into deterministic IDs."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

from themis.errors import SpecValidationError
from themis.specs.foundational import (
    EvaluationSpec,
    OutputTransformSpec,
    TaskSpec,
    TraceEvaluationSpec,
)
from themis.types.enums import ErrorCode


@dataclass(frozen=True, slots=True)
class ResolvedOutputTransform:
    """One output transform paired with its deterministic content hash."""

    spec: OutputTransformSpec
    transform_hash: str


@dataclass(frozen=True, slots=True)
class ResolvedEvaluation:
    """One evaluation resolved against its referenced transform."""

    spec: EvaluationSpec
    transform: ResolvedOutputTransform | None
    evaluation_hash: str


@dataclass(frozen=True, slots=True)
class ResolvedTraceEvaluation:
    """One trace evaluation paired with its deterministic content hash."""

    spec: TraceEvaluationSpec
    trace_score_hash: str


@dataclass(frozen=True, slots=True)
class ResolvedTaskStages:
    """Resolved transform and evaluation stages for one task."""

    output_transforms: tuple[ResolvedOutputTransform, ...]
    evaluations: tuple[ResolvedEvaluation, ...]
    trace_evaluations: tuple[ResolvedTraceEvaluation, ...] = field(
        default_factory=tuple
    )

    def output_transform_by_hash(
        self,
        transform_hash: str,
    ) -> ResolvedOutputTransform | None:
        """Return one resolved output transform by its deterministic hash."""
        return next(
            (
                output_transform
                for output_transform in self.output_transforms
                if output_transform.transform_hash == transform_hash
            ),
            None,
        )

    def evaluation_by_hash(
        self,
        evaluation_hash: str,
    ) -> ResolvedEvaluation | None:
        """Return one resolved evaluation by its deterministic hash."""
        return next(
            (
                evaluation
                for evaluation in self.evaluations
                if evaluation.evaluation_hash == evaluation_hash
            ),
            None,
        )

    def trace_evaluation_by_hash(
        self,
        trace_score_hash: str,
    ) -> ResolvedTraceEvaluation | None:
        """Return one resolved trace evaluation by its deterministic hash."""
        return next(
            (
                evaluation
                for evaluation in self.trace_evaluations
                if evaluation.trace_score_hash == trace_score_hash
            ),
            None,
        )


def resolve_task_stages(task: TaskSpec) -> ResolvedTaskStages:
    """Resolve transform references and compute deterministic stage identities."""
    transform_identities: dict[str, str] = {}
    resolved_transforms: list[ResolvedOutputTransform] = []
    for transform_spec in task.output_transforms:
        _record_short_hash_identity(
            stage_kind="output transform",
            stage_label=transform_spec.name,
            short_hash=transform_spec.spec_hash,
            canonical_hash=transform_spec.canonical_hash,
            known_identities=transform_identities,
        )
        resolved_transforms.append(
            ResolvedOutputTransform(
                spec=transform_spec,
                transform_hash=transform_spec.spec_hash,
            )
        )
    resolved_transform_tuple = tuple(resolved_transforms)
    transforms_by_name = {
        transform.spec.name: transform for transform in resolved_transform_tuple
    }
    evaluation_identities: dict[str, str] = {}
    resolved_evaluations = []
    for evaluation in task.evaluations:
        resolved_transform = _resolve_transform_reference(
            evaluation, transforms_by_name
        )
        evaluation_hash = _evaluation_hash(evaluation, resolved_transform)
        _record_short_hash_identity(
            stage_kind="evaluation",
            stage_label=evaluation.name,
            short_hash=evaluation_hash,
            canonical_hash=_evaluation_canonical_hash(evaluation, resolved_transform),
            known_identities=evaluation_identities,
        )
        resolved_evaluations.append(
            ResolvedEvaluation(
                spec=evaluation,
                transform=resolved_transform,
                evaluation_hash=evaluation_hash,
            )
        )
    trace_evaluation_identities: dict[str, str] = {}
    resolved_trace_evaluations: list[ResolvedTraceEvaluation] = []
    for trace_evaluation in task.trace_evaluations:
        trace_score_hash = _hash_payload(
            trace_evaluation.canonical_dict(),
            short=True,
        )
        canonical_trace_score_hash = _hash_payload(
            trace_evaluation.canonical_dict(),
            short=False,
        )
        _record_short_hash_identity(
            stage_kind="trace evaluation",
            stage_label=trace_evaluation.name,
            short_hash=trace_score_hash,
            canonical_hash=canonical_trace_score_hash,
            known_identities=trace_evaluation_identities,
        )
        resolved_trace_evaluations.append(
            ResolvedTraceEvaluation(
                spec=trace_evaluation,
                trace_score_hash=trace_score_hash,
            )
        )
    return ResolvedTaskStages(
        output_transforms=resolved_transform_tuple,
        evaluations=tuple(resolved_evaluations),
        trace_evaluations=tuple(resolved_trace_evaluations),
    )


def _resolve_transform_reference(
    evaluation: EvaluationSpec,
    transforms_by_name: dict[str, ResolvedOutputTransform],
) -> ResolvedOutputTransform | None:
    transform_name = evaluation.transform
    if transform_name is None:
        return None
    try:
        return transforms_by_name[transform_name]
    except KeyError as exc:  # pragma: no cover - TaskSpec validation should catch this.
        raise ValueError(
            f"Evaluation '{evaluation.name}' references unknown output transform "
            f"'{transform_name}'."
        ) from exc


def _evaluation_hash(
    evaluation: EvaluationSpec, transform: ResolvedOutputTransform | None
) -> str:
    return _hash_payload(
        _evaluation_payload(
            evaluation,
            transform,
            use_canonical_transform_hash=False,
        ),
        short=True,
    )


def _evaluation_canonical_hash(
    evaluation: EvaluationSpec, transform: ResolvedOutputTransform | None
) -> str:
    return _hash_payload(
        _evaluation_payload(
            evaluation,
            transform,
            use_canonical_transform_hash=True,
        ),
        short=False,
    )


def _evaluation_payload(
    evaluation: EvaluationSpec,
    transform: ResolvedOutputTransform | None,
    *,
    use_canonical_transform_hash: bool,
) -> dict[str, object]:
    return {
        "evaluation": evaluation.canonical_dict(),
        "transform_hash": (
            None
            if transform is None
            else (
                transform.spec.canonical_hash
                if use_canonical_transform_hash
                else transform.transform_hash
            )
        ),
    }


def _hash_payload(payload: object, *, short: bool) -> str:
    encoded = json.dumps(
        payload, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    if short:
        return digest[:12]
    return digest


def _record_short_hash_identity(
    *,
    stage_kind: str,
    stage_label: str,
    short_hash: str,
    canonical_hash: str,
    known_identities: dict[str, str],
) -> None:
    existing_canonical_hash = known_identities.get(short_hash)
    if existing_canonical_hash is None:
        known_identities[short_hash] = canonical_hash
        return
    if existing_canonical_hash == canonical_hash:
        return
    raise SpecValidationError(
        code=ErrorCode.SCHEMA_MISMATCH,
        message=(
            f"{stage_kind} short hash collision: '{stage_label}' shares "
            f"'{short_hash}' with a different canonical identity"
        ),
        details={
            "stage_kind": stage_kind,
            "stage_label": stage_label,
            "short_hash": short_hash,
            "canonical_hash": canonical_hash,
            "existing_canonical_hash": existing_canonical_hash,
        },
    )
