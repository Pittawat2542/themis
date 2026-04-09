"""Helpers for deriving stable dataset-scoped case identities."""

from __future__ import annotations

from pydantic import computed_field

from themis.core.base import FrozenModel


def case_key_for(dataset_id: str, case_id: str) -> str:
    """Return a stable string key for one dataset-scoped case."""

    return f"{len(dataset_id)}:{dataset_id}:{case_id}"


def resolve_case_key(
    *,
    case_id: str,
    dataset_id: str | None = None,
    case_key: str | None = None,
) -> str:
    """Resolve the best available case key for stored or live payloads."""

    if case_key:
        return case_key
    if dataset_id:
        return case_key_for(dataset_id, case_id)
    return case_id


class CaseRef(FrozenModel):
    """Dataset-scoped identity for one case."""

    dataset_id: str
    case_id: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def case_key(self) -> str:
        return case_key_for(self.dataset_id, self.case_id)
