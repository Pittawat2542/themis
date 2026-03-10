"""Comparison read models and summary-row based statistical table assembly."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from themis._optional import import_optional
from themis.types.events import ScoreRow, TrialSummaryRow

if TYPE_CHECKING:
    from themis.stats.stats_engine import StatsEngine


class ComparisonRow(BaseModel):
    """Paired statistical comparison for one task/metric/model pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    metric_id: str
    baseline_model_id: str
    treatment_model_id: str
    pair_count: int
    baseline_mean: float
    treatment_mean: float
    delta_mean: float
    p_value: float
    adjusted_p_value: float
    adjustment_method: str
    ci_lower: float
    ci_upper: float
    ci_level: float
    method: str


class ComparisonTable(BaseModel):
    """Comparison-oriented read model over paired statistical outputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: list[ComparisonRow] = Field(default_factory=list)


def _score_frame(
    trial_summaries: list[TrialSummaryRow],
    score_rows: list[ScoreRow],
):
    pd = import_optional("pandas", extra="stats")

    trial_metadata = {
        trial.trial_hash: {
            "model_id": trial.model_id,
            "task_id": trial.task_id,
            "item_id": trial.item_id,
        }
        for trial in trial_summaries
        if trial.model_id is not None
        and trial.task_id is not None
        and trial.item_id is not None
    }
    rows: list[dict[str, str | float]] = []
    for row in score_rows:
        metadata = trial_metadata.get(row.trial_hash)
        if metadata is None:
            continue
        rows.append(
            {
                "trial_hash": row.trial_hash,
                "candidate_id": row.candidate_id,
                "metric_id": row.metric_id,
                "score": row.score,
                "model_id": metadata["model_id"],
                "task_id": metadata["task_id"],
                "item_id": metadata["item_id"],
            }
        )
    return pd.DataFrame(rows)


def _pair_models(
    model_ids: list[str],
    *,
    baseline_model_id: str | None = None,
    treatment_model_id: str | None = None,
) -> list[tuple[str, str]]:
    unique_model_ids = sorted(set(model_ids))
    if baseline_model_id and treatment_model_id:
        return (
            [(baseline_model_id, treatment_model_id)]
            if baseline_model_id in unique_model_ids
            and treatment_model_id in unique_model_ids
            else []
        )
    if baseline_model_id:
        return [
            (baseline_model_id, model_id)
            for model_id in unique_model_ids
            if model_id != baseline_model_id
        ]
    if treatment_model_id:
        return [
            (model_id, treatment_model_id)
            for model_id in unique_model_ids
            if model_id != treatment_model_id
        ]
    return list(combinations(unique_model_ids, 2))


def build_comparison_table(
    trial_summaries: list[TrialSummaryRow],
    score_rows: list[ScoreRow],
    *,
    stats_engine: "StatsEngine | None" = None,
    metric_id: str | None = None,
    task_id: str | None = None,
    baseline_model_id: str | None = None,
    treatment_model_id: str | None = None,
    p_value_correction: str = "none",
) -> ComparisonTable:
    """Build paired comparison rows from trial summaries and metric scores."""
    stats_module = import_optional("themis.stats.stats_engine", extra="stats")

    engine = stats_engine or stats_module.StatsEngine()
    df = _score_frame(trial_summaries, score_rows)
    if df.empty:
        return ComparisonTable()

    if metric_id is not None:
        df = df[df["metric_id"] == metric_id]
    if task_id is not None:
        df = df[df["task_id"] == task_id]
    if df.empty:
        return ComparisonTable()

    trial_scores = (
        df.groupby(["task_id", "metric_id", "model_id", "item_id"], dropna=False)[
            "score"
        ]
        .mean()
        .reset_index(name="trial_score")
    )

    rows: list[ComparisonRow] = []
    for (current_task_id, current_metric_id), group in trial_scores.groupby(
        ["task_id", "metric_id"], dropna=False
    ):
        pivot = group.pivot_table(
            index="item_id",
            columns="model_id",
            values="trial_score",
            aggfunc="mean",
        )
        if pivot.empty:
            continue

        for baseline_model, treatment_model in _pair_models(
            list(pivot.columns),
            baseline_model_id=baseline_model_id,
            treatment_model_id=treatment_model_id,
        ):
            paired = pivot[[baseline_model, treatment_model]].dropna()
            if paired.empty:
                continue

            result = engine.paired_bootstrap(
                baseline_scores=paired[baseline_model].to_numpy(),
                treatment_scores=paired[treatment_model].to_numpy(),
            )
            rows.append(
                ComparisonRow(
                    task_id=str(current_task_id),
                    metric_id=str(current_metric_id),
                    baseline_model_id=baseline_model,
                    treatment_model_id=treatment_model,
                    pair_count=int(len(paired)),
                    baseline_mean=result.baseline_mean or 0.0,
                    treatment_mean=result.treatment_mean or 0.0,
                    delta_mean=result.delta_mean or 0.0,
                    p_value=result.p_value or 0.0,
                    adjusted_p_value=result.p_value or 0.0,
                    adjustment_method="none",
                    ci_lower=result.ci_lower or 0.0,
                    ci_upper=result.ci_upper or 0.0,
                    ci_level=result.ci_level,
                    method=result.method,
                )
            )

    adjusted_p_values = engine.adjust_p_values(
        [row.p_value for row in rows],
        method=p_value_correction,
    )
    rows = [
        row.model_copy(
            update={
                "adjusted_p_value": adjusted_p_values[index],
                "adjustment_method": p_value_correction,
            }
        )
        for index, row in enumerate(rows)
    ]
    rows.sort(
        key=lambda row: (
            row.task_id,
            row.metric_id,
            row.baseline_model_id,
            row.treatment_model_id,
        )
    )
    return ComparisonTable(rows=rows)
