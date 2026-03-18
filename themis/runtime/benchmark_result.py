"""Benchmark-native result facade over stored run projections."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Iterator

from themis.overlays import OverlaySelection
from themis.runtime.comparison import build_comparison_table
from themis.runtime.experiment_result import ExperimentResult
from themis.types.enums import PValueCorrection
from themis.types.events import ScoreRow, TrialSummaryRow
from themis.types.json_types import JSONDict, JSONValueType


@dataclass(frozen=True, slots=True)
class ArtifactBundle:
    """Paths for benchmark-native handoff artifacts."""

    aggregate_json_path: Path
    summary_markdown_path: Path


class BenchmarkResult(ExperimentResult):
    """Public result facade that speaks benchmark-native semantics."""

    def __init__(
        self,
        *,
        projection_repo,
        trial_hashes: list[str],
        transform_hashes: list[str] | None = None,
        evaluation_hashes: list[str] | None = None,
        active_transform_hash: str | None = None,
        active_evaluation_hash: str | None = None,
        benchmark_id: str | None = None,
        slice_ids: list[str] | None = None,
        prompt_variant_ids: list[str] | None = None,
    ) -> None:
        super().__init__(
            projection_repo=projection_repo,
            trial_hashes=trial_hashes,
            transform_hashes=transform_hashes,
            evaluation_hashes=evaluation_hashes,
            active_transform_hash=active_transform_hash,
            active_evaluation_hash=active_evaluation_hash,
        )
        self.benchmark_id = benchmark_id
        self.slice_ids = list(slice_ids or [])
        self.prompt_variant_ids = list(prompt_variant_ids or [])

    def aggregate(
        self,
        *,
        group_by: list[str],
        metric_id: str | None = None,
    ) -> list[JSONDict]:
        """Aggregate score rows using benchmark-native summary fields."""

        summaries = {row.trial_hash: row for row in self.iter_trial_summaries()}
        self._validate_group_by_keys(summaries.values(), group_by)
        groups: dict[tuple[JSONValueType, ...], list[float]] = {}
        for row in self._iter_scores(metric_id=metric_id):
            summary = summaries.get(row.trial_hash)
            if summary is None:
                continue
            key_payload = self._group_payload(summary, row, group_by)
            groups.setdefault(tuple(key_payload.values()), []).append(row.score)

        results: list[JSONDict] = []
        for key, scores in sorted(
            groups.items(), key=lambda item: self._sort_group_key(item[0])
        ):
            payload = dict(zip(group_by, key, strict=True))
            payload["mean"] = sum(scores) / len(scores)
            payload["count"] = len(scores)
            results.append(payload)
        return results

    def paired_compare(
        self,
        *,
        metric_id: str,
        group_by: str = "slice_id",
        baseline_model_id: str | None = None,
        treatment_model_id: str | None = None,
        p_value_correction: PValueCorrection | str = PValueCorrection.NONE,
    ) -> list[JSONDict]:
        """Return paired comparisons by one benchmark grouping key."""

        trial_summaries = list(self.iter_trial_summaries())
        self._validate_group_by_keys(trial_summaries, [group_by])
        relevant_scores = list(self._iter_scores(metric_id=metric_id))
        scores_by_trial: dict[str, list[ScoreRow]] = {}
        for row in relevant_scores:
            scores_by_trial.setdefault(row.trial_hash, []).append(row)

        summaries_by_group: dict[JSONValueType, list[TrialSummaryRow]] = {}
        for summary in trial_summaries:
            group_value = self._resolve_group_value(
                summary,
                group_by,
                metric_id=metric_id,
            )
            summaries_by_group.setdefault(group_value, []).append(summary)

        comparison_rows: list[JSONDict] = []
        for group_value in sorted(summaries_by_group, key=self._group_value_sort_key):
            group_summaries = summaries_by_group[group_value]
            group_trial_hashes = {summary.trial_hash for summary in group_summaries}
            group_scores = [
                score
                for trial_hash in group_trial_hashes
                for score in scores_by_trial.get(trial_hash, [])
            ]
            table = build_comparison_table(
                group_summaries,
                group_scores,
                metric_id=metric_id,
                baseline_model_id=baseline_model_id,
                treatment_model_id=treatment_model_id,
                p_value_correction=p_value_correction,
            )
            for comparison_row in table.rows:
                payload = {
                    group_by: group_value,
                    "metric_id": comparison_row.metric_id,
                    "baseline_model_id": comparison_row.baseline_model_id,
                    "treatment_model_id": comparison_row.treatment_model_id,
                    "pair_count": comparison_row.pair_count,
                    "baseline_mean": comparison_row.baseline_mean,
                    "treatment_mean": comparison_row.treatment_mean,
                    "delta_mean": comparison_row.delta_mean,
                    "p_value": comparison_row.p_value,
                    "adjusted_p_value": comparison_row.adjusted_p_value,
                    "adjustment_method": comparison_row.adjustment_method,
                    "ci_lower": comparison_row.ci_lower,
                    "ci_upper": comparison_row.ci_upper,
                    "ci_level": comparison_row.ci_level,
                    "method": comparison_row.method,
                }
                comparison_rows.append(payload)
        return comparison_rows

    def for_transform(self, transform_hash: str) -> "BenchmarkResult":
        return BenchmarkResult(
            projection_repo=self.projection_repo,
            trial_hashes=self.trial_hashes,
            transform_hashes=self.transform_hashes,
            evaluation_hashes=self.evaluation_hashes,
            active_transform_hash=transform_hash,
            active_evaluation_hash=None,
            benchmark_id=self.benchmark_id,
            slice_ids=self.slice_ids,
            prompt_variant_ids=self.prompt_variant_ids,
        )

    def for_evaluation(self, evaluation_hash: str) -> "BenchmarkResult":
        return BenchmarkResult(
            projection_repo=self.projection_repo,
            trial_hashes=self.trial_hashes,
            transform_hashes=self.transform_hashes,
            evaluation_hashes=self.evaluation_hashes,
            active_transform_hash=None,
            active_evaluation_hash=evaluation_hash,
            benchmark_id=self.benchmark_id,
            slice_ids=self.slice_ids,
            prompt_variant_ids=self.prompt_variant_ids,
        )

    def persist_artifacts(
        self,
        *,
        storage_root: str | Path,
    ) -> ArtifactBundle:
        """Persist a small aggregate bundle for operator handoff."""

        root = Path(storage_root)
        root.mkdir(parents=True, exist_ok=True)
        aggregate_rows = self.aggregate(
            group_by=["model_id", "slice_id", "metric_id", "prompt_variant_id"]
        )
        scope = self._scope_metadata()
        scope_suffix = scope["overlay_key"].replace(":", "-")
        aggregate_path = root / f"benchmark-aggregate-{scope_suffix}.json"
        summary_path = root / f"benchmark-summary-{scope_suffix}.md"
        aggregate_path.write_text(
            json.dumps(
                {
                    "benchmark_id": self.benchmark_id,
                    "scope": scope,
                    "rows": aggregate_rows,
                },
                indent=2,
                sort_keys=True,
            )
        )
        summary_lines = ["# Benchmark Summary", ""]
        for row in aggregate_rows:
            mean_value = self._float_value(row, "mean")
            count_value = self._int_value(row, "count")
            summary_lines.append(
                "- "
                f"scope={scope['overlay_key']} "
                f"model={row.get('model_id')} "
                f"slice={row.get('slice_id')} "
                f"metric={row.get('metric_id')} "
                f"prompt={row.get('prompt_variant_id')} "
                f"mean={mean_value:.4f} "
                f"count={count_value}"
            )
        summary_path.write_text("\n".join(summary_lines) + "\n")
        return ArtifactBundle(
            aggregate_json_path=aggregate_path,
            summary_markdown_path=summary_path,
        )

    def _iter_scores(self, *, metric_id: str | None) -> Iterator[ScoreRow]:
        yield from self.projection_repo.iter_candidate_scores(
            trial_hashes=self.trial_hashes,
            metric_id=metric_id,
            evaluation_hash=self.active_evaluation_hash,
        )

    def _group_payload(
        self,
        summary: TrialSummaryRow,
        score_row: ScoreRow,
        group_by: list[str],
    ) -> JSONDict:
        payload: JSONDict = {}
        for key in group_by:
            payload[key] = self._resolve_group_value(
                summary,
                key,
                metric_id=score_row.metric_id,
            )
        return payload

    def _resolve_group_value(
        self,
        summary: TrialSummaryRow,
        key: str,
        *,
        metric_id: str | None = None,
    ) -> JSONValueType:
        if key == "metric_id":
            return metric_id
        if key in {
            "benchmark_id",
            "slice_id",
            "prompt_variant_id",
            "model_id",
            "item_id",
            "status",
        }:
            return getattr(summary, key)
        if key in summary.dimensions:
            return summary.dimensions[key]
        return None

    def _sort_group_key(
        self, values: tuple[JSONValueType, ...]
    ) -> tuple[tuple[int, str], ...]:
        return tuple(self._group_value_sort_key(value) for value in values)

    def _group_value_sort_key(self, value: JSONValueType) -> tuple[int, str]:
        if value is None:
            return (0, "")
        return (1, str(value))

    def _float_value(self, row: JSONDict, key: str) -> float:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{key} must be numeric, got {value!r}")
        return float(value)

    def _int_value(self, row: JSONDict, key: str) -> int:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{key} must be an int, got {value!r}")
        return value

    def _scope_metadata(self) -> dict[str, str]:
        return OverlaySelection(
            transform_hash=self.active_transform_hash,
            evaluation_hash=self.active_evaluation_hash,
        ).metadata()

    def _validate_group_by_keys(
        self,
        summaries: Iterable[TrialSummaryRow],
        group_by: list[str],
    ) -> None:
        supported_keys = {
            "metric_id",
            "benchmark_id",
            "slice_id",
            "prompt_variant_id",
            "model_id",
            "item_id",
            "status",
        }
        dimension_keys = {key for summary in summaries for key in summary.dimensions}
        unknown_keys = sorted(set(group_by) - supported_keys - dimension_keys)
        if unknown_keys:
            raise ValueError(f"Unsupported group_by key: {', '.join(unknown_keys)}")
