from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import wandb
else:
    try:
        import wandb
    except ImportError:
        wandb = None  # type: ignore

from themis.config.schema import WandbConfig
from themis.core.entities import ExperimentReport


class WandbTracker:
    def __init__(self, config: WandbConfig):
        if wandb is None:
            from themis.exceptions import DependencyError

            raise DependencyError(
                "wandb is not installed. Install with: pip install wandb"
            )
        self.config = config

    def init(self, experiment_config: dict) -> None:
        if not self.config.enable:
            return
        wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            tags=self.config.tags,
            config=experiment_config,
        )

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to WandB.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number
        """
        if not self.config.enable:
            return
        wandb.log(metrics, step=step)

    def log_results(self, report: ExperimentReport) -> None:
        if not self.config.enable:
            return
        summary = {
            "total_samples": report.metadata.get("total_samples"),
            "successful_generations": report.metadata.get("successful_generations"),
            "failed_generations": report.metadata.get("failed_generations"),
            "evaluation_failures": report.metadata.get("evaluation_failures"),
        }
        for name, aggregate in report.evaluation_report.metrics.items():
            summary[f"{name}_mean"] = aggregate.mean
        wandb.summary.update(summary)

        records_table = wandb.Table(
            columns=[
                "sample_id",
                "prompt",
                "raw_response",
                "error",
                "metric_scores",
            ]
        )
        for record in report.generation_results:
            eval_record = next(
                (
                    r
                    for r in report.evaluation_report.records
                    if r.sample_id == record.task.metadata.get("dataset_id")
                ),
                None,
            )
            records_table.add_data(
                record.task.metadata.get("dataset_id"),
                record.task.prompt.text,
                [record.output.text] if record.output else [],
                record.error.message if record.error else None,
                {s.metric_name: s.value for s in eval_record.scores}
                if eval_record
                else None,
            )
        wandb.log({"generation_results": records_table})

    def finalize(self) -> None:
        """Finalize the WandB run."""
        if not self.config.enable:
            return
        wandb.finish()
