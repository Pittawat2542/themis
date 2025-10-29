"""Experiment orchestrator primitives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Sequence

from themis.config.schema import IntegrationsConfig
from themis.core.entities import (ExperimentFailure, ExperimentReport, GenerationRecord, GenerationTask, MetricScore, EvaluationRecord)
from themis.evaluation import pipeline as evaluation_pipeline
from themis.evaluation.reports import EvaluationFailure
from themis.generation import plan as generation_plan
from themis.generation import runner as generation_runner
from themis.experiment import storage as experiment_storage
from themis.integrations.huggingface import HuggingFaceHubUploader
from themis.integrations.wandb import WandbTracker





class ExperimentOrchestrator:
    def __init__(
        self,
        *,
        generation_plan: generation_plan.GenerationPlan,
        generation_runner: generation_runner.GenerationRunner,
        evaluation_pipeline: evaluation_pipeline.EvaluationPipeline,
        storage: experiment_storage.ExperimentStorage | None = None,
        integrations_config: IntegrationsConfig | None = None,
    ) -> None:
        self._plan = generation_plan
        self._runner = generation_runner
        self._evaluation = evaluation_pipeline
        self._storage = storage
        self._integrations_config = integrations_config or IntegrationsConfig()
        self._wandb_tracker = (
            WandbTracker(self._integrations_config.wandb)
            if self._integrations_config.wandb.enable
            else None
        )
        self._huggingface_hub_uploader = (
            HuggingFaceHubUploader(self._integrations_config.huggingface_hub)
            if self._integrations_config.huggingface_hub.enable
            else None
        )

    def run(
        self,
        dataset: Sequence[dict[str, object]] | None = None,
        *,
        dataset_loader: Callable[[], Sequence[dict[str, object]]] | None = None,
        max_samples: int | None = None,
        run_id: str | None = None,
        resume: bool = True,
        cache_results: bool = True,
        on_result: Callable[[GenerationRecord], None] | None = None,
    ) -> ExperimentReport:
        if self._wandb_tracker:
            self._wandb_tracker.init(
                {
                    "max_samples": max_samples,
                    "run_id": run_id,
                    "resume": resume,
                }
            )
        dataset_list = self._resolve_dataset(
            dataset=dataset, dataset_loader=dataset_loader, run_id=run_id
        )
        selected_dataset = (
            dataset_list[:max_samples] if max_samples is not None else dataset_list
        )
        run_identifier = run_id or self._default_run_id()

        if self._storage is not None and dataset_list:
            self._storage.cache_dataset(run_identifier, dataset_list)

        tasks = list(self._plan.expand(selected_dataset))
        cached_records = self._load_cached_records(run_identifier) if resume else {}
        cached_evaluations = (
            self._load_cached_evaluations(run_identifier) if resume else {}
        )
        generation_results: list[GenerationRecord] = []
        failures: list[ExperimentFailure] = []

        pending_tasks: list[GenerationTask] = []
        pending_records: list[GenerationRecord] = []
        pending_keys: list[str] = []
        cached_eval_records: list[EvaluationRecord] = []
        for task in tasks:
            cache_key = experiment_storage.task_cache_key(task)
            cached = cached_records.get(cache_key)
            if cached is not None:
                generation_results.append(cached)
                if cached.error:
                    failures.append(
                        ExperimentFailure(
                            sample_id=cached.task.metadata.get("dataset_id"),
                            message=cached.error.message,
                        )
                    )
                evaluation = cached_evaluations.get(cache_key)
                if evaluation is not None:
                    cached_eval_records.append(evaluation)
                else:
                    pending_records.append(cached)
                    pending_keys.append(cache_key)
                if on_result:
                    on_result(cached)
            else:
                pending_tasks.append(task)

        if pending_tasks:
            for record in self._runner.run(pending_tasks):
                generation_results.append(record)
                if record.error:
                    failures.append(
                        ExperimentFailure(
                            sample_id=record.task.metadata.get("dataset_id"),
                            message=record.error.message,
                        )
                    )
                cache_key = experiment_storage.task_cache_key(record.task)
                if self._storage is not None and cache_results:
                    self._storage.append_record(
                        run_identifier, record, cache_key=cache_key
                    )
                pending_records.append(record)
                pending_keys.append(cache_key)
                if on_result:
                    on_result(record)

        if pending_records:
            new_evaluation_report = self._evaluation.evaluate(pending_records)
        else:
            new_evaluation_report = evaluation_pipeline.EvaluationReport(
                metrics={}, failures=[], records=[]
            )

        if self._storage is not None:
            for record, evaluation in zip(
                pending_records, new_evaluation_report.records
            ):
                self._storage.append_evaluation(run_identifier, record, evaluation)

        evaluation_report = self._combine_evaluations(
            cached_eval_records, new_evaluation_report
        )
        metadata = {
            "total_samples": len(selected_dataset),
            "successful_generations": sum(
                1 for result in generation_results if not result.error
            ),
            "failed_generations": sum(
                1 for result in generation_results if result.error
            ),
            "run_id": run_identifier,
            "evaluation_failures": sum(
                1 for record in evaluation_report.records if record.failures
            )
            + len(evaluation_report.failures),
        }

        report = ExperimentReport(
            generation_results=generation_results,
            evaluation_report=evaluation_report,
            failures=failures,
            metadata=metadata,
        )
        if self._wandb_tracker:
            self._wandb_tracker.log_results(report)
        if self._huggingface_hub_uploader and self._storage:
            self._huggingface_hub_uploader.upload_results(
                report, self._storage.get_run_path(run_identifier)
            )
        return report

    def _default_run_id(self) -> str:
        return datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")

    def _resolve_dataset(
        self,
        *,
        dataset: Sequence[dict[str, object]] | None,
        dataset_loader: Callable[[], Sequence[dict[str, object]]] | None,
        run_id: str | None,
    ) -> list[dict[str, object]]:
        if dataset is not None:
            return list(dataset)
        if dataset_loader is not None:
            return list(dataset_loader())
        if self._storage is not None and run_id is not None:
            return self._storage.load_dataset(run_id)
        raise ValueError(
            "No dataset provided. Supply `dataset=` rows, a `dataset_loader`, "
            "or set `run_id` with storage configured so cached data can be reloaded."
        )

    def _load_cached_records(
        self, run_id: str
    ) -> dict[str, GenerationRecord]:
        if self._storage is None:
            return {}
        return self._storage.load_cached_records(run_id)

    def _load_cached_evaluations(
        self, run_id: str
    ) -> dict[str, EvaluationRecord]:
        if self._storage is None:
            return {}
        return self._storage.load_cached_evaluations(run_id)

    def _combine_evaluations(
        self,
        cached_records: list[EvaluationRecord],
        new_report: evaluation_pipeline.EvaluationReport,
    ) -> evaluation_pipeline.EvaluationReport:
        all_records = list(cached_records) + list(new_report.records)
        per_metric: dict[str, list[MetricScore]] = {}
        for record in all_records:
            for score in record.scores:
                per_metric.setdefault(score.metric_name, []).append(score)

        aggregates: dict[str, evaluation_pipeline.MetricAggregate] = {}
        for name, scores in per_metric.items():
            mean = sum(score.value for score in scores) / len(scores) if scores else 0.0
            aggregates[name] = evaluation_pipeline.MetricAggregate(
                name=name,
                count=len(scores),
                mean=mean,
                per_sample=scores,
            )

        failures = list(new_report.failures)
        for record in cached_records:
            for message in record.failures:
                failures.append(
                    EvaluationFailure(sample_id=record.sample_id, message=message)
                )

        return evaluation_pipeline.EvaluationReport(
            metrics=aggregates,
            failures=failures,
            records=all_records,
        )
