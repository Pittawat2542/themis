"""Catalog definition for Procbench."""

from __future__ import annotations

from dataclasses import replace

from themis import BenchmarkDefinition

from ...common import build_procbench_benchmark, register_procbench, summarize_mcq
from .dataset import BuiltinProcbenchDatasetProvider

_SUPPORTED_PROCBENCH_TASK_IDS = tuple(f"task{index:02d}" for index in range(1, 24))


def _preview_rows(_definition: BenchmarkDefinition) -> list[dict[str, object]]:
    return [
        {
            "problem_name": "task01_0000",
            "prompt": "Sort the string.",
            "task_name": "task01",
            "label": {"final": "abct", "intermediate": ["bac"]},
        }
    ]


DEFINITION = BenchmarkDefinition(
    benchmark_id="procbench",
    family="catalog",
    primary_metric_id="procbench_final_accuracy",
    requires_judge=False,
    metadata={
        "dataset_id": "ifujisawa/procbench",
        "split": "train",
        "task_ids": list(_SUPPORTED_PROCBENCH_TASK_IDS),
    },
    builder=build_procbench_benchmark,
    registrar=register_procbench,
    summarizer=summarize_mcq,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinProcbenchDatasetProvider(huggingface_loader=huggingface_loader)
    ),
    preview_rows_loader=_preview_rows,
)


def supported_procbench_task_ids() -> tuple[str, ...]:
    return _SUPPORTED_PROCBENCH_TASK_IDS


def build_procbench_definition(task_ids: list[str]) -> BenchmarkDefinition:
    benchmark_id = "procbench"
    if len(task_ids) == 1:
        benchmark_id = f"procbench:{task_ids[0]}"
    return replace(
        DEFINITION,
        benchmark_id=benchmark_id,
        metadata={
            **DEFINITION.metadata,
            "task_ids": list(task_ids),
        },
    )
