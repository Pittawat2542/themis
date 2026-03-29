"""Benchmark-specific adapter helpers."""

from __future__ import annotations

from themis.core.base import JSONValue


def apply_benchmark_adapter(
    adapter: str | None,
    *,
    base_name: str,
    benchmark_id: str,
    spec: dict[str, object],
    variant: str | None,
) -> dict[str, object]:
    del variant
    payload: dict[str, object] = {
        "candidate_policy": {"num_samples": 1},
        "workflow_overrides": {},
        "dataset_metadata": {},
        "sample_case_id": f"{base_name}-sample-1",
        "sample_case_input": {"prompt": f"sample prompt for {benchmark_id}"},
        "sample_case_expected_output": {"answer": f"sample answer for {base_name}"},
        "sample_case_metadata": {},
    }
    if adapter is None:
        return payload
    if adapter == "rubric_qa":
        payload.update(_rubric_qa_adapter(base_name, benchmark_id))
        return payload
    if adapter == "panel_qa":
        payload.update(_panel_qa_adapter(base_name, benchmark_id))
        return payload
    if adapter == "code_generation":
        payload.update(_code_generation_adapter(base_name, benchmark_id, spec))
        return payload
    raise ValueError(f"Unknown benchmark adapter: {adapter}")


def _rubric_qa_adapter(base_name: str, benchmark_id: str) -> dict[str, object]:
    return {
        "metric_ids": ["builtin/llm_rubric"],
        "judge_model_ids": ["builtin/demo_judge"],
        "workflow_overrides": {
            "rubric": "Judge whether the scientific answer is correct and well-supported."
            if base_name == "frontierscience"
            else "Judge whether the answer is correct, specific, and well-supported."
        },
        "sample_case_input": {"question": f"Answer the benchmark question for {benchmark_id}."},
        "sample_case_expected_output": {"answer": f"sample answer for {base_name}"},
    }


def _panel_qa_adapter(base_name: str, benchmark_id: str) -> dict[str, object]:
    return {
        "metric_ids": ["builtin/panel_of_judges"],
        "judge_model_ids": ["builtin/demo_judge", "builtin/demo_judge"],
        "workflow_overrides": {"rubric": "Judge whether the answer is correct and sufficiently supported."},
        "sample_case_input": {"question": f"Answer the benchmark question for {benchmark_id}."},
        "sample_case_expected_output": {"answer": f"sample answer for {base_name}"},
    }


def _code_generation_adapter(
    base_name: str,
    benchmark_id: str,
    spec: dict[str, object],
) -> dict[str, object]:
    backends = ",".join(list(spec.get("supported_execution_backends", [])))
    return {
        "candidate_policy": {"num_samples": 2},
        "reducer_id": "builtin/best_of_n",
        "judge_model_ids": ["builtin/demo_judge"],
        "dataset_metadata": {
            "execution_kind": "code_generation",
            "supported_execution_backends": backends,
        },
        "sample_case_input": {
            "problem": f"Solve the programming task for {benchmark_id}.",
            "language": "python",
        },
        "sample_case_expected_output": {
            "solution": "def solve():\n    return 4",
            "language": "python",
        },
        "sample_case_metadata": {"benchmark_family": base_name},
    }
