"""Benchmark-specific adapter helpers."""

from __future__ import annotations

from collections.abc import Iterable


def apply_benchmark_adapter(
    adapter: str | None,
    *,
    base_name: str,
    benchmark_id: str,
    spec: dict[str, object],
    variant: str | None,
) -> dict[str, object]:
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
        payload.update(_code_generation_adapter(base_name, benchmark_id, spec, variant))
        return payload
    if adapter == "rolebench":
        payload.update(_rolebench_adapter(base_name, benchmark_id, variant))
        return payload
    if adapter == "procbench":
        payload.update(_procbench_adapter(base_name, benchmark_id, variant))
        return payload
    if adapter == "mmmlu":
        payload.update(_mmmlu_adapter(base_name, benchmark_id, variant))
        return payload
    if adapter == "superchem":
        payload.update(_superchem_adapter(base_name, benchmark_id, variant))
        return payload
    if adapter == "hle":
        payload.update(_hle_adapter(base_name, benchmark_id, variant))
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
        "sample_case_input": {
            "question": f"Answer the benchmark question for {benchmark_id}."
        },
        "sample_case_expected_output": {"answer": f"sample answer for {base_name}"},
    }


def _panel_qa_adapter(base_name: str, benchmark_id: str) -> dict[str, object]:
    return {
        "metric_ids": ["builtin/panel_of_judges"],
        "judge_model_ids": ["builtin/demo_judge", "builtin/demo_judge"],
        "workflow_overrides": {
            "rubric": "Judge whether the answer is correct and sufficiently supported."
        },
        "sample_case_input": {
            "question": f"Answer the benchmark question for {benchmark_id}."
        },
        "sample_case_expected_output": {"answer": f"sample answer for {base_name}"},
    }


def _code_generation_adapter(
    base_name: str,
    benchmark_id: str,
    spec: dict[str, object],
    variant: str | None,
) -> dict[str, object]:
    backends = ",".join(_coerce_string_list(spec.get("supported_execution_backends")))
    return {
        "candidate_policy": {"num_samples": 2},
        "selector_id": "builtin/best_of_n",
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
        "sample_case_metadata": {
            "benchmark_family": base_name,
            "variant": variant or "",
        },
    }


def _rolebench_adapter(
    base_name: str, benchmark_id: str, variant: str | None
) -> dict[str, object]:
    variant_name = variant or "default"
    return {
        "metric_ids": ["builtin/llm_rubric"],
        "judge_model_ids": ["builtin/demo_judge"],
        "workflow_overrides": {
            "rubric": "Judge whether the response follows the requested role behavior and generalizes correctly."
        },
        "sample_case_input": {
            "instruction": f"Respond as required for {benchmark_id}.",
            "role": variant_name,
        },
        "sample_case_expected_output": {
            "answer": f"sample role response for {base_name}"
        },
        "sample_case_metadata": {"variant": variant_name},
    }


def _procbench_adapter(
    base_name: str, benchmark_id: str, variant: str | None
) -> dict[str, object]:
    task_id = variant or "task01"
    return {
        "metric_ids": ["builtin/llm_rubric"],
        "judge_model_ids": ["builtin/demo_judge"],
        "workflow_overrides": {
            "rubric": "Judge whether the procedure was followed correctly and completely."
        },
        "sample_case_input": {
            "task": f"Complete procbench {task_id} for {benchmark_id}.",
        },
        "sample_case_expected_output": {
            "answer": f"sample procedure result for {base_name}"
        },
        "sample_case_metadata": {"task_id": task_id},
    }


def _mmmlu_adapter(
    base_name: str, benchmark_id: str, variant: str | None
) -> dict[str, object]:
    language = variant or "default"
    return {
        "sample_case_input": {
            "question": f"Answer the MMMLU question for {benchmark_id}.",
            "language": language,
        },
        "sample_case_expected_output": {"answer": f"sample answer for {base_name}"},
        "sample_case_metadata": {"language_config": language},
    }


def _superchem_adapter(
    base_name: str, benchmark_id: str, variant: str | None
) -> dict[str, object]:
    language = variant or "en"
    return {
        "sample_case_input": {
            "question": f"Answer the chemistry question for {benchmark_id}.",
            "language": language,
        },
        "sample_case_expected_output": {
            "answer": f"sample chemistry answer for {base_name}"
        },
        "sample_case_metadata": {"language": language},
    }


def _hle_adapter(
    base_name: str, benchmark_id: str, variant: str | None
) -> dict[str, object]:
    domains = variant or "default"
    return {
        "metric_ids": ["builtin/panel_of_judges"],
        "judge_model_ids": ["builtin/demo_judge", "builtin/demo_judge"],
        "workflow_overrides": {
            "rubric": "Judge whether the response demonstrates high-level expertise across the requested domains."
        },
        "sample_case_input": {
            "question": f"Answer the HLE question for {benchmark_id}.",
            "domains": domains,
        },
        "sample_case_expected_output": {
            "answer": f"sample expert answer for {base_name}"
        },
        "sample_case_metadata": {"domains": domains},
    }


def _coerce_string_list(value: object | None) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, dict)):
        raise ValueError(f"Expected iterable of strings, got {type(value).__name__}")
    items = list(value)
    if any(not isinstance(item, str) for item in items):
        raise ValueError("Expected iterable of strings")
    return items
