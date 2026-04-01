from __future__ import annotations

import importlib.util
from pathlib import Path


EXAMPLES_ROOT = Path(__file__).resolve().parents[1] / "examples" / "docs"


def _load_module(name: str):
    path = EXAMPLES_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"docs_example_{name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_python_tutorial_examples_run() -> None:
    for name in (
        "first_evaluate",
        "first_experiment",
        "llm_judged_evaluation",
        "advanced_run",
        "custom_parser",
        "custom_reducer",
        "observability",
        "pure_metrics",
        "workflow_metrics",
        "external_execution",
    ):
        module = _load_module(name)
        payload = module.run_example()

        assert payload["status"] == "completed"
        assert payload["run_id"]


def test_persisted_run_example_uses_sqlite_and_reports(tmp_path: Path) -> None:
    module = _load_module("persisted_run")

    payload = module.run_example(tmp_path)

    assert payload["status"] == "completed"
    assert payload["state_status"] == "completed"
    assert Path(payload["store_path"]).is_file()
    assert payload["report_preview"][0] == "# Run Report"


def test_custom_component_examples_run() -> None:
    generator_module = _load_module("custom_generator")
    metric_module = _load_module("custom_metric")
    trace_module = _load_module("trace_capture")
    openai_module = _load_module("provider_openai")
    vllm_module = _load_module("provider_vllm")
    langgraph_module = _load_module("provider_langgraph")
    observability_module = _load_module("observability")

    generator_payload = generator_module.run_example()
    metric_payload = metric_module.run_example()
    trace_payload = trace_module.run_example()
    openai_payload = openai_module.run_example()
    vllm_payload = vllm_module.run_example()
    langgraph_payload = langgraph_module.run_example()
    observability_payload = observability_module.run_example()

    assert generator_payload["status"] == "completed"
    assert metric_payload["status"] == "completed"
    assert "metric/exact_answer" in metric_payload["score_ids"]
    assert trace_payload["status"] == "completed"
    assert trace_payload["generation_trace_count"] == 1
    assert openai_payload["status"] == "completed"
    assert vllm_payload["status"] == "completed"
    assert langgraph_payload["status"] == "completed"
    assert openai_payload["artifact_keys"]
    assert vllm_payload["api_mode"] in {"responses", "chat_completions"}
    assert langgraph_payload["trace_steps"] >= 1
    assert observability_payload["status"] == "completed"
    assert "run" in observability_payload["span_names"]
    assert any(
        call.startswith("before_generate:")
        for call in observability_payload["subscriber_calls"]
    )


def test_rejudge_bundle_example_round_trips() -> None:
    module = _load_module("rejudge_bundle")

    payload = module.run_example()

    assert payload["imported"] is True
    assert payload["run_id"] == payload["replayed_run_id"]
