from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from themis import PromptMessage, PromptTurnSpec, ToolSpec
from themis.specs.experiment import RuntimeContext
from themis.types.enums import PromptRole


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_example_module(example_name: str):
    module_path = PROJECT_ROOT / "examples" / example_name
    spec = importlib.util.spec_from_file_location(
        example_name.replace(".", "_"), module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_run_example_uses_deterministic_inference_ids() -> None:
    module = _load_example_module("05_resume_run.py")
    engine = module.ResumeEngine()

    first = engine.infer(None, {"item_id": "item-1", "answer": "7"}, None)
    second = engine.infer(None, {"item_id": "item-1", "answer": "7"}, None)

    assert first.inference is not None
    assert second.inference is not None
    assert first.inference.spec_hash == "inf_item-1"
    assert second.inference.spec_hash == "inf_item-1"


def test_agent_eval_example_executes_registered_tool_handler() -> None:
    module = _load_example_module("10_agent_eval.py")
    engine = module.ScriptedAgentEngine()
    seen: dict[str, object] = {}

    def calculator_handler(arguments: object) -> dict[str, str]:
        seen["arguments"] = arguments
        return {"value": "42"}

    trial = SimpleNamespace(
        item_id="item-1",
        prompt=SimpleNamespace(
            messages=[
                PromptMessage(role=PromptRole.SYSTEM, content="System"),
                PromptMessage(role=PromptRole.USER, content="What is 6 * 7?"),
            ],
            follow_up_turns=[
                PromptTurnSpec(
                    messages=[
                        PromptMessage(role=PromptRole.USER, content="Double-check it.")
                    ]
                )
            ],
        ),
        tools=[
            ToolSpec(
                id="calculator",
                description="Benchmark arithmetic tool.",
                input_schema={"type": "object"},
            )
        ],
    )

    result = engine.infer(
        trial,
        {"item_id": "wrong-item", "answer": "not-used"},
        RuntimeContext(tool_handlers={"calculator": calculator_handler}),
    )

    assert seen["arguments"] == {"expression": "6 * 7"}
    assert result.inference.spec_hash == "inf_item-1"
    assert result.inference.raw_text == "42"
    assert result.inference.conversation is None
    assert result.conversation is not None
    assert result.conversation.events[2].payload.result["value"] == "42"


def test_catalog_builtin_example_uses_fixture_loader() -> None:
    module = _load_example_module("13_catalog_builtin_benchmark.py")

    rows = module.load_fixture_rows("TIGER-Lab/MMLU-Pro", "test", None)

    assert rows == [
        {
            "item_id": "mmlu-pro-1",
            "question": "Which planet is known as the Red Planet?",
            "options": ["Venus", "Mars", "Jupiter", "Mercury"],
            "answer": "B",
            "answer_index": 1,
            "category": "astronomy",
            "src": "fixture",
        }
    ]
