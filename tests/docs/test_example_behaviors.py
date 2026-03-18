from __future__ import annotations

import importlib.util
from pathlib import Path


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
