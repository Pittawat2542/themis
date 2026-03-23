"""Catalog runtime engines."""

from __future__ import annotations

from themis.contracts.protocols import InferenceResult
from themis.records import InferenceRecord

from .._coercion import _context_item_id, _expected_demo_response
from .._openai import _run_openai_chat_inference


class DemoEngine:
    """Offline engine that echoes the expected answer for smoke tests."""

    def infer(self, trial, context, runtime):
        del trial, runtime
        raw_text = _expected_demo_response(context)
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inference_{_context_item_id(context)}",
                raw_text=raw_text,
            )
        )


class OpenAIChatEngine:
    """Minimal OpenAI chat-completions adapter for catalog workflows."""

    def infer(self, trial, context, runtime):
        extras = dict(trial.model.extras)
        raw_base_url = extras.get("base_url")
        return _run_openai_chat_inference(
            trial,
            context,
            runtime,
            base_url=(
                str(raw_base_url).rstrip("/") if isinstance(raw_base_url, str) else None
            ),
            provider_label="OpenAI",
            missing_extra="providers-openai",
        )
