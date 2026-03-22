"""Catalog runtime engines."""

from __future__ import annotations

from themis.contracts.protocols import InferenceResult
from themis.records import InferenceRecord

from ..common import (
    _context_item_id,
    _expected_demo_response,
    _run_openai_chat_inference,
)


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
        return _run_openai_chat_inference(
            trial,
            context,
            runtime,
            base_url=None,
            provider_label="OpenAI",
            missing_extra="providers-openai",
        )


class OpenAICompatibleChatEngine:
    """Minimal OpenAI-compatible chat-completions adapter for catalog workflows."""

    def infer(self, trial, context, runtime):
        extras = dict(trial.model.extras)
        base_url = str(extras.get("base_url", "http://127.0.0.1:8000/v1")).rstrip("/")
        return _run_openai_chat_inference(
            trial,
            context,
            runtime,
            base_url=base_url,
            provider_label="OpenAI-compatible endpoint",
            missing_extra="providers-openai",
        )
