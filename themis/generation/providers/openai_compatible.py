"""OpenAI-compatible chat completion provider."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Dict

import httpx

from themis.core import entities as core_entities
from themis.interfaces import ModelProvider
from themis.providers import register_provider


@dataclass
class OpenAICompatibleProvider(ModelProvider):
    base_url: str
    api_key: str | None = None
    timeout: int = 30
    model_mapping: Dict[str, str] | None = None
    n_parallel: int = 4

    def __post_init__(self) -> None:
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        self._semaphore = threading.Semaphore(max(1, self.n_parallel))

    def generate(
        self, task: core_entities.GenerationTask
    ) -> core_entities.GenerationRecord:  # type: ignore[override]
        payload = self._build_payload(task)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with self._semaphore:
            response = self._client.post(
                "/chat/completions", headers=headers, json=payload
            )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return core_entities.GenerationRecord(
            task=task,
            output=core_entities.ModelOutput(text=text, raw=data),
            error=None,
            metrics={
                "completion_tokens": usage.get("completion_tokens"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "response_tokens": usage.get("completion_tokens"),
            },
        )

    def _build_payload(self, task: core_entities.GenerationTask) -> dict[str, Any]:
        mapped_model = (
            self.model_mapping.get(task.model.identifier)
            if self.model_mapping
            else task.model.identifier
        )
        messages = [
            {
                "role": "system",
                "content": task.prompt.metadata.get(
                    "system_prompt", "You are a helpful AI."
                ),
            },
            {"role": "user", "content": task.prompt.text},
        ]
        return {
            "model": mapped_model,
            "messages": messages,
            "temperature": task.sampling.temperature,
            "top_p": task.sampling.top_p,
            "max_tokens": task.sampling.max_tokens,
            "stream": False,
        }


register_provider("openai-compatible", OpenAICompatibleProvider)


__all__ = ["OpenAICompatibleProvider"]
