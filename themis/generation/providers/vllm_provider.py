"""vLLM provider using AsyncLLMEngine."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Dict, List

from themis.core import entities as core_entities
from themis.interfaces import ModelProvider
from themis.providers import register_provider


class VLLMProvider(ModelProvider):
    def __init__(
        self,
        *,
        model: str,
        tensor_parallel_size: int = 1,
        max_parallel: int = 2,
        engine_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        self._model_name = model
        self._tp_size = max(1, tensor_parallel_size)
        self._max_parallel = max(1, max_parallel)
        self._engine_kwargs = engine_kwargs or {}
        self._engines = self._create_engines()
        self._engine_lock = threading.Lock()
        self._rr_index = 0
        self._semaphore = threading.Semaphore(self._max_parallel)

    def generate(
        self, task: core_entities.GenerationTask
    ) -> core_entities.GenerationRecord:  # type: ignore[override]
        with self._semaphore:
            engine = self._select_engine()
            text, raw = asyncio.run(self._run_generation(engine, task))
        metrics = {k: v for k, v in raw.items() if k != "chunks"}
        return core_entities.GenerationRecord(
            task=task,
            output=core_entities.ModelOutput(text=text, raw=raw),
            error=None,
            metrics=metrics,
        )

    async def _run_generation(self, engine, task: core_entities.GenerationTask):
        SamplingParams = self._sampling_params_cls
        sampling_params = SamplingParams(
            temperature=task.sampling.temperature,
            top_p=task.sampling.top_p,
            max_tokens=None
            if task.sampling.max_tokens < 0
            else task.sampling.max_tokens,
        )
        dataset_id = task.metadata.get("dataset_id", "sample")
        request_id = f"themis-{dataset_id}-{time.time_ns()}"
        chunks: List[str] = []
        tokenizer = getattr(engine, "tokenizer", None)
        async for output in engine.generate(
            prompt=task.prompt.text,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            if output.outputs:
                chunks.append(output.outputs[0].text)
        final_text = chunks[-1] if chunks else ""
        metrics = {"chunks": chunks}
        if tokenizer is not None:
            try:
                metrics["prompt_tokens"] = len(tokenizer.encode(task.prompt.text))
                metrics["response_tokens"] = len(tokenizer.encode(final_text))
            except Exception:  # pragma: no cover
                pass
        return final_text, metrics

    def _select_engine(self):
        with self._engine_lock:
            engine = self._engines[self._rr_index]
            self._rr_index = (self._rr_index + 1) % len(self._engines)
        return engine

    def _create_engines(self):
        AsyncLLMEngine, SamplingParams = self._load_vllm_classes()
        self._sampling_params_cls = SamplingParams
        engine_count = self._determine_engine_count()
        engines = []
        for idx in range(engine_count):
            engine = AsyncLLMEngine(
                model=self._model_name,
                tensor_parallel_size=self._tp_size,
                **self._engine_kwargs,
            )
            engines.append(engine)
        return engines

    def _determine_engine_count(self) -> int:
        device_count = 0
        try:
            import torch

            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
        except ImportError:
            device_count = 0
        if device_count and device_count % self._tp_size == 0:
            return max(1, device_count // self._tp_size)
        return 1

    def count_tokens(self, text: str) -> int | None:
        tokenizer = (
            getattr(self._engines[0], "tokenizer", None) if self._engines else None
        )
        if tokenizer is None:
            return None
        try:
            return len(tokenizer.encode(text))
        except Exception:
            return None

    @staticmethod
    def _load_vllm_classes():
        try:
            from vllm import AsyncLLMEngine, SamplingParams
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "vLLM is not installed. Install via `pip install vllm` to use VLLMProvider."
            ) from exc
        return AsyncLLMEngine, SamplingParams


register_provider("vllm", VLLMProvider)


__all__ = ["VLLMProvider"]
