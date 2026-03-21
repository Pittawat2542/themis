"""Explicit starter components for quick-start CLIs and generated scaffolds."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random
from time import perf_counter
from typing import Any, cast

from themis._optional import import_optional
from themis.contracts.protocols import InferenceResult
from themis.errors import InferenceError, RetryableProviderError
from themis.extractors.builtin import _normalize_text
from themis.records import InferenceRecord, MetricScore, TokenUsage
from themis.registry import EngineCapabilities, PluginRegistry
from themis.types.enums import DatasetSource, ErrorCode, ResponseFormat, SamplingKind


def build_starter_registry(provider: str) -> PluginRegistry:
    """Build a registry containing the starter metrics and one requested engine."""

    registry = PluginRegistry()
    register_starter_metrics(registry)
    register_starter_engine(registry, provider)
    return registry


def register_starter_metrics(registry: PluginRegistry) -> None:
    """Register the curated starter metric set on an existing registry."""

    registry.register_metric("exact_match", ExactMatchMetric())
    registry.register_metric("normalized_exact_match", NormalizedExactMatchMetric())
    registry.register_metric("choice_accuracy", ChoiceAccuracyMetric())
    registry.register_metric("numeric_exact_match", NumericExactMatchMetric())


def register_starter_engine(registry: PluginRegistry, provider: str) -> None:
    """Register one starter inference engine on an existing registry."""

    if provider == "demo":
        registry.register_inference_engine(
            "demo",
            DemoEngine(),
            capabilities=EngineCapabilities(supports_seed=True),
        )
        return
    if provider == "openai":
        registry.register_inference_engine(
            "openai",
            OpenAIChatEngine(),
            capabilities=EngineCapabilities(supports_seed=True),
        )
        return
    if provider == "openai_compatible":
        registry.register_inference_engine(
            "openai_compatible",
            OpenAICompatibleChatEngine(),
            capabilities=EngineCapabilities(supports_seed=True),
        )
        return
    raise ValueError(f"Unsupported quick-start provider '{provider}'.")


class DemoEngine:
    """Offline engine that echoes the expected answer for smoke tests."""

    def infer(self, trial, context, runtime):
        del trial, runtime
        raw_text = _expected_text(context)
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inference_{_context_item_id(context)}",
                raw_text=raw_text,
            )
        )


class OpenAIChatEngine:
    """Minimal OpenAI chat-completions adapter for starter workflows."""

    def infer(self, trial, context, runtime):
        return _run_openai_chat_inference(
            trial,
            runtime,
            base_url=None,
            provider_label="OpenAI",
            missing_extra="providers-openai",
        )


class OpenAICompatibleChatEngine:
    """Minimal OpenAI-compatible chat-completions adapter for starter workflows."""

    def infer(self, trial, context, runtime):
        del context
        extras = dict(trial.model.extras)
        base_url = str(extras.get("base_url", "http://127.0.0.1:8000/v1")).rstrip("/")
        return _run_openai_chat_inference(
            trial,
            runtime,
            base_url=base_url,
            provider_label="OpenAI-compatible endpoint",
            missing_extra="providers-openai",
        )


class ExactMatchMetric:
    """Exact string match against the normalized starter expected field."""

    def score(self, trial, candidate, context):
        del trial
        actual = candidate.inference.raw_text if candidate.inference is not None else ""
        return MetricScore(
            metric_id="exact_match",
            value=float(actual == _expected_text(context)),
        )


class NormalizedExactMatchMetric:
    """Case- and punctuation-insensitive exact match using parsed output."""

    def score(self, trial, candidate, context):
        del trial
        extraction = candidate.best_extraction()
        parsed = (
            extraction.parsed_answer
            if extraction is not None and extraction.success
            else ""
        )
        return MetricScore(
            metric_id="normalized_exact_match",
            value=float(str(parsed) == _normalize_text(_expected_text(context))),
        )


class ChoiceAccuracyMetric:
    """Exact multiple-choice letter match using parsed output."""

    def score(self, trial, candidate, context):
        del trial
        extraction = candidate.best_extraction()
        parsed = (
            extraction.parsed_answer
            if extraction is not None and extraction.success
            else ""
        )
        expected = _expected_text(context).strip().upper()
        return MetricScore(
            metric_id="choice_accuracy",
            value=float(str(parsed).strip().upper() == expected),
        )


class NumericExactMatchMetric:
    """Numeric exact match using the parsed first-number surface."""

    def score(self, trial, candidate, context):
        del trial
        extraction = candidate.best_extraction()
        parsed = (
            extraction.parsed_answer
            if extraction is not None and extraction.success
            else None
        )
        actual = _coerce_float(parsed)
        expected = _coerce_float(_expected_text(context))
        return MetricScore(
            metric_id="numeric_exact_match",
            value=float(
                actual is not None and expected is not None and actual == expected
            ),
        )


class StarterDatasetProvider:
    """Dataset provider covering inline, local-file, and HuggingFace starters."""

    def __init__(
        self,
        *,
        memory_rows: list[dict[str, object]] | None = None,
        huggingface_loader=None,
        local_loader=None,
    ) -> None:
        self._memory_rows = list(memory_rows or [])
        self._huggingface_loader = huggingface_loader or load_huggingface_rows
        self._local_loader = local_loader or load_local_rows

    def scan(self, slice_spec, query):
        dataset = slice_spec.dataset
        if dataset.source == DatasetSource.MEMORY:
            rows = list(self._memory_rows)
        elif dataset.source == DatasetSource.LOCAL:
            dataset_path = dataset.dataset_id or dataset.data_dir
            if dataset_path is None:
                raise ValueError("Local starter datasets require a dataset path.")
            rows = self._local_loader(Path(dataset_path))
        elif dataset.source == DatasetSource.HUGGINGFACE:
            if dataset.dataset_id is None:
                raise ValueError("HuggingFace starter datasets require a dataset_id.")
            rows = self._huggingface_loader(
                dataset.dataset_id,
                dataset.split,
                dataset.revision,
            )
        else:
            raise ValueError(f"Unsupported starter dataset source '{dataset.source}'.")
        return _apply_query(rows, query)


def load_local_rows(path: Path) -> list[dict[str, object]]:
    """Load starter dataset rows from JSONL or CSV."""

    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, object]] = []
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"JSONL row {line_number} in {path.name} must be an object."
                )
            rows.append(dict(payload))
        return _assign_missing_item_ids(rows)
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as fh:
            return _assign_missing_item_ids([dict(row) for row in csv.DictReader(fh)])
    raise ValueError("Starter local datasets must use .jsonl or .csv.")


def load_huggingface_rows(
    dataset_id: str,
    split: str,
    revision: str | None = None,
) -> list[dict[str, object]]:
    """Load starter dataset rows from a HuggingFace dataset identifier."""

    datasets = import_optional("datasets", extra="datasets")
    dataset = datasets.load_dataset(dataset_id, split=split, revision=revision)
    return _assign_missing_item_ids([dict(row) for row in dataset])


def _run_openai_chat_inference(
    trial,
    runtime,
    *,
    base_url: str | None,
    provider_label: str,
    missing_extra: str,
) -> InferenceResult:
    if trial.params.response_format not in (None, ResponseFormat.TEXT):
        raise InferenceError(
            code=ErrorCode.PLUGIN_INCOMPATIBLE,
            message=f"{provider_label} starter engine currently supports text responses only.",
            details={"response_format": str(trial.params.response_format)},
        )
    openai = import_optional("openai", extra=missing_extra)
    extras = dict(trial.model.extras)
    timeout_seconds = float(extras.get("timeout_seconds", 60.0))
    client_kwargs: dict[str, object] = {"timeout": timeout_seconds}
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    api_key = (
        _runtime_secret(runtime, "OPENAI_API_KEY")
        or _runtime_secret(runtime, "OPENAI_COMPAT_API_KEY")
        or extras.get("api_key")
        or "dummy"
    )
    client_kwargs["api_key"] = str(api_key)
    client = openai.OpenAI(**client_kwargs)
    messages = [message.model_dump(mode="json") for message in trial.prompt.messages]
    request_kwargs: dict[str, Any] = {
        "model": trial.model.model_id,
        "messages": messages,
        "temperature": trial.params.temperature,
        "max_tokens": trial.params.max_tokens,
    }
    if trial.params.top_p is not None:
        request_kwargs["top_p"] = trial.params.top_p
    if trial.params.stop_sequences:
        request_kwargs["stop"] = trial.params.stop_sequences
    if trial.params.seed is not None:
        request_kwargs["seed"] = trial.params.seed & 0xFFFFFFFF
    if trial.params.logprobs is not None:
        request_kwargs["logprobs"] = True
        request_kwargs["top_logprobs"] = trial.params.logprobs
    if trial.params.top_k is not None:
        extra_body = cast(
            dict[str, object], request_kwargs.setdefault("extra_body", {})
        )
        extra_body["top_k"] = trial.params.top_k
    for key, value in trial.params.extras.items():
        extra_body = cast(
            dict[str, object], request_kwargs.setdefault("extra_body", {})
        )
        extra_body[key] = value

    start = perf_counter()
    try:
        response = client.chat.completions.create(**request_kwargs)
    except openai.AuthenticationError as exc:
        raise InferenceError(
            code=ErrorCode.PROVIDER_AUTH,
            message=f"{provider_label} rejected authentication.",
            details={"body": getattr(exc, "body", None)},
        ) from exc
    except openai.RateLimitError as exc:
        raise RetryableProviderError(
            code=ErrorCode.PROVIDER_RATE_LIMIT,
            message=f"{provider_label} rate limited the request.",
            details={"body": getattr(exc, "body", None)},
        ) from exc
    except openai.APIConnectionError as exc:
        raise RetryableProviderError(
            code=ErrorCode.PROVIDER_UNAVAILABLE,
            message=f"Could not reach {provider_label}: {exc}",
        ) from exc
    except openai.APIStatusError as exc:
        error_cls = RetryableProviderError if exc.status_code >= 500 else InferenceError
        raise error_cls(
            code=ErrorCode.PROVIDER_UNAVAILABLE,
            message=f"{provider_label} returned HTTP {exc.status_code}.",
            details={"body": getattr(exc, "body", None)},
        ) from exc

    choice = response.choices[0] if getattr(response, "choices", None) else None
    if choice is None or getattr(choice, "message", None) is None:
        raise InferenceError(
            code=ErrorCode.PARSE_ERROR,
            message=f"{provider_label} returned no message choices.",
            details={"provider_request_id": getattr(response, "id", None)},
        )
    latency_ms = (perf_counter() - start) * 1000
    usage = getattr(response, "usage", None)
    return InferenceResult(
        inference=InferenceRecord(
            spec_hash=f"inference_{trial.item_id}",
            raw_text=_coerce_message_text(choice.message.content),
            finish_reason=getattr(choice, "finish_reason", None),
            latency_ms=latency_ms,
            provider_request_id=getattr(response, "id", None),
            token_usage=TokenUsage(
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            ),
        )
    )


def _apply_query(rows: list[dict[str, object]], query) -> list[dict[str, object]]:
    filtered = list(rows)
    if query.metadata_filters:
        filtered = [
            row
            for row in filtered
            if all(
                str(row.get(key, "")) == value
                for key, value in query.metadata_filters.items()
            )
        ]
    if query.item_ids:
        wanted = set(query.item_ids)
        filtered = [row for row in filtered if str(row.get("item_id")) in wanted]
    if query.kind == SamplingKind.ALL:
        return filtered
    count = query.count or 0
    if query.kind == SamplingKind.SUBSET:
        if query.seed is None:
            return filtered[:count]
        if count >= len(filtered):
            return filtered
        return random.Random(query.seed).sample(filtered, count)
    if query.kind == SamplingKind.STRATIFIED:
        field = query.strata_field
        if not field:
            return filtered
        buckets: dict[str, list[dict[str, object]]] = {}
        for row in filtered:
            buckets.setdefault(str(row.get(field, "")), []).append(row)
        randomizer = random.Random(query.seed)
        samples: list[dict[str, object]] = []
        for bucket_rows in buckets.values():
            if len(bucket_rows) <= count:
                samples.extend(bucket_rows)
            else:
                samples.extend(randomizer.sample(bucket_rows, count))
        return samples
    return filtered


def _assign_missing_item_ids(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        payload = dict(row)
        payload.setdefault("item_id", f"item-{index}")
        normalized.append(payload)
    return normalized


def _context_item_id(context: object) -> str:
    item_id = getattr(context, "item_id", None)
    if item_id is not None:
        return str(item_id)
    if hasattr(context, "get"):
        resolved = context.get("item_id")  # type: ignore[attr-defined]
        if resolved is not None:
            return str(resolved)
    return "item"


def _expected_text(context: object) -> str:
    if hasattr(context, "get"):
        resolved = context.get("expected")  # type: ignore[attr-defined]
        if resolved is not None:
            return _coerce_text(resolved)
        fallback = context.get("answer")  # type: ignore[attr-defined]
        if fallback is not None:
            return _coerce_text(fallback)
    return ""


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (str, int, float)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def _coerce_message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "".join(text_parts)
    return ""


def _runtime_secret(runtime, key: str) -> str | None:
    secrets = getattr(runtime, "secrets", {}) or {}
    value = secrets.get(key)
    if value is None:
        return None
    if hasattr(value, "get_secret_value"):
        return value.get_secret_value()
    return str(value)
