"""Shared runtime mechanics for orchestration internals."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
from collections.abc import Callable, Mapping, Sequence
from time import monotonic
from uuid import uuid4

from themis.core.base import JSONValue
from themis.core.config import RuntimeConfig, Stage
from themis.core.events import FailureEvidence, RunEvent
from themis.core.evidence import EvidenceWriter
from themis.core.protocols import EventSubscriber
from themis.core.store import AppendResult, RunStore
from themis.core.stores.memory import InMemoryRunStore
from themis.core.security import EvidenceSanitizer

ConnectionLikeErrors = (ConnectionError, OSError)
_RETRY_JITTER_RNG = random.Random()
_LOGGER = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """Token-bucket limiter with a single-token bucket to smooth request rate."""

    def __init__(
        self,
        requests_per_minute: int,
        *,
        monotonic_clock: Callable[[], float] = monotonic,
        initial_tokens: float = 1.0,
    ) -> None:
        self._requests_per_minute = max(1, requests_per_minute)
        self._tokens = initial_tokens
        self._monotonic = monotonic_clock
        self._updated_at = self._monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        for _ in range(max(0, tokens)):
            while True:
                async with self._lock:
                    now = self._monotonic()
                    self._refill(now)
                    if self._tokens >= 1.0:
                        self._tokens -= 1.0
                        break
                    wait_time = (1.0 - self._tokens) / self._tokens_per_second
                await asyncio.sleep(wait_time)

    async def update_limit(self, requests_per_minute: int) -> None:
        async with self._lock:
            now = self._monotonic()
            self._refill(now)
            self._requests_per_minute = max(1, requests_per_minute)
            self._tokens = min(self._tokens, 1.0)

    @property
    def _tokens_per_second(self) -> float:
        return self._requests_per_minute / 60.0

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self._updated_at)
        self._tokens = min(1.0, self._tokens + elapsed * self._tokens_per_second)
        self._updated_at = now


class RuntimeSupport:
    """Runtime primitives shared by the run coordinator and case pipeline."""

    def __init__(
        self,
        *,
        store: RunStore,
        runtime: RuntimeConfig,
        subscribers: Sequence[EventSubscriber],
        monotonic_clock: Callable[[], float] = monotonic,
    ) -> None:
        self.store = store
        self.runtime = runtime
        self.subscribers = list(subscribers)
        self._monotonic = monotonic_clock
        self.global_semaphore = asyncio.Semaphore(runtime.max_concurrent_tasks)
        self.stage_semaphores = {
            Stage.GENERATE: asyncio.Semaphore(
                max(
                    1,
                    runtime.stage_concurrency.get(
                        Stage.GENERATE, runtime.max_concurrent_tasks
                    ),
                )
            ),
            Stage.JUDGE: asyncio.Semaphore(
                max(
                    1,
                    runtime.stage_concurrency.get(
                        Stage.JUDGE, runtime.max_concurrent_tasks
                    ),
                )
            ),
            Stage.SELECT: asyncio.Semaphore(
                max(
                    1,
                    runtime.stage_concurrency.get(
                        Stage.SELECT, runtime.max_concurrent_tasks
                    ),
                )
            ),
            Stage.REDUCE: asyncio.Semaphore(
                max(
                    1,
                    runtime.stage_concurrency.get(
                        Stage.REDUCE, runtime.max_concurrent_tasks
                    ),
                )
            ),
            Stage.PARSE: asyncio.Semaphore(
                max(
                    1,
                    runtime.stage_concurrency.get(
                        Stage.PARSE, runtime.max_concurrent_tasks
                    ),
                )
            ),
            Stage.SCORE: asyncio.Semaphore(
                max(
                    1,
                    runtime.stage_concurrency.get(
                        Stage.SCORE, runtime.max_concurrent_tasks
                    ),
                )
            ),
        }
        self._provider_semaphores: dict[str, asyncio.Semaphore] = {}
        self._provider_limiters: dict[str, TokenBucketRateLimiter] = {}
        self._provider_token_limiters: dict[str, TokenBucketRateLimiter] = {}
        self.evidence_sanitizer = EvidenceSanitizer(runtime.evidence_retention)
        self.attempt_id = str(uuid4())
        self.evidence_writer = EvidenceWriter(runtime.evidence_queue_capacity)

    def provider_semaphore(self, provider_key: str) -> asyncio.Semaphore:
        if provider_key not in self._provider_semaphores:
            limit = max(
                1,
                self.runtime.provider_concurrency.get(
                    provider_key, self.runtime.max_concurrent_tasks
                ),
            )
            self._provider_semaphores[provider_key] = asyncio.Semaphore(limit)
        return self._provider_semaphores[provider_key]

    def provider_limiter(self, provider_key: str) -> TokenBucketRateLimiter | None:
        existing = self._provider_limiters.get(provider_key)
        if existing is not None:
            return existing
        requests_per_minute = self.runtime.provider_rate_limits.get(provider_key)
        if requests_per_minute is None:
            return None
        self._provider_limiters[provider_key] = TokenBucketRateLimiter(
            requests_per_minute,
            monotonic_clock=self._monotonic,
        )
        return self._provider_limiters[provider_key]

    def provider_token_limiter(
        self, provider_key: str
    ) -> TokenBucketRateLimiter | None:
        tokens_per_minute = self.runtime.provider_token_limits.get(provider_key)
        if tokens_per_minute is None:
            return None
        if provider_key not in self._provider_token_limiters:
            self._provider_token_limiters[provider_key] = TokenBucketRateLimiter(
                tokens_per_minute,
                monotonic_clock=self._monotonic,
            )
        return self._provider_token_limiters[provider_key]

    async def update_rate_limit(
        self,
        provider_key: str | None,
        artifacts: Mapping[str, object] | None,
    ) -> None:
        if provider_key is None or not artifacts:
            return
        rate_limit = artifacts.get("rate_limit")
        if not isinstance(rate_limit, dict):
            return
        requests_per_minute = rate_limit.get("requests_per_minute")
        if isinstance(requests_per_minute, int):
            limiter = self._provider_limiters.get(provider_key)
            if limiter is None:
                self._provider_limiters[provider_key] = TokenBucketRateLimiter(
                    requests_per_minute,
                    monotonic_clock=self._monotonic,
                    initial_tokens=0.0,
                )
            else:
                await limiter.update_limit(requests_per_minute)

    async def persist_event(self, event: RunEvent) -> None:
        if not event.attempt_id:
            updates: dict[str, object] = {"attempt_id": self.attempt_id}
            failure = getattr(event, "failure", None)
            if isinstance(failure, FailureEvidence) and not failure.attempt_id:
                updates["failure"] = failure.model_copy(
                    update={"attempt_id": self.attempt_id}
                )
            event = event.model_copy(update=updates)
        event = self.evidence_sanitizer.event(event)
        result: AppendResult | None = None
        for attempt in range(self.runtime.store_retry_attempts):
            try:
                result = await self.evidence_writer.call(
                    self.store.persist_event,
                    event,
                    timeout=self.runtime.persistence_timeout_seconds,
                )
                break
            except Exception as exc:
                if (
                    classify_retryable_error(exc) is None
                    or attempt + 1 == self.runtime.store_retry_attempts
                ):
                    raise
                await asyncio.sleep(self.runtime.store_retry_delay)
        if isinstance(result, AppendResult) and not result.inserted:
            return
        for subscriber in self.subscribers:
            try:
                await self.evidence_writer.call(
                    subscriber.on_event,
                    event,
                    timeout=self.runtime.subscriber_timeout_seconds,
                )
            except Exception as exc:
                _LOGGER.warning(
                    "event subscriber failed",
                    extra={
                        "run_id": event.run_id,
                        "attempt_id": event.attempt_id,
                        "event_id": event.event_id,
                        "exception_class": type(exc).__qualname__,
                    },
                )

    async def store_blob(self, blob: bytes, media_type: str) -> str:
        blob = self.evidence_sanitizer.blob(blob, media_type)
        for attempt in range(self.runtime.store_retry_attempts):
            try:
                return await self.evidence_writer.call(
                    self.store.store_blob,
                    blob,
                    media_type,
                    timeout=self.runtime.persistence_timeout_seconds,
                )
            except Exception as exc:
                if (
                    classify_retryable_error(exc) is None
                    or attempt + 1 == self.runtime.store_retry_attempts
                ):
                    raise
                await asyncio.sleep(self.runtime.store_retry_delay)
        raise RuntimeError("unreachable")

    async def load_stage_cache(
        self, stage_name: str, cache_key: str
    ) -> JSONValue | None:
        if isinstance(self.store, InMemoryRunStore):
            return None
        return await self.evidence_writer.call(
            self.store.load_stage_cache,
            stage_name,
            cache_key,
            timeout=self.runtime.persistence_timeout_seconds,
        )

    async def store_stage_cache(
        self, stage_name: str, cache_key: str, payload: JSONValue
    ) -> None:
        if isinstance(self.store, InMemoryRunStore):
            return
        sanitized = self.evidence_sanitizer.value(payload)
        await self.evidence_writer.call(
            self.store.store_stage_cache,
            stage_name,
            cache_key,
            sanitized,
            timeout=self.runtime.persistence_timeout_seconds,
        )

    async def aclose(self) -> None:
        await self.evidence_writer.aclose()


def classify_retryable_error(exc: Exception) -> dict[str, JSONValue] | None:
    if bool(getattr(exc, "retryable", False)):
        return {"reason": "explicit_retryable"}
    if isinstance(exc, TimeoutError | asyncio.TimeoutError):
        return {"reason": "timeout"}
    if isinstance(exc, ConnectionLikeErrors):
        return {"reason": "connection"}
    status_code = getattr(exc, "status_code", None)
    retry_after_s = getattr(exc, "retry_after_s", None)
    if status_code == 429:
        payload: dict[str, JSONValue] = {"reason": "rate_limit"}
        if isinstance(retry_after_s, (int, float)):
            payload["retry_after_s"] = float(retry_after_s)
        return payload
    if isinstance(status_code, int) and 500 <= status_code < 600:
        payload = {"reason": "server_error"}
        if isinstance(retry_after_s, (int, float)):
            payload["retry_after_s"] = float(retry_after_s)
        return payload
    return None


def failure_evidence(
    exc: Exception,
    *,
    stage: str,
    component_id: str | None = None,
    attempt_id: str = "",
    retry_attempt: int = 1,
) -> FailureEvidence:
    """Build stable failure evidence without persisting an unsafe traceback."""

    retryable = classify_retryable_error(exc) is not None
    return FailureEvidence(
        error_code=f"{stage}_{'retryable' if retryable else 'failed'}",
        exception_class=f"{type(exc).__module__}.{type(exc).__qualname__}",
        stage=stage,
        component_id=component_id,
        retryable=retryable,
        retry_attempt=retry_attempt,
        attempt_id=attempt_id,
    )


def retry_delay_seconds(
    *,
    base_delay: float,
    backoff: float,
    attempt: int,
    retry_after_s: JSONValue | None = None,
    rng: random.Random | None = None,
) -> float:
    computed_delay = max(0.0, base_delay) * (max(1.0, backoff) ** attempt)
    jitter_window = computed_delay * 0.1
    jittered_delay = computed_delay
    if jitter_window > 0:
        jitter_rng = rng or _RETRY_JITTER_RNG
        jittered_delay = max(
            0.0,
            computed_delay + jitter_rng.uniform(-jitter_window, jitter_window),
        )
    if isinstance(retry_after_s, (int, float)):
        return max(jittered_delay, float(retry_after_s))
    return jittered_delay


def observed_token_cost(token_usage: Mapping[str, int] | None) -> int:
    if not token_usage:
        return 1
    total = token_usage.get("total_tokens")
    if isinstance(total, int) and total > 0:
        return total
    return max(
        1, sum(value for value in token_usage.values() if isinstance(value, int))
    )


def stable_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
