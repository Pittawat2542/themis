from dataclasses import dataclass
import threading
import time

import pytest

from themis.core import entities as core_entities
from themis.generation import runner as generation_runner


@dataclass
class FakeModelProvider:
    latency_ms: int = 10

    def __post_init__(self) -> None:
        self.calls: list[core_entities.GenerationTask] = []

    def execute(
        self, task: core_entities.GenerationTask
    ) -> core_entities.GenerationRecord:
        self.calls.append(task)
        payload = core_entities.ModelOutput(
            text=f"{task.prompt.text} :: {task.model.identifier}"
        )
        return core_entities.GenerationRecord(
            task=task,
            output=payload,
            error=None,
            metrics={"latency_ms": self.latency_ms},
        )


def build_task(
    prompt: str, model: str, sampling: core_entities.SamplingConfig
) -> core_entities.GenerationTask:
    prompt_spec = core_entities.PromptSpec(name="tmp", template="{prompt}")
    prompt_render = core_entities.PromptRender(
        spec=prompt_spec,
        text=prompt,
        context={"prompt": prompt},
        metadata={"template_style": "tmp"},
    )
    model_spec = core_entities.ModelSpec(identifier=model, provider="test")
    return core_entities.GenerationTask(
        prompt=prompt_render,
        model=model_spec,
        sampling=sampling,
        metadata={"template_style": "tmp"},
    )


def test_runner_invokes_model_client_and_returns_structured_results():
    sampling = core_entities.SamplingConfig(temperature=0.2, top_p=0.9, max_tokens=32)
    requests = [
        build_task("Explain Monte Carlo", "gpt-4o", sampling),
        build_task("Explain rejection sampling", "gpt-4o", sampling),
    ]

    provider = FakeModelProvider()
    runner = generation_runner.GenerationRunner(executor=provider)

    results = list(runner.run(requests))

    assert len(results) == 2
    assert results[0].task.prompt.text == "Explain Monte Carlo"
    assert results[0].output.text == "Explain Monte Carlo :: gpt-4o"
    assert results[0].task.metadata["template_style"] == "tmp"
    assert provider.calls[0].sampling.temperature == pytest.approx(0.2)
    assert results[0].metrics["latency_ms"] == 10
    assert "generation_time_ms" in results[0].metrics
    assert "prompt_chars" in results[0].metrics
    assert "response_chars" in results[0].metrics


def test_runner_attaches_failures_to_result_stream():
    class FlakyProvider(FakeModelProvider):
        def execute(self, task: core_entities.GenerationTask):  # type: ignore[override]
            if "fail" in task.prompt.text:
                raise RuntimeError("boom")
            return super().execute(task)

    sampling = core_entities.SamplingConfig(temperature=1.0, top_p=0.9, max_tokens=16)
    requests = [
        build_task("Explain stable diffusion", "gpt-4o", sampling),
        build_task("fail on purpose", "gpt-4o", sampling),
    ]

    runner = generation_runner.GenerationRunner(
        executor=FlakyProvider(), retry_initial_delay=0.0
    )

    results = list(runner.run(requests))

    assert results[1].error is not None
    assert "boom" in results[1].error.message
    assert results[0].error is None


def test_runner_retries_transient_failures_and_records_attempts():
    class TransientProvider(FakeModelProvider):
        def __post_init__(self):  # type: ignore[override]
            super().__post_init__()
            self.failures = 0

        def execute(self, task: core_entities.GenerationTask):  # type: ignore[override]
            if self.failures < 2:
                self.failures += 1
                raise RuntimeError("temporary outage")
            return super().execute(task)

    sampling = core_entities.SamplingConfig(temperature=0.1, top_p=0.9, max_tokens=16)
    requests = [build_task("Recover please", "gpt-4o", sampling)]

    runner = generation_runner.GenerationRunner(
        executor=TransientProvider(), retry_initial_delay=0.0, max_retries=3
    )

    results = list(runner.run(requests))

    assert results[0].error is None
    assert results[0].metrics["generation_attempts"] == 3
    retry_errors = results[0].metrics["retry_errors"]
    assert isinstance(retry_errors, list)
    assert len(retry_errors) == 2
    assert all("temporary outage" in entry["error"] for entry in retry_errors)


def test_runner_stops_after_max_retries_and_reports_cause():
    class AlwaysFailProvider(FakeModelProvider):
        def execute(self, task: core_entities.GenerationTask):  # type: ignore[override]
            raise RuntimeError("permanent failure")

    sampling = core_entities.SamplingConfig(temperature=0.1, top_p=0.9, max_tokens=16)
    requests = [build_task("Never works", "gpt-4o", sampling)]

    runner = generation_runner.GenerationRunner(
        executor=AlwaysFailProvider(), max_retries=2, retry_initial_delay=0.0
    )

    results = list(runner.run(requests))

    assert results[0].error is not None
    assert "permanent failure" in results[0].error.message
    assert results[0].metrics["generation_attempts"] == 2
    assert len(results[0].metrics["retry_errors"]) == 2


def test_runner_parallel_execution():
    class SlowProvider(FakeModelProvider):
        def __post_init__(self) -> None:  # type: ignore[override]
            super().__post_init__()
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def execute(self, task: core_entities.GenerationTask):  # type: ignore[override]
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.05)
            record = super().execute(task)
            with self.lock:
                self.active -= 1
            return record

    provider = SlowProvider()
    sampling = core_entities.SamplingConfig(temperature=0.2, top_p=0.9, max_tokens=32)
    requests = [build_task(f"Task {i}", "gpt-4o", sampling) for i in range(4)]

    runner = generation_runner.GenerationRunner(executor=provider, max_parallel=3)
    list(runner.run(requests))

    assert provider.max_active >= 2
    assert provider.max_active <= 3


def test_runner_parallel_yields_completion_order():
    class VariableLatencyProvider(FakeModelProvider):
        def execute(self, task: core_entities.GenerationTask):  # type: ignore[override]
            if task.prompt.text == "slow":
                time.sleep(0.08)
            else:
                time.sleep(0.01)
            return super().execute(task)

    provider = VariableLatencyProvider()
    sampling = core_entities.SamplingConfig(temperature=0.2, top_p=0.9, max_tokens=32)
    requests = [
        build_task("slow", "gpt-4o", sampling),
        build_task("fast", "gpt-4o", sampling),
    ]

    runner = generation_runner.GenerationRunner(executor=provider, max_parallel=2)
    results = list(runner.run(requests))

    assert [record.task.prompt.text for record in results] == ["fast", "slow"]


def test_runner_streams_iterables_without_full_materialization():
    provider = FakeModelProvider()
    sampling = core_entities.SamplingConfig(temperature=0.2, top_p=0.9, max_tokens=32)

    class TrackingTasks:
        def __init__(self, total: int):
            self.total = total
            self.requested = 0

        def __iter__(self):
            for idx in range(self.total):
                self.requested += 1
                yield build_task(f"Task {idx}", "gpt-4o", sampling)

    tasks = TrackingTasks(total=20)
    runner = generation_runner.GenerationRunner(executor=provider, max_parallel=1)
    stream = runner.run(tasks)

    first = next(stream)
    assert first.task.prompt.text == "Task 0"
    assert tasks.requested == 1

    remaining = list(stream)
    assert len(remaining) == 19


def test_runner_bounds_in_flight_task_submission():
    class SlowProvider(FakeModelProvider):
        def execute(self, task: core_entities.GenerationTask):  # type: ignore[override]
            time.sleep(0.03)
            return super().execute(task)

    provider = SlowProvider()
    sampling = core_entities.SamplingConfig(temperature=0.2, top_p=0.9, max_tokens=32)

    class TrackingTasks:
        def __init__(self, total: int):
            self.total = total
            self.requested = 0

        def __iter__(self):
            for idx in range(self.total):
                self.requested += 1
                yield build_task(f"Task {idx}", "gpt-4o", sampling)

    tasks = TrackingTasks(total=10)
    runner = generation_runner.GenerationRunner(
        executor=provider,
        max_parallel=2,
        max_in_flight_tasks=3,
    )
    stream = runner.run(tasks)

    first = next(stream)
    assert first.output is not None
    assert tasks.requested <= 3
    assert len([first, *list(stream)]) == 10


def test_runner_retries_when_provider_returns_retryable_error_record():
    class RetryableErrorProvider:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, task: core_entities.GenerationTask):
            self.calls += 1
            if self.calls < 3:
                return core_entities.GenerationRecord(
                    task=task,
                    output=None,
                    error=core_entities.ModelError(
                        message="temporary outage",
                        kind="ProviderError",
                    ),
                    metrics={},
                )
            return core_entities.GenerationRecord(
                task=task,
                output=core_entities.ModelOutput(text="ok"),
                error=None,
                metrics={},
            )

    sampling = core_entities.SamplingConfig(temperature=0.1, top_p=0.9, max_tokens=16)
    requests = [build_task("Recover from error-record", "gpt-4o", sampling)]
    provider = RetryableErrorProvider()
    runner = generation_runner.GenerationRunner(
        executor=provider, max_retries=3, retry_initial_delay=0.0
    )

    result = list(runner.run(requests))[0]
    assert result.error is None
    assert result.metrics["generation_attempts"] == 3
    assert len(result.metrics["retry_errors"]) == 2


def test_runner_does_not_retry_non_retryable_error_record():
    class NonRetryableErrorProvider:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, task: core_entities.GenerationTask):
            self.calls += 1
            return core_entities.GenerationRecord(
                task=task,
                output=None,
                error=core_entities.ModelError(
                    message="authentication failed",
                    kind="AuthenticationError",
                ),
                metrics={},
            )

    sampling = core_entities.SamplingConfig(temperature=0.1, top_p=0.9, max_tokens=16)
    requests = [build_task("No retry expected", "gpt-4o", sampling)]
    provider = NonRetryableErrorProvider()
    runner = generation_runner.GenerationRunner(
        executor=provider, max_retries=3, retry_initial_delay=0.0
    )

    result = list(runner.run(requests))[0]
    assert result.error is not None
    assert provider.calls == 1
    assert result.metrics["generation_attempts"] == 1
