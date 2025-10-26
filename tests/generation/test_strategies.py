from themis.core import entities as core_entities
from themis.generation import strategies


def make_task():
    prompt_spec = core_entities.PromptSpec(name="t", template="{problem}")
    prompt = core_entities.PromptRender(
        spec=prompt_spec, text="Solve", context={"problem": "x"}, metadata={}
    )
    model = core_entities.ModelSpec(identifier="fake", provider="fake")
    sampling = core_entities.SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=16)
    return core_entities.GenerationTask(
        prompt=prompt, model=model, sampling=sampling, metadata={"subject": "demo"}
    )


def make_record(task, text="result"):
    return core_entities.GenerationRecord(
        task=task,
        output=core_entities.ModelOutput(text=text),
        error=None,
        metrics={"latency_ms": 1},
    )


def test_repeated_sampling_strategy_expands_and_aggregates():
    task = make_task()
    strategy = strategies.RepeatedSamplingStrategy(attempts=3, metadata_label="shot")

    expanded = list(strategy.expand(task))
    assert len(expanded) == 3
    assert expanded[0].metadata["shot"] == 0
    assert expanded[2].metadata["shot"] == 2

    records = [
        make_record(expanded[0], text="first"),
        make_record(expanded[1], text="better"),
        make_record(expanded[2]),
    ]
    aggregated = strategy.aggregate(task, records)

    assert aggregated.metrics["attempt_count"] == 3
    assert aggregated.output.text == "first"
    assert (
        len(aggregated.attempts) == 0
    )  # aggregation populates only metrics; runner attaches attempts


def test_single_attempt_strategy_passthrough():
    task = make_task()
    strategy = strategies.SingleAttemptStrategy()
    [expanded_task] = list(strategy.expand(task))
    record = make_record(expanded_task)

    aggregated = strategy.aggregate(task, [record])
    assert aggregated.output.text == "result"
