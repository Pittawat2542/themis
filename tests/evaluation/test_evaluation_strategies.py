from themis.core import entities as core_entities
from themis.evaluation import strategies


def make_record(text="answer"):
    prompt_spec = core_entities.PromptSpec(name="t", template="{problem}")
    prompt = core_entities.PromptRender(
        spec=prompt_spec, text="Problem", context={}, metadata={}
    )
    model = core_entities.ModelSpec(identifier="fake", provider="fake")
    sampling = core_entities.SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=16)
    task = core_entities.GenerationTask(prompt=prompt, model=model, sampling=sampling)
    record = core_entities.GenerationRecord(
        task=task,
        output=core_entities.ModelOutput(text=text),
        error=None,
        metrics={},
    )
    return record


def test_attempt_aware_strategy_average_scores():
    strategy = strategies.AttemptAwareEvaluationStrategy(average_attempts=True)
    parent = make_record()
    attempt1 = make_record(text="a")
    attempt2 = make_record(text="b")
    parent.attempts = [attempt1, attempt2]

    items = list(strategy.prepare(parent))
    assert len(items) == 2

    score1 = core_entities.MetricScore(metric_name="ExactMatch", value=1.0, metadata={})
    score2 = core_entities.MetricScore(metric_name="ExactMatch", value=0.0, metadata={})

    aggregated = strategy.aggregate(parent, [score1, score2])
    assert aggregated[0].value == 0.5
