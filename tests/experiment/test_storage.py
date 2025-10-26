from themis.core import entities as core_entities
from themis.experiment import storage as experiment_storage


def make_record(sample_id: str, answer: str) -> core_entities.GenerationRecord:
    prompt_spec = core_entities.PromptSpec(name="math", template="Solve {problem}")
    prompt_render = core_entities.PromptRender(
        spec=prompt_spec, text="Solve 1+1", context={"problem": "1+1"}, metadata={}
    )
    sampling = core_entities.SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=32)
    model_spec = core_entities.ModelSpec(identifier="fake", provider="test")
    task = core_entities.GenerationTask(
        prompt=prompt_render,
        model=model_spec,
        sampling=sampling,
        metadata={"dataset_id": sample_id},
        reference=core_entities.Reference(kind="answer", value=answer),
    )
    return core_entities.GenerationRecord(
        task=task,
        output=core_entities.ModelOutput(text=answer),
        error=None,
        metrics={"latency_ms": 10},
    )


def test_experiment_storage_roundtrip(tmp_path):
    storage = experiment_storage.ExperimentStorage(tmp_path)
    dataset = [{"id": "1", "problem": "1+1"}]
    storage.cache_dataset("run-1", dataset)

    loaded_dataset = storage.load_dataset("run-1")
    assert loaded_dataset == dataset

    record = make_record("1", "2")
    key = experiment_storage.task_cache_key(record.task)
    storage.append_record("run-1", record, cache_key=key)

    cached = storage.load_cached_records("run-1")
    assert key in cached
    assert cached[key].output.text == "2"

    score = core_entities.MetricScore(
        metric_name="ExactMatch", value=1.0, details={}, metadata={}
    )
    evaluation = core_entities.EvaluationRecord(sample_id="1", scores=[score])
    storage.append_evaluation("run-1", record, evaluation)

    cached_eval = storage.load_cached_evaluations("run-1")
    assert key in cached_eval
    assert cached_eval[key].scores[0].value == 1.0
