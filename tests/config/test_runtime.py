from unittest import mock

import pytest

from themis.config import runtime, schema


def _inline_samples():
    return [
        {
            "unique_id": "sample-1",
            "problem": "What is 1 + 1?",
            "answer": "2",
            "subject": "arithmetic",
        },
        {
            "unique_id": "sample-2",
            "problem": "What is 2 + 2?",
            "answer": "4",
            "subject": "arithmetic",
        },
    ]


def test_run_experiment_from_inline_config():
    config = schema.ExperimentConfig(
        dataset=schema.DatasetConfig(source="inline", inline_samples=_inline_samples()),
        generation=schema.GenerationConfig(
            model_identifier="fake-math-llm",
            provider=schema.ProviderConfig(name="fake", options={"seed": 1}),
            sampling=schema.SamplingConfig(
                temperature=0.0,
                top_p=0.95,
                max_tokens=256,
            ),
            runner=schema.RunnerConfig(
                max_parallel=1,
                max_retries=1,
                retry_initial_delay=0.0,
                retry_backoff_multiplier=2.0,
                retry_max_delay=1.0,
            ),
        ),
    )

    dataset = runtime.load_dataset_from_config(config)
    callback = mock.Mock()

    report = runtime.run_experiment_from_config(
        config, dataset=dataset, on_result=lambda record: callback(record)
    )

    assert report.metadata["total_samples"] == 2
    summary = runtime.summarize_report_for_config(config, report)
    assert "Evaluated 2 samples" in summary
    assert callback.call_count == 2


def test_inline_dataset_requires_rows():
    config = schema.ExperimentConfig(
        dataset=schema.DatasetConfig(source="inline", inline_samples=[])
    )

    with pytest.raises(ValueError, match="inline_samples must contain at least one row"):
        runtime.load_dataset_from_config(config)
