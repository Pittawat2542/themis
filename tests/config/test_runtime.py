import json
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


def _default_generation_config() -> schema.GenerationConfig:
    return schema.GenerationConfig(
        model_identifier="fake-math-llm",
        provider=schema.ProviderConfig(name="fake", options={"seed": 7}),
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
    )


def test_run_supergpqa_config_with_local_dataset(tmp_path):
    data_dir = tmp_path / "supergpqa"
    data_dir.mkdir()
    sample_path = data_dir / "row.json"
    sample_payload = {
        "question": "What is 1 + 3?",
        "choices": ["3", "4", "5", "6"],
        "answer": 1,
        "subject": "math",
    }
    sample_path.write_text(json.dumps(sample_payload), encoding="utf-8")

    config = schema.ExperimentConfig(
        name="supergpqa_zero_shot",
        dataset=schema.DatasetConfig(
            source="local",
            data_dir=str(data_dir),
            split="test",
        ),
        generation=_default_generation_config(),
    )

    dataset = runtime.load_dataset_from_config(config)
    report = runtime.run_experiment_from_config(config, dataset=dataset)

    assert report.metadata["total_samples"] == 1
    summary = runtime.summarize_report_for_config(config, report)
    assert "Accuracy" in summary


def test_run_mmlu_pro_config_with_local_dataset(tmp_path):
    data_dir = tmp_path / "mmlu-pro"
    data_dir.mkdir()
    sample_path = data_dir / "row.json"
    sample_payload = {
        "question": "Capital of Spain?",
        "choices": ["Barcelona", "Madrid", "Seville", "Valencia"],
        "answer": "B",
        "subject": "geography",
    }
    sample_path.write_text(json.dumps(sample_payload), encoding="utf-8")

    config = schema.ExperimentConfig(
        name="mmlu_pro_zero_shot",
        dataset=schema.DatasetConfig(
            source="local",
            data_dir=str(data_dir),
            split="test",
        ),
        generation=_default_generation_config(),
    )

    dataset = runtime.load_dataset_from_config(config)
    report = runtime.run_experiment_from_config(config, dataset=dataset)

    assert report.metadata["total_samples"] == 1
    summary = runtime.summarize_report_for_config(config, report)
    assert "Accuracy" in summary


@pytest.mark.parametrize(
    "experiment_name",
    [
        "aime24_zero_shot",
        "aime25_zero_shot",
        "amc23_zero_shot",
        "olympiadbench_zero_shot",
        "beyondaime_zero_shot",
    ],
)
def test_run_competition_math_configs(tmp_path, experiment_name):
    data_dir = tmp_path / experiment_name
    data_dir.mkdir()
    sample_path = data_dir / "row.json"
    sample_payload = {
        "problem": "Compute 1 + 2",
        "solution": "Add numbers",
        "answer": "3",
        "subject": "algebra",
    }
    sample_path.write_text(json.dumps(sample_payload), encoding="utf-8")

    config = schema.ExperimentConfig(
        name=experiment_name,
        dataset=schema.DatasetConfig(
            source="local",
            data_dir=str(data_dir),
            split="test",
        ),
        generation=_default_generation_config(),
    )

    dataset = runtime.load_dataset_from_config(config)
    report = runtime.run_experiment_from_config(config, dataset=dataset)

    assert report.metadata["total_samples"] == 1
    summary = runtime.summarize_report_for_config(config, report)
    assert "Exact match" in summary
