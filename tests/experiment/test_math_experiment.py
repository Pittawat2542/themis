import sys

import pytest

from themis.evaluation import math_verify_utils
from themis.experiment import math as math_experiment
from themis.generation import clients


def test_math_zero_shot_experiment_runs_end_to_end():
    dataset = [
        {
            "unique_id": "math-1",
            "problem": "What is 2 + 3?",
            "answer": "5",
            "subject": "arithmetic",
            "level": 1,
        }
    ]

    experiment = math_experiment.build_math500_zero_shot_experiment(
        model_client=clients.FakeMathModelClient(seed=7)
    )

    report = experiment.run(dataset)

    assert report.metadata["total_samples"] == 1
    assert len(report.generation_results) == 1
    assert report.evaluation_report.metrics["ExactMatch"].count == 1
    if math_verify_utils.math_verify_available():
        assert report.evaluation_report.metrics["MathVerifyAccuracy"].count == 1


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="math-verify has multiprocessing issues on Windows",
)
def test_run_math500_zero_shot_helper():
    dataset = [
        {
            "unique_id": "math-1",
            "problem": "What is 1 + 1?",
            "answer": "2",
            "subject": "arithmetic",
            "level": 1,
        },
        {
            "unique_id": "math-2",
            "problem": "What is 3 + 4?",
            "answer": "7",
            "subject": "arithmetic",
            "level": 1,
        },
    ]

    report = math_experiment.run_math500_zero_shot(
        dataset,
        model_client=clients.FakeMathModelClient(seed=13),
        max_samples=2,
    )

    assert report.metadata["total_samples"] == 2
    assert report.evaluation_report.metrics["ExactMatch"].mean == 1.0
    if math_verify_utils.math_verify_available():
        assert report.evaluation_report.metrics["MathVerifyAccuracy"].mean == 1.0
