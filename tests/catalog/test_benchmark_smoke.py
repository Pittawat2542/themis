from __future__ import annotations

import pytest

from themis.catalog import load
from themis.core.builtins import resolve_generator_component
from themis.core.contexts import GenerateContext
from themis.core.models import GenerationResult


BENCHMARK_IDS = [
    "aime_2025",
    "aime_2026",
    "aethercode",
    "apex_2025",
    "babe",
    "beyond_aime",
    "encyclo_k",
    "frontierscience",
    "gpqa_diamond",
    "healthbench",
    "hle:math,reasoning",
    "hmmt_feb_2025",
    "hmmt_nov_2025",
    "humaneval:mini",
    "humaneval_plus:noextreme",
    "imo_answerbench",
    "livecodebench",
    "lpfqa",
    "mmlu_pro",
    "mmmlu:thai",
    "codeforces",
    "phybench",
    "procbench:task07",
    "rolebench:role_generalization_eng",
    "simpleqa_verified",
    "superchem:en",
    "supergpqa",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark_id", BENCHMARK_IDS)
async def test_benchmark_catalog_smoke_loads_experiment_and_invokes_generator(benchmark_id: str) -> None:
    benchmark = load(benchmark_id)
    experiment = benchmark.build_experiment()
    dataset = experiment.datasets[0]
    case = dataset.cases[0]
    generator = resolve_generator_component(experiment.generation.generator)

    result = await generator.generate(
        case,
        GenerateContext(
            run_id="smoke-run",
            case_id=case.case_id,
            seed=experiment.seeds[0] if experiment.seeds else 7,
        ),
    )

    assert case.input is not None
    assert isinstance(result, GenerationResult)
    assert result.candidate_id.startswith(case.case_id)
    if isinstance(case.input, dict):
        assert case.input
