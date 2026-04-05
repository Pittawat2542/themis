from __future__ import annotations

from typing import cast

import pytest

from themis.catalog import load
from themis.catalog.benchmarks import BenchmarkDefinition
from themis.core.builtins import resolve_generator_component
from themis.core.contexts import GenerateContext
from themis.core.models import GenerationResult
from tests.catalog_ids import catalog_benchmark_ids


BENCHMARK_IDS = catalog_benchmark_ids()


def test_benchmark_catalog_smoke_ids_cover_manifest_entries() -> None:
    assert BENCHMARK_IDS == catalog_benchmark_ids()


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark_id", BENCHMARK_IDS)
async def test_benchmark_catalog_smoke_loads_experiment_and_invokes_generator(
    benchmark_id: str,
) -> None:
    benchmark = cast(BenchmarkDefinition, load(benchmark_id))
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
