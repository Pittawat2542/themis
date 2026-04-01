from __future__ import annotations

from themis.catalog.benchmarks import benchmark_specs


OPEN_VARIANT_EXAMPLES = {
    "hle": "math,reasoning",
    "humaneval": "mini",
    "humaneval_plus": "noextreme",
    "mmmlu": "thai",
    "procbench": "task07",
}


def _string_list(value: object) -> list[str]:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return []


def catalog_benchmark_ids() -> list[str]:
    benchmark_ids: list[str] = []
    for benchmark_id, spec in benchmark_specs().items():
        benchmark_ids.append(benchmark_id)
        for variant in _string_list(spec.get("variants", [])):
            benchmark_ids.append(f"{benchmark_id}:{variant}")
        if spec.get("variant_mode") == "open":
            benchmark_ids.append(
                f"{benchmark_id}:{OPEN_VARIANT_EXAMPLES[benchmark_id]}"
            )
    return benchmark_ids
