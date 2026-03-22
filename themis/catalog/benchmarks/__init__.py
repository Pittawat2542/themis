"""Built-in benchmark registry for the catalog package."""

from __future__ import annotations

from themis import BenchmarkDefinition
from .aime_2025 import DEFINITION as AIME_2025_DEFINITION
from .aime_2026 import DEFINITION as AIME_2026_DEFINITION
from .aethercode import DEFINITION as AETHERCODE_DEFINITION
from .apex_2025 import DEFINITION as APEX_2025_DEFINITION
from .beyond_aime import DEFINITION as BEYOND_AIME_DEFINITION
from .encyclo_k import DEFINITION as ENCYCLO_K_DEFINITION
from .healthbench import DEFINITION as HEALTHBENCH_DEFINITION
from .hle import (
    DEFINITION as HLE_DEFINITION,
    build_hle_definition,
    supported_hle_variant_ids,
)
from .hmmt_feb_2025 import DEFINITION as HMMT_FEB_2025_DEFINITION
from .hmmt_nov_2025 import DEFINITION as HMMT_NOV_2025_DEFINITION
from .imo_answerbench import DEFINITION as IMO_ANSWERBENCH_DEFINITION
from .lpfqa import DEFINITION as LPFQA_DEFINITION
from .livecodebench import DEFINITION as LIVECODEBENCH_DEFINITION
from .mmlu_pro import DEFINITION as MMLU_PRO_DEFINITION
from .codeforces import DEFINITION as CODEFORCES_DEFINITION
from .simpleqa_verified import DEFINITION as SIMPLEQA_VERIFIED_DEFINITION
from .supergpqa import DEFINITION as SUPERGPQA_DEFINITION

_CATALOG_BENCHMARKS: dict[str, BenchmarkDefinition] = {
    definition.benchmark_id: definition
    for definition in (
        AIME_2025_DEFINITION,
        AIME_2026_DEFINITION,
        AETHERCODE_DEFINITION,
        APEX_2025_DEFINITION,
        BEYOND_AIME_DEFINITION,
        MMLU_PRO_DEFINITION,
        SUPERGPQA_DEFINITION,
        ENCYCLO_K_DEFINITION,
        SIMPLEQA_VERIFIED_DEFINITION,
        HEALTHBENCH_DEFINITION,
        HMMT_FEB_2025_DEFINITION,
        HMMT_NOV_2025_DEFINITION,
        IMO_ANSWERBENCH_DEFINITION,
        LPFQA_DEFINITION,
        HLE_DEFINITION,
        CODEFORCES_DEFINITION,
        LIVECODEBENCH_DEFINITION,
    )
}


def list_catalog_benchmarks() -> list[str]:
    return sorted(_CATALOG_BENCHMARKS)


def get_catalog_benchmark(name: str) -> BenchmarkDefinition:
    normalized = name.strip().lower().replace("-", "_")
    if normalized == "hle":
        supported = ", ".join(supported_hle_variant_ids())
        raise ValueError(
            "Built-in benchmark 'hle' requires explicit HLE variants. "
            f"Supported variants: {supported}. "
            "Examples: hle:text_only, hle:no_tool, hle:text_only,no_tool."
        )
    if normalized.startswith("hle:"):
        raw_variant_ids = [
            part.strip() for part in normalized.split(":", 1)[1].split(",")
        ]
        if not raw_variant_ids or any(not part for part in raw_variant_ids):
            raise ValueError(
                "Built-in benchmark 'hle' requires one or more HLE variant ids after ':'."
            )
        supported_variant_ids = set(supported_hle_variant_ids())
        duplicate_variant_ids = sorted(
            {
                variant_id
                for variant_id in raw_variant_ids
                if raw_variant_ids.count(variant_id) > 1
            }
        )
        if duplicate_variant_ids:
            raise ValueError(
                "Built-in benchmark 'hle' received duplicate HLE variant ids: "
                + ", ".join(duplicate_variant_ids)
            )
        unknown_variant_ids = sorted(
            variant_id
            for variant_id in raw_variant_ids
            if variant_id not in supported_variant_ids
        )
        if unknown_variant_ids:
            raise ValueError(
                "Built-in benchmark 'hle' received unknown HLE variant ids: "
                + ", ".join(unknown_variant_ids)
            )
        return build_hle_definition(raw_variant_ids)
    if normalized not in _CATALOG_BENCHMARKS:
        raise ValueError(
            "Unknown built-in benchmark. Choose one of: "
            + ", ".join(sorted(_CATALOG_BENCHMARKS))
        )
    return _CATALOG_BENCHMARKS[normalized]


__all__ = ["get_catalog_benchmark", "list_catalog_benchmarks"]
