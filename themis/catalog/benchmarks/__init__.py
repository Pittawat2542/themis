"""Built-in benchmark registry for the catalog package."""

from __future__ import annotations

from themis import BenchmarkDefinition
from .aime_2025 import DEFINITION as AIME_2025_DEFINITION
from .aime_2026 import DEFINITION as AIME_2026_DEFINITION
from .aethercode import DEFINITION as AETHERCODE_DEFINITION
from .apex_2025 import DEFINITION as APEX_2025_DEFINITION
from .beyond_aime import DEFINITION as BEYOND_AIME_DEFINITION
from .babe import DEFINITION as BABE_DEFINITION
from .encyclo_k import DEFINITION as ENCYCLO_K_DEFINITION
from .frontierscience import DEFINITION as FRONTIERSCIENCE_DEFINITION
from .gpqa_diamond import DEFINITION as GPQA_DIAMOND_DEFINITION
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
from .mmmlu import (
    DEFINITION as MMMLU_DEFINITION,
    build_mmmlu_definition,
    supported_mmmlu_config_names,
)
from .phybench import DEFINITION as PHYBENCH_DEFINITION
from .procbench import (
    DEFINITION as PROCBENCH_DEFINITION,
    build_procbench_definition,
    supported_procbench_task_ids,
)
from .codeforces import DEFINITION as CODEFORCES_DEFINITION
from .simpleqa_verified import DEFINITION as SIMPLEQA_VERIFIED_DEFINITION
from .superchem import (
    DEFINITION as SUPERCHEM_DEFINITION,
    build_superchem_definition,
    supported_superchem_languages,
)
from .supergpqa import DEFINITION as SUPERGPQA_DEFINITION

_CATALOG_BENCHMARKS: dict[str, BenchmarkDefinition] = {
    definition.benchmark_id: definition
    for definition in (
        AIME_2025_DEFINITION,
        AIME_2026_DEFINITION,
        AETHERCODE_DEFINITION,
        APEX_2025_DEFINITION,
        BABE_DEFINITION,
        BEYOND_AIME_DEFINITION,
        GPQA_DIAMOND_DEFINITION,
        MMLU_PRO_DEFINITION,
        MMMLU_DEFINITION,
        SUPERGPQA_DEFINITION,
        ENCYCLO_K_DEFINITION,
        PHYBENCH_DEFINITION,
        SIMPLEQA_VERIFIED_DEFINITION,
        HEALTHBENCH_DEFINITION,
        FRONTIERSCIENCE_DEFINITION,
        HMMT_FEB_2025_DEFINITION,
        HMMT_NOV_2025_DEFINITION,
        IMO_ANSWERBENCH_DEFINITION,
        LPFQA_DEFINITION,
        HLE_DEFINITION,
        CODEFORCES_DEFINITION,
        LIVECODEBENCH_DEFINITION,
        PROCBENCH_DEFINITION,
        SUPERCHEM_DEFINITION,
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
    if normalized.startswith("mmmlu:"):
        config_name = normalized.split(":", 1)[1].strip()
        supported_configs = {
            name.lower(): name for name in supported_mmmlu_config_names()
        }
        if config_name not in {name.lower() for name in supported_mmmlu_config_names()}:
            raise ValueError(
                "Built-in benchmark 'mmmlu' received unknown config name: "
                f"{config_name}"
            )
        return build_mmmlu_definition(supported_configs[config_name])
    if normalized.startswith("procbench:"):
        task_id = normalized.split(":", 1)[1].strip()
        supported_task_ids = set(supported_procbench_task_ids())
        if task_id not in supported_task_ids:
            raise ValueError(
                f"Built-in benchmark 'procbench' received unknown task id: {task_id}"
            )
        return build_procbench_definition([task_id])
    if normalized.startswith("superchem:"):
        language = normalized.split(":", 1)[1].strip()
        supported_languages = set(supported_superchem_languages())
        if language not in supported_languages:
            raise ValueError(
                f"Built-in benchmark 'superchem' received unknown language: {language}"
            )
        return build_superchem_definition(language)
    if normalized not in _CATALOG_BENCHMARKS:
        raise ValueError(
            "Unknown built-in benchmark. Choose one of: "
            + ", ".join(sorted(_CATALOG_BENCHMARKS))
        )
    return _CATALOG_BENCHMARKS[normalized]


__all__ = ["get_catalog_benchmark", "list_catalog_benchmarks"]
