"""Dataset provider and cached oracle helpers for HumanEval."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, cast
from urllib import request

from themis.specs.foundational import DatasetSpec

from ...datasets.common import BuiltinDatasetProvider, CatalogNormalizedRows

DEFAULT_HUMANEVAL_PLUS_VERSION = "v0.1.10"
_CACHE_ROOT = Path.home() / ".cache" / "themis" / "humaneval"

type HumanEvalRow = dict[str, object]
type HumanEvalOracle = dict[str, object]
type HumanEvalOracleCache = dict[str, HumanEvalOracle]
type HumanEvalRowLoader = Callable[..., list[HumanEvalRow]]


class BuiltinHumanEvalDatasetProvider(BuiltinDatasetProvider):
    def __init__(
        self,
        *,
        mini: bool = False,
        noextreme: bool = False,
        version: str = DEFAULT_HUMANEVAL_PLUS_VERSION,
        score_variant: str = "base",
        huggingface_loader: HumanEvalRowLoader | None = None,
    ) -> None:
        super().__init__(
            huggingface_loader=huggingface_loader or load_humaneval_plus_rows
        )
        self._mini = mini
        self._noextreme = noextreme
        self._version = version
        self._score_variant = score_variant

    def load_rows(self, dataset: DatasetSpec) -> list[HumanEvalRow]:
        del dataset
        loader = cast(HumanEvalRowLoader, self._huggingface_loader)
        return loader(
            mini=self._mini,
            noextreme=self._noextreme,
            version=self._version,
        )

    def normalize_loaded_rows(
        self,
        rows: list[dict[str, object]],
        dataset: object,
    ) -> CatalogNormalizedRows:
        del dataset
        with_oracles = attach_ground_truth(rows)
        normalized = [
            _normalize_humaneval_row(row, score_variant=self._score_variant)
            for row in with_oracles
        ]
        return CatalogNormalizedRows(rows=normalized)

    def prepare_rows(
        self,
        rows: list[dict[str, object]],
        dataset_or_slice: object,
    ) -> CatalogNormalizedRows:
        score_variant = self._score_variant
        slice_id = getattr(dataset_or_slice, "slice_id", None)
        if isinstance(slice_id, str) and "plus" in slice_id:
            score_variant = "plus"
        with_oracles = attach_ground_truth(rows)
        normalized = [
            _normalize_humaneval_row(row, score_variant=score_variant)
            for row in with_oracles
        ]
        return CatalogNormalizedRows(rows=normalized)


def parse_humaneval_variants(
    base_benchmark_id: str,
    raw_name: str,
) -> tuple[bool, bool, str]:
    prefix = f"{base_benchmark_id}:"
    if not raw_name.startswith(prefix):
        return False, False, DEFAULT_HUMANEVAL_PLUS_VERSION
    raw_tokens = [token.strip() for token in raw_name.split(":", 1)[1].split(",")]
    if not raw_tokens or any(not token for token in raw_tokens):
        raise ValueError(
            f"Built-in benchmark '{base_benchmark_id}' requires non-empty variant tokens."
        )
    if len(raw_tokens) != len(set(raw_tokens)):
        raise ValueError(
            f"Built-in benchmark '{base_benchmark_id}' received duplicate variant tokens."
        )
    mini = False
    noextreme = False
    version: str | None = None
    for token in raw_tokens:
        if token == "mini":
            mini = True
            continue
        if token == "noextreme":
            noextreme = True
            continue
        if token.startswith("v"):
            if version is not None:
                raise ValueError(
                    f"Built-in benchmark '{base_benchmark_id}' received multiple version tokens."
                )
            version = token
            continue
        raise ValueError(
            f"Built-in benchmark '{base_benchmark_id}' received unknown variant token: {token}"
        )
    if mini and noextreme:
        raise ValueError(
            f"Built-in benchmark '{base_benchmark_id}' cannot combine mini and noextreme."
        )
    return mini, noextreme, version or DEFAULT_HUMANEVAL_PLUS_VERSION


def load_humaneval_plus_rows(
    *,
    mini: bool = False,
    noextreme: bool = False,
    version: str = DEFAULT_HUMANEVAL_PLUS_VERSION,
    override_path: str | None = None,
    download: bool = True,
    urlopen=request.urlopen,
) -> list[HumanEvalRow]:
    if mini and noextreme:
        raise ValueError("HumanEval variants cannot combine mini and noextreme.")
    path = resolve_humaneval_plus_path(
        mini=mini,
        noextreme=noextreme,
        version=version,
        override_path=override_path,
        download=download,
        urlopen=urlopen,
    )
    return _stream_jsonl(path)


def resolve_humaneval_plus_path(
    *,
    mini: bool,
    noextreme: bool,
    version: str,
    override_path: str | None = None,
    download: bool = True,
    urlopen=request.urlopen,
) -> Path:
    env_override = override_path or os.getenv("HUMANEVAL_OVERRIDE_PATH")
    if env_override:
        return Path(env_override)
    cache_path = _cache_dataset_path(mini=mini, noextreme=noextreme, version=version)
    if cache_path.exists() or not download:
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_url = _humaneval_plus_url(mini=mini, noextreme=noextreme, version=version)
    with urlopen(dataset_url) as response:
        compressed = response.read()
    payload = gzip.decompress(compressed).decode("utf-8")
    cache_path.write_text(payload)
    return cache_path


def attach_ground_truth(rows: list[HumanEvalRow]) -> list[HumanEvalRow]:
    if not rows:
        return []
    cache_path = _ground_truth_cache_path(rows)
    if cache_path.exists():
        cached = _oracle_cache(_json_loads_unbounded(cache_path.read_text()))
    else:
        cached = _build_ground_truth_cache(rows)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(_json_dumps_unbounded(cached, sort_keys=True))
    enriched: list[HumanEvalRow] = []
    for row in rows:
        payload = dict(row)
        oracle = cached[str(payload["task_id"])]
        payload["base_expected"] = oracle["base_expected"]
        payload["plus_expected"] = oracle["plus_expected"]
        payload["base_time_limits"] = oracle["base_time_limits"]
        payload["plus_time_limits"] = oracle["plus_time_limits"]
        enriched.append(payload)
    return enriched


def _build_ground_truth_cache(rows: list[HumanEvalRow]) -> HumanEvalOracleCache:
    cache: HumanEvalOracleCache = {}
    for row in rows:
        prompt = str(row["prompt"])
        solution = str(row["canonical_solution"])
        entry_point = str(row["entry_point"])
        base_inputs = _normalize_input_vectors(row.get("base_input"))
        plus_inputs = _normalize_input_vectors(row.get("plus_input"))
        base_expected, base_times = _trusted_exec(
            prompt + solution, base_inputs, entry_point
        )
        plus_expected, plus_times = _trusted_exec(
            prompt + solution, plus_inputs, entry_point
        )
        cache[str(row["task_id"])] = {
            "base_expected": base_expected,
            "plus_expected": plus_expected,
            "base_time_limits": [_derive_time_limit(value) for value in base_times],
            "plus_time_limits": [_derive_time_limit(value) for value in plus_times],
        }
    return cache


def _normalize_humaneval_row(
    row: HumanEvalRow,
    *,
    score_variant: str,
) -> HumanEvalRow:
    base_inputs = _normalize_input_vectors(row.get("base_input"))
    plus_inputs = _normalize_input_vectors(row.get("plus_input"))
    prompt = str(row.get("prompt", ""))
    entry_point = str(row.get("entry_point", ""))
    base_expected = _object_list(row.get("base_expected"))
    plus_expected = _object_list(row.get("plus_expected"))
    base_time_limits = _object_list(row.get("base_time_limits"))
    plus_time_limits = _object_list(row.get("plus_time_limits"))
    return {
        **row,
        "item_id": str(row.get("task_id", "")),
        "language": "python",
        "execution_mode": "function",
        "function_name": entry_point,
        "score_variant": score_variant,
        "prompt_text": (
            "Write a complete Python solution for the following task. "
            "Return only Python code.\n\n"
            f"{prompt.rstrip()}\n"
        ),
        "official_tests": [
            _serialized_test_case(inp, out)
            for inp, out in zip(base_inputs, base_expected, strict=True)
        ],
        "plus_tests": [
            _serialized_test_case(inp, out)
            for inp, out in zip(plus_inputs, plus_expected, strict=True)
        ],
        "atol": _coerce_float(row.get("atol"), default=0.0),
        "base_expected": base_expected,
        "plus_expected": plus_expected,
        "base_time_limits": base_time_limits,
        "plus_time_limits": plus_time_limits,
    }


def _serialized_test_case(inp: object, out: object) -> dict[str, str]:
    return {
        "input": _json_dumps_unbounded(inp),
        "output": _json_dumps_unbounded(out),
    }


def _trusted_exec(
    code: str,
    inputs: list[list[object]],
    entry_point: str,
) -> tuple[list[object], list[float]]:
    exec_globals: dict[str, object] = {}
    exec(code, exec_globals)
    target = _resolve_callable(exec_globals, entry_point)
    outputs: list[object] = []
    runtimes: list[float] = []
    for raw_input in inputs:
        args = copy.deepcopy(raw_input)
        start = time.perf_counter()
        outputs.append(target(*args))
        runtimes.append(time.perf_counter() - start)
    return outputs, runtimes


def _resolve_callable(exec_globals: dict[str, object], entry_point: str):
    candidate = exec_globals.get(entry_point)
    if callable(candidate):
        return candidate
    solution_cls = exec_globals.get("Solution")
    if isinstance(solution_cls, type) and hasattr(solution_cls, entry_point):
        return getattr(solution_cls(), entry_point)
    raise ValueError(f"Missing callable {entry_point!r} in trusted solution.")


def _normalize_input_vectors(value: object) -> list[list[object]]:
    if not isinstance(value, list):
        return []
    normalized: list[list[object]] = []
    for item in value:
        if isinstance(item, list):
            normalized.append(item)
        elif isinstance(item, tuple):
            normalized.append(list(item))
        else:
            normalized.append([item])
    return normalized


def _derive_time_limit(runtime_seconds: float) -> float:
    return max(0.05, 2.0 * runtime_seconds, 0.1)


def _cache_dataset_path(*, mini: bool, noextreme: bool, version: str) -> Path:
    suffix = ""
    if mini:
        suffix = "-Mini"
    elif noextreme:
        suffix = "-NoExtreme"
    return _CACHE_ROOT / f"HumanEvalPlus{suffix}-{version}.jsonl"


def _ground_truth_cache_path(rows: list[HumanEvalRow]) -> Path:
    payload = _json_dumps_unbounded(rows, sort_keys=True).encode("utf-8")
    digest = hashlib.md5(payload).hexdigest()
    return _CACHE_ROOT / f"{digest}.groundtruth.json"


def _humaneval_plus_url(*, mini: bool, noextreme: bool, version: str) -> str:
    extra = ""
    if mini:
        extra = "-Mini"
    elif noextreme:
        extra = "-NoExtreme"
    return (
        "https://github.com/evalplus/humanevalplus_release/releases/download/"
        f"{version}/HumanEvalPlus{extra}.jsonl.gz"
    )


def _stream_jsonl(path: Path) -> list[HumanEvalRow]:
    rows: list[HumanEvalRow] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(_row_payload(_json_loads_unbounded(line)))
    return rows


def _json_loads_unbounded(payload: str) -> object:
    return _with_unbounded_int_strings(lambda: json.loads(payload))


def _json_dumps_unbounded(value: object, *, sort_keys: bool = False) -> str:
    return _with_unbounded_int_strings(lambda: json.dumps(value, sort_keys=sort_keys))


def _with_unbounded_int_strings(callback: Callable[[], Any]) -> Any:
    set_limit = getattr(sys, "set_int_max_str_digits", None)
    get_limit = getattr(sys, "get_int_max_str_digits", None)
    if set_limit is None or get_limit is None:
        return callback()
    previous_limit = int(get_limit())
    if previous_limit == 0:
        return callback()
    try:
        set_limit(0)
        return callback()
    finally:
        set_limit(previous_limit)


def _row_payload(value: object) -> HumanEvalRow:
    if not isinstance(value, dict):
        raise ValueError("HumanEval rows must be JSON objects.")
    return {str(key): item for key, item in value.items()}


def _oracle_cache(value: object) -> HumanEvalOracleCache:
    if not isinstance(value, dict):
        raise ValueError("HumanEval oracle cache must be a JSON object.")
    cache: HumanEvalOracleCache = {}
    for key, item in value.items():
        if not isinstance(item, dict):
            raise ValueError("HumanEval oracle cache entries must be JSON objects.")
        cache[str(key)] = {
            str(field): field_value for field, field_value in item.items()
        }
    return cache


def _object_list(value: object) -> list[object]:
    if not isinstance(value, list):
        return []
    return list(value)


def _coerce_float(value: object, *, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default
