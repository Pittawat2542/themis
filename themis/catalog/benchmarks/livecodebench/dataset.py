"""Dataset provider for LiveCodeBench code generation."""

from __future__ import annotations

import base64
from collections.abc import Callable
import json
import random
from typing import cast
from urllib import request
import zlib

from themis.benchmark.query import DatasetQuerySpec
from themis.specs.foundational import DatasetSpec
from themis.types.enums import DatasetSource, SamplingKind

from ..._http import DEFAULT_HTTP_TIMEOUT_SECONDS, iter_jsonl_url
from ...datasets._normalizers import _metadata_dict
from ...datasets._providers import (
    BuiltinDatasetProvider,
    _apply_query,
    _row_metadata_value,
)
from ...datasets._types import CatalogNormalizedRows

DEFAULT_LIVECODEBENCH_VERSION_TAG = "release_v6"

_RELEASE_FILES: dict[str, list[str]] = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
    ],
    "release_v6": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ],
}
_RELEASE_FILES["release_latest"] = list(_RELEASE_FILES["release_v6"])
for version_index in range(1, 7):
    version_name = f"v{version_index}"
    _RELEASE_FILES[version_name] = [
        "test.jsonl" if version_index == 1 else f"test{version_index}.jsonl"
    ]
for start in range(1, 7):
    for end in range(start + 1, 7):
        key = f"v{start}_v{end}"
        _RELEASE_FILES[key] = [
            "test.jsonl" if index == 1 else f"test{index}.jsonl"
            for index in range(start, end + 1)
        ]


class BuiltinLiveCodeBenchDatasetProvider(BuiltinDatasetProvider):
    def __init__(
        self,
        *,
        version_tag: str = DEFAULT_LIVECODEBENCH_VERSION_TAG,
        huggingface_loader=None,
    ) -> None:
        super().__init__(
            huggingface_loader=huggingface_loader or _load_livecodebench_rows
        )
        self._version_tag = version_tag

    def scan(self, slice_spec, query):
        dataset = slice_spec.dataset
        if dataset.source != DatasetSource.HUGGINGFACE or dataset.dataset_id is None:
            raise ValueError(
                "Built-in benchmark dataset providers require a HuggingFace dataset_id."
            )
        if self._huggingface_loader is _load_livecodebench_rows:
            if (
                query.kind == SamplingKind.ALL
                and not query.item_ids
                and not query.metadata_filters
            ):
                return super().scan(slice_spec, query)
            return self._scan_streaming(slice_spec, query)
        rows = self.load_rows(dataset)
        normalized = self.prepare_rows(rows, slice_spec)
        filtered = _apply_query(normalized.rows, query)
        self._last_scan_stats = {
            **normalized.stats,
            "loaded_count": len(rows),
            "normalized_count": len(normalized.rows),
            "returned_count": len(filtered),
        }
        return filtered

    def load_rows(self, dataset: DatasetSpec) -> list[dict[str, object]]:
        if dataset.dataset_id is None:
            raise ValueError(
                "Built-in HuggingFace dataset providers require a dataset_id."
            )
        loader = cast(Callable[..., list[dict[str, object]]], self._huggingface_loader)
        return loader(
            dataset.dataset_id,
            dataset.split,
            dataset.revision,
            version_tag=self._version_tag,
        )

    def normalize_loaded_rows(
        self,
        rows: list[dict[str, object]],
        dataset: object,
    ) -> CatalogNormalizedRows:
        del dataset
        normalized: list[dict[str, object]] = []
        skipped_missing_tests = 0
        for row in rows:
            payload = _normalize_livecodebench_row(dict(row))
            if payload is None:
                skipped_missing_tests += 1
                continue
            normalized.append(payload)
        return CatalogNormalizedRows(
            rows=normalized,
            stats={"skipped_missing_tests_count": skipped_missing_tests},
        )

    def _scan_streaming(self, slice_spec, query: DatasetQuerySpec):
        dataset = slice_spec.dataset
        if dataset.dataset_id is None:
            raise ValueError(
                "Built-in HuggingFace dataset providers require a dataset_id."
            )
        wanted_item_ids = set(query.item_ids)
        randomizer = random.Random(query.seed) if query.seed is not None else None
        selected: list[dict[str, object]] = []
        strata_buckets: dict[str, list[dict[str, object]]] = {}
        strata_seen: dict[str, int] = {}
        seen_matching = 0
        loaded_count = 0
        normalized_count = 0
        returned_count = 0
        for raw_row in _iter_livecodebench_rows(
            dataset.dataset_id,
            dataset.split,
            dataset.revision,
            version_tag=self._version_tag,
        ):
            loaded_count += 1
            payload = _normalize_livecodebench_row(raw_row)
            if payload is None:
                continue
            normalized_count += 1
            if wanted_item_ids and str(payload.get("item_id")) not in wanted_item_ids:
                continue
            if query.metadata_filters and not all(
                _row_metadata_value(payload, key) == value
                for key, value in query.metadata_filters.items()
            ):
                continue

            if query.kind == SamplingKind.ALL:
                selected.append(payload)
                continue

            count = query.count or 0
            if query.kind == SamplingKind.SUBSET:
                seen_matching += 1
                if query.seed is None:
                    if len(selected) < count:
                        selected.append(payload)
                        if (
                            not wanted_item_ids
                            and not query.metadata_filters
                            and len(selected) >= count
                        ):
                            break
                    continue
                if len(selected) < count:
                    selected.append(payload)
                    continue
                assert randomizer is not None
                replacement_index = randomizer.randint(0, seen_matching - 1)
                if replacement_index < count:
                    selected[replacement_index] = payload
                continue

            if query.kind == SamplingKind.STRATIFIED:
                strata_field = query.strata_field or ""
                bucket_key = _row_metadata_value(payload, strata_field)
                bucket = strata_buckets.setdefault(bucket_key, [])
                if query.seed is None:
                    if len(bucket) < count:
                        bucket.append(payload)
                    continue
                seen_in_bucket = strata_seen.get(bucket_key, 0) + 1
                strata_seen[bucket_key] = seen_in_bucket
                if len(bucket) < count:
                    bucket.append(payload)
                    continue
                assert randomizer is not None
                replacement_index = randomizer.randint(0, seen_in_bucket - 1)
                if replacement_index < count:
                    bucket[replacement_index] = payload

        if query.kind == SamplingKind.STRATIFIED:
            selected = [
                row for bucket_rows in strata_buckets.values() for row in bucket_rows
            ]
        returned_count = len(selected)
        self._last_scan_stats = {
            "loaded_count": loaded_count,
            "normalized_count": normalized_count,
            "returned_count": returned_count,
        }
        return selected


def _load_livecodebench_rows(
    dataset_id: str,
    split: str,
    revision: str | None = None,
    *,
    version_tag: str = DEFAULT_LIVECODEBENCH_VERSION_TAG,
    urlopen=request.urlopen,
) -> list[dict[str, object]]:
    return list(
        _iter_livecodebench_rows(
            dataset_id,
            split,
            revision,
            version_tag=version_tag,
            urlopen=urlopen,
        )
    )


def _iter_livecodebench_rows(
    dataset_id: str,
    split: str,
    revision: str | None = None,
    *,
    version_tag: str = DEFAULT_LIVECODEBENCH_VERSION_TAG,
    urlopen=request.urlopen,
):
    if split != "test":
        raise ValueError("LiveCodeBench only provides the 'test' split.")
    for filename in _livecodebench_filenames(version_tag):
        for payload in iter_jsonl_url(
            _livecodebench_file_url(dataset_id, revision, filename),
            urlopen=urlopen,
            timeout=DEFAULT_HTTP_TIMEOUT_SECONDS,
        ):
            if not isinstance(payload, dict):
                raise ValueError("LiveCodeBench rows must be JSON objects.")
            yield dict(payload)


def _livecodebench_filenames(version_tag: str) -> list[str]:
    normalized = version_tag.strip()
    if normalized not in _RELEASE_FILES:
        raise ValueError(
            "Unsupported LiveCodeBench version_tag. Choose one of: "
            + ", ".join(sorted(_RELEASE_FILES))
        )
    return list(_RELEASE_FILES[normalized])


def _livecodebench_file_url(
    dataset_id: str,
    revision: str | None,
    filename: str,
) -> str:
    ref = revision or "main"
    return (
        f"https://huggingface.co/datasets/{dataset_id}/resolve/{ref}/{filename}"
        "?download=true"
    )


def _normalize_livecodebench_row(
    payload: dict[str, object],
) -> dict[str, object] | None:
    tests = _decode_livecodebench_tests(payload.get("private_test_cases"))
    if not tests:
        tests = _decode_livecodebench_tests(payload.get("public_test_cases"))
    if not tests:
        return None
    execution_mode = _livecodebench_execution_mode(tests)
    payload["item_id"] = str(payload.get("question_id", payload.get("item_id", "")))
    payload["prompt_text"] = _livecodebench_prompt(
        question_content=str(payload.get("question_content", "")).strip(),
        starter_code=str(payload.get("starter_code", "") or "").rstrip(),
        execution_mode=execution_mode,
    )
    payload["language"] = "python"
    payload["execution_mode"] = execution_mode
    payload["input_mode"] = execution_mode
    payload["official_tests"] = tests
    function_name = _livecodebench_function_name(payload.get("metadata"))
    if execution_mode == "function":
        if function_name is None:
            raise ValueError(
                "livecodebench functional rows require metadata.func_name."
            )
        payload["function_name"] = function_name
    metadata = _metadata_dict(
        payload,
        ["platform", "contest_id", "difficulty", "contest_date"],
    )
    if function_name is not None:
        metadata["function_name"] = function_name
    payload["metadata"] = metadata
    return payload


def _decode_livecodebench_tests(value: object) -> list[dict[str, str]]:
    if isinstance(value, list):
        raw_tests = value
    elif isinstance(value, str):
        if not value.strip():
            return []
        try:
            raw_tests = json.loads(value)
        except json.JSONDecodeError:
            try:
                decoded = zlib.decompress(base64.b64decode(value))
                raw_tests = json.loads(decoded.decode("utf-8"))
            except (
                ValueError,
                zlib.error,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:
                raise ValueError(
                    "LiveCodeBench private_test_cases must be JSON or base64+zlib-compressed JSON."
                ) from exc
    else:
        return []
    if not isinstance(raw_tests, list):
        return []
    tests: list[dict[str, str]] = []
    for entry in raw_tests:
        if not isinstance(entry, dict):
            continue
        raw_input = entry.get("input")
        raw_output = entry.get("output")
        if not isinstance(raw_input, str) or not isinstance(raw_output, str):
            continue
        payload = {"input": raw_input, "output": raw_output}
        raw_testtype = entry.get("testtype")
        if isinstance(raw_testtype, str) and raw_testtype.strip():
            normalized_testtype = raw_testtype.strip().lower()
            if normalized_testtype != "stdin":
                payload["testtype"] = normalized_testtype
        tests.append(payload)
    return tests


def _livecodebench_execution_mode(tests: list[dict[str, str]]) -> str:
    if any(test.get("testtype") == "functional" for test in tests):
        return "function"
    return "stdio"


def _livecodebench_function_name(value: object) -> str | None:
    if isinstance(value, dict):
        candidate = value.get("func_name")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        return None
    if isinstance(value, str) and value.strip():
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return None
        return _livecodebench_function_name(payload)
    return None


def _livecodebench_prompt(
    *,
    question_content: str,
    starter_code: str,
    execution_mode: str,
) -> str:
    if execution_mode == "function" and starter_code:
        return (
            "Write a Python 3 solution for the following problem. Return only code.\n\n"
            f"Problem:\n{question_content}\n\n"
            "Starter code:\n"
            f"{starter_code}\n\n"
            "Preserve the same callable interface shown in the starter code."
        )
    return (
        "Write a Python 3 program that solves the following problem. "
        "Return only code.\n\n"
        f"Problem:\n{question_content}"
    )
