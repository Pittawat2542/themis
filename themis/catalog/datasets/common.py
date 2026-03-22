"""Dataset providers and Hugging Face loading helpers for catalog benchmarks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import csv
from dataclasses import dataclass, field
import inspect
import json
from pathlib import Path
import random
import re

from themis._optional import import_optional
from themis.prompting import render_prompt_messages
from themis import PromptMessage
from themis.specs.foundational import DatasetSpec
from themis.types.enums import DatasetSource, PromptRole, SamplingKind
from themis.types.json_types import JSONDict
from themis.types.json_validation import validate_json_dict

_HLE_RESPONSE_TEMPLATE = """Explanation: {explanation}
Answer: {answer}
Confidence: {confidence}%"""

type CatalogRow = dict[str, object]
type CatalogPromptMessage = dict[str, object]
type CatalogMetadataLoader = Callable[[str, str | None], JSONDict]
type CatalogRowLoader = Callable[..., list[CatalogRow]]
type CatalogRowNormalizer = Callable[
    [list[CatalogRow], object], "CatalogNormalizedRows"
]


@dataclass(frozen=True, slots=True)
class CatalogNormalizedRows:
    rows: list[CatalogRow]
    stats: JSONDict = field(default_factory=dict)


class CatalogDatasetProvider:
    """Dataset provider covering inline, local-file, and HuggingFace catalogs."""

    def __init__(
        self,
        *,
        memory_rows: list[CatalogRow] | None = None,
        huggingface_loader: CatalogRowLoader | None = None,
        local_loader: Callable[[Path], list[CatalogRow]] | None = None,
        row_normalizer: CatalogRowNormalizer | None = None,
    ) -> None:
        self._memory_rows = list(memory_rows or [])
        self._huggingface_loader = huggingface_loader or load_huggingface_rows
        self._local_loader = local_loader or load_local_rows
        self._row_normalizer = row_normalizer
        self._last_scan_stats: JSONDict = {}

    def scan(self, slice_spec, query):
        dataset = slice_spec.dataset
        if dataset.source == DatasetSource.MEMORY:
            rows = list(self._memory_rows)
        elif dataset.source == DatasetSource.LOCAL:
            dataset_path = dataset.dataset_id or dataset.data_dir
            if dataset_path is None:
                raise ValueError("Local catalog datasets require a dataset path.")
            rows = self._local_loader(Path(dataset_path))
        elif dataset.source == DatasetSource.HUGGINGFACE:
            if dataset.dataset_id is None:
                raise ValueError("HuggingFace catalog datasets require a dataset_id.")
            rows = _invoke_huggingface_loader(
                self._huggingface_loader,
                dataset.dataset_id,
                dataset.split,
                dataset.revision,
                config_name=dataset.config_name,
            )
        else:
            raise ValueError(f"Unsupported catalog dataset source '{dataset.source}'.")
        normalized = self.prepare_rows(rows, dataset)
        filtered = _apply_query(normalized.rows, query)
        self._last_scan_stats = {
            **normalized.stats,
            "loaded_count": len(rows),
            "normalized_count": len(normalized.rows),
            "returned_count": len(filtered),
        }
        return filtered

    def prepare_rows(
        self,
        rows: list[CatalogRow],
        dataset_or_slice: object,
    ) -> CatalogNormalizedRows:
        return _normalize_rows_for_provider(
            rows, dataset_or_slice, self._row_normalizer
        )

    def last_scan_stats(self) -> JSONDict:
        return dict(self._last_scan_stats)


class BuiltinDatasetProvider(CatalogDatasetProvider):
    """Benchmark-aware Hugging Face dataset provider base used by built-ins."""

    def __init__(self, *, huggingface_loader=None) -> None:
        super().__init__(huggingface_loader=huggingface_loader)

    def scan(self, slice_spec, query):
        dataset = slice_spec.dataset
        if dataset.source != DatasetSource.HUGGINGFACE or dataset.dataset_id is None:
            raise ValueError(
                "Built-in benchmark dataset providers require a HuggingFace dataset_id."
            )
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
        return _invoke_huggingface_loader(
            self._huggingface_loader,
            dataset.dataset_id,
            dataset.split,
            dataset.revision,
            config_name=dataset.config_name,
        )

    def normalize_loaded_rows(
        self,
        rows: list[CatalogRow],
        dataset: object,
    ) -> CatalogNormalizedRows:
        return CatalogNormalizedRows(rows=[dict(row) for row in rows])

    def prepare_rows(
        self,
        rows: list[CatalogRow],
        dataset_or_slice: object,
    ) -> CatalogNormalizedRows:
        return _normalize_rows_for_provider(
            rows, dataset_or_slice, self.normalize_loaded_rows
        )


class BuiltinMCQDatasetProvider(BuiltinDatasetProvider):
    def __init__(
        self,
        *,
        metadata_keys: list[str],
        huggingface_loader=None,
    ) -> None:
        super().__init__(huggingface_loader=huggingface_loader or load_huggingface_rows)
        self._metadata_keys = list(metadata_keys)

    def normalize_loaded_rows(
        self,
        rows: list[CatalogRow],
        dataset: object,
    ) -> CatalogNormalizedRows:
        return _normalize_mcq_rows(rows, dataset, metadata_keys=self._metadata_keys)


def load_local_rows(path: Path) -> list[CatalogRow]:
    """Load catalog dataset rows from JSONL or CSV."""

    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, object]] = []
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"JSONL row {line_number} in {path.name} must be an object."
                )
            rows.append(dict(payload))
        return _assign_missing_item_ids(rows)
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as fh:
            return _assign_missing_item_ids([dict(row) for row in csv.DictReader(fh)])
    raise ValueError("Catalog local datasets must use .jsonl or .csv.")


def load_huggingface_rows(
    dataset_id: str,
    split: str,
    revision: str | None = None,
    config_name: str | None = None,
    *,
    datasets_module=None,
) -> list[CatalogRow]:
    """Load catalog dataset rows from a HuggingFace dataset identifier."""

    datasets = datasets_module or import_optional("datasets", extra="datasets")
    try:
        dataset = _datasets_load_dataset(
            datasets,
            dataset_id,
            split=split,
            revision=revision,
            config_name=config_name,
        )
    except Exception as exc:
        if _should_retry_huggingface_streaming(dataset_id, exc, datasets):
            dataset = _datasets_load_dataset(
                datasets,
                dataset_id,
                split=split,
                revision=revision,
                config_name=config_name,
                streaming=True,
            )
        else:
            raise
    dataset = _prepare_huggingface_dataset_for_iteration(dataset, datasets)
    return _assign_missing_item_ids([dict(row) for row in dataset])


def load_hle_rows(
    dataset_id: str,
    split: str,
    revision: str | None = None,
    config_name: str | None = None,
    *,
    datasets_module=None,
) -> list[CatalogRow]:
    """Load HLE rows without forcing image decoding during iteration."""

    datasets = datasets_module or import_optional("datasets", extra="datasets")
    dataset = _datasets_load_dataset(
        datasets,
        dataset_id,
        split=split,
        revision=revision,
        config_name=config_name,
    )
    dataset = _prepare_huggingface_dataset_for_iteration(dataset, datasets)
    return _assign_missing_item_ids([dict(row) for row in dataset])


def load_healthbench_rows(
    dataset_id: str,
    split: str,
    revision: str | None = None,
    config_name: str | None = None,
    *,
    datasets_module=None,
) -> list[CatalogRow]:
    """Load HealthBench rows with a dataset-specific streaming fallback."""

    datasets = datasets_module or import_optional("datasets", extra="datasets")
    try:
        dataset = _datasets_load_dataset(
            datasets,
            dataset_id,
            split=split,
            revision=revision,
            config_name=config_name,
        )
    except Exception as exc:
        if _should_retry_huggingface_streaming(dataset_id, exc, datasets):
            dataset = _datasets_load_dataset(
                datasets,
                dataset_id,
                split=split,
                revision=revision,
                config_name=config_name,
                streaming=True,
            )
        else:
            raise
    return _assign_missing_item_ids([dict(row) for row in dataset])


def inspect_huggingface_dataset(
    dataset_id: str,
    *,
    config_name: str | None = None,
    split: str = "test",
    revision: str | None = None,
    metadata_loader: CatalogMetadataLoader | None = None,
    row_loader: CatalogRowLoader | None = None,
    max_samples: int = 3,
    datasets_module=None,
) -> JSONDict:
    datasets = datasets_module or import_optional("datasets", extra="datasets")
    metadata = (
        metadata_loader(dataset_id, revision)
        if metadata_loader is not None
        else _default_dataset_metadata(
            dataset_id,
            revision,
            datasets_module=datasets,
        )
    )
    rows = (
        _invoke_huggingface_loader(
            row_loader,
            dataset_id,
            split,
            revision,
            config_name=config_name,
        )
        if row_loader is not None
        else load_huggingface_rows(
            dataset_id,
            split,
            revision,
            config_name,
            datasets_module=datasets,
        )
    )
    return validate_json_dict(
        {
            "dataset_id": dataset_id,
            "config_name": config_name,
            "split": split,
            "revision": revision,
            "splits": _string_list(metadata.get("splits")),
            "gated": bool(metadata.get("gated", False)),
            "modalities": _string_list(metadata.get("modalities")),
            "fields": _infer_field_types(rows),
            "row_count": len(rows),
            "samples": _json_rows(rows[:max_samples], label="catalog dataset samples"),
            **_suggest_dataset_wiring(rows),
        },
        label="catalog dataset inspection",
    )


def _apply_query(rows: list[CatalogRow], query) -> list[CatalogRow]:
    filtered = list(rows)
    if query.metadata_filters:
        filtered = [
            row
            for row in filtered
            if all(
                _row_metadata_value(row, key) == value
                for key, value in query.metadata_filters.items()
            )
        ]
    if query.item_ids:
        wanted = set(query.item_ids)
        filtered = [row for row in filtered if str(row.get("item_id")) in wanted]
    if query.kind == SamplingKind.ALL:
        return filtered
    count = query.count or 0
    if query.kind == SamplingKind.SUBSET:
        if query.seed is None:
            return filtered[:count]
        if count >= len(filtered):
            return filtered
        return random.Random(query.seed).sample(filtered, count)
    if query.kind == SamplingKind.STRATIFIED:
        field = query.strata_field
        if not field:
            return filtered
        buckets: dict[str, list[CatalogRow]] = {}
        for row in filtered:
            buckets.setdefault(_row_metadata_value(row, field), []).append(row)
        randomizer = random.Random(query.seed)
        samples: list[CatalogRow] = []
        for bucket_rows in buckets.values():
            if len(bucket_rows) <= count:
                samples.extend(bucket_rows)
            else:
                samples.extend(randomizer.sample(bucket_rows, count))
        return samples
    return filtered


def _datasets_load_dataset(
    datasets_module,
    dataset_id: str,
    *,
    split: str,
    revision: str | None,
    config_name: str | None,
    streaming: bool = False,
):
    kwargs: dict[str, object] = {
        "split": split,
        "revision": revision,
    }
    if streaming:
        kwargs["streaming"] = True
    if config_name is None:
        return datasets_module.load_dataset(dataset_id, **kwargs)
    return datasets_module.load_dataset(dataset_id, config_name, **kwargs)


def _invoke_huggingface_loader(
    loader: CatalogRowLoader,
    dataset_id: str,
    split: str,
    revision: str | None,
    *,
    config_name: str | None,
) -> list[CatalogRow]:
    signature = inspect.signature(loader)
    if "config_name" in signature.parameters:
        return loader(
            dataset_id,
            split,
            revision,
            config_name=config_name,
        )
    return loader(dataset_id, split, revision)


def _assign_missing_item_ids(rows: list[CatalogRow]) -> list[CatalogRow]:
    normalized: list[CatalogRow] = []
    for index, row in enumerate(rows, start=1):
        payload = dict(row)
        payload.setdefault("item_id", payload.get("id", f"item-{index}"))
        normalized.append(payload)
    return normalized


def _render_string_template(template: str, payload: CatalogRow) -> str:
    message = PromptMessage(role=PromptRole.USER, content=template)
    rendered = render_prompt_messages([message], payload, strict=True)[0]["content"]
    if not isinstance(rendered, str):
        raise ValueError("Catalog dataset transforms require string prompt content.")
    return rendered


def _apply_dataset_transforms(
    rows: list[CatalogRow],
    dataset: DatasetSpec,
) -> list[CatalogRow]:
    transformed_rows = [dict(row) for row in rows]
    for transform in dataset.transforms:
        if transform.kind == "rename":
            for row in transformed_rows:
                row[transform.field] = row.get(transform.source_field)
            continue
        if transform.kind == "jinja":
            for row in transformed_rows:
                row[transform.field] = _render_string_template(transform.template, row)
            continue
        if transform.kind == "python":
            raise ValueError(
                "DatasetSpec python transforms are not supported by CatalogDatasetProvider."
            )
    return transformed_rows


def _normalize_rows_for_provider(
    rows: list[CatalogRow],
    dataset_or_slice: object,
    row_normalizer: CatalogRowNormalizer | None,
) -> CatalogNormalizedRows:
    dataset = getattr(dataset_or_slice, "dataset", dataset_or_slice)
    if not isinstance(dataset, DatasetSpec):
        raise ValueError("Catalog dataset normalization requires a DatasetSpec.")
    assigned = _assign_missing_item_ids(rows)
    normalized = (
        row_normalizer(assigned, dataset_or_slice)
        if row_normalizer is not None
        else CatalogNormalizedRows(rows=assigned)
    )
    transformed = _apply_dataset_transforms(normalized.rows, dataset)
    return CatalogNormalizedRows(rows=transformed, stats=normalized.stats)


def _metadata_dict(payload: CatalogRow, keys: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for key in keys:
        value = payload.get(key)
        if value is not None:
            if isinstance(value, list):
                metadata[key] = ", ".join(str(item) for item in value)
            else:
                metadata[key] = str(value)
    return metadata


def _format_options_text(options: object) -> str:
    if not isinstance(options, list):
        return str(options)
    lines: list[str] = []
    for index, option in enumerate(options):
        label = "ABCDEFGHIJ"[index] if index < 10 else str(index + 1)
        lines.append(f"{label}. {option}")
    return "\n".join(lines)


def _extract_lpfqa_reference_answer(text: str) -> str:
    marker = "<参考答案>"
    if marker not in text:
        return text
    segment = text.split(marker, maxsplit=1)[1]
    segment = segment.lstrip("：:").strip()
    return segment.split("<评估要点>", maxsplit=1)[0].strip()


def _normalize_mcq_rows(
    rows: list[CatalogRow],
    dataset: object,
    *,
    metadata_keys: list[str],
) -> CatalogNormalizedRows:
    del dataset
    normalized: list[CatalogRow] = []
    for row in rows:
        payload = dict(row)
        payload["options_text"] = _format_options_text(payload.get("options"))
        payload["metadata"] = _metadata_dict(payload, metadata_keys)
        normalized.append(payload)
    return CatalogNormalizedRows(rows=normalized)


def _normalize_gpqa_diamond_rows(
    rows: list[CatalogRow],
    dataset: object,
) -> CatalogNormalizedRows:
    del dataset
    normalized: list[CatalogRow] = []
    for row in rows:
        payload = dict(row)
        question, options = _parse_gpqa_diamond_question(
            str(payload.get("question", ""))
        )
        payload["question"] = question
        payload["options"] = options
        payload["expected"] = str(payload.get("answer", "")).strip().upper()
        payload["metadata"] = {}
        payload["options_text"] = _format_options_text(options)
        normalized.append(payload)
    return CatalogNormalizedRows(rows=normalized)


def _normalize_babe_rows(
    rows: list[CatalogRow],
    dataset: object,
) -> CatalogNormalizedRows:
    del dataset
    normalized: list[CatalogRow] = []
    for row in rows:
        payload = dict(row)
        label_value = payload.get("label", 0)
        label = int(label_value) if isinstance(label_value, int | bool | float) else 0
        payload["item_id"] = str(payload.get("uuid", payload.get("item_id", "")))
        payload["question"] = str(payload.get("text", ""))
        payload["options"] = ["Entirely factual", "Opinionated or subjective"]
        payload["expected"] = "A" if label == 0 else "B"
        payload["metadata"] = _metadata_dict(
            payload, ["outlet", "topic", "type", "label_opinion"]
        )
        payload["options_text"] = _format_options_text(payload["options"])
        normalized.append(payload)
    return CatalogNormalizedRows(rows=normalized)


def _normalize_mmmlu_rows(
    rows: list[CatalogRow],
    dataset: object,
) -> CatalogNormalizedRows:
    del dataset
    normalized: list[CatalogRow] = []
    for row in rows:
        payload = dict(row)
        payload["item_id"] = str(payload.get("Unnamed: 0", payload.get("item_id", "")))
        payload["question"] = str(payload.get("Question", ""))
        payload["options"] = [
            str(payload.get("A", "")),
            str(payload.get("B", "")),
            str(payload.get("C", "")),
            str(payload.get("D", "")),
        ]
        payload["expected"] = str(payload.get("Answer", "")).strip().upper()
        metadata = _metadata_dict(payload, ["Subject"])
        if "Subject" in payload:
            metadata["subject"] = str(payload["Subject"])
            metadata.pop("Subject", None)
        payload["metadata"] = metadata
        payload["options_text"] = _format_options_text(payload["options"])
        normalized.append(payload)
    return CatalogNormalizedRows(rows=normalized)


def _normalize_simpleqa_rows(
    rows: list[CatalogRow],
    dataset: object,
) -> CatalogNormalizedRows:
    del dataset
    normalized: list[CatalogRow] = []
    for row in rows:
        payload = dict(row)
        payload["item_id"] = str(payload.get("original_index", payload["item_id"]))
        payload["metadata"] = _metadata_dict(
            payload,
            ["topic", "answer_type", "multi_step", "requires_reasoning"],
        )
        payload.setdefault("expected_response", str(payload.get("answer", "")))
        normalized.append(payload)
    return CatalogNormalizedRows(rows=normalized)


def _normalize_frontierscience_rows(
    rows: list[CatalogRow],
    dataset: object,
) -> CatalogNormalizedRows:
    del dataset
    normalized: list[CatalogRow] = []
    for row in rows:
        payload = dict(row)
        payload["prompt_text"] = str(payload.get("problem", ""))
        payload["expected_response"] = str(payload.get("answer", ""))
        payload["metadata"] = {
            "subject": str(payload.get("subject", "")),
            "task_group_id": str(payload.get("task_group_id", "")),
        }
        normalized.append(payload)
    return CatalogNormalizedRows(rows=normalized)


def _normalize_healthbench_rows(
    rows: list[CatalogRow],
    dataset: object,
) -> CatalogNormalizedRows:
    del dataset
    normalized: list[CatalogRow] = []
    for row in rows:
        payload = dict(row)
        payload["item_id"] = str(payload.get("prompt_id", payload["item_id"]))
        prompt_messages = _prompt_messages_from_payload(payload)
        payload["prompt_messages"] = prompt_messages
        payload["prompt_text"] = "\n\n".join(
            f"{message['role']}: {message['content']}" for message in prompt_messages
        )
        payload["metadata"] = {
            "prompt_id": str(payload.get("prompt_id", payload["item_id"]))
        }
        ideal_data = payload.get("ideal_completions_data")
        if isinstance(ideal_data, dict):
            ideal_completion = ideal_data.get("ideal_completion")
            if isinstance(ideal_completion, str):
                payload["expected_response"] = ideal_completion
        normalized.append(payload)
    return CatalogNormalizedRows(rows=normalized)


def _normalize_lpfqa_rows(
    rows: list[CatalogRow],
    dataset: object,
) -> CatalogNormalizedRows:
    del dataset
    normalized: list[CatalogRow] = []
    for row in rows:
        payload = dict(row)
        payload["item_id"] = str(payload.get("prompt_id", payload["item_id"]))
        payload["metadata"] = _metadata_dict(payload, ["primary_domain"])
        payload["expected_response"] = _extract_lpfqa_reference_answer(
            str(payload.get("response_reference", ""))
        )
        normalized.append(payload)
    return CatalogNormalizedRows(rows=normalized)


def _normalize_phybench_rows(
    rows: list[CatalogRow],
    dataset: object,
) -> CatalogNormalizedRows:
    del dataset
    normalized: list[CatalogRow] = []
    for row in rows:
        payload = dict(row)
        payload["item_id"] = str(payload.get("id", payload.get("item_id", "")))
        payload["problem"] = str(payload.get("content", ""))
        payload["answer"] = str(payload.get("answer", "")).strip()
        payload["metadata"] = _metadata_dict(payload, ["tag"])
        normalized.append(payload)
    return CatalogNormalizedRows(rows=normalized)


def _normalize_procbench_rows(
    rows: list[CatalogRow],
    dataset: object,
) -> CatalogNormalizedRows:
    del dataset
    normalized: list[CatalogRow] = []
    for row in rows:
        payload = dict(row)
        label = payload.get("label")
        final_value: object = None
        if isinstance(label, dict):
            final_value = label.get("final")
        payload["item_id"] = str(
            payload.get("problem_name", payload.get("item_id", ""))
        )
        payload["prompt_text"] = str(payload.get("prompt", ""))
        payload["expected"] = final_value
        payload["metadata"] = _metadata_dict(payload, ["task_name", "example_name"])
        normalized.append(payload)
    return CatalogNormalizedRows(rows=normalized)


def _normalize_superchem_rows(
    rows: list[CatalogRow],
    dataset_or_slice: object,
) -> CatalogNormalizedRows:
    language = "en"
    dimensions = getattr(dataset_or_slice, "dimensions", {})
    if isinstance(dimensions, dict):
        resolved_language = dimensions.get("language")
        if isinstance(resolved_language, str) and resolved_language in {"en", "zh"}:
            language = resolved_language
    normalized: list[CatalogRow] = []
    for row in rows:
        payload = dict(row)
        options = _superchem_options(payload, language=language)
        expected = _superchem_answer(payload, language=language)
        question = str(payload.get(f"question_{language}", ""))
        prompt_text = (
            f"{question}\n\nOptions:\n{_format_options_text(options)}\n\n"
            "Return the best option letter only."
        )
        prompt_parts: list[dict[str, object]] = [{"type": "text", "text": prompt_text}]
        for image_url in _superchem_question_images(payload):
            prompt_parts.append({"type": "image_url", "image_url": image_url})
        payload["item_id"] = str(payload.get("uuid", payload.get("item_id", "")))
        payload["question"] = question
        payload["options"] = options
        payload["expected"] = expected
        payload["prompt_text"] = prompt_text
        payload["prompt_messages"] = [
            {"role": "user", "content": prompt_parts},
        ]
        metadata = _metadata_dict(payload, ["field", "question_type"])
        metadata["language"] = language
        payload["metadata"] = metadata
        payload["options_text"] = _format_options_text(options)
        normalized.append(payload)
    return CatalogNormalizedRows(rows=normalized)


def _normalize_hle_rows(
    rows: list[CatalogRow],
    dataset: DatasetSpec,
) -> CatalogNormalizedRows:
    del dataset
    normalized: list[CatalogRow] = []
    skipped = 0
    for row in rows:
        payload = dict(row)
        image = payload.get("image")
        if isinstance(image, str) and image.strip():
            skipped += 1
            continue
        payload["item_id"] = str(payload.get("id", payload["item_id"]))
        payload["metadata"] = {"text_only": "true"}
        payload["expected_response"] = _HLE_RESPONSE_TEMPLATE.format(
            explanation="Demo benchmark answer.",
            answer=str(payload.get("answer", "")),
            confidence=100,
        )
        normalized.append(payload)
    return CatalogNormalizedRows(
        rows=normalized,
        stats={"skipped_image_count": skipped},
    )


def _normalize_math_short_answer_rows(
    rows: list[CatalogRow],
    dataset: object,
) -> CatalogNormalizedRows:
    del dataset
    normalized: list[CatalogRow] = []
    for row in rows:
        payload = dict(row)
        payload["item_id"] = str(payload.get("problem_idx", payload.get("item_id", "")))
        payload["problem"] = str(payload.get("problem", ""))
        payload["answer"] = str(payload.get("answer", "")).strip()
        payload["metadata"] = _metadata_dict(
            payload,
            ["problem_idx", "problem_type", "source"],
        )
        normalized.append(payload)
    return CatalogNormalizedRows(rows=normalized)


def _normalize_imo_answerbench_rows(
    rows: list[CatalogRow],
    dataset: object,
) -> CatalogNormalizedRows:
    del dataset
    normalized: list[CatalogRow] = []
    for row in rows:
        payload = dict(row)
        payload["item_id"] = str(payload.get("Problem ID", payload.get("item_id", "")))
        payload["problem"] = str(payload.get("Problem", payload.get("problem", "")))
        payload["answer"] = str(
            payload.get("Short Answer", payload.get("answer", ""))
        ).strip()
        metadata: dict[str, str] = {}
        for source_key, target_key in (
            ("Category", "category"),
            ("Subcategory", "subcategory"),
            ("Source", "source"),
        ):
            if payload.get(source_key) is not None:
                metadata[target_key] = str(payload[source_key])
        payload["metadata"] = metadata
        normalized.append(payload)
    return CatalogNormalizedRows(rows=normalized)


def _prompt_messages_from_payload(payload: CatalogRow) -> list[CatalogPromptMessage]:
    prompt = payload.get("prompt")
    if not isinstance(prompt, list):
        return []
    messages: list[CatalogPromptMessage] = []
    for entry in prompt:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = entry.get("content")
        if isinstance(role, str) and _is_prompt_content(content):
            messages.append({"role": role, "content": content})
    return messages


def _prompt_messages_from_context(context: object) -> list[CatalogPromptMessage]:
    if isinstance(context, Mapping):
        payload = context.get("prompt_messages")
        if isinstance(payload, list):
            messages: list[CatalogPromptMessage] = []
            for entry in payload:
                if (
                    isinstance(entry, dict)
                    and isinstance(entry.get("role"), str)
                    and _is_prompt_content(entry.get("content"))
                ):
                    messages.append(
                        {
                            "role": str(entry["role"]),
                            "content": entry["content"],
                        }
                    )
            return messages
    return []


def _is_prompt_content(value: object) -> bool:
    if isinstance(value, str):
        return True
    if not isinstance(value, list):
        return False
    for part in value:
        if not isinstance(part, dict):
            return False
        part_type = part.get("type")
        if part_type == "text" and isinstance(part.get("text"), str):
            continue
        if part_type == "image_url" and isinstance(part.get("image_url"), str):
            continue
        return False
    return True


_GPQA_OPTION_RE = re.compile(r"^([a-z])\)\s*(.+)$")
_GPQA_MAPPING_RE = re.compile(r"^([A-Z])\.\s*([a-z])$")


def _parse_gpqa_diamond_question(question: str) -> tuple[str, list[str]]:
    lines = [line.strip() for line in question.splitlines() if line.strip()]
    lower_options: dict[str, str] = {}
    upper_mapping: dict[str, str] = {}
    question_lines: list[str] = []
    saw_option_block = False
    for line in lines:
        option_match = _GPQA_OPTION_RE.match(line)
        if option_match is not None:
            saw_option_block = True
            lower_options[option_match.group(1)] = option_match.group(2).strip()
            continue
        mapping_match = _GPQA_MAPPING_RE.match(line)
        if mapping_match is not None:
            upper_mapping[mapping_match.group(1)] = mapping_match.group(2)
            continue
        if not saw_option_block:
            question_lines.append(line)
    options = [
        lower_options[upper_mapping[label]]
        for label in ("A", "B", "C", "D")
        if label in upper_mapping and upper_mapping[label] in lower_options
    ]
    return "\n".join(question_lines).strip(), options


def _superchem_options(payload: CatalogRow, *, language: str) -> list[str]:
    raw_options = payload.get(f"options_{language}")
    if isinstance(raw_options, dict):
        return [
            str(raw_options[key])
            for key in sorted(raw_options)
            if isinstance(raw_options.get(key), str)
        ]
    return []


def _superchem_answer(payload: CatalogRow, *, language: str) -> str:
    raw_answer = payload.get(f"answer_{language}")
    if isinstance(raw_answer, list) and raw_answer:
        return str(raw_answer[0]).strip().upper()
    return ""


def _superchem_question_images(payload: CatalogRow) -> list[str]:
    raw_images = payload.get("question_images")
    if isinstance(raw_images, list):
        return [str(item) for item in raw_images if isinstance(item, str) and item]
    if isinstance(raw_images, dict):
        return [str(key) for key in raw_images if isinstance(key, str) and key]
    return []


def _row_metadata_value(row: CatalogRow, key: str) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and key in metadata:
        return str(metadata[key])
    return str(row.get(key, ""))


def _prepare_huggingface_dataset_for_iteration(dataset, datasets_module):
    image_type = getattr(datasets_module, "Image", None)
    features = getattr(dataset, "features", None)
    if (
        image_type is None
        or features is None
        or not hasattr(features, "items")
        or not hasattr(dataset, "cast_column")
    ):
        return dataset
    prepared = dataset
    for column_name, feature in features.items():
        if isinstance(feature, image_type) and bool(getattr(feature, "decode", False)):
            prepared = prepared.cast_column(column_name, image_type(decode=False))
    return prepared


def _should_retry_huggingface_streaming(
    dataset_id: str,
    exc: Exception,
    datasets_module,
) -> bool:
    if dataset_id != "openai/healthbench":
        return False
    dataset_generation_error = getattr(datasets_module, "DatasetGenerationError", None)
    if dataset_generation_error is not None and isinstance(
        exc, dataset_generation_error
    ):
        return True
    exceptions_module = getattr(datasets_module, "exceptions", None)
    nested_dataset_generation_error = getattr(
        exceptions_module, "DatasetGenerationError", None
    )
    if nested_dataset_generation_error is None:
        return False
    return isinstance(exc, nested_dataset_generation_error)


def _default_dataset_metadata(
    dataset_id: str,
    revision: str | None,
    *,
    datasets_module=None,
) -> JSONDict:
    datasets = datasets_module or import_optional("datasets", extra="datasets")
    try:
        builder = datasets.load_dataset_builder(dataset_id, revision=revision)
    except Exception as exc:
        message = str(exc).lower()
        return validate_json_dict(
            {
                "dataset_id": dataset_id,
                "gated": "agree to share your contact information" in message
                or "access" in message
                and "denied" in message,
                "splits": [],
                "modalities": [],
                "error": str(exc),
            },
            label="catalog dataset metadata",
        )
    features = getattr(builder.info, "features", {}) or {}
    splits = sorted(list((getattr(builder.info, "splits", {}) or {}).keys()))
    modalities = _infer_modalities_from_features(features)
    return validate_json_dict(
        {
            "dataset_id": dataset_id,
            "gated": False,
            "splits": splits,
            "modalities": modalities,
        },
        label="catalog dataset metadata",
    )


def _infer_modalities_from_features(features: object) -> list[str]:
    text_like = False
    image_like = False
    serialized = json.dumps(str(features)).lower()
    if "image" in serialized:
        image_like = True
    if "string" in serialized or "value('string')" in serialized:
        text_like = True
    modalities: list[str] = []
    if text_like:
        modalities.append("text")
    if image_like:
        modalities.append("image")
    return modalities


def _infer_field_types(rows: list[CatalogRow]) -> JSONDict:
    field_types: JSONDict = {}
    for row in rows:
        for key, value in row.items():
            if key in field_types or value is None:
                continue
            field_types[key] = type(value).__name__
    return validate_json_dict(field_types, label="catalog dataset fields")


def _suggest_dataset_wiring(rows: list[CatalogRow]) -> JSONDict:
    if not rows:
        return validate_json_dict(
            {
                "suggested_prompt_field": None,
                "suggested_answer_field": None,
                "suggested_item_id_field": None,
                "suggested_metadata_keys": [],
            },
            label="catalog dataset suggestions",
        )
    row = rows[0]
    field_names = {str(key) for key in row}
    prompt_field = _first_present_field(
        field_names,
        ["problem", "question", "prompt", "Problem"],
    )
    answer_field = _first_present_field(
        field_names,
        ["answer", "expected", "Short Answer", "answer_letter"],
    )
    item_id_field = _first_present_field(
        field_names,
        ["problem_idx", "Problem ID", "item_id", "id", "question_id", "prompt_id"],
    )
    ignored = {
        prompt_field,
        answer_field,
        item_id_field,
        "item_id",
        "__index_level_0__",
    }
    metadata_keys: list[str] = []
    for key, value in row.items():
        normalized_key = str(key)
        if normalized_key in ignored or value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata_keys.append(normalized_key)
            continue
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            metadata_keys.append(normalized_key)
    return validate_json_dict(
        {
            "suggested_prompt_field": prompt_field,
            "suggested_answer_field": answer_field,
            "suggested_item_id_field": item_id_field,
            "suggested_metadata_keys": metadata_keys,
        },
        label="catalog dataset suggestions",
    )


def _first_present_field(field_names: set[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in field_names:
            return candidate
    return None


def _json_rows(rows: list[CatalogRow], *, label: str) -> list[JSONDict]:
    return [
        validate_json_dict(dict(row), label=f"{label}[{index}]")
        for index, row in enumerate(rows)
    ]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]
