"""Row normalizers for catalog benchmark datasets."""

from __future__ import annotations

import re

from themis.specs.foundational import DatasetSpec

from ._prompts import _HLE_RESPONSE_TEMPLATE, _prompt_messages_from_payload
from ._types import CatalogNormalizedRows, CatalogRow


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
