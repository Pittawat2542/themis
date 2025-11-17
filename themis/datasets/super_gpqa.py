"""Helpers for working with the m-a-p/SuperGPQA dataset."""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Sequence

from pydantic import BaseModel, Field, ValidationInfo, field_validator

_DATASET_NAME = "m-a-p/SuperGPQA"
_CHOICE_LABELS = tuple(string.ascii_uppercase)


def _index_to_label(index: int) -> str:
    if 0 <= index < len(_CHOICE_LABELS):
        return _CHOICE_LABELS[index]
    return str(index)


def _normalize_answer(value: Any, *, total_choices: int | None = None) -> str:
    if isinstance(value, int):
        return _index_to_label(value)
    if isinstance(value, float):
        as_int = int(value)
        if as_int == value:
            return _index_to_label(as_int)
    text = str(value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith("option "):
        text = text.split(" ", 1)[-1]
    if lowered.startswith("choice "):
        text = text.split(" ", 1)[-1]
    if text.isdigit():
        index = int(text)
        if total_choices is None or 0 <= index < total_choices:
            return _index_to_label(index)
    text = text.strip().rstrip(".")
    if len(text) == 1 and text.isalpha():
        return text.upper()
    if total_choices is not None:
        mapping = {str(idx): _index_to_label(idx) for idx in range(total_choices)}
        normalized = mapping.get(text)
        if normalized:
            return normalized
    return text


class SuperGpqaSample(BaseModel):
    unique_id: str
    question: str
    choices: list[str]
    answer: str
    subject: str = Field(default="unknown")
    metadata: dict[str, Any] = Field(default_factory=dict)
    choice_labels: list[str] = Field(default_factory=list)

    @field_validator("choices", mode="before")
    @classmethod
    def _ensure_choices(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, dict):
            # Sort by key to keep deterministic order
            return [str(v) for _, v in sorted(value.items())]
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        raise TypeError("choices must be a sequence or mapping")

    @field_validator("choice_labels", mode="before")
    @classmethod
    def _build_choice_labels(cls, value: Any, info: ValidationInfo) -> list[str]:
        if value:
            return [str(item) for item in value]
        choices = info.data.get("choices") if hasattr(info, "data") else None
        total = len(choices) if isinstance(choices, list) else 0
        return [*_CHOICE_LABELS[:total]]

    @field_validator("answer", mode="before")
    @classmethod
    def _normalize_answer_field(cls, value: Any, info: ValidationInfo) -> str:
        choices = info.data.get("choices") if hasattr(info, "data") else None
        total = len(choices) if isinstance(choices, list) else None
        return _normalize_answer(value, total_choices=total)

    def to_generation_example(self) -> dict[str, Any]:
        effective_labels = (
            list(self.choice_labels)
            if self.choice_labels
            else list(_CHOICE_LABELS[: len(self.choices)])
        )
        return {
            "unique_id": self.unique_id,
            "question": self.question,
            "choices": list(self.choices),
            "choice_labels": effective_labels,
            "answer": self.answer,
            "subject": self.subject,
            "metadata": dict(self.metadata),
        }


def load_super_gpqa(
    *,
    split: str = "test",
    limit: int | None = None,
    source: str = "huggingface",
    data_dir: str | Path | None = None,
    subjects: Sequence[str] | None = None,
) -> List[SuperGpqaSample]:
    """Load SuperGPQA samples from Hugging Face or a local directory."""

    if source not in {"huggingface", "local"}:
        raise ValueError(
            f"Unsupported source '{source}'. Expected one of: 'huggingface', 'local'."
        )

    if source == "huggingface":
        rows = _load_from_huggingface(split=split)
    else:
        if data_dir is None:
            raise ValueError(
                "data_dir must be provided when source='local'. "
                "Pass dataset.data_dir in configs or --data-dir on the CLI."
            )
        rows = _load_from_local(Path(data_dir))

    samples: list[SuperGpqaSample] = []
    selected_subjects = {s.lower() for s in subjects} if subjects else None
    for index, row in enumerate(rows, start=1):
        subject = _extract_subject(row)
        if selected_subjects and subject.lower() not in selected_subjects:
            continue
        sample = _row_to_sample(row, index=index, subject=subject)
        samples.append(sample)
        if limit is not None and len(samples) >= limit:
            break
    return samples


def _extract_subject(row: dict[str, Any]) -> str:
    for key in ("subject", "category", "field", "domain", "track"):
        value = row.get(key)
        if value:
            return str(value)
    return "unknown"


def _row_to_sample(row: dict[str, Any], *, index: int, subject: str) -> SuperGpqaSample:
    unique_id = (
        row.get("id")
        or row.get("question_id")
        or row.get("unique_id")
        or f"supergpqa-{index:05d}"
    )
    question = row.get("question") or row.get("Question") or row.get("prompt") or ""
    choices = _extract_choices(row)
    answer = (
        row.get("answer")
        or row.get("Answer")
        or row.get("correct_answer")
        or row.get("correct")
        or ""
    )
    metadata_keys = {
        "question",
        "Question",
        "prompt",
        "choices",
        "options",
        "answer",
        "Answer",
        "correct_answer",
        "correct",
    }
    metadata = {key: value for key, value in row.items() if key not in metadata_keys}
    sample = SuperGpqaSample.model_validate(
        {
            "unique_id": str(unique_id),
            "question": str(question),
            "choices": choices,
            "answer": answer,
            "subject": str(subject),
            "metadata": metadata,
        }
    )
    return sample


def _extract_choices(row: dict[str, Any]) -> list[str]:
    candidates = row.get("choices") or row.get("options") or row.get("Choices")
    if isinstance(candidates, dict):
        return [str(value) for _, value in sorted(candidates.items())]
    if isinstance(candidates, (list, tuple)):
        return [str(item) for item in candidates]

    choice_keys = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "choice_a",
        "choice_b",
        "choice_c",
        "choice_d",
        "choice_e",
        "choice_f",
        "option_a",
        "option_b",
        "option_c",
        "option_d",
        "option_e",
        "option_f",
    ]
    collected: list[str] = []
    for key in choice_keys:
        if key in row:
            collected.append(str(row[key]))
    if collected:
        return collected
    return []


def _load_from_huggingface(*, split: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "datasets is required to load SuperGPQA from Hugging Face. Install it via `uv pip install '.[hf]'`."
        ) from exc

    dataset = load_dataset(_DATASET_NAME, split=split)
    for row in dataset:
        yield dict(row)


def _load_from_local(root: Path) -> Iterator[dict[str, Any]]:
    if not root.exists():
        raise FileNotFoundError(f"Local dataset directory not found: {root}")

    for path in root.rglob("*"):
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as handle:
                row = json.load(handle)
                row.setdefault("id", path.stem)
                yield row
        elif path.suffix.lower() in {".jsonl", ".ndjson"}:
            with path.open("r", encoding="utf-8") as handle:
                for line_num, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    row.setdefault("id", f"{path.stem}-{line_num}")
                    yield row


__all__ = ["SuperGpqaSample", "load_super_gpqa"]
