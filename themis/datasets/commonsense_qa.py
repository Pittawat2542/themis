"""Helpers for working with the tau/commonsense_qa dataset."""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Sequence

from pydantic import BaseModel, Field, ValidationInfo, field_validator

_DATASET_NAME = "tau/commonsense_qa"
_CHOICE_LABELS = tuple(string.ascii_uppercase)


class CommonsenseQaSample(BaseModel):
    unique_id: str
    question: str
    choices: list[str]
    answer: str
    concept: str = Field(default="")
    metadata: dict[str, Any] = Field(default_factory=dict)
    choice_labels: list[str] = Field(default_factory=list)

    @field_validator("choices", mode="before")
    @classmethod
    def _ensure_choices(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, dict):
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
            "concept": self.concept,
            "metadata": dict(self.metadata),
        }


def load_commonsense_qa(
    *,
    split: str = "validation", # Test set usually has no labels
    limit: int | None = None,
    source: str = "huggingface",
    data_dir: str | Path | None = None,
) -> List[CommonsenseQaSample]:
    """Load CommonsenseQA samples from Hugging Face or a local directory."""

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

    samples: list[CommonsenseQaSample] = []
    for index, row in enumerate(rows, start=1):
        sample = _row_to_sample(row, index=index)
        samples.append(sample)
        if limit is not None and len(samples) >= limit:
            break
    return samples


def _row_to_sample(row: dict[str, Any], *, index: int) -> CommonsenseQaSample:
    unique_id = (
        row.get("id")
        or row.get("unique_id")
        or f"csqa-{index:05d}"
    )
    question = row.get("question") or ""
    
    # CommonsenseQA format:
    # choices: {'label': ['A', 'B', ...], 'text': ['text1', 'text2', ...]}
    # answerKey: 'A'
    
    choices_data = row.get("choices") or {}
    choices = []
    choice_labels = []
    
    if isinstance(choices_data, dict):
        labels = choices_data.get("label") or []
        texts = choices_data.get("text") or []
        
        # Zip and sort by label
        zipped = sorted(zip(labels, texts), key=lambda x: x[0])
        for label, text in zipped:
            choices.append(str(text))
            choice_labels.append(str(label))
            
    answer = str(row.get("answerKey") or "")
    concept = str(row.get("question_concept") or "")

    metadata_keys = {
        "question", "choices", "answerKey", "question_concept", "id"
    }
    metadata = {key: value for key, value in row.items() if key not in metadata_keys}
    
    return CommonsenseQaSample(
        unique_id=str(unique_id),
        question=str(question),
        choices=choices,
        choice_labels=choice_labels,
        answer=answer,
        concept=concept,
        metadata=metadata,
    )


def _load_from_huggingface(*, split: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "datasets is required to load CommonsenseQA from Hugging Face. Install it via `uv pip install '.[hf]'`."
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


__all__ = ["CommonsenseQaSample", "load_commonsense_qa"]
