"""Helpers for working with the openlifescienceai/medmcqa dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Sequence

from pydantic import BaseModel, Field, field_validator

_DATASET_NAME = "openlifescienceai/medmcqa"
_CHOICE_LABELS = ["A", "B", "C", "D"]


class MedMcqaSample(BaseModel):
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
            return [str(v) for _, v in sorted(value.items())]
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        raise TypeError("choices must be a sequence or mapping")

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


def load_medmcqa(
    *,
    split: str = "test",
    limit: int | None = None,
    source: str = "huggingface",
    data_dir: str | Path | None = None,
    subset: str | None = None,
) -> List[MedMcqaSample]:
    """Load MedMCQA samples from Hugging Face or a local directory."""

    if source not in {"huggingface", "local"}:
        raise ValueError(
            f"Unsupported source '{source}'. Expected one of: 'huggingface', 'local'."
        )

    if source == "huggingface":
        rows = _load_from_huggingface(split=split, subset=subset)
    else:
        if data_dir is None:
            raise ValueError(
                "data_dir must be provided when source='local'. "
                "Pass dataset.data_dir in configs or --data-dir on the CLI."
            )
        rows = _load_from_local(Path(data_dir))

    samples: list[MedMcqaSample] = []
    for index, row in enumerate(rows, start=1):
        sample = _row_to_sample(row, index=index)
        samples.append(sample)
        if limit is not None and len(samples) >= limit:
            break
    return samples


def _row_to_sample(row: dict[str, Any], *, index: int) -> MedMcqaSample:
    unique_id = (
        row.get("id")
        or row.get("unique_id")
        or f"medmcqa-{index:05d}"
    )
    question = row.get("question") or ""
    
    # MedMCQA has 'opa', 'opb', 'opc', 'opd'
    choices = [
        str(row.get("opa") or ""),
        str(row.get("opb") or ""),
        str(row.get("opc") or ""),
        str(row.get("opd") or ""),
    ]
    
    # Answer is an integer 0-3 (sometimes 1-4 depending on version, but usually 0-3 in HF)
    # Let's check. HF dataset viewer says 'cop': 1 (meaning option B if 1-based, or B if 0-based?)
    # Usually it's 0-3 or 1-4.
    # Checking dataset info: "cop: Correct option (1-4)"
    # So we need to map 1->A, 2->B, 3->C, 4->D
    
    cop = row.get("cop")
    answer = ""
    if cop is not None:
        try:
            cop_int = int(cop)
            if 1 <= cop_int <= 4:
                answer = _CHOICE_LABELS[cop_int - 1]
        except (ValueError, TypeError):
            pass

    subject = row.get("subject_name") or "medicine"

    metadata_keys = {
        "question", "opa", "opb", "opc", "opd", "cop", "subject_name", "id"
    }
    metadata = {key: value for key, value in row.items() if key not in metadata_keys}
    
    return MedMcqaSample(
        unique_id=str(unique_id),
        question=str(question),
        choices=choices,
        answer=answer,
        subject=str(subject),
        metadata=metadata,
    )


def _load_from_huggingface(*, split: str, subset: str | None) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "datasets is required to load MedMCQA from Hugging Face. Install it via `uv pip install '.[hf]'`."
        ) from exc

    dataset = load_dataset(_DATASET_NAME, subset, split=split)
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


__all__ = ["MedMcqaSample", "load_medmcqa"]
