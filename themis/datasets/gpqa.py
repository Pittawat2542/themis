"""Helpers for working with the math-ai/gpqa dataset."""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Sequence

from pydantic import BaseModel, Field, ValidationInfo, field_validator

_DATASET_NAME = "math-ai/gpqa"
_CHOICE_LABELS = tuple(string.ascii_uppercase)


class GpqaSample(BaseModel):
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
            "subject": self.subject,
            "metadata": dict(self.metadata),
        }


def load_gpqa(
    *,
    split: str = "test",
    limit: int | None = None,
    source: str = "huggingface",
    data_dir: str | Path | None = None,
    subset: str = "gpqa_diamond",
) -> List[GpqaSample]:
    """Load GPQA samples from Hugging Face or a local directory."""

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

    samples: list[GpqaSample] = []
    for index, row in enumerate(rows, start=1):
        sample = _row_to_sample(row, index=index)
        samples.append(sample)
        if limit is not None and len(samples) >= limit:
            break
    return samples


def _row_to_sample(row: dict[str, Any], *, index: int) -> GpqaSample:
    unique_id = (
        row.get("id")
        or row.get("unique_id")
        or f"gpqa-{index:05d}"
    )
    question = row.get("Question") or row.get("question") or ""
    
    # GPQA usually has 'Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3'
    # We need to shuffle them or just present them. For simplicity, we'll just collect them.
    # However, standard GPQA format in HF might be different.
    # Let's assume the HF format: 'Question', 'Correct Answer', 'Incorrect Answer 1', ...
    
    correct_answer = row.get("Correct Answer") or row.get("correct_answer") or ""
    incorrect_answers = []
    for i in range(1, 4):
        inc = row.get(f"Incorrect Answer {i}") or row.get(f"incorrect_answer_{i}")
        if inc:
            incorrect_answers.append(str(inc))
            
    # If choices are already present (e.g. processed version), use them
    choices = row.get("choices") or row.get("options")
    if not choices:
        # We need to form choices. For now, let's put correct answer first (should be shuffled in real eval)
        # But wait, if we put correct answer first always, the model might learn.
        # Ideally, we should shuffle. But to keep it deterministic for now without extra deps,
        # let's just list them. The evaluator should handle permutation if needed, 
        # or we should shuffle here if we want to present them as A, B, C, D.
        # For this implementation, I will just append them.
        choices = [correct_answer] + incorrect_answers
        # Note: In a real evaluation pipeline, you'd want to shuffle these and track the correct index.
        # But since we are just loading, we'll leave it as is.
        # Actually, let's sort them to be deterministic if we can't shuffle safely.
        choices.sort()
        
    # Determine the answer label
    try:
        answer_idx = choices.index(correct_answer)
        answer = _CHOICE_LABELS[answer_idx]
    except ValueError:
        answer = "" # Should not happen if correct_answer is in choices

    metadata_keys = {
        "Question", "question", "Correct Answer", "correct_answer", 
        "Incorrect Answer 1", "incorrect_answer_1",
        "Incorrect Answer 2", "incorrect_answer_2",
        "Incorrect Answer 3", "incorrect_answer_3",
        "choices", "options"
    }
    metadata = {key: value for key, value in row.items() if key not in metadata_keys}
    
    return GpqaSample(
        unique_id=str(unique_id),
        question=str(question),
        choices=choices,
        answer=answer,
        metadata=metadata,
    )


def _load_from_huggingface(*, split: str, subset: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "datasets is required to load GPQA from Hugging Face. Install it via `uv pip install '.[hf]'`."
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


__all__ = ["GpqaSample", "load_gpqa"]
