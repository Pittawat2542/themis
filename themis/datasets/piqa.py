"""Helpers for working with the ybisk/piqa dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Sequence

from pydantic import BaseModel, Field, field_validator

_DATASET_NAME = "ybisk/piqa"


class PiqaSample(BaseModel):
    unique_id: str
    goal: str
    choices: list[str]
    answer: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("choices", mode="before")
    @classmethod
    def _ensure_choices(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        raise TypeError("choices must be a sequence")

    def to_generation_example(self) -> dict[str, Any]:
        return {
            "unique_id": self.unique_id,
            "goal": self.goal,
            "choices": list(self.choices),
            "answer": self.answer,
            "metadata": dict(self.metadata),
        }


def load_piqa(
    *,
    split: str = "validation", # Test set usually has no labels
    limit: int | None = None,
    source: str = "huggingface",
    data_dir: str | Path | None = None,
) -> List[PiqaSample]:
    """Load PIQA samples from Hugging Face or a local directory."""

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

    samples: list[PiqaSample] = []
    for index, row in enumerate(rows, start=1):
        sample = _row_to_sample(row, index=index)
        samples.append(sample)
        if limit is not None and len(samples) >= limit:
            break
    return samples


def _row_to_sample(row: dict[str, Any], *, index: int) -> PiqaSample:
    unique_id = (
        row.get("id")
        or row.get("unique_id")
        or f"piqa-{index:05d}"
    )
    goal = row.get("goal") or ""
    
    # PIQA has 'sol1', 'sol2'
    choices = [
        str(row.get("sol1") or ""),
        str(row.get("sol2") or ""),
    ]
    
    # label is integer 0 or 1
    label = row.get("label")
    answer = ""
    if label is not None:
        try:
            label_int = int(label)
            if 0 <= label_int < len(choices):
                answer = choices[label_int]
        except (ValueError, TypeError):
            pass

    metadata_keys = {
        "goal", "sol1", "sol2", "label", "id"
    }
    metadata = {key: value for key, value in row.items() if key not in metadata_keys}
    
    return PiqaSample(
        unique_id=str(unique_id),
        goal=str(goal),
        choices=choices,
        answer=answer,
        metadata=metadata,
    )


def _load_from_huggingface(*, split: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "datasets is required to load PIQA from Hugging Face. Install it via `uv pip install '.[hf]'`."
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


__all__ = ["PiqaSample", "load_piqa"]
