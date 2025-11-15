"""Helpers for competition-style math benchmarks from Hugging Face."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Sequence

from pydantic import BaseModel, Field, field_validator


class CompetitionMathSample(BaseModel):
    unique_id: str
    problem: str
    solution: str
    answer: str
    subject: str = Field(default="unknown")
    level: str | int = Field(default="unknown")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def _ensure_metadata(cls, value: Any) -> dict[str, Any]:
        return dict(value or {})

    @field_validator("level", mode="before")
    @classmethod
    def _normalize_level(cls, value: Any) -> str | int:
        if value is None or value == "":
            return "unknown"
        try:
            return int(value)
        except (TypeError, ValueError):
            return str(value)

    def to_generation_example(self) -> dict[str, Any]:
        payload = {
            "unique_id": self.unique_id,
            "problem": self.problem,
            "solution": self.solution,
            "answer": self.answer,
            "subject": self.subject,
            "level": self.level,
        }
        payload.update(self.metadata)
        return payload


def load_competition_math(
    *,
    dataset: str,
    split: str = "test",
    limit: int | None = None,
    source: str = "huggingface",
    data_dir: str | Path | None = None,
    subjects: Sequence[str] | None = None,
    subset: str | None = None,
) -> List[CompetitionMathSample]:
    """Load math competition samples from Hugging Face or a local directory."""

    if source not in {"huggingface", "local"}:
        raise ValueError(
            f"Unsupported source '{source}'. Expected one of: 'huggingface', 'local'."
        )

    if source == "huggingface":
        rows = _load_from_huggingface(dataset=dataset, split=split, subset=subset)
    else:
        if data_dir is None:
            raise ValueError(
                "data_dir must be provided when source='local'. "
                "Pass dataset.data_dir in configs or --data-dir on the CLI."
            )
        rows = _load_from_local(Path(data_dir))

    samples: list[CompetitionMathSample] = []
    selected_subjects = {s.lower() for s in subjects} if subjects else None
    for index, row in enumerate(rows, start=1):
        subject = _extract_subject(row) or "unknown"
        if selected_subjects and subject.lower() not in selected_subjects:
            continue
        sample = _row_to_sample(
            row=row,
            index=index,
            dataset=dataset,
            subject=subject,
        )
        samples.append(sample)
        if limit is not None and len(samples) >= limit:
            break
    return samples


def _load_from_huggingface(
    *, dataset: str, split: str, subset: str | None
) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "datasets is required to load competition math benchmarks from Hugging Face. "
            "Install it via `uv pip install '.[hf]'`."
        ) from exc

    if subset:
        hf_dataset = load_dataset(dataset, subset, split=split)
    else:
        hf_dataset = load_dataset(dataset, split=split)
    for row in hf_dataset:
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


def _extract_subject(row: dict[str, Any]) -> str | None:
    for key in (
        "subject",
        "category",
        "topic",
        "domain",
        "contest",
        "source",
        "level",
    ):
        value = row.get(key)
        if value:
            return str(value)
    return None


def _extract_problem(row: dict[str, Any]) -> str:
    for key in (
        "problem",
        "problem_text",
        "problem_statement",
        "question",
        "prompt",
        "problem_markdown",
    ):
        value = row.get(key)
        if value:
            return str(value)
    return ""


def _extract_solution(row: dict[str, Any]) -> str:
    for key in (
        "solution",
        "solution_text",
        "solution_markdown",
        "answer_explanation",
        "worked_solution",
        "reasoning",
    ):
        value = row.get(key)
        if value:
            return str(value)
    return ""


def _extract_answer(row: dict[str, Any]) -> str:
    for key in (
        "answer",
        "final_answer",
        "ground_truth",
        "answer_text",
        "answer_value",
    ):
        value = row.get(key)
        if value is not None:
            return str(value).strip()
    return ""


def _extract_level(row: dict[str, Any]) -> str | int:
    for key in ("difficulty", "level", "year"):
        value = row.get(key)
        if value:
            return value
    return "unknown"


def _row_to_sample(
    *,
    row: dict[str, Any],
    index: int,
    dataset: str,
    subject: str,
) -> CompetitionMathSample:
    unique_id = (
        row.get("id")
        or row.get("problem_id")
        or row.get("unique_id")
        or f"{dataset.replace('/', '-')}-{index:05d}"
    )
    problem = _extract_problem(row)
    solution = _extract_solution(row)
    answer = _extract_answer(row)
    level = _extract_level(row)
    core_keys = {
        "id",
        "problem_id",
        "unique_id",
        "problem",
        "problem_text",
        "problem_statement",
        "question",
        "prompt",
        "problem_markdown",
        "solution",
        "solution_text",
        "solution_markdown",
        "answer_explanation",
        "worked_solution",
        "reasoning",
        "answer",
        "final_answer",
        "ground_truth",
        "answer_text",
        "answer_value",
        "difficulty",
        "level",
        "year",
        "subject",
        "category",
        "topic",
        "domain",
        "contest",
        "source",
    }
    metadata = {key: value for key, value in row.items() if key not in core_keys}
    sample = CompetitionMathSample.model_validate(
        {
            "unique_id": str(unique_id),
            "problem": problem,
            "solution": solution,
            "answer": answer,
            "subject": str(subject),
            "level": level,
            "metadata": metadata,
        }
    )
    return sample


__all__ = ["CompetitionMathSample", "load_competition_math"]
