"""Dataset source specifications and materialization helpers."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import cast

from pydantic import Field

from themis.core.base import HashableModel, JSONValue
from themis.core.dataset_inputs import (
    dataset_from_huggingface,
    dataset_from_jsonl,
)
from themis.core.models import Case, Dataset


class DatasetSourceSpec(HashableModel):
    """Declarative dataset source used for identity and rematerialization."""

    target: str
    dataset_id: str
    revision: str | None = None
    source_id: str | None = None
    source_revision: str | None = None
    source_fingerprint: str | None = None
    kwargs: dict[str, JSONValue] = Field(default_factory=dict)
    transform_kwargs: dict[str, JSONValue] = Field(default_factory=dict)
    provenance_metadata: dict[str, str] = Field(default_factory=dict)


def inline_dataset_source(dataset: Dataset) -> DatasetSourceSpec:
    """Convert an in-memory dataset into a reproducible inline source spec."""

    return DatasetSourceSpec(
        target="inline",
        dataset_id=dataset.dataset_id,
        revision=dataset.revision,
        source_id=dataset.dataset_id,
        source_revision=dataset.revision,
        source_fingerprint=dataset.compute_hash(),
        kwargs={
            "cases": [case.model_dump(mode="json") for case in dataset.cases],
            "metadata": cast(dict[str, JSONValue], dict(dataset.metadata)),
        },
    )


def jsonl_dataset_source(
    path: str | Path,
    *,
    dataset_id: str | None = None,
    revision: str | None = None,
) -> DatasetSourceSpec:
    source_path = Path(path).expanduser().resolve()
    return DatasetSourceSpec(
        target="jsonl",
        dataset_id=dataset_id or source_path.stem,
        revision=revision,
        source_id=str(source_path),
        source_revision=revision,
        source_fingerprint=_file_fingerprint(source_path),
        kwargs={"path": str(source_path)},
        provenance_metadata={"path": str(source_path)},
    )


def huggingface_dataset_source(
    *,
    dataset_name: str,
    split: str,
    input_field: str,
    expected_output_field: str | None = None,
    case_id_field: str | None = None,
    config_name: str | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
    dataset_id: str | None = None,
) -> DatasetSourceSpec:
    kwargs: dict[str, JSONValue] = {
        "dataset_name": dataset_name,
        "split": split,
        "input_field": input_field,
    }
    if expected_output_field is not None:
        kwargs["expected_output_field"] = expected_output_field
    if case_id_field is not None:
        kwargs["case_id_field"] = case_id_field
    if config_name is not None:
        kwargs["config_name"] = config_name
    if revision is not None:
        kwargs["revision"] = revision
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    source_id = dataset_name if config_name is None else f"{dataset_name}:{config_name}"
    return DatasetSourceSpec(
        target="huggingface",
        dataset_id=dataset_id or dataset_name,
        revision=revision or split,
        source_id=source_id,
        source_revision=revision or split,
        source_fingerprint=_hash_payload(kwargs),
        kwargs=kwargs,
        provenance_metadata={"dataset_name": dataset_name, "split": split},
    )


def catalog_dataset_source(
    *,
    benchmark_name: str,
    dataset_id: str,
    revision: str | None = None,
    provenance_metadata: dict[str, str] | None = None,
) -> DatasetSourceSpec:
    metadata = {"benchmark_name": benchmark_name, **dict(provenance_metadata or {})}
    return DatasetSourceSpec(
        target="catalog",
        dataset_id=dataset_id,
        revision=revision,
        source_id=benchmark_name,
        source_revision=revision,
        source_fingerprint=_hash_payload(
            {
                "benchmark_name": benchmark_name,
                "dataset_id": dataset_id,
                "revision": revision,
            }
        ),
        kwargs={"name": benchmark_name},
        provenance_metadata=metadata,
    )


def resolved_source_id(spec: DatasetSourceSpec) -> str:
    return spec.source_id or spec.dataset_id


def resolved_source_revision(spec: DatasetSourceSpec) -> str | None:
    return spec.source_revision or spec.revision


def resolved_source_fingerprint(spec: DatasetSourceSpec) -> str:
    if spec.source_fingerprint is not None:
        return spec.source_fingerprint
    return _hash_payload(
        {
            "target": spec.target,
            "dataset_id": spec.dataset_id,
            "revision": spec.revision,
            "source_id": resolved_source_id(spec),
            "source_revision": resolved_source_revision(spec),
            "transform_kwargs": spec.transform_kwargs,
            "provenance_metadata": cast(
                dict[str, JSONValue], dict(spec.provenance_metadata)
            ),
        }
    )


def materialize_dataset_source(spec: DatasetSourceSpec) -> Dataset:
    """Materialize a dataset from a declarative source spec."""

    if spec.target == "inline":
        cases_payload = cast(list[dict[str, JSONValue]], spec.kwargs.get("cases", []))
        cases = [Case.model_validate(payload) for payload in cases_payload]
        raw_metadata = spec.kwargs.get("metadata", {})
        metadata = (
            cast(dict[str, str], dict(raw_metadata))
            if isinstance(raw_metadata, dict)
            else {}
        )
        return Dataset(
            dataset_id=spec.dataset_id,
            revision=spec.revision,
            cases=cases,
            metadata=metadata,
        )

    if spec.target == "jsonl":
        return dataset_from_jsonl(
            Path(str(spec.kwargs["path"])),
            dataset_id=spec.dataset_id,
            revision=spec.revision,
        )

    if spec.target == "huggingface":
        return dataset_from_huggingface(
            dataset_name=str(spec.kwargs.get("dataset_name", spec.dataset_id)),
            config_name=cast(str | None, spec.kwargs.get("config_name")),
            split=str(spec.kwargs["split"]),
            revision=cast(str | None, spec.kwargs.get("revision")),
            cache_dir=cast(str | None, spec.kwargs.get("cache_dir")),
            input_field=str(spec.kwargs["input_field"]),
            expected_output_field=cast(
                str | None, spec.kwargs.get("expected_output_field")
            ),
            case_id_field=cast(str | None, spec.kwargs.get("case_id_field")),
        ).model_copy(
            update={
                "dataset_id": spec.dataset_id,
                "revision": spec.revision or cast(str | None, spec.kwargs.get("split")),
                "metadata": {
                    "source": "huggingface",
                    **spec.provenance_metadata,
                },
            }
        )

    if spec.target == "catalog":
        from themis.catalog.benchmarks import (
            load_benchmark as load_benchmark_definition,
        )

        benchmark_name = str(spec.kwargs["name"])
        definition = load_benchmark_definition(benchmark_name)
        dataset = definition.materialize_dataset()
        return dataset.model_copy(
            update={
                "dataset_id": spec.dataset_id or dataset.dataset_id,
                "revision": spec.revision or dataset.revision,
                "metadata": {**dataset.metadata, **spec.provenance_metadata},
            }
        )

    raise ValueError(f"Unsupported dataset source target: {spec.target}")


def materialize_dataset_sources(specs: list[DatasetSourceSpec]) -> list[Dataset]:
    """Materialize all configured dataset sources."""

    return [materialize_dataset_source(spec) for spec in specs]


def dataset_materialization_receipt(
    spec: DatasetSourceSpec, dataset: Dataset
) -> dict[str, JSONValue]:
    """Capture the resolved source and materialization receipt for one dataset."""

    receipt: dict[str, JSONValue] = {
        "target": spec.target,
        "source_id": resolved_source_id(spec),
        "source_revision": resolved_source_revision(spec),
        "source_fingerprint": resolved_source_fingerprint(spec),
        "dataset_fingerprint": dataset.compute_hash(),
        "case_count": len(dataset.cases),
        "dataset_metadata": cast(dict[str, JSONValue], dict(dataset.metadata)),
        "provenance_metadata": cast(
            dict[str, JSONValue], dict(spec.provenance_metadata)
        ),
    }
    if spec.target == "jsonl":
        receipt["path"] = str(spec.kwargs["path"])
    if spec.target == "huggingface":
        receipt["dataset_name"] = str(spec.kwargs.get("dataset_name", spec.dataset_id))
        receipt["split"] = str(spec.kwargs["split"])
        if "config_name" in spec.kwargs:
            receipt["config_name"] = spec.kwargs["config_name"]
    if spec.target == "catalog":
        receipt["benchmark_name"] = str(spec.kwargs["name"])
    return receipt


def _hash_payload(payload: dict[str, JSONValue]) -> str:
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_fingerprint(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
