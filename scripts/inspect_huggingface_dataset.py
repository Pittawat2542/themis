#!/usr/bin/env python3
"""Inspect a Hugging Face dataset and optionally save the summary as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from themis.catalog import inspect_huggingface_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a Hugging Face dataset for Themis catalog benchmark wiring."
        )
    )
    parser.add_argument(
        "dataset_specs",
        nargs="+",
        help=(
            "One or more dataset repo ids, optionally as dataset_id:split or "
            "dataset_id@config_name:split."
        ),
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to inspect. Defaults to 'test'.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional dataset revision, tag, or commit.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Maximum number of sample rows to include. Defaults to 3.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to write the JSON summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summaries: list[dict[str, object]] = []
    for dataset_spec in args.dataset_specs:
        dataset_id, config_name, split = _parse_dataset_spec(
            dataset_spec, default_split=args.split
        )
        summaries.append(
            inspect_huggingface_dataset(
                dataset_id,
                config_name=config_name,
                split=split,
                revision=args.revision,
                max_samples=args.samples,
            )
        )
    payload: object = summaries[0] if len(summaries) == 1 else summaries
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)

    if args.save:
        destination = Path(args.save)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n")
    return 0


def _parse_dataset_spec(
    spec: str, *, default_split: str
) -> tuple[str, str | None, str]:
    dataset_ref = spec
    split = default_split
    if ":" in spec:
        dataset_ref, split = spec.rsplit(":", maxsplit=1)
    if not dataset_ref or not split:
        raise ValueError(
            "Dataset specs must use dataset_id, dataset_id:split, or dataset_id@config:split format."
        )
    if "@" not in dataset_ref:
        return dataset_ref, None, split
    dataset_id, config_name = dataset_ref.rsplit("@", maxsplit=1)
    if not dataset_id or not config_name:
        raise ValueError(
            "Dataset specs with configs must use dataset_id@config or dataset_id@config:split."
        )
    return dataset_id, config_name, split


if __name__ == "__main__":
    raise SystemExit(main())
