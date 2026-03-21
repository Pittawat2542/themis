#!/usr/bin/env python3
"""Inspect a Hugging Face dataset and optionally save the summary as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from themis.starter_catalog import inspect_huggingface_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a Hugging Face dataset for Themis starter benchmark wiring."
        )
    )
    parser.add_argument("dataset_id", help="Hugging Face dataset repo id.")
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
    summary = inspect_huggingface_dataset(
        args.dataset_id,
        split=args.split,
        revision=args.revision,
        max_samples=args.samples,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)

    if args.save:
        destination = Path(args.save)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
