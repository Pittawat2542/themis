"""Utilities for loading experiment configs via Hydra/OmegaConf."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from omegaconf import OmegaConf

from . import schema


def load_experiment_config(
    config_path: Path,
    overrides: Iterable[str] | None = None,
) -> schema.ExperimentConfig:
    """Load and validate an experiment config file with optional overrides."""

    base = OmegaConf.structured(schema.ExperimentConfig)
    file_conf = OmegaConf.load(config_path)
    merged = OmegaConf.merge(base, file_conf)

    if overrides:
        override_conf = OmegaConf.from_dotlist(list(overrides))
        merged = OmegaConf.merge(merged, override_conf)

    return OmegaConf.to_object(merged)
