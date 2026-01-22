"""Preset configurations for common benchmarks and models.

This module provides automatic configuration for popular benchmarks,
eliminating the need for manual setup of prompts, metrics, and extractors.
"""

from themis.presets.benchmarks import get_benchmark_preset, list_benchmarks
from themis.presets.models import parse_model_name

__all__ = ["get_benchmark_preset", "list_benchmarks", "parse_model_name"]
