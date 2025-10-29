"""Dataset loading utilities for the prompt engineering experiment."""

from __future__ import annotations

from pathlib import Path

from .config import DatasetConfig


def load_dataset(config: DatasetConfig) -> list[dict[str, object]]:
    """Load dataset based on configuration."""
    
    if config.kind == "demo":
        return _load_demo_dataset(config)
    elif config.kind == "math500_hf":
        return _load_math500_hf(config)
    elif config.kind == "math500_local":
        return _load_math500_local(config)
    elif config.kind == "inline":
        return config.samples
    else:
        raise ValueError(f"Unknown dataset kind: {config.kind}")


def _load_demo_dataset(config: DatasetConfig) -> list[dict[str, object]]:
    """Load a simple demo dataset with math problems."""
    
    demo_data = [
        {
            "unique_id": "demo_1",
            "problem": "What is 15 + 27?",
            "answer": "42",
            "subject": "arithmetic",
            "level": "easy",
            "dataset_name": "demo"
        },
        {
            "unique_id": "demo_2", 
            "problem": "If x + 5 = 12, what is x?",
            "answer": "7",
            "subject": "algebra",
            "level": "easy",
            "dataset_name": "demo"
        },
        {
            "unique_id": "demo_3",
            "problem": "What is the capital of France?",
            "answer": "Paris",
            "subject": "geography",
            "level": "easy",
            "dataset_name": "demo"
        }
    ]
    
    # Apply limit if specified
    if config.limit is not None:
        demo_data = demo_data[:config.limit]
    
    return demo_data


def _load_math500_hf(config: DatasetConfig) -> list[dict[str, object]]:
    """Load Math500 dataset from HuggingFace."""
    # This would use HuggingFace datasets in a real implementation
    # For now, we'll return an empty list as a placeholder
    raise NotImplementedError("Math500 HuggingFace loading not implemented in this example")


def _load_math500_local(config: DatasetConfig) -> list[dict[str, object]]:
    """Load Math500 dataset from local path."""
    # This would load from a local path in a real implementation
    # For now, we'll return an empty list as a placeholder
    raise NotImplementedError("Math500 local loading not implemented in this example")


__all__ = ["load_dataset"]