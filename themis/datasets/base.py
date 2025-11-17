"""Base dataset implementation with schema support.

This module provides a base class that implements common dataset operations
like filtering, limiting, and stratification.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Any, Callable, Iterable

from themis.datasets import schema as dataset_schema

logger = logging.getLogger(__name__)


class BaseDataset:
    """Base implementation for dataset classes that implement DatasetAdapter protocol.

    This class provides a reusable implementation of common dataset operations
    including filtering, limiting, and stratification. It satisfies the
    DatasetAdapter protocol by implementing iter_samples().

    The class implements the structural DatasetAdapter protocol without
    explicit inheritance, using duck typing. At runtime, instances will
    satisfy isinstance(obj, DatasetAdapter) checks.

    Subclasses should provide the initial samples, schema, and metadata.

    Protocol Compliance:
        Implements DatasetAdapter protocol via iter_samples() method

    Examples:
        class MyDataset(BaseDataset):
            def __init__(self):
                samples = [
                    {"id": "1", "problem": "What is 2+2?", "answer": "4"},
                    {"id": "2", "problem": "What is 3+3?", "answer": "6"},
                ]
                schema = DatasetSchema(
                    id_field="id",
                    reference_field="answer",
                    required_fields={"id", "problem", "answer"},
                )
                metadata = DatasetMetadata(
                    name="SimpleArithmetic",
                    version="1.0",
                    total_samples=2,
                )
                super().__init__(samples, schema, metadata)

        # Verify protocol compliance
        >>> from themis.interfaces import DatasetAdapter
        >>> dataset = MyDataset()
        >>> isinstance(dataset, DatasetAdapter)  # True
    """

    def __init__(
        self,
        samples: Iterable[dict[str, Any]],
        schema: dataset_schema.DatasetSchema,
        metadata: dataset_schema.DatasetMetadata,
        validate: bool = True,
    ):
        """Initialize dataset.

        Args:
            samples: Iterable of sample dictionaries
            schema: Dataset schema
            metadata: Dataset metadata
            validate: Whether to validate samples against schema (default: True)

        Raises:
            ValueError: If validation is enabled and samples don't match schema
        """
        self._samples = list(samples)
        self._schema = schema
        self._metadata = metadata

        if validate:
            self._validate_all()

        # Update metadata total if not set
        if self._metadata.total_samples is None:
            self._metadata = dataset_schema.DatasetMetadata(
                **{**self._metadata.__dict__, "total_samples": len(self._samples)}
            )

    def _validate_all(self) -> None:
        """Validate all samples against schema."""
        logger.debug(
            "Validating %d samples for dataset %s",
            len(self._samples),
            self._metadata.name,
        )

        for i, sample in enumerate(self._samples):
            try:
                self._schema.validate_sample(sample)
            except ValueError as e:
                logger.error("Validation failed for sample %d: %s", i, e)
                raise ValueError(f"Sample {i} validation failed: {e}") from e

        logger.debug("All samples validated successfully")

    def iter_samples(self) -> Iterable[dict[str, Any]]:
        """Iterate over dataset samples."""
        return iter(self._samples)

    def get_schema(self) -> dataset_schema.DatasetSchema:
        """Get the dataset schema."""
        return self._schema

    def get_metadata(self) -> dataset_schema.DatasetMetadata:
        """Get dataset metadata."""
        return self._metadata

    def filter(self, predicate: Callable[[dict[str, Any]], bool]) -> BaseDataset:
        """Return filtered view of dataset.

        Args:
            predicate: Function that returns True for samples to keep

        Returns:
            New BaseDataset with filtered samples
        """
        filtered_samples = [s for s in self._samples if predicate(s)]
        logger.debug(
            "Filtered dataset from %d to %d samples",
            len(self._samples),
            len(filtered_samples),
        )

        return BaseDataset(
            samples=filtered_samples,
            schema=self._schema,
            metadata=self._metadata,
            validate=False,  # Already validated
        )

    def limit(self, n: int) -> BaseDataset:
        """Return dataset limited to first n samples.

        Args:
            n: Maximum number of samples

        Returns:
            New BaseDataset with limited samples
        """
        limited_samples = self._samples[:n]
        logger.debug(
            "Limited dataset from %d to %d samples",
            len(self._samples),
            len(limited_samples),
        )

        return BaseDataset(
            samples=limited_samples,
            schema=self._schema,
            metadata=self._metadata,
            validate=False,
        )

    def stratify(
        self, field: str, distribution: dict[str, float], seed: int | None = None
    ) -> BaseDataset:
        """Return stratified sample of dataset.

        Args:
            field: Field to stratify by
            distribution: Desired distribution (values should sum to ~1.0)
            seed: Random seed for reproducibility

        Returns:
            New BaseDataset with stratified samples

        Raises:
            ValueError: If field doesn't exist or distribution is invalid
        """
        # Group samples by field value
        groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for sample in self._samples:
            if field not in sample:
                raise ValueError(f"Field '{field}' not found in sample")
            groups[sample[field]].append(sample)

        # Validate distribution
        total_dist = sum(distribution.values())
        if not (0.99 <= total_dist <= 1.01):
            logger.warning("Distribution values sum to %f, expected ~1.0", total_dist)

        # Calculate sample sizes for each group
        total_samples = len(self._samples)
        stratified_samples = []

        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        for value, desired_ratio in distribution.items():
            if value not in groups:
                logger.warning(
                    "Value '%s' specified in distribution but not found in dataset",
                    value,
                )
                continue

            group_samples = groups[value]
            n_samples = int(total_samples * desired_ratio)
            n_samples = min(n_samples, len(group_samples))  # Can't exceed available

            # Sample from group
            sampled = rng.sample(group_samples, n_samples)
            stratified_samples.extend(sampled)

        logger.debug(
            "Stratified dataset by field '%s' from %d to %d samples",
            field,
            len(self._samples),
            len(stratified_samples),
        )

        return BaseDataset(
            samples=stratified_samples,
            schema=self._schema,
            metadata=self._metadata,
            validate=False,
        )

    def shuffle(self, seed: int | None = None) -> BaseDataset:
        """Return shuffled dataset.

        Args:
            seed: Random seed for reproducibility

        Returns:
            New BaseDataset with shuffled samples
        """
        shuffled = list(self._samples)
        if seed is not None:
            random.Random(seed).shuffle(shuffled)
        else:
            random.shuffle(shuffled)

        return BaseDataset(
            samples=shuffled,
            schema=self._schema,
            metadata=self._metadata,
            validate=False,
        )

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get sample by index."""
        return self._samples[idx]


__all__ = ["BaseDataset"]
