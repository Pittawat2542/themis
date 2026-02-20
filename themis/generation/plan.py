"""Generation planning primitives.

This module provides generation planning with Cartesian product expansion.

The GenerationPlan class expands a dataset into generation tasks by creating
all possible combinations of:
- Dataset rows
- Prompt templates
- Model specifications
- Sampling configurations

Example:
    >>> plan = GenerationPlan(
    ...     templates=[template1, template2],
    ...     models=[model1, model2],
    ...     sampling_parameters=[config1]
    ... )
    >>> tasks = list(plan.expand(dataset))
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from themis.core import entities as core_entities
from themis.generation import templates


@dataclass
class GenerationPlan:
    templates: Sequence[templates.PromptTemplate]
    models: Sequence[core_entities.ModelSpec]
    sampling_parameters: Sequence[core_entities.SamplingConfig]
    dataset_id_field: str = "id"
    reference_field: str | None = "expected"
    metadata_fields: Sequence[str] = field(default_factory=tuple)
    context_builder: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    def expand(
        self, dataset: Sequence[dict[str, object]]
    ) -> Iterator[core_entities.GenerationTask]:
        for row in dataset:
            row_dict = dict(row)
            context = self._build_context(row_dict)
            dataset_id = row_dict.get(self.dataset_id_field)
            reference = (
                row_dict.get(self.reference_field) if self.reference_field else None
            )
            for template in self.templates:
                rendered_prompt = template.render_prompt(context)
                base_metadata = self._build_metadata(template, dataset_id, row_dict)
                for model in self.models:
                    for sampling in self.sampling_parameters:
                        yield core_entities.GenerationTask(
                            prompt=rendered_prompt,
                            model=model,
                            sampling=sampling,
                            metadata=dict(base_metadata),
                            reference=self._build_reference(reference),
                        )

    def _build_context(self, row: dict[str, Any]) -> dict[str, Any]:
        if self.context_builder is None:
            return dict(row)
        return self.context_builder(dict(row))

    def _build_metadata(
        self,
        template: templates.PromptTemplate,
        dataset_id: Any,
        row: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = {
            f"template_{key}": value for key, value in (template.metadata or {}).items()
        }
        if dataset_id is not None:
            metadata["dataset_id"] = dataset_id

        # If metadata_fields is explicitly specified, use it as a filter (existing behavior)
        # Otherwise, include all fields by default (new behavior for custom metrics)
        if self.metadata_fields:
            # Explicit filter - only include specified fields
            for field_name in self.metadata_fields:
                if field_name in row:
                    metadata[field_name] = row[field_name]
        else:
            # No filter - include all fields except those used for other purposes
            for field_name, field_value in row.items():
                if field_name not in (self.dataset_id_field, self.reference_field):
                    metadata[field_name] = field_value

        return metadata

    def _build_reference(
        self, raw_reference: Any | None
    ) -> core_entities.Reference | None:
        if raw_reference is None:
            return None
        return core_entities.Reference(
            kind=self.reference_field or "reference", value=raw_reference
        )


__all__ = ["GenerationPlan"]
