"""Generation planning primitives.

This module provides generation planning with flexible expansion strategies:

1. GenerationPlan: Traditional Cartesian product expansion
2. FlexibleGenerationPlan: Pluggable expansion strategies
3. Expansion strategies:
   - CartesianExpansionStrategy: Full Cartesian product (default)
   - FilteredExpansionStrategy: Filter specific combinations
   - ConditionalExpansionStrategy: Route based on conditions
   - ChainedExpansionStrategy: Chain multiple strategies

Example (Traditional):
    >>> plan = GenerationPlan(
    ...     templates=[template1, template2],
    ...     models=[model1, model2],
    ...     sampling_parameters=[config1]
    ... )
    >>> tasks = list(plan.expand(dataset))

Example (Filtered):
    >>> plan = FlexibleGenerationPlan(
    ...     templates=[template1, template2],
    ...     models=[model1, model2],
    ...     sampling_parameters=[config1],
    ...     expansion_strategy=FilteredExpansionStrategy(
    ...         task_filter=lambda row, tpl, mdl, smp: mdl.identifier != "gpt-4" or row.get("difficulty") == "hard"
    ...     )
    ... )
    >>> tasks = list(plan.expand(dataset))  # Only creates GPT-4 tasks for hard problems
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Protocol, Sequence

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
    ) -> Dict[str, Any]:
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


# ============================================================================
# Flexible Generation Planning with Expansion Strategies
# ============================================================================


@dataclass
class PlanContext:
    """Context passed to expansion strategies.

    Contains all the information needed to expand dataset rows into tasks.
    """

    templates: Sequence[templates.PromptTemplate]
    models: Sequence[core_entities.ModelSpec]
    sampling_parameters: Sequence[core_entities.SamplingConfig]
    dataset_id_field: str
    reference_field: str | None
    metadata_fields: Sequence[str]
    context_builder: Callable[[dict], dict] | None


class ExpansionStrategy(Protocol):
    """Strategy for expanding dataset into generation tasks.

    Different strategies can control which combinations of
    (row, template, model, sampling) are generated.
    """

    def expand(
        self,
        dataset: Sequence[dict[str, Any]],
        context: PlanContext,
    ) -> Iterator[core_entities.GenerationTask]:
        """Expand dataset rows into generation tasks.

        Args:
            dataset: Dataset rows to expand
            context: Plan context with templates, models, etc.

        Yields:
            Generation tasks
        """
        ...


class CartesianExpansionStrategy:
    """Traditional Cartesian product expansion (default behavior).

    Generates all possible combinations of:
    - Each row in dataset
    - Each template
    - Each model
    - Each sampling configuration

    This is the default expansion strategy used by GenerationPlan.
    """

    def expand(
        self,
        dataset: Sequence[dict[str, Any]],
        context: PlanContext,
    ) -> Iterator[core_entities.GenerationTask]:
        """Expand using Cartesian product."""
        for row in dataset:
            row_dict = dict(row)
            ctx = (
                context.context_builder(row_dict)
                if context.context_builder
                else row_dict
            )
            dataset_id = row_dict.get(context.dataset_id_field)
            reference = (
                row_dict.get(context.reference_field)
                if context.reference_field
                else None
            )

            for template in context.templates:
                rendered = template.render_prompt(ctx)
                base_metadata = self._build_metadata(
                    template, dataset_id, row_dict, context
                )

                for model in context.models:
                    for sampling in context.sampling_parameters:
                        yield core_entities.GenerationTask(
                            prompt=rendered,
                            model=model,
                            sampling=sampling,
                            metadata=dict(base_metadata),
                            reference=self._build_reference(reference, context),
                        )

    def _build_metadata(
        self,
        template: templates.PromptTemplate,
        dataset_id: Any,
        row: dict[str, Any],
        context: PlanContext,
    ) -> Dict[str, Any]:
        """Build metadata dict for task."""
        metadata = {
            f"template_{key}": value for key, value in (template.metadata or {}).items()
        }
        if dataset_id is not None:
            metadata["dataset_id"] = dataset_id
        
        # If metadata_fields is explicitly specified, use it as a filter (existing behavior)
        # Otherwise, include all fields by default (new behavior for custom metrics)
        if context.metadata_fields:
            # Explicit filter - only include specified fields
            for field_name in context.metadata_fields:
                if field_name in row:
                    metadata[field_name] = row[field_name]
        else:
            # No filter - include all fields except those used for other purposes
            for field_name, field_value in row.items():
                if field_name not in (context.dataset_id_field, context.reference_field):
                    metadata[field_name] = field_value
        
        return metadata

    def _build_reference(
        self, raw_reference: Any | None, context: PlanContext
    ) -> core_entities.Reference | None:
        """Build reference object."""
        if raw_reference is None:
            return None
        return core_entities.Reference(
            kind=context.reference_field or "reference", value=raw_reference
        )


class FilteredExpansionStrategy:
    """Expansion strategy that filters specific combinations.

    Only generates tasks that pass the filter function. Useful for:
    - Expensive models only on hard problems
    - Specific templates for specific models
    - Conditional generation based on metadata

    Example:
        >>> # Only use GPT-4 on hard problems
        >>> strategy = FilteredExpansionStrategy(
        ...     task_filter=lambda row, tpl, mdl, smp: (
        ...         mdl.identifier != "gpt-4" or row.get("difficulty") == "hard"
        ...     )
        ... )
    """

    def __init__(
        self,
        task_filter: Callable[
            [
                dict[str, Any],  # row
                templates.PromptTemplate,  # template
                core_entities.ModelSpec,  # model
                core_entities.SamplingConfig,  # sampling
            ],
            bool,
        ],
    ):
        """Initialize filtered expansion strategy.

        Args:
            task_filter: Function that returns True if task should be generated
        """
        self._filter = task_filter
        self._base_strategy = CartesianExpansionStrategy()

    def expand(
        self,
        dataset: Sequence[dict[str, Any]],
        context: PlanContext,
    ) -> Iterator[core_entities.GenerationTask]:
        """Expand with filtering."""
        for row in dataset:
            row_dict = dict(row)
            ctx = (
                context.context_builder(row_dict)
                if context.context_builder
                else row_dict
            )
            dataset_id = row_dict.get(context.dataset_id_field)
            reference = (
                row_dict.get(context.reference_field)
                if context.reference_field
                else None
            )

            for template in context.templates:
                rendered = template.render_prompt(ctx)
                base_metadata = self._base_strategy._build_metadata(
                    template, dataset_id, row_dict, context
                )

                for model in context.models:
                    for sampling in context.sampling_parameters:
                        # Check if this combination should be generated
                        if self._filter(row_dict, template, model, sampling):
                            yield core_entities.GenerationTask(
                                prompt=rendered,
                                model=model,
                                sampling=sampling,
                                metadata=dict(base_metadata),
                                reference=self._base_strategy._build_reference(
                                    reference, context
                                ),
                            )


class ConditionalExpansionStrategy:
    """Expansion strategy that routes to different strategies based on conditions.

    Evaluates conditions in order and uses the first matching strategy.
    Falls back to default strategy if no conditions match.

    Example:
        >>> # Use different strategies for math vs code problems
        >>> strategy = ConditionalExpansionStrategy(
        ...     rules=[
        ...         (lambda row: row.get("type") == "math", math_strategy),
        ...         (lambda row: row.get("type") == "code", code_strategy),
        ...     ],
        ...     default=default_strategy
        ... )
    """

    def __init__(
        self,
        rules: list[tuple[Callable[[dict], bool], ExpansionStrategy]],
        default: ExpansionStrategy,
    ):
        """Initialize conditional expansion strategy.

        Args:
            rules: List of (condition, strategy) tuples
            default: Default strategy if no conditions match
        """
        self._rules = rules
        self._default = default

    def expand(
        self,
        dataset: Sequence[dict[str, Any]],
        context: PlanContext,
    ) -> Iterator[core_entities.GenerationTask]:
        """Expand using conditional routing."""
        # Group rows by which strategy applies
        strategy_groups: dict[int, list[dict]] = {}

        for row in dataset:
            # Find first matching rule
            matched = False
            for rule_idx, (condition, strategy) in enumerate(self._rules):
                if condition(row):
                    strategy_groups.setdefault(rule_idx, []).append(row)
                    matched = True
                    break

            if not matched:
                strategy_groups.setdefault(-1, []).append(row)

        # Expand each group with its strategy
        for rule_idx, group_rows in strategy_groups.items():
            if rule_idx == -1:
                strategy = self._default
            else:
                strategy = self._rules[rule_idx][1]

            yield from strategy.expand(group_rows, context)


class ChainedExpansionStrategy:
    """Expansion strategy that chains multiple strategies.

    Applies multiple strategies in sequence, yielding tasks from all of them.
    Useful for combining different expansion approaches.

    Example:
        >>> # Generate baseline tasks + additional high-temperature samples for hard problems
        >>> strategy = ChainedExpansionStrategy([
        ...     CartesianExpansionStrategy(),
        ...     FilteredExpansionStrategy(
        ...         task_filter=lambda row, tpl, mdl, smp: (
        ...             row.get("difficulty") == "hard" and smp.temperature > 0.5
        ...         )
        ...     )
        ... ])
    """

    def __init__(self, strategies: Sequence[ExpansionStrategy]):
        """Initialize chained expansion strategy.

        Args:
            strategies: List of strategies to apply in sequence
        """
        self._strategies = strategies

    def expand(
        self,
        dataset: Sequence[dict[str, Any]],
        context: PlanContext,
    ) -> Iterator[core_entities.GenerationTask]:
        """Expand using all strategies in sequence."""
        for strategy in self._strategies:
            yield from strategy.expand(dataset, context)


@dataclass
class FlexibleGenerationPlan:
    """Generation plan with pluggable expansion strategy.

    Allows controlling how dataset rows are expanded into generation tasks
    using different expansion strategies.

    Example:
        >>> # Filter expensive model to hard problems only
        >>> plan = FlexibleGenerationPlan(
        ...     templates=[template1, template2],
        ...     models=[cheap_model, expensive_model],
        ...     sampling_parameters=[config],
        ...     expansion_strategy=FilteredExpansionStrategy(
        ...         task_filter=lambda row, tpl, mdl, smp: (
        ...             mdl.identifier != "expensive" or row.get("difficulty") == "hard"
        ...         )
        ...     )
        ... )
    """

    templates: Sequence[templates.PromptTemplate]
    models: Sequence[core_entities.ModelSpec]
    sampling_parameters: Sequence[core_entities.SamplingConfig]
    expansion_strategy: ExpansionStrategy | None = None
    dataset_id_field: str = "id"
    reference_field: str | None = "expected"
    metadata_fields: Sequence[str] = field(default_factory=tuple)
    context_builder: Callable[[dict], dict] | None = None

    def expand(
        self, dataset: Sequence[dict[str, object]]
    ) -> Iterator[core_entities.GenerationTask]:
        """Expand dataset into generation tasks using strategy.

        Args:
            dataset: Dataset rows to expand

        Yields:
            Generation tasks
        """
        context = PlanContext(
            templates=self.templates,
            models=self.models,
            sampling_parameters=self.sampling_parameters,
            dataset_id_field=self.dataset_id_field,
            reference_field=self.reference_field,
            metadata_fields=self.metadata_fields,
            context_builder=self.context_builder,
        )

        strategy = self.expansion_strategy or CartesianExpansionStrategy()
        yield from strategy.expand(dataset, context)
