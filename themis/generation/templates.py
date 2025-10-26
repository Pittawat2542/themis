"""Prompt template primitives for Themis generation domain."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List

from themis.core import entities as core_entities


class TemplateRenderingError(RuntimeError):
    """Raised when a prompt template cannot be rendered."""


@dataclass
class PromptTemplate:
    """Represents a format string and associated metadata."""

    name: str
    template: str
    metadata: Dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self._spec = core_entities.PromptSpec(
            name=self.name,
            template=self.template,
            metadata=dict(self.metadata or {}),
        )

    def render(self, **kwargs: Any) -> str:
        try:
            return self.template.format(**kwargs)
        except KeyError as exc:  # pragma: no cover - defensive path
            missing = exc.args[0]
            raise TemplateRenderingError(
                f"Missing template variable: {missing}"
            ) from exc

    def expand_variants(
        self,
        *,
        base_context: Dict[str, Any],
        variant_values: Dict[str, Iterable[Any]],
    ) -> List[core_entities.PromptRender]:
        """Generate prompts for the cross-product of variant fields."""

        if not variant_values:
            return [self._render_context(base_context)]

        keys = sorted(variant_values.keys())
        prompts: list[core_entities.PromptRender] = []
        for combo in product(*(variant_values[key] for key in keys)):
            combo_context = dict(base_context)
            combo_context.update(dict(zip(keys, combo)))
            prompts.append(self._render_context(combo_context))
        return prompts

    def render_prompt(self, context: Dict[str, Any]) -> core_entities.PromptRender:
        """Render the template to a core PromptRender."""
        return self._render_context(context)

    def _render_context(self, context: Dict[str, Any]) -> core_entities.PromptRender:
        prompt_text = self.render(**context)
        metadata = dict(self.metadata or {})
        return core_entities.PromptRender(
            spec=self._spec,
            text=prompt_text,
            context=dict(context),
            metadata=metadata,
        )
