"""Prompt-oriented configuration models and rendering helpers."""

from __future__ import annotations

import json
from typing import cast

from pydantic import Field

from themis.core.base import HashableModel, JSONValue


class PromptSpec(HashableModel):
    """Generic prompt instructions and structured prompt material."""

    instructions: str | None = None
    prefix: str | None = None
    suffix: str | None = None
    blocks: list[dict[str, JSONValue]] = Field(default_factory=list)

    def render_input(self, prompt_input: JSONValue) -> JSONValue:
        """Render prompt-oriented input for provider adapters."""

        if not any(
            (
                self.instructions,
                self.prefix,
                self.suffix,
                self.blocks,
            )
        ):
            return prompt_input
        sections = self.render_sections()
        sections.append(f"Input:\n{_render_prompt_value(prompt_input)}")
        return "\n\n".join(sections)

    def render_sections(self) -> list[str]:
        """Render prompt sections that can prefix a prompt body."""

        sections: list[str] = []
        if self.instructions:
            sections.append(f"Instructions:\n{self.instructions}")
        if self.prefix:
            sections.append(self.prefix)
        for index, block in enumerate(self.blocks, start=1):
            title = block.get("title")
            if isinstance(title, str) and title:
                payload = {key: value for key, value in block.items() if key != "title"}
                sections.append(
                    "\n".join(
                        [
                            f"{title}:",
                            _render_prompt_value(cast(JSONValue, payload)),
                        ]
                    )
                )
                continue
            sections.append(
                "\n".join(
                    [
                        f"Block {index}:",
                        _render_prompt_value(cast(JSONValue, block)),
                    ]
                )
            )
        if self.suffix:
            sections.append(self.suffix)
        return sections


def render_prompt_spec(prompt_spec: PromptSpec | None, body: str) -> str:
    """Render a complete prompt body with optional prompt-spec sections."""

    if prompt_spec is None:
        return body
    sections = prompt_spec.render_sections()
    sections.append(body)
    return "\n\n".join(section for section in sections if section)


def _render_prompt_value(value: JSONValue) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)
