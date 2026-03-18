from __future__ import annotations

from themis import PromptMessage
from themis.prompting import render_prompt_messages
from themis.types.enums import PromptRole


def test_render_prompt_messages_uses_benchmark_namespaces() -> None:
    rendered = render_prompt_messages(
        [
            PromptMessage(
                role=PromptRole.SYSTEM,
                content="You are judging {slice.dimensions[source]} in {prompt.family}.",
            ),
            PromptMessage(
                role=PromptRole.USER,
                content="Question: {item.question} [{runtime.run_labels[phase]}]",
            ),
        ],
        {
            "item": {"question": "2 + 2"},
            "slice": {"dimensions": {"source": "medqa"}},
            "prompt": {"family": "qa"},
            "runtime": {"run_labels": {"phase": "smoke"}},
        },
    )

    assert rendered == [
        {"role": "system", "content": "You are judging medqa in qa."},
        {"role": "user", "content": "Question: 2 + 2 [smoke]"},
    ]


def test_render_prompt_messages_preserves_placeholder_for_deep_missing_access() -> None:
    rendered = render_prompt_messages(
        [
            PromptMessage(
                role=PromptRole.USER,
                content="Question: {item.question[text]}",
            )
        ],
        {},
    )

    assert rendered == [{"role": "user", "content": "Question: {item}"}]
