"""Tests for BenchmarkSpec.preview() dry-run prompt rendering."""

from __future__ import annotations


from themis import (
    BenchmarkSpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ModelSpec,
    PromptMessage,
    PromptTurnSpec,
    PromptVariantSpec,
    ScoreSpec,
    SliceSpec,
)
from themis.specs.foundational import DatasetSpec, GenerationSpec
from themis.types.enums import DatasetSource, PromptRole


def _make_benchmark(
    prompt_variants: list[PromptVariantSpec] | None = None,
) -> BenchmarkSpec:
    if prompt_variants is None:
        prompt_variants = [
            PromptVariantSpec(
                id="qa-default",
                family="qa",
                messages=[
                    PromptMessage(
                        role=PromptRole.SYSTEM,
                        content="You are a helpful assistant.",
                    ),
                    PromptMessage(
                        role=PromptRole.USER,
                        content="Solve: {item.question}",
                    ),
                ],
            )
        ]
    return BenchmarkSpec(
        benchmark_id="preview-test",
        models=[ModelSpec(model_id="gpt-4", provider="openai")],
        slices=[
            SliceSpec(
                slice_id="arithmetic",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                prompt_variant_ids=[v.id for v in prompt_variants],
                generation=GenerationSpec(),
                scores=[ScoreSpec(name="default", metrics=["exact_match"])],
            )
        ],
        prompt_variants=prompt_variants,
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
    )


class TestBenchmarkSpecPreview:
    def test_preview_returns_list_of_rendered_variant_dicts(self) -> None:
        benchmark = _make_benchmark()
        result = benchmark.preview({"question": "2 + 2"})
        assert isinstance(result, list)
        assert len(result) == 1

    def test_preview_contains_prompt_variant_id(self) -> None:
        benchmark = _make_benchmark()
        result = benchmark.preview({"question": "2 + 2"})
        assert result[0]["prompt_variant_id"] == "qa-default"

    def test_preview_renders_item_fields_into_messages(self) -> None:
        benchmark = _make_benchmark()
        result = benchmark.preview({"question": "2 + 2"})
        messages = result[0]["messages"]
        assert any("Solve: 2 + 2" in m["content"] for m in messages)

    def test_preview_renders_all_roles(self) -> None:
        benchmark = _make_benchmark()
        result = benchmark.preview({"question": "2 + 2"})
        roles = {m["role"] for m in result[0]["messages"]}
        assert "system" in roles
        assert "user" in roles

    def test_preview_multiple_variants_returns_one_entry_each(self) -> None:
        variants = [
            PromptVariantSpec(
                id="v1",
                messages=[
                    PromptMessage(role=PromptRole.USER, content="Q: {item.question}")
                ],
            ),
            PromptVariantSpec(
                id="v2",
                messages=[
                    PromptMessage(
                        role=PromptRole.SYSTEM, content="Think step by step."
                    ),
                    PromptMessage(
                        role=PromptRole.USER, content="Solve: {item.question}"
                    ),
                ],
            ),
        ]
        benchmark = _make_benchmark(prompt_variants=variants)
        result = benchmark.preview({"question": "1 + 1"})
        assert len(result) == 2
        ids = {r["prompt_variant_id"] for r in result}
        assert ids == {"v1", "v2"}

    def test_preview_filter_by_prompt_variant_ids(self) -> None:
        variants = [
            PromptVariantSpec(
                id="v1",
                messages=[PromptMessage(role=PromptRole.USER, content="{item.q}")],
            ),
            PromptVariantSpec(
                id="v2",
                messages=[PromptMessage(role=PromptRole.USER, content="{item.q}")],
            ),
        ]
        benchmark = _make_benchmark(prompt_variants=variants)
        result = benchmark.preview({"q": "test"}, prompt_variant_ids=["v1"])
        assert len(result) == 1
        assert result[0]["prompt_variant_id"] == "v1"

    def test_preview_includes_follow_up_turns_when_present(self) -> None:
        variant = PromptVariantSpec(
            id="multi-turn",
            messages=[PromptMessage(role=PromptRole.USER, content="Q: {item.q}")],
            follow_up_turns=[
                PromptTurnSpec(
                    messages=[
                        PromptMessage(
                            role=PromptRole.USER, content="Double-check: {item.q}"
                        )
                    ]
                )
            ],
        )
        benchmark = _make_benchmark(prompt_variants=[variant])
        result = benchmark.preview({"q": "5 + 3"})
        assert "follow_up_turns" in result[0]
        turns = result[0]["follow_up_turns"]
        assert len(turns) == 1
        assert "Double-check: 5 + 3" in turns[0]["messages"][0]["content"]

    def test_preview_static_variables_available_via_prompt_namespace(self) -> None:
        variant = PromptVariantSpec(
            id="with-vars",
            messages=[
                PromptMessage(
                    role=PromptRole.USER,
                    content="({prompt.tone}) {item.question}",
                )
            ],
            variables={"tone": "concise"},
        )
        benchmark = _make_benchmark(prompt_variants=[variant])
        result = benchmark.preview({"question": "What is 3+3?"})
        assert "(concise)" in result[0]["messages"][0]["content"]

    def test_preview_returns_empty_list_for_unknown_filter_ids(self) -> None:
        benchmark = _make_benchmark()
        result = benchmark.preview(
            {"question": "x"}, prompt_variant_ids=["nonexistent"]
        )
        assert result == []
