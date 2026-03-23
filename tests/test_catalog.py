from __future__ import annotations

from collections.abc import Sequence
import io
import inspect
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

import themis.catalog as catalog
from themis import BenchmarkDefinition, DatasetQuerySpec
from themis.benchmark.specs import DatasetSliceSpec
from themis.catalog.datasets import common as dataset_common
from themis.catalog.datasets.common import (
    _normalize_babe_rows,
    _normalize_frontierscience_rows,
    _normalize_gpqa_diamond_rows,
    _normalize_healthbench_rows,
    _normalize_imo_answerbench_rows,
    _normalize_mmmlu_rows,
    _normalize_math_short_answer_rows,
    _normalize_phybench_rows,
    _normalize_procbench_rows,
    _normalize_superchem_rows,
)
from themis.catalog.runtime.metrics.common import MathEquivalenceMetric
from themis.errors import SpecValidationError, ThemisError
from themis.catalog.runtime.common import (
    _build_judge_spec,
    _coerce_usage_int,
    _openai_mcp_tool_payload,
    _openai_response_input_message,
    _run_openai_chat_inference,
)
from themis.contracts.protocols import InferenceResult
from themis.specs.experiment import (
    InferenceParamsSpec,
    PromptMessage,
    PromptTemplateSpec,
    RuntimeContext,
)
from themis.specs.foundational import (
    DatasetSpec,
    JinjaTransform,
    McpServerSpec,
    ModelSpec,
    PythonTransform,
    RenameFieldTransform,
)
from themis.records.conversation import ToolCallEvent, ToolResultEvent
from themis.types.enums import DatasetSource, ErrorCode, PromptRole
from themis.types.events import ScoreRow
from themis.types.json_types import JSONDict


class _StubProjectionRepo:
    def __init__(self, score_rows: Sequence[Any]) -> None:
        self._score_rows = list(score_rows)

    def iter_candidate_scores(
        self,
        *,
        trial_hashes: list[str] | None = None,
        metric_id: str | None = None,
        evaluation_hash: str | None = None,
    ):
        del evaluation_hash
        allowed_hashes = set(trial_hashes or [])
        for row in self._score_rows:
            if allowed_hashes and row.trial_hash not in allowed_hashes:
                continue
            if metric_id is not None and row.metric_id != metric_id:
                continue
            yield row


class _StubResult:
    def __init__(
        self,
        score_rows: Sequence[Any],
        *,
        scan_stats: JSONDict | None = None,
        trial_summaries: Sequence[Any] | None = None,
    ) -> None:
        self.projection_repo = _StubProjectionRepo(score_rows)
        self.trial_hashes = sorted(
            {str(getattr(row, "trial_hash")) for row in score_rows}
        )
        self.active_evaluation_hash = None
        self._builtin_scan_stats = dict(scan_stats or {})
        self._trial_summaries = list(trial_summaries or [])

    def iter_trial_summaries(self):
        yield from self._trial_summaries


def _row_mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return {str(key): item for key, item in value.items()}


def _json_mapping(value: object) -> JSONDict:
    assert isinstance(value, dict)
    return cast(JSONDict, value)


def _first_preview_message(preview: Sequence[JSONDict]) -> dict[str, object]:
    messages = preview[0].get("messages")
    assert isinstance(messages, list)
    assert messages
    return _row_mapping(messages[0])


def test_public_catalog_lists_requested_benchmarks() -> None:
    assert set(catalog.list_catalog_benchmarks()) == {
        "aime_2025",
        "aime_2026",
        "aethercode",
        "apex_2025",
        "babe",
        "beyond_aime",
        "codeforces",
        "encyclo_k",
        "frontierscience",
        "gpqa_diamond",
        "healthbench",
        "hle",
        "hmmt_feb_2025",
        "hmmt_nov_2025",
        "humaneval",
        "humaneval_plus",
        "imo_answerbench",
        "livecodebench",
        "lpfqa",
        "mmlu_pro",
        "mmmlu",
        "phybench",
        "procbench",
        "rolebench",
        "simpleqa_verified",
        "superchem",
        "supergpqa",
    }


@pytest.mark.parametrize(
    ("benchmark_id", "expected_dataset_id", "expected_config_name"),
    [
        ("mmmlu", "openai/MMMLU", "default"),
        ("mmmlu:AR_XY", "openai/MMMLU", "AR_XY"),
        ("procbench:task01", "ifujisawa/procbench", "task01"),
        (
            "rolebench:instruction_generalization_eng",
            "ZenMoore/RoleBench",
            "instruction_generalization_eng",
        ),
        (
            "rolebench:role_generalization_eng",
            "ZenMoore/RoleBench",
            "role_generalization_eng",
        ),
        ("superchem", "ZehuaZhao/SUPERChem", "default"),
        ("superchem:zh", "ZehuaZhao/SUPERChem", "default"),
    ],
)
def test_catalog_variant_benchmarks_expose_expected_hf_config_name(
    benchmark_id: str,
    expected_dataset_id: str,
    expected_config_name: str,
) -> None:
    definition = catalog.get_catalog_benchmark(benchmark_id)
    requires_judge = definition.requires_judge
    benchmark = definition.build_benchmark(
        model_id="demo-model",
        provider="demo",
        judge_model_id="judge-model" if requires_judge else None,
        judge_provider="demo" if requires_judge else None,
    )

    assert benchmark.slices[0].dataset.dataset_id == expected_dataset_id
    assert benchmark.slices[0].dataset.config_name == expected_config_name


def test_get_catalog_benchmark_accepts_rolebench_variant_ids() -> None:
    definition = catalog.get_catalog_benchmark(
        "rolebench:instruction_generalization_eng"
    )

    assert definition.benchmark_id == "rolebench:instruction_generalization_eng"
    assert definition.metadata["variant_ids"] == ["instruction_generalization_eng"]


def test_get_catalog_benchmark_rejects_unknown_rolebench_variant_ids() -> None:
    with pytest.raises(ValueError, match="unknown RoleBench variant"):
        catalog.get_catalog_benchmark("rolebench:missing")


def test_rolebench_aggregate_builds_one_slice_per_variant() -> None:
    definition = catalog.get_catalog_benchmark("rolebench")

    benchmark = definition.build_benchmark(model_id="demo-model", provider="demo")

    assert benchmark.benchmark_id == "rolebench"
    assert [slice_spec.slice_id for slice_spec in benchmark.slices] == [
        "rolebench-instruction_generalization_eng",
        "rolebench-role_generalization_eng",
    ]
    assert [slice_spec.dimensions for slice_spec in benchmark.slices] == [
        {"rolebench_variant": "instruction_generalization_eng"},
        {"rolebench_variant": "role_generalization_eng"},
    ]
    assert [slice_spec.dataset.dataset_id for slice_spec in benchmark.slices] == [
        "ZenMoore/RoleBench",
        "ZenMoore/RoleBench",
    ]
    assert [slice_spec.dataset.split for slice_spec in benchmark.slices] == [
        "test",
        "test",
    ]
    assert [slice_spec.dataset.config_name for slice_spec in benchmark.slices] == [
        "instruction_generalization_eng",
        "role_generalization_eng",
    ]
    assert [slice_spec.scores[0].metrics for slice_spec in benchmark.slices] == [
        ["rolebench_rouge_l_f1"],
        ["rolebench_rouge_l_f1"],
    ]


def test_render_preview_formats_rolebench_prompt_from_fixture_sample() -> None:
    definition = catalog.get_catalog_benchmark(
        "rolebench:instruction_generalization_eng"
    )

    preview = definition.render_preview(model_id="demo-model", provider="demo")

    messages = cast(list[dict[str, object]], preview[0]["messages"])
    assert messages[0]["role"] == "system"
    assert "You are {role}" not in str(messages[0]["content"])
    assert "assigned one personality role" in str(messages[0]["content"])
    assert messages[1]["role"] == "user"
    assert "question" not in str(messages[1]["content"]).lower()


def test_builtin_runtime_defaults_use_8192_generator_tokens() -> None:
    definition = catalog.get_catalog_benchmark("mmlu_pro")

    config = definition.build_runtime_config(
        model_id="demo-model",
        provider="demo",
    )

    assert config.max_tokens == 8192


def test_catalog_definitions_use_generic_benchmark_definition_and_metadata() -> None:
    definition = catalog.get_catalog_benchmark("mmlu_pro")

    assert isinstance(definition, BenchmarkDefinition)


def test_get_catalog_benchmark_accepts_humaneval_variants() -> None:
    definition = catalog.get_catalog_benchmark("humaneval_plus:mini,v0.1.10")

    assert definition.benchmark_id == "humaneval_plus:mini,v0.1.10"
    assert definition.metadata["variant"] == "plus"
    assert definition.metadata["mini"] is True
    assert definition.metadata["noextreme"] is False
    assert definition.metadata["version"] == "v0.1.10"


@pytest.mark.parametrize(
    ("benchmark_id", "message"),
    [
        ("humaneval:mini,noextreme", "cannot combine"),
        ("humaneval:mini,mini", "duplicate"),
        ("humaneval:v0.1.9,v0.1.10", "multiple version"),
        ("humaneval:unknown", "unknown"),
    ],
)
def test_get_catalog_benchmark_rejects_invalid_humaneval_variants(
    benchmark_id: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        catalog.get_catalog_benchmark(benchmark_id)


def test_openai_compatible_benchmarks_use_env_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://127.0.0.1:1234/v1")
    definition = catalog.get_catalog_benchmark("aime_2026")

    benchmark = definition.build_benchmark(
        model_id="demo-model",
        provider="openai_compatible",
    )

    assert benchmark.models[0].extras["base_url"] == "http://127.0.0.1:1234/v1"


def test_builtin_judge_spec_defaults_use_8192_tokens() -> None:
    judge_spec = _build_judge_spec(
        model_id="judge-model",
        provider="demo",
    )

    assert judge_spec.params.max_tokens == 8192


def test_openai_chat_inference_uses_responses_api_for_mcp_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(
        id="resp_123",
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
        ),
        output=[
            SimpleNamespace(
                type="mcp_call",
                id="mcp_1",
                server_label="dice",
                name="roll",
                arguments='{"expression":"2d4+1"}',
            ),
            SimpleNamespace(
                type="mcp_call_output",
                id="mcp_1_out",
                call_id="mcp_1",
                output='{"value":"6"}',
            ),
            SimpleNamespace(type="message"),
        ],
        output_text="6",
    )
    responses = MagicMock()
    responses.create.return_value = response
    client = SimpleNamespace(responses=responses)
    openai_module = SimpleNamespace(OpenAI=MagicMock(return_value=client))
    monkeypatch.setattr(
        "themis.catalog.runtime.common.import_optional",
        lambda name, extra: openai_module,
    )

    trial = SimpleNamespace(
        trial_id="trial_mcp_123",
        model=ModelSpec(model_id="gpt-5", provider="openai"),
        prompt=PromptTemplateSpec(
            messages=[PromptMessage(role=PromptRole.USER, content="Roll 2d4+1")]
        ),
        params=InferenceParamsSpec(
            max_tokens=64,
            temperature=0.1,
            top_p=0.9,
            seed=7,
            extras={"reasoning": {"effort": "medium"}},
        ),
        mcp_servers=[
            McpServerSpec(
                id="dice",
                server_label="dice",
                server_url="https://dmcp-server.deno.dev/sse",
                allowed_tools=["roll"],
                require_approval="never",
                authorization_secret_name="DICE_TOKEN",
            )
        ],
    )

    result = _run_openai_chat_inference(
        trial,
        context={},
        runtime=RuntimeContext(secrets={"DICE_TOKEN": SecretStr("secret-token")}),
        base_url=None,
        provider_label="OpenAI",
        missing_extra="providers-openai",
    )

    assert isinstance(result, InferenceResult)
    assert result.inference.spec_hash == "inference_trial_mcp_123"
    assert result.inference.raw_text == "6"
    assert result.inference.provider_request_id == "resp_123"
    assert result.inference.token_usage is not None
    assert result.inference.token_usage.total_tokens == 18
    create_kwargs = responses.create.call_args.kwargs
    assert create_kwargs["model"] == "gpt-5"
    assert create_kwargs["input"][0]["content"][0]["text"] == "Roll 2d4+1"
    assert create_kwargs["max_output_tokens"] == 64
    assert create_kwargs["extra_body"] == {"reasoning": {"effort": "medium"}}
    assert create_kwargs["tools"] == [
        {
            "type": "mcp",
            "server_label": "dice",
            "server_url": "https://dmcp-server.deno.dev/sse",
            "allowed_tools": ["roll"],
            "authorization": "secret-token",
            "require_approval": "never",
        }
    ]
    assert result.conversation is not None
    assert isinstance(result.conversation.events[0], ToolCallEvent)
    assert result.conversation.events[0].payload.tool_name == "dice:roll"
    assert isinstance(result.conversation.events[1], ToolResultEvent)


def test_openai_chat_inference_rejects_non_object_mcp_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(
        id="resp_bad_args",
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
        ),
        output=[
            SimpleNamespace(
                type="mcp_call",
                id="mcp_1",
                server_label="dice",
                name="roll",
                arguments='["2d4+1"]',
            ),
        ],
        output_text="",
    )
    responses = MagicMock()
    responses.create.return_value = response
    client = SimpleNamespace(responses=responses)
    openai_module = SimpleNamespace(OpenAI=MagicMock(return_value=client))
    monkeypatch.setattr(
        "themis.catalog.runtime.common.import_optional",
        lambda name, extra: openai_module,
    )

    trial = SimpleNamespace(
        trial_id="trial_mcp_bad_args",
        model=ModelSpec(model_id="gpt-5", provider="openai"),
        prompt=PromptTemplateSpec(
            messages=[PromptMessage(role=PromptRole.USER, content="Roll 2d4+1")]
        ),
        params=InferenceParamsSpec(max_tokens=64),
        mcp_servers=[
            McpServerSpec(
                id="dice",
                server_label="dice",
                server_url="https://dmcp-server.deno.dev/sse",
                allowed_tools=["roll"],
                require_approval="never",
                authorization_secret_name="DICE_TOKEN",
            )
        ],
    )

    with pytest.raises(SpecValidationError, match="MCP tool arguments"):
        _run_openai_chat_inference(
            trial,
            context={},
            runtime=RuntimeContext(secrets={"DICE_TOKEN": SecretStr("secret-token")}),
            base_url=None,
            provider_label="OpenAI",
            missing_extra="providers-openai",
        )


def test_coerce_usage_int_returns_none_for_invalid_strings() -> None:
    assert _coerce_usage_int("") is None
    assert _coerce_usage_int("unknown") is None


def test_openai_response_input_message_preserves_falsy_scalars() -> None:
    assert _openai_response_input_message({"role": "user", "content": 0}) == {
        "role": "user",
        "content": [{"type": "input_text", "text": "0"}],
    }
    assert _openai_response_input_message({"role": "user", "content": False}) == {
        "role": "user",
        "content": [{"type": "input_text", "text": "False"}],
    }


def test_openai_mcp_tool_payload_rejects_approval_gated_servers() -> None:
    with pytest.raises(ThemisError, match="approval-gated"):
        _openai_mcp_tool_payload(
            McpServerSpec(
                id="calendar",
                server_label="google_calendar",
                connector_id="connector_googlecalendar",
                require_approval="always",
            ),
            RuntimeContext(),
        )


def test_openai_chat_inference_rejects_mcp_without_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openai_module = SimpleNamespace(OpenAI=MagicMock())
    monkeypatch.setattr(
        "themis.catalog.runtime.common.import_optional",
        lambda name, extra: openai_module,
    )
    trial = SimpleNamespace(
        model=ModelSpec(model_id="gpt-5", provider="openai"),
        prompt=PromptTemplateSpec(
            messages=[PromptMessage(role=PromptRole.USER, content="Hello")]
        ),
        params=InferenceParamsSpec(max_tokens=64),
        mcp_servers=[
            McpServerSpec(
                id="calendar",
                server_label="google_calendar",
                connector_id="connector_googlecalendar",
                require_approval="never",
                authorization_secret_name="GOOGLE_TOKEN",
            )
        ],
    )

    with pytest.raises(ThemisError, match="GOOGLE_TOKEN"):
        _run_openai_chat_inference(
            trial,
            context={},
            runtime=RuntimeContext(),
            base_url=None,
            provider_label="OpenAI",
            missing_extra="providers-openai",
        )


@pytest.mark.parametrize(
    ("benchmark_id", "dataset_id", "split"),
    [
        ("mmlu_pro", "TIGER-Lab/MMLU-Pro", "test"),
        ("supergpqa", "m-a-p/SuperGPQA", "train"),
        ("gpqa_diamond", "fingertap/GPQA-Diamond", "test"),
        ("encyclo_k", "m-a-p/Encyclo-K", "test"),
        ("babe", "mediabiasgroup/BABE", "test"),
    ],
)
def test_mcq_benchmark_builders_use_expected_dataset_defaults(
    benchmark_id: str,
    dataset_id: str,
    split: str,
) -> None:
    definition = catalog.get_catalog_benchmark(benchmark_id)

    benchmark = definition.build_benchmark(model_id="demo-model", provider="demo")

    assert benchmark.benchmark_id == benchmark_id
    assert benchmark.slices[0].dataset.dataset_id == dataset_id
    assert benchmark.slices[0].dataset.split == split
    assert benchmark.slices[0].parses[0].extractors[0].id == "choice_letter"
    assert benchmark.slices[0].scores[0].metrics == ["choice_accuracy"]


@pytest.mark.parametrize(
    ("benchmark_id", "dataset_id", "split"),
    [
        ("aime_2026", "MathArena/aime_2026", "train"),
        ("aime_2025", "MathArena/aime_2025", "train"),
        ("hmmt_feb_2025", "MathArena/hmmt_feb_2025", "train"),
        ("hmmt_nov_2025", "MathArena/hmmt_nov_2025", "train"),
        ("apex_2025", "MathArena/apex_2025", "train"),
        ("beyond_aime", "ByteDance-Seed/BeyondAIME", "test"),
        ("imo_answerbench", "Hwilner/imo-answerbench", "train"),
        ("phybench", "Eureka-Lab/PHYBench", "train"),
    ],
)
def test_math_benchmark_builders_use_expected_dataset_defaults(
    benchmark_id: str,
    dataset_id: str,
    split: str,
) -> None:
    definition = catalog.get_catalog_benchmark(benchmark_id)

    benchmark = definition.build_benchmark(model_id="demo-model", provider="demo")

    assert benchmark.benchmark_id == benchmark_id
    assert benchmark.slices[0].dataset.dataset_id == dataset_id
    assert benchmark.slices[0].dataset.split == split
    assert benchmark.slices[0].parses[0].extractors[0].id == "math_answer"
    assert benchmark.slices[0].scores[0].metrics == ["math_equivalence"]


@pytest.mark.parametrize(
    "benchmark_id",
    ["simpleqa_verified", "healthbench", "lpfqa", "hle:text_only", "frontierscience"],
)
def test_judge_backed_benchmark_builders_require_explicit_judge_config(
    benchmark_id: str,
) -> None:
    definition = catalog.get_catalog_benchmark(benchmark_id)

    with pytest.raises(ValueError, match="judge"):
        definition.build_benchmark(model_id="demo-model", provider="demo")


def test_render_preview_uses_dataset_native_messages_for_healthbench() -> None:
    definition = catalog.get_catalog_benchmark("healthbench")

    preview = definition.render_preview(
        model_id="demo-model",
        provider="demo",
        judge_model_id="judge-model",
        judge_provider="demo",
    )

    message = _first_preview_message(preview)

    assert message["role"] == "user"
    assert "postpartum depression" in str(message["content"])


def test_healthbench_row_normalizer_populates_prompt_text_for_runtime_rendering() -> (
    None
):
    normalized = _normalize_healthbench_rows(
        [
            {
                "item_id": "hb-1",
                "prompt_id": "hb-1",
                "prompt": [
                    {"role": "system", "content": "Stay concise."},
                    {"role": "user", "content": "How should I treat a burn?"},
                ],
                "rubrics": [],
            }
        ],
        DatasetSpec(source=DatasetSource.HUGGINGFACE, dataset_id="openai/healthbench"),
    )

    assert normalized.rows[0]["prompt_text"] == (
        "system: Stay concise.\n\nuser: How should I treat a burn?"
    )


def test_render_preview_formats_mcq_prompt_from_fixture_sample() -> None:
    definition = catalog.get_catalog_benchmark("mmlu_pro")

    preview = definition.render_preview(model_id="demo-model", provider="demo")
    message = _first_preview_message(preview)

    assert "Question:" in str(message["content"])
    assert "Return the best option letter only." in str(message["content"])


def test_render_preview_formats_hle_prompt_without_template_errors() -> None:
    definition = catalog.get_catalog_benchmark("hle:text_only")

    preview = definition.render_preview(
        model_id="demo-model",
        provider="demo",
        judge_model_id="judge-model",
        judge_provider="demo",
    )

    content = str(_first_preview_message(preview)["content"])
    assert "Explanation:" in content
    assert "Answer:" in content
    assert "Confidence:" in content


def test_get_catalog_benchmark_requires_explicit_hle_variants() -> None:
    with pytest.raises(ValueError, match="explicit HLE variants"):
        catalog.get_catalog_benchmark("hle")


def test_get_catalog_benchmark_accepts_encoded_hle_variant_ids() -> None:
    definition = catalog.get_catalog_benchmark("hle:text_only,no_tool")

    assert definition.benchmark_id == "hle:text_only,no_tool"
    assert definition.metadata["variant_ids"] == ["text_only", "no_tool"]


def test_get_catalog_benchmark_rejects_unknown_hle_variant_ids() -> None:
    with pytest.raises(ValueError, match="unknown HLE variant"):
        catalog.get_catalog_benchmark("hle:missing")


def test_get_catalog_benchmark_rejects_duplicate_hle_variant_ids() -> None:
    with pytest.raises(ValueError, match="duplicate HLE variant"):
        catalog.get_catalog_benchmark("hle:text_only,text_only")


def test_hle_build_benchmark_emits_separate_slices_and_prompt_variants_per_variant() -> (
    None
):
    definition = catalog.get_catalog_benchmark("hle:text_only,no_tool")

    benchmark = definition.build_benchmark(
        model_id="demo-model",
        provider="demo",
        judge_model_id="judge-model",
        judge_provider="demo",
    )

    assert benchmark.benchmark_id == "hle:text_only,no_tool"
    assert [slice_spec.slice_id for slice_spec in benchmark.slices] == [
        "hle-text_only",
        "hle-no_tool",
    ]
    assert [slice_spec.prompt_variant_ids for slice_spec in benchmark.slices] == [
        ["hle-text_only-default"],
        ["hle-no_tool-default"],
    ]
    assert [slice_spec.dimensions for slice_spec in benchmark.slices] == [
        {"hle_variant": "text_only"},
        {"hle_variant": "no_tool"},
    ]
    assert [variant.id for variant in benchmark.prompt_variants] == [
        "hle-text_only-default",
        "hle-no_tool-default",
    ]


def test_procbench_aggregate_builds_one_slice_per_task_variant() -> None:
    definition = catalog.get_catalog_benchmark("procbench")

    benchmark = definition.build_benchmark(model_id="demo-model", provider="demo")

    assert len(benchmark.slices) == 23
    assert benchmark.slices[0].dataset.config_name == "task01"
    assert benchmark.slices[-1].dataset.config_name == "task23"
    assert benchmark.slices[0].scores[0].metrics == ["procbench_final_accuracy"]


def test_hle_preview_renders_one_entry_per_selected_variant() -> None:
    definition = catalog.get_catalog_benchmark("hle:text_only,no_tool")

    preview = definition.render_preview(
        model_id="demo-model",
        provider="demo",
        judge_model_id="judge-model",
        judge_provider="demo",
    )

    assert [entry["prompt_variant_id"] for entry in preview] == [
        "hle-text_only-default",
        "hle-no_tool-default",
    ]


def test_render_preview_formats_math_prompt_with_boxed_answer_instruction() -> None:
    definition = catalog.get_catalog_benchmark("aime_2026")

    preview = definition.render_preview(model_id="demo-model", provider="demo")

    content = str(_first_preview_message(preview)["content"])
    assert "Problem:" in content
    assert "\\boxed{" in content


def test_starter_dataset_provider_applies_supported_dataset_transforms() -> None:
    provider = catalog.CatalogDatasetProvider(
        memory_rows=[{"question": "2 + 2", "answer": "4"}]
    )
    slice_spec = DatasetSliceSpec(
        benchmark_id="transform-demo",
        slice_id="qa",
        dataset=DatasetSpec(
            source=DatasetSource.MEMORY,
            transforms=[
                RenameFieldTransform(
                    field="rendered_question",
                    source_field="question",
                ),
                JinjaTransform(
                    field="prompt_text",
                    template="Solve: {rendered_question}",
                ),
            ],
        ),
    )

    rows = list(provider.scan(slice_spec, DatasetQuerySpec()))
    first_row = _row_mapping(rows[0])

    assert first_row["rendered_question"] == "2 + 2"
    assert first_row["prompt_text"] == "Solve: 2 + 2"


def test_starter_dataset_provider_rejects_python_dataset_transforms() -> None:
    provider = catalog.CatalogDatasetProvider(
        memory_rows=[{"question": "2 + 2", "answer": "4"}]
    )
    slice_spec = DatasetSliceSpec(
        benchmark_id="transform-demo",
        slice_id="qa",
        dataset=DatasetSpec(
            source=DatasetSource.MEMORY,
            transforms=[
                PythonTransform(
                    field="normalized",
                    config={"callable": "demo.normalize"},
                )
            ],
        ),
    )

    with pytest.raises(ValueError, match="python"):
        list(provider.scan(slice_spec, DatasetQuerySpec()))


def test_math_short_answer_row_normalizer_maps_matharena_fields() -> None:
    normalized = _normalize_math_short_answer_rows(
        [
            {
                "problem_idx": 4,
                "problem": "Find x.",
                "answer": 17,
                "problem_type": ["Algebra"],
                "source": "fixture",
            }
        ],
        DatasetSpec(source=DatasetSource.HUGGINGFACE, dataset_id="MathArena/aime_2026"),
    )

    row = _row_mapping(normalized.rows[0])
    metadata = _row_mapping(row["metadata"])

    assert row["item_id"] == "4"
    assert row["problem"] == "Find x."
    assert row["answer"] == "17"
    assert metadata["problem_idx"] == "4"
    assert metadata["problem_type"] == "Algebra"
    assert metadata["source"] == "fixture"


def test_math_short_answer_row_normalizer_maps_imo_answerbench_fields() -> None:
    normalized = _normalize_imo_answerbench_rows(
        [
            {
                "Problem ID": "imo-1",
                "Problem": "Prove something.",
                "Short Answer": "\\frac{3}{2}",
                "Category": "Geometry",
                "Subcategory": "3d_geometry",
                "Source": "Sharygin 2008",
            }
        ],
        DatasetSpec(
            source=DatasetSource.HUGGINGFACE,
            dataset_id="Hwilner/imo-answerbench",
        ),
    )

    row = _row_mapping(normalized.rows[0])
    metadata = _row_mapping(row["metadata"])

    assert row["item_id"] == "imo-1"
    assert row["problem"] == "Prove something."
    assert row["answer"] == "\\frac{3}{2}"
    assert metadata["category"] == "Geometry"
    assert metadata["subcategory"] == "3d_geometry"
    assert metadata["source"] == "Sharygin 2008"


def test_gpqa_diamond_row_normalizer_parses_inline_option_mapping() -> None:
    normalized = _normalize_gpqa_diamond_rows(
        [
            {
                "question": (
                    "Which option is correct?\n\n"
                    "a) alpha\nb) beta\nc) gamma\nd) delta\n\n"
                    "A. d\nB. a\nC. b\nD. c"
                ),
                "answer": "D",
            }
        ],
        DatasetSpec(
            source=DatasetSource.HUGGINGFACE,
            dataset_id="fingertap/GPQA-Diamond",
        ),
    )

    row = _row_mapping(normalized.rows[0])

    assert row["question"] == "Which option is correct?"
    assert row["options"] == ["delta", "alpha", "beta", "gamma"]
    assert row["expected"] == "D"


def test_babe_row_normalizer_maps_binary_labels_to_choice_benchmark() -> None:
    normalized = _normalize_babe_rows(
        [
            {
                "uuid": "babe-1",
                "text": "This is an article lead.",
                "label": 1,
                "outlet": "Outlet",
                "topic": "topic-a",
                "type": "left",
                "label_opinion": "Expresses writer's opinion",
            }
        ],
        DatasetSpec(
            source=DatasetSource.HUGGINGFACE,
            dataset_id="mediabiasgroup/BABE",
        ),
    )

    row = _row_mapping(normalized.rows[0])
    metadata = _row_mapping(row["metadata"])

    assert row["item_id"] == "babe-1"
    assert row["options"] == ["Entirely factual", "Opinionated or subjective"]
    assert row["expected"] == "B"
    assert metadata["outlet"] == "Outlet"
    assert metadata["topic"] == "topic-a"


def test_mmmlu_row_normalizer_maps_fixed_choice_columns() -> None:
    normalized = _normalize_mmmlu_rows(
        [
            {
                "Unnamed: 0": 7,
                "Question": "What is 2 + 2?",
                "A": "1",
                "B": "4",
                "C": "3",
                "D": "5",
                "Answer": "B",
                "Subject": "math",
            }
        ],
        DatasetSpec(source=DatasetSource.HUGGINGFACE, dataset_id="openai/MMMLU"),
    )

    row = _row_mapping(normalized.rows[0])
    metadata = _row_mapping(row["metadata"])

    assert row["item_id"] == "7"
    assert row["options"] == ["1", "4", "3", "5"]
    assert row["expected"] == "B"
    assert metadata["subject"] == "math"


def test_phybench_row_normalizer_maps_problem_and_answer_fields() -> None:
    normalized = _normalize_phybench_rows(
        [
            {
                "id": 42,
                "tag": "OPTICS",
                "content": "Find the focal length.",
                "answer": "\\frac{1}{2}",
            }
        ],
        DatasetSpec(
            source=DatasetSource.HUGGINGFACE,
            dataset_id="Eureka-Lab/PHYBench",
        ),
    )

    row = _row_mapping(normalized.rows[0])
    metadata = _row_mapping(row["metadata"])

    assert row["item_id"] == "42"
    assert row["problem"] == "Find the focal length."
    assert row["answer"] == "\\frac{1}{2}"
    assert metadata["tag"] == "OPTICS"


def test_frontierscience_row_normalizer_keeps_rubric_and_metadata() -> None:
    normalized = _normalize_frontierscience_rows(
        [
            {
                "problem": "Solve the physics problem.",
                "answer": "Points: 1.0, Item: derive the formula.",
                "subject": "physics",
                "task_group_id": "group-1",
            }
        ],
        DatasetSpec(
            source=DatasetSource.HUGGINGFACE,
            dataset_id="openai/frontierscience",
        ),
    )

    row = _row_mapping(normalized.rows[0])
    metadata = _row_mapping(row["metadata"])

    assert row["prompt_text"] == "Solve the physics problem."
    assert row["expected_response"] == "Points: 1.0, Item: derive the formula."
    assert metadata["subject"] == "physics"
    assert metadata["task_group_id"] == "group-1"


def test_procbench_row_normalizer_exposes_final_label_only_for_v1() -> None:
    normalized = _normalize_procbench_rows(
        [
            {
                "problem_name": "task01_0001",
                "prompt": "Do the thing.",
                "task_name": "task01",
                "label": {"final": ["a", "b"], "intermediate": ["x"]},
            }
        ],
        DatasetSpec(
            source=DatasetSource.HUGGINGFACE,
            dataset_id="ifujisawa/procbench",
        ),
    )

    row = _row_mapping(normalized.rows[0])
    metadata = _row_mapping(row["metadata"])

    assert row["item_id"] == "task01_0001"
    assert row["prompt_text"] == "Do the thing."
    assert row["expected"] == ["a", "b"]
    assert "intermediate" not in row
    assert metadata["task_name"] == "task01"


def test_superchem_row_normalizer_builds_multimodal_prompt_for_language_variant() -> (
    None
):
    normalized = _normalize_superchem_rows(
        [
            {
                "uuid": "chem-1",
                "field": "chemistry",
                "question_type": "multiple_choice",
                "question_en": "What is shown?",
                "question_zh": "图中显示了什么？",
                "question_images": ["/tmp/chem-1.png"],
                "options_en": {"A": "Alpha", "B": "Beta"},
                "options_zh": {"A": "甲", "B": "乙"},
                "answer_en": ["B"],
                "answer_zh": ["B"],
            }
        ],
        DatasetSliceSpec(
            benchmark_id="superchem:zh",
            slice_id="superchem-zh",
            dimensions={"language": "zh"},
            dataset=DatasetSpec(
                source=DatasetSource.HUGGINGFACE,
                dataset_id="ZehuaZhao/SUPERChem",
            ),
        ),
    )

    row = _row_mapping(normalized.rows[0])
    prompt_messages = row["prompt_messages"]
    metadata = _row_mapping(row["metadata"])

    assert isinstance(prompt_messages, list)
    assert prompt_messages[0]["role"] == "user"
    assert isinstance(prompt_messages[0]["content"], list)
    assert prompt_messages[0]["content"][0]["type"] == "text"
    assert prompt_messages[0]["content"][1]["type"] == "image_url"
    assert row["expected"] == "B"
    assert metadata["language"] == "zh"


def test_build_catalog_registry_registers_multiple_providers() -> None:
    registry = catalog.build_catalog_registry(["demo", "openai"])

    assert registry.has_inference_engine("demo")
    assert registry.has_inference_engine("openai")
    assert registry.has_metric("choice_accuracy")
    assert registry.has_metric("math_equivalence")
    assert inspect.isclass(registry.get_metric_registration("choice_accuracy").factory)
    assert inspect.isclass(registry.get_inference_engine_registration("demo").factory)


@pytest.mark.parametrize(
    ("benchmark_id", "metric_type", "metric_module"),
    [
        (
            "simpleqa_verified",
            "SimpleQAVerifiedJudgeMetric",
            "themis.catalog.benchmarks.simpleqa_verified.metric",
        ),
        (
            "healthbench",
            "HealthBenchRubricMetric",
            "themis.catalog.benchmarks.healthbench.metric",
        ),
        (
            "lpfqa",
            "LPFQAJudgeMetric",
            "themis.catalog.benchmarks.lpfqa.metric",
        ),
        (
            "hle:text_only",
            "HLEJudgeMetric",
            "themis.catalog.benchmarks.hle.metric",
        ),
        (
            "frontierscience",
            "FrontierScienceJudgeMetric",
            "themis.catalog.benchmarks.frontierscience.metric",
        ),
    ],
)
def test_judge_backed_benchmarks_use_judge_modules_grouped_by_type(
    benchmark_id: str,
    metric_type: str,
    metric_module: str,
) -> None:
    definition = catalog.get_catalog_benchmark(benchmark_id)
    registry = catalog.build_catalog_registry("demo")

    definition.register_required_components(
        registry,
        judge_model_id="judge-model",
        judge_provider="demo",
    )

    assert definition.primary_metric_id is not None
    registration = registry.get_metric_registration(definition.primary_metric_id)
    metric = registry.get_metric(definition.primary_metric_id)

    assert callable(registration.factory)
    assert type(metric).__name__ == metric_type
    assert type(metric).__module__ == metric_module


@pytest.mark.parametrize(
    ("benchmark_id", "provider_type", "provider_module"),
    [
        (
            "mmlu_pro",
            "BuiltinMMLUProDatasetProvider",
            "themis.catalog.benchmarks.mmlu_pro",
        ),
        (
            "supergpqa",
            "BuiltinSuperGPQADatasetProvider",
            "themis.catalog.benchmarks.supergpqa",
        ),
        (
            "encyclo_k",
            "BuiltinEncycloKDatasetProvider",
            "themis.catalog.benchmarks.encyclo_k",
        ),
        (
            "gpqa_diamond",
            "BuiltinGPQADiamondDatasetProvider",
            "themis.catalog.benchmarks.gpqa_diamond",
        ),
        (
            "babe",
            "BuiltinBABEDatasetProvider",
            "themis.catalog.benchmarks.babe",
        ),
        (
            "simpleqa_verified",
            "BuiltinSimpleQAVerifiedDatasetProvider",
            "themis.catalog.benchmarks.simpleqa_verified.dataset",
        ),
        (
            "healthbench",
            "BuiltinHealthBenchDatasetProvider",
            "themis.catalog.benchmarks.healthbench.dataset",
        ),
        (
            "lpfqa",
            "BuiltinLPFQADatasetProvider",
            "themis.catalog.benchmarks.lpfqa.dataset",
        ),
        (
            "hle:text_only",
            "BuiltinHLEDatasetProvider",
            "themis.catalog.benchmarks.hle.dataset",
        ),
        (
            "rolebench",
            "BuiltinRoleBenchDatasetProvider",
            "themis.catalog.benchmarks.rolebench.dataset",
        ),
        (
            "aime_2026",
            "BuiltinMathArenaDatasetProvider",
            "themis.catalog.benchmarks.aime_2026",
        ),
        (
            "aime_2025",
            "BuiltinMathArenaDatasetProvider",
            "themis.catalog.benchmarks.aime_2025",
        ),
        (
            "hmmt_feb_2025",
            "BuiltinMathArenaDatasetProvider",
            "themis.catalog.benchmarks.hmmt_feb_2025",
        ),
        (
            "hmmt_nov_2025",
            "BuiltinMathArenaDatasetProvider",
            "themis.catalog.benchmarks.hmmt_nov_2025",
        ),
        (
            "apex_2025",
            "BuiltinMathArenaDatasetProvider",
            "themis.catalog.benchmarks.apex_2025",
        ),
        (
            "beyond_aime",
            "BuiltinBeyondAIMEDatasetProvider",
            "themis.catalog.benchmarks.beyond_aime",
        ),
        (
            "imo_answerbench",
            "BuiltinIMOAnswerBenchDatasetProvider",
            "themis.catalog.benchmarks.imo_answerbench",
        ),
        (
            "phybench",
            "BuiltinPHYBenchDatasetProvider",
            "themis.catalog.benchmarks.phybench",
        ),
        (
            "frontierscience",
            "BuiltinFrontierScienceDatasetProvider",
            "themis.catalog.benchmarks.frontierscience.dataset",
        ),
        (
            "mmmlu",
            "BuiltinMMMLUDatasetProvider",
            "themis.catalog.benchmarks.mmmlu",
        ),
        (
            "procbench",
            "BuiltinProcbenchDatasetProvider",
            "themis.catalog.benchmarks.procbench.dataset",
        ),
        (
            "superchem",
            "BuiltinSuperChemDatasetProvider",
            "themis.catalog.benchmarks.superchem.dataset",
        ),
    ],
)
def test_builtin_benchmarks_use_benchmark_specific_dataset_providers(
    benchmark_id: str,
    provider_type: str,
    provider_module: str,
) -> None:
    definition = catalog.get_catalog_benchmark(benchmark_id)

    provider = definition.build_dataset_provider()

    assert type(provider).__name__ == provider_type
    assert type(provider).__module__ == provider_module


@pytest.mark.parametrize(
    "benchmark_id",
    ["simpleqa_verified", "healthbench", "lpfqa", "hle:text_only", "rolebench"],
)
def test_specialized_dataset_providers_inline_their_own_implementation(
    benchmark_id: str,
) -> None:
    definition = catalog.get_catalog_benchmark(benchmark_id)

    provider = definition.build_dataset_provider()
    direct_base_name = type(provider).__bases__[0].__name__

    assert direct_base_name == "BuiltinDatasetProvider"


def test_simpleqa_summary_uses_f1_and_attempted_math_from_metric_details() -> None:
    definition = catalog.get_catalog_benchmark("simpleqa_verified")
    rows = [
        ScoreRow(
            trial_hash="trial-1",
            candidate_id="cand-1",
            metric_id="simpleqa_verified_score",
            score=1.0,
            details={"grade": "CORRECT", "attempted": True},
        ),
        ScoreRow(
            trial_hash="trial-2",
            candidate_id="cand-2",
            metric_id="simpleqa_verified_score",
            score=0.0,
            details={"grade": "INCORRECT", "attempted": True},
        ),
        ScoreRow(
            trial_hash="trial-3",
            candidate_id="cand-3",
            metric_id="simpleqa_verified_score",
            score=0.0,
            details={"grade": "NOT_ATTEMPTED", "attempted": False},
        ),
    ]

    summary = definition.summarize_result(_StubResult(rows))

    assert summary["count"] == 3
    assert summary["correct_rate"] == pytest.approx(1 / 3)
    assert summary["attempted_rate"] == pytest.approx(2 / 3)
    assert summary["accuracy_given_attempted"] == pytest.approx(0.5)
    assert summary["f1"] == pytest.approx(0.4)


def test_healthbench_summary_reports_mean_score_and_tag_breakdowns() -> None:
    definition = catalog.get_catalog_benchmark("healthbench")
    rows = [
        ScoreRow(
            trial_hash="trial-1",
            candidate_id="cand-1",
            metric_id="healthbench_score",
            score=1.0,
            details={"example_tags": ["theme:communication", "axis:safety"]},
        ),
        ScoreRow(
            trial_hash="trial-2",
            candidate_id="cand-2",
            metric_id="healthbench_score",
            score=0.5,
            details={"example_tags": ["theme:communication"]},
        ),
    ]

    summary = definition.summarize_result(_StubResult(rows))
    tag_means = _json_mapping(summary["tag_means"])

    assert summary["count"] == 2
    assert summary["mean_overall_score"] == pytest.approx(0.75)
    assert tag_means["theme:communication"] == pytest.approx(0.75)
    assert tag_means["axis:safety"] == pytest.approx(1.0)


def test_hle_summary_reports_accuracy_ci_calibration_and_skipped_images() -> None:
    definition = catalog.get_catalog_benchmark("hle:text_only")
    rows = [
        ScoreRow(
            trial_hash="trial-1",
            candidate_id="cand-1",
            metric_id="hle_accuracy",
            score=1.0,
            details={"correct": True, "confidence": 100},
        ),
        ScoreRow(
            trial_hash="trial-2",
            candidate_id="cand-2",
            metric_id="hle_accuracy",
            score=0.0,
            details={"correct": False, "confidence": 20},
        ),
    ]

    summary = definition.summarize_result(
        _StubResult(rows, scan_stats={"skipped_image_count": 4})
    )
    confidence_interval_half_width = summary["confidence_interval_half_width"]

    assert summary["count"] == 2
    assert summary["accuracy"] == pytest.approx(0.5)
    assert isinstance(confidence_interval_half_width, int | float)
    assert confidence_interval_half_width > 0.0
    assert summary["calibration_error"] == pytest.approx(0.3)
    assert summary["skipped_image_count"] == 4


def test_hle_summary_groups_multi_variant_runs_by_variant() -> None:
    definition = catalog.get_catalog_benchmark("hle:text_only,no_tool")
    rows = [
        SimpleNamespace(
            trial_hash="trial-1",
            candidate_id="cand-1",
            metric_id="hle_accuracy",
            score=1.0,
            details={"correct": True, "confidence": 100},
        ),
        SimpleNamespace(
            trial_hash="trial-2",
            candidate_id="cand-2",
            metric_id="hle_accuracy",
            score=0.0,
            details={"correct": False, "confidence": 20},
        ),
    ]

    summary = definition.summarize_result(
        _StubResult(
            rows,
            trial_summaries=[
                SimpleNamespace(
                    trial_hash="trial-1",
                    slice_id="hle-text_only",
                    dimensions={"hle_variant": "text_only"},
                ),
                SimpleNamespace(
                    trial_hash="trial-2",
                    slice_id="hle-no_tool",
                    dimensions={"hle_variant": "no_tool"},
                ),
            ],
        )
    )

    assert summary["variant_ids"] == ["text_only", "no_tool"]
    variants = _json_mapping(summary["variants"])
    assert _json_mapping(variants["text_only"])["accuracy"] == pytest.approx(1.0)
    assert _json_mapping(variants["no_tool"])["accuracy"] == pytest.approx(0.0)


def test_hle_dataset_provider_applies_text_only_variant_filtering() -> None:
    definition = catalog.get_catalog_benchmark("hle:text_only")
    provider = cast(Any, definition.build_dataset_provider())
    slice_spec = DatasetSliceSpec(
        benchmark_id="hle:text_only",
        slice_id="hle-text_only",
        dimensions={"hle_variant": "text_only"},
        dataset=DatasetSpec(source=DatasetSource.HUGGINGFACE, dataset_id="cais/hle"),
    )

    rows = provider.prepare_rows(
        [
            {"id": "hle-1", "question": "Text row", "answer": "4", "image": ""},
            {"id": "hle-2", "question": "Image row", "answer": "5", "image": "img"},
        ],
        slice_spec,
    )

    assert [row["item_id"] for row in rows.rows] == ["hle-1"]
    assert rows.stats["skipped_image_count"] == 1
    assert rows.rows[0]["metadata"] == {"hle_variant": "text_only", "text_only": "true"}


def test_hle_dataset_provider_keeps_image_rows_for_no_tool_variant() -> None:
    definition = catalog.get_catalog_benchmark("hle:no_tool")
    provider = cast(Any, definition.build_dataset_provider())
    slice_spec = DatasetSliceSpec(
        benchmark_id="hle:no_tool",
        slice_id="hle-no_tool",
        dimensions={"hle_variant": "no_tool"},
        dataset=DatasetSpec(source=DatasetSource.HUGGINGFACE, dataset_id="cais/hle"),
    )

    rows = provider.prepare_rows(
        [
            {"id": "hle-1", "question": "Text row", "answer": "4", "image": ""},
            {"id": "hle-2", "question": "Image row", "answer": "5", "image": "img"},
        ],
        slice_spec,
    )

    assert [row["item_id"] for row in rows.rows] == ["hle-1", "hle-2"]
    assert rows.stats["skipped_image_count"] == 0
    assert rows.rows[1]["metadata"] == {"hle_variant": "no_tool"}


def test_rolebench_loader_fetches_exact_variant_files_and_combines_subsets() -> None:
    from themis.catalog.benchmarks.rolebench.dataset import _load_rolebench_rows

    file_map = {
        "https://huggingface.co/datasets/ZenMoore/RoleBench/resolve/main/profiles-eng/desc.json": '{"Wizard":"Speaks cryptically."}',
        "https://huggingface.co/datasets/ZenMoore/RoleBench/resolve/main/rolebench-eng/instruction-generalization/general/test.jsonl": "\n".join(
            [
                '{"role":"Wizard","question":"General question?","generated":["General answer."]}',
                "",
            ]
        ),
        "https://huggingface.co/datasets/ZenMoore/RoleBench/resolve/main/rolebench-eng/instruction-generalization/role_specific/test.jsonl": '{"role":"Wizard","question":"Specific question?","generated":["Specific answer."]}',
    }

    def _fake_urlopen(request: object):
        url = getattr(request, "full_url", request)
        assert isinstance(url, str)
        return io.BytesIO(file_map[url].encode("utf-8"))

    rows = _load_rolebench_rows(
        "ZenMoore/RoleBench",
        "test",
        None,
        config_name="instruction_generalization_eng",
        urlopen=_fake_urlopen,
    )

    assert len(rows) == 2
    assert [row["subset"] for row in rows] == ["general", "role_specific"]
    assert rows[0]["desc"] == "Speaks cryptically."
    assert rows[1]["question"] == "Specific question?"


def test_rolebench_dataset_provider_normalizes_expected_ids_and_metadata() -> None:
    definition = catalog.get_catalog_benchmark(
        "rolebench:instruction_generalization_eng"
    )
    provider = cast(Any, definition.build_dataset_provider())
    slice_spec = DatasetSliceSpec(
        benchmark_id="rolebench:instruction_generalization_eng",
        slice_id="rolebench-instruction_generalization_eng",
        dimensions={"rolebench_variant": "instruction_generalization_eng"},
        dataset=DatasetSpec(
            source=DatasetSource.HUGGINGFACE,
            dataset_id="ZenMoore/RoleBench",
            config_name="instruction_generalization_eng",
        ),
    )

    rows = provider.prepare_rows(
        [
            {
                "role": "Wizard",
                "desc": "Speaks cryptically.",
                "question": "What is the prophecy?",
                "generated": ["The moon will dim."],
                "subset": "general",
                "source_line_number": 7,
            }
        ],
        slice_spec,
    )

    assert (
        rows.rows[0]["item_id"] == "rolebench-instruction_generalization_eng-general-7"
    )
    assert rows.rows[0]["expected"] == "The moon will dim."
    assert rows.rows[0]["metadata"] == {
        "rolebench_variant": "instruction_generalization_eng",
        "subset": "general",
        "role": "Wizard",
    }


def test_rolebench_dataset_provider_uses_config_name_to_isolate_variant_loader() -> (
    None
):
    definition = catalog.get_catalog_benchmark("rolebench")
    benchmark = definition.build_benchmark(model_id="demo-model", provider="demo")
    seen_config_names: list[str | None] = []

    def _loader(
        dataset_id: str,
        split: str,
        revision: str | None,
        *,
        config_name: str | None = None,
    ) -> list[dict[str, object]]:
        del dataset_id, split, revision
        seen_config_names.append(config_name)
        return [
            {
                "role": "Wizard",
                "desc": "Speaks cryptically.",
                "question": "What is the prophecy?",
                "generated": ["The moon will dim."],
                "subset": "general",
                "source_line_number": 1,
            }
        ]

    provider = cast(Any, definition.build_dataset_provider(huggingface_loader=_loader))

    first_rows = cast(
        list[dict[str, object]], provider.scan(benchmark.slices[0], DatasetQuerySpec())
    )
    second_rows = cast(
        list[dict[str, object]], provider.scan(benchmark.slices[1], DatasetQuerySpec())
    )

    assert seen_config_names == [
        "instruction_generalization_eng",
        "role_generalization_eng",
    ]
    first_metadata = cast(dict[str, object], first_rows[0]["metadata"])
    second_metadata = cast(dict[str, object], second_rows[0]["metadata"])
    assert first_metadata["rolebench_variant"] == "instruction_generalization_eng"
    assert second_metadata["rolebench_variant"] == "role_generalization_eng"


def test_inspect_huggingface_dataset_uses_loader_hooks_for_schema_and_samples() -> None:
    summary = catalog.inspect_huggingface_dataset(
        "demo/qa",
        split="test",
        metadata_loader=lambda dataset_id, revision: {
            "dataset_id": dataset_id,
            "gated": False,
            "splits": ["train", "test"],
            "modalities": ["text"],
        },
        row_loader=lambda dataset_id, split, revision: [
            {"question": "2 + 2", "answer": "4"},
            {"question": "3 + 3", "answer": "6"},
        ],
    )
    fields = _json_mapping(summary["fields"])
    samples = summary["samples"]

    assert summary["dataset_id"] == "demo/qa"
    assert summary["splits"] == ["train", "test"]
    assert summary["modalities"] == ["text"]
    assert summary["row_count"] == 2
    assert fields["question"] == "str"
    assert isinstance(samples, list)
    assert len(samples) == 2
    assert summary["suggested_prompt_field"] == "question"
    assert summary["suggested_answer_field"] == "answer"
    assert summary["suggested_item_id_field"] is None
    assert summary["suggested_metadata_keys"] == []


def test_inspect_huggingface_dataset_emits_math_wiring_hints() -> None:
    summary = catalog.inspect_huggingface_dataset(
        "MathArena/aime_2026",
        split="train",
        metadata_loader=lambda dataset_id, revision: {
            "dataset_id": dataset_id,
            "gated": False,
            "splits": ["train"],
            "modalities": ["text"],
        },
        row_loader=lambda dataset_id, split, revision: [
            {
                "problem_idx": 1,
                "problem": "Find x.",
                "answer": "42",
                "problem_type": ["Algebra"],
            }
        ],
    )

    assert summary["suggested_prompt_field"] == "problem"
    assert summary["suggested_answer_field"] == "answer"
    assert summary["suggested_item_id_field"] == "problem_idx"
    assert summary["suggested_metadata_keys"] == ["problem_type"]


def test_math_equivalence_metric_scores_equivalent_answers(monkeypatch) -> None:
    metric = MathEquivalenceMetric()

    fake_math_verify = SimpleNamespace(
        parse=lambda value: f"parsed:{value}",
        verify=lambda gold, answer: gold == "parsed:0.5" and answer == "parsed:1/2",
    )
    monkeypatch.setattr(
        "themis.catalog.runtime.metrics.common.import_optional",
        lambda module_name, *, extra: fake_math_verify,
    )
    candidate = SimpleNamespace(
        best_extraction=lambda: SimpleNamespace(success=True, parsed_answer="1/2")
    )

    score = metric.score(None, candidate, {"expected": "0.5"})

    assert score.metric_id == "math_equivalence"
    assert score.value == 1.0
    assert score.details["candidate_answer"] == "1/2"
    assert score.details["gold_answer"] == "0.5"


def test_math_equivalence_metric_returns_install_hint_when_optional_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric = MathEquivalenceMetric()

    def _raise_missing_optional(module_name: str, *, extra: str):
        raise ThemisError(
            code=ErrorCode.MISSING_OPTIONAL_DEPENDENCY,
            message=f'Install it with `uv add "themis-eval[{extra}]"`.',
        )

    monkeypatch.setattr(
        "themis.catalog.runtime.metrics.common.import_optional",
        _raise_missing_optional,
    )
    candidate = SimpleNamespace(
        best_extraction=lambda: SimpleNamespace(success=True, parsed_answer="1/2")
    )

    score = metric.score(None, candidate, {"expected": "0.5"})

    assert score.value == 0.0
    assert score.error == 'Install it with `uv add "themis-eval[math]"`.'


def test_rolebench_rouge_metric_scores_overlap_using_rouge_l_f1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from themis.catalog.benchmarks.rolebench.metric import RoleBenchRougeMetric

    metric = RoleBenchRougeMetric()

    class _FakeRougeScorer:
        def __init__(self, metrics: list[str], *, use_stemmer: bool) -> None:
            assert metrics == ["rougeL"]
            assert use_stemmer is True

        def score(self, target: str, prediction: str) -> dict[str, object]:
            assert target == "The moon will dim tonight."
            assert prediction == "The moon will dim."
            return {"rougeL": SimpleNamespace(precision=0.9, recall=0.6, fmeasure=0.72)}

    monkeypatch.setattr(
        "themis.catalog.benchmarks.rolebench.metric.import_optional",
        lambda module_name, *, extra: SimpleNamespace(RougeScorer=_FakeRougeScorer),
    )
    candidate = SimpleNamespace(
        inference=SimpleNamespace(raw_text="The moon will dim.")
    )

    score = metric.score(None, candidate, {"expected": "The moon will dim tonight."})

    assert score.metric_id == "rolebench_rouge_l_f1"
    assert score.value == pytest.approx(0.72)
    assert score.details == {
        "precision": pytest.approx(0.9),
        "recall": pytest.approx(0.6),
        "f1": pytest.approx(0.72),
    }


def test_rolebench_rouge_metric_returns_install_hint_when_optional_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from themis.catalog.benchmarks.rolebench.metric import RoleBenchRougeMetric

    metric = RoleBenchRougeMetric()

    def _raise_missing_optional(module_name: str, *, extra: str):
        raise ThemisError(
            code=ErrorCode.MISSING_OPTIONAL_DEPENDENCY,
            message=f'Install it with `uv add "themis-eval[{extra}]"`.',
        )

    monkeypatch.setattr(
        "themis.catalog.benchmarks.rolebench.metric.import_optional",
        _raise_missing_optional,
    )
    candidate = SimpleNamespace(
        inference=SimpleNamespace(raw_text="The moon will dim.")
    )

    score = metric.score(None, candidate, {"expected": "The moon will dim tonight."})

    assert score.value == 0.0
    assert score.error == 'Install it with `uv add "themis-eval[text-metrics]"`.'


def test_rolebench_summary_groups_multi_variant_runs_by_variant() -> None:
    definition = catalog.get_catalog_benchmark("rolebench")
    rows = [
        SimpleNamespace(
            trial_hash="trial-1",
            candidate_id="cand-1",
            metric_id="rolebench_rouge_l_f1",
            score=0.8,
            prompt_variant_id="rolebench-instruction_generalization_eng-default",
        ),
        SimpleNamespace(
            trial_hash="trial-2",
            candidate_id="cand-2",
            metric_id="rolebench_rouge_l_f1",
            score=0.5,
            prompt_variant_id="rolebench-role_generalization_eng-default",
        ),
    ]

    summary = definition.summarize_result(
        _StubResult(
            rows,
            trial_summaries=[
                SimpleNamespace(
                    trial_hash="trial-1",
                    slice_id="rolebench-instruction_generalization_eng",
                    dimensions={"rolebench_variant": "instruction_generalization_eng"},
                ),
                SimpleNamespace(
                    trial_hash="trial-2",
                    slice_id="rolebench-role_generalization_eng",
                    dimensions={"rolebench_variant": "role_generalization_eng"},
                ),
            ],
        )
    )

    assert summary["metric_id"] == "rolebench_rouge_l_f1"
    assert summary["count"] == 2
    assert summary["mean"] == pytest.approx(0.65)
    assert summary["variant_ids"] == [
        "instruction_generalization_eng",
        "role_generalization_eng",
    ]
    variants = _json_mapping(summary["variants"])
    assert _json_mapping(variants["instruction_generalization_eng"])[
        "mean"
    ] == pytest.approx(0.8)
    assert _json_mapping(variants["role_generalization_eng"])["mean"] == pytest.approx(
        0.5
    )


def test_catalog_preview_rows_are_available_for_all_builtin_benchmarks() -> None:
    for benchmark_id in catalog.list_catalog_benchmarks():
        resolved_benchmark_id = (
            "hle:text_only" if benchmark_id == "hle" else benchmark_id
        )
        definition = catalog.get_catalog_benchmark(resolved_benchmark_id)
        assert definition.preview_rows_loader is not None
        rows = definition.preview_rows_loader(definition)
        assert rows
        assert isinstance(rows[0], dict)


def test_load_huggingface_rows_disables_image_decoding_for_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeImageFeature:
        def __init__(self, *, decode: bool = True) -> None:
            self.decode = decode

    class _FakeDataset:
        def __init__(self) -> None:
            self.features = {
                "question": "text",
                "image_preview": _FakeImageFeature(),
                "rationale_image": _FakeImageFeature(),
            }
            self.cast_calls: list[tuple[str, bool]] = []

        def cast_column(self, name: str, feature: object):
            assert isinstance(feature, _FakeImageFeature)
            self.cast_calls.append((name, feature.decode))
            self.features[name] = feature
            return self

        def __iter__(self):
            yield {"question": "What is 2 + 2?", "answer": "4"}

    fake_dataset = _FakeDataset()

    class _FakeDatasetsModule:
        Image = _FakeImageFeature

        @staticmethod
        def load_dataset(dataset_id: str, *, split: str, revision: str | None = None):
            assert dataset_id == "cais/hle"
            assert split == "test"
            assert revision is None
            return fake_dataset

    monkeypatch.setattr(
        dataset_common,
        "import_optional",
        lambda module_name, *, extra: _FakeDatasetsModule,
    )

    rows = catalog.load_huggingface_rows("cais/hle", "test")

    assert rows == [{"item_id": "item-1", "question": "What is 2 + 2?", "answer": "4"}]
    assert fake_dataset.cast_calls == [
        ("image_preview", False),
        ("rationale_image", False),
    ]


def test_load_huggingface_rows_retries_healthbench_with_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DatasetGenerationError(Exception):
        pass

    class _FakeStreamingDataset:
        features = {"prompt": "list", "rubrics": "list"}

        def __iter__(self):
            yield {
                "prompt_id": "hb-1",
                "prompt": [{"role": "user", "content": "Help me."}],
                "rubrics": [{"criterion": "safe", "points": 1, "tags": []}],
            }

    class _FakeDatasetsModule:
        DatasetGenerationError = _DatasetGenerationError

        @staticmethod
        def load_dataset(
            dataset_id: str,
            *,
            split: str,
            revision: str | None = None,
            streaming: bool = False,
        ):
            assert dataset_id == "openai/healthbench"
            assert split == "test"
            assert revision is None
            if streaming:
                return _FakeStreamingDataset()
            raise _DatasetGenerationError("broken non-streaming cast")

    monkeypatch.setattr(
        dataset_common,
        "import_optional",
        lambda module_name, *, extra: _FakeDatasetsModule,
    )

    rows = catalog.load_huggingface_rows("openai/healthbench", "test")

    assert rows == [
        {
            "item_id": "item-1",
            "prompt_id": "hb-1",
            "prompt": [{"role": "user", "content": "Help me."}],
            "rubrics": [{"criterion": "safe", "points": 1, "tags": []}],
        }
    ]
