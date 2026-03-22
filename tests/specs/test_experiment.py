from themis.types.enums import PromptRole
import importlib
import importlib.util

import pytest
from pydantic import ValidationError

from themis.specs.experiment import (
    BatchExecutionBackendSpec,
    ExecutionPolicySpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ItemSamplingSpec,
    LocalExecutionBackendSpec,
    PostgresBlobStorageSpec,
    ProjectSpec,
    PromptMessage,
    PromptTurnSpec,
    SqliteBlobStorageSpec,
    StorageSpec,
    TrialSpec,
    ExperimentSpec,
    PromptTemplateSpec,
    RuntimeContext,
    WorkerPoolExecutionBackendSpec,
)
from themis.specs.foundational import (
    DatasetSpec,
    ExtractorChainSpec,
    EvaluationSpec,
    GenerationSpec,
    McpServerSpec,
    ModelSpec,
    OutputTransformSpec,
    TaskSpec,
    ToolSpec,
)
from themis.errors import SpecValidationError
from themis.types.enums import (
    DatasetSource,
    CompressionCodec,
    ErrorCode,
    ResponseFormat,
    SamplingKind,
    StorageBackend,
)


def _make_model():
    return ModelSpec(model_id="gpt-4", provider="openai")


def _make_task():
    return TaskSpec(
        task_id="t1",
        dataset=DatasetSpec(source=DatasetSource.MEMORY),
        generation=GenerationSpec(),
        output_transforms=[
            OutputTransformSpec(
                name="json",
                extractor_chain=ExtractorChainSpec(extractors=["json"]),
            )
        ],
        evaluations=[EvaluationSpec(name="judge", transform="json", metrics=["em"])],
    )


def _task_resolution_module():
    assert importlib.util.find_spec("themis.orchestration.task_resolution") is not None
    return importlib.import_module("themis.orchestration.task_resolution")


def test_inference_params():
    params = InferenceParamsSpec(
        temperature=0.5,
        top_p=0.9,
        max_tokens=100,
        response_format=ResponseFormat.JSON,
    )
    assert params.temperature == 0.5
    assert params.top_p == 0.9
    assert params.max_tokens == 100
    assert params.response_format == ResponseFormat.JSON
    assert params.model_dump(mode="json")["response_format"] == "json"


def test_prompt_template_spec():
    spec = PromptTemplateSpec(
        messages=[{"role": "developer", "content": "Hello {name}"}],
        follow_up_turns=[
            {
                "messages": [
                    {"role": "user", "content": "Question: {name}?"},
                    {"role": "assistant", "content": "Thinking..."},
                ]
            }
        ],
    )
    assert spec.messages[0].content == "Hello {name}"
    assert spec.messages[0].role == PromptRole.DEVELOPER
    assert spec.follow_up_turns == [
        PromptTurnSpec(
            messages=[
                PromptMessage(role=PromptRole.USER, content="Question: {name}?"),
                PromptMessage(role=PromptRole.ASSISTANT, content="Thinking..."),
            ]
        )
    ]


def test_prompt_template_spec_rejects_unknown_message_keys():
    with pytest.raises(ValidationError):
        PromptTemplateSpec(
            messages=[{"role": "user", "content": "Hello", "extra": "bad"}]
        )


def test_prompt_message_requires_role_and_content():
    message = PromptMessage(role=PromptRole.DEVELOPER, content="Done")

    assert message.role == "developer"
    assert message.content == "Done"

    with pytest.raises(ValidationError):
        PromptMessage.model_validate({"foo": "bar"})


def test_prompt_message_accepts_multimodal_content_parts() -> None:
    message = PromptMessage.model_validate(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this figure."},
                {"type": "image_url", "image_url": "https://example.com/plot.png"},
            ],
        }
    )

    assert isinstance(message.content, list)
    rendered = message.model_dump(mode="json")
    assert rendered["content"][0]["type"] == "text"
    assert rendered["content"][1]["type"] == "image_url"


def test_dataset_spec_accepts_huggingface_config_name() -> None:
    dataset = DatasetSpec(
        source=DatasetSource.HUGGINGFACE,
        dataset_id="openai/MMMLU",
        config_name="AR_XY",
        split="test",
    )

    assert dataset.config_name == "AR_XY"


def test_trial_spec_validation():
    trial = TrialSpec(
        trial_id="t_1",
        model=_make_model(),
        task=_make_task(),
        item_id="item_0",
        prompt=PromptTemplateSpec(messages=[{"role": "user", "content": "Q: {q}"}]),
        params=InferenceParamsSpec(),
        tools=[
            ToolSpec(
                id="search",
                description="Search",
                input_schema={"type": "object"},
            )
        ],
        mcp_servers=[
            McpServerSpec(
                id="dice",
                server_label="dice",
                server_url="https://dmcp-server.deno.dev/sse",
            )
        ],
        candidate_count=5,
    )
    assert trial.candidate_count == 5
    assert trial.tools[0].id == "search"
    assert trial.mcp_servers[0].id == "dice"

    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        TrialSpec(
            trial_id="t_2",
            model=_make_model(),
            task=_make_task(),
            item_id="item_0",
            prompt=PromptTemplateSpec(messages=[]),
            params=InferenceParamsSpec(),
            candidate_count=0,
        )


def test_experiment_spec():
    spec = ExperimentSpec(
        models=[_make_model()],
        tasks=[_make_task()],
        prompt_templates=[PromptTemplateSpec(messages=[])],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(temperature=0.0)]),
        tools=[
            ToolSpec(
                id="search",
                description="Search",
                input_schema={"type": "object"},
            )
        ],
        mcp_servers=[
            McpServerSpec(
                id="dice",
                server_label="dice",
                server_url="https://dmcp-server.deno.dev/sse",
            )
        ],
        item_sampling=ItemSamplingSpec(kind="all"),
    )
    assert len(spec.models) == 1
    assert spec.item_sampling.kind == SamplingKind.ALL
    assert spec.tools[0].id == "search"
    assert spec.mcp_servers[0].id == "dice"


def test_experiment_spec_empty_validation():
    with pytest.raises(ValidationError, match="must have at least one model"):
        ExperimentSpec(
            models=[],
            tasks=[_make_task()],
            prompt_templates=[PromptTemplateSpec(messages=[])],
            inference_grid=InferenceGridSpec(
                params=[InferenceParamsSpec(temperature=0.0)]
            ),
        )


def test_project_spec():
    proj = ProjectSpec(
        project_name="eval_run_1",
        researcher_id="researcher_1",
        global_seed=42,
        storage=SqliteBlobStorageSpec(root_dir="./runs/eval_run_1"),
        execution_policy=ExecutionPolicySpec(),
        tools=[
            ToolSpec(
                id="search",
                description="Search",
                input_schema={"type": "object"},
            )
        ],
        mcp_servers=[
            McpServerSpec(
                id="calendar",
                server_label="google_calendar",
                connector_id="connector_googlecalendar",
                authorization_secret_name="GOOGLE_CALENDAR_TOKEN",
            )
        ],
    )
    assert proj.project_name == "eval_run_1"
    assert proj.storage.backend == StorageBackend.SQLITE_BLOB
    assert proj.storage.compression == CompressionCodec.ZSTD
    assert isinstance(proj.execution_backend, LocalExecutionBackendSpec)
    assert proj.execution_backend.kind == "local"
    assert proj.tools[0].id == "search"
    assert proj.mcp_servers[0].connector_id == "connector_googlecalendar"


def test_project_spec_rejects_duplicate_mcp_server_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate MCP server id"):
        ProjectSpec(
            project_name="eval_run_1",
            researcher_id="researcher_1",
            global_seed=42,
            storage=SqliteBlobStorageSpec(root_dir="./runs/eval_run_1"),
            execution_policy=ExecutionPolicySpec(),
            mcp_servers=[
                McpServerSpec(
                    id="dice",
                    server_label="dice",
                    server_url="https://dmcp-server.deno.dev/sse",
                ),
                McpServerSpec(
                    id="dice",
                    server_label="dice-override",
                    server_url="https://example.com/mcp",
                ),
            ],
        )


def test_runtime_context_excludes_tool_handlers_from_json_dump():
    runtime = RuntimeContext(
        run_labels={"phase": "smoke"},
        tool_handlers={"search": object()},
    )

    assert runtime.model_dump(mode="json") == {
        "secrets": {},
        "environment": {},
        "run_labels": {"phase": "smoke"},
        "candidate_seed": None,
        "resume": None,
        "tool_handler_versions": {},
    }


def test_runtime_context_tool_handler_versions_defaults_empty() -> None:
    runtime = RuntimeContext()
    assert runtime.tool_handler_versions == {}


def test_runtime_context_tool_handler_versions_stored_and_serialized() -> None:
    runtime = RuntimeContext(
        tool_handler_versions={"search": "1.2.0", "calculator": "0.3.1"}
    )
    assert runtime.tool_handler_versions["search"] == "1.2.0"
    assert runtime.tool_handler_versions["calculator"] == "0.3.1"
    dumped = runtime.model_dump(mode="json")
    assert dumped["tool_handler_versions"] == {"search": "1.2.0", "calculator": "0.3.1"}


def test_runtime_context_tool_handler_versions_independent_of_tool_handlers() -> None:
    """Versions are persisted; handlers are excluded from serialization."""
    runtime = RuntimeContext(
        tool_handlers={"search": object()},
        tool_handler_versions={"search": "2.0.0"},
    )
    dumped = runtime.model_dump(mode="json")
    assert "tool_handlers" not in dumped
    assert dumped["tool_handler_versions"] == {"search": "2.0.0"}


def test_project_spec_supports_postgres_blob_storage():
    proj = ProjectSpec.model_validate(
        {
            "project_name": "eval_run_1",
            "researcher_id": "researcher_1",
            "global_seed": 42,
            "storage": {
                "backend": "postgres_blob",
                "database_url": "postgresql://localhost:5432/themis",
                "blob_root_dir": "./runs/eval_run_1/blobs",
            },
            "execution_policy": {},
        }
    )

    assert isinstance(proj.storage, PostgresBlobStorageSpec)
    assert proj.storage.backend == StorageBackend.POSTGRES_BLOB
    assert proj.storage.database_url == "postgresql://localhost:5432/themis"
    assert proj.storage.blob_root_dir == "./runs/eval_run_1/blobs"


def test_project_spec_supports_worker_pool_and_batch_execution_backends():
    worker_pool_project = ProjectSpec.model_validate(
        {
            "project_name": "eval_run_1",
            "researcher_id": "researcher_1",
            "global_seed": 42,
            "storage": {
                "backend": "sqlite_blob",
                "root_dir": "./runs/eval_run_1",
            },
            "execution_policy": {},
            "execution_backend": {
                "kind": "worker_pool",
                "lease_ttl_seconds": 180,
                "poll_interval_seconds": 5,
                "worker_tags": ["gpu:a100", "provider:openai"],
            },
        }
    )
    batch_project = ProjectSpec.model_validate(
        {
            "project_name": "eval_run_2",
            "researcher_id": "researcher_2",
            "global_seed": 42,
            "storage": {
                "backend": "sqlite_blob",
                "root_dir": "./runs/eval_run_2",
            },
            "execution_policy": {},
            "execution_backend": {
                "kind": "batch",
                "provider": "openai",
                "poll_interval_seconds": 30,
                "max_batch_items": 250,
            },
        }
    )

    assert isinstance(
        worker_pool_project.execution_backend, WorkerPoolExecutionBackendSpec
    )
    assert worker_pool_project.execution_backend.worker_tags == [
        "gpu:a100",
        "provider:openai",
    ]
    assert isinstance(batch_project.execution_backend, BatchExecutionBackendSpec)
    assert batch_project.execution_backend.provider == "openai"


def test_execution_policy_supports_retryable_error_codes():
    policy = ExecutionPolicySpec(
        max_retries=4,
        retryable_error_codes=[
            ErrorCode.PROVIDER_TIMEOUT,
            ErrorCode.PROVIDER_RATE_LIMIT,
        ],
    )

    assert policy.max_retries == 4
    assert policy.retryable_error_codes == [
        ErrorCode.PROVIDER_TIMEOUT,
        ErrorCode.PROVIDER_RATE_LIMIT,
    ]


def test_runtime_context_defaults_and_immutability():
    runtime = RuntimeContext()

    assert runtime.secrets == {}
    assert runtime.environment == {}
    assert runtime.run_labels == {}

    with pytest.raises(Exception):
        runtime.environment = {"suite": "tests"}


def test_inference_grid_spec_tracks_param_bases_and_overrides():
    spec = InferenceGridSpec(
        params=[InferenceParamsSpec(temperature=0.0, max_tokens=32)],
        overrides={"max_tokens": [32, 64], "top_p": [0.9]},
    )

    assert len(spec.params) == 1
    assert spec.overrides["max_tokens"] == [32, 64]


def test_item_sampling_subset_requires_count():
    with pytest.raises(ValidationError, match="requires a positive count"):
        ItemSamplingSpec(kind="subset")


def test_item_sampling_stratified_requires_strata_field():
    with pytest.raises(ValidationError, match="requires strata_field"):
        ItemSamplingSpec(kind="stratified", count=2)


def test_item_sampling_classmethods():
    assert ItemSamplingSpec.all().kind == SamplingKind.ALL

    subset = ItemSamplingSpec.subset(2, seed=7)
    assert subset.kind == SamplingKind.SUBSET
    assert subset.count == 2
    assert subset.seed == 7

    stratified = ItemSamplingSpec.stratified(3, strata_field="difficulty", seed=9)
    assert stratified.kind == SamplingKind.STRATIFIED
    assert stratified.count == 3
    assert stratified.strata_field == "difficulty"
    assert stratified.seed == 9


def test_item_sampling_supports_item_ids_and_metadata_filters():
    sampling = ItemSamplingSpec(
        kind="subset",
        count=1,
        item_ids=["item-1", "item-3"],
        metadata_filters={"difficulty": "hard", "split": "eval"},
    )

    assert sampling.item_ids == ["item-1", "item-3"]
    assert sampling.metadata_filters == {"difficulty": "hard", "split": "eval"}


def test_item_sampling_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        ItemSamplingSpec(kind="unknown")


def test_project_spec_rejects_string_seed_under_strict_mode():
    with pytest.raises(ValidationError):
        ProjectSpec(
            project_name="eval_run_1",
            researcher_id="researcher_1",
            global_seed="42",
            storage=StorageSpec(backend="sqlite_blob", root_dir="./runs/eval_run_1"),
            execution_policy=ExecutionPolicySpec(),
        )


def test_inference_params_rejects_string_numbers_under_strict_mode():
    with pytest.raises(ValidationError):
        InferenceParamsSpec(max_tokens="100")


def test_runtime_context_resume_is_typed():
    with pytest.raises(ValidationError):
        RuntimeContext.model_validate({"resume": object()})


def test_transform_identity_is_deterministic():
    transform = OutputTransformSpec(
        name="json",
        extractor_chain=ExtractorChainSpec(extractors=["json"]),
    )
    assert (
        transform.spec_hash
        == OutputTransformSpec(
            name="json",
            extractor_chain=ExtractorChainSpec(extractors=["json"]),
        ).spec_hash
    )


def test_transform_identity_includes_display_name():
    transform_a = OutputTransformSpec(
        name="json",
        extractor_chain=ExtractorChainSpec(extractors=["json"]),
    )
    transform_b = OutputTransformSpec(
        name="parsed_output",
        extractor_chain=ExtractorChainSpec(extractors=["json"]),
    )

    assert transform_a.spec_hash != transform_b.spec_hash


def test_evaluation_identity_changes_when_referenced_transform_changes():
    module = _task_resolution_module()
    resolve_task_stages = getattr(module, "resolve_task_stages")

    task = TaskSpec(
        task_id="qa",
        dataset=DatasetSpec(source=DatasetSource.MEMORY),
        generation=GenerationSpec(),
        output_transforms=[
            OutputTransformSpec(
                name="json",
                extractor_chain=ExtractorChainSpec(extractors=["json"]),
            )
        ],
        evaluations=[EvaluationSpec(name="judge", transform="json", metrics=["em"])],
    )
    resolved_a = resolve_task_stages(task)

    task_b = task.model_copy(
        update={
            "output_transforms": [
                OutputTransformSpec(
                    name="json",
                    extractor_chain=ExtractorChainSpec(extractors=["regex"]),
                )
            ]
        }
    )
    resolved_b = resolve_task_stages(task_b)

    assert (
        resolved_a.evaluations[0].evaluation_hash
        != resolved_b.evaluations[0].evaluation_hash
    )


def test_evaluation_identity_includes_display_name():
    module = _task_resolution_module()
    resolve_task_stages = getattr(module, "resolve_task_stages")

    task_a = TaskSpec(
        task_id="qa",
        dataset=DatasetSpec(source=DatasetSource.MEMORY),
        generation=GenerationSpec(),
        output_transforms=[
            OutputTransformSpec(
                name="json",
                extractor_chain=ExtractorChainSpec(extractors=["json"]),
            )
        ],
        evaluations=[EvaluationSpec(name="judge_a", transform="json", metrics=["em"])],
    )
    task_b = task_a.model_copy(
        update={
            "evaluations": [
                EvaluationSpec(name="judge_b", transform="json", metrics=["em"])
            ]
        }
    )

    resolved_a = resolve_task_stages(task_a)
    resolved_b = resolve_task_stages(task_b)

    assert (
        resolved_a.evaluations[0].evaluation_hash
        != resolved_b.evaluations[0].evaluation_hash
    )


def test_resolve_task_stages_rejects_transform_short_hash_collisions(monkeypatch):
    module = _task_resolution_module()
    resolve_task_stages = getattr(module, "resolve_task_stages")

    def fake_compute_hash(self: OutputTransformSpec, *, short: bool = False) -> str:
        extractor_id = self.extractor_chain.extractors[0].id
        if extractor_id == "json":
            full_hash = (
                "collision1234aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            )
        else:
            full_hash = (
                "collision1234bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            )
        if short:
            return full_hash[:12]
        return full_hash

    monkeypatch.setattr(OutputTransformSpec, "compute_hash", fake_compute_hash)

    task = TaskSpec(
        task_id="qa",
        dataset=DatasetSpec(source=DatasetSource.MEMORY),
        generation=GenerationSpec(),
        output_transforms=[
            OutputTransformSpec(
                name="json",
                extractor_chain=ExtractorChainSpec(extractors=["json"]),
            ),
            OutputTransformSpec(
                name="regex",
                extractor_chain=ExtractorChainSpec(extractors=["regex"]),
            ),
        ],
    )

    with pytest.raises(SpecValidationError, match="short hash collision"):
        resolve_task_stages(task)
