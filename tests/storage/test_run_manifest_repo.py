from __future__ import annotations

import pytest

from themis.benchmark.specs import (
    BenchmarkSpec,
    PromptVariantSpec,
    ScoreSpec,
    SliceSpec,
)
from themis.errors import StorageError
from themis.orchestration.run_manifest import RunManifest, StageWorkItem, WorkItemStatus
from themis.specs.experiment import (
    ExperimentSpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    PromptMessage,
    PromptTemplateSpec,
)
from themis.specs.foundational import DatasetSpec, GenerationSpec, ModelSpec, TaskSpec
from themis.storage.run_manifest_repo import RunManifestRepository
from themis.storage.sqlite_schema import DatabaseManager
from themis.types.enums import DatasetSource, ErrorCode, PromptRole, RunStage


def _manifest() -> RunManifest:
    return RunManifest(
        run_id="run-1",
        backend_kind="local",
        benchmark_spec=BenchmarkSpec(
            benchmark_id="benchmark-1",
            models=[ModelSpec(model_id="mock-model", provider="mock")],
            slices=[
                SliceSpec(
                    slice_id="slice-1",
                    dataset=DatasetSpec(source=DatasetSource.MEMORY),
                    generation=GenerationSpec(),
                    scores=[ScoreSpec(name="default", metrics=["accuracy"])],
                )
            ],
            prompt_variants=[
                PromptVariantSpec(
                    id="baseline",
                    family="qa",
                    messages=[
                        PromptMessage(
                            role=PromptRole.USER, content="Answer the question."
                        )
                    ],
                )
            ],
            inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
        ),
        experiment_spec=ExperimentSpec(
            models=[ModelSpec(model_id="mock-model", provider="mock")],
            tasks=[
                TaskSpec(
                    task_id="task",
                    dataset=DatasetSpec(source=DatasetSource.MEMORY),
                    generation=GenerationSpec(),
                )
            ],
            prompt_templates=[PromptTemplateSpec(id="baseline", messages=[])],
            inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
        ),
        work_items=[
            StageWorkItem(
                work_item_id="work-1",
                stage=RunStage.GENERATION,
                status=WorkItemStatus.PENDING,
                trial_hash="trial-1",
                candidate_index=0,
                candidate_id="candidate-1",
            )
        ],
    )


def test_update_work_item_raises_for_unknown_work_item(tmp_path) -> None:
    manager = DatabaseManager(f"sqlite:///{tmp_path}/manifest.db")
    manager.initialize()
    repo = RunManifestRepository(manager)
    manifest = _manifest()
    repo.save_manifest(manifest)

    with pytest.raises(StorageError) as exc_info:
        repo.update_work_item(
            manifest.run_id,
            "missing-work-item",
            status=WorkItemStatus.RUNNING,
        )

    assert exc_info.value.code == ErrorCode.STORAGE_WRITE
    assert "missing-work-item" in exc_info.value.message


def test_save_and_load_manifest_round_trips_benchmark_source_spec(tmp_path) -> None:
    manager = DatabaseManager(f"sqlite:///{tmp_path}/manifest.db")
    manager.initialize()
    repo = RunManifestRepository(manager)
    manifest = _manifest()

    repo.save_manifest(manifest)
    loaded = repo.get_manifest(manifest.run_id)

    assert loaded is not None
    assert loaded.source_kind == "benchmark"
    assert loaded.benchmark_spec is not None
    assert loaded.benchmark_spec.benchmark_id == "benchmark-1"
