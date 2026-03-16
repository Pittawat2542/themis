from __future__ import annotations

import pytest

from themis.errors import StorageError
from themis.orchestration.run_manifest import RunManifest, StageWorkItem, WorkItemStatus
from themis.specs.experiment import (
    ExperimentSpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    PromptTemplateSpec,
)
from themis.specs.foundational import DatasetSpec, GenerationSpec, ModelSpec, TaskSpec
from themis.storage.run_manifest_repo import RunManifestRepository
from themis.storage.sqlite_schema import DatabaseManager
from themis.types.enums import DatasetSource, ErrorCode, RunStage


def _manifest() -> RunManifest:
    return RunManifest(
        run_id="run-1",
        backend_kind="local",
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
