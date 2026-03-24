from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from themis import _optional
from themis.errors import ThemisError
from themis.records.trial import TrialRecord
from themis.runtime.experiment_result import ExperimentResult
from themis.specs.experiment import InferenceParamsSpec, PromptTemplateSpec, TrialSpec
from themis.specs.foundational import DatasetSpec, GenerationSpec, ModelSpec, TaskSpec
from themis.types.enums import ErrorCode, RecordStatus, DatasetSource


class EmptyProjectionRepository:
    def __init__(self, trial_hash: str) -> None:
        self.trial_hash = trial_hash

    def get_trial_record(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ):
        del transform_hash, evaluation_hash
        trial = TrialSpec(
            trial_id="optional_trial",
            model=ModelSpec(model_id="gpt-4o-mini", provider="openai"),
            task=TaskSpec(
                task_id="math",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
            ),
            item_id="item-1",
            prompt=PromptTemplateSpec(id="baseline", messages=[]),
            params=InferenceParamsSpec(),
        )
        if trial_hash != self.trial_hash:
            return None
        return TrialRecord(
            spec_hash=trial_hash,
            status=RecordStatus.OK,
            candidates=[],
            trial_spec=trial,
        )

    def iter_candidate_scores(self, **kwargs):
        return iter(())


def test_experiment_result_compare_honors_stats_extra_boundary(monkeypatch):
    trial = TrialSpec(
        trial_id="optional_trial",
        model=ModelSpec(model_id="gpt-4o-mini", provider="openai"),
        task=TaskSpec(
            task_id="math",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
        ),
        item_id="item-1",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
    )
    result = ExperimentResult(
        projection_repo=EmptyProjectionRepository(trial.spec_hash),
        trial_hashes=[trial.spec_hash],
    )

    def raise_missing_optional(module_name: str, *, extra: str):
        raise ThemisError(
            code=ErrorCode.MISSING_OPTIONAL_DEPENDENCY,
            message=f'Install it with `uv add "themis-eval[{extra}]"`.',
        )

    monkeypatch.setattr(
        "themis.runtime.experiment_result.import_optional", raise_missing_optional
    )

    with pytest.raises(ThemisError, match=r"themis-eval\[stats\]"):
        result.compare()


def test_import_optional_uses_uv_add_install_hint(monkeypatch) -> None:
    def raise_import_error(module_name: str) -> object:
        raise ImportError(module_name)

    monkeypatch.setattr(_optional.importlib, "import_module", raise_import_error)

    with pytest.raises(ThemisError, match=r'uv add "themis-eval\[stats\]"'):
        _optional.import_optional("pandas", extra="stats")


def test_pyproject_optional_dependency_groups_match_v2_surface():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]

    expected = {
        "compression",
        "datasets",
        "dev",
        "docs",
        "extractors",
        "math",
        "providers-openai",
        "providers-litellm",
        "providers-vllm",
        "stats",
        "telemetry",
        "storage-postgres",
        "all",
    }
    removed = {
        "cli",
        "code",
        "config",
        "langfuse",
        "nlp",
        "server",
        "ui",
        "viz",
    }

    assert expected.issubset(extras)
    assert removed.isdisjoint(extras)


def test_providers_vllm_extra_is_linux_only_in_packaging_metadata() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    providers_vllm = pyproject["project"]["optional-dependencies"]["providers-vllm"]

    assert providers_vllm == ["vllm>=0.17.0; sys_platform == 'linux'"]


def test_text_metrics_extra_contains_sampling_text_dependencies() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    text_metrics = pyproject["project"]["optional-dependencies"]["text-metrics"]

    assert "bert-score>=0.3.13" in text_metrics
    assert "evaluate>=0.4.3" in text_metrics
    assert "nltk>=3.9" in text_metrics
    assert "rouge-score>=0.1.2" in text_metrics
    assert "sacrebleu>=2.5.1" in text_metrics
    assert "scikit-learn>=1.5.0" in text_metrics
