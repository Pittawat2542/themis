"""Trial planning logic for expanding experiment matrices into planned trials."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import random
from collections.abc import Collection, Mapping, Sequence

from themis.benchmark.compiler import compile_benchmark
from themis.benchmark.query import DatasetQuerySpec
from themis.benchmark.specs import BenchmarkSpec, DatasetSliceSpec
from themis.contracts.protocols import DatasetItem, DatasetLoader, DatasetProvider
from themis.errors import SpecValidationError
from themis.registry.compatibility import (
    TrialStage,
    validate_trial,
    validate_trial_for_stages,
)
from themis.registry.plugin_registry import PluginRegistry
from themis.specs.experiment import (
    DataItemContext,
    ExperimentSpec,
    ItemSamplingSpec,
    PromptTemplateSpec,
    TrialSpec,
)
from themis.types.enums import ErrorCode, SamplingKind
from themis.types.json_types import JSONValueType
from themis.types.json_validation import validate_json_dict, validate_json_value


@dataclass(frozen=True, slots=True)
class PlannedTrial:
    """One trial spec paired with its immutable execution-side item context."""

    trial_spec: TrialSpec
    dataset_context: DataItemContext


class TrialPlanner:
    """Expands an experiment into deterministic trial specs and validates compatibility."""

    def __init__(
        self,
        dataset_loader: DatasetLoader | None = None,
        dataset_provider: DatasetProvider | None = None,
        registry: PluginRegistry | None = None,
    ) -> None:
        self.dataset_loader = dataset_loader
        self.dataset_provider = dataset_provider
        self.registry = registry

    def plan_benchmark(
        self,
        benchmark: BenchmarkSpec,
        *,
        required_stages: Collection[TrialStage] | None = None,
    ) -> list[PlannedTrial]:
        """Compile and plan one benchmark specification."""

        return self.plan_experiment(
            compile_benchmark(benchmark),
            required_stages=required_stages,
        )

    def plan_experiment(
        self,
        experiment: ExperimentSpec,
        *,
        required_stages: Collection[TrialStage] | None = None,
    ) -> list[PlannedTrial]:
        """Expand an experiment specification into deterministic planned trials."""
        trials: list[PlannedTrial] = []
        task_prompts = {
            task.task_id: self._resolve_task_prompts(experiment, task)
            for task in experiment.tasks
        }

        task_items: dict[str, list[DataItemContext]] = {}
        if self.dataset_provider or self.dataset_loader:
            for task in experiment.tasks:
                query = self._effective_dataset_query(experiment, task)
                if self.dataset_provider is not None:
                    provider_query = self._provider_query(query)
                    items = self.dataset_provider.scan(
                        self._dataset_slice_spec(task),
                        provider_query,
                    )
                    task_items[task.task_id] = [
                        self._coerce_data_item_context(item) for item in items
                    ]
                else:
                    assert self.dataset_loader is not None
                    items = self.dataset_loader.load_task_items(task)
                    sampling = self._sampling_query(experiment, query)
                    task_items[task.task_id] = [
                        self._coerce_data_item_context(item)
                        for item in self._sample_items(items, sampling)
                    ]
        elif experiment.tasks:
            raise SpecValidationError(
                code=ErrorCode.MISSING_OPTIONAL_DEPENDENCY,
                message=(
                    "Experiment requires dataset items but no dataset_loader was "
                    "provided and no dataset_provider was provided."
                ),
            )
        params_grid = experiment.inference_grid.expand()
        for model in experiment.models:
            for task in experiment.tasks:
                items = task_items.get(task.task_id, [])
                prompts = task_prompts[task.task_id]
                for prompt, params in itertools.product(prompts, params_grid):
                    for item in items:
                        item_str = json.dumps(
                            {
                                "item_id": item.item_id,
                                "payload": item.payload,
                                "metadata": item.metadata,
                            },
                            sort_keys=True,
                        )

                        composite = "".join(
                            [
                                model.spec_hash,
                                task.spec_hash,
                                prompt.spec_hash,
                                params.spec_hash,
                                item_str,
                            ]
                        ).encode("utf-8")
                        trial_id = f"trial_{hashlib.sha256(composite).hexdigest()[:12]}"
                        trial = TrialSpec(
                            trial_id=trial_id,
                            model=model,
                            task=task,
                            prompt=prompt,
                            params=params,
                            item_id=item.item_id,
                            candidate_count=experiment.num_samples,
                            metadata=validate_json_dict(
                                {
                                    "benchmark_id": task.benchmark_id,
                                    "slice_id": task.slice_id or task.task_id,
                                    "prompt_variant_id": prompt.id,
                                    "dimensions": dict(task.dimensions),
                                },
                                label="TrialSpec.metadata",
                            ),
                        )
                        if self.registry is not None:
                            if required_stages is None:
                                validate_trial(trial, self.registry)
                            else:
                                validate_trial_for_stages(
                                    trial,
                                    self.registry,
                                    stages=required_stages,
                                )
                        trials.append(
                            PlannedTrial(trial_spec=trial, dataset_context=item)
                        )

        return trials

    def _resolve_task_prompts(
        self,
        experiment: ExperimentSpec,
        task,
    ) -> list[PromptTemplateSpec]:
        prompts = list(experiment.prompt_templates)
        if task.allowed_prompt_template_ids is not None:
            allowed_prompt_id_set = set(task.allowed_prompt_template_ids)
            resolved = [
                prompt for prompt in prompts if prompt.id in allowed_prompt_id_set
            ]
            if resolved:
                return resolved
            selector_preview = ", ".join(sorted(allowed_prompt_id_set))
            raise SpecValidationError(
                code=ErrorCode.SCHEMA_MISMATCH,
                message=(
                    "Task "
                    f"'{task.slice_id or task.task_id}' matched no prompt "
                    f"templates for explicit selector(s): {selector_preview}."
                ),
            )
        if task.prompt_family_filters is not None:
            allowed_families = set(task.prompt_family_filters)
            resolved = [
                prompt for prompt in prompts if prompt.family in allowed_families
            ]
            if resolved:
                return resolved
            family_preview = ", ".join(sorted(allowed_families))
            raise SpecValidationError(
                code=ErrorCode.SCHEMA_MISMATCH,
                message=(
                    "Task "
                    f"'{task.slice_id or task.task_id}' matched no prompt "
                    f"templates for family selector(s): {family_preview}."
                ),
            )
        return prompts

    def _dataset_slice_spec(self, task) -> DatasetSliceSpec:
        return DatasetSliceSpec(
            benchmark_id=task.benchmark_id,
            slice_id=task.slice_id or task.task_id,
            dataset=task.dataset,
            dimensions=dict(task.dimensions),
        )

    def _coerce_data_item_context(self, item: object) -> DataItemContext:
        if isinstance(item, DataItemContext):
            return item

        if isinstance(item, Mapping):
            payload = self._json_dict(dict(item))
            item_id_value = payload.get("item_id") or payload.get("id")
            metadata_value = payload.get("metadata")
            metadata = (
                {str(key): str(value) for key, value in metadata_value.items()}
                if isinstance(metadata_value, dict)
                else {}
            )
            item_id = (
                str(item_id_value)
                if item_id_value is not None
                else self._fallback_item_id(payload)
            )
            return DataItemContext(item_id=item_id, payload=payload, metadata=metadata)

        scalar_value = self._json_value(item)
        item_id = str(item) if item is not None else "null"
        return DataItemContext(item_id=item_id, payload={"value": scalar_value})

    def _sample_items(
        self,
        items: Sequence[DatasetItem],
        sampling: ItemSamplingSpec | DatasetQuerySpec,
    ) -> list[DatasetItem]:
        item_list = self._filter_items(list(items), sampling)
        if sampling.kind == SamplingKind.ALL or not item_list:
            return item_list

        if sampling.kind == SamplingKind.SUBSET:
            count = min(sampling.count or len(item_list), len(item_list))
            if sampling.seed is None:
                return item_list[:count]
            indices = sorted(
                random.Random(sampling.seed).sample(range(len(item_list)), count)
            )
            return [item_list[index] for index in indices]

        strata_field = sampling.strata_field
        assert strata_field is not None

        buckets: dict[object, list[DatasetItem]] = {}
        for item in item_list:
            if isinstance(item, dict):
                key = item.get(strata_field)
            else:
                key = getattr(item, strata_field, None)
            if key is None:
                raise SpecValidationError(
                    code=ErrorCode.SCHEMA_MISMATCH,
                    message=f"Cannot stratify item without field '{strata_field}'.",
                )
            buckets.setdefault(key, []).append(item)

        count = min(sampling.count or len(item_list), len(item_list))
        picker = random.Random(sampling.seed)
        sampled: list[DatasetItem] = []
        remaining = count
        bucket_items = list(buckets.values())

        for index, bucket in enumerate(bucket_items):
            buckets_left = len(bucket_items) - index
            target = max(1, remaining // buckets_left)
            take = min(target, len(bucket), remaining)
            if take == len(bucket):
                sampled.extend(bucket)
            else:
                sampled.extend(picker.sample(bucket, take))
            remaining = count - len(sampled)
            if remaining <= 0:
                break

        return sampled[:count]

    def _filter_items(
        self,
        items: list[DatasetItem],
        sampling: ItemSamplingSpec | DatasetQuerySpec,
    ) -> list[DatasetItem]:
        filtered = list(items)
        if sampling.item_ids:
            allowed_item_ids = set(sampling.item_ids)
            filtered = [
                item
                for item in filtered
                if self._coerce_data_item_context(item).item_id in allowed_item_ids
            ]
        if sampling.metadata_filters:
            filtered = [
                item
                for item in filtered
                if self._metadata_matches(
                    self._coerce_data_item_context(item),
                    sampling.metadata_filters,
                )
            ]
        return filtered

    def _metadata_matches(
        self,
        item: DataItemContext,
        filters: Mapping[str, str],
    ) -> bool:
        for key, expected_value in filters.items():
            if item.metadata.get(key) != expected_value:
                return False
        return True

    def _fallback_item_id(self, payload: JSONValueType) -> str:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]

    def _coerce_dataset_query(
        self, query: object
    ) -> ItemSamplingSpec | DatasetQuerySpec | dict[str, JSONValueType]:
        if query is None:
            return DatasetQuerySpec()
        if isinstance(query, (ItemSamplingSpec, DatasetQuerySpec)):
            return query
        if isinstance(query, Mapping):
            payload = validate_json_dict(dict(query), label="task.dataset_query")
            try:
                return DatasetQuerySpec.model_validate(payload)
            except Exception:
                try:
                    return ItemSamplingSpec.model_validate(payload)
                except Exception:
                    return payload
        raise SpecValidationError(
            code=ErrorCode.SCHEMA_MISMATCH,
            message=f"Unsupported dataset query payload: {type(query).__name__}",
        )

    def _effective_dataset_query(
        self,
        experiment: ExperimentSpec,
        task,
    ) -> ItemSamplingSpec | DatasetQuerySpec | dict[str, JSONValueType]:
        query = self._coerce_dataset_query(task.dataset_query)
        if isinstance(query, dict):
            return query
        if self._is_default_sampling(query) and not self._is_default_sampling(
            experiment.item_sampling
        ):
            return experiment.item_sampling
        return query

    def _provider_query(
        self,
        query: ItemSamplingSpec | DatasetQuerySpec | dict[str, JSONValueType],
    ) -> DatasetQuerySpec:
        if isinstance(query, DatasetQuerySpec):
            return query
        if isinstance(query, ItemSamplingSpec):
            return DatasetQuerySpec.model_validate(query.model_dump(mode="json"))
        raise SpecValidationError(
            code=ErrorCode.SCHEMA_MISMATCH,
            message=(
                "Dataset providers require a DatasetQuerySpec-compatible query, "
                "but the task carries a loader-side JSON mapping."
            ),
        )

    def _sampling_query(
        self,
        experiment: ExperimentSpec,
        query: ItemSamplingSpec | DatasetQuerySpec | dict[str, JSONValueType],
    ) -> ItemSamplingSpec | DatasetQuerySpec:
        if isinstance(query, dict):
            if not self._is_default_sampling(experiment.item_sampling):
                return experiment.item_sampling
            return ItemSamplingSpec()
        return query

    def _is_default_sampling(
        self, sampling: ItemSamplingSpec | DatasetQuerySpec
    ) -> bool:
        return (
            sampling.kind == SamplingKind.ALL
            and sampling.count is None
            and sampling.seed is None
            and sampling.strata_field is None
            and not sampling.item_ids
            and not sampling.metadata_filters
            and not getattr(sampling, "projected_fields", [])
        )

    def _json_value(self, data: object) -> JSONValueType:
        return validate_json_value(data, label="dataset item")

    def _json_dict(self, data: Mapping[str, object]) -> dict[str, JSONValueType]:
        return validate_json_dict(dict(data), label="dataset item")
