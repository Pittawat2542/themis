"""Trial planning logic for expanding experiment matrices into planned trials."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import random
from collections.abc import Mapping
from typing import Any

from themis.errors.exceptions import SpecValidationError
from themis.registry.compatibility import CompatibilityChecker
from themis.specs.experiment import (
    DataItemContext,
    ExperimentSpec,
    ItemSamplingSpec,
    TrialSpec,
)
from themis.types.enums import ErrorCode
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
        dataset_loader: Any = None,
        registry: Any = None,
        compatibility_checker: CompatibilityChecker | None = None,
    ):
        self.dataset_loader = dataset_loader
        self.registry = registry
        self.compatibility_checker = compatibility_checker or CompatibilityChecker()

    def plan_experiment(self, experiment: ExperimentSpec) -> list[PlannedTrial]:
        """Expand an experiment specification into deterministic planned trials."""
        trials: list[PlannedTrial] = []

        task_items: dict[str, list[DataItemContext]] = {}
        if self.dataset_loader:
            for task in experiment.tasks:
                items = self.dataset_loader.load_task_items(task)
                task_items[task.task_id] = [
                    self._coerce_data_item_context(item)
                    for item in self._sample_items(items, experiment.item_sampling)
                ]
        elif experiment.tasks:
            raise SpecValidationError(
                code=ErrorCode.MISSING_OPTIONAL_DEPENDENCY,
                message="Experiment requires dataset items but no dataset_loader was provided.",
            )

        grid = itertools.product(
            experiment.models,
            experiment.tasks,
            experiment.prompt_templates,
            experiment.inference_grid.expand(),
        )

        for model, task, prompt, params in grid:
            items = task_items.get(task.task_id, [])

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
                )
                if self.registry is not None:
                    self.compatibility_checker.validate_trial(trial, self.registry)
                trials.append(PlannedTrial(trial_spec=trial, dataset_context=item))

        return trials

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

    def _sample_items(self, items: list[Any], sampling: ItemSamplingSpec) -> list[Any]:
        sampling.validate_semantic()
        item_list = list(items)
        if sampling.kind == "all" or not item_list:
            return item_list

        if sampling.kind == "subset":
            count = min(sampling.count or len(item_list), len(item_list))
            if sampling.seed is None:
                return item_list[:count]
            indices = sorted(
                random.Random(sampling.seed).sample(range(len(item_list)), count)
            )
            return [item_list[index] for index in indices]

        strata_field = sampling.strata_field
        assert strata_field is not None

        buckets: dict[Any, list[Any]] = {}
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
        sampled: list[Any] = []
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

    def _fallback_item_id(self, payload: JSONValueType) -> str:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]

    def _json_value(self, data: object) -> JSONValueType:
        return validate_json_value(data, label="dataset item")

    def _json_dict(self, data: Mapping[str, object]) -> dict[str, JSONValueType]:
        return validate_json_dict(dict(data), label="dataset item")
