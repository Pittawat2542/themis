from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from huggingface_hub import HfApi
else:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        HfApi = None  # type: ignore

from themis.config.schema import HuggingFaceHubConfig
from themis.core.entities import ExperimentReport


def to_dict(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, (list, tuple)):
        return [to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {key: to_dict(value) for key, value in obj.items()}
    return obj


class HuggingFaceHubUploader:
    def __init__(self, config: HuggingFaceHubConfig):
        if HfApi is None:
            raise ImportError(
                "huggingface_hub is not installed. Install with: pip install huggingface_hub"
            )
        self.config = config
        self.api = HfApi()

    def upload_results(self, report: ExperimentReport, storage_path: Path) -> None:
        if not self.config.enable or not self.config.repository:
            return

        report_dict = to_dict(report)

        # Upload the full report as a JSON file
        report_path = storage_path / "report.json"
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=4)

        self.api.upload_file(
            path_or_fileobj=str(report_path),
            path_in_repo=f"{report.metadata.get('run_id')}/report.json",
            repo_id=self.config.repository,
            repo_type="dataset",
        )

        # Upload individual generation results
        for record in report.generation_results:
            record_dict = to_dict(record)
            record_path = (
                storage_path / f"{record.task.metadata.get('dataset_id')}.json"
            )
            with open(record_path, "w") as f:
                json.dump(record_dict, f, indent=4)
            self.api.upload_file(
                path_or_fileobj=str(record_path),
                path_in_repo=f"{report.metadata.get('run_id')}/generations/{record.task.metadata.get('dataset_id')}.json",
                repo_id=self.config.repository,
                repo_type="dataset",
            )
