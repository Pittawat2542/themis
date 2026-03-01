from unittest.mock import patch
from themis.config.schema import HuggingFaceHubConfig
from themis.integrations.huggingface import HuggingFaceHubUploader
from themis.core.entities import ExperimentReport
from themis.evaluation.reports import EvaluationReport


@patch("themis.integrations.huggingface.HfApi")
def test_huggingface_upload_results(mock_hf, tmp_path):
    config = HuggingFaceHubConfig(enable=True, repository="test/repo")
    uploader = HuggingFaceHubUploader(config)

    report = ExperimentReport(
        metadata={"run_id": "test-run"},
        generation_results=[],
        evaluation_report=EvaluationReport(metrics={}, records=[], failures=[]),
        failures=[],
    )

    uploader.upload_results(report, tmp_path)

    # Check that HfApi was called to upload the report.json
    mock_hf.return_value.upload_file.assert_called_with(
        path_or_fileobj=str(tmp_path / "report.json"),
        path_in_repo="test-run/report.json",
        repo_id="test/repo",
        repo_type="dataset",
    )
