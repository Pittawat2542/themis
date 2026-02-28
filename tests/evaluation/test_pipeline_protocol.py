"""Tests for EvaluationPipelineContract protocol compliance."""

from themis.evaluation.pipeline import EvaluationPipelineContract


class TestEvaluationPipelineProtocol:
    """Verify the protocol exposes metric_names and evaluation_fingerprint."""

    def test_pipeline_has_metric_names_property(self):
        """EvaluationPipeline must expose metric_names."""
        from themis.evaluation import pipeline, extractors, metrics

        p = pipeline.EvaluationPipeline(
            extractor=extractors.IdentityExtractor(),
            metrics=[metrics.ExactMatch(), metrics.ResponseLength()],
        )
        names = p.metric_names
        assert "ExactMatch" in names
        assert "ResponseLength" in names

    def test_pipeline_fingerprint_includes_metrics(self):
        """evaluation_fingerprint must always include metric names."""
        from themis.evaluation import pipeline, extractors, metrics

        p = pipeline.EvaluationPipeline(
            extractor=extractors.IdentityExtractor(),
            metrics=[metrics.ExactMatch()],
        )
        fp = p.evaluation_fingerprint()
        assert "metrics" in fp
        assert any("ExactMatch" in m for m in fp["metrics"])

    def test_pipeline_is_protocol_compliant(self):
        """EvaluationPipeline must satisfy EvaluationPipelineContract."""
        from themis.evaluation import pipeline, extractors, metrics

        p = pipeline.EvaluationPipeline(
            extractor=extractors.IdentityExtractor(),
            metrics=[metrics.ExactMatch()],
        )
        assert isinstance(p, EvaluationPipelineContract)
