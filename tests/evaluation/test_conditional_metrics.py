"""Conditional tests for NLP and code metrics with mocked dependencies.

These tests verify metric logic WITHOUT requiring the actual libraries
(sacrebleu, rouge-score, nltk, bert-score, codebleu) to be installed.
Each test mocks the lazy import so the class can be instantiated, then
verifies that `compute()` correctly delegates to the library and
transforms the result into a proper MetricScore.
"""

import sys
import types
from types import SimpleNamespace
from unittest import mock

import pytest

from themis.core.entities import MetricScore
from themis.exceptions import DependencyError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_import(module_path: str):
    """Force a fresh import of a metric module (bypasses __init__.py caching)."""
    key = module_path
    saved = sys.modules.pop(key, None)
    # Also pop parent __init__ to avoid stale re-exports
    parent = ".".join(key.split(".")[:-1])
    saved_parent = sys.modules.pop(parent, None)
    try:
        mod = __import__(key, fromlist=["_"])
        return mod
    finally:
        # Restore so other tests aren't affected
        if saved is not None:
            sys.modules[key] = saved
        if saved_parent is not None:
            sys.modules[parent] = saved_parent


# ===================================================================
# BLEU
# ===================================================================


class TestBLEU:
    """Test BLEU metric with mocked sacrebleu."""

    def test_dependency_error_when_missing(self):
        """Should raise DependencyError when sacrebleu is not installed."""
        with mock.patch.dict(sys.modules, {"sacrebleu": None}):
            mod = _fresh_import("themis.evaluation.metrics.nlp.bleu")
            with pytest.raises(DependencyError, match="sacrebleu"):
                mod.BLEU()

    def test_compute_returns_metric_score(self):
        """Verify compute() transforms sacrebleu output correctly."""
        # Build a fake sacrebleu module
        fake_score = SimpleNamespace(
            score=45.23,
            precisions=[80.0, 50.0, 30.0, 20.0],
            bp=0.95,
            sys_len=50,
            ref_len=55,
        )

        class FakeSacreBLEU:
            def __init__(self, **kwargs):
                pass

            def sentence_score(self, hypothesis, references):
                return fake_score

        fake_sacrebleu = types.ModuleType("sacrebleu")
        fake_sacrebleu.BLEU = FakeSacreBLEU

        with mock.patch.dict(sys.modules, {"sacrebleu": fake_sacrebleu}):
            mod = _fresh_import("themis.evaluation.metrics.nlp.bleu")
            metric = mod.BLEU()
            result = metric.compute(
                prediction="The cat sat on the mat",
                references=["The cat is on the mat"],
            )

        assert isinstance(result, MetricScore)
        assert result.metric_name == "bleu"
        assert result.value == pytest.approx(0.4523)
        assert result.details["brevity_penalty"] == 0.95
        assert result.details["precision_1"] == pytest.approx(0.80)

    def test_attributes(self):
        """Verify metric has correct name and requires_reference."""

        class FakeSacreBLEU:
            def __init__(self, **kwargs):
                pass

        fake_sacrebleu = types.ModuleType("sacrebleu")
        fake_sacrebleu.BLEU = FakeSacreBLEU

        with mock.patch.dict(sys.modules, {"sacrebleu": fake_sacrebleu}):
            mod = _fresh_import("themis.evaluation.metrics.nlp.bleu")
            metric = mod.BLEU()
            assert metric.name == "bleu"
            assert metric.requires_reference is True


# ===================================================================
# ROUGE
# ===================================================================


class TestROUGE:
    """Test ROUGE metric with mocked rouge-score."""

    def test_dependency_error_when_missing(self):
        with mock.patch.dict(
            sys.modules, {"rouge_score": None, "rouge_score.rouge_scorer": None}
        ):
            mod = _fresh_import("themis.evaluation.metrics.nlp.rouge")
            with pytest.raises(DependencyError, match="rouge-score"):
                mod.ROUGE()

    def test_compute_returns_metric_score(self):
        fake_rouge_result = SimpleNamespace(precision=0.75, recall=0.80, fmeasure=0.77)

        class FakeRougeScorer:
            def __init__(self, rouge_types, use_stemmer=True):
                self._types = rouge_types

            def score(self, reference, hypothesis):
                return {t: fake_rouge_result for t in self._types}

        # Build fake module hierarchy
        fake_rouge_scorer_mod = types.ModuleType("rouge_score.rouge_scorer")
        fake_rouge_scorer_mod.RougeScorer = FakeRougeScorer

        fake_rouge_score = types.ModuleType("rouge_score")
        fake_rouge_score.rouge_scorer = fake_rouge_scorer_mod

        with mock.patch.dict(
            sys.modules,
            {
                "rouge_score": fake_rouge_score,
                "rouge_score.rouge_scorer": fake_rouge_scorer_mod,
            },
        ):
            mod = _fresh_import("themis.evaluation.metrics.nlp.rouge")
            metric = mod.ROUGE()
            result = metric.compute(
                prediction="The quick brown fox",
                references=["A quick brown fox"],
            )

        assert isinstance(result, MetricScore)
        assert result.value == pytest.approx(0.77)
        assert result.details["precision"] == pytest.approx(0.75)
        assert result.details["recall"] == pytest.approx(0.80)


# ===================================================================
# BERTScore
# ===================================================================


class TestBERTScore:
    """Test BERTScore with mocked bert-score."""

    def test_dependency_error_when_missing(self):
        with mock.patch.dict(sys.modules, {"bert_score": None}):
            mod = _fresh_import("themis.evaluation.metrics.nlp.bertscore")
            with pytest.raises(DependencyError, match="bert-score"):
                mod.BERTScore()

    def test_compute_returns_metric_score(self):
        class FakeTensor:
            def __init__(self, values):
                self._values = values

            def argmax(self):
                return SimpleNamespace(
                    item=lambda: self._values.index(max(self._values))
                )

            def __getitem__(self, idx):
                return SimpleNamespace(item=lambda: self._values[idx])

        def fake_score(
            cands,
            refs,
            model_type=None,
            lang=None,
            rescale_with_baseline=True,
            device=None,
            verbose=False,
        ):
            P = FakeTensor([0.85])
            R = FakeTensor([0.90])
            F1 = FakeTensor([0.87])
            return P, R, F1

        fake_bert_score = types.ModuleType("bert_score")
        fake_bert_score.score = fake_score

        with mock.patch.dict(sys.modules, {"bert_score": fake_bert_score}):
            mod = _fresh_import("themis.evaluation.metrics.nlp.bertscore")
            metric = mod.BERTScore()
            result = metric.compute(
                prediction="The cat sat on the mat",
                references=["A cat is on the mat"],
            )

        assert isinstance(result, MetricScore)
        assert result.metric_name == "bertscore"
        assert result.value == pytest.approx(0.87)
        assert result.details["precision"] == pytest.approx(0.85)


# ===================================================================
# METEOR
# ===================================================================


class TestMETEOR:
    """Test METEOR with mocked nltk."""

    def test_dependency_error_when_missing(self):
        with mock.patch.dict(
            sys.modules,
            {
                "nltk": None,
                "nltk.translate": None,
                "nltk.translate.meteor_score": None,
            },
        ):
            mod = _fresh_import("themis.evaluation.metrics.nlp.meteor")
            with pytest.raises(DependencyError, match="nltk"):
                mod.METEOR()

    def test_compute_returns_metric_score(self):
        def fake_meteor_score(references, hypothesis, alpha=0.9, beta=3.0, gamma=0.5):
            return 0.82

        fake_meteor_mod = types.ModuleType("nltk.translate.meteor_score")
        fake_meteor_mod.meteor_score = fake_meteor_score

        fake_translate = types.ModuleType("nltk.translate")
        fake_translate.meteor_score = fake_meteor_mod

        fake_nltk = types.ModuleType("nltk")
        fake_nltk.translate = fake_translate
        # nltk.data.find used during init
        fake_data = types.ModuleType("nltk.data")
        fake_data.find = mock.MagicMock()  # pretend data exists
        fake_nltk.data = fake_data
        fake_nltk.download = mock.MagicMock()

        with mock.patch.dict(
            sys.modules,
            {
                "nltk": fake_nltk,
                "nltk.translate": fake_translate,
                "nltk.translate.meteor_score": fake_meteor_mod,
                "nltk.data": fake_data,
            },
        ):
            mod = _fresh_import("themis.evaluation.metrics.nlp.meteor")
            metric = mod.METEOR()
            result = metric.compute(
                prediction="The cat sat on the mat",
                references=["The cat is on the mat"],
            )

        assert isinstance(result, MetricScore)
        assert result.metric_name == "meteor"
        assert result.value == pytest.approx(0.82)


# ===================================================================
# CodeBLEU
# ===================================================================


class TestCodeBLEU:
    """Test CodeBLEU with mocked codebleu library."""

    def test_dependency_error_when_missing(self):
        with mock.patch.dict(sys.modules, {"codebleu": None}):
            mod = _fresh_import("themis.evaluation.metrics.code.codebleu")
            with pytest.raises(DependencyError, match="codebleu"):
                mod.CodeBLEU()

    def test_compute_returns_metric_score(self):
        def fake_calc_codebleu(references, predictions, lang, weights):
            return {
                "codebleu": 0.75,
                "ngram_match_score": 0.80,
                "weighted_ngram_match_score": 0.70,
                "syntax_match_score": 0.85,
                "dataflow_match_score": 0.65,
            }

        fake_codebleu_mod = types.ModuleType("codebleu")
        fake_codebleu_mod.calc_codebleu = fake_calc_codebleu

        with mock.patch.dict(sys.modules, {"codebleu": fake_codebleu_mod}):
            mod = _fresh_import("themis.evaluation.metrics.code.codebleu")
            metric = mod.CodeBLEU(lang="python")
            result = metric.compute(
                prediction="def add(a, b): return a + b",
                references=["def add(x, y): return x + y"],
            )

        assert isinstance(result, MetricScore)
        assert result.metric_name == "codebleu"
        assert result.value == pytest.approx(0.75)
        assert result.details["syntax_match_score"] == pytest.approx(0.85)


# ===================================================================
# PassAtK (no external deps, test directly)
# ===================================================================


class TestPassAtK:
    """Test PassAtK and estimate_pass_at_k (no external dependencies)."""

    def test_estimate_pass_at_k_basic(self):
        from themis.evaluation.metrics.code.pass_at_k import estimate_pass_at_k

        # All passed
        assert estimate_pass_at_k(n=10, c=10, k=1) == 1.0
        # None passed
        assert estimate_pass_at_k(n=10, c=0, k=1) == pytest.approx(0.0)
        # Simple case
        assert estimate_pass_at_k(n=10, c=3, k=1) == pytest.approx(0.3)

    def test_compute_with_valid_prediction(self):
        from themis.evaluation.metrics.code.pass_at_k import PassAtK

        metric = PassAtK(k=1)
        result = metric.compute(
            prediction={"samples": ["code1", "code2"], "test_results": [True, False]},
            references=[],
        )
        assert isinstance(result, MetricScore)
        assert result.value == pytest.approx(0.5)

    def test_compute_with_invalid_prediction(self):
        from themis.evaluation.metrics.code.pass_at_k import PassAtK

        metric = PassAtK(k=1)
        result = metric.compute(prediction="invalid", references=[])
        assert result.value == 0.0
        assert "error" in result.details


# ===================================================================
# ExecutionAccuracy (no external deps, test directly)
# ===================================================================


class TestExecutionAccuracy:
    """Test ExecutionAccuracy metric (uses only stdlib multiprocessing)."""

    def test_simple_function_execution(self):
        from themis.evaluation.metrics.code.execution import ExecutionAccuracy

        metric = ExecutionAccuracy(timeout=2.0)
        result = metric.compute(
            prediction="def add(a, b):\n    return a + b",
            references=[
                {
                    "inputs": [(2, 3), (0, 0)],
                    "expected": [5, 0],
                    "function_name": "add",
                }
            ],
        )
        assert isinstance(result, MetricScore)
        assert result.value == pytest.approx(1.0)

    def test_failing_function(self):
        from themis.evaluation.metrics.code.execution import ExecutionAccuracy

        metric = ExecutionAccuracy(timeout=2.0)
        result = metric.compute(
            prediction="def add(a, b):\n    return a - b",
            references=[
                {
                    "inputs": [(2, 3)],
                    "expected": [5],
                    "function_name": "add",
                }
            ],
        )
        assert result.value == pytest.approx(0.0)

    def test_invalid_prediction_format(self):
        from themis.evaluation.metrics.code.execution import ExecutionAccuracy

        metric = ExecutionAccuracy(timeout=1.0)
        result = metric.compute(prediction="not a dict", references=[])
        assert result.value == pytest.approx(0.0)
