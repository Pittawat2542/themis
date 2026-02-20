"""Tests for comparison reports module."""

from themis.comparison.reports import ComparisonReport, ComparisonResult, WinLossMatrix
from themis.comparison.statistics import StatisticalTestResult


class TestComparisonResult:
    """Tests for ComparisonResult."""

    def test_comparison_result_creation(self):
        """Test creating a comparison result."""
        result = ComparisonResult(
            metric_name="accuracy",
            run_a_id="run-1",
            run_b_id="run-2",
            run_a_mean=0.8,
            run_b_mean=0.7,
            delta=0.1,
            delta_percent=14.3,
            winner="run-1",
        )

        assert result.metric_name == "accuracy"
        assert result.delta > 0
        assert result.winner == "run-1"

    def test_comparison_result_with_test(self):
        """Test comparison result with statistical test."""
        test_result = StatisticalTestResult(
            test_name="t-test",
            statistic=2.5,
            p_value=0.01,
            significant=True,
        )

        result = ComparisonResult(
            metric_name="f1",
            run_a_id="a",
            run_b_id="b",
            run_a_mean=0.9,
            run_b_mean=0.8,
            delta=0.1,
            delta_percent=12.5,
            winner="a",
            test_result=test_result,
        )

        assert result.is_significant()
        assert result.test_result.p_value == 0.01

    def test_comparison_result_summary(self):
        """Test summary generation."""
        result = ComparisonResult(
            metric_name="accuracy",
            run_a_id="gpt4",
            run_b_id="claude",
            run_a_mean=0.85,
            run_b_mean=0.80,
            delta=0.05,
            delta_percent=6.25,
            winner="gpt4",
        )

        summary = result.summary()
        assert "accuracy" in summary
        assert "gpt4" in summary
        assert "claude" in summary
        assert "0.85" in summary or "0.850" in summary
        assert "↑" in summary  # Positive change arrow

    def test_comparison_result_summary_handles_ci_only_test_results(self):
        test_result = StatisticalTestResult(
            test_name="bootstrap",
            statistic=0.1,
            p_value=None,
            significant=True,
            inference_mode="ci_only",
        )
        result = ComparisonResult(
            metric_name="accuracy",
            run_a_id="a",
            run_b_id="b",
            run_a_mean=0.8,
            run_b_mean=0.7,
            delta=0.1,
            delta_percent=14.3,
            winner="a",
            test_result=test_result,
        )

        summary = result.summary()
        assert "CI-only" in summary


class TestWinLossMatrix:
    """Tests for WinLossMatrix."""

    def test_win_loss_matrix_creation(self):
        """Test creating a win/loss matrix."""
        run_ids = ["run-1", "run-2", "run-3"]
        matrix = [
            ["—", "win", "win"],
            ["loss", "—", "tie"],
            ["loss", "tie", "—"],
        ]

        wl_matrix = WinLossMatrix(
            run_ids=run_ids,
            metric_name="accuracy",
            matrix=matrix,
            win_counts={"run-1": 2, "run-2": 0, "run-3": 0},
            loss_counts={"run-1": 0, "run-2": 1, "run-3": 2},
            tie_counts={"run-1": 0, "run-2": 1, "run-3": 1},
        )

        assert len(wl_matrix.run_ids) == 3
        assert wl_matrix.metric_name == "accuracy"

    def test_get_result(self):
        """Test getting comparison result from matrix."""
        run_ids = ["a", "b"]
        matrix = [["—", "win"], ["loss", "—"]]

        wl_matrix = WinLossMatrix(
            run_ids=run_ids,
            metric_name="score",
            matrix=matrix,
        )

        assert wl_matrix.get_result("a", "b") == "win"
        assert wl_matrix.get_result("b", "a") == "loss"
        assert wl_matrix.get_result("a", "c") == "unknown"

    def test_rank_runs(self):
        """Test ranking runs by wins/losses."""
        run_ids = ["run-1", "run-2", "run-3"]

        wl_matrix = WinLossMatrix(
            run_ids=run_ids,
            metric_name="accuracy",
            matrix=[],
            win_counts={"run-1": 2, "run-2": 1, "run-3": 0},
            loss_counts={"run-1": 0, "run-2": 1, "run-3": 2},
            tie_counts={"run-1": 0, "run-2": 0, "run-3": 0},
        )

        rankings = wl_matrix.rank_runs()

        # Should be sorted by wins (desc), then losses (asc)
        assert rankings[0][0] == "run-1"  # 2 wins, 0 losses
        assert rankings[1][0] == "run-2"  # 1 win, 1 loss
        assert rankings[2][0] == "run-3"  # 0 wins, 2 losses

    def test_to_table(self):
        """Test table generation."""
        run_ids = ["a", "b"]
        matrix = [["—", "win"], ["loss", "—"]]

        wl_matrix = WinLossMatrix(
            run_ids=run_ids,
            metric_name="score",
            matrix=matrix,
            win_counts={"a": 1, "b": 0},
            loss_counts={"a": 0, "b": 1},
            tie_counts={"a": 0, "b": 0},
        )

        table = wl_matrix.to_table()

        assert "Run" in table
        assert "win" in table
        assert "loss" in table
        assert "Summary" in table


class TestComparisonReport:
    """Tests for ComparisonReport."""

    def test_comparison_report_creation(self):
        """Test creating a comparison report."""
        report = ComparisonReport(
            run_ids=["run-1", "run-2"],
            metrics=["accuracy", "f1"],
        )

        assert len(report.run_ids) == 2
        assert len(report.metrics) == 2
        assert len(report.pairwise_results) == 0  # Initially empty

    def test_get_comparison(self):
        """Test getting specific comparison."""
        result = ComparisonResult(
            metric_name="accuracy",
            run_a_id="a",
            run_b_id="b",
            run_a_mean=0.9,
            run_b_mean=0.8,
            delta=0.1,
            delta_percent=12.5,
            winner="a",
        )

        report = ComparisonReport(
            run_ids=["a", "b"],
            metrics=["accuracy"],
            pairwise_results=[result],
        )

        found = report.get_comparison("a", "b", "accuracy")
        assert found is not None
        assert found.winner == "a"

        not_found = report.get_comparison("a", "b", "f1")
        assert not_found is None

    def test_get_metric_results(self):
        """Test getting all results for a metric."""
        result1 = ComparisonResult(
            metric_name="accuracy",
            run_a_id="a",
            run_b_id="b",
            run_a_mean=0.9,
            run_b_mean=0.8,
            delta=0.1,
            delta_percent=12.5,
            winner="a",
        )

        result2 = ComparisonResult(
            metric_name="accuracy",
            run_a_id="a",
            run_b_id="c",
            run_a_mean=0.9,
            run_b_mean=0.85,
            delta=0.05,
            delta_percent=5.9,
            winner="a",
        )

        report = ComparisonReport(
            run_ids=["a", "b", "c"],
            metrics=["accuracy"],
            pairwise_results=[result1, result2],
        )

        results = report.get_metric_results("accuracy")
        assert len(results) == 2

    def test_summary(self):
        """Test report summary generation."""
        report = ComparisonReport(
            run_ids=["run-1", "run-2"],
            metrics=["accuracy"],
            best_run_per_metric={"accuracy": "run-1"},
            overall_best_run="run-1",
        )

        summary = report.summary()

        assert "COMPARISON REPORT" in summary
        assert "run-1" in summary
        assert "run-2" in summary
        assert "accuracy" in summary

    def test_to_dict(self):
        """Test converting report to dictionary."""
        result = ComparisonResult(
            metric_name="accuracy",
            run_a_id="a",
            run_b_id="b",
            run_a_mean=0.9,
            run_b_mean=0.8,
            delta=0.1,
            delta_percent=12.5,
            winner="a",
        )

        report = ComparisonReport(
            run_ids=["a", "b"],
            metrics=["accuracy"],
            pairwise_results=[result],
            best_run_per_metric={"accuracy": "a"},
            overall_best_run="a",
        )

        data = report.to_dict()

        assert "run_ids" in data
        assert "metrics" in data
        assert "pairwise_results" in data
        assert "best_run_per_metric" in data
        assert data["overall_best_run"] == "a"
        assert len(data["pairwise_results"]) == 1
