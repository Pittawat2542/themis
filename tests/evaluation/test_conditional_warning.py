import sys
import warnings


def test_conditional_warning_deferred():
    # Ensure module is not loaded yet so we can test the import
    if "themis.evaluation.conditional" in sys.modules:
        del sys.modules["themis.evaluation.conditional"]

    # 1. Importing the module should NOT emit a warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        import themis.evaluation.conditional as cond

        # Ensure _WARNED gets reset for the test
        cond._WARNED = False

        # Verify no FutureWarning about experimental features
        experimental_warnings = [
            warn
            for warn in w
            if issubclass(warn.category, FutureWarning)
            and "conditional evaluation pipelines are currently experimental"
            in str(warn.message)
        ]
        assert len(experimental_warnings) == 0, (
            "Module import emitted the warning prematurely!"
        )

    # 2. Instantiating a class SHOULD emit a warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from themis.interfaces import Metric
        from themis.core.entities import MetricScore

        class DummyMetric(Metric):
            name = "dummy"

            def compute(self, *, prediction, references, metadata=None) -> MetricScore:
                return MetricScore(self.name, 1.0)

        # Instantiate
        cond.ConditionalMetric(metric=DummyMetric(), condition=lambda r: True)

        experimental_warnings = [
            warn
            for warn in w
            if issubclass(warn.category, FutureWarning)
            and "conditional evaluation pipelines are currently experimental"
            in str(warn.message)
        ]
        assert len(experimental_warnings) == 1, (
            "Instantiation did not emit the experimental warning!"
        )

    # 3. Second instantiation should NOT emit the warning again
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        cond.ConditionalMetric(metric=DummyMetric(), condition=lambda r: False)

        experimental_warnings = [
            warn
            for warn in w
            if issubclass(warn.category, FutureWarning)
            and "conditional evaluation pipelines are currently experimental"
            in str(warn.message)
        ]
        assert len(experimental_warnings) == 0, (
            "Second instantiation emitted the warning again!"
        )
