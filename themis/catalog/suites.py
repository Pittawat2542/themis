"""Suite definitions over benchmark-backed executable definitions."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from themis.catalog.benchmarks import get_benchmark_kit, list_benchmark_kits
from themis.catalog.benchmarks.kits import _sample_dataset
from themis.catalog.benchmarks import load_benchmark
from themis.core.base import FrozenModel
from themis.core.results import RunResult, RunStatus
from themis.core.store import RunStore
from themis.core.stores import InMemoryRunStore


class SuiteAggregation(StrEnum):
    """How suite-level reporting should aggregate child run claims."""

    NONE = "none"
    AVERAGE = "average"
    AVERAGE_OF_AVERAGES = "average_of_averages"
    DISPLAY_ONLY = "display_only"


class SuiteItem(FrozenModel):
    """One benchmark or nested suite reference in a suite definition."""

    benchmark_id: str | None = None
    suite_id: str | None = None

    @model_validator(mode="after")
    def _validate_single_reference(self) -> SuiteItem:
        if (self.benchmark_id is None) == (self.suite_id is None):
            raise ValueError("SuiteItem requires exactly one of benchmark_id or suite_id")
        return self


class SuiteDefinition(FrozenModel):
    """Named automation surface that expands to benchmark executable definitions."""

    suite_id: str
    items: list[SuiteItem] = Field(default_factory=list)
    aggregation: SuiteAggregation = SuiteAggregation.AVERAGE
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class SuiteExpansionItem(FrozenModel):
    """One resolved benchmark item from suite expansion."""

    benchmark_id: str
    source_suite_ids: list[str]


class SuiteExpansion(FrozenModel):
    """Deterministic expansion of one suite into benchmark ids."""

    suite_id: str
    items: list[SuiteExpansionItem]
    aggregation: SuiteAggregation

    @property
    def benchmark_ids(self) -> list[str]:
        return [item.benchmark_id for item in self.items]


class SuiteRunItem(FrozenModel):
    """Result for one run launched by a suite."""

    benchmark_id: str
    run_id: str
    status: RunStatus


class SuiteRunResult(FrozenModel):
    """Result of running every executable item in a suite."""

    suite_id: str
    runs: list[SuiteRunItem]

    @property
    def run_ids(self) -> list[str]:
        return [item.run_id for item in self.runs]


_REGISTERED_SUITES: dict[str, SuiteDefinition] = {}


def register_suite(suite: SuiteDefinition) -> SuiteDefinition:
    """Register a suite definition for the current process."""

    if suite.suite_id in list_benchmark_kits():
        raise ValueError(f"Suite id {suite.suite_id!r} collides with a benchmark id")
    existing = _REGISTERED_SUITES.get(suite.suite_id)
    if existing is not None and existing != suite:
        raise ValueError(f"Duplicate suite id: {suite.suite_id}")
    _REGISTERED_SUITES[suite.suite_id] = suite
    return suite


def get_suite(suite_id: str) -> SuiteDefinition:
    """Return a registered suite definition."""

    _ensure_builtin_suites()
    try:
        return _REGISTERED_SUITES[suite_id]
    except KeyError as exc:
        raise ValueError(f"Unknown suite: {suite_id}") from exc


def list_suites(*, tags: list[str] | None = None) -> list[str]:
    """List suite ids, optionally filtered by tags."""

    _ensure_builtin_suites()
    requested_tags = set(tags or [])
    return sorted(
        suite_id
        for suite_id, suite in _REGISTERED_SUITES.items()
        if not requested_tags or requested_tags.issubset(set(suite.tags))
    )


def expand_suite(suite_id: str) -> SuiteExpansion:
    """Expand a suite to benchmark items in deterministic order."""

    suite = get_suite(suite_id)
    return SuiteExpansion(
        suite_id=suite.suite_id,
        items=_expand_items(suite, stack=[]),
        aggregation=suite.aggregation,
    )


def run_suite(suite_id: str, *, store: RunStore | None = None) -> SuiteRunResult:
    """Run each expanded suite item as a normal Themis run."""

    run_store = store or InMemoryRunStore()
    run_store.initialize()
    expansion = expand_suite(suite_id)
    runs: list[SuiteRunItem] = []
    for item in expansion.items:
        definition = load_benchmark(item.benchmark_id)
        experiment = get_benchmark_kit(item.benchmark_id).build_experiment(
            dataset=_sample_dataset(definition)
        )
        result: RunResult = experiment.run(store=run_store)
        _tag_suite_run(run_store, result.run_id, suite_id=suite_id)
        runs.append(
            SuiteRunItem(
                benchmark_id=item.benchmark_id,
                run_id=result.run_id,
                status=result.status,
            )
        )
    return SuiteRunResult(suite_id=suite_id, runs=runs)


def _expand_items(
    suite: SuiteDefinition, *, stack: list[str]
) -> list[SuiteExpansionItem]:
    if suite.suite_id in stack:
        cycle = " -> ".join([*stack, suite.suite_id])
        raise ValueError(f"Suite expansion cycle detected: {cycle}")
    next_stack = [*stack, suite.suite_id]
    expanded: list[SuiteExpansionItem] = []
    for item in suite.items:
        if item.benchmark_id is not None:
            expanded.append(
                SuiteExpansionItem(
                    benchmark_id=item.benchmark_id,
                    source_suite_ids=next_stack,
                )
            )
        elif item.suite_id is not None:
            expanded.extend(_expand_items(get_suite(item.suite_id), stack=next_stack))
    return expanded


def _tag_suite_run(store: RunStore, run_id: str, *, suite_id: str) -> None:
    record = store.get_run_record(run_id)
    current_tags = list(record.tags) if record is not None else []
    suite_tag = f"suite:{suite_id}"
    if suite_tag not in current_tags:
        current_tags.append(suite_tag)
    store.update_run_record(run_id, tags=current_tags)


def _ensure_builtin_suites() -> None:
    if "math-core" in _REGISTERED_SUITES:
        return
    register_suite(
        SuiteDefinition(
            suite_id="math-core",
            items=[
                SuiteItem(benchmark_id="aime_2025"),
                SuiteItem(benchmark_id="mmlu_pro"),
            ],
            aggregation=SuiteAggregation.AVERAGE,
            description="Small deterministic math-oriented starter suite.",
            tags=["builtin", "math"],
        )
    )
    register_suite(
        SuiteDefinition(
            suite_id="qa-core",
            items=[
                SuiteItem(benchmark_id="frontierscience"),
                SuiteItem(benchmark_id="simpleqa_verified"),
            ],
            aggregation=SuiteAggregation.AVERAGE,
            description="Small deterministic QA starter suite.",
            tags=["builtin", "qa"],
        )
    )
    register_suite(
        SuiteDefinition(
            suite_id="code-core",
            items=[SuiteItem(benchmark_id="humaneval_plus")],
            aggregation=SuiteAggregation.DISPLAY_ONLY,
            description="Small deterministic code starter suite.",
            tags=["builtin", "code"],
        )
    )
    register_suite(
        SuiteDefinition(
            suite_id="general-core",
            items=[
                SuiteItem(suite_id="math-core"),
                SuiteItem(suite_id="qa-core"),
            ],
            aggregation=SuiteAggregation.AVERAGE_OF_AVERAGES,
            description="Starter suite combining math and QA coverage.",
            tags=["builtin", "general"],
        )
    )
