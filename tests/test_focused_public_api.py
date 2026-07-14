from __future__ import annotations

import themis.analysis as analysis
import themis.artifacts as artifacts
import themis.catalog as catalog
import themis.components as components
import themis.metrics as metrics
import themis.presets as presets
import themis.runtime as runtime
import themis.storage as storage


EXPECTED_EXPORTS = {
    components: """
        AggregationResult Case Candidate CandidateSetSubject CandidateSubject
        CandidateReducer CandidateSelector ConversationTrace ConversationSubject
        EvalScoreContext EvaluationExecution EvaluationWorkflow GenerationContext
        GenerationTurn Generator JudgeCall JudgeModel JudgeResponse Message
        MetricDirection MetricInterpretation MetricResult MetricSubjectKind
        ParsedJudgment ParsedOutput ParseContext Parser PromptSpec PureMetric
        RenderedJudgePrompt ReducedCandidate ReduceContext ScoreContext ScoreError
        SelectContext SeedCapability StreamEvent TraceStep TraceSubject
        WorkflowFailure WorkflowMetric WorkflowRunner WorkflowStatus
        WorkflowSubjectKind WorkflowTrace
    """,
    runtime: """
        EventSubscriber ExecutionResourcePlan EvidenceRetention ExistingRunPolicy
        RunEstimate Stage TracingProvider estimate resource_plan
    """,
    analysis: """
        ComparisonSummary MetricComparison MetricDirection MetricSummary Reporter
        ReporterProtocol StatsEngine StatsSummary available_reporters create_reporter
        get_attempt_history get_case_audit get_evaluation_execution
        get_execution_state get_projection get_run_record get_score_claim_history
        get_run_snapshot get_telemetry_summary query_run_records register_reporter
        resolve_run_id resolve_run_record snapshot_report
    """,
    storage: """
        AppendResult EventRecord InMemoryRunStore JsonlRunStore MongoDbRunStore
        PostgresRunStore ProjectionConsistency ProjectionFreshness ProjectionRead
        RunStore RunStoreBase SqliteRunStore StorageConfig available_store_backends
        create_run_store jsonl_store memory_store mongodb_store postgres_store
        register_store_backend sqlite_store
    """,
    artifacts: """
        EvaluationBundle EvaluationBundleRecord GenerationBundle
        GenerationBundleRecord ParseBundle ParseBundleRecord ReductionBundle
        ReductionBundleRecord ScoreBundle ScoreBundleRecord export_evaluation_bundle
        export_generation_bundle export_parse_bundle export_reduction_bundle
        export_score_bundle import_evaluation_bundle import_generation_bundle
        import_parse_bundle import_reduction_bundle import_score_bundle
    """,
    presets: """
        EvaluationPreset ExperimentPreset GenerationPreset Preset
        PresetApplication RuntimePreset apply_preset get_preset list_presets
    """,
    catalog: """
        BenchmarkCatalogEntry BenchmarkDefinition BenchmarkExperimentDefaults
        BenchmarkKit BenchmarkValidationCheck BenchmarkValidationResult
        SuiteAggregation SuiteDefinition SuiteExpansion
        SuiteExpansionItem SuiteItem SuiteRunItem SuiteRunResult
        build_benchmark_experiment builtin_component_refs expand_suite get_benchmark
        get_benchmark_kit get_suite list_benchmark_ids list_benchmark_kits
        list_benchmarks list_component_ids list_suites load register_suite run
        run_suite validate_benchmark
    """,
    metrics: "exact_match",
}


def test_focused_module_exports_match_the_v6_contract() -> None:
    for module, expected in EXPECTED_EXPORTS.items():
        names = set(expected.split())
        assert set(module.__all__) == names
        assert all(hasattr(module, name) for name in names)


def test_metrics_module_builds_the_documented_exact_match_metric() -> None:
    metric = metrics.exact_match()

    assert metric.component_id == "builtin/exact_match"
    assert metric.version == "1.0"
