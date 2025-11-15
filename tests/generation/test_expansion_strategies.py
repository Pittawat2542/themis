"""Tests for flexible generation planning and expansion strategies."""

from themis.core import entities
from themis.generation import plan, templates


def test_cartesian_expansion_strategy_basic():
    """Test basic Cartesian product expansion."""
    dataset = [
        {"id": "1", "problem": "test1"},
        {"id": "2", "problem": "test2"},
    ]

    template = templates.PromptTemplate(name="t1", template="{problem}")
    model1 = entities.ModelSpec(identifier="m1", provider="fake")
    model2 = entities.ModelSpec(identifier="m2", provider="fake")
    sampling = entities.SamplingConfig(temperature=0.0, top_p=0.95, max_tokens=100)

    context = plan.PlanContext(
        templates=[template],
        models=[model1, model2],
        sampling_parameters=[sampling],
        dataset_id_field="id",
        reference_field=None,
        metadata_fields=(),
        context_builder=None,
    )

    strategy = plan.CartesianExpansionStrategy()
    tasks = list(strategy.expand(dataset, context))

    # Should generate: 2 rows × 1 template × 2 models × 1 sampling = 4 tasks
    assert len(tasks) == 4

    # Check all combinations are present
    task_combinations = [(t.metadata["dataset_id"], t.model.identifier) for t in tasks]
    assert ("1", "m1") in task_combinations
    assert ("1", "m2") in task_combinations
    assert ("2", "m1") in task_combinations
    assert ("2", "m2") in task_combinations


def test_filtered_expansion_strategy_model_filter():
    """Test filtered expansion with model filtering."""
    dataset = [
        {"id": "easy-1", "difficulty": "easy"},
        {"id": "hard-1", "difficulty": "hard"},
    ]

    template = templates.PromptTemplate(name="t1", template="Problem")
    cheap_model = entities.ModelSpec(identifier="cheap", provider="fake")
    expensive_model = entities.ModelSpec(identifier="expensive", provider="fake")
    sampling = entities.SamplingConfig(temperature=0.0, top_p=0.95, max_tokens=100)

    # Only use expensive model on hard problems
    def task_filter(row, tpl, mdl, smp):
        if mdl.identifier == "cheap":
            return True
        return mdl.identifier == "expensive" and row.get("difficulty") == "hard"

    context = plan.PlanContext(
        templates=[template],
        models=[cheap_model, expensive_model],
        sampling_parameters=[sampling],
        dataset_id_field="id",
        reference_field=None,
        metadata_fields=["difficulty"],
        context_builder=None,
    )

    strategy = plan.FilteredExpansionStrategy(task_filter=task_filter)
    tasks = list(strategy.expand(dataset, context))

    # Should generate: 2 easy with cheap + 2 hard with both = 3 tasks
    # (easy-1: cheap, hard-1: cheap, hard-1: expensive)
    assert len(tasks) == 3

    # Check combinations
    task_info = [(t.metadata["dataset_id"], t.model.identifier) for t in tasks]
    assert ("easy-1", "cheap") in task_info
    assert ("hard-1", "cheap") in task_info
    assert ("hard-1", "expensive") in task_info
    assert ("easy-1", "expensive") not in task_info  # Filtered out


def test_filtered_expansion_strategy_reduces_count():
    """Test that filtering actually reduces task count."""
    dataset = [{"id": f"sample-{i}"} for i in range(10)]

    template = templates.PromptTemplate(name="t1", template="Problem")
    model = entities.ModelSpec(identifier="model", provider="fake")
    sampling = entities.SamplingConfig(temperature=0.0, top_p=0.95, max_tokens=100)

    context = plan.PlanContext(
        templates=[template],
        models=[model],
        sampling_parameters=[sampling],
        dataset_id_field="id",
        reference_field=None,
        metadata_fields=(),
        context_builder=None,
    )

    # Cartesian should generate 10 tasks
    cartesian = plan.CartesianExpansionStrategy()
    cartesian_tasks = list(cartesian.expand(dataset, context))
    assert len(cartesian_tasks) == 10

    # Filtered (only even samples) should generate 5 tasks
    def even_filter(row, tpl, mdl, smp):
        sample_id = row.get("id", "")
        return sample_id.endswith(("0", "2", "4", "6", "8"))

    filtered = plan.FilteredExpansionStrategy(task_filter=even_filter)
    filtered_tasks = list(filtered.expand(dataset, context))
    assert len(filtered_tasks) == 5


def test_conditional_expansion_strategy_routing():
    """Test conditional expansion routes to different strategies."""
    dataset = [
        {"id": "math-1", "type": "math"},
        {"id": "math-2", "type": "math"},
        {"id": "code-1", "type": "code"},
    ]

    template = templates.PromptTemplate(name="t1", template="Problem")
    model = entities.ModelSpec(identifier="model", provider="fake")
    sampling1 = entities.SamplingConfig(temperature=0.0, top_p=0.95, max_tokens=100)
    sampling2 = entities.SamplingConfig(temperature=0.7, top_p=0.95, max_tokens=100)

    context = plan.PlanContext(
        templates=[template],
        models=[model],
        sampling_parameters=[sampling1, sampling2],
        dataset_id_field="id",
        reference_field=None,
        metadata_fields=["type"],
        context_builder=None,
    )

    # Math problems: only low temperature
    math_strategy = plan.FilteredExpansionStrategy(
        task_filter=lambda row, tpl, mdl, smp: (
            row.get("type") == "math" and smp.temperature == 0.0
        )
    )

    # Code problems: both temperatures
    code_strategy = plan.CartesianExpansionStrategy()

    # Default: low temperature only
    default_strategy = plan.FilteredExpansionStrategy(
        task_filter=lambda row, tpl, mdl, smp: smp.temperature == 0.0
    )

    conditional = plan.ConditionalExpansionStrategy(
        rules=[
            (lambda row: row.get("type") == "math", math_strategy),
            (lambda row: row.get("type") == "code", code_strategy),
        ],
        default=default_strategy,
    )

    tasks = list(conditional.expand(dataset, context))

    # Math: 2 samples × 1 temp = 2 tasks
    # Code: 1 sample × 2 temps = 2 tasks
    # Total: 4 tasks
    assert len(tasks) == 4

    # Check math tasks have temp=0.0
    math_tasks = [t for t in tasks if t.metadata.get("type") == "math"]
    assert len(math_tasks) == 2
    assert all(t.sampling.temperature == 0.0 for t in math_tasks)

    # Check code tasks have both temperatures
    code_tasks = [t for t in tasks if t.metadata.get("type") == "code"]
    assert len(code_tasks) == 2
    temps = sorted([t.sampling.temperature for t in code_tasks])
    assert temps == [0.0, 0.7]


def test_conditional_expansion_strategy_default():
    """Test conditional expansion uses default for unmatched rows."""
    dataset = [
        {"id": "1", "type": "known"},
        {"id": "2", "type": "unknown"},
    ]

    template = templates.PromptTemplate(name="t1", template="Problem")
    model = entities.ModelSpec(identifier="model", provider="fake")
    sampling = entities.SamplingConfig(temperature=0.0, top_p=0.95, max_tokens=100)

    context = plan.PlanContext(
        templates=[template],
        models=[model],
        sampling_parameters=[sampling],
        dataset_id_field="id",
        reference_field=None,
        metadata_fields=["type"],
        context_builder=None,
    )

    # Strategy for known type: generate tasks
    known_strategy = plan.CartesianExpansionStrategy()

    # Default strategy: skip all
    default_strategy = plan.FilteredExpansionStrategy(
        task_filter=lambda row, tpl, mdl, smp: False  # Skip all
    )

    conditional = plan.ConditionalExpansionStrategy(
        rules=[
            (lambda row: row.get("type") == "known", known_strategy),
        ],
        default=default_strategy,
    )

    tasks = list(conditional.expand(dataset, context))

    # Only "known" type should generate tasks
    assert len(tasks) == 1
    assert tasks[0].metadata.get("type") == "known"


def test_chained_expansion_strategy():
    """Test chained expansion combines multiple strategies."""
    dataset = [
        {"id": "1", "difficulty": "easy"},
        {"id": "2", "difficulty": "hard"},
    ]

    template = templates.PromptTemplate(name="t1", template="Problem")
    model = entities.ModelSpec(identifier="model", provider="fake")
    sampling1 = entities.SamplingConfig(temperature=0.0, top_p=0.95, max_tokens=100)
    sampling2 = entities.SamplingConfig(temperature=0.9, top_p=0.95, max_tokens=100)

    context = plan.PlanContext(
        templates=[template],
        models=[model],
        sampling_parameters=[sampling1, sampling2],
        dataset_id_field="id",
        reference_field=None,
        metadata_fields=["difficulty"],
        context_builder=None,
    )

    # Strategy 1: All samples with temp=0.0
    baseline_strategy = plan.FilteredExpansionStrategy(
        task_filter=lambda row, tpl, mdl, smp: smp.temperature == 0.0
    )

    # Strategy 2: Hard samples with temp=0.9
    exploration_strategy = plan.FilteredExpansionStrategy(
        task_filter=lambda row, tpl, mdl, smp: (
            row.get("difficulty") == "hard" and smp.temperature == 0.9
        )
    )

    chained = plan.ChainedExpansionStrategy([baseline_strategy, exploration_strategy])
    tasks = list(chained.expand(dataset, context))

    # Baseline: 2 samples × temp=0.0 = 2 tasks
    # Exploration: 1 hard sample × temp=0.9 = 1 task
    # Total: 3 tasks
    assert len(tasks) == 3

    # Check we have the right combinations
    task_info = [(t.metadata.get("difficulty"), t.sampling.temperature) for t in tasks]
    assert ("easy", 0.0) in task_info
    assert ("hard", 0.0) in task_info
    assert ("hard", 0.9) in task_info


def test_flexible_generation_plan_default_strategy():
    """Test FlexibleGenerationPlan uses Cartesian by default."""
    dataset = [{"id": "1"}, {"id": "2"}]

    template = templates.PromptTemplate(name="t1", template="Problem")
    model1 = entities.ModelSpec(identifier="m1", provider="fake")
    model2 = entities.ModelSpec(identifier="m2", provider="fake")
    sampling = entities.SamplingConfig(temperature=0.0, top_p=0.95, max_tokens=100)

    # No expansion_strategy specified
    flexible_plan = plan.FlexibleGenerationPlan(
        templates=[template],
        models=[model1, model2],
        sampling_parameters=[sampling],
    )

    tasks = list(flexible_plan.expand(dataset))

    # Should use Cartesian: 2 × 1 × 2 × 1 = 4 tasks
    assert len(tasks) == 4


def test_flexible_generation_plan_with_filter():
    """Test FlexibleGenerationPlan with filtered strategy."""
    dataset = [
        {"id": "easy-1", "difficulty": "easy"},
        {"id": "hard-1", "difficulty": "hard"},
    ]

    template = templates.PromptTemplate(name="t1", template="Problem")
    cheap = entities.ModelSpec(identifier="cheap", provider="fake")
    expensive = entities.ModelSpec(identifier="expensive", provider="fake")
    sampling = entities.SamplingConfig(temperature=0.0, top_p=0.95, max_tokens=100)

    # Only expensive model on hard problems
    def expensive_on_hard(row, tpl, mdl, smp):
        if mdl.identifier == "cheap":
            return True
        return mdl.identifier == "expensive" and row.get("difficulty") == "hard"

    flexible_plan = plan.FlexibleGenerationPlan(
        templates=[template],
        models=[cheap, expensive],
        sampling_parameters=[sampling],
        expansion_strategy=plan.FilteredExpansionStrategy(
            task_filter=expensive_on_hard
        ),
        metadata_fields=["difficulty"],
    )

    tasks = list(flexible_plan.expand(dataset))

    # Should generate: 2 with cheap + 1 hard with expensive = 3 tasks
    assert len(tasks) == 3

    # Verify filtering worked
    expensive_tasks = [t for t in tasks if t.model.identifier == "expensive"]
    assert len(expensive_tasks) == 1
    assert expensive_tasks[0].metadata.get("difficulty") == "hard"


def test_flexible_generation_plan_metadata_fields():
    """Test that metadata fields are properly included."""
    dataset = [{"id": "1", "difficulty": "easy", "category": "math", "extra": "ignore"}]

    template = templates.PromptTemplate(name="t1", template="Problem")
    model = entities.ModelSpec(identifier="model", provider="fake")
    sampling = entities.SamplingConfig(temperature=0.0, top_p=0.95, max_tokens=100)

    flexible_plan = plan.FlexibleGenerationPlan(
        templates=[template],
        models=[model],
        sampling_parameters=[sampling],
        metadata_fields=["difficulty", "category"],  # Select specific fields
    )

    tasks = list(flexible_plan.expand(dataset))
    assert len(tasks) == 1

    task = tasks[0]
    assert task.metadata.get("difficulty") == "easy"
    assert task.metadata.get("category") == "math"
    assert "extra" not in task.metadata  # Should not be included


def test_flexible_generation_plan_context_builder():
    """Test FlexibleGenerationPlan with custom context builder."""
    dataset = [{"id": "1", "raw_problem": "What is 2+2?"}]

    template = templates.PromptTemplate(name="t1", template="{formatted_problem}")
    model = entities.ModelSpec(identifier="model", provider="fake")
    sampling = entities.SamplingConfig(temperature=0.0, top_p=0.95, max_tokens=100)

    # Context builder transforms raw data
    def build_context(row):
        return {
            "formatted_problem": f"Q: {row['raw_problem']}\nA:",
            "original": row["raw_problem"],
        }

    flexible_plan = plan.FlexibleGenerationPlan(
        templates=[template],
        models=[model],
        sampling_parameters=[sampling],
        context_builder=build_context,
    )

    tasks = list(flexible_plan.expand(dataset))
    assert len(tasks) == 1

    # Check that template was rendered with transformed context
    task = tasks[0]
    assert "Q: What is 2+2?" in task.prompt.text
    assert "A:" in task.prompt.text


def test_expansion_strategy_with_multiple_templates():
    """Test expansion strategies work correctly with multiple templates."""
    dataset = [{"id": "1"}]

    t1 = templates.PromptTemplate(name="direct", template="Q: {id}")
    t2 = templates.PromptTemplate(name="cot", template="Think step by step: {id}")
    model = entities.ModelSpec(identifier="model", provider="fake")
    sampling = entities.SamplingConfig(temperature=0.0, top_p=0.95, max_tokens=100)

    # Filter: only use CoT for hard problems
    def cot_filter(row, tpl, mdl, smp):
        if tpl.name == "direct":
            return True
        return tpl.name == "cot" and row.get("difficulty") == "hard"

    context = plan.PlanContext(
        templates=[t1, t2],
        models=[model],
        sampling_parameters=[sampling],
        dataset_id_field="id",
        reference_field=None,
        metadata_fields=(),
        context_builder=None,
    )

    # Easy sample: should only get direct template
    easy_dataset = [{"id": "1", "difficulty": "easy"}]
    strategy = plan.FilteredExpansionStrategy(task_filter=cot_filter)
    easy_tasks = list(strategy.expand(easy_dataset, context))
    assert len(easy_tasks) == 1
    assert easy_tasks[0].prompt.spec.name == "direct"

    # Hard sample: should get both templates
    hard_dataset = [{"id": "1", "difficulty": "hard"}]
    hard_tasks = list(strategy.expand(hard_dataset, context))
    assert len(hard_tasks) == 2
    template_names = {t.prompt.spec.name for t in hard_tasks}
    assert template_names == {"direct", "cot"}


def test_expansion_preserves_reference():
    """Test that expansion strategies preserve reference values."""
    dataset = [{"id": "1", "problem": "test", "answer": "42"}]

    template = templates.PromptTemplate(name="t1", template="{problem}")
    model = entities.ModelSpec(identifier="model", provider="fake")
    sampling = entities.SamplingConfig(temperature=0.0, top_p=0.95, max_tokens=100)

    context = plan.PlanContext(
        templates=[template],
        models=[model],
        sampling_parameters=[sampling],
        dataset_id_field="id",
        reference_field="answer",
        metadata_fields=(),
        context_builder=None,
    )

    strategy = plan.CartesianExpansionStrategy()
    tasks = list(strategy.expand(dataset, context))

    assert len(tasks) == 1
    assert tasks[0].reference is not None
    assert tasks[0].reference.value == "42"
    assert tasks[0].reference.kind == "answer"
