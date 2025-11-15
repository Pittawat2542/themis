import pytest

from themis.core import entities as core_entities
from themis.experiment import builder as experiment_builder
from themis.generation import templates
from themis.project import Project, ProjectExperiment


def _experiment_definition(
    name: str = "demo",
) -> experiment_builder.ExperimentDefinition:
    template = templates.PromptTemplate(
        name=f"{name}-template", template="Solve {problem}"
    )
    sampling = core_entities.SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=32)
    model_spec = core_entities.ModelSpec(identifier=f"{name}-model", provider="fake")
    binding = experiment_builder.ModelBinding(spec=model_spec, provider_name="fake")
    return experiment_builder.ExperimentDefinition(
        templates=[template],
        sampling_parameters=[sampling],
        model_bindings=[binding],
    )


def test_project_registers_experiments_and_merges_metadata():
    project = Project(
        project_id="proj-alpha",
        name="Alpha",
        metadata={"owner": "eval", "priority": "low"},
        tags=("math",),
    )

    experiment = project.create_experiment(
        name="baseline",
        definition=_experiment_definition("baseline"),
        metadata={"priority": "high", "notes": "first pass"},
        tags=("baseline",),
    )

    assert project.get_experiment("baseline") is experiment
    assert project.list_experiment_names() == ("baseline",)
    merged = project.metadata_for_experiment("baseline")
    assert merged == {"owner": "eval", "priority": "high", "notes": "first pass"}


def test_project_rejects_duplicate_experiment_names():
    definition = _experiment_definition()
    exp = ProjectExperiment(name="baseline", definition=definition)
    project = Project(project_id="proj", name="Demo", experiments=(exp,))

    assert project.get_experiment("baseline") is exp

    with pytest.raises(ValueError):
        project.create_experiment(name="baseline", definition=definition)


def test_project_detects_duplicates_on_initialization():
    definition = _experiment_definition()
    exp_a = ProjectExperiment(name="shared", definition=definition)
    exp_b = ProjectExperiment(name="shared", definition=definition)

    with pytest.raises(ValueError):
        Project(project_id="proj", name="Demo", experiments=(exp_a, exp_b))
