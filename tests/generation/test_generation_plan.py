from collections import Counter

from themis.core import entities as core_entities
from themis.generation import plan as generation_plan
from themis.generation import templates


def make_sampling(temperature: float, top_p: float, max_tokens: int):
    return core_entities.SamplingConfig(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


def make_model(identifier: str) -> core_entities.ModelSpec:
    return core_entities.ModelSpec(identifier=identifier, provider="test")


def test_plan_expands_cross_product_for_dataset_models_and_sampling():
    template = templates.PromptTemplate(
        name="baseline",
        template="Summarize {topic} in three sentences.",
    )
    plan = generation_plan.GenerationPlan(
        templates=[template],
        models=[make_model("gpt-4o"), make_model("gpt-4o-mini")],
        sampling_parameters=[
            make_sampling(temperature=0.1, top_p=0.9, max_tokens=128),
            make_sampling(temperature=0.8, top_p=0.95, max_tokens=256),
        ],
    )
    dataset = [
        {"id": "sample-1", "topic": "diffusion"},
        {"id": "sample-2", "topic": "transformers"},
    ]

    requests = list(plan.expand(dataset))

    assert len(requests) == 8
    assert {req.model.identifier for req in requests} == {"gpt-4o", "gpt-4o-mini"}
    assert {req.prompt.context["topic"] for req in requests} == {
        "diffusion",
        "transformers",
    }
    assert Counter(req.sampling.temperature for req in requests) == Counter(
        {0.1: 4, 0.8: 4}
    )
    assert all(req.prompt.text.startswith("Summarize") for req in requests)


def test_plan_carries_template_metadata_into_each_request():
    template = templates.PromptTemplate(
        name="detailed",
        template="Provide a {style} explanation of {topic}.",
        metadata={"style": "detailed"},
    )
    plan = generation_plan.GenerationPlan(
        templates=[template],
        models=[make_model("gpt-4o")],
        sampling_parameters=[make_sampling(0.3, 0.9, 64)],
    )
    dataset = [{"id": "sample-1", "topic": "graph search", "style": "tutorial"}]

    [request] = list(plan.expand(dataset))

    assert request.prompt.spec.name == "detailed"
    assert request.metadata["template_style"] == "detailed"
    assert request.metadata["dataset_id"] == "sample-1"
    assert request.prompt.context["style"] == "tutorial"


def test_plan_supports_custom_identifier_and_reference_fields():
    template = templates.PromptTemplate(
        name="math-json",
        template="Solve: {problem}",
    )
    plan = generation_plan.GenerationPlan(
        templates=[template],
        models=[make_model("fake")],
        sampling_parameters=[make_sampling(0.0, 1.0, 128)],
        dataset_id_field="unique_id",
        reference_field="answer",
        metadata_fields=("subject", "level"),
        context_builder=lambda row: {"problem": row["problem"]},
    )
    dataset = [
        {
            "unique_id": "math-1",
            "problem": "What is 1+1?",
            "answer": "2",
            "subject": "arithmetic",
            "level": 1,
        }
    ]

    [request] = list(plan.expand(dataset))

    assert request.metadata["dataset_id"] == "math-1"
    assert request.metadata["subject"] == "arithmetic"
    assert request.metadata["level"] == 1
    assert request.reference.value == "2"
    assert request.prompt.context == {"problem": "What is 1+1?"}
