import json

from themis.core import entities as core_entities
from themis.generation import clients


def make_task(prompt: str) -> core_entities.GenerationTask:
    prompt_spec = core_entities.PromptSpec(name="tmp", template="{prompt}")
    prompt_render = core_entities.PromptRender(
        spec=prompt_spec,
        text=prompt,
        context={"prompt": prompt},
    )
    sampling = core_entities.SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=64)
    model = core_entities.ModelSpec(identifier="fake", provider="test")
    return core_entities.GenerationTask(
        prompt=prompt_render,
        model=model,
        sampling=sampling,
    )


def test_fake_math_model_client_returns_structured_json():
    client = clients.FakeMathModelClient(seed=42)
    response = client.generate(make_task("What is 2 + 2?"))

    payload = json.loads(response.output.text)
    assert payload["answer"] == "4"
    assert not payload["answer"].startswith("\\boxed")
    assert "reasoning" in payload


def test_fake_math_model_client_can_handle_polar_conversion():
    client = clients.FakeMathModelClient(seed=2)
    response = client.generate(
        make_task("Convert the point (0,3) to polar coordinates.")
    )

    payload = json.loads(response.output.text)
    assert payload["answer"].startswith("\\left( 3")
    assert not payload["answer"].startswith("\\boxed")
