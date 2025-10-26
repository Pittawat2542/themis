import pytest

from themis.generation import templates


class TestPromptTemplate:
    def test_render_fills_all_variables(self):
        template = templates.PromptTemplate(
            name="baseline",
            template="Summarize {topic} for a {audience} audience.",
        )

        rendered = template.render(
            topic="diffusion models", audience="college students"
        )

        assert rendered == "Summarize diffusion models for a college students audience."

    def test_missing_variables_raise_meaningful_error(self):
        template = templates.PromptTemplate(
            name="baseline",
            template="Explain {topic} in {language}.",
        )

        with pytest.raises(templates.TemplateRenderingError):
            template.render(topic="variance propagation")

    def test_variant_expansion_handles_multiple_fields(self):
        template = templates.PromptTemplate(
            name="tone",
            template="Give a {tone} summary of {topic} that is {length} in length.",
        )

        base_context = {"topic": "Monte Carlo", "length": "short"}
        variants = {"tone": ["neutral", "playful"], "length": ["short", "long"]}

        prompts = template.expand_variants(
            base_context=base_context, variant_values=variants
        )

        assert {p.prompt_text for p in prompts} == {
            "Give a neutral summary of Monte Carlo that is short in length.",
            "Give a playful summary of Monte Carlo that is short in length.",
            "Give a neutral summary of Monte Carlo that is long in length.",
            "Give a playful summary of Monte Carlo that is long in length.",
        }
        assert all(p.template_name == "tone" for p in prompts)
