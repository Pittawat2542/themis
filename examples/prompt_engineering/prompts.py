"""Prompt template definitions for the prompt engineering experiment."""

from themis.generation import templates


def create_prompt_templates(config):
    """Create prompt templates from configuration."""
    templates_list = []
    for variant in config.prompt_variants:
        template = templates.PromptTemplate(
            name=variant.name,
            template=variant.template,
            metadata=variant.metadata
        )
        templates_list.append(template)
    return templates_list


# Predefined prompt templates for convenience
ZERO_SHOT_MATH = templates.PromptTemplate(
    name="zero-shot-math",
    template="""
You are an expert mathematician. Solve the problem below and respond with a JSON object
containing `answer` and `reasoning` keys only.

Problem:
{problem}
    """.strip(),
    metadata={"strategy": "zero-shot", "domain": "math"}
)

FEW_SHOT_MATH = templates.PromptTemplate(
    name="few-shot-math", 
    template="""
You are an expert mathematician. Solve the problem below and respond with a JSON object
containing `answer` and `reasoning` keys only.

Example:
Problem: What is 2+2?
Answer: {{"answer": "4", "reasoning": "Simple addition of 2 and 2"}}

Problem:
{problem}

Answer:
    """.strip(),
    metadata={"strategy": "few-shot", "domain": "math"}
)

CHAIN_OF_THOUGHT_MATH = templates.PromptTemplate(
    name="chain-of-thought-math",
    template="""
You are an expert mathematician. Think through the problem step-by-step, then provide your final answer.

Problem: {problem}

Step-by-step reasoning:
    """.strip(),
    metadata={"strategy": "chain-of-thought", "domain": "math"}
)

# More prompt templates for different domains
ZERO_SHOT_GENERAL = templates.PromptTemplate(
    name="zero-shot-general",
    template="Please answer the following: {question}",
    metadata={"strategy": "zero-shot", "domain": "general"}
)

FEW_SHOT_GENERAL = templates.PromptTemplate(
    name="few-shot-general",
    template="""
Here's an example of how to answer:
Question: What is the capital of France?
Answer: Paris

Now answer: {question}
    """.strip(),
    metadata={"strategy": "few-shot", "domain": "general"}
)

__all__ = [
    "create_prompt_templates",
    "ZERO_SHOT_MATH",
    "FEW_SHOT_MATH", 
    "CHAIN_OF_THOUGHT_MATH",
    "ZERO_SHOT_GENERAL",
    "FEW_SHOT_GENERAL",
]