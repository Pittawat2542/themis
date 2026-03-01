"""Multiple Choice Question (MCQ) benchmark presets and factories."""

from __future__ import annotations

import string
from typing import Any, Sequence

from themis.exceptions import ConfigurationError
from themis.generation.templates import PromptTemplate
from themis.presets.core import BenchmarkPreset, register_benchmark


def _to_dict_samples(samples: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        sample.to_generation_example()
        if hasattr(sample, "to_generation_example")
        else dict(sample)
        for sample in samples
    ]


def _format_mcq_options(choices: Sequence[str], labels: Sequence[str]) -> str:
    return "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))


def _normalize_mcq_answer(
    answer: Any,
    choices: Sequence[str],
    labels: Sequence[str],
) -> str:
    if answer is None:
        return ""
    if isinstance(answer, bool):
        return str(answer)
    if isinstance(answer, (int, float)):
        index = int(answer)
        if 1 <= index <= len(choices):
            return labels[index - 1]
        if 0 <= index < len(choices):
            return labels[index]
    text = str(answer).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith("option "):
        text = text.split(" ", 1)[-1].strip()
    if lowered.startswith("choice "):
        text = text.split(" ", 1)[-1].strip()
    if len(text) >= 2 and text[1] in {".", ")", ":", "-"}:
        text = text[0]
    if len(text) == 1 and text.isalpha():
        letter = text.upper()
        if letter in labels:
            return letter
    for idx, choice in enumerate(choices):
        if text == str(choice).strip():
            return labels[idx]
    for idx, choice in enumerate(choices):
        if text.lower() == str(choice).strip().lower():
            return labels[idx]
    return text


def _normalize_mcq_samples(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for sample in samples:
        row = dict(sample)
        choices = row.get("choices") or row.get("options")
        if not isinstance(choices, (list, tuple)):
            normalized.append(row)
            continue
        choices_list = [str(choice) for choice in choices]
        labels = row.get("choice_labels")
        if isinstance(labels, (list, tuple)) and labels:
            labels_list = [str(label) for label in labels][: len(choices_list)]
        else:
            labels_list = list(string.ascii_uppercase[: len(choices_list)])
        row["choices"] = choices_list
        row["choice_labels"] = labels_list
        row["options"] = _format_mcq_options(choices_list, labels_list)
        row["answer"] = _normalize_mcq_answer(
            row.get("answer"),
            choices_list,
            labels_list,
        )
        normalized.append(row)
    return normalized


def _create_mcq_preset(
    name: str,
    dataset_name: str,
    prompt_template_str: str,
    description: str,
    dataset_kwargs: dict[str, Any] | None = None,
    metadata_fields: tuple[str, ...] = ("subject",),
    reference_field: str = "answer",
) -> BenchmarkPreset:
    """Create a generic MCQ benchmark preset."""
    from themis.evaluation.extractors.identity_extractor import IdentityExtractor
    from themis.evaluation.metrics.exact_match import ExactMatch

    kwargs = dataset_kwargs or {}

    def load_dataset(limit: int | None = None) -> Sequence[dict[str, Any]]:
        if dataset_name == "mmlu-pro":
            from themis.datasets.mmlu_pro import load_mmlu_pro

            samples = load_mmlu_pro(limit=limit, **kwargs)
        elif dataset_name == "super_gpqa":
            from themis.datasets.super_gpqa import load_super_gpqa

            samples = load_super_gpqa(limit=limit, **kwargs)
        elif dataset_name == "gpqa":
            from themis.datasets.gpqa import load_gpqa

            samples = load_gpqa(limit=limit, **kwargs)
        elif dataset_name == "medmcqa":
            from themis.datasets.medmcqa import load_medmcqa

            samples = load_medmcqa(limit=limit, **kwargs)
        elif dataset_name == "med_qa":
            from themis.datasets.med_qa import load_med_qa

            samples = load_med_qa(limit=limit, **kwargs)
        elif dataset_name == "sciq":
            from themis.datasets.sciq import load_sciq

            samples = load_sciq(limit=limit, **kwargs)
        elif dataset_name == "commonsense_qa":
            from themis.datasets.commonsense_qa import load_commonsense_qa

            samples = load_commonsense_qa(limit=limit, **kwargs)
        elif dataset_name == "piqa":
            from themis.datasets.piqa import load_piqa

            samples = load_piqa(limit=limit, **kwargs)
        elif dataset_name == "social_i_qa":
            from themis.datasets.social_i_qa import load_social_i_qa

            samples = load_social_i_qa(limit=limit, **kwargs)
        elif dataset_name == "coqa":
            from themis.datasets.coqa import load_coqa

            samples = load_coqa(limit=limit, **kwargs)
            return _to_dict_samples(samples)  # CoQA doesn't need _normalize_mcq_samples
        else:
            raise ConfigurationError(f"Unsupported MCQ dataset loader: {dataset_name}")

        return _normalize_mcq_samples(_to_dict_samples(samples))

    prompt_template = PromptTemplate(
        name=f"{name}-zero-shot",
        template=prompt_template_str,
    )

    return BenchmarkPreset(
        name=name,
        prompt_template=prompt_template,
        metrics=[ExactMatch()],
        extractor=IdentityExtractor(),
        dataset_loader=load_dataset,
        metadata_fields=metadata_fields,
        reference_field=reference_field,
        dataset_id_field="unique_id",
        description=description,
    )


def _create_mmlu_pro_preset() -> BenchmarkPreset:
    return _create_mcq_preset(
        name="mmlu-pro",
        dataset_name="mmlu-pro",
        prompt_template_str=(
            "Answer the following multiple choice question.\n\n"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Answer (letter):"
        ),
        description="MMLU-Pro professional-level multiple choice questions",
        dataset_kwargs={"source": "huggingface", "split": "test"},
    )


def _create_supergpqa_preset() -> BenchmarkPreset:
    return _create_mcq_preset(
        name="supergpqa",
        dataset_name="super_gpqa",
        prompt_template_str=(
            "Answer the following science question.\n\n"
            "Question: {question}\n\n"
            "Choices:\n{options}\n\n"
            "Answer (letter):"
        ),
        description="SuperGPQA graduate-level science questions",
        dataset_kwargs={"source": "huggingface", "split": "test"},
    )


def _create_gpqa_preset() -> BenchmarkPreset:
    return _create_mcq_preset(
        name="gpqa",
        dataset_name="gpqa",
        prompt_template_str=(
            "Answer the following question.\n\n"
            "Question: {question}\n\n"
            "Choices:\n{options}\n\n"
            "Answer (letter):"
        ),
        description="GPQA graduate-level science questions",
        dataset_kwargs={"source": "huggingface", "split": "test", "subset": "default"},
    )


def _create_medmcqa_preset() -> BenchmarkPreset:
    return _create_mcq_preset(
        name="medmcqa",
        dataset_name="medmcqa",
        prompt_template_str=(
            "Answer the following medical multiple choice question.\n\n"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Answer (letter):"
        ),
        description="MedMCQA medical entrance exam questions",
        dataset_kwargs={"source": "huggingface", "split": "test"},
    )


def _create_med_qa_preset() -> BenchmarkPreset:
    return _create_mcq_preset(
        name="med_qa",
        dataset_name="med_qa",
        prompt_template_str=(
            "Answer the following medical multiple choice question.\n\n"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Answer (letter):"
        ),
        description="MedQA multiple choice medical QA benchmark",
        dataset_kwargs={"source": "huggingface", "split": "test"},
    )


def _create_sciq_preset() -> BenchmarkPreset:
    return _create_mcq_preset(
        name="sciq",
        dataset_name="sciq",
        prompt_template_str=(
            "Answer the following science question.\n\n"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Answer (letter):"
        ),
        description="SciQ science multiple choice questions",
        dataset_kwargs={"source": "huggingface", "split": "test"},
        metadata_fields=(),
    )


def _create_commonsense_qa_preset() -> BenchmarkPreset:
    return _create_mcq_preset(
        name="commonsense_qa",
        dataset_name="commonsense_qa",
        prompt_template_str=(
            "Answer the following commonsense question.\n\n"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Answer (letter):"
        ),
        description="CommonsenseQA multiple choice reasoning benchmark",
        dataset_kwargs={"source": "huggingface", "split": "validation"},
        metadata_fields=("concept",),
    )


def _create_piqa_preset() -> BenchmarkPreset:
    return _create_mcq_preset(
        name="piqa",
        dataset_name="piqa",
        prompt_template_str=(
            "Choose the best answer for the goal.\n\n"
            "Goal: {goal}\n\n"
            "Options:\n{options}\n\n"
            "Answer (letter):"
        ),
        description="PIQA physical commonsense reasoning benchmark",
        dataset_kwargs={"source": "huggingface", "split": "validation"},
        metadata_fields=(),
    )


def _create_social_i_qa_preset() -> BenchmarkPreset:
    return _create_mcq_preset(
        name="social_i_qa",
        dataset_name="social_i_qa",
        prompt_template_str=(
            "Answer the question based on the social context.\n\n"
            "Context: {context}\n"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Answer (letter):"
        ),
        description="Social IQA commonsense reasoning benchmark",
        dataset_kwargs={"source": "huggingface", "split": "validation"},
        metadata_fields=(),
    )


def _create_coqa_preset() -> BenchmarkPreset:
    return _create_mcq_preset(
        name="coqa",
        dataset_name="coqa",
        prompt_template_str=(
            "Answer the question based on the passage.\n\n"
            "Passage: {story}\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
        description="CoQA conversational question answering benchmark",
        dataset_kwargs={"source": "huggingface", "split": "validation"},
        metadata_fields=("turn",),
    )


def _register_mcq_benchmarks() -> None:
    """Register all built-in MCQ benchmarks."""
    register_benchmark(_create_mmlu_pro_preset())
    register_benchmark(_create_supergpqa_preset())
    register_benchmark(_create_gpqa_preset())
    register_benchmark(_create_medmcqa_preset())
    register_benchmark(_create_med_qa_preset())
    register_benchmark(_create_sciq_preset())
    register_benchmark(_create_commonsense_qa_preset())
    register_benchmark(_create_piqa_preset())
    register_benchmark(_create_social_i_qa_preset())
    register_benchmark(_create_coqa_preset())
