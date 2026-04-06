"""Benchmark dataset materializers."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from themis.catalog.loaders import (
    BenchmarkSourceRequest,
    load_huggingface_raw_rows,
    load_huggingface_rows,
)
from themis.core.base import JSONValue
from themis.core.models import Case, Dataset

DatasetRowLoader = Callable[[BenchmarkSourceRequest], list[dict[str, object]]]

_MATH_FAMILY_IDS = {
    "aime_2025",
    "aime_2026",
    "apex_2025",
    "beyond_aime",
    "hmmt_feb_2025",
    "hmmt_nov_2025",
    "phybench",
}
_MCQ_FAMILY_IDS = {"supergpqa", "encyclo_k"}


def materialize_benchmark_dataset(
    definition,
    *,
    loader: DatasetRowLoader | None = None,
) -> Dataset:
    row_loader = loader or _default_loader
    base_name = definition.base_benchmark_id
    if base_name in _MATH_FAMILY_IDS:
        return _materialize_math_family(definition, row_loader=row_loader)
    if base_name == "imo_answerbench":
        return _materialize_imo_answerbench(definition, row_loader=row_loader)
    if base_name == "mmlu_pro":
        return _materialize_mmlu_pro(definition, row_loader=row_loader)
    if base_name in _MCQ_FAMILY_IDS:
        return _materialize_mcq_family(definition, row_loader=row_loader)
    if base_name == "babe":
        return _materialize_babe(definition, row_loader=row_loader)
    if base_name == "gpqa_diamond":
        return _materialize_gpqa_diamond(definition, row_loader=row_loader)
    if base_name == "mmmlu":
        return _materialize_mmmlu(definition, row_loader=row_loader)
    if base_name == "superchem":
        return _materialize_superchem(definition, row_loader=row_loader)
    if base_name == "frontierscience":
        return _materialize_frontierscience(definition, row_loader=row_loader)
    if base_name == "healthbench":
        return _materialize_healthbench(definition, row_loader=row_loader)
    if base_name == "lpfqa":
        return _materialize_lpfqa(definition, row_loader=row_loader)
    if base_name == "simpleqa_verified":
        return _materialize_simpleqa(definition, row_loader=row_loader)
    if base_name == "hle":
        return _materialize_hle(definition, row_loader=row_loader)
    if base_name == "procbench":
        return _materialize_procbench(definition, row_loader=row_loader)
    if base_name == "rolebench":
        return _materialize_rolebench(definition, row_loader=row_loader)
    if base_name == "codeforces":
        return _materialize_codeforces(definition, row_loader=row_loader)
    if base_name == "aethercode":
        return _materialize_aethercode(definition, row_loader=row_loader)
    if base_name == "livecodebench":
        return _materialize_livecodebench(definition, row_loader=row_loader)
    if base_name in {"humaneval", "humaneval_plus"}:
        return _materialize_humaneval(definition, row_loader=row_loader)
    raise ValueError(
        f"Benchmark materialization is not implemented for {definition.base_benchmark_id}"
    )


def _default_loader(
    request: BenchmarkSourceRequest,
) -> list[dict[str, object]]:
    if request.source_kind == "huggingface_dataset":
        return load_huggingface_rows(
            request.dataset_id,
            request.split,
            request.revision,
            config_name=request.config_name,
        )
    if request.source_kind == "huggingface_raw_files":
        return load_huggingface_raw_rows(
            request.dataset_id,
            files=request.files,
            revision=request.revision,
        )
    raise ValueError(f"Unknown benchmark source kind: {request.source_kind}")


def _load_rows(
    definition,
    row_loader: DatasetRowLoader,
    *,
    config_name: str | None = None,
    revision: str | None = None,
    files: list[str] | None = None,
) -> list[dict[str, object]]:
    return row_loader(
        BenchmarkSourceRequest(
            source_kind=definition.source_kind,
            dataset_id=definition.dataset_id,
            split=definition.split,
            revision=revision,
            config_name=config_name,
            files=list(files or []),
        )
    )


def _materialize_math_family(definition, *, row_loader: DatasetRowLoader) -> Dataset:
    rows = _load_rows(definition, row_loader, revision=definition.dataset_revision)
    cases = [
        Case(
            case_id=str(
                row.get("problem_idx", row.get("id", f"{definition.split}-{index}"))
            ),
            input=(
                "Solve the following math problem. "
                "Return only the final answer in \\boxed{...}.\n\n"
                f"Problem:\n{str(row.get('problem', row.get('content', '')))}"
            ),
            expected_output={"answer": str(row.get("answer", "")).strip()},
            metadata=_string_metadata(row, ["problem_type", "source", "tag"]),
        )
        for index, row in enumerate(rows, start=1)
    ]
    return _dataset(definition, cases)


def _materialize_imo_answerbench(
    definition, *, row_loader: DatasetRowLoader
) -> Dataset:
    rows = _load_rows(definition, row_loader, revision=definition.dataset_revision)
    cases = []
    for index, row in enumerate(rows, start=1):
        cases.append(
            Case(
                case_id=str(row.get("Problem ID", f"{definition.split}-{index}")),
                input=(
                    "Solve the following math problem. "
                    "Return only the final answer in \\boxed{...}.\n\n"
                    f"Problem:\n{str(row.get('Problem', ''))}"
                ),
                expected_output={
                    "answer": str(
                        row.get("Short Answer", row.get("answer", ""))
                    ).strip()
                },
                metadata={
                    key: str(row[source_key])
                    for source_key, key in (
                        ("Category", "category"),
                        ("Subcategory", "subcategory"),
                        ("Source", "source"),
                    )
                    if row.get(source_key) is not None
                },
            )
        )
    return _dataset(definition, cases)


def _materialize_mmlu_pro(definition, *, row_loader: DatasetRowLoader) -> Dataset:
    rows = _load_rows(definition, row_loader)
    return _dataset(
        definition,
        [
            Case(
                case_id=str(
                    row.get("item_id", row.get("id", f"{definition.split}-{index}"))
                ),
                input=_mcq_prompt(
                    question=str(row.get("question", "")),
                    options=_normalize_options(row.get("options")),
                ),
                expected_output={"choice": str(row.get("answer", "")).strip().upper()},
                metadata=_string_metadata(row, ["category", "src"]),
            )
            for index, row in enumerate(rows, start=1)
        ],
    )


def _materialize_mcq_family(definition, *, row_loader: DatasetRowLoader) -> Dataset:
    rows = _load_rows(definition, row_loader, revision=definition.dataset_revision)
    cases: list[Case] = []
    for index, row in enumerate(rows, start=1):
        options = _normalize_options(row.get("options"))
        answer = str(row.get("answer_letter", row.get("answer", ""))).strip().upper()
        cases.append(
            Case(
                case_id=str(row.get("item_id", f"{definition.split}-{index}")),
                input=_mcq_prompt(
                    question=str(row.get("question", "")), options=options
                ),
                expected_output={"choice": answer},
                metadata=_string_metadata(
                    row, ["discipline", "field", "subfield", "difficulty"]
                ),
            )
        )
    return _dataset(definition, cases)


def _materialize_babe(definition, *, row_loader: DatasetRowLoader) -> Dataset:
    rows = _load_rows(definition, row_loader, revision=definition.dataset_revision)
    cases = []
    for index, row in enumerate(rows, start=1):
        label_value = row.get("label", 0)
        label = int(label_value) if isinstance(label_value, int | float | bool) else 0
        options = ["Entirely factual", "Opinionated or subjective"]
        question = str(row.get("text", ""))
        cases.append(
            Case(
                case_id=str(row.get("uuid", f"{definition.split}-{index}")),
                input=_mcq_prompt(question=question, options=options),
                expected_output={"choice": "A" if label == 0 else "B"},
                metadata=_string_metadata(
                    row, ["outlet", "topic", "type", "label_opinion"]
                ),
            )
        )
    return _dataset(definition, cases)


def _materialize_gpqa_diamond(definition, *, row_loader: DatasetRowLoader) -> Dataset:
    rows = _load_rows(definition, row_loader, revision=definition.dataset_revision)
    cases = []
    for index, row in enumerate(rows, start=1):
        question, options = _parse_gpqa_diamond_question(str(row.get("question", "")))
        cases.append(
            Case(
                case_id=str(row.get("item_id", f"{definition.split}-{index}")),
                input=_mcq_prompt(question=question, options=options),
                expected_output={"choice": str(row.get("answer", "")).strip().upper()},
                metadata={},
            )
        )
    return _dataset(definition, cases)


def _materialize_mmmlu(definition, *, row_loader: DatasetRowLoader) -> Dataset:
    config_name = definition.variant or "default"
    rows = _load_rows(
        definition,
        row_loader,
        revision=definition.dataset_revision,
        config_name=config_name,
    )
    cases = []
    for index, row in enumerate(rows, start=1):
        options = [
            str(row.get("A", "")),
            str(row.get("B", "")),
            str(row.get("C", "")),
            str(row.get("D", "")),
        ]
        cases.append(
            Case(
                case_id=str(row.get("Unnamed: 0", f"{definition.split}-{index}")),
                input=_mcq_prompt(
                    question=str(row.get("Question", "")), options=options
                ),
                expected_output={"choice": str(row.get("Answer", "")).strip().upper()},
                metadata={
                    "subject": str(row.get("Subject", "")),
                    "language": config_name,
                },
            )
        )
    return _dataset(definition, cases)


def _materialize_superchem(definition, *, row_loader: DatasetRowLoader) -> Dataset:
    language = definition.variant or "en"
    rows = _load_rows(
        definition,
        row_loader,
        revision=definition.dataset_revision,
        config_name="default",
    )
    cases = []
    for index, row in enumerate(rows, start=1):
        options = _superchem_options(row, language=language)
        question = str(row.get(f"question_{language}", ""))
        image_urls = _string_list(row.get("question_images"))
        prompt_text = _mcq_prompt(question=question, options=options)
        if image_urls:
            prompt_text = "\n\n".join(
                [prompt_text, "Images:\n" + "\n".join(image_urls)]
            )
        cases.append(
            Case(
                case_id=str(row.get("uuid", f"{definition.split}-{index}")),
                input=prompt_text,
                expected_output={"choice": _superchem_answer(row, language=language)},
                metadata={
                    **_string_metadata(row, ["field", "question_type"]),
                    "language": language,
                },
            )
        )
    return _dataset(definition, cases)


def _materialize_frontierscience(
    definition, *, row_loader: DatasetRowLoader
) -> Dataset:
    rows = _load_rows(definition, row_loader, revision=definition.dataset_revision)
    cases = [
        Case(
            case_id=str(row.get("item_id", f"{definition.split}-{index}")),
            input=str(row.get("problem", "")),
            expected_output={"rubric": str(row.get("answer", ""))},
            metadata=_string_metadata(row, ["subject", "task_group_id"]),
        )
        for index, row in enumerate(rows, start=1)
    ]
    return _dataset(definition, cases)


def _materialize_healthbench(definition, *, row_loader: DatasetRowLoader) -> Dataset:
    rows = _load_rows(definition, row_loader, revision=definition.dataset_revision)
    cases = []
    for index, row in enumerate(rows, start=1):
        prompt_messages = _prompt_messages_from_payload(row)
        prompt_text = "\n\n".join(
            f"{message['role']}: {message['content']}" for message in prompt_messages
        )
        expected_output = _json_dict(
            {
                "rubrics": row.get("rubrics", []),
                "ideal_completion": _ideal_completion(row),
            }
        )
        cases.append(
            Case(
                case_id=str(row.get("prompt_id", f"{definition.split}-{index}")),
                input=prompt_text,
                expected_output=expected_output,
                metadata={
                    "prompt_id": str(
                        row.get("prompt_id", f"{definition.split}-{index}")
                    ),
                    "example_tags": ", ".join(_string_list(row.get("example_tags"))),
                },
            )
        )
    return _dataset(definition, cases)


def _materialize_lpfqa(definition, *, row_loader: DatasetRowLoader) -> Dataset:
    rows = _load_rows(definition, row_loader, revision=definition.dataset_revision)
    cases = []
    for index, row in enumerate(rows, start=1):
        cases.append(
            Case(
                case_id=str(row.get("prompt_id", f"{definition.split}-{index}")),
                input=str(row.get("prompt", "")),
                expected_output=_json_dict(
                    {
                        "response_reference": str(row.get("response_reference", "")),
                        "judge_prompt_template": str(
                            row.get(
                                "judge_prompt_template",
                                "{response_reference}\n{response}",
                            )
                        ),
                        "judge_system_prompt": str(row.get("judge_system_prompt", "")),
                    }
                ),
                metadata=_string_metadata(row, ["primary_domain"]),
            )
        )
    return _dataset(definition, cases)


def _materialize_simpleqa(definition, *, row_loader: DatasetRowLoader) -> Dataset:
    rows = _load_rows(definition, row_loader, revision=definition.dataset_revision)
    cases = []
    for index, row in enumerate(rows, start=1):
        cases.append(
            Case(
                case_id=str(row.get("original_index", f"{definition.split}-{index}")),
                input=str(row.get("problem", "")),
                expected_output={"answer": str(row.get("answer", ""))},
                metadata=_string_metadata(
                    row, ["topic", "answer_type", "multi_step", "requires_reasoning"]
                ),
            )
        )
    return _dataset(definition, cases)


def _materialize_hle(definition, *, row_loader: DatasetRowLoader) -> Dataset:
    variant = definition.variant or ""
    variant_tokens = {token.strip() for token in variant.split(",") if token.strip()}
    rows = _load_rows(definition, row_loader, revision=definition.dataset_revision)
    cases = []
    for index, row in enumerate(rows, start=1):
        image = row.get("image")
        if "text_only" in variant_tokens and isinstance(image, str) and image.strip():
            continue
        cases.append(
            Case(
                case_id=str(row.get("id", f"{definition.split}-{index}")),
                input=str(row.get("question", "")),
                expected_output={"answer": str(row.get("answer", ""))},
                metadata={"variant": variant or "text_only"},
            )
        )
    return _dataset(definition, cases)


def _materialize_procbench(definition, *, row_loader: DatasetRowLoader) -> Dataset:
    rows = _load_rows(definition, row_loader, revision=definition.dataset_revision)
    requested_task = definition.variant
    cases = []
    for index, row in enumerate(rows, start=1):
        task_name = str(row.get("task_name", ""))
        if requested_task is not None and task_name != requested_task:
            continue
        label = row.get("label")
        final_value = label.get("final") if isinstance(label, dict) else None
        cases.append(
            Case(
                case_id=str(row.get("problem_name", f"{definition.split}-{index}")),
                input=str(row.get("prompt", "")),
                expected_output=_json_dict({"answer": final_value}),
                metadata=_string_metadata(row, ["task_name", "example_name"]),
            )
        )
    return _dataset(definition, cases)


def _materialize_rolebench(definition, *, row_loader: DatasetRowLoader) -> Dataset:
    variants = (
        [definition.variant]
        if definition.variant
        else ["instruction_generalization_eng", "role_generalization_eng"]
    )
    cases: list[Case] = []
    for variant in variants:
        assert variant is not None
        rows = _load_rows(
            definition,
            row_loader,
            revision=None
            if definition.source_kind == "huggingface_raw_files"
            else definition.dataset_revision,
            config_name=None
            if definition.source_kind == "huggingface_raw_files"
            else variant,
            files=[_rolebench_source_file(definition, variant)]
            if definition.source_kind == "huggingface_raw_files"
            else None,
        )
        for index, row in enumerate(rows, start=1):
            generated = row.get("generated")
            expected = (
                str(generated[0]) if isinstance(generated, list) and generated else ""
            )
            prompt = (
                f"You are {str(row.get('role', ''))}, your description is: {str(row.get('desc', ''))}. "
                "Answer the following question while staying fully in character.\n\n"
                f"Question:\n{str(row.get('question', ''))}"
            )
            cases.append(
                Case(
                    case_id=str(
                        row.get(
                            "item_id",
                            f"rolebench-{variant}-{row.get('subset', 'general')}-{index}",
                        )
                    ),
                    input=prompt,
                    expected_output={"answer": expected},
                    metadata={
                        "variant": variant,
                        "subset": str(row.get("subset", "general")),
                        "role": str(row.get("role", "")),
                    },
                )
            )
    return _dataset(definition, cases)


def _materialize_codeforces(definition, *, row_loader: DatasetRowLoader) -> Dataset:
    rows = _load_rows(definition, row_loader, config_name=definition.dataset_revision)
    cases = []
    for index, row in enumerate(rows, start=1):
        if str(row.get("input_mode", "")).strip().lower() != "stdio":
            continue
        if row.get("interaction_format"):
            continue
        if not bool(row.get("official_tests_complete", False)):
            continue
        cases.append(
            Case(
                case_id=str(row.get("id", f"{definition.split}-{index}")),
                input=str(row.get("prompt", "")),
                expected_output=_json_dict(
                    {
                        "language": str(row.get("language", "python")),
                        "execution_mode": str(row.get("input_mode", "stdio")),
                        "official_tests": row.get("official_tests", []),
                        "time_limit": row.get("time_limit"),
                        "memory_limit": row.get("memory_limit"),
                        "generated_checker": row.get("generated_checker"),
                        "checker_language": row.get("checker_language"),
                    }
                ),
                metadata=_string_metadata(
                    row, ["contest_id", "language", "rating", "input_mode"]
                ),
            )
        )
    return _dataset(definition, cases)


def _materialize_aethercode(definition, *, row_loader: DatasetRowLoader) -> Dataset:
    rows = _load_rows(definition, row_loader, config_name=definition.dataset_revision)
    cases = []
    for index, row in enumerate(rows, start=1):
        tests = _normalize_tests(row.get("test_cases", row.get("official_tests")))
        if not tests:
            continue
        prompt = (
            "Write a C++17 program that solves the following problem. Return only code.\n\n"
            f"Problem:\n{str(row.get('description', row.get('prompt_text', '')))}"
        )
        cases.append(
            Case(
                case_id=str(row.get("id", f"{definition.split}-{index}")),
                input=prompt,
                expected_output=_json_dict(
                    {
                        "language": "cpp",
                        "execution_mode": "stdio",
                        "official_tests": tests,
                        "time_limit": row.get("time_limit"),
                        "generated_checker": row.get(
                            "checker", row.get("generated_checker")
                        ),
                        "checker_language": "cpp",
                    }
                ),
                metadata=_string_metadata(
                    row,
                    ["difficulty", "contest_category", "contest_name", "date", "year"],
                ),
            )
        )
    return _dataset(definition, cases)


def _materialize_livecodebench(definition, *, row_loader: DatasetRowLoader) -> Dataset:
    rows = _load_rows(
        definition,
        row_loader,
        revision=None
        if definition.source_kind == "huggingface_raw_files"
        else definition.dataset_revision,
        files=definition.source_files if definition.source_kind == "huggingface_raw_files" else None,
    )
    cases = []
    for index, row in enumerate(rows, start=1):
        tests = _normalize_tests(row.get("public_tests", row.get("official_tests")))
        if not tests:
            continue
        prompt = str(row.get("prompt", row.get("question_content", "")))
        cases.append(
            Case(
                case_id=str(
                    row.get(
                        "question_id", row.get("item_id", f"{definition.split}-{index}")
                    )
                ),
                input=prompt,
                expected_output=_json_dict(
                    {
                        "language": str(row.get("language", "python")),
                        "execution_mode": str(
                            row.get("execution_mode", row.get("input_mode", "stdio"))
                        ),
                        "official_tests": tests,
                        "function_name": row.get("function_name"),
                        "time_limit": row.get("time_limit"),
                    }
                ),
                metadata=_string_metadata(
                    row, ["platform", "contest_id", "difficulty", "contest_date"]
                ),
            )
        )
    return _dataset(definition, cases)


def _materialize_humaneval(definition, *, row_loader: DatasetRowLoader) -> Dataset:
    rows = _load_rows(
        definition,
        row_loader,
        revision=definition.dataset_revision,
        config_name=definition.variant,
    )
    cases = []
    score_variant = (
        "plus" if definition.base_benchmark_id == "humaneval_plus" else "base"
    )
    for index, row in enumerate(rows, start=1):
        prompt = str(row.get("prompt", "")).rstrip()
        canonical_solution = str(row.get("canonical_solution", "")).rstrip()
        entry_point = str(row.get("entry_point", ""))
        tests = _humaneval_tests(row.get("base_input", []))
        plus_tests = _humaneval_tests(row.get("plus_input", []))
        reference_solution = _humaneval_reference_solution(
            prompt=prompt, canonical_solution=canonical_solution
        )
        cases.append(
            Case(
                case_id=str(row.get("task_id", f"{definition.split}-{index}")),
                input=(
                    "Write a complete Python solution for the following task. "
                    "Return only Python code.\n\n"
                    f"{prompt}\n"
                ),
                expected_output=_json_dict(
                    {
                        "language": "python",
                        "execution_mode": "function",
                        "function_name": entry_point,
                        "official_tests": tests,
                        "plus_tests": plus_tests,
                        "reference_solution": reference_solution,
                        "solution": reference_solution,
                        "score_variant": score_variant,
                    }
                ),
                metadata={
                    "variant": definition.variant or "",
                    "entry_point": entry_point,
                },
            )
        )
    return _dataset(definition, cases)


def _dataset(definition, cases: list[Case]) -> Dataset:
    return Dataset(
        dataset_id=definition.dataset_id,
        revision=definition.split,
        metadata={
            "split": definition.split,
            "benchmark_id": definition.benchmark_id,
            "dataset_revision": definition.dataset_revision or "",
            "requires_code_execution": str(definition.requires_code_execution).lower(),
            "supported_execution_backends": ",".join(
                definition.supported_execution_backends
            ),
        },
        cases=cases,
    )


def _rolebench_source_file(definition, variant: str) -> str:
    return definition.source_file_map.get(variant, f"{variant}.jsonl")


def _normalize_options(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(option) for option in value]


def _mcq_prompt(*, question: str, options: list[str]) -> str:
    options_text = "\n".join(
        f"{'ABCDEFGHIJ'[index]}. {option}" for index, option in enumerate(options)
    )
    return (
        f"Question:\n{question}\n\nOptions:\n{options_text}\n\n"
        "Return the best option letter only."
    )


def _string_metadata(row: dict[str, object], keys: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            metadata[key] = ", ".join(str(item) for item in value)
            continue
        metadata[key] = str(value)
    return metadata


_GPQA_OPTION_RE = re.compile(r"^([a-z])\)\s*(.+)$")
_GPQA_MAPPING_RE = re.compile(r"^([A-Z])\.\s*([a-z])$")


def _parse_gpqa_diamond_question(question: str) -> tuple[str, list[str]]:
    lines = [line.strip() for line in question.splitlines() if line.strip()]
    lower_options: dict[str, str] = {}
    upper_mapping: dict[str, str] = {}
    question_lines: list[str] = []
    saw_option_block = False
    for line in lines:
        option_match = _GPQA_OPTION_RE.match(line)
        if option_match is not None:
            saw_option_block = True
            lower_options[option_match.group(1)] = option_match.group(2).strip()
            continue
        mapping_match = _GPQA_MAPPING_RE.match(line)
        if mapping_match is not None:
            upper_mapping[mapping_match.group(1)] = mapping_match.group(2)
            continue
        if not saw_option_block:
            question_lines.append(line)
    options = [
        lower_options[upper_mapping[label]]
        for label in ("A", "B", "C", "D")
        if label in upper_mapping and upper_mapping[label] in lower_options
    ]
    return "\n".join(question_lines).strip(), options


def _superchem_options(row: dict[str, object], *, language: str) -> list[str]:
    raw_options = row.get(f"options_{language}")
    if not isinstance(raw_options, dict):
        return []
    return [
        str(raw_options[key])
        for key in sorted(raw_options)
        if isinstance(raw_options.get(key), str)
    ]


def _superchem_answer(row: dict[str, object], *, language: str) -> str:
    raw_answer = row.get(f"answer_{language}")
    if isinstance(raw_answer, list) and raw_answer:
        return str(raw_answer[0]).strip().upper()
    return ""


def _prompt_messages_from_payload(row: dict[str, object]) -> list[dict[str, object]]:
    prompt = row.get("prompt")
    if not isinstance(prompt, list):
        return []
    messages: list[dict[str, object]] = []
    for entry in prompt:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = entry.get("content")
        if isinstance(role, str) and isinstance(content, str):
            messages.append({"role": role, "content": content})
    return messages


def _ideal_completion(row: dict[str, object]) -> str:
    payload = row.get("ideal_completions_data")
    if isinstance(payload, dict) and isinstance(payload.get("ideal_completion"), str):
        return str(payload["ideal_completion"])
    return ""


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def _normalize_tests(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    tests: list[dict[str, str]] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        raw_input = entry.get("input")
        raw_output = entry.get("output")
        if isinstance(raw_input, str) and isinstance(raw_output, str):
            tests.append({"input": raw_input, "output": raw_output})
    return tests


def _humaneval_tests(inputs: object) -> list[dict[str, str]]:
    if not isinstance(inputs, list):
        return []
    return [{"input": _json_dump(item)} for item in inputs]


def _humaneval_reference_solution(*, prompt: str, canonical_solution: str) -> str:
    return f"{prompt}\n{canonical_solution}\n".strip() + "\n"


def _json_dump(value: object) -> str:
    return json.dumps(_json(value), sort_keys=True)


def _json(value: object) -> JSONValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    return str(value)


def _json_dict(value: dict[str, object]) -> dict[str, JSONValue]:
    return {key: _json(item) for key, item in value.items()}
