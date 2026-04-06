from __future__ import annotations

import pytest
from themis.catalog.loaders import BenchmarkSourceRequest


@pytest.fixture(autouse=True)
def _install_catalog_fixture_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    from themis.catalog.benchmarks import materializers

    def request_key(
        source_kind: str,
        dataset_id: str,
        split: str,
        revision: str | None = None,
        config_name: str | None = None,
        files: tuple[str, ...] = (),
    ) -> tuple[str, str, str, str | None, str | None, tuple[str, ...]]:
        return (source_kind, dataset_id, split, revision, config_name, files)

    rows_by_request: dict[
        tuple[str, str, str, str | None, str | None, tuple[str, ...]],
        list[dict[str, object]],
    ] = {
        request_key("huggingface_dataset", "MathArena/aime_2025", "train"): [
            {
                "problem_idx": 2,
                "problem": "Compute 3 + 3.",
                "answer": "6",
                "problem_type": ["Algebra"],
                "source": "fixture",
            }
        ],
        request_key("huggingface_dataset", "MathArena/aime_2026", "train"): [
            {
                "problem_idx": 1,
                "problem": "What is 2 + 2?",
                "answer": "4",
                "problem_type": ["Algebra"],
                "source": "fixture",
            }
        ],
        request_key("huggingface_dataset", "MathArena/apex_2025", "train"): [
            {
                "problem_idx": 5,
                "problem": "Find the value of 9 + 10.",
                "answer": "19",
                "source": "fixture",
            }
        ],
        request_key("huggingface_dataset", "ByteDance-Seed/BeyondAIME", "test"): [
            {
                "item_id": "beyond-aime-1",
                "problem": "What is 7 times 6?",
                "answer": "42",
                "source": "fixture",
            }
        ],
        request_key("huggingface_dataset", "MathArena/hmmt_feb_2025", "train"): [
            {
                "problem_idx": 3,
                "problem": "What is 5 + 7?",
                "answer": "12",
                "problem_type": ["Number Theory"],
                "source": "fixture",
            }
        ],
        request_key("huggingface_dataset", "MathArena/hmmt_nov_2025", "train"): [
            {
                "problem_idx": 4,
                "problem": "Evaluate 8 - 3.",
                "answer": "5",
                "problem_type": ["Algebra"],
                "source": "fixture",
            }
        ],
        request_key("huggingface_dataset", "Eureka-Lab/PHYBench", "train"): [
            {
                "id": 1,
                "tag": "MECHANICS",
                "content": "Find the acceleration.",
                "answer": "9.8",
            }
        ],
        request_key("huggingface_dataset", "Hwilner/imo-answerbench", "train"): [
            {
                "Problem ID": "imo-1",
                "Problem": "What is 10 + 10?",
                "Short Answer": "20",
                "Category": "algebra",
                "Subcategory": "sum",
                "Source": "fixture",
            }
        ],
        request_key("huggingface_dataset", "TIGER-Lab/MMLU-Pro", "test"): [
            {
                "item_id": "mmlu-pro-1",
                "question": "Which planet is known as the Red Planet?",
                "options": ["Venus", "Mars", "Jupiter", "Mercury"],
                "answer": "B",
                "category": "astronomy",
                "src": "fixture",
            }
        ],
        request_key("huggingface_dataset", "m-a-p/SuperGPQA", "train"): [
            {
                "item_id": "supergpqa-1",
                "question": "Which gas is most abundant in Earth's atmosphere?",
                "options": ["Oxygen", "Hydrogen", "Nitrogen", "Carbon dioxide"],
                "answer_letter": "C",
                "discipline": "science",
                "field": "chemistry",
                "subfield": "atmospheric chemistry",
                "difficulty": "medium",
            }
        ],
        request_key("huggingface_dataset", "m-a-p/Encyclo-K", "test"): [
            {
                "item_id": "encyclo-k-1",
                "question": "What is the capital city of Canada?",
                "options": ["Toronto", "Vancouver", "Ottawa", "Montreal"],
                "answer_letter": "C",
                "discipline": "humanities",
                "field": "geography",
                "subfield": "capitals",
                "difficulty": "easy",
            }
        ],
        request_key("huggingface_dataset", "mediabiasgroup/BABE", "test"): [
            {
                "uuid": "babe-1",
                "text": "This article lead is written in an opinionated voice.",
                "label": 1,
                "outlet": "Example Outlet",
                "topic": "media",
                "type": "left",
                "label_opinion": "Expresses writer's opinion",
            }
        ],
        request_key("huggingface_dataset", "fingertap/GPQA-Diamond", "test"): [
            {
                "question": "Which option is correct?\n\na) alpha\nb) beta\nc) gamma\nd) delta\n\nA. d\nB. a\nC. b\nD. c",
                "answer": "D",
            }
        ],
        request_key("huggingface_dataset", "openai/MMMLU", "test", None, "default"): [
            {
                "Unnamed: 0": 0,
                "Question": "What is 2 + 2?",
                "A": "1",
                "B": "4",
                "C": "3",
                "D": "5",
                "Answer": "B",
                "Subject": "math",
            }
        ],
        request_key("huggingface_dataset", "openai/MMMLU", "test", None, "ZH_CN"): [
            {
                "Unnamed: 0": 1,
                "Question": "中国的首都是哪里？",
                "A": "上海",
                "B": "北京",
                "C": "广州",
                "D": "深圳",
                "Answer": "B",
                "Subject": "geography",
            }
        ],
        request_key(
            "huggingface_raw_files",
            "ZehuaZhao/SUPERChem",
            "train",
            None,
            None,
            ("SUPERChem-500.parquet",),
        ): [
            {
                "uuid": "chem-1",
                "field": "chemistry",
                "question_type": "multiple_choice",
                "question_en": "What is shown?",
                "question_zh": "图中显示了什么？",
                "question_images": ["https://example.test/chem-1.png"],
                "options_en": {"A": "Alpha", "B": "Beta"},
                "options_zh": {"A": "甲", "B": "乙"},
                "answer_en": ["B"],
                "answer_zh": ["B"],
            }
        ],
        request_key("huggingface_dataset", "openai/frontierscience", "test"): [
            {
                "item_id": "frontierscience-1",
                "problem": "Derive the requested expression.",
                "answer": "Points: 1.0, Item: derive the expression correctly.",
                "subject": "physics",
                "task_group_id": "group-1",
            }
        ],
        request_key(
            "huggingface_raw_files",
            "openai/healthbench",
            "test",
            None,
            None,
            ("2025-05-07-06-14-12_oss_eval.jsonl",),
        ): [
            {
                "prompt_id": "healthbench-1",
                "prompt": [
                    {
                        "role": "user",
                        "content": "I think I may have postpartum depression. What should I do right now?",
                    }
                ],
                "rubrics": [
                    {
                        "criterion": "Encourages immediate professional support for urgent symptoms.",
                        "points": 2,
                        "tags": ["theme:communication", "axis:safety"],
                    }
                ],
                "example_tags": ["theme:communication", "axis:safety"],
                "ideal_completions_data": {
                    "ideal_completion": "Reach out to a healthcare professional or emergency support if you feel unsafe."
                },
            }
        ],
        request_key("huggingface_dataset", "m-a-p/LPFQA", "train"): [
            {
                "prompt_id": "lpfqa-1",
                "prompt": "Translate the phrase 'good morning' into Japanese.",
                "response_reference": "<参考答案>: おはようございます\n<评估要点>: The answer should convey a polite morning greeting.",
                "judge_prompt_template": "Reference:\n{response_reference}\n\nResponse:\n{response}",
                "judge_system_prompt": "You are a careful grading assistant.",
                "primary_domain": "translation",
            }
        ],
        request_key("huggingface_dataset", "google/simpleqa-verified", "eval"): [
            {
                "original_index": 1,
                "problem": "What is the chemical symbol for gold?",
                "answer": "Au",
                "topic": "chemistry",
                "answer_type": "short_text",
                "multi_step": False,
                "requires_reasoning": False,
            }
        ],
        request_key("huggingface_dataset", "cais/hle", "test"): [
            {
                "id": "hle-1",
                "question": "What is 12 multiplied by 12?",
                "answer": "144",
                "image": "",
            }
        ],
        request_key(
            "huggingface_raw_files",
            "ifujisawa/procbench",
            "train",
            None,
            None,
            ("combined_dataset.jsonl",),
        ): [
            {
                "problem_name": "task01_0000",
                "prompt": "Sort the string.",
                "task_name": "task01",
                "label": {"final": "abct", "intermediate": ["bac"]},
            },
            {
                "problem_name": "task03_0000",
                "prompt": "Reverse the string.",
                "task_name": "task03",
                "label": {"final": "cba"},
            },
            {
                "problem_name": "task07_0000",
                "prompt": "Complete task 07.",
                "task_name": "task07",
                "label": {"final": "done"},
            },
        ],
        request_key(
            "huggingface_raw_files",
            "ZenMoore/RoleBench",
            "test",
            None,
            None,
            ("rolebench-eng/instruction-generalization/general/test.jsonl",),
        ): [
            {
                "role": "Wizard",
                "question": "What would you say to a lost traveler?",
                "generated": ["Follow the silver river until dawn."],
            }
        ],
        request_key(
            "huggingface_raw_files",
            "ZenMoore/RoleBench",
            "test",
            None,
            None,
            ("rolebench-eng/instruction-generalization/role_specific/test.jsonl",),
        ): [
            {
                "role": "Archivist",
                "question": "How should I preserve a fragile manuscript?",
                "generated": ["Store it in a stable, dry, low-light archive."],
                "type": ["script_based"],
            }
        ],
        request_key(
            "huggingface_raw_files",
            "ZenMoore/RoleBench",
            "test",
            None,
            None,
            ("rolebench-eng/role-generalization/general/test.jsonl",),
        ): [
            {
                "role": "Historian",
                "question": "Why do archives matter?",
                "generated": ["They preserve evidence and memory."],
            }
        ],
        request_key(
            "huggingface_raw_files",
            "ZenMoore/RoleBench",
            "test",
            None,
            None,
            ("rolebench-eng/role-generalization/role_specific/test.jsonl",),
        ): [
            {
                "role": "Captain",
                "question": "How do you stay calm in a storm?",
                "generated": ["By trusting the crew and reading the seas."],
                "type": "script_based",
            }
        ],
        request_key(
            "huggingface_dataset",
            "open-r1/codeforces",
            "test",
            None,
            "verifiable-prompts",
        ): [
            {
                "id": "1A",
                "contest_id": "1",
                "title": "Theatre Square",
                "prompt": "Write a Python program that solves the problem.",
                "input_mode": "stdio",
                "official_tests_complete": True,
                "official_tests": [{"input": "6 6 4\n", "output": "4\n"}],
                "language": "python",
                "rating": 1000,
                "time_limit": 1.0,
                "memory_limit": 256.0,
            }
        ],
        request_key(
            "huggingface_dataset", "m-a-p/AetherCode", "test", None, "v1_2024"
        ): [
            {
                "id": "60173",
                "description": "Compute the answer.",
                "test_cases": [{"input": "1\n", "output": "2\n"}],
                "time_limit": 1000,
                "difficulty": "Easy",
                "contest_category": "ICPC",
                "contest_name": "Fixture Contest",
                "date": "2024/11/17",
                "year": 2024,
            }
        ],
        request_key(
            "huggingface_raw_files",
            "livecodebench/code_generation_lite",
            "test",
            None,
            None,
            ("test6.jsonl",),
        ): [
            {
                "question_id": "lcb-1",
                "prompt": "Write a Python function that returns 4.",
                "public_tests": [{"input": "", "output": "4\n"}],
                "language": "python",
                "execution_mode": "stdio",
                "platform": "leetcode",
                "contest_id": "abc387",
                "difficulty": "easy",
                "contest_date": "2025-01-04T00:00:00",
            }
        ],
        request_key("huggingface_dataset", "evalplus/humanevalplus", "test"): [
            {
                "task_id": "HumanEval/0",
                "entry_point": "add",
                "prompt": 'def add(a: int, b: int) -> int:\n    """Return the sum of two integers."""\n',
                "canonical_solution": "    return a + b\n",
                "test": (
                    "def check(candidate):\n"
                    "    assert candidate(1, 2) == 3\n"
                    "    assert candidate(4, 5) == 9\n"
                ),
            }
        ],
    }

    def fake_loader(request: BenchmarkSourceRequest) -> list[dict[str, object]]:
        key = request_key(
            request.source_kind,
            request.dataset_id,
            request.split,
            request.revision,
            request.config_name,
            tuple(request.files),
        )
        try:
            return [dict(row) for row in rows_by_request[key]]
        except KeyError:
            raise AssertionError(
                f"Unexpected benchmark source request: {request.model_dump()}"
            ) from None

    monkeypatch.setattr(materializers, "_default_loader", fake_loader)


@pytest.fixture(autouse=True)
def _mock_code_executors(monkeypatch: pytest.MonkeyPatch) -> None:
    from themis.catalog.builtins.code_execution import (
        PistonSandboxExecutor,
        SandboxExecutionResult,
        SandboxFusionExecutor,
    )

    def mock_execute(*args: object, **kwargs: object) -> SandboxExecutionResult:
        return SandboxExecutionResult(
            stdout="mocked",
            stderr="",
            return_code=0,
            status="ok",
        )

    monkeypatch.setattr(PistonSandboxExecutor, "execute", mock_execute)
    monkeypatch.setattr(SandboxFusionExecutor, "execute", mock_execute)
