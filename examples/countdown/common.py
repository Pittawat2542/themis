from __future__ import annotations

import ast
import json
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import themis
from datasets import load_dataset
from themis.core import entities as core
from themis.evaluation.extractors import IdentityExtractor
from themis.evaluation.pipeline import EvaluationPipeline
from themis.interfaces import Metric as MetricInterface, ModelProvider

DEFAULT_API_BASE = "http://localhost:1234/api/v1/chat"
DEFAULT_MODEL = "localchat:qwen/qwen3-1.7b"
DEFAULT_STORAGE = ".cache/experiments"

BASELINE_PROMPT = (
    "Solve this Countdown puzzle.\n"
    "Numbers: {numbers_str}\n"
    "Target: {target}\n"
    "Return only one valid arithmetic expression using +, -, *, /."
)

CANDIDATE_PROMPT = (
    "You are solving Countdown.\n"
    "Numbers: {numbers_str}\n"
    "Target: {target}\n"
    "Rules: each source number can be used at most once; intermediate values must "
    "be positive integers.\n"
    "Output ONLY one arithmetic expression."
)


class LocalChatProvider(ModelProvider):
    def __init__(
        self,
        *,
        api_base: str = DEFAULT_API_BASE,
        timeout: float = 60.0,
        **_: Any,
    ) -> None:
        self.api_base = api_base
        self.timeout = timeout

    def generate(self, task: core.GenerationTask) -> core.GenerationRecord:
        payload = {
            "model": task.model.identifier,
            "system_prompt": "Return only one arithmetic expression.",
            "input": task.prompt.text,
        }
        req = urllib.request.Request(
            self.api_base,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))

            output_items = raw.get("output", [])
            message_text = ""
            for item in output_items:
                if isinstance(item, dict) and item.get("type") == "message":
                    message_text = str(item.get("content", ""))
                    break
            if not message_text:
                message_text = str(raw)

            stats = raw.get("stats") or {}
            usage = {
                "prompt_tokens": int(stats.get("input_tokens", 0) or 0),
                "completion_tokens": int(stats.get("total_output_tokens", 0) or 0),
                "total_tokens": int(stats.get("input_tokens", 0) or 0)
                + int(stats.get("total_output_tokens", 0) or 0),
            }
            return core.GenerationRecord(
                task=task,
                output=core.ModelOutput(text=message_text, raw=raw, usage=usage),
                error=None,
                metrics={},
            )
        except Exception as exc:
            return core.GenerationRecord(
                task=task,
                output=None,
                error=core.ModelError(message=str(exc), kind="provider_error"),
                metrics={},
            )


def load_countdown_for_themis(
    limit: int = 20,
    split: str = "train",
) -> list[dict[str, Any]]:
    ds = load_dataset("predibase/countdown", split=f"{split}[:{limit}]")
    rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(ds):
        numbers = [int(x) for x in raw["nums"]]
        target = int(raw["target"])
        rows.append(
            {
                "id": str(raw.get("id", idx)),
                "numbers": numbers,
                "numbers_str": ", ".join(str(n) for n in numbers),
                "target": target,
                "reference": {"numbers": numbers, "target": target},
            }
        )
    return rows


def _extract_expression(prediction: Any) -> str:
    text = str(prediction).strip()
    if not text:
        return ""

    match = re.search(r"expression\\s*:\\s*(.+)", text, flags=re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    text = text.replace("**", "").replace("`", "")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    expr_pattern = re.compile(r"^[0-9\\s\\+\\-\\*/\\(\\)]+$")
    for line in reversed(lines):
        if expr_pattern.match(line):
            candidate = line
            tokens = candidate.split()
            for end in range(len(tokens), 0, -1):
                trial = " ".join(tokens[:end]).strip()
                if not trial:
                    continue
                try:
                    ast.parse(trial, mode="eval")
                    return trial
                except SyntaxError:
                    continue
            return candidate

    if not lines:
        return ""

    candidate = re.sub(r"[^0-9\\+\\-\\*/\\(\\)\\s]", "", lines[-1]).strip()
    tokens = candidate.split()
    for end in range(len(tokens), 0, -1):
        trial = " ".join(tokens[:end]).strip()
        if not trial:
            continue
        try:
            ast.parse(trial, mode="eval")
            return trial
        except SyntaxError:
            continue
    return candidate


def _evaluate_countdown_expr(
    expr: str,
    source_numbers: Sequence[int],
) -> tuple[bool, int | None, str | None]:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        return False, None, f"syntax_error:{exc}"

    remaining = Counter(int(x) for x in source_numbers)

    def eval_node(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, int):
                raise ValueError("non_integer_constant")
            value = int(node.value)
            if value <= 0:
                raise ValueError("non_positive_constant")
            if remaining[value] <= 0:
                raise ValueError(f"number_not_available:{value}")
            remaining[value] -= 1
            return value
        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            if isinstance(node.op, ast.Add):
                out = left + right
            elif isinstance(node.op, ast.Sub):
                out = left - right
                if out <= 0:
                    raise ValueError("non_positive_intermediate")
            elif isinstance(node.op, ast.Mult):
                out = left * right
            elif isinstance(node.op, ast.Div):
                if right == 0:
                    raise ValueError("division_by_zero")
                if left % right != 0:
                    raise ValueError("non_integer_division")
                out = left // right
                if out <= 0:
                    raise ValueError("non_positive_intermediate")
            else:
                raise ValueError("operator_not_allowed")
            if out <= 0:
                raise ValueError("non_positive_intermediate")
            return out
        raise ValueError(f"node_not_allowed:{type(node).__name__}")

    try:
        value = eval_node(tree)
    except ValueError as exc:
        return False, None, str(exc)
    return True, value, None


@dataclass
class CountdownValidity(MetricInterface):
    name: str = "CountdownValidity"

    def compute(
        self,
        *,
        prediction: Any,
        references: Sequence[Any],
        metadata: dict[str, Any] | None = None,
    ) -> core.MetricScore:
        md = dict(metadata or {})
        if not references or not isinstance(references[0], dict):
            return core.MetricScore(
                metric_name=self.name,
                value=0.0,
                details={"error": "invalid_reference"},
                metadata=md,
            )

        ref = references[0]
        numbers = ref.get("numbers")
        target = ref.get("target")
        if not isinstance(numbers, list) or target is None:
            return core.MetricScore(
                metric_name=self.name,
                value=0.0,
                details={"error": "reference_missing_numbers_or_target"},
                metadata=md,
            )

        expr = _extract_expression(prediction)
        if not expr:
            return core.MetricScore(
                metric_name=self.name,
                value=0.0,
                details={"error": "empty_expression"},
                metadata=md,
            )

        ok, value, error = _evaluate_countdown_expr(expr, numbers)
        if not ok:
            return core.MetricScore(
                metric_name=self.name,
                value=0.0,
                details={"error": error, "expression": expr},
                metadata=md,
            )

        solved = float(value == int(target))
        details: dict[str, Any] = {
            "expression": expr,
            "computed_value": value,
            "target": int(target),
            "is_valid_expression": True,
        }
        if solved == 0.0:
            details["error"] = "target_mismatch"

        return core.MetricScore(
            metric_name=self.name,
            value=solved,
            details=details,
            metadata=md,
        )


def register_countdown_extensions() -> None:
    themis.register_provider("localchat", LocalChatProvider)
    themis.register_metric("countdown_validity", CountdownValidity)


def build_pipeline() -> EvaluationPipeline:
    return EvaluationPipeline(
        extractor=IdentityExtractor(),
        metrics=[CountdownValidity()],
    )
