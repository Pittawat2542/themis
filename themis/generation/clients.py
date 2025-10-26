"""Model provider implementations used for experiments."""

from __future__ import annotations

import json
import math
import random
import re
from typing import Tuple

from themis.core import entities as core_entities
from themis.interfaces import ModelProvider
from themis.providers import register_provider


class FakeMathModelClient(ModelProvider):
    """A lightweight heuristic provider used for math experiments."""

    _POINT_PATTERN = re.compile(
        r"point\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", re.IGNORECASE
    )
    _ARITHMETIC_PATTERN = re.compile(
        r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)"
    )

    def __init__(
        self, *, seed: int | None = None, default_answer: str = "unknown"
    ) -> None:
        self._rng = random.Random(seed)
        self._default_answer = default_answer

    def generate(
        self, task: core_entities.GenerationTask
    ) -> core_entities.GenerationRecord:  # type: ignore[override]
        prompt_text = task.prompt.text
        answer, reason = self._solve(prompt_text)
        expect_boxed = bool(task.metadata.get("template_expect_boxed"))
        if expect_boxed and "\\boxed{" not in answer:
            answer = f"\\boxed{{{answer}}}"
        payload = {
            "answer": answer,
            "reasoning": reason,
            "model": task.model.identifier,
        }
        latency = self._rng.randint(8, 18)
        return core_entities.GenerationRecord(
            task=task,
            output=core_entities.ModelOutput(text=json.dumps(payload), raw=payload),
            error=None,
            metrics={"latency_ms": latency},
        )

    def _solve(self, prompt: str) -> Tuple[str, str]:
        prompt_lower = prompt.lower()
        polar = self._solve_polar_coordinates(prompt_lower)
        if polar is not None:
            return polar

        arithmetic = self._solve_arithmetic(prompt_lower)
        if arithmetic is not None:
            return arithmetic

        return self._default_answer, "Unable to derive answer with heuristic solver."

    def _solve_polar_coordinates(self, prompt_lower: str) -> Tuple[str, str] | None:
        if "polar" not in prompt_lower:
            return None
        match = self._POINT_PATTERN.search(prompt_lower)
        if not match:
            return None
        x = int(match.group(1))
        y = int(match.group(2))
        radius_squared = x * x + y * y
        radius = math.sqrt(radius_squared)
        if math.isclose(radius, round(radius)):
            radius_str = str(int(round(radius)))
        else:
            radius_str = f"\\sqrt{{{radius_squared}}}"
        theta = math.atan2(y, x)
        theta_str = self._format_theta(theta)
        answer = f"\\left( {radius_str}, {theta_str} \\right)"
        reasoning = f"Converted rectangular point ({x}, {y}) into polar coordinates."
        return answer, reasoning

    def _format_theta(self, theta: float) -> str:
        tau = 2 * math.pi
        theta = theta % tau
        multiples = {
            0: "0",
            math.pi / 6: "\\frac{\\pi}{6}",
            math.pi / 4: "\\frac{\\pi}{4}",
            math.pi / 3: "\\frac{\\pi}{3}",
            math.pi / 2: "\\frac{\\pi}{2}",
            math.pi: "\\pi",
            3 * math.pi / 2: "\\frac{3\\pi}{2}",
        }
        for value, label in multiples.items():
            if math.isclose(theta, value, abs_tol=1e-6):
                return label
        if math.isclose(theta, 5 * math.pi / 6, abs_tol=1e-6):
            return "\\frac{5\\pi}{6}"
        if math.isclose(theta, 7 * math.pi / 6, abs_tol=1e-6):
            return "\\frac{7\\pi}{6}"
        if math.isclose(theta, 4 * math.pi / 3, abs_tol=1e-6):
            return "\\frac{4\\pi}{3}"
        return f"{theta:.3f}"

    def _solve_arithmetic(self, prompt_lower: str) -> Tuple[str, str] | None:
        if "what is" not in prompt_lower and "compute" not in prompt_lower:
            return None
        match = self._ARITHMETIC_PATTERN.search(prompt_lower)
        if not match:
            return None
        left = float(match.group(1))
        op = match.group(2)
        right = float(match.group(3))
        if op == "+":
            result = left + right
        elif op == "-":
            result = left - right
        elif op == "*":
            result = left * right
        elif op == "/":
            if right == 0:
                return "undefined", "Division by zero encountered."
            result = left / right
        else:
            return None
        if result.is_integer():
            answer = str(int(result))
        else:
            answer = f"{result:.3f}"
        reasoning = f"Evaluated {left} {op} {right} using arithmetic solver."
        return answer, reasoning

    def count_tokens(self, text: str) -> int:
        return len(text.split())


__all__ = ["FakeMathModelClient"]


register_provider("fake", FakeMathModelClient)
