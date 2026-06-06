"""Small reusable catalog components for evaluation kit composition."""

from __future__ import annotations

from dataclasses import dataclass, field

from themis.core.base import JSONValue


class LowercaseTransform:
    component_id = "builtin/lowercase_transform"
    version = "1.0"

    def fingerprint(self) -> str:
        return "builtin-lowercase-transform-fingerprint"

    def apply(self, value: JSONValue) -> JSONValue:
        if isinstance(value, str):
            return value.lower()
        if isinstance(value, list):
            return [self.apply(item) for item in value]
        if isinstance(value, dict):
            return {key: self.apply(item) for key, item in value.items()}
        return value


@dataclass(frozen=True)
class DefaultBaselinePack:
    component_id: str = "builtin/default_baseline_pack"
    version: str = "1.0"
    component_ids: list[str] = field(
        default_factory=lambda: [
            "builtin/demo_generator",
            "builtin/json_identity",
            "builtin/exact_match",
            "builtin/f1",
        ]
    )

    def fingerprint(self) -> str:
        return "builtin-default-baseline-pack-fingerprint"


class ParserAblationTemplate:
    component_id = "builtin/parser_ablation_template"
    version = "1.0"

    def fingerprint(self) -> str:
        return "builtin-parser-ablation-template-fingerprint"

    def variants(self, parser_id: str) -> list[str]:
        return [f"{parser_id}::disabled", f"{parser_id}::fallback_only"]


@dataclass(frozen=True)
class TokenBudgetSweep:
    component_id: str = "builtin/token_budget_sweep"
    version: str = "1.0"
    budgets: list[int] = field(default_factory=lambda: [128, 512, 2048])

    def fingerprint(self) -> str:
        return "builtin-token-budget-sweep-fingerprint"
