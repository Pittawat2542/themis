"""Manifest-backed runtime component resolvers."""

from __future__ import annotations

from typing import cast

from themis.catalog.registry import load_component
from themis.core.protocols import (
    CandidateReducer,
    CandidateSelector,
    Generator,
    JudgeModel,
    LLMMetric,
    Parser,
    PureMetric,
    SelectionMetric,
    TraceMetric,
)

BuiltinMetric = PureMetric | LLMMetric | SelectionMetric | TraceMetric


def _resolve(value: object, *, kind: str) -> object:
    if isinstance(value, str):
        return load_component(value, kind=kind)
    return value


def resolve_generator_component(value: object) -> Generator:
    return cast(Generator, _resolve(value, kind="generator"))


def resolve_selector_component(value: object) -> CandidateSelector:
    return cast(CandidateSelector, _resolve(value, kind="selector"))


def resolve_reducer_component(value: object) -> CandidateReducer:
    return cast(CandidateReducer, _resolve(value, kind="reducer"))


def resolve_parser_component(value: object) -> Parser:
    return cast(Parser, _resolve(value, kind="parser"))


def resolve_metric_component(value: object) -> BuiltinMetric:
    return cast(BuiltinMetric, _resolve(value, kind="metric"))


def resolve_judge_model_component(value: object) -> JudgeModel:
    return cast(JudgeModel, _resolve(value, kind="judge_model"))
