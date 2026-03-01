"""Smoke tests for experimental and newly added modules.
These tests simply ensure that the modules can be imported and their core classes
instantiated without crashing, establishing baseline coverage.
"""

import pytest


def test_core_conversation_smoke():
    from themis.core.conversation import ConversationContext, Message

    msg = Message(role="user", content="hello")
    ctx = ConversationContext(messages=[msg])
    assert len(ctx) == 1
    assert ctx.to_prompt() == "user: hello"


def test_core_tools_smoke():
    from themis.core.tools import ToolDefinition, ToolRegistry

    def mock_handler(args):
        return True

    tool = ToolDefinition(
        name="test", description="test", parameters={}, handler=mock_handler
    )
    registry = ToolRegistry()
    registry.register(tool)
    assert registry.get("test") is not None


def test_core_types_smoke():
    from themis.core.types import validate_type

    assert callable(validate_type)


def test_evaluation_conditional_smoke():
    from themis.evaluation.conditional import ConditionalMetric, select_by_difficulty
    from themis.evaluation.metrics.exact_match import ExactMatch

    metric = ConditionalMetric(metric=ExactMatch(), condition=lambda x: True)
    assert metric.name is not None

    selector = select_by_difficulty([], [], [])
    assert callable(selector)


def test_experiment_share_smoke():
    from themis.experiment.share import create_share_pack

    assert callable(create_share_pack)


def test_generation_agentic_runner_smoke():
    from themis.generation.agentic_runner import AgenticRunner

    # Just import check here; deeper tests in test_agentic_runner.py
    assert AgenticRunner is not None


def test_generation_conversation_runner_smoke():
    from themis.generation.conversation_runner import ConversationRunner

    assert ConversationRunner is not None


def test_generation_turn_strategies_smoke():
    from themis.generation.turn_strategies import FixedSequenceTurnStrategy

    strategy = FixedSequenceTurnStrategy(messages=["test"])
    assert strategy is not None


def test_cli_main_smoke():
    from themis.cli.__main__ import app

    assert app is not None


def test_integrations_huggingface_smoke():
    pytest.importorskip("huggingface_hub")
    from themis.integrations.huggingface import HuggingFaceHubUploader

    assert HuggingFaceHubUploader is not None


def test_integrations_wandb_smoke():
    pytest.importorskip("wandb")
    from themis.integrations.wandb import WandbTracker

    assert WandbTracker is not None
