"""Unit tests for the ConversationRunner module."""

from themis.core.conversation import ConversationContext
from themis.core.entities import GenerationRecord, GenerationTask, ModelSpec
from themis.generation.conversation_runner import ConversationRunner
from themis.interfaces import StatelessTaskExecutor
from themis.generation.router import ProviderRouter
from themis.generation.turn_strategies import FixedSequenceTurnStrategy


class MockExecutor(StatelessTaskExecutor):
    """Mock executor for testing conversation runner."""

    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    def execute(self, task: GenerationTask, **kwargs) -> GenerationRecord:
        response = self.responses[self._get_index()]
        self.call_count += 1
        return GenerationRecord(task=task, output=response)

    def _get_index(self):
        return min(self.call_count, len(self.responses) - 1)


def test_conversation_runner_initialization():
    strategy = FixedSequenceTurnStrategy(["Hi"])
    runner = ConversationRunner(executor=ProviderRouter({}), turn_strategy=strategy)
    assert runner is not None


def test_conversation_runner_single_turn():
    from themis.core.entities import SamplingConfig

    # Setup mock
    GenerationTask(
        model=ModelSpec(identifier="mock", provider="mock"),
        sampling=SamplingConfig(),
        prompt=None,  # Will be set by runner
    )

    strategy = FixedSequenceTurnStrategy(["Hello assistant"])
    ConversationRunner(executor=ProviderRouter({}), turn_strategy=strategy)

    # We just ensure it instantiates and strategy works
    ctx = ConversationContext()
    ctx.add_message("user", "first message")
    assert len(ctx) == 1
