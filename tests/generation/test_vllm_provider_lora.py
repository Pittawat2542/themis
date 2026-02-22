import sys
import types

from themis.core import entities as core_entities


def build_task_with_lora():
    prompt_spec = core_entities.PromptSpec(name="tmp", template="{prompt}")
    prompt_render = core_entities.PromptRender(
        spec=prompt_spec, text="Say hi", context={"prompt": "Say hi"}, metadata={}
    )
    model_spec = core_entities.ModelSpec(identifier="vllm-model", provider="vllm")
    sampling = core_entities.SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=16)

    # Task with LoRA metadata
    metadata = {
        "dataset_id": "sample-1",
        "lora_path": "/path/to/adapter",
        "lora_name": "test-adapter",
    }

    return core_entities.GenerationTask(
        prompt=prompt_render,
        model=model_spec,
        sampling=sampling,
        metadata=metadata,
    )


def fake_vllm_module_with_lora(monkeypatch):
    class FakeOutput:
        def __init__(self, text):
            self.outputs = [types.SimpleNamespace(text=text)]

    class FakeAsyncLLMEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            # Prepare to capture the last generate call arguments for verification
            self.last_generate_kwargs = {}

        def generate(self, **kwargs):
            self.last_generate_kwargs = kwargs

            async def _gen():
                yield FakeOutput(f"{kwargs['prompt']} :: chunk")

            return _gen()

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLoraRequest:
        def __init__(self, lora_name, lora_int_id, lora_path):
            self.lora_name = lora_name
            self.lora_int_id = lora_int_id
            self.lora_path = lora_path

    # Create the fake module structure
    module = types.SimpleNamespace(
        AsyncLLMEngine=FakeAsyncLLMEngine, SamplingParams=FakeSamplingParams
    )

    # Mock vllm.lora.request submodule
    lora_request_module = types.SimpleNamespace(LoraRequest=FakeLoraRequest)

    # Needs a bit more complex mocking for submodules if they are imported directly
    # But since the code does `from vllm.lora.request import LoraRequest`
    # We need to mock `vllm` and `vllm.lora.request` in `sys.modules`

    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setitem(
        sys.modules, "vllm.lora", types.SimpleNamespace(request=lora_request_module)
    )
    monkeypatch.setitem(sys.modules, "vllm.lora.request", lora_request_module)


def fake_torch_module(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 4

    module = types.SimpleNamespace(cuda=FakeCuda())
    monkeypatch.setitem(sys.modules, "torch", module)


def test_vllm_provider_lora_request(monkeypatch):
    fake_vllm_module_with_lora(monkeypatch)
    fake_torch_module(monkeypatch)

    from themis.generation.providers.vllm_provider import VLLMProvider

    # Initialize provider
    provider = VLLMProvider(model="demo", tensor_parallel_size=1)

    # Check if LoraRequest class was loaded
    assert hasattr(provider, "_lora_request_cls")

    # Create task with LoRA
    task = build_task_with_lora()

    # Generate
    record = provider.execute(task)

    # Verify result (basic check)
    assert "chunk" in record.output.text

    # Verify LoraRequest was passed to the engine
    engine = provider._engines[
        0
    ]  # Access the engine (assuming single engine or RR index 0 was used)
    # Note: Since _engines might have rotated, we might need to check which one was used.
    # But with 1 engine (implied by test), it should be fine.

    last_kwargs = engine.last_generate_kwargs
    assert "lora_request" in last_kwargs
    lora_req = last_kwargs["lora_request"]

    assert lora_req is not None
    assert lora_req.lora_name == "test-adapter"
    assert lora_req.lora_path == "/path/to/adapter"
    assert hasattr(lora_req, "lora_int_id")


def test_vllm_provider_no_lora_request(monkeypatch):
    fake_vllm_module_with_lora(monkeypatch)
    fake_torch_module(monkeypatch)

    from themis.generation.providers.vllm_provider import VLLMProvider

    provider = VLLMProvider(model="demo", tensor_parallel_size=1)

    # Create task WITHOUT LoRA
    prompt_spec = core_entities.PromptSpec(name="tmp", template="{prompt}")
    prompt_render = core_entities.PromptRender(
        spec=prompt_spec, text="Say hi", context={"prompt": "Say hi"}, metadata={}
    )
    model_spec = core_entities.ModelSpec(identifier="vllm-model", provider="vllm")
    sampling = core_entities.SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=16)

    task_no_lora = core_entities.GenerationTask(
        prompt=prompt_render,
        model=model_spec,
        sampling=sampling,
        metadata={"dataset_id": "sample-2"},  # No lora_path
    )

    provider.execute(task_no_lora)

    engine = provider._engines[0]
    last_kwargs = engine.last_generate_kwargs

    # Should not have lora_request in kwargs (or it should be None, depending on implementation
    # but our implementation creates dict conditionally)
    assert "lora_request" not in last_kwargs
