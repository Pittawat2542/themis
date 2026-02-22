import sys
import types


from themis.core import entities as core_entities


def build_task():
    prompt_spec = core_entities.PromptSpec(name="tmp", template="{prompt}")
    prompt_render = core_entities.PromptRender(
        spec=prompt_spec, text="Say hi", context={"prompt": "Say hi"}, metadata={}
    )
    model_spec = core_entities.ModelSpec(identifier="vllm-model", provider="vllm")
    sampling = core_entities.SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=16)
    return core_entities.GenerationTask(
        prompt=prompt_render,
        model=model_spec,
        sampling=sampling,
        metadata={"dataset_id": "sample-1"},
    )


def fake_vllm_module(monkeypatch):
    class FakeOutput:
        def __init__(self, text):
            self.outputs = [types.SimpleNamespace(text=text)]

    class FakeAsyncLLMEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self, *, prompt, sampling_params, request_id):
            async def _gen():
                yield FakeOutput(f"{prompt} :: chunk")

            return _gen()

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module = types.SimpleNamespace(
        AsyncLLMEngine=FakeAsyncLLMEngine, SamplingParams=FakeSamplingParams
    )
    monkeypatch.setitem(sys.modules, "vllm", module)

    # Mock vllm.lora.request
    lora_request_module = types.SimpleNamespace(
        LoraRequest=lambda *args, **kwargs: None
    )
    lora_module = types.SimpleNamespace(request=lora_request_module)
    monkeypatch.setitem(sys.modules, "vllm.lora", lora_module)
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


def test_vllm_provider_multiple_engines(monkeypatch):
    fake_vllm_module(monkeypatch)
    fake_torch_module(monkeypatch)

    from themis.generation.providers.vllm_provider import VLLMProvider

    provider = VLLMProvider(model="demo", tensor_parallel_size=2)

    assert len(provider._engines) == 2

    record = provider.execute(build_task())
    assert "chunk" in record.output.text
