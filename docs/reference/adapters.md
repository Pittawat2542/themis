---
title: Adapters reference
diataxis: reference
audience: users configuring provider-backed generators
goal: Document the generator adapter entry points and their intended use.
---

# Adapters reference

## Available adapters

| Name | Kind | Use when | Key constraints / notes |
| --- | --- | --- | --- |
| `themis.adapters.openai(...)` | Provider adapter | An OpenAI or OpenAI-compatible endpoint should handle generation while Themis owns evaluation and storage | Install `themis-eval[openai]` or inject a compatible client |
| `themis.adapters.vllm(...)` | Provider adapter | A vLLM or other OpenAI-compatible local endpoint should handle generation | Install `themis-eval[vllm]` on Linux or inject a compatible client |
| `themis.adapters.langgraph(...)` | Graph adapter | A LangGraph workflow should act as the generator | Pass a graph with `invoke()` or `ainvoke()` |
| `themis.adapters.anthropic(...)` | Provider adapter | Claude generation or judging should feed Themis telemetry | Inject an Anthropic-compatible client |
| `themis.adapters.bedrock(...)` | Provider adapter | AWS Bedrock Converse generation or judging should feed Themis telemetry | Inject a Bedrock Runtime-compatible client |
| `themis.adapters.gemini(...)` | Provider adapter | Gemini generation or judging should feed Themis telemetry | Inject a Google Gen AI-compatible client |
| `themis.adapters.azure_openai(...)` | Provider adapter | Azure-hosted OpenAI generation or judging should feed Themis telemetry | Inject an Azure OpenAI-compatible client |
| `themis.adapters.ollama(...)` | Provider adapter | A local Ollama endpoint should provide generation or judging | Inject an Ollama-compatible client |
| `themis.adapters.litellm(...)` | Provider adapter | LiteLLM should route generation or judging across providers | Inject the LiteLLM module or a compatible client |

Use adapters when Themis should still own planning, reduction, parsing, scoring, storage, and inspection, but a provider or graph runtime should perform generation.

## Provider notes

| Option | Best for | Persistence / runtime behavior | Caveats |
| --- | --- | --- | --- |
| OpenAI | Managed or self-hosted OpenAI-compatible endpoints | Themis still owns planning, storage, parsing, and scoring | Requires credentials or an injected client |
| vLLM | Local or self-hosted inference servers that speak an OpenAI-compatible API | Behaves like another provider endpoint inside Themis runtime controls | The packaged extra is Linux-only |
| LangGraph | Existing graph-driven generation workflows | Themis treats the graph as the generator while preserving the rest of the runtime | Best trace capture depends on `astream_events()` availability |
| Anthropic, Bedrock, Gemini, Azure OpenAI, Ollama, and LiteLLM | Provider-specific generation or judging with shared telemetry | Themis normalizes provider responses, retries, usage, and request metadata | These constructors require injected compatible clients; they do not install provider SDKs automatically |
