# Agent Evals And Tools

Use this reference when the user is building an agent-style benchmark instead
of a simple single-turn evaluation.

## Use This Pattern When

- the prompt starts with a bootstrap message sequence
- `system` or `developer` messages matter
- the benchmark scripts follow-up turns
- the engine needs an explicit tool list
- the engine returns tool call or tool result traces

Stay with the normal benchmark flow when there is one rendered prompt and no
tools.

## Authoring Flow

1. Put shared tool definitions on `ProjectSpec.tools`.
2. Define `BenchmarkSpec` as usual.
3. Use `PromptVariantSpec.messages` for the bootstrap sequence.
4. Use `PromptVariantSpec.follow_up_turns` for scripted continuation.
5. Add benchmark-local tool definitions on `BenchmarkSpec.tools` when a tool is
   benchmark-specific or should override a project tool by id.
6. Use `SliceSpec.tool_ids` to select the tools exposed on each trial.
7. Register matching opaque handlers with `PluginRegistry.register_tool(...)`.
8. Build the orchestrator and run the benchmark normally.

## Merge Semantics

- project tools are the base
- benchmark tools override same-id project tools
- slice selection is explicit
- if `tool_ids` is empty, the trial receives no tools

Do not attach tools to prompt variants in this workflow.

## Engine Contract

Benchmark-native engines receive:

- `trial.prompt.messages`
- `trial.prompt.follow_up_turns`
- `trial.tools`
- `runtime.tool_handlers`

Use that boundary directly. Do not re-render benchmark prompts inside the
engine.

Themis does not:

- implement the agent loop
- execute tools
- define a standard tool invoke interface

The engine still owns provider calls, agent control flow, tool execution, and
deciding when to return the terminal assistant answer.

## Canonical Example

Point users to `examples/10_agent_eval.py` when they ask for:

- bootstrap `system` and `developer` prompts
- follow-up turns
- tool declaration and selection
- benchmark override of a project tool
- access to `trial.tools` and `runtime.tool_handlers`
- returned tool-call and tool-result traces

## Out Of Scope

Do not send users to retired experiment/task APIs for this flow.

Do not describe Themis as an agent runtime. It is the orchestration and
benchmark layer around the engine.
