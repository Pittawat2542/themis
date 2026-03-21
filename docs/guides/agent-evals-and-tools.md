# Author Agent Evaluations and Tools

Use this guide when your benchmark is no longer just one rendered user prompt
plus one terminal answer.

Typical cases:

- the engine runs an internal ReAct or agent loop
- the prompt needs bootstrap `system` and `developer` messages
- the benchmark scripts one or more follow-up turns
- the engine needs an explicit tool list and matching runtime handlers

For the engine-only contract, see [Build a Provider Engine](provider-engines.md).
For the runnable reference implementation, see [Example Catalog](examples.md)
and run `examples/10_agent_eval.py`.

## When To Use This Pattern

Stay with the normal benchmark flow when:

- one rendered prompt is enough
- there are no scripted follow-up turns
- the engine does not need tools

Use the agent pattern when:

- prompt setup is a bootstrap message sequence, not just one user turn
- the benchmark needs scripted multi-turn continuation
- the engine owns its own tool loop or multi-step execution
- you want tool traces preserved in the returned conversation

## Author Bootstrap Messages and Follow-Up Turns

`PromptVariantSpec.messages` is the bootstrap message sequence. It can include
`system`, `developer`, and `user` messages.

`PromptVariantSpec.follow_up_turns` is the ordered scripted continuation. Each
turn is a `PromptTurnSpec` with one or more messages.

```python
PromptVariantSpec(
    id="agent-default",
    family="agent",
    messages=[
        PromptMessage(role=PromptRole.SYSTEM, content="You are a careful agent."),
        PromptMessage(
            role=PromptRole.DEVELOPER,
            content="Use tools before you answer.",
        ),
        PromptMessage(
            role=PromptRole.USER,
            content="Solve: {item.question}",
        ),
    ],
    follow_up_turns=[
        PromptTurnSpec(
            messages=[
                PromptMessage(
                    role=PromptRole.USER,
                    content="Double check the answer for {item.question}.",
                )
            ]
        )
    ],
)
```

For benchmark-native runs, Themis renders both the bootstrap messages and every
follow-up turn before the engine is called.

## Declare and Select Tools

Use `ToolSpec` for the serializable tool definition seen by the engine.

- `ProjectSpec.tools` defines reusable base tools
- `BenchmarkSpec.tools` adds or overrides benchmark-local tools by `id`
- `SliceSpec.tool_ids` selects the exact tools exposed on each trial

```python
project = ProjectSpec(
    ...,
    tools=[
        ToolSpec(
            id="calculator",
            description="Base arithmetic tool.",
            input_schema={"type": "object"},
        )
    ],
)

benchmark = BenchmarkSpec(
    ...,
    tools=[
        ToolSpec(
            id="calculator",
            description="Benchmark-specific arithmetic tool.",
            input_schema={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        )
    ],
    slices=[
        SliceSpec(
            ...,
            tool_ids=["calculator"],
        )
    ],
)
```

The merge model is:

- project tools are the base
- benchmark tools override same-id project tools
- slice selection is explicit; if `tool_ids` is empty, no tools are exposed

## What The Engine Receives

At inference time, benchmark-native engines receive:

- `trial.prompt.messages`: rendered bootstrap messages
- `trial.prompt.follow_up_turns`: rendered scripted continuation
- `trial.tools`: selected serializable tool specs for that trial
- `runtime.tool_handlers`: matching opaque runtime handlers keyed by tool id

```python
class MyAgentEngine:
    def infer(self, trial, context, runtime):
        bootstrap = [message.model_dump(mode="json") for message in trial.prompt.messages]
        follow_up_turns = [
            [message.model_dump(mode="json") for message in turn.messages]
            for turn in trial.prompt.follow_up_turns
        ]
        tools = [tool.model_dump(mode="json") for tool in trial.tools]
        handlers = runtime.tool_handlers
        del bootstrap, follow_up_turns, tools, handlers, context
        ...
```

Register opaque runtime handlers on the `PluginRegistry`:

```python
registry.register_tool("calculator", my_calculator_handler)
```

Only handlers for the selected trial tools are injected into
`runtime.tool_handlers`.
If any selected tool does not resolve to a handler, Themis raises an error
before inference runs. Provide the handler either through
`PluginRegistry.register_tool(...)` or `RuntimeContext.tool_handlers`.

For direct compiler/planner usage outside `Orchestrator`, pass project tools
explicitly so benchmark-local overrides use the same merge rules:

```python
experiment = compile_benchmark(benchmark, project_tools=project.tools)
planned_trials = planner.plan_benchmark(benchmark, project_tools=project.tools)
```

## What Themis Does Not Do

This pattern gives the engine the right prompt and tool envelope. It does not
turn Themis into an agent runtime.

Themis does not:

- implement a built-in ReAct loop
- execute tool calls on behalf of the engine
- define a standard tool invocation protocol
- attach tools to prompt variants

The engine still owns:

- provider request construction
- agent control flow
- actual tool invocation behavior
- deciding when to stop and return the terminal assistant answer

## Canonical Example

`examples/10_agent_eval.py` is the canonical advanced example for:

- bootstrap `system` and `developer` messages
- scripted follow-up turns
- project and benchmark tool declarations
- slice-level tool selection
- engine access to `trial.tools` and `runtime.tool_handlers`
- returned tool-call and tool-result conversation events
