---
title: Capture traces and conversations
diataxis: how-to
audience: users who need post-run inspection or trace-level evaluation
goal: Show how to emit trace and conversation artifacts from generation.
---

# Capture traces and conversations

Goal: store trace and conversation artifacts so you can inspect or score them later.

When to use this:

Use this guide when generated final output alone is not enough to explain or evaluate model behavior.

## Procedure

Populate `GenerationResult.trace` and `GenerationResult.conversation` inside your generator.

```python
--8<-- "examples/docs/trace_capture.py"
```

--8<-- "docs/_snippets/how-to/trace-inspection-note.md"

## Variants

| Variant | Best when | Tradeoff | Related APIs / commands |
| --- | --- | --- | --- |
| Trace only | You need execution breadcrumbs without full turn-by-turn conversation state | Less useful for prompt-review workflows | `GenerationResult.trace` |
| Conversation only | You need the conversational exchange for later inspection or judging | Provides less internal execution structure than a trace | `GenerationResult.conversation` |
| Full inspection workflows | You want both trace-level and conversation-level evidence persisted for later analysis | More storage and generator implementation work | `GenerationResult.trace`, `GenerationResult.conversation`, persistent stores |

## Expected result

The run should expose trace-oriented projections and make those artifacts available for later inspection.

## Troubleshooting

- [Artifacts and inspection](../explanation/artifacts-and-inspection.md)
- [Data models reference](../reference/data-models.md)
