---
title: Docs review checklist
diataxis: project
audience: contributors and reviewers
goal: Provide a compact checklist for reviewing documentation changes.
---

# Docs review checklist

Use this checklist when reviewing documentation changes:

- the page belongs to exactly one quadrant or the `Project` section
- front matter includes `diataxis`, `audience`, and `goal`
- code snippets come from `examples/docs/*.py` or source-of-truth files
- glossary terms are used consistently
- reference pages stay dry and precise
- tutorials teach one workflow
- how-to guides solve one task
- explanation pages answer what it is, when it matters, what the user provides, what Themis provides, and what to inspect when things go wrong
- option inventories use tables when readers need help choosing among commands, variants, providers, components, or backends
- inventory tables include at least one decision-helping column such as `Use when`, `Best for`, `Tradeoff`, or `Notes`
- bullets stay reserved for procedures, conceptual definitions, troubleshooting links, and short takeaways
- diagrams, if present, fit the quadrant: explanation for mental models, landing pages for routing, how-to for flow clarification, tutorials only sparingly, reference preferably as tables
- each diagram has a short lead-in sentence and a one-sentence interpretation after it
