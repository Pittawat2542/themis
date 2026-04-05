---
title: Writing guide
diataxis: project
audience: docs contributors
goal: Provide authoring rules for Diátaxis pages in this repo.
---

# Writing guide

Authoring rules:

- every substantive page needs front matter with `diataxis`, `audience`, and `goal`
- tutorials teach one linear workflow
- how-to guides solve one task and assume motivation already exists
- reference pages stay dry, exact, and lookup-oriented
- explanation pages prioritize mental models and design reasoning
- explanation pages may use Mermaid diagrams when a visual clarifies a mental model, boundary, lifecycle, or ownership split
- start-here and other landing pages may use Mermaid diagrams for routing and decision support
- how-to guides may use Mermaid diagrams only when they clarify a task flow or meaningful branch in the procedure
- tutorials may include diagrams sparingly, but the step sequence remains the primary teaching device
- reference pages should prefer tables and terse structure over diagrams unless a visual is the clearest lookup aid
- if a section mostly enumerates available options and a reader would reasonably ask which one to choose, prefer a table over a flat bullet list
- option-inventory tables must include at least one decision-helping column such as `Use when`, `Best for`, `Tradeoff`, `Notes`, or `Key constraints / notes`
- keep bullets for procedures, conceptual definitions, troubleshooting links, and short narrative takeaways rather than forcing every list into a table
- cross-link glossary terms from first meaningful use in major pages
- do not hand-copy code that already exists in `examples/docs/`
