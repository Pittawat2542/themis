1. Resume if eval is interrupted without re-eval entirely
2. Generation-only
3. Evaluation-only (bring your own results and create a custom script to map into themis-compatible format)
4. Let's say I used the generation only and then use outside batch API from OpenAI for evaluation with LM judge then I create a script to convert result back to themis-compatible format for the rest of the flow to continue?
5. Let's say I'm done with generation and evaluation. Later, I want to add new model, new benchmark, or new metric and it will only generate or evaluate the new stuffs.
6. Does themis save intermediate results from all stages in storage-efficient way. Let's say I want to pull generated responses for analysis or extracted responses for another evaluation, or just re-extract responses.
7. Can I easily add new benchmarks, metrics, models, evaluators, extractors, prompts, etc. to an experiement?
8. Can I define my experiment purely in code (code as config) while still having the option to use a config file?
9. Can I version my experiment configs so I can reproduce the exact same run (model, prompt, benchmark version) months later?
10. Can I diff two experiment configs to quickly understand what changed between runs?
11. Can I fix random seeds for generation to ensure deterministic, reproducible outputs?
12. Can I run multiple models or benchmarks concurrently across multiple GPUs or machines?
13. Can I parallelize across API-based models the same way I would for local GPU models?
14. Does it support batch APIs (e.g., OpenAI Batch API) where the pipeline submits a batch job, pauses, and automatically resumes once the batch is complete?
15. Does it gracefully handle API timeouts, quota exhaustion, or transient failures mid-run with retries?
16. Can I do a dry-run to estimate token cost before committing to a full eval?
17. Can it automatically compile results across all models × benchmarks into an aggregate leaderboard or summary table?
18. Can I compute confidence intervals or p-values when comparing two models?
19. Can I run eval on just a specific slice of a benchmark (e.g., hard examples, a specific category)?
20. Can I sweep over multiple prompt templates for the same benchmark and compare results?
21. Can I grid search over generation parameters (temperature, top-p) without re-running unaffected configs?
22. Can I easily toggle between zero-shot and few-shot without rewriting benchmark logic?
23. Can I use multiple judges simultaneously, each with its own judge prompt and parsing logic, on the same set of responses?
24. Can I drill down into individual examples where a model scored poorly to diagnose failure modes?
25. Can I tag and categorize failure cases (refusals, hallucinations, format errors) for qualitative analysis?
26. Can I surface examples where the extractor returned null or invalid output so I can fix edge cases?
27. Can it detect if the same (model, benchmark, config) combo was already evaluated and skip or warn me?
28. Can I export final results to CSV, JSON, or a database for downstream reporting or dashboarding?
29. Can it efficiently store and retrieve very large generated responses without running into memory issues?