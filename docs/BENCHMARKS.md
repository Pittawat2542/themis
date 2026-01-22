# Benchmarks Reference

Complete reference for all built-in benchmarks in Themis.

## Overview

Themis includes 6 carefully selected benchmarks covering:
- **Math Reasoning**: GSM8K, MATH500, AIME24
- **General Knowledge**: MMLU-Pro, SuperGPQA
- **Quick Testing**: Demo

---

## Math Benchmarks

### demo

**Quick testing benchmark with minimal samples**

- **Dataset Size**: 10 samples
- **Domain**: Elementary arithmetic
- **Difficulty**: Easy
- **Source**: Subset of GSM8K
- **Metrics**: ExactMatch, MathVerify
- **License**: MIT

**Usage:**
```python
from themis import evaluate

# Test without API key
result = evaluate(benchmark="demo", model="fake-math-llm")

# Test with real model
result = evaluate(benchmark="demo", model="gpt-4")
```

**Example Problems:**
- "Janet's ducks lay 16 eggs per day. She eats three for breakfast..."
- "A robe takes 2 bolts of blue fiber and half that much white fiber..."

**When to use:**
- Testing your setup
- Verifying Themis installation
- Quick experiments without API costs

---

### gsm8k

**Grade School Math 8K - Elementary math word problems**

- **Dataset Size**: 8,500+ problems
- **Domain**: Grade school mathematics (K-8)
- **Difficulty**: Elementary to middle school
- **Source**: [GSM8K Paper](https://arxiv.org/abs/2110.14168)
- **Metrics**: ExactMatch, MathVerify
- **License**: MIT

**Usage:**
```python
result = evaluate(benchmark="gsm8k", model="gpt-4", limit=100)
```

**Example Problems:**
- Multi-step arithmetic word problems
- Real-world scenarios (money, measurements, time)
- Requires basic arithmetic and reasoning

**Prompt Template:**
```
Problem: {prompt}
Solution:
```

**Evaluation:**
- Extracts final numerical answer
- Compares with reference using exact match and math verification

**Typical Scores:**
- GPT-4: ~90-95%
- GPT-3.5: ~55-60%
- Claude-3: ~90-95%

---

### math500

**MATH500 - Advanced competition-level math**

- **Dataset Size**: 500 problems
- **Domain**: High school to competition math
- **Difficulty**: Advanced (AIME, AMC, IMO level)
- **Source**: [MATH Dataset](https://arxiv.org/abs/2103.03874)
- **Metrics**: ExactMatch, MathVerify
- **License**: MIT

**Usage:**
```python
result = evaluate(benchmark="math500", model="gpt-4")
```

**Example Problems:**
- Algebra, geometry, calculus, number theory
- Requires multiple steps and deep reasoning
- Often includes LaTeX formatting

**Prompt Template:**
```
Solve this problem: {problem}
```

**Typical Scores:**
- GPT-4: ~30-40%
- GPT-3.5: ~5-10%
- Claude-3: ~35-45%

---

### aime24

**AIME 2024 - American Invitational Mathematics Examination**

- **Dataset Size**: 30 problems
- **Domain**: Competition mathematics
- **Difficulty**: Very challenging (top high school level)
- **Source**: 2024 AIME exam
- **Metrics**: ExactMatch, MathVerify
- **License**: Public domain (exam questions)

**Usage:**
```python
result = evaluate(benchmark="aime24", model="gpt-4")
```

**Example Problems:**
- Advanced algebra, geometry, combinatorics
- Integer answers from 000 to 999
- Requires deep mathematical reasoning

**Prompt Template:**
```
Problem: {problem}

Solution:
```

**Typical Scores:**
- GPT-4: ~10-15% (3-5 out of 30)
- GPT-3.5: ~0-5%
- Claude-3: ~10-20%

**Note**: This is an extremely challenging benchmark. Even top models struggle.

---

## Knowledge Benchmarks

### mmlu_pro

**MMLU-Pro - Massive Multitask Language Understanding (Professional)**

- **Dataset Size**: Thousands of questions
- **Domain**: Multiple subjects (STEM, humanities, social sciences)
- **Difficulty**: College and professional level
- **Source**: [MMLU-Pro](https://arxiv.org/abs/2406.01574)
- **Metrics**: ExactMatch
- **License**: MIT

**Usage:**
```python
result = evaluate(benchmark="mmlu_pro", model="gpt-4", limit=1000)
```

**Subjects:**
- Physics, Chemistry, Biology
- History, Law, Economics
- Computer Science, Mathematics
- And many more

**Format**: Multiple choice questions

**Typical Scores:**
- GPT-4: ~70-75%
- GPT-3.5: ~45-50%
- Claude-3: ~70-75%

---

### supergpqa

**SuperGPQA - Advanced reasoning and knowledge**

- **Dataset Size**: Varies
- **Domain**: Expert-level questions
- **Difficulty**: Graduate level and beyond
- **Source**: [GPQA](https://arxiv.org/abs/2311.12022)
- **Metrics**: ExactMatch
- **License**: CC-BY

**Usage:**
```python
result = evaluate(benchmark="supergpqa", model="gpt-4")
```

**Characteristics:**
- Requires expert knowledge
- Multi-step reasoning
- Often requires combining multiple concepts

**Typical Scores:**
- GPT-4: ~35-40%
- GPT-3.5: ~20-25%
- Claude-3: ~40-45%

---

## Benchmark Comparison

| Benchmark | Size | Difficulty | Domain | Typical Time |
|-----------|------|------------|--------|--------------|
| demo | 10 | Easy | Math | < 1 min |
| gsm8k | 8,500 | Medium | Math | 30-60 min |
| math500 | 500 | Hard | Math | 5-10 min |
| aime24 | 30 | Very Hard | Math | 1-2 min |
| mmlu_pro | Large | Medium-Hard | Knowledge | 20-40 min |
| supergpqa | Medium | Very Hard | Knowledge | 10-20 min |

*Times are estimates with GPT-4 and 8 workers*

---

## Usage Recommendations

### For Quick Testing
→ Use `demo` (10 samples, no API key needed)

### For Math Evaluation
→ Start with `gsm8k` (most widely used)
→ Then try `math500` for harder problems
→ Use `aime24` for extreme difficulty

### For General Knowledge
→ Use `mmlu_pro` for broad coverage
→ Use `supergpqa` for expert-level questions

### For Research Papers
→ Use `gsm8k` and `math500` (widely reported)
→ Include multiple benchmarks for comprehensive evaluation

---

## Adding Custom Benchmarks

You can register your own benchmarks:

```python
from themis.presets.benchmarks import BenchmarkPreset, register_benchmark
from themis.generation.templates import PromptTemplate
from themis.evaluation.metrics.math import ExactMatch
from themis.evaluation.extractors import LastLineExtractor

def my_dataset_loader(limit=None):
    data = load_my_data()
    return data[:limit] if limit else data

preset = BenchmarkPreset(
    name="my-benchmark",
    prompt_template=PromptTemplate(template="Q: {question}\nA:"),
    metrics=[ExactMatch()],
    extractor=LastLineExtractor(),
    dataset_loader=my_dataset_loader,
    description="My custom benchmark",
)

register_benchmark(preset)

# Use it
result = evaluate(benchmark="my-benchmark", model="gpt-4")
```

---

## Benchmark Statistics

### GSM8K
- Problems: 8,500
- Average length: 3-5 sentences
- Topics: Addition, subtraction, multiplication, division, percentages, ratios
- Format: Natural language word problems
- Evaluation: Numerical answer extraction

### MATH500
- Problems: 500
- Average length: 5-10 sentences
- Topics: Algebra, geometry, calculus, number theory, combinatorics
- Format: LaTeX mathematical notation
- Evaluation: Symbolic and numerical matching

### AIME24
- Problems: 30
- Average length: 3-8 sentences
- Topics: Advanced competition mathematics
- Format: Natural language with mathematical notation
- Evaluation: Integer answers (000-999)

---

## Best Practices

### 1. Start with Demo

Always test with `demo` first:
```python
result = evaluate(benchmark="demo", model="fake-math-llm")
```

### 2. Use Appropriate Limits

For expensive models:
```python
# Test with small sample
result = evaluate(benchmark="gsm8k", model="gpt-4", limit=100)

# Full evaluation once validated
result = evaluate(benchmark="gsm8k", model="gpt-4")
```

### 3. Choose Right Benchmark

Match benchmark to your use case:
- Testing math ability → gsm8k or math500
- Competition performance → aime24
- General knowledge → mmlu_pro
- Expert reasoning → supergpqa

### 4. Report Multiple Benchmarks

For research, evaluate on multiple benchmarks:
```python
benchmarks = ["gsm8k", "math500", "aime24"]

for benchmark in benchmarks:
    result = evaluate(
        benchmark=benchmark,
        model="gpt-4",
        run_id=f"{benchmark}-gpt4",
    )
```

---

## See Also

- [Presets API](api/presets.md) - Benchmark API reference
- [Evaluation Guide](guides/evaluation.md) - Using benchmarks
- [Custom Datasets](guides/evaluation.md#custom-datasets) - Create your own
