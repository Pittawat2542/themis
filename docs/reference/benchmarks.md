# Benchmarks Reference

Complete reference for all built-in benchmarks in Themis.

## Overview

Themis includes 19 built-in benchmarks covering:
- **Math Reasoning**: GSM8K, MATH500, GSM-Symbolic, AIME24, AIME25, AMC23, OlympiadBench, BeyondAIME
- **Knowledge & Science**: MMLU-Pro, SuperGPQA, GPQA, SciQ
- **Medicine**: MedMCQA, MedQA
- **Commonsense**: CommonsenseQA, PIQA, Social IQA
- **Conversational QA**: CoQA
- **Quick Testing**: Demo

---

## Benchmark Catalog (Summary)

| Benchmark | Domain | Format | Notes |
| --- | --- | --- | --- |
| demo | Quick testing | Short QA | Built-in tiny dataset |
| gsm8k | Math | Free-form | Grade school math word problems |
| math500 | Math | Free-form | Competition math (MATH) |
| gsm-symbolic | Math | Free-form | Symbolic math variations |
| aime24 | Math | Free-form | AIME 2024 |
| aime25 | Math | Free-form | AIME 2025 |
| amc23 | Math | Free-form | AMC 2023 |
| olympiadbench | Math | Free-form | Olympiad-style problems |
| beyondaime | Math | Free-form | Advanced contest problems |
| mmlu-pro | Knowledge | MCQ (letter) | Professional-level subjects |
| supergpqa | Science | MCQ (letter) | Graduate-level science |
| gpqa | Science | MCQ (letter) | Graduate-level QA |
| sciq | Science | MCQ (letter) | Science questions |
| medmcqa | Medicine | MCQ (letter) | Medical exams |
| med_qa | Medicine | MCQ (letter) | Medical QA |
| commonsense_qa | Commonsense | MCQ (letter) | Commonsense reasoning |
| piqa | Commonsense | MCQ (letter) | Physical commonsense |
| social_i_qa | Commonsense | MCQ (letter) | Social reasoning |
| coqa | Conversational | Free-form | Multi-turn QA |

---

## Math Benchmarks

### demo

**Quick testing benchmark with minimal samples**

- **Dataset Size**: 10 samples
- **Domain**: Elementary arithmetic
- **Difficulty**: Easy
- **Source**: Subset of GSM8K
- **Metrics**: ExactMatch, MathVerifyAccuracy
- **License**: MIT

**Usage:**
```python
from themis import evaluate

# Test without API key
result = evaluate("demo", model="fake-math-llm")

# Test with real model
result = evaluate("demo", model="gpt-4")
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
- **Metrics**: ExactMatch, MathVerifyAccuracy
- **License**: MIT

**Usage:**
```python
result = evaluate("gsm8k", model="gpt-4", limit=100)
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
- **Metrics**: ExactMatch, MathVerifyAccuracy
- **License**: MIT

**Usage:**
```python
result = evaluate("math500", model="gpt-4")
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
- **Metrics**: ExactMatch, MathVerifyAccuracy
- **License**: Public domain (exam questions)

**Usage:**
```python
result = evaluate("aime24", model="gpt-4")
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

### mmlu-pro

**MMLU-Pro - Massive Multitask Language Understanding (Professional)**

- **Dataset Size**: Thousands of questions
- **Domain**: Multiple subjects (STEM, humanities, social sciences)
- **Difficulty**: College and professional level
- **Source**: [MMLU-Pro](https://arxiv.org/abs/2406.01574)
- **Metrics**: ExactMatch
- **License**: MIT

**Usage:**
```python
result = evaluate("mmlu-pro", model="gpt-4", limit=1000)
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
result = evaluate("supergpqa", model="gpt-4")
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

## Science & Medical Benchmarks

### gpqa

**GPQA - Graduate-level science questions**

- **Domain**: Science
- **Format**: Multiple choice (letter)
- **Metrics**: ExactMatch

**Usage:**
```python
result = evaluate("gpqa", model="gpt-4")
```

---

### sciq

**SciQ - Science question answering**

- **Domain**: Science
- **Format**: Multiple choice (letter)
- **Metrics**: ExactMatch

**Usage:**
```python
result = evaluate("sciq", model="gpt-4", limit=100)
```

---

### medmcqa

**MedMCQA - Medical entrance exam questions**

- **Domain**: Medicine
- **Format**: Multiple choice (letter)
- **Metrics**: ExactMatch

**Usage:**
```python
result = evaluate("medmcqa", model="gpt-4", limit=200)
```

---

### med_qa

**MedQA - Medical QA benchmark**

- **Domain**: Medicine
- **Format**: Multiple choice (letter)
- **Metrics**: ExactMatch

**Usage:**
```python
result = evaluate("med_qa", model="gpt-4", limit=200)
```

---

## Commonsense Benchmarks

### commonsense_qa

**CommonsenseQA - Commonsense reasoning**

- **Domain**: Commonsense
- **Format**: Multiple choice (letter)
- **Metrics**: ExactMatch
- **Note**: Uses the validation split because test labels are not public.

**Usage:**
```python
result = evaluate("commonsense_qa", model="gpt-4", limit=200)
```

---

### piqa

**PIQA - Physical commonsense reasoning**

- **Domain**: Commonsense
- **Format**: Multiple choice (letter)
- **Metrics**: ExactMatch
- **Note**: Uses the validation split because test labels are not public.

**Usage:**
```python
result = evaluate("piqa", model="gpt-4", limit=200)
```

---

### social_i_qa

**Social IQA - Social reasoning**

- **Domain**: Commonsense
- **Format**: Multiple choice (letter)
- **Metrics**: ExactMatch
- **Note**: Uses the validation split because test labels are not public.

**Usage:**
```python
result = evaluate("social_i_qa", model="gpt-4", limit=200)
```

---

## Conversational QA Benchmarks

### coqa

**CoQA - Conversational question answering**

- **Domain**: Conversational QA
- **Format**: Free-form
- **Metrics**: ExactMatch
- **Note**: Uses the validation split because test labels are not public.

**Usage:**
```python
result = evaluate("coqa", model="gpt-4", limit=200)
```

---

## Additional Math Benchmarks

### gsm-symbolic

**GSM-Symbolic - Symbolic variants of GSM8K**

- **Domain**: Math
- **Format**: Free-form
- **Metrics**: MathVerifyAccuracy

**Usage:**
```python
result = evaluate("gsm-symbolic", model="gpt-4", limit=200)
```

---

### aime25

**AIME 2025 - Competition math**

- **Domain**: Math
- **Format**: Free-form
- **Metrics**: MathVerifyAccuracy

**Usage:**
```python
result = evaluate("aime25", model="gpt-4")
```

---

### amc23

**AMC 2023 - Competition math**

- **Domain**: Math
- **Format**: Free-form
- **Metrics**: MathVerifyAccuracy

**Usage:**
```python
result = evaluate("amc23", model="gpt-4")
```

---

### olympiadbench

**OlympiadBench - Olympiad-style problems**

- **Domain**: Math
- **Format**: Free-form
- **Metrics**: MathVerifyAccuracy

**Usage:**
```python
result = evaluate("olympiadbench", model="gpt-4")
```

---

### beyondaime

**BeyondAIME - Advanced contest math**

- **Domain**: Math
- **Format**: Free-form
- **Metrics**: MathVerifyAccuracy

**Usage:**
```python
result = evaluate("beyondaime", model="gpt-4")
```

---

## Benchmark Comparison (Core Set)

| Benchmark | Size | Difficulty | Domain | Typical Time |
|-----------|------|------------|--------|--------------|
| demo | 10 | Easy | Math | < 1 min |
| gsm8k | 8,500 | Medium | Math | 30-60 min |
| math500 | 500 | Hard | Math | 5-10 min |
| aime24 | 30 | Very Hard | Math | 1-2 min |
| mmlu-pro | Large | Medium-Hard | Knowledge | 20-40 min |
| supergpqa | Medium | Very Hard | Knowledge | 10-20 min |

*Times are estimates with GPT-4 and 8 workers*

---

## Coverage Gaps by Domain

- **Multimodal**: No built-in image/audio/video benchmarks.
- **Tool-use / agentic**: No built-in tool-use or multi-step tool benchmarks.
- **Long-context**: No dedicated long-context datasets (100K+ tokens).
- **Safety/alignment**: No built-in safety or red-teaming benchmarks.

If these domains are critical, integrate external benchmarks via custom datasets.

## Usage Recommendations

### For Quick Testing
→ Use `demo` (10 samples, no API key needed)

### For Math Evaluation
→ Start with `gsm8k` (most widely used)
→ Then try `math500` for harder problems
→ Use `aime24`/`aime25`/`amc23`/`olympiadbench` for competition difficulty
→ Use `gsm-symbolic` to test symbolic variants

### For General Knowledge
→ Use `mmlu-pro` for broad coverage
→ Use `supergpqa` for expert-level questions

### For Science & Medical
→ Use `gpqa` or `sciq` for science MCQ
→ Use `medmcqa` or `med_qa` for medical QA

### For Commonsense Reasoning
→ Use `commonsense_qa`, `piqa`, and `social_i_qa` for diverse commonsense tasks

### For Conversational QA
→ Use `coqa` for multi-turn question answering

### For Research Papers
→ Use `gsm8k` and `math500` (widely reported)
→ Include multiple benchmarks for comprehensive evaluation

---

## Adding Custom Benchmarks

You can register your own benchmarks:

```python
from themis.presets.benchmarks import BenchmarkPreset, register_benchmark
from themis.generation.templates import PromptTemplate
from themis.evaluation import extractors, metrics

def my_dataset_loader(limit=None):
    data = load_my_data()
    return data[:limit] if limit else data

preset = BenchmarkPreset(
    name="my-benchmark",
    prompt_template=PromptTemplate(name="custom", template="Q: {question}\nA:"),
    metrics=[metrics.ExactMatch()],
    extractor=extractors.IdentityExtractor(),
    dataset_loader=my_dataset_loader,
    description="My custom benchmark",
)

register_benchmark(preset)

# Use it
result = evaluate("my-benchmark", model="gpt-4")
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
result = evaluate("demo", model="fake-math-llm")
```

### 2. Use Appropriate Limits

For expensive models:
```python
# Test with small sample
result = evaluate("gsm8k", model="gpt-4", limit=100)

# Full evaluation once validated
result = evaluate("gsm8k", model="gpt-4")
```

### 3. Choose Right Benchmark

Match benchmark to your use case:
- Testing math ability → gsm8k or math500
- Competition performance → aime24
- General knowledge → mmlu-pro
- Expert reasoning → supergpqa

### 4. Report Multiple Benchmarks

For research, evaluate on multiple benchmarks:
```python
benchmarks = ["gsm8k", "math500", "aime24"]

for benchmark in benchmarks:
    result = evaluate(benchmark,
        model="gpt-4",
        run_id=f"{benchmark}-gpt4",
    )
```

---

## See Also

- [Presets API](../api/presets.md) - Benchmark API reference
- [Evaluation Guide](../guides/evaluation.md) - Using benchmarks
- [Custom Datasets](../guides/evaluation.md#custom-dataset) - Create your own
