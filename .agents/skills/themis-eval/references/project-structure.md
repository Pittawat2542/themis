# Project Structure

Use this reference when the user wants the ideal manual project shape that
matches `themis init`, or when you are about to run the structure generator
script.

## Default Local Starter

Matches `themis init <path>` without `--benchmark`:

```text
<project-root>/
  project.toml
  .env.example
  README.md
  data/
    sample.jsonl
  <package_name>/
    __init__.py
    __main__.py
    app.py
    settings.py
    registry.py
    benchmarks/
      __init__.py
      default.py
    datasets/
      __init__.py
      local_file.py
```

## Built-In Benchmark Starter

Matches `themis init <path> --benchmark <id>`:

```text
<project-root>/
  project.toml
  .env.example
  README.md
  <package_name>/
    __init__.py
    __main__.py
    app.py
    settings.py
    registry.py
    benchmarks/
      __init__.py
      default.py
    datasets/
      __init__.py
      builtin.py
```

## Generator Script

Prefer the script when the user wants the tree materialized in a real folder:

```bash
python .agents/skills/themis-eval/scripts/generate_project_structure.py \
  --target /path/to/project \
  --package-name starter_eval \
  --mode default
```

```bash
python .agents/skills/themis-eval/scripts/generate_project_structure.py \
  --target /path/to/project \
  --package-name starter_eval \
  --mode builtin
```

The script only creates the shape and placeholder files. Use `themis init`
instead when the user wants the full generated starter contents from the CLI.
