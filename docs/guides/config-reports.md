# Generate Config Reports

Use config reports when you need a reproducibility artifact for the exact
project and benchmark configuration.

```python
from themis import generate_config_report

bundle = {"project": project, "benchmark": benchmark}

markdown_report = generate_config_report(bundle, format="markdown")
full_json_report = generate_config_report(bundle, format="json", verbosity="full")
```

Supported built-in formats:

- `json`
- `yaml`
- `markdown`
- `latex`

Use `verbosity="default"` for a concise handoff and `verbosity="full"` for the
complete collected tree.
