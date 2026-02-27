"""011_yaml_registry.py

This example demonstrates how to run an evaluation entirely through Themis's
declarative YAML registry system, avoiding custom Python pipeline code.
"""

from pathlib import Path
from themis.config.schema import ExperimentConfig
from themis.config.runtime import run_experiment_from_config

# 1. Define our declarative layout inline (usually this lives in `config.yaml`)
yaml_content = """
name: "custom_yaml_eval"
task: "custom"
task_options:
  template: "Please answer the following question: {question}"
  dataset_id_field: "id"
  reference_field: "reference"

dataset:
    source: "inline"
    inline_samples:
      - id: "1"
        question: "What is the capital of France?"
        reference: "Paris"
      - id: "2"
        question: "Respond strictly with '<answer>42</answer>'"
        reference: "42"

generation:
    model_identifier: "gpt-4o-mini"
    provider:
        name: "fake"
        options: {}
    sampling:
        temperature: 0.0
        max_tokens: 50

pipeline:
    extractor:
        name: "regex"
        options:
            pattern: "(?sm)<answer>(.*?)</answer>|([a-zA-Z]+)"
    metrics:
        - name: "exact_match"
          options:
            case_sensitive: false

storage:
    default_path: "./outputs/yaml-runs"
"""


def main() -> None:
    # 2. Write the YAML to disk for the demonstration
    config_path = Path("custom_eval.yaml")
    config_path.write_text(yaml_content)

    print("Running evaluation from custom_eval.yaml...")

    # 3. Load the config using Hydra schema validation
    # This automatically picks up environment variables specified via ${oc.env}
    config = ExperimentConfig.from_file(config_path)

    # 4. Construct and Run the experiment seamlessly
    report = run_experiment_from_config(config)

    print("\nExperiment Complete!")
    print("--------------------------------------------------")
    print(f"Total Samples    : {report.metadata.get('total_samples')}")
    print(f"Failures         : {len(report.failures)}")
    print("Metrics:")
    for metric_name, aggregate in report.evaluation_report.metrics.items():
        print(f"  {metric_name}: {aggregate.mean:.3f}")

    # Cleanup
    if config_path.exists():
        config_path.unlink()


if __name__ == "__main__":
    main()
