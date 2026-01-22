"""Example: Comparing Multiple Runs

This example shows how to compare results from multiple experiment runs
using the comparison engine with statistical tests.
"""

from pathlib import Path

from themis.comparison import compare_runs
from themis.comparison.statistics import StatisticalTest


def main():
    """Compare two runs and display results."""
    
    # Define runs to compare
    # (In practice, these would be from actual experiment runs)
    run_ids = ["run-gpt4", "run-claude"]
    
    # Storage path where runs are saved
    storage_path = Path(".cache/experiments")
    
    # Check if storage exists
    if not storage_path.exists():
        print(f"Storage path {storage_path} does not exist.")
        print("Run some evaluations first before comparing.")
        return
    
    try:
        # Run comparison with bootstrap confidence intervals
        report = compare_runs(
            run_ids=run_ids,
            storage_path=storage_path,
            metrics=None,  # None = compare all available metrics
            statistical_test=StatisticalTest.BOOTSTRAP,
            alpha=0.05,  # 95% confidence level
        )
        
        # Print summary
        print(report.summary(include_details=True))
        
        # Access specific comparisons
        accuracy_result = report.get_comparison("run-gpt4", "run-claude", "ExactMatch")
        if accuracy_result:
            print(f"\nExactMatch: {accuracy_result.summary()}")
        
        # Check overall winner
        print(f"\nOverall best run: {report.overall_best_run}")
        
        # Export to file
        output = Path("comparison_report.json")
        import json
        output.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nReport exported to {output}")
    
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure the specified runs exist in storage.")


if __name__ == "__main__":
    # Note: This example requires actual runs in storage to compare
    # Run evaluations first using 01_quickstart.py or 02_custom_dataset.py
    # with different --run-id values
    
    print("Comparison Example")
    print("="*60)
    print()
    print("To use this example with real data:")
    print("1. Run evaluations with different configurations:")
    print("   themis eval demo --model fake-math-llm --limit 10 --run-id run-a")
    print("   themis eval demo --model fake-math-llm --limit 10 --run-id run-b")
    print()
    print("2. Then compare them:")
    print("   themis compare run-a run-b")
    print()
    print("For now, showing how the comparison API works programmatically:")
    print()
    
    # Show example of programmatic comparison
    from themis.comparison.statistics import t_test, bootstrap_confidence_interval
    
    # Simulated results from two models
    model_a_scores = [0.8, 0.85, 0.82, 0.88, 0.79]
    model_b_scores = [0.75, 0.78, 0.77, 0.80, 0.74]
    
    # Perform t-test
    t_result = t_test(model_a_scores, model_b_scores, paired=True)
    print("T-test result:")
    print(f"  {t_result}")
    print()
    
    # Perform bootstrap
    boot_result = bootstrap_confidence_interval(
        model_a_scores, model_b_scores, n_bootstrap=1000, seed=42
    )
    print("Bootstrap result:")
    print(f"  {boot_result}")
    print()
    
    # Interpret
    if boot_result.significant:
        print("✓ The difference is statistically significant!")
    else:
        print("  The difference is not statistically significant")
    
    print()
    print("="*60)
    print("See themis/comparison/ for more details on the comparison engine")
