"""Run the documentation examples as a CI smoke test."""

import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def _reset_example_cache_dirs(root_dir: Path = Path(".")) -> None:
    for cache_dir in [root_dir / ".cache", root_dir / ".themis_cache"]:
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)


def _run_example(script: Path, *, root_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=root_dir,
        capture_output=True,
        text=True,
    )


def main():
    root_dir = Path(__file__).resolve().parents[2]
    examples_dir = root_dir / "examples" / "docs"

    if not examples_dir.exists():
        print(f"Examples directory not found at {examples_dir}")
        sys.exit(1)

    python_files = sorted(examples_dir.glob("*.py"))

    success = True

    print("Starting example validation...")

    for script in python_files:
        print(f"Running {script.name}...")
        with TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            _reset_example_cache_dirs(run_dir)
            result = _run_example(script, root_dir=run_dir)
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(
                result.stderr,
                file=sys.stderr,
                end="" if result.stderr.endswith("\n") else "\n",
            )

        if result.returncode == 0:
            print(f"{script.name} succeeded.\n")
        else:
            print(f"{script.name} exited with code {result.returncode}\n")
            success = False

    if not success:
        print("Example validation failed.")
        sys.exit(1)

    print("All examples validated successfully.")


if __name__ == "__main__":
    main()
