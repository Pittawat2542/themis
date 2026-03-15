"""Run the current examples directory as a CI smoke test."""

import sys
import runpy
from pathlib import Path


def main():
    root_dir = Path(__file__).parent.parent.parent
    examples_dir = root_dir / "examples"

    if not examples_dir.exists():
        print(f"Examples directory not found at {examples_dir}")
        sys.exit(1)

    python_files = sorted(examples_dir.glob("*.py"))

    success = True

    print("Starting example validation...")

    import shutil

    for script in python_files:
        for cache_dir in [Path(".cache"), Path(".themis_cache")]:
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)

        print(f"Running {script.name}...")
        try:
            runpy.run_path(str(script), run_name="__main__")
            print(f"{script.name} succeeded.\n")
        except SystemExit as exc:
            if isinstance(exc.code, int) and exc.code != 0:
                print(f"{script.name} exited with code {exc.code}\n")
                success = False
            elif exc.code is not None and not isinstance(exc.code, int):
                print(f"{script.name} exited with error: {exc.code}\n")
                success = False
            else:
                print(f"{script.name} succeeded (sys.exit(0)).\n")
        except Exception as exc:
            print(f"{script.name} failed: {exc}\n")
            success = False

    if not success:
        print("Example validation failed.")
        sys.exit(1)

    print("All examples validated successfully.")


if __name__ == "__main__":
    main()
