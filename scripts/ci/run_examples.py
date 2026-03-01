"""CI script to run all examples with mocked LLM responses."""

import os
import sys
import runpy
from pathlib import Path
from unittest.mock import patch

# Set dummy keys to bypass provider initialization checks
os.environ["OPENAI_API_KEY"] = "sk-dummy"
os.environ["ANTHROPIC_API_KEY"] = "sk-dummy"
os.environ["LITELLM_API_KEY"] = "sk-dummy"


def mock_litellm_completion(*args, **kwargs):
    import litellm

    class MockMessage:
        content = "4"
        role = "assistant"

    class MockChoice:
        message = MockMessage()
        finish_reason = "stop"
        index = 0

    class MockUsage:
        prompt_tokens = 10
        completion_tokens = 10
        total_tokens = 20

    class MockResponse(litellm.ModelResponse):
        choices = [MockChoice()]
        usage = MockUsage()
        model = kwargs.get("model", "mock-model")
        id = "mock-id"

    return MockResponse()


async def mock_litellm_acompletion(*args, **kwargs):
    return mock_litellm_completion(*args, **kwargs)


def main():
    root_dir = Path(__file__).parent.parent.parent
    examples_dir = root_dir / "examples"

    if not examples_dir.exists():
        print(f"Examples directory not found at {examples_dir}")
        sys.exit(1)

    python_files = sorted(examples_dir.glob("*.py"))

    # Exclude examples that start servers or enter infinite loops
    exclude = {"05_api_server.py", "09_research_loop.py"}

    success = True

    print("Starting example validation...")

    import shutil

    # Patch litellm extensively
    with (
        patch("litellm.completion", side_effect=mock_litellm_completion),
        patch("litellm.acompletion", side_effect=mock_litellm_acompletion),
    ):
        for script in python_files:
            if script.name in exclude:
                print(f"⏭️ Skipping {script.name}")
                continue

            # Clean up cache to prevent run_id collisions
            for cache_dir in [Path(".themis_cache"), Path(".cache")]:
                if cache_dir.exists():
                    shutil.rmtree(cache_dir, ignore_errors=True)

            print(f"▶️ Running {script.name}...")
            try:
                # Run the script in its own module namespace
                runpy.run_path(str(script), run_name="__main__")
                print(f"✅ {script.name} succeeded.\n")
            except SystemExit as e:
                if isinstance(e.code, int) and e.code != 0:
                    print(f"❌ {script.name} exited with code {e.code}\n")
                    success = False
                elif e.code is not None and not isinstance(e.code, int):
                    print(f"❌ {script.name} exited with error: {e.code}\n")
                    success = False
                else:
                    print(f"✅ {script.name} succeeded (sys.exit(0)).\n")
            except Exception as e:
                print(f"❌ {script.name} failed: {e}\n")
                success = False

    if not success:
        print("❌ Example validation failed.")
        sys.exit(1)

    print("✅ All examples validated successfully.")


if __name__ == "__main__":
    main()
