from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def _run(*args: str, cwd: Path | None = None) -> None:
    result = subprocess.run(args, cwd=cwd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    dist_dir = repo_root / "dist"
    wheels = sorted(dist_dir.glob("themis_eval-*.whl"))
    if not wheels:
        print("no built wheel found in dist/", file=sys.stderr)
        return 1

    wheel = wheels[-1]
    with tempfile.TemporaryDirectory(prefix="themis-wheel-smoke-") as temp_dir:
        temp_path = Path(temp_dir)
        venv_dir = temp_path / "venv"
        _run(sys.executable, "-m", "venv", str(venv_dir))

        bindir = "Scripts" if sys.platform == "win32" else "bin"
        python = venv_dir / bindir / "python"
        pip = venv_dir / bindir / "pip"

        _run(str(pip), "install", str(wheel))
        _run(
            str(python),
            "-c",
            (
                "from themis import Experiment, __all__; "
                "assert 'Experiment' in __all__; "
                "assert Experiment is not None"
            ),
            cwd=temp_path,
        )

    print(f"wheel smoke test passed for {wheel.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
