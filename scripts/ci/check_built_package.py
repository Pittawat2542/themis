"""Smoke-test the built wheel in a clean virtual environment."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


EXPECTED_ROOT_EXPORTS = {
    "__version__",
    "BenchmarkDefinition",
    "BenchmarkDefinitionConfig",
    "Orchestrator",
    "BenchmarkResult",
    "BenchmarkSpec",
    "DatasetQuerySpec",
    "ProjectSpec",
    "ExecutionPolicySpec",
    "InferenceGridSpec",
    "InferenceParamsSpec",
    "McpServerSpec",
    "PromptMessage",
    "PromptTurnSpec",
    "PromptVariantSpec",
    "ModelSpec",
    "ToolSpec",
    "ParseSpec",
    "ScoreSpec",
    "SliceSpec",
    "TraceScoreSpec",
    "StorageConfig",
    "StorageSpec",
    "SqliteBlobStorageSpec",
    "PostgresBlobStorageSpec",
    "EngineCapabilities",
    "PluginRegistry",
    "build_benchmark_definition_project",
    "generate_config_report",
}


def _venv_bin_dir(env_dir: Path) -> Path:
    return env_dir / ("Scripts" if os.name == "nt" else "bin")


def _python_name() -> str:
    return "python.exe" if os.name == "nt" else "python"


def _quickcheck_name() -> str:
    return "themis-quickcheck.exe" if os.name == "nt" else "themis-quickcheck"


def _verification_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _import_check_snippet(site_packages_dir: Path) -> str:
    site_packages = str(site_packages_dir.resolve())
    expected_exports = sorted(EXPECTED_ROOT_EXPORTS)
    return (
        "from pathlib import Path; "
        "import themis; "
        f"expected = {expected_exports!r}; "
        "missing = sorted(set(expected) - set(themis.__all__)); "
        "assert not missing, missing; "
        f"site_packages = Path({site_packages!r}).resolve(); "
        "module_path = Path(themis.__file__).resolve(); "
        "assert site_packages in module_path.parents, (module_path, site_packages)"
    )


def _site_packages_dir(python_exe: Path) -> Path:
    result = subprocess.run(
        [
            str(python_exe),
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _read_project_version(pyproject_path: Path) -> str:
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    return str(data["project"]["version"])


def _select_built_wheel(dist_dir: Path, *, project_version: str) -> Path:
    wheel_name = f"themis_eval-{project_version}-py3-none-any.whl"
    wheel_path = dist_dir / wheel_name
    if not wheel_path.exists():
        raise SystemExit(
            f"Built wheel for version {project_version} not found under dist/. "
            f"Expected {wheel_name}. Run `uv build` first."
        )
    return wheel_path.resolve()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    dist_dir = Path("dist")
    project_version = _read_project_version(repo_root / "pyproject.toml")
    wheel_path = _select_built_wheel(dist_dir, project_version=project_version)
    with tempfile.TemporaryDirectory(prefix="themis-wheel-smoke-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        env_dir = Path(tmpdir) / "venv"
        subprocess.run(
            [
                "uv",
                "venv",
                "--python",
                sys.executable,
                str(env_dir),
            ],
            check=True,
        )
        bin_dir = _venv_bin_dir(env_dir)
        python_exe = bin_dir / _python_name()
        quickcheck_exe = bin_dir / _quickcheck_name()
        verification_env = _verification_env()
        site_packages_dir = _site_packages_dir(python_exe)

        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python_exe),
                str(wheel_path),
            ],
            check=True,
            cwd=tmpdir_path,
            env=verification_env,
        )
        subprocess.run(
            [
                str(python_exe),
                "-c",
                _import_check_snippet(site_packages_dir),
            ],
            check=True,
            cwd=tmpdir_path,
            env=verification_env,
        )
        subprocess.run(
            [str(quickcheck_exe), "--help"],
            check=True,
            stdout=subprocess.DEVNULL,
            cwd=tmpdir_path,
            env=verification_env,
        )

    print(f"Built wheel smoke test passed for {wheel_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
