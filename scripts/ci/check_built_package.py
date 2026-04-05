from __future__ import annotations

import subprocess
import sys
import tempfile
import tarfile
import zipfile
from pathlib import Path
import tomllib


EXCLUDED_ARCHIVE_PATHS = (
    "tests/",
    "docs/",
    "examples/",
    "site/",
    "dist/",
    "queue/",
    "runs/",
    ".agent/",
    ".agents/",
)
EXCLUDED_ARCHIVE_NAMES = {
    ".coverage",
    "coverage.json",
    "coverage.xml",
    "REQUIREMENTS.md",
    "V4_PLAN.md",
    "skills-lock.json",
    "uv.lock",
}


def _run(*args: str, cwd: Path | None = None) -> None:
    result = subprocess.run(args, cwd=cwd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _assert_archive_contents(path: Path, names: list[str]) -> None:
    for name in names:
        normalized = name
        if path.suffixes[-2:] == [".tar", ".gz"] and "/" in name:
            normalized = name.split("/", 1)[1]
        if any(normalized.startswith(prefix) for prefix in EXCLUDED_ARCHIVE_PATHS):
            raise SystemExit(f"{path.name} unexpectedly contains excluded path: {normalized}")
        if normalized in EXCLUDED_ARCHIVE_NAMES:
            raise SystemExit(f"{path.name} unexpectedly contains excluded file: {normalized}")


def _inspect_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        _assert_archive_contents(path, names)
        if "themis/__init__.py" not in names:
            raise SystemExit(f"{path.name} does not contain themis/__init__.py")
        metadata_name = next(
            (name for name in names if name.endswith(".dist-info/METADATA")),
            None,
        )
        entry_points_name = next(
            (name for name in names if name.endswith(".dist-info/entry_points.txt")),
            None,
        )
        if metadata_name is None or entry_points_name is None:
            raise SystemExit(f"{path.name} is missing dist-info metadata")
        metadata = archive.read(metadata_name).decode()
        entry_points = archive.read(entry_points_name).decode()
        if "Version: 4.0.0" not in metadata:
            raise SystemExit(f"{path.name} metadata does not contain the final release version")
        if "themis = themis.cli:main" not in entry_points:
            raise SystemExit(f"{path.name} is missing the themis CLI entry point")


def _inspect_sdist(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
        _assert_archive_contents(path, names)
        if not any(name.endswith("/themis/__init__.py") for name in names):
            raise SystemExit(f"{path.name} does not contain the themis package")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    dist_dir = repo_root / "dist"
    project_version = tomllib.loads(
        (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    wheels = sorted(dist_dir.glob(f"themis_eval-{project_version}-*.whl"))
    sdists = sorted(dist_dir.glob(f"themis_eval-{project_version}.tar.gz"))
    if not wheels:
        print(
            f"no built wheel found in dist/ for version {project_version}",
            file=sys.stderr,
        )
        return 1
    if not sdists:
        print(
            f"no built sdist found in dist/ for version {project_version}",
            file=sys.stderr,
        )
        return 1

    wheel = wheels[-1]
    sdist = sdists[-1]
    _inspect_wheel(wheel)
    _inspect_sdist(sdist)

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

    print(f"distribution smoke test passed for {wheel.name} and {sdist.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
