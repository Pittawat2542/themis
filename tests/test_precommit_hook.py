from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path


def _run(
    args: list[str], *, cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _init_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init", "-q"], cwd=repo)
    _run(["git", "config", "user.name", "Test User"], cwd=repo)
    _run(["git", "config", "user.email", "test@example.com"], cwd=repo)

    scripts_dir = repo / "scripts" / "ci"
    scripts_dir.mkdir(parents=True)
    source_script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "ci"
        / "check_staged_python.sh"
    )
    shutil.copy2(source_script, scripts_dir / "check_staged_python.sh")
    (scripts_dir / "check_staged_python.sh").chmod(
        (scripts_dir / "check_staged_python.sh").stat().st_mode | stat.S_IXUSR
    )
    return repo, scripts_dir / "check_staged_python.sh"


def _fake_uv(bin_dir: Path, log_path: Path) -> None:
    bin_dir.mkdir()
    script = bin_dir / "uv"
    script.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$FAKE_UV_LOG"
for arg in "$@"; do
  if [[ -f "$arg" ]] && grep -q 'BROKEN' "$arg"; then
    echo "saw broken content" >&2
    exit 1
  fi
done
""",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    if os.name == "nt":
        windows_script = bin_dir / "uv.bat"
        windows_script.write_text(
            "@echo off\n"
            "python -c \"from pathlib import Path; import os, sys; "
            "args = sys.argv[1:]; "
            "log = Path(os.environ['FAKE_UV_LOG']); "
            "with log.open('a', encoding='utf-8') as fh: fh.write(' '.join(args) + '\\n'); "
            "bad = False; "
            "for arg in args:\n"
            "    p = Path(arg)\n"
            "    if p.is_file() and 'BROKEN' in p.read_text(encoding='utf-8'):\n"
            "        bad = True\n"
            "if bad:\n"
            "    print('saw broken content', file=sys.stderr)\n"
            "    sys.exit(1)\"\n",
            encoding="utf-8",
        )


def _script_command(script: Path) -> list[str]:
    if os.name != "nt":
        return [str(script)]
    bash = shutil.which("bash")
    assert bash is not None, "bash is required to test the pre-commit hook on Windows"
    return [bash, str(script)]


def test_check_staged_python_skips_when_no_python_files_are_staged(
    tmp_path: Path,
) -> None:
    repo, script = _init_repo(tmp_path)
    log_path = tmp_path / "uv.log"
    _fake_uv(tmp_path / "bin", log_path)

    (repo / "README.md").write_text("docs\n", encoding="utf-8")
    _run(["git", "add", "README.md"], cwd=repo)

    env = dict(os.environ)
    env["PATH"] = f"{tmp_path / 'bin'}{os.pathsep}{env['PATH']}"
    env["FAKE_UV_LOG"] = str(log_path)
    result = _run(_script_command(script), cwd=repo, env=env)

    assert result.returncode == 0
    assert not log_path.exists()


def test_check_staged_python_uses_staged_snapshot_not_worktree(tmp_path: Path) -> None:
    repo, script = _init_repo(tmp_path)
    log_path = tmp_path / "uv.log"
    _fake_uv(tmp_path / "bin", log_path)

    tracked = repo / "sample.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    _run(["git", "add", "sample.py"], cwd=repo)
    _run(["git", "commit", "-m", "seed"], cwd=repo)

    tracked.write_text("value = 2\n", encoding="utf-8")
    _run(["git", "add", "sample.py"], cwd=repo)
    tracked.write_text("BROKEN = (\n", encoding="utf-8")

    env = dict(os.environ)
    env["PATH"] = f"{tmp_path / 'bin'}{os.pathsep}{env['PATH']}"
    env["FAKE_UV_LOG"] = str(log_path)
    result = _run(_script_command(script), cwd=repo, env=env)

    assert result.returncode == 0, result.stderr
    logged = log_path.read_text(encoding="utf-8")
    assert "ruff check" in logged
    assert "py_compile" in logged
    assert "mypy" in logged
    assert "sample.py" in logged
