from __future__ import annotations

import importlib
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO

from cyclopts.exceptions import CycloptsError

from themis.cli.app import app


@dataclass(frozen=True)
class CliResult:
    returncode: int
    stdout: str
    stderr: str


def run_cli(*args: str, env: dict[str, str] | None = None) -> CliResult:
    stdout = StringIO()
    stderr = StringIO()
    old_environ = os.environ.copy()
    old_sys_path = list(sys.path)
    old_datasets_module = sys.modules.get("datasets")

    if env:
        os.environ.update(env)
        pythonpath = env.get("PYTHONPATH")
        if pythonpath:
            for path in reversed(pythonpath.split(os.pathsep)):
                if path:
                    sys.path.insert(0, path)
            sys.modules.pop("datasets", None)
            importlib.invalidate_caches()

    try:
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = app(
                    args,
                    exit_on_error=False,
                    result_action="return_int_as_exit_code_else_zero",
                )
        except SystemExit as exc:
            if isinstance(exc.code, int):
                code = exc.code
            else:
                if exc.code is not None:
                    print(exc.code, file=stderr)
                code = 1
        except CycloptsError as exc:
            print(str(exc), file=stderr)
            code = 1
        else:
            code = int(result) if isinstance(result, int) else 0

        return CliResult(
            returncode=code,
            stdout=stdout.getvalue(),
            stderr=stderr.getvalue(),
        )
    finally:
        os.environ.clear()
        os.environ.update(old_environ)
        sys.path[:] = old_sys_path
        if old_datasets_module is None:
            sys.modules.pop("datasets", None)
        else:
            sys.modules["datasets"] = old_datasets_module
        importlib.invalidate_caches()
