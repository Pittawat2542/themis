from __future__ import annotations

from pathlib import Path

import pytest

from themis.cli import main as cli_main


def test_serve_resolves_storage_path_and_invokes_uvicorn(tmp_path, monkeypatch):
    pytest.importorskip("uvicorn")
    import uvicorn

    calls: dict[str, object] = {}

    def _fake_create_app(*, storage_path):
        calls["storage_path"] = storage_path
        return object()

    def _fake_run(app_instance, **kwargs):
        calls["app_instance"] = app_instance
        calls["kwargs"] = kwargs

    monkeypatch.setattr("themis.server.create_app", _fake_create_app)
    monkeypatch.setattr(uvicorn, "run", _fake_run)

    exit_code = cli_main.serve(
        storage=str(tmp_path),
        host="127.0.0.1",
        port=9123,
        reload=True,
    )

    assert exit_code == 0
    assert Path(calls["storage_path"]) == tmp_path
    kwargs = calls["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 9123
    assert kwargs["reload"] is True
