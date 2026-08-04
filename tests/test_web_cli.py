from __future__ import annotations

from pathlib import Path

import pytest

from posetestbot.web.app import create_app
from posetestbot.web.cli import run_web_server
from posetestbot.web.runtime import (
    RUNTIME_EXTENSION_KEY,
    WebRuntime,
    WebSettings,
    get_web_runtime,
)


def test_web_server_shutdown_releases_runner_jobs() -> None:
    class FakeApp:
        def run(self, **_kwargs) -> None:
            raise RuntimeError("server stopped")

    class FakeRunner:
        def __init__(self) -> None:
            self.shutdown_calls = 0

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    runner = FakeRunner()

    with pytest.raises(RuntimeError, match="server stopped"):
        run_web_server(FakeApp(), runner)

    assert runner.shutdown_calls == 1


def test_app_factory_injects_runtime_runner_and_server_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRunner:
        def __init__(self) -> None:
            self.shutdown_calls = 0

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    runner = FakeRunner()
    settings = WebSettings(
        host="127.0.0.8",
        port=54321,
        debug=True,
        job_root=tmp_path / "jobs",
    )
    runtime = WebRuntime(settings=settings, job_runner=runner)  # type: ignore[arg-type]
    app = create_app(runtime=runtime)
    calls: list[dict] = []

    def stop_server(**kwargs) -> None:
        calls.append(kwargs)
        raise RuntimeError("server stopped")

    monkeypatch.setattr(app, "run", stop_server)

    assert app.extensions[RUNTIME_EXTENSION_KEY] is runtime
    with app.app_context():
        assert get_web_runtime() is runtime

    with pytest.raises(RuntimeError, match="server stopped"):
        run_web_server(app)

    assert calls == [
        {
            "host": "127.0.0.8",
            "port": 54321,
            "debug": True,
            "use_reloader": False,
        }
    ]
    assert runner.shutdown_calls == 1
