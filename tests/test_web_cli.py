from __future__ import annotations

import pytest

from posetestbot.web.cli import run_web_server


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
