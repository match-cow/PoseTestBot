"""Web-process settings and background-job ownership.

Route modules resolve the runner from the active Flask application.  The
module-level proxy keeps direct helper calls and older test doubles working
without making an individual blueprint own process-wide state.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flask import current_app, has_app_context

from posetestbot.jobs.runner import LocalJobRunner
from posetestbot.web.paths import APP_ROOT


RUNTIME_EXTENSION_KEY = "posetestbot.web_runtime"


def _env_bool(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class WebSettings:
    host: str
    port: int
    debug: bool
    job_root: Path
    cluster_url: str = "http://127.0.0.1:8765"
    cluster_token: str | None = None
    cluster_enabled: bool = False

    @classmethod
    def from_environment(cls) -> WebSettings:
        return cls(
            host=os.environ.get("POSETESTBOT_WEB_HOST", "0.0.0.0"),
            port=int(os.environ.get("POSETESTBOT_WEB_PORT", "5000")),
            debug=_env_bool("POSETESTBOT_WEB_DEBUG", default=False),
            job_root=APP_ROOT / "working_data" / "jobs",
            cluster_url=os.environ.get(
                "POSETESTBOT_CLUSTER_URL", "http://127.0.0.1:8765"
            ),
            cluster_token=os.environ.get("POSETESTBOT_CLUSTER_API_TOKEN") or None,
            cluster_enabled=_env_bool("POSETESTBOT_CLUSTER_ENABLED", default=False),
        )


@dataclass
class WebRuntime:
    settings: WebSettings
    job_runner: LocalJobRunner
    cluster_client: Any | None = None


_default_runtime: WebRuntime | None = None
_default_runtime_lock = threading.Lock()


def create_web_runtime(
    *,
    settings: WebSettings | None = None,
    job_runner: LocalJobRunner | None = None,
) -> WebRuntime:
    selected_settings = settings or WebSettings.from_environment()
    cluster_client = None
    if selected_settings.cluster_enabled and selected_settings.cluster_token:
        from posetestbot.cluster.client import ClusterControllerClient

        cluster_client = ClusterControllerClient(
            selected_settings.cluster_url,
            selected_settings.cluster_token,
        )
    return WebRuntime(
        settings=selected_settings,
        job_runner=job_runner or LocalJobRunner(selected_settings.job_root),
        cluster_client=cluster_client,
    )


def default_web_runtime() -> WebRuntime:
    global _default_runtime
    with _default_runtime_lock:
        if _default_runtime is None:
            _default_runtime = create_web_runtime()
        return _default_runtime


def get_web_runtime() -> WebRuntime:
    if has_app_context():
        runtime = current_app.extensions.get(RUNTIME_EXTENSION_KEY)
        if runtime is not None:
            return runtime
    return default_web_runtime()


def get_job_runner() -> LocalJobRunner:
    return get_web_runtime().job_runner


def get_cluster_client():
    runtime = get_web_runtime()
    if runtime.cluster_client is None:
        raise RuntimeError("Cluster controller token is not configured")
    return runtime.cluster_client


class _CurrentJobRunnerProxy:
    """Resolve job-runner operations through the active application runtime."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_job_runner(), name)


job_runner = _CurrentJobRunnerProxy()
