"""Installed command-line entry point for the PoseTestBot web UI."""

from __future__ import annotations

import signal
import threading


def run_web_server(
    app_instance=None,
    runner=None,
) -> None:
    """Run Flask and always release jobs that may own lab hardware."""

    if app_instance is None:
        from posetestbot.web.app import app

        app_instance = app

    from posetestbot.web.runtime import WebSettings, get_web_runtime

    settings = None
    app_context = getattr(app_instance, "app_context", None)
    if app_context is not None:
        with app_context():
            runtime = get_web_runtime()
            settings = runtime.settings
            if runner is None:
                runner = runtime.job_runner
    if runner is None:
        from posetestbot.web.app import app

        app_instance = app
        with app_instance.app_context():
            runtime = get_web_runtime()
            settings = runtime.settings
            runner = runtime.job_runner
    if settings is None:
        settings = WebSettings.from_environment()

    previous_sigterm = None
    if threading.current_thread() is threading.main_thread():
        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def request_shutdown(signum, _frame) -> None:
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, request_shutdown)
    try:
        app_instance.run(
            host=settings.host,
            port=settings.port,
            debug=settings.debug,
            use_reloader=False,
        )
    finally:
        runner.shutdown()
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)


def main() -> None:
    run_web_server()


if __name__ == "__main__":
    main()
