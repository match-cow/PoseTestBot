"""Installed command-line entry point for the PoseTestBot web UI."""

from __future__ import annotations

import signal
import threading


def run_web_server(
    app_instance=None,
    runner=None,
) -> None:
    """Run Flask and always release jobs that may own lab hardware."""

    if app_instance is None or runner is None:
        from posetestbot.web.app import app
        from posetestbot.web.legacy import job_runner

        app_instance = app if app_instance is None else app_instance
        runner = job_runner if runner is None else runner
    from posetestbot.web.legacy import WEB_DEBUG, WEB_HOST, WEB_PORT

    previous_sigterm = None
    if threading.current_thread() is threading.main_thread():
        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def request_shutdown(signum, _frame) -> None:
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, request_shutdown)
    try:
        app_instance.run(
            host=WEB_HOST,
            port=WEB_PORT,
            debug=WEB_DEBUG,
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
