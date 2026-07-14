"""Installed command-line entry point for the PoseTestBot web UI."""

from __future__ import annotations

import signal
import threading

from posetestbot.web.app import WEB_DEBUG, WEB_HOST, WEB_PORT, app
from posetestbot.web.legacy import job_runner


def run_web_server(
    app_instance=app,
    runner=job_runner,
) -> None:
    """Run Flask and always release jobs that may own lab hardware."""

    previous_sigterm = None
    if threading.current_thread() is threading.main_thread():
        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def request_shutdown(signum, _frame) -> None:
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, request_shutdown)
    try:
        app_instance.run(host=WEB_HOST, port=WEB_PORT, debug=WEB_DEBUG)
    finally:
        runner.shutdown()
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)


def main() -> None:
    run_web_server()


if __name__ == "__main__":
    main()
