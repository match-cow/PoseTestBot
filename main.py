import os
import signal
import subprocess
import webbrowser
from threading import Timer

WEB_BROWSER_HOST = os.environ.get("POSETESTBOT_BROWSER_HOST", "127.0.0.1")
WEB_PORT = os.environ.get("POSETESTBOT_WEB_PORT", "5000")


def open_browser():
    webbrowser.open_new(f"http://{WEB_BROWSER_HOST}:{WEB_PORT}/")


def main():
    print("Starting web interface")
    web_interface = subprocess.Popen(["uv", "run", "python", "web_interface.py"])
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def request_shutdown(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, request_shutdown)
    print("Started web interface on all interfaces by default")
    browser_timer = Timer(1, open_browser)
    browser_timer.start()
    print(f"Local browser will open at http://{WEB_BROWSER_HOST}:{WEB_PORT}/ shortly")
    print(f"Other devices can use http://<this-machine-ip>:{WEB_PORT}/")
    try:
        web_interface.wait()
    finally:
        browser_timer.cancel()
        if web_interface.poll() is None:
            web_interface.terminate()
            try:
                web_interface.wait(timeout=10)
            except subprocess.TimeoutExpired:
                web_interface.kill()
                web_interface.wait()
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
