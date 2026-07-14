import os
import webbrowser
from threading import Timer

from posetestbot.web.cli import run_web_server

WEB_BROWSER_HOST = os.environ.get("POSETESTBOT_BROWSER_HOST", "127.0.0.1")
WEB_PORT = os.environ.get("POSETESTBOT_WEB_PORT", "5000")


def open_browser():
    webbrowser.open_new(f"http://{WEB_BROWSER_HOST}:{WEB_PORT}/")


def main():
    print("Starting web interface")
    print("Started web interface on all interfaces by default")
    browser_timer = Timer(1, open_browser)
    browser_timer.start()
    print(f"Local browser will open at http://{WEB_BROWSER_HOST}:{WEB_PORT}/ shortly")
    print(f"Other devices can use http://<this-machine-ip>:{WEB_PORT}/")
    try:
        run_web_server()
    finally:
        browser_timer.cancel()


if __name__ == "__main__":
    main()
