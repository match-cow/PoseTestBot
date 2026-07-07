import os
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
    print("Started web interface on all interfaces by default")
    Timer(1, open_browser).start()
    print(f"Local browser will open at http://{WEB_BROWSER_HOST}:{WEB_PORT}/ shortly")
    print(f"Other devices can use http://<this-machine-ip>:{WEB_PORT}/")
    web_interface.wait()


if __name__ == "__main__":
    main()
