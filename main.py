import subprocess
import webbrowser
from threading import Timer
import sys

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

def main():
    print("starting server")
    web_interface = subprocess.Popen([sys.executable, "web_interface.py"])
    print("Started web interface")
    Timer(1, open_browser).start()
    print("Browser will open in new tab shortly")
    web_interface.wait()


if __name__ == "__main__":
    main()
