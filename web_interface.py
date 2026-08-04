"""Compatibility entrypoint for the PoseTestBot Flask web UI."""

from posetestbot.web.app import app as app
from posetestbot.web.cli import main


if __name__ == "__main__":
    main()
