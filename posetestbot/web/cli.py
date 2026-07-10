"""Installed command-line entry point for the PoseTestBot web UI."""

from __future__ import annotations

from posetestbot.web.app import WEB_DEBUG, WEB_HOST, WEB_PORT, app


def main() -> None:
    app.run(host=WEB_HOST, port=WEB_PORT, debug=WEB_DEBUG)


if __name__ == "__main__":
    main()
