"""Compatibility entrypoint for the PoseTestBot Flask web UI."""

from posetestbot.web.app import WEB_DEBUG, WEB_HOST, WEB_PORT, app


if __name__ == "__main__":
    app.run(host=WEB_HOST, port=WEB_PORT, debug=WEB_DEBUG)
