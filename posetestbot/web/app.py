"""PoseTestBot Flask application factory."""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, send_from_directory

from posetestbot.web.legacy import WEB_DEBUG, WEB_HOST, WEB_PORT
from posetestbot.web.legacy import app as legacy_api
from posetestbot.web.routes.overview import overview_bp
from posetestbot.web.routes.pages import pages_bp
from posetestbot.web.routes.sensors import sensors_bp


BRAND_ASSET_DIR = Path(__file__).resolve().parents[2] / "assets"
BRAND_LOGO_FILENAME = "cow200.png"


class _PreviewPollLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return '"GET /sensors/previews' not in message


def _install_preview_poll_log_filter() -> None:
    logger = logging.getLogger("werkzeug")
    if any(isinstance(item, _PreviewPollLogFilter) for item in logger.filters):
        return
    logger.addFilter(_PreviewPollLogFilter())


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
        static_url_path="/static",
    )

    @app.get("/assets/cow200.png")
    def brand_logo():
        return send_from_directory(
            BRAND_ASSET_DIR,
            BRAND_LOGO_FILENAME,
            max_age=86400,
            mimetype="image/png",
        )

    app.register_blueprint(pages_bp)
    app.register_blueprint(sensors_bp)
    app.register_blueprint(overview_bp)
    app.register_blueprint(legacy_api)
    _install_preview_poll_log_filter()
    return app


app = create_app()


if __name__ == "__main__":
    app.run(host=WEB_HOST, port=WEB_PORT, debug=WEB_DEBUG)
