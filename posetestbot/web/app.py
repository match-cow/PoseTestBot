"""PoseTestBot Flask application factory."""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, send_from_directory

from posetestbot.web.legacy import app as legacy_api
from posetestbot.web.routes.monitoring import monitoring_bp
from posetestbot.web.routes.overview import overview_bp
from posetestbot.web.routes.pages import pages_bp
from posetestbot.web.routes.sensors import sensors_bp
from posetestbot.web.routes.ui import ui_bp
from posetestbot.web.security import install_request_security


BRAND_ASSET_DIR = Path(__file__).resolve().parent / "static"
CELL_ASSET_DIR = BRAND_ASSET_DIR / "cell"


class _PreviewPollLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        successful_poll = message.rstrip().endswith('" 200 -')
        noisy_preview_get = any(
            marker in message
            for marker in (
                '"GET /sensors/previews',
            )
        )
        return not (successful_poll and noisy_preview_get)


def _install_preview_poll_log_filter() -> None:
    logger = logging.getLogger("werkzeug")
    if any(isinstance(item, _PreviewPollLogFilter) for item in logger.filters):
        return
    logger.addFilter(_PreviewPollLogFilter())


def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder="static",
        static_url_path="/static",
    )
    install_request_security(app)

    @app.get("/assets/cow_dark.png", defaults={"filename": "cow_dark.png"})
    @app.get("/assets/cow_light.png", defaults={"filename": "cow_light.png"})
    @app.get(
        "/assets/cow_favicon.png",
        defaults={"filename": "cow_favicon.png"},
    )
    def brand_asset(filename: str):
        return send_from_directory(
            BRAND_ASSET_DIR,
            filename,
            max_age=86400,
            mimetype="image/png",
        )

    @app.get("/assets/cell/template_HRI_LBR_all_center_v2.svg")
    def hri_cell_template():
        return send_from_directory(
            CELL_ASSET_DIR,
            "template_HRI_LBR_all_center_v2.svg",
            max_age=86400,
            mimetype="image/svg+xml",
            conditional=True,
        )

    app.register_blueprint(pages_bp)
    app.register_blueprint(sensors_bp)
    app.register_blueprint(monitoring_bp)
    app.register_blueprint(overview_bp)
    app.register_blueprint(ui_bp)
    app.register_blueprint(legacy_api)
    _install_preview_poll_log_filter()
    return app


app = create_app()


if __name__ == "__main__":
    from posetestbot.web.cli import run_web_server
    from posetestbot.web.legacy import job_runner

    run_web_server(app, job_runner)
