"""PoseTestBot Flask application factory."""

from __future__ import annotations

from flask import Flask

from posetestbot.web.legacy import WEB_DEBUG, WEB_HOST, WEB_PORT
from posetestbot.web.legacy import app as legacy_api
from posetestbot.web.routes.overview import overview_bp
from posetestbot.web.routes.pages import pages_bp
from posetestbot.web.routes.sensors import sensors_bp


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
        static_url_path="/static",
    )
    app.register_blueprint(pages_bp)
    app.register_blueprint(sensors_bp)
    app.register_blueprint(overview_bp)
    app.register_blueprint(legacy_api)
    return app


app = create_app()


if __name__ == "__main__":
    app.run(host=WEB_HOST, port=WEB_PORT, debug=WEB_DEBUG)
