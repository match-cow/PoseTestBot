"""Bundled operator-console page route."""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, send_from_directory


pages_bp = Blueprint("pages", __name__)
UI_BUILD_DIR = Path(__file__).resolve().parents[1] / "static" / "ui"


@pages_bp.get("/")
def index():
    index_path = UI_BUILD_DIR / "index.html"
    if not index_path.is_file():
        return (
            jsonify(
                {
                    "output": (
                        "The bundled operator console is missing. Run "
                        "`bun run build` in frontend/."
                    )
                }
            ),
            503,
        )
    return send_from_directory(UI_BUILD_DIR, "index.html", max_age=0)
