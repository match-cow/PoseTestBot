"""HTML page routes."""

from __future__ import annotations

from flask import Blueprint, render_template

from posetestbot.config import DEFAULT_ROBOT_PORT, LAB_ROBOT_IP
from posetestbot.pipeline.sequences import PIPELINE_SEQUENCES, list_pipeline_sequences


pages_bp = Blueprint("pages", __name__)


@pages_bp.get("/")
def index():
    return render_template(
        "index.html",
        sequences=list_pipeline_sequences(PIPELINE_SEQUENCES),
        robot_control_defaults={
            "robot_ip": LAB_ROBOT_IP,
            "robot_port": DEFAULT_ROBOT_PORT,
        },
    )
