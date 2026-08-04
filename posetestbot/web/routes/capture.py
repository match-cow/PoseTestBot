"""Capture planning, execution evidence, and capture-job APIs."""

from flask import Blueprint

from posetestbot.web import route_support


capture_bp = Blueprint("capture", __name__)


@capture_bp.get("/capture/jobs")
def list_capture_jobs():
    return route_support.list_capture_jobs()


@capture_bp.get("/capture/status")
def capture_execution_status():
    return route_support.capture_execution_status()


@capture_bp.post("/capture/jobs/<job_id>/stop")
def stop_capture_job(job_id: str):
    return route_support.stop_capture_job(job_id)


@capture_bp.route("/capture-plan", methods=["GET", "POST"])
def capture_plan_endpoint():
    return route_support.capture_plan_endpoint()


@capture_bp.route("/capture-plan/preflight", methods=["GET", "POST"])
def capture_plan_preflight_endpoint():
    return route_support.capture_plan_preflight_endpoint()


@capture_bp.route("/capture-plan/execution", methods=["GET", "POST"])
def capture_plan_execution_endpoint():
    return route_support.capture_plan_execution_endpoint()
