"""Pipeline metadata, run configuration, preflight, and queueing APIs."""

from flask import Blueprint

from posetestbot.web import route_support


pipeline_bp = Blueprint("pipeline", __name__)


@pipeline_bp.get("/pipeline/stages")
def pipeline_stages():
    return route_support.pipeline_stages()


@pipeline_bp.get("/pipeline/stages/<stage_id>")
def pipeline_stage(stage_id: str):
    return route_support.pipeline_stage(stage_id)


@pipeline_bp.get("/pipeline/sequences")
def pipeline_sequences():
    return route_support.pipeline_sequences()


@pipeline_bp.get("/pipeline/workflows")
def pipeline_workflows():
    return route_support.pipeline_workflows()


@pipeline_bp.get("/pipeline/sequences/<sequence_id>")
def pipeline_sequence(sequence_id: str):
    return route_support.pipeline_sequence(sequence_id)


@pipeline_bp.get("/pipeline/recommendations")
def pipeline_recommendations():
    return route_support.pipeline_recommendations()


@pipeline_bp.route("/run-config", methods=["GET", "POST"])
def run_config():
    return route_support.run_config()


@pipeline_bp.post("/pipeline/run")
def run_pipeline_stage():
    return route_support.run_pipeline_stage()


@pipeline_bp.route("/pipeline/preflight", methods=["GET", "POST"])
def pipeline_preflight():
    return route_support.pipeline_preflight()


@pipeline_bp.post("/pipeline/run-sequence")
def run_pipeline_sequence():
    return route_support.run_pipeline_sequence()


@pipeline_bp.post("/pipeline/run-config")
def run_pipeline_from_config():
    return route_support.run_pipeline_from_config()
