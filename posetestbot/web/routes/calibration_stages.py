"""Compatibility calibration-stage report and write APIs."""

from flask import Blueprint

from posetestbot.web import route_support


calibration_stages_bp = Blueprint("calibration_stages", __name__)


@calibration_stages_bp.route("/calibration/preflight", methods=["GET", "POST"])
def calibration_preflight_endpoint():
    return route_support.calibration_preflight_endpoint()


@calibration_stages_bp.route("/calibration/observations", methods=["GET", "POST"])
def calibration_observations_endpoint():
    return route_support.calibration_observations_endpoint()


@calibration_stages_bp.route("/calibration/candidates", methods=["GET", "POST"])
def calibration_candidates_endpoint():
    return route_support.calibration_candidates_endpoint()


@calibration_stages_bp.route("/calibration/solver", methods=["GET", "POST"])
def calibration_solver_endpoint():
    return route_support.calibration_solver_endpoint()


@calibration_stages_bp.route("/calibration/validation", methods=["GET", "POST"])
def calibration_validation_endpoint():
    return route_support.calibration_validation_endpoint()
