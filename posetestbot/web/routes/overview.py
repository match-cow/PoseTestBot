"""Workflow overview payload for the web UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from flask import Blueprint, jsonify, request

from posetestbot.io.artifacts import (
    BOP_DIR,
    BOP_EXPORT_MANIFEST,
    CALIBRATION_CANDIDATES,
    CALIBRATION_OBSERVATIONS,
    CALIBRATION_PREFLIGHT_REPORT,
    CALIBRATION_PROFILES,
    CALIBRATION_SOLVER_REPORT,
    CALIBRATION_VALIDATION_REPORT,
    CAPTURE_EXECUTION_PLAN,
    CAPTURE_EXECUTION_REPORT,
    CAPTURE_EXECUTION_STATUS,
    CAPTURE_PLAN,
    CAPTURE_PLAN_PREFLIGHT_REPORT,
    HARDWARE_STATUS_REPORT,
    RUN_CONFIG,
    RUN_PREFLIGHT_REPORT,
    SYNC_QUALITY_REPORT,
    SYNC_REPORT,
)
from posetestbot.pipeline.recommendations import build_pipeline_recommendations
from posetestbot.pipeline.run_config import (
    load_run_config_for_run_root,
    sequence_plan_from_run_config,
)


overview_bp = Blueprint("overview", __name__)


WORKFLOW_SECTIONS = [
    {
        "id": "overview",
        "label": "Overview",
        "artifacts": [RUN_CONFIG, RUN_PREFLIGHT_REPORT],
    },
    {
        "id": "sensors",
        "label": "Sensors",
        "artifacts": [HARDWARE_STATUS_REPORT],
    },
    {
        "id": "run_setup",
        "label": "Run Setup",
        "artifacts": [RUN_CONFIG, CAPTURE_PLAN],
    },
    {
        "id": "preflight",
        "label": "Preflight",
        "artifacts": [
            RUN_PREFLIGHT_REPORT,
            CAPTURE_PLAN_PREFLIGHT_REPORT,
            HARDWARE_STATUS_REPORT,
        ],
    },
    {
        "id": "capture",
        "label": "Capture",
        "artifacts": [
            CAPTURE_PLAN,
            CAPTURE_EXECUTION_PLAN,
            CAPTURE_EXECUTION_STATUS,
            CAPTURE_EXECUTION_REPORT,
        ],
    },
    {
        "id": "sync",
        "label": "Sync",
        "artifacts": [SYNC_REPORT, SYNC_QUALITY_REPORT],
    },
    {
        "id": "calibration",
        "label": "Calibration",
        "artifacts": [
            CALIBRATION_PREFLIGHT_REPORT,
            CALIBRATION_OBSERVATIONS,
            CALIBRATION_CANDIDATES,
            CALIBRATION_SOLVER_REPORT,
            CALIBRATION_VALIDATION_REPORT,
            CALIBRATION_PROFILES,
        ],
    },
    {
        "id": "bop",
        "label": "BOP Export",
        "artifacts": [f"{BOP_DIR}/{BOP_EXPORT_MANIFEST}"],
    },
    {"id": "jobs", "label": "Jobs", "artifacts": []},
]


def _json_if_present(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with open(path, "r") as f:
            value = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _artifact_chip(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    value = _json_if_present(path)
    status = None
    if isinstance(value, Mapping):
        status = value.get("overall_status", value.get("status"))
    return {
        "path": relative_path,
        "exists": path.is_file(),
        "status": status if isinstance(status, str) else None,
    }


def _section_status(chips: list[Mapping[str, Any]]) -> str:
    if not chips:
        return "available"
    existing = [chip for chip in chips if chip.get("exists")]
    if not existing:
        return "pending"
    if any(chip.get("status") in {"error", "failed", "blocked"} for chip in existing):
        return "blocked"
    if len(existing) == len(chips):
        return "complete"
    return "in_progress"


def _section_summaries(root: Path) -> list[dict[str, Any]]:
    sections = []
    for section in WORKFLOW_SECTIONS:
        chips = [_artifact_chip(root, path) for path in section["artifacts"]]
        sections.append(
            {
                "id": section["id"],
                "label": section["label"],
                "status": _section_status(chips),
                "artifacts": chips,
            }
        )
    return sections


def _sequence_steps(config: Mapping[str, Any] | None, root: Path) -> list[dict[str, Any]]:
    if not config:
        return []
    try:
        plan = sequence_plan_from_run_config(config)
    except ValueError:
        return []
    steps = []
    artifact_lookup = {
        "run_preflight": [RUN_PREFLIGHT_REPORT],
        "capture_plan": [CAPTURE_PLAN],
        "capture_plan_preflight": [CAPTURE_PLAN_PREFLIGHT_REPORT],
        "capture_execution_plan": [CAPTURE_EXECUTION_PLAN],
        "capture_execution": [CAPTURE_EXECUTION_REPORT],
        "sync_run": [SYNC_REPORT],
        "sync_quality": [SYNC_QUALITY_REPORT],
        "calibration_preflight": [CALIBRATION_PREFLIGHT_REPORT],
        "calibration_observations": [CALIBRATION_OBSERVATIONS],
        "calibration_candidates": [CALIBRATION_CANDIDATES],
        "calibration_solver": [CALIBRATION_SOLVER_REPORT],
        "calibration_validation": [CALIBRATION_VALIDATION_REPORT],
        "bop_export": [f"{BOP_DIR}/{BOP_EXPORT_MANIFEST}"],
    }
    for index, step in enumerate(plan.steps, start=1):
        artifact_paths = artifact_lookup.get(step.stage_id, [])
        chips = [_artifact_chip(root, path) for path in artifact_paths]
        steps.append(
            {
                "index": index,
                "id": step.id,
                "stage_id": step.stage_id,
                "label": step.stage_label,
                "description": step.stage_id,
                "status": _section_status(chips) if chips else "available",
                "artifacts": chips,
                "resources": list(step.resources),
                "command": list(step.command),
            }
        )
    return steps


@overview_bp.get("/ui/overview")
def ui_overview():
    run_root = request.args.get("run_root")
    if not run_root:
        return jsonify({"output": "Missing run_root"}), 400
    root = Path(run_root)
    config = None
    config_error = None
    if root.exists():
        try:
            config = load_run_config_for_run_root(root)
        except FileNotFoundError:
            config_error = None
        except ValueError as exc:
            config_error = str(exc)
    else:
        config_error = f"Run root not found: {root}"

    recommendations = []
    recommendation_error = None
    if root.exists():
        try:
            recommendations = build_pipeline_recommendations(root).get(
                "recommendations", []
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            recommendation_error = str(exc)

    return jsonify(
        {
            "run_root": root.as_posix(),
            "config": config,
            "config_error": config_error,
            "sidebar": _section_summaries(root),
            "steps": _sequence_steps(config, root),
            "recommendations": recommendations,
            "recommendation_error": recommendation_error,
        }
    )
