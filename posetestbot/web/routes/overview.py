"""Workflow overview payload for the web UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from flask import Blueprint, jsonify, request

from posetestbot.io.artifacts import (
    BLENDERPROC_RENDER_PLAN,
    BOP_DIR,
    BOP_EXPORT_MANIFEST,
    CALIBRATION_CANDIDATES,
    CALIBRATION_OBSERVATIONS,
    CALIBRATION_PREFLIGHT_REPORT,
    CALIBRATION_PROFILES,
    CALIBRATION_SOLVER_REPORT,
    CALIBRATION_VALIDATION_REPORT,
    CAMERA_RECTIFICATION_REPORT,
    CAPTURE_EXECUTION_PLAN,
    CAPTURE_EXECUTION_REPORT,
    CAPTURE_EXECUTION_STATUS,
    CAPTURE_PLAN,
    CAPTURE_PLAN_PREFLIGHT_REPORT,
    HARDWARE_STATUS_REPORT,
    RUN_CONFIG,
    RUN_PREFLIGHT_REPORT,
    SYNC_QUALITY_REPORT,
)
from posetestbot.pipeline.recommendations import build_pipeline_recommendations
from posetestbot.pipeline.run_config import (
    load_run_config_for_run_root,
    sequence_plan_from_run_config,
)
from posetestbot.sync.calibration_policy import (
    resolve_calibration_profile_sync_policy,
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
        # Per-camera sync reports live below processed/synchronized/.  The
        # run-level quality report is the validated aggregate for the guided UI.
        "artifacts": [SYNC_QUALITY_REPORT],
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
        "artifacts": [
            CAMERA_RECTIFICATION_REPORT,
            BLENDERPROC_RENDER_PLAN,
            f"{BOP_DIR}/{BOP_EXPORT_MANIFEST}",
        ],
    },
    {"id": "jobs", "label": "Jobs", "artifacts": []},
]


def _calibration_sync_overview(
    root: Path,
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    pointer = (
        config.get("calibration_profile_selection")
        if isinstance(config, Mapping)
        else None
    )
    if pointer is None:
        return {
            "status": "not_configured",
            "sensors": [],
        }
    bundle_sha256 = (
        pointer.get("bundle_sha256") if isinstance(pointer, Mapping) else None
    )
    try:
        policy = resolve_calibration_profile_sync_policy(root)
        if policy is None:
            raise ValueError(
                "Run config does not bind a selected calibration timing policy"
            )
        display_names: dict[str, str] = {}
        capture = config.get("capture")
        if isinstance(capture, Mapping):
            sensors = capture.get("sensors")
            if isinstance(sensors, list):
                for sensor in sensors:
                    if not isinstance(sensor, Mapping):
                        continue
                    sensor_type = sensor.get("sensor_type")
                    device_id = sensor.get("device_id")
                    display_name = sensor.get("display_name")
                    if all(
                        isinstance(value, str) and value
                        for value in (sensor_type, device_id, display_name)
                    ):
                        display_names[f"{sensor_type}:{device_id}"] = display_name
        return {
            "status": "ready",
            "bundle_sha256": policy["bundle_sha256"],
            "sensors": [
                {
                    **dict(sensor),
                    "sensor_name": display_names.get(
                        str(sensor["sensor_key"]),
                        str(sensor["sensor_name"]),
                    ),
                }
                for sensor in policy["sensors"]
            ],
        }
    except Exception as exc:
        return {
            "status": "error",
            "bundle_sha256": bundle_sha256,
            "sensors": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


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
    exists = path.is_file()
    if not exists:
        return {"path": relative_path, "exists": False, "status": None}
    if path.is_symlink():
        return {"path": relative_path, "exists": True, "status": "invalid"}
    value = _json_if_present(path)
    if value is None:
        return {"path": relative_path, "exists": True, "status": "invalid"}
    status = _validated_artifact_status(relative_path, value)
    return {
        "path": relative_path,
        "exists": True,
        "status": status,
    }


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _validated_artifact_status(
    relative_path: str, value: Mapping[str, Any]
) -> str:
    """Return an explicit guided-workflow state for durable JSON evidence."""

    declared = value.get("overall_status", value.get("status"))
    if relative_path == CAPTURE_EXECUTION_REPORT:
        if value.get("schema_version") != "capture_execution_report.v1":
            return "invalid"
        return str(declared) if isinstance(declared, str) else "invalid"
    if relative_path == SYNC_QUALITY_REPORT:
        if (
            value.get("schema_version") != "sync_quality_report.v2"
            or not _positive_int(value.get("sensor_count"))
            or not isinstance(value.get("sensors"), list)
            or len(value["sensors"]) != value["sensor_count"]
            or not isinstance(value.get("checks"), list)
        ):
            return "invalid"
        return str(declared) if declared in {"ok", "warning", "error"} else "invalid"
    if relative_path == CALIBRATION_PROFILES:
        profiles = value.get("profiles")
        if (
            value.get("schema_version") not in {"calibration.v1", "calibration.v2"}
            or not isinstance(profiles, list)
            or not profiles
            or not all(isinstance(profile, Mapping) for profile in profiles)
        ):
            return "invalid"
        return (
            "complete"
            if all(profile.get("status") == "valid" for profile in profiles)
            else "needs_validation"
        )
    if relative_path == CAMERA_RECTIFICATION_REPORT:
        if (
            value.get("schema_version") != "camera_rectification.v1"
            or not _positive_int(value.get("sensor_count"))
            or not _positive_int(value.get("frame_count"))
            or not isinstance(value.get("sensors"), list)
            or len(value["sensors"]) != value["sensor_count"]
        ):
            return "invalid"
        return "complete"
    if relative_path == f"{BOP_DIR}/{BOP_EXPORT_MANIFEST}":
        exports = value.get("exports")
        if (
            value.get("schema_version")
            not in {"bop_export_manifest.v3", "bop_export_manifest.v4"}
            or not isinstance(exports, list)
            or not exports
            or not all(isinstance(item, Mapping) for item in exports)
        ):
            return "invalid"
        return "complete"
    return str(declared) if isinstance(declared, str) else "present"


def _section_status(chips: list[Mapping[str, Any]]) -> str:
    if not chips:
        return "available"
    existing = [chip for chip in chips if chip.get("exists")]
    if not existing:
        return "pending"
    if any(
        chip.get("status")
        in {"error", "failed", "blocked", "invalid", "canceled", "cancelled"}
        for chip in existing
    ):
        return "blocked"
    if any(
        chip.get("status")
        in {"needs_validation", "queued", "running", "in_progress", "waiting"}
        for chip in existing
    ):
        return "in_progress"
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
        "sync_run": [SYNC_QUALITY_REPORT],
        "sync_quality": [SYNC_QUALITY_REPORT],
        "calibration_preflight": [CALIBRATION_PREFLIGHT_REPORT],
        "calibration_observations": [CALIBRATION_OBSERVATIONS],
        "calibration_candidates": [CALIBRATION_CANDIDATES],
        "calibration_solver": [CALIBRATION_SOLVER_REPORT],
        "calibration_validation": [CALIBRATION_VALIDATION_REPORT],
        "camera_rectification": [CAMERA_RECTIFICATION_REPORT],
        "blenderproc_render": [BLENDERPROC_RENDER_PLAN],
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
            "calibration_sync": _calibration_sync_overview(root, config),
            "sidebar": _section_summaries(root),
            "steps": _sequence_steps(config, root),
            "recommendations": recommendations,
            "recommendation_error": recommendation_error,
        }
    )
