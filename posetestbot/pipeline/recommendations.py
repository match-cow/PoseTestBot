"""Artifact-driven next-step recommendations for acquisition-focused runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from posetestbot.io.artifacts import (
    ARUCO_COVERAGE_REPORT,
    ARUCO_POSE_ESTIMATION,
    BLENDERPROC_RENDER_PLAN,
    BOP_DIR,
    BOP_EXPORT_MANIFEST,
    BOP_TARGETS_BOP19,
    CALIBRATION_CANDIDATES,
    CALIBRATION_OBSERVATIONS,
    CALIBRATION_PREFLIGHT_REPORT,
    CALIBRATION_PROFILES,
    CALIBRATION_SOLVER_REPORT,
    CALIBRATION_TARGET_POSE_ESTIMATION,
    CALIBRATION_VALIDATION_REPORT,
    CAPTURE_EXECUTION_PLAN,
    CAPTURE_EXECUTION_REPORT,
    CAPTURE_PLAN,
    CAPTURE_PLAN_PREFLIGHT_REPORT,
    CHARUCO_POSE_ESTIMATION,
    CHECKERBOARD_POSE_ESTIMATION,
    PIPELINE_SEQUENCE_PLAN,
    PROCESSED_DIR,
    RAW_ROBOT_EE_POSES,
    REWRITE_GATE_REPORT,
    REWRITE_STATUS_REPORT,
    RUN_CONFIG,
    RUN_PREFLIGHT_REPORT,
    SYNC_QUALITY_REPORT,
    SYNCHRONIZED_DIR,
)
from posetestbot.pipeline.preflight import run_preflight_queue_summary
from posetestbot.pipeline.rewrite_gate import build_rewrite_status_report
from posetestbot.pipeline.run_config import (
    sequence_plan_from_run_config,
    validate_run_config,
)
from posetestbot.pipeline.stages import build_pipeline_job


SCHEMA_VERSION = "pipeline_recommendations.v1"
RAW_SENSOR_PREFIXES = ("realsense_", "luxonis_", "zed_2i_")
CALIBRATION_TARGET_POSE_ARTIFACTS = (
    ARUCO_POSE_ESTIMATION,
    CHARUCO_POSE_ESTIMATION,
    CHECKERBOARD_POSE_ESTIMATION,
    CALIBRATION_TARGET_POSE_ESTIMATION,
)


@dataclass(frozen=True)
class PipelineRecommendation:
    """One operator-facing suggestion for moving a run to the next artifact."""

    id: str
    label: str
    description: str
    reason: str
    priority: int
    action_type: str
    command: list[str] = field(default_factory=list)
    endpoint: str | None = None
    method: str | None = None
    stage_id: str | None = None
    sequence_id: str | None = None
    expected_artifacts: list[str] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)
    blocks_on: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_if_present(path: Path) -> object | None:
    if not path.is_file():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _status_summary(
    root: Path,
    artifact_name: str,
    *,
    ready_key: str,
    missing_blocker: str,
    invalid_blocker: str,
    failed_blocker: str,
) -> dict[str, Any]:
    path = root / artifact_name
    if not path.is_file():
        return {"overall_status": None, ready_key: False, "blocker": missing_blocker}
    value = _json_if_present(path)
    if not isinstance(value, Mapping):
        return {"overall_status": None, ready_key: False, "blocker": invalid_blocker}
    overall_status = value.get("overall_status", value.get("status"))
    if not isinstance(overall_status, str):
        return {"overall_status": None, ready_key: False, "blocker": invalid_blocker}
    if overall_status in {"error", "failed", "blocked"}:
        return {
            "overall_status": overall_status,
            ready_key: False,
            "blocker": failed_blocker,
        }
    return {"overall_status": overall_status, ready_key: True, "blocker": None}


def _run_config_summary(root: Path) -> dict[str, Any]:
    path = root / RUN_CONFIG
    if not path.is_file():
        return {
            "ready_for_pipeline": False,
            "blocker": "missing_run_config",
            "error": None,
            "config": None,
        }
    value = _json_if_present(path)
    if not isinstance(value, Mapping):
        return {
            "ready_for_pipeline": False,
            "blocker": "invalid_run_config",
            "error": f"{RUN_CONFIG} is not a JSON object.",
            "config": None,
        }
    try:
        validate_run_config(value)
    except Exception as exc:
        return {
            "ready_for_pipeline": False,
            "blocker": "invalid_run_config",
            "error": str(exc),
            "config": value,
        }
    return {
        "ready_for_pipeline": True,
        "blocker": None,
        "error": None,
        "config": value,
    }


def _capture_plan_summary(root: Path) -> dict[str, Any]:
    path = root / CAPTURE_PLAN
    if not path.is_file():
        return {
            "ready_for_preflight": False,
            "blocker": "missing_capture_plan",
            "command_count": 0,
        }
    value = _json_if_present(path)
    if not isinstance(value, Mapping) or value.get("schema_version") != "capture_plan.v1":
        return {
            "ready_for_preflight": False,
            "blocker": "invalid_capture_plan",
            "command_count": 0,
        }
    commands = value.get("commands")
    if not isinstance(commands, list) or not commands:
        return {
            "ready_for_preflight": False,
            "blocker": "empty_capture_plan",
            "command_count": 0,
        }
    receiver_count = sum(
        1
        for command in commands
        if isinstance(command, Mapping)
        and command.get("role") == "robot_pose_receiver"
    )
    if receiver_count != 1:
        return {
            "ready_for_preflight": False,
            "blocker": "missing_robot_pose_receiver",
            "command_count": len(commands),
        }
    return {
        "ready_for_preflight": True,
        "blocker": None,
        "command_count": len(commands),
    }


def _capture_execution_plan_summary(root: Path) -> dict[str, Any]:
    path = root / CAPTURE_EXECUTION_PLAN
    if not path.is_file():
        return {
            "status": None,
            "ready_to_execute": False,
            "blocker": "missing_capture_execution_plan",
        }
    value = _json_if_present(path)
    if not isinstance(value, Mapping):
        return {
            "status": None,
            "ready_to_execute": False,
            "blocker": "invalid_capture_execution_plan",
        }
    status = value.get("status")
    ready_to_execute = value.get("ready_to_execute")
    if ready_to_execute is True and status in {"ok", "warning"}:
        return {"status": status, "ready_to_execute": True, "blocker": None}
    return {
        "status": status if isinstance(status, str) else None,
        "ready_to_execute": False,
        "blocker": "failed_capture_execution_plan",
    }


def _capture_execution_report_summary(root: Path) -> dict[str, Any]:
    path = root / CAPTURE_EXECUTION_REPORT
    if not path.is_file():
        return {
            "status": None,
            "ready_for_sync": False,
            "blocker": "missing_capture_execution_report",
        }
    value = _json_if_present(path)
    if not isinstance(value, Mapping):
        return {
            "status": None,
            "ready_for_sync": False,
            "blocker": "invalid_capture_execution_report",
        }
    status = value.get("status")
    if status != "succeeded":
        return {
            "status": status if isinstance(status, str) else None,
            "ready_for_sync": False,
            "blocker": "failed_capture_execution_report",
        }
    return {"status": status, "ready_for_sync": True, "blocker": None}


def _sync_quality_summary(root: Path) -> dict[str, Any]:
    return _status_summary(
        root,
        SYNC_QUALITY_REPORT,
        ready_key="ready_for_bop",
        missing_blocker="missing_sync_quality_report",
        invalid_blocker="invalid_sync_quality_report",
        failed_blocker="failed_sync_quality_report",
    )


def _aruco_coverage_summary(root: Path) -> dict[str, Any]:
    return _status_summary(
        root,
        ARUCO_COVERAGE_REPORT,
        ready_key="ready_for_calibration",
        missing_blocker="missing_aruco_coverage_report",
        invalid_blocker="invalid_aruco_coverage_report",
        failed_blocker="failed_aruco_coverage_report",
    )


def _calibration_stage_summary(root: Path, artifact_name: str, label: str) -> dict[str, Any]:
    return _status_summary(
        root,
        artifact_name,
        ready_key=f"ready_for_{label}",
        missing_blocker=f"missing_{label}",
        invalid_blocker=f"invalid_{label}",
        failed_blocker=f"failed_{label}",
    )


def _bop_export_summary(root: Path) -> dict[str, Any]:
    path = root / BOP_DIR / BOP_EXPORT_MANIFEST
    if not path.is_file():
        return {
            "ready_for_dataset_use": False,
            "blocker": "missing_bop_export_manifest",
            "export_count": 0,
            "target_count": 0,
            "model_count": 0,
        }
    manifest = _json_if_present(path)
    if not isinstance(manifest, Mapping):
        return {
            "ready_for_dataset_use": False,
            "blocker": "invalid_bop_export_manifest",
            "export_count": 0,
            "target_count": 0,
            "model_count": 0,
        }
    exports = manifest.get("exports")
    export_count = len(exports) if isinstance(exports, list) else 0
    targets = _json_if_present(root / BOP_DIR / BOP_TARGETS_BOP19)
    target_count = len(targets) if isinstance(targets, list) else 0
    object_models = manifest.get("object_models")
    model_count = len(object_models) if isinstance(object_models, list) else 0
    if export_count == 0:
        blocker = "empty_bop_export_manifest"
    elif target_count == 0:
        blocker = "missing_bop_targets"
    elif model_count == 0:
        blocker = "missing_bop_models"
    else:
        blocker = None
    return {
        "ready_for_dataset_use": blocker is None,
        "blocker": blocker,
        "export_count": export_count,
        "target_count": target_count,
        "model_count": model_count,
    }


def _pipeline_sequence_plan_summary(root: Path) -> dict[str, Any]:
    path = root / PIPELINE_SEQUENCE_PLAN
    if not path.is_file():
        return {
            "ready_for_queue": False,
            "blocker": "missing_pipeline_sequence_plan",
            "step_count": 0,
        }
    value = _json_if_present(path)
    if not isinstance(value, Mapping):
        return {
            "ready_for_queue": False,
            "blocker": "invalid_pipeline_sequence_plan",
            "step_count": 0,
        }
    steps = value.get("steps")
    step_count = len(steps) if isinstance(steps, list) else 0
    return {
        "ready_for_queue": step_count > 0,
        "blocker": None if step_count > 0 else "empty_pipeline_sequence_plan",
        "step_count": step_count,
    }


def _synchronized_root(run_root: Path) -> Path:
    return run_root / PROCESSED_DIR / SYNCHRONIZED_DIR


def _synchronized_sensor_dirs(run_root: Path) -> list[Path]:
    root = _synchronized_root(run_root)
    if not root.is_dir():
        return []
    return [child for child in sorted(root.iterdir()) if child.is_dir()]


def _has_raw_sensor_folders(run_root: Path) -> bool:
    if not run_root.is_dir():
        return False
    return any(
        child.is_dir() and child.name.startswith(RAW_SENSOR_PREFIXES)
        for child in run_root.iterdir()
    )


def _has_target_pose_outputs(run_root: Path) -> bool:
    return any(
        (sensor / artifact_name).is_file()
        for sensor in _synchronized_sensor_dirs(run_root)
        for artifact_name in CALIBRATION_TARGET_POSE_ARTIFACTS
    )


def _has_aruco_outputs(run_root: Path) -> bool:
    return any(
        (sensor / ARUCO_POSE_ESTIMATION).is_file()
        for sensor in _synchronized_sensor_dirs(run_root)
    )


def _has_blenderproc_prepared(run_root: Path) -> bool:
    return any(
        (sensor / "blenderproc" / "objects.json").is_file()
        for sensor in _synchronized_sensor_dirs(run_root)
    )


def _run_config_sequence_has_stage(run_root: Path, stage_id: str) -> bool:
    value = _json_if_present(run_root / RUN_CONFIG)
    if not isinstance(value, Mapping):
        return False
    try:
        plan = sequence_plan_from_run_config(value)
    except Exception:
        return False
    return any(step.stage_id == stage_id for step in plan.steps)


def _stage_recommendation(
    *,
    recommendation_id: str,
    stage_id: str,
    run_root: Path,
    label: str,
    description: str,
    reason: str,
    priority: int,
    expected_artifacts: list[str],
    options: Mapping[str, Any] | None = None,
    endpoint: str = "/pipeline/run",
    blocks_on: list[str] | None = None,
) -> PipelineRecommendation:
    job = build_pipeline_job(stage_id=stage_id, run_root=run_root, options=options)
    return PipelineRecommendation(
        id=recommendation_id,
        label=label,
        description=description,
        reason=reason,
        priority=priority,
        action_type="stage",
        command=job.command,
        endpoint=endpoint,
        method="POST",
        stage_id=stage_id,
        expected_artifacts=expected_artifacts,
        resources=job.resources,
        blocks_on=list(blocks_on or []),
    )


def _command_recommendation(
    *,
    recommendation_id: str,
    label: str,
    description: str,
    reason: str,
    priority: int,
    command: list[str],
    expected_artifacts: list[str],
    action_type: str = "command",
    endpoint: str | None = None,
    method: str | None = None,
    blocks_on: list[str] | None = None,
) -> PipelineRecommendation:
    return PipelineRecommendation(
        id=recommendation_id,
        label=label,
        description=description,
        reason=reason,
        priority=priority,
        action_type=action_type,
        command=command,
        endpoint=endpoint,
        method=method,
        expected_artifacts=expected_artifacts,
        resources=["disk_io"],
        blocks_on=list(blocks_on or []),
    )


def build_pipeline_recommendations(run_root: str | Path) -> dict[str, Any]:
    """Build read-only recommendations from current acquisition artifacts."""

    root = Path(run_root)
    run_config = _run_config_summary(root)
    config = run_config.get("config")
    preflight_queue = (
        run_preflight_queue_summary(root, config)
        if isinstance(config, Mapping)
        else None
    )
    capture_plan = _capture_plan_summary(root)
    capture_plan_preflight = _status_summary(
        root,
        CAPTURE_PLAN_PREFLIGHT_REPORT,
        ready_key="ready_for_execution_plan",
        missing_blocker="missing_capture_plan_preflight",
        invalid_blocker="invalid_capture_plan_preflight",
        failed_blocker="failed_capture_plan_preflight",
    )
    capture_execution_plan = _capture_execution_plan_summary(root)
    capture_execution_report = _capture_execution_report_summary(root)
    sync_quality = _sync_quality_summary(root)
    aruco_coverage = _aruco_coverage_summary(root)
    calibration_preflight = _calibration_stage_summary(
        root,
        CALIBRATION_PREFLIGHT_REPORT,
        "calibration_preflight",
    )
    calibration_observations = _calibration_stage_summary(
        root,
        CALIBRATION_OBSERVATIONS,
        "calibration_observations",
    )
    calibration_solver = _calibration_stage_summary(
        root,
        CALIBRATION_SOLVER_REPORT,
        "calibration_solver",
    )
    calibration_candidates = _calibration_stage_summary(
        root,
        CALIBRATION_CANDIDATES,
        "calibration_candidates",
    )
    calibration_validation = _calibration_stage_summary(
        root,
        CALIBRATION_VALIDATION_REPORT,
        "calibration_validation",
    )
    bop_export = _bop_export_summary(root)
    pipeline_sequence_plan = _pipeline_sequence_plan_summary(root)
    synchronized_sensors = _synchronized_sensor_dirs(root)
    rewrite_status = build_rewrite_status_report(root)

    facts: dict[str, Any] = {
        "run_root": root.as_posix(),
        "has_run_config": (root / RUN_CONFIG).is_file(),
        "run_config_ready_for_pipeline": run_config["ready_for_pipeline"],
        "run_config_blocker": run_config["blocker"],
        "run_config_error": run_config["error"],
        "has_run_preflight": (root / RUN_PREFLIGHT_REPORT).is_file(),
        "run_preflight_ready_for_queue": (
            bool(preflight_queue and preflight_queue.get("ready_for_queue"))
        ),
        "run_preflight_queue_blocker": (
            preflight_queue.get("queue_blocker")
            if isinstance(preflight_queue, Mapping)
            else "missing_run_config"
        ),
        "has_pipeline_sequence_plan": (root / PIPELINE_SEQUENCE_PLAN).is_file(),
        "pipeline_sequence_plan_ready_for_queue": pipeline_sequence_plan[
            "ready_for_queue"
        ],
        "pipeline_sequence_plan_step_count": pipeline_sequence_plan["step_count"],
        "has_capture_plan": (root / CAPTURE_PLAN).is_file(),
        "capture_plan_ready_for_preflight": capture_plan["ready_for_preflight"],
        "capture_plan_blocker": capture_plan["blocker"],
        "capture_plan_command_count": capture_plan["command_count"],
        "has_capture_plan_preflight": (root / CAPTURE_PLAN_PREFLIGHT_REPORT).is_file(),
        "capture_plan_preflight_ready_for_execution_plan": capture_plan_preflight[
            "ready_for_execution_plan"
        ],
        "capture_plan_preflight_blocker": capture_plan_preflight["blocker"],
        "has_capture_execution_plan": (root / CAPTURE_EXECUTION_PLAN).is_file(),
        "capture_execution_plan_ready_to_execute": capture_execution_plan[
            "ready_to_execute"
        ],
        "capture_execution_plan_blocker": capture_execution_plan["blocker"],
        "has_capture_execution_report": (root / CAPTURE_EXECUTION_REPORT).is_file(),
        "capture_execution_report_ready_for_sync": capture_execution_report[
            "ready_for_sync"
        ],
        "capture_execution_report_blocker": capture_execution_report["blocker"],
        "has_raw_robot_poses": (root / RAW_ROBOT_EE_POSES).is_file(),
        "has_raw_sensor_folders": _has_raw_sensor_folders(root),
        "synchronized_sensor_count": len(synchronized_sensors),
        "has_synchronized_sensors": bool(synchronized_sensors),
        "has_sync_quality": (root / SYNC_QUALITY_REPORT).is_file(),
        "sync_quality_ready_for_bop": sync_quality["ready_for_bop"],
        "sync_quality_blocker": sync_quality["blocker"],
        "has_aruco_outputs": _has_aruco_outputs(root),
        "has_target_pose_outputs": _has_target_pose_outputs(root),
        "has_aruco_coverage": (root / ARUCO_COVERAGE_REPORT).is_file(),
        "aruco_coverage_ready_for_calibration": aruco_coverage[
            "ready_for_calibration"
        ],
        "has_calibration_profiles": (root / CALIBRATION_PROFILES).is_file(),
        "has_calibration_preflight": (root / CALIBRATION_PREFLIGHT_REPORT).is_file(),
        "calibration_preflight_ready": calibration_preflight[
            "ready_for_calibration_preflight"
        ],
        "has_calibration_observations": (root / CALIBRATION_OBSERVATIONS).is_file(),
        "calibration_observations_ready": calibration_observations[
            "ready_for_calibration_observations"
        ],
        "has_calibration_solver": (root / CALIBRATION_SOLVER_REPORT).is_file(),
        "calibration_solver_ready": calibration_solver["ready_for_calibration_solver"],
        "has_calibration_candidates": (root / CALIBRATION_CANDIDATES).is_file(),
        "calibration_candidates_ready": calibration_candidates[
            "ready_for_calibration_candidates"
        ],
        "has_calibration_validation": (root / CALIBRATION_VALIDATION_REPORT).is_file(),
        "calibration_validation_ready": calibration_validation[
            "ready_for_calibration_validation"
        ],
        "has_blenderproc_prepared": _has_blenderproc_prepared(root),
        "has_blenderproc_render_plan": (root / BLENDERPROC_RENDER_PLAN).is_file(),
        "has_bop_export": (root / BOP_DIR / BOP_EXPORT_MANIFEST).is_file(),
        "bop_export_ready_for_dataset_use": bop_export["ready_for_dataset_use"],
        "bop_export_blocker": bop_export["blocker"],
        "bop_export_count": bop_export["export_count"],
        "bop_target_count": bop_export["target_count"],
        "bop_model_count": bop_export["model_count"],
        "has_rewrite_gate_report": (root / REWRITE_GATE_REPORT).is_file(),
        "has_rewrite_status_report": (root / REWRITE_STATUS_REPORT).is_file(),
        "rewrite_status": rewrite_status["overall_status"],
        "rewrite_next_gate": rewrite_status.get("next_gate"),
    }

    recommendations: list[PipelineRecommendation] = []
    if not facts["has_run_config"]:
        recommendations.append(
            _command_recommendation(
                recommendation_id="create_run_config",
                label="Create run config",
                description="Write the operator intent artifact for this run.",
                reason=f"{RUN_CONFIG} is missing.",
                priority=10,
                command=[
                    "uv",
                    "run",
                    "python",
                    "scripts/create_run_config.py",
                    root.as_posix(),
                ],
                expected_artifacts=[RUN_CONFIG],
            )
        )
    elif run_config["ready_for_pipeline"]:
        if not facts["run_preflight_ready_for_queue"]:
            recommendations.append(
                _stage_recommendation(
                    recommendation_id="write_run_preflight",
                    stage_id="run_preflight",
                    run_root=root,
                    label="Write run preflight",
                    description="Snapshot run-config, robot, sensor, and runtime readiness.",
                    reason=(
                        f"{RUN_PREFLIGHT_REPORT} is "
                        f"{facts['run_preflight_queue_blocker']}."
                    ),
                    priority=20,
                    expected_artifacts=[RUN_PREFLIGHT_REPORT],
                    options={"write": True},
                )
            )
        elif not pipeline_sequence_plan["ready_for_queue"]:
            recommendations.append(
                _command_recommendation(
                    recommendation_id="plan_saved_sequence",
                    label="Plan saved sequence",
                    description="Write a dependency-aware sequence plan for inspection.",
                    reason=f"{PIPELINE_SEQUENCE_PLAN} is missing or empty.",
                    priority=25,
                    command=[
                        "uv",
                        "run",
                        "python",
                        "scripts/run_pipeline_sequence.py",
                        root.as_posix(),
                        "--sequence",
                        str(config["pipeline"]["sequence_id"]),
                        "--plan-only",
                    ],
                    expected_artifacts=[PIPELINE_SEQUENCE_PLAN],
                )
            )

    if run_config["ready_for_pipeline"] and not capture_plan["ready_for_preflight"]:
        recommendations.append(
            _stage_recommendation(
                recommendation_id="write_capture_plan",
                stage_id="capture_plan",
                run_root=root,
                label="Write capture plan",
                description="Build capture commands without opening cameras or moving the robot.",
                reason=f"{CAPTURE_PLAN} is {capture_plan['blocker']}.",
                priority=30,
                expected_artifacts=[CAPTURE_PLAN],
            )
        )
    if capture_plan["ready_for_preflight"] and not capture_plan_preflight[
        "ready_for_execution_plan"
    ]:
        recommendations.append(
            _stage_recommendation(
                recommendation_id="preflight_capture_plan",
                stage_id="capture_plan_preflight",
                run_root=root,
                label="Preflight capture plan",
                description="Validate command shape, safety gates, and optional sensor checks.",
                reason=f"{CAPTURE_PLAN_PREFLIGHT_REPORT} is {capture_plan_preflight['blocker']}.",
                priority=35,
                expected_artifacts=[CAPTURE_PLAN_PREFLIGHT_REPORT],
            )
        )
    if capture_plan_preflight["ready_for_execution_plan"] and not capture_execution_plan[
        "ready_to_execute"
    ]:
        recommendations.append(
            _stage_recommendation(
                recommendation_id="write_capture_execution_plan",
                stage_id="capture_execution_plan",
                run_root=root,
                label="Write capture execution plan",
                description="Select safe capture commands before process supervision.",
                reason=f"{CAPTURE_EXECUTION_PLAN} is {capture_execution_plan['blocker']}.",
                priority=40,
                expected_artifacts=[CAPTURE_EXECUTION_PLAN],
            )
        )
    if capture_execution_plan["ready_to_execute"] and not capture_execution_report[
        "ready_for_sync"
    ]:
        recommendations.append(
            _stage_recommendation(
                recommendation_id="run_capture_execution",
                stage_id="capture_execution",
                run_root=root,
                label="Run supervised capture",
                description="Execute selected capture commands with supervisor logging.",
                reason=f"{CAPTURE_EXECUTION_REPORT} is {capture_execution_report['blocker']}.",
                priority=45,
                expected_artifacts=[CAPTURE_EXECUTION_REPORT],
            )
        )

    if facts["has_raw_sensor_folders"] and not facts["has_synchronized_sensors"]:
        recommendations.append(
            _stage_recommendation(
                recommendation_id="sync_run",
                stage_id="sync_run",
                run_root=root,
                label="Synchronize raw sensors",
                description="Create processed/synchronized sensor folders without changing raw captures.",
                reason="Raw sensor folders exist but synchronized outputs are missing.",
                priority=50,
                expected_artifacts=[f"{PROCESSED_DIR}/{SYNCHRONIZED_DIR}"],
            )
        )
    if facts["has_synchronized_sensors"] and not sync_quality["ready_for_bop"]:
        recommendations.append(
            _stage_recommendation(
                recommendation_id="write_sync_quality",
                stage_id="sync_quality",
                run_root=root,
                label="Write sync quality report",
                description="Check frame match ratios, dropped frames, and nearest-pose deltas.",
                reason=f"{SYNC_QUALITY_REPORT} is {sync_quality['blocker']}.",
                priority=55,
                expected_artifacts=[SYNC_QUALITY_REPORT],
            )
        )

    if facts["has_synchronized_sensors"] and not facts["has_aruco_outputs"]:
        recommendations.append(
            _stage_recommendation(
                recommendation_id="run_aruco",
                stage_id="aruco",
                run_root=root,
                label="Run ArUco target detection",
                description="Generate target-pose observations for calibration support.",
                reason="Synchronized sensors exist but ArUco outputs are missing.",
                priority=60,
                expected_artifacts=[ARUCO_POSE_ESTIMATION],
            )
        )
    if facts["has_aruco_outputs"] and not aruco_coverage["ready_for_calibration"]:
        recommendations.append(
            _stage_recommendation(
                recommendation_id="write_aruco_coverage",
                stage_id="aruco_coverage",
                run_root=root,
                label="Write ArUco coverage",
                description="Summarize target detection coverage before calibration extraction.",
                reason=f"{ARUCO_COVERAGE_REPORT} is {aruco_coverage['blocker']}.",
                priority=65,
                expected_artifacts=[ARUCO_COVERAGE_REPORT],
            )
        )
    if facts["has_target_pose_outputs"] and not calibration_observations[
        "ready_for_calibration_observations"
    ]:
        recommendations.append(
            _stage_recommendation(
                recommendation_id="build_calibration_observations",
                stage_id="calibration_observations",
                run_root=root,
                label="Build calibration observations",
                description="Extract solver-ready target and robot pose observations.",
                reason=f"{CALIBRATION_OBSERVATIONS} is {calibration_observations['blocker']}.",
                priority=70,
                expected_artifacts=[CALIBRATION_OBSERVATIONS],
            )
        )
    if calibration_observations["ready_for_calibration_observations"] and not calibration_solver[
        "ready_for_calibration_solver"
    ]:
        recommendations.append(
            _stage_recommendation(
                recommendation_id="solve_calibration",
                stage_id="calibration_solver",
                run_root=root,
                label="Solve calibration",
                description="Solve needs-validation calibration profiles from observations.",
                reason=f"{CALIBRATION_SOLVER_REPORT} is {calibration_solver['blocker']}.",
                priority=75,
                expected_artifacts=[CALIBRATION_SOLVER_REPORT],
            )
        )
    if calibration_observations["ready_for_calibration_observations"] and not calibration_candidates[
        "ready_for_calibration_candidates"
    ]:
        recommendations.append(
            _stage_recommendation(
                recommendation_id="build_calibration_candidates",
                stage_id="calibration_candidates",
                run_root=root,
                label="Build calibration candidates",
                description="Average observation-derived transforms into inspectable candidates.",
                reason=f"{CALIBRATION_CANDIDATES} is {calibration_candidates['blocker']}.",
                priority=76,
                expected_artifacts=[CALIBRATION_CANDIDATES],
            )
        )
    if (
        calibration_solver["ready_for_calibration_solver"]
        or calibration_candidates["ready_for_calibration_candidates"]
    ) and not calibration_validation["ready_for_calibration_validation"]:
        recommendations.append(
            _stage_recommendation(
                recommendation_id="validate_calibration",
                stage_id="calibration_validation",
                run_root=root,
                label="Validate calibration",
                description="Gate solved or candidate profiles before optional promotion.",
                reason=f"{CALIBRATION_VALIDATION_REPORT} is {calibration_validation['blocker']}.",
                priority=80,
                expected_artifacts=[CALIBRATION_VALIDATION_REPORT],
            )
        )

    if facts["sync_quality_ready_for_bop"] and not facts["has_blenderproc_prepared"]:
        recommendations.append(
            _stage_recommendation(
                recommendation_id="prepare_blenderproc",
                stage_id="blenderproc_prepare",
                run_root=root,
                label="Prepare BlenderProc inputs",
                description="Prepare object and camera inputs for optional GT rendering.",
                reason="Sync quality is ready but BlenderProc inputs are missing.",
                priority=90,
                expected_artifacts=["blenderproc/objects.json"],
            )
        )
    if facts["has_blenderproc_prepared"] and not facts["has_blenderproc_render_plan"]:
        recommendations.append(
            _stage_recommendation(
                recommendation_id="plan_blenderproc_render",
                stage_id="blenderproc_render",
                run_root=root,
                label="Plan BlenderProc render",
                description="Write a dry-run render plan for optional GT and masks.",
                reason=f"{BLENDERPROC_RENDER_PLAN} is missing.",
                priority=95,
                expected_artifacts=[BLENDERPROC_RENDER_PLAN],
                options={"dry_run": True},
            )
        )
    if facts["sync_quality_ready_for_bop"] and not bop_export["ready_for_dataset_use"]:
        recommendations.append(
            _stage_recommendation(
                recommendation_id="export_bop_dataset",
                stage_id="bop_export",
                run_root=root,
                label="Export BOP dataset",
                description="Write BOP scene folders, targets, model metadata, and frame maps.",
                reason=f"{BOP_EXPORT_MANIFEST} is {bop_export['blocker']}.",
                priority=100,
                expected_artifacts=[
                    f"{BOP_DIR}/{BOP_EXPORT_MANIFEST}",
                    f"{BOP_DIR}/{BOP_TARGETS_BOP19}",
                ],
            )
        )

    next_actions = rewrite_status.get("next_actions")
    if isinstance(next_actions, list):
        for index, action in enumerate(next_actions[:3], start=1):
            if not isinstance(action, Mapping):
                continue
            command = action.get("command")
            if not isinstance(command, list):
                continue
            recommendations.append(
                _command_recommendation(
                    recommendation_id=f"rewrite_next_action_{index}",
                    label=str(action.get("label") or "Audit acquisition gate"),
                    description="Advance the acquisition-only rewrite gate evidence.",
                    reason=str(action.get("reason") or "Rewrite status is blocked."),
                    priority=120 + index,
                    command=[str(part) for part in command],
                    expected_artifacts=[REWRITE_GATE_REPORT],
                    blocks_on=[
                        str(item)
                        for item in action.get("blocks_on", [])
                        if isinstance(item, str)
                    ],
                )
            )

    recommendations.sort(key=lambda item: (item.priority, item.id))
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "run_root": root.as_posix(),
        "facts": facts,
        "recommendations": [recommendation.to_dict() for recommendation in recommendations],
    }
