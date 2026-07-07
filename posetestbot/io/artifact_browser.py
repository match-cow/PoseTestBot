"""Run artifact discovery and safe previews for PoseTestBot."""

from __future__ import annotations

import base64
import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from posetestbot.bop.writer import mesh_vertices
from posetestbot.evaluation.bop_toolkit import (
    BOP19_RESULT_HEADER,
    validate_bop19_result_file,
)
from posetestbot.io.artifacts import (
    ACCURACY_ARUCO_HRC_HUB,
    ACCURACY_HRC_HUB,
    ALL_RESULTS_JSON,
    ARUCO_COVERAGE_REPORT,
    BOP_COCO_ANNOTATIONS,
    BOP_DIR,
    BOP_EVALUATION_PLAN,
    BOP_EVALUATION_REPORT,
    BOP_EXPORT_MANIFEST,
    BOP_FRAME_MAP_JSON,
    BOP_MULTIVIEW_TARGETS,
    BOP_RESULT_EXPORT_MANIFEST,
    BOP_TARGETS_BOP19,
    CAPTURE_EXECUTION_LOGS_DIR,
    CAPTURE_EXECUTION_PLAN,
    CAPTURE_EXECUTION_REPORT,
    CAPTURE_EXECUTION_STATUS,
    CAPTURE_PLAN,
    CAPTURE_PLAN_PREFLIGHT_REPORT,
    CAPTURE_REHEARSAL_REPORT,
    REALSENSE_CAPTURE_SMOKE_REPORT,
    CALIBRATION_OBSERVATIONS,
    CALIBRATION_CANDIDATES,
    CALIBRATION_PREFLIGHT_REPORT,
    CALIBRATION_PROFILES_FROM_OBSERVATIONS,
    CALIBRATION_PROFILES_SOLVED,
    CALIBRATION_SOLVER_REPORT,
    CALIBRATION_VALIDATION_REPORT,
    DATASET_MANIFEST,
    DEPTH_DIR,
    EVALUATION_DIR,
    FOUNDATIONPOSE_PLAN,
    HARDWARE_STATUS_REPORT,
    MEGAPOSE_PLAN,
    METRIC_REPORT_CSV,
    METRIC_REPORT_JSON,
    METRIC_REPORT_XLSX,
    METRICS_DIR,
    MODELS_DIR,
    PIPELINE_SEQUENCE_PLAN,
    RGB_DIR,
    RESULTS_DIR,
    REWRITE_GATE_REPORT,
    REWRITE_STATUS_REPORT,
    RUN_CONFIG,
    RUN_PREFLIGHT_REPORT,
    SAM6D_PLAN,
    SYNC_QUALITY_REPORT,
)
from posetestbot.io.manifest import load_run_manifest
from posetestbot.pipeline.preflight import run_preflight_queue_summary
from posetestbot.pipeline.run_config import validate_run_config


TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
ACCURACY_ARTIFACTS = {
    ACCURACY_HRC_HUB,
    ACCURACY_ARUCO_HRC_HUB,
}
ESTIMATOR_PLAN_ARTIFACTS = {
    FOUNDATIONPOSE_PLAN: "foundationpose",
    MEGAPOSE_PLAN: "megapose",
    SAM6D_PLAN: "sam6d",
}
BOP_EVALUATION_CRITICAL_CHECKS = frozenset(
    {
        "result_file",
        "bop_root",
        "dataset_folder",
        "targets_file",
        "models_folder",
        "models_info",
        "model_files",
    }
)
METRIC_KEYS = (
    "AP_p",
    "ap_x",
    "ap_y",
    "ap_z",
    "ap_a",
    "ap_b",
    "ap_c",
    "RP_i",
    "RP_a",
    "RP_b",
    "RP_c",
)
RAW_SAMPLE_KEYS = ("x", "y", "z", "a", "b", "c")


class ArtifactPathError(ValueError):
    """Raised when an artifact path is invalid or outside the run root."""


@dataclass(frozen=True)
class ArtifactRecord:
    key: str
    source: str
    path: str
    relative_path: str | None
    kind: str
    exists: bool
    preview_type: str
    size_bytes: int | None = None
    modified_at: str | None = None
    child_count: int | None = None
    summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["display_label"] = _artifact_display_label(
            key=self.key,
            exists=self.exists,
            kind=self.kind,
            summary=self.summary,
        )
        return payload


def _string_list(value: Any, *, limit: int = 3) -> list[str]:
    if not isinstance(value, list):
        return []
    items = [str(item) for item in value if item is not None]
    if len(items) <= limit:
        return items
    return [*items[:limit], f"+{len(items) - limit} more"]


def _truthy_label(value: Any) -> str | None:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return None


def _short_label_value(value: str, *, limit: int = 72) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "..."


def _capture_execution_report_blocker(status: object) -> str | None:
    if status == "succeeded":
        return None
    if isinstance(status, str):
        return "failed_capture_execution_report"
    return "invalid_capture_execution_report"


def _rewrite_gate_blocker(status: object) -> str | None:
    if status == "ready":
        return None
    if status == "blocked":
        return "blocked_rewrite_gate"
    if isinstance(status, str):
        return "failed_rewrite_gate"
    return "invalid_rewrite_gate"


def _rewrite_blocker_names(next_blockers: object, *, limit: int = 5) -> list[str]:
    if not isinstance(next_blockers, list):
        return []
    return [
        str(blocker.get("name"))
        for blocker in next_blockers
        if isinstance(blocker, Mapping) and blocker.get("name")
    ][:limit]


def _rewrite_blocker_messages(
    next_blockers: object,
    *,
    limit: int = 5,
) -> list[str]:
    if not isinstance(next_blockers, list):
        return []
    return [
        str(blocker.get("message"))
        for blocker in next_blockers
        if isinstance(blocker, Mapping) and blocker.get("message")
    ][:limit]


def _rewrite_blocker_detail_summary(
    next_blockers: object,
    *,
    limit: int = 5,
) -> dict[str, list[str]]:
    if not isinstance(next_blockers, list):
        return {"diagnostics": [], "hints": [], "blocked_checks": []}

    diagnostics: list[str] = []
    hints: list[str] = []
    blocked_checks: list[str] = []
    for blocker in next_blockers:
        if not isinstance(blocker, Mapping):
            continue
        details = blocker.get("details")
        if not isinstance(details, Mapping):
            continue
        error_checks = details.get("error_checks")
        if isinstance(error_checks, list):
            for check in error_checks:
                if not isinstance(check, Mapping):
                    continue
                name = check.get("name")
                message = check.get("message")
                if name or message:
                    value = str(name) if name else ""
                    if message:
                        value = f"{value}: {message}" if value else str(message)
                    blocked_checks.append(value)
        sensor_diagnostics = details.get("sensor_diagnostics")
        if isinstance(sensor_diagnostics, list):
            for diagnostic in sensor_diagnostics:
                if not isinstance(diagnostic, Mapping):
                    continue
                message = diagnostic.get("message")
                if message:
                    diagnostics.append(str(message))
                diagnostic_hints = diagnostic.get("hints")
                if isinstance(diagnostic_hints, list):
                    hints.extend(str(hint) for hint in diagnostic_hints if hint)

    return {
        "diagnostics": diagnostics[:limit],
        "hints": hints[:limit],
        "blocked_checks": blocked_checks[:limit],
    }


def _problem_checks_from_report(
    value: Mapping[str, Any],
    *,
    blocking_statuses: set[str] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    blocking_statuses = blocking_statuses or {"error", "blocked", "failed"}
    checks = value.get("checks")
    if not isinstance(checks, list):
        checks = value.get("gates")
    if not isinstance(checks, list):
        return []
    problems: list[dict[str, Any]] = []
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        status = str(check.get("status") or "")
        if status not in blocking_statuses:
            continue
        problems.append(
            {
                "name": check.get("name"),
                "status": status,
                "message": check.get("message"),
            }
        )
        if len(problems) >= limit:
            break
    return problems


def _capture_rehearsal_report_blocker(
    status: object,
    raw_pose_count: object,
) -> str | None:
    if status != "succeeded":
        if isinstance(status, str):
            return "failed_capture_rehearsal_report"
        return "invalid_capture_rehearsal_report"
    if not isinstance(raw_pose_count, int):
        return "invalid_capture_rehearsal_report"
    if raw_pose_count <= 0:
        return "empty_capture_rehearsal_report"
    return None


def _run_config_readiness(value: Mapping[str, Any]) -> tuple[bool, str | None, str | None]:
    try:
        validate_run_config(value)
    except Exception as exc:
        return False, "invalid_run_config", str(exc)
    return True, None, None


def _capture_plan_blocker(schema_version: object, commands: object) -> str | None:
    if schema_version != "capture_plan.v1":
        return "invalid_capture_plan"
    if not isinstance(commands, list):
        return "invalid_capture_plan"
    if not commands:
        return "empty_capture_plan"
    receiver_count = sum(
        1
        for command in commands
        if isinstance(command, Mapping)
        and command.get("role") == "robot_pose_receiver"
    )
    if receiver_count != 1:
        return "missing_robot_pose_receiver"
    return None


def _capture_plan_preflight_blocker(status: object) -> str | None:
    if status in {"ok", "warning"}:
        return None
    if status == "error":
        return "failed_capture_plan_preflight"
    return "invalid_capture_plan_preflight"


def _hardware_status_report_blocker(status: object) -> str | None:
    if status in {"ok", "warning"}:
        return None
    if status == "error":
        return "failed_hardware_status_report"
    return "invalid_hardware_status_report"


def _capture_execution_plan_blocker(status: object, ready_to_execute: object) -> str | None:
    if ready_to_execute is True and status in {"ok", "warning"}:
        return None
    if isinstance(status, str):
        return "failed_capture_execution_plan"
    return "invalid_capture_execution_plan"


def _bop_evaluation_report_blocker(status: object) -> str | None:
    if status in {"planned", "succeeded"}:
        return None
    if isinstance(status, str):
        return "failed_bop_evaluation_report"
    return "invalid_bop_evaluation_report"


def _bop_evaluation_critical_check_counts(checks: object) -> tuple[int, int]:
    if not isinstance(checks, list) or not checks:
        return 0, 0
    seen: set[str] = set()
    failed = 0
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        name = check.get("name")
        if not isinstance(name, str) or name not in BOP_EVALUATION_CRITICAL_CHECKS:
            continue
        seen.add(name)
        if not bool(check.get("ok", False)):
            failed += 1
    missing = len(BOP_EVALUATION_CRITICAL_CHECKS - seen)
    return failed, missing


def _bop_evaluation_plan_blocker(
    schema_version: object,
    result: object,
    command: object,
    environment: object,
) -> str | None:
    if schema_version != "bop_evaluation_plan.v1":
        return "invalid_bop_evaluation_plan"
    if not isinstance(result, Mapping):
        return "invalid_bop_evaluation_plan"
    if not isinstance(result.get("filename"), str) or not result.get("filename"):
        return "invalid_bop_evaluation_plan"
    if not isinstance(result.get("path"), str) or not result.get("path"):
        return "invalid_bop_evaluation_plan"
    if not isinstance(command, list) or not all(
        isinstance(item, str) for item in command
    ):
        return "invalid_bop_evaluation_plan"
    if not command:
        return "empty_bop_evaluation_plan"
    if not isinstance(environment, Mapping):
        return "invalid_bop_evaluation_plan"
    if not isinstance(environment.get("BOP_PATH"), str) or not environment.get(
        "BOP_PATH"
    ):
        return "invalid_bop_evaluation_plan"
    return None


def _sync_quality_report_blocker(status: object) -> str | None:
    if status in {"ok", "warning"}:
        return None
    if status == "error":
        return "failed_sync_quality_report"
    return "invalid_sync_quality_report"


def _aruco_coverage_report_blocker(status: object) -> str | None:
    if status in {"ok", "warning"}:
        return None
    if status == "error":
        return "failed_aruco_coverage_report"
    return "invalid_aruco_coverage_report"


def _calibration_preflight_blocker(status: object) -> str | None:
    if status in {"ok", "warning"}:
        return None
    if status == "error":
        return "failed_calibration_preflight"
    return "invalid_calibration_preflight"


def _calibration_observations_blocker(status: object) -> str | None:
    if status in {"ok", "warning"}:
        return None
    if status == "error":
        return "failed_calibration_observations"
    return "invalid_calibration_observations"


def _calibration_solver_blocker(status: object) -> str | None:
    if status in {"ok", "warning"}:
        return None
    if status == "error":
        return "failed_calibration_solver"
    return "invalid_calibration_solver"


def _calibration_candidates_blocker(status: object) -> str | None:
    if status in {"ok", "warning"}:
        return None
    if status == "error":
        return "failed_calibration_candidates"
    return "invalid_calibration_candidates"


def _calibration_profile_collection_blocker(
    schema_version: object,
    profiles: object,
) -> str | None:
    if schema_version != "calibration.v1":
        return "invalid_calibration_profile_collection"
    if not isinstance(profiles, list):
        return "invalid_calibration_profile_collection"
    if not profiles:
        return "empty_calibration_profile_collection"
    allowed_statuses = {"needs_validation", "valid"}
    for profile in profiles:
        if not isinstance(profile, Mapping):
            return "invalid_calibration_profile_collection"
        status = profile.get("status")
        if status not in allowed_statuses:
            return "invalid_calibration_profile_collection"
    return None


def _calibration_validation_blocker(status: object) -> str | None:
    if status in {"ok", "warning"}:
        return None
    if status == "error":
        return "failed_calibration_validation"
    return "invalid_calibration_validation"


def _bop_result_export_blocker(results: object, usable_result_count: int) -> str | None:
    if not isinstance(results, list):
        return "invalid_bop_result_export_manifest"
    if usable_result_count > 0:
        return None
    return "missing_bop_result_csv"


def _bop_export_blocker(exports: object) -> str | None:
    if not isinstance(exports, list):
        return "invalid_bop_export_manifest"
    if exports:
        return None
    return "empty_bop_export_manifest"


def _bop_targets_blocker(targets: object) -> str | None:
    if not isinstance(targets, list):
        return "invalid_bop_targets"
    if not targets:
        return "empty_bop_targets"
    required_fields = ("scene_id", "im_id", "obj_id", "inst_count")
    for target in targets:
        if not isinstance(target, Mapping):
            return "invalid_bop_targets"
        for field in required_fields:
            try:
                int(target[field])
            except (KeyError, TypeError, ValueError):
                return "invalid_bop_targets"
    return None


def _bop_multiview_targets_blocker(
    schema_version: object,
    targets: object,
) -> str | None:
    if schema_version != "posetestbot_bop_multiview_targets.v1":
        return "invalid_bop_multiview_targets"
    if not isinstance(targets, list):
        return "invalid_bop_multiview_targets"
    if not targets:
        return "empty_bop_multiview_targets"
    for target in targets:
        if not isinstance(target, Mapping):
            return "invalid_bop_multiview_targets"
        try:
            int(target["obj_id"])
            view_count = int(target["view_count"])
        except (KeyError, TypeError, ValueError):
            return "invalid_bop_multiview_targets"
        views = target.get("views")
        if view_count <= 0 or not isinstance(views, list) or not views:
            return "invalid_bop_multiview_targets"
    return None


def _bop_coco_annotations_blocker(
    schema_version: object,
    images: object,
    categories: object,
    annotations: object,
) -> str | None:
    if schema_version != "posetestbot_coco_annotations.v1":
        return "invalid_bop_coco_annotations"
    if (
        not isinstance(images, list)
        or not isinstance(categories, list)
        or not isinstance(annotations, list)
    ):
        return "invalid_bop_coco_annotations"
    if not images:
        return "empty_bop_coco_annotations"
    if not categories:
        return "missing_bop_coco_categories"
    return None


def _bop_models_info_blocker(models_info: object) -> str | None:
    if not isinstance(models_info, Mapping):
        return "invalid_bop_models_info"
    if not models_info:
        return "empty_bop_models_info"
    for obj_id, model_info in models_info.items():
        try:
            int(obj_id)
        except (TypeError, ValueError):
            return "invalid_bop_models_info"
        if not isinstance(model_info, Mapping):
            return "invalid_bop_models_info"
    return None


def _metric_report_blocker(rows: object, dashboard: object) -> str | None:
    if not isinstance(rows, list) or not isinstance(dashboard, Mapping):
        return "invalid_metric_report"
    if rows:
        return None
    return "empty_metric_report"


def _pipeline_sequence_plan_blocker(
    schema_version: object,
    steps: object,
) -> str | None:
    if schema_version != "pipeline_sequence_plan.v1":
        return "invalid_pipeline_sequence_plan"
    if not isinstance(steps, list):
        return "invalid_pipeline_sequence_plan"
    if not steps:
        return "empty_pipeline_sequence_plan"
    for step in steps:
        if not isinstance(step, Mapping):
            return "invalid_pipeline_sequence_plan"
        if not isinstance(step.get("id"), str):
            return "invalid_pipeline_sequence_plan"
        if not isinstance(step.get("stage_id"), str):
            return "invalid_pipeline_sequence_plan"
        command = step.get("command")
        if not isinstance(command, list) or not all(
            isinstance(item, str) for item in command
        ):
            return "invalid_pipeline_sequence_plan"
    return None


def _estimator_plan_blocker(
    jobs: object,
    estimator_id: object,
    expected_estimator_id: str,
) -> str | None:
    if not isinstance(estimator_id, str) or estimator_id != expected_estimator_id:
        return "invalid_estimator_plan"
    if not isinstance(jobs, list):
        return "invalid_estimator_plan"
    if jobs:
        return None
    return "empty_estimator_plan"


def _artifact_display_label(
    *,
    key: str,
    exists: bool,
    kind: str,
    summary: Mapping[str, Any] | None,
) -> str:
    bits = [key, "exists" if exists else "missing", kind]
    if not summary:
        return " · ".join(bits)

    summary_type = summary.get("type")
    if isinstance(summary_type, str):
        bits.append(summary_type)

    status = summary.get("overall_status", summary.get("status"))
    if status is not None:
        bits.append(f"status={status}")

    sequence_id = summary.get("sequence_id")
    if isinstance(sequence_id, str):
        bits.append(f"sequence={sequence_id}")

    for label, field in (
        ("steps", "step_count"),
        ("sensors", "sensor_count"),
        ("frames", "frame_count"),
        ("rows", "row_count"),
        ("results", "result_count"),
        ("checks", "check_count"),
        ("jobs", "job_count"),
    ):
        value = summary.get(field)
        if isinstance(value, int):
            bits.append(f"{label}={value}")

    robot_mode = summary.get("robot_mode")
    if isinstance(robot_mode, str):
        bits.append(f"robot={robot_mode}")

    object_folder = summary.get("object_folder")
    if isinstance(object_folder, str) and object_folder:
        bits.append(f"objects={object_folder}")

    resources = _string_list(summary.get("resources"))
    if resources:
        bits.append(f"resources={','.join(resources)}")

    profile_paths = _string_list(summary.get("calibration_profile_paths"), limit=2)
    profile_path = summary.get("profile_path")
    calibration_profiles = summary.get("calibration_profiles")
    has_calibration_profiles = _truthy_label(summary.get("has_calibration_profiles"))
    if profile_paths:
        bits.append(f"calibration={','.join(profile_paths)}")
    elif isinstance(profile_path, str) and profile_path:
        bits.append(f"calibration={profile_path}")
    elif isinstance(calibration_profiles, str) and calibration_profiles:
        bits.append(f"calibration={calibration_profiles}")
    elif has_calibration_profiles is not None:
        bits.append(f"calibration={has_calibration_profiles}")

    matched_profile_ids = _string_list(summary.get("matched_profile_ids"), limit=2)
    if matched_profile_ids:
        bits.append(f"matched={','.join(matched_profile_ids)}")

    check_status_counts = summary.get("check_status_counts")
    if isinstance(check_status_counts, Mapping) and check_status_counts:
        status_bits = [
            f"{status_key}:{check_status_counts[status_key]}"
            for status_key in sorted(check_status_counts)
        ]
        bits.append(f"check_status={','.join(status_bits)}")

    preflight_ready = summary.get("preflight_ready_for_queue")
    preflight_blocker = summary.get("preflight_queue_blocker")
    if preflight_ready is True:
        bits.append("preflight=ready")
    elif isinstance(preflight_blocker, str) and preflight_blocker:
        bits.append(f"preflight={preflight_blocker}")

    run_config_ready = summary.get("run_config_ready_for_pipeline")
    run_config_blocker = summary.get("run_config_blocker")
    if run_config_ready is True:
        bits.append("run_config=ready")
    elif isinstance(run_config_blocker, str) and run_config_blocker:
        bits.append(f"run_config={run_config_blocker}")

    sequence_plan_ready = summary.get("pipeline_sequence_plan_ready_for_queue")
    sequence_plan_blocker = summary.get("pipeline_sequence_plan_blocker")
    if sequence_plan_ready is True:
        bits.append("sequence_plan=ready")
    elif isinstance(sequence_plan_blocker, str) and sequence_plan_blocker:
        bits.append(f"sequence_plan={sequence_plan_blocker}")

    hardware_ready = summary.get("hardware_status_ready_for_capture")
    hardware_blocker = summary.get("hardware_status_blocker")
    if hardware_ready is True:
        bits.append("hardware_status=ready")
    elif isinstance(hardware_blocker, str) and hardware_blocker:
        bits.append(f"hardware_status={hardware_blocker}")

    capture_ready = summary.get("ready_for_downstream")
    capture_blocker = summary.get("capture_execution_report_blocker")
    if capture_ready is True:
        bits.append("capture=ready")
    elif isinstance(capture_blocker, str) and capture_blocker:
        bits.append(f"capture={capture_blocker}")

    rewrite_gate_ready = summary.get("rewrite_gate_ready")
    rewrite_gate_blocker = summary.get("rewrite_gate_blocker")
    if rewrite_gate_ready is True:
        bits.append("rewrite_gate=ready")
    elif isinstance(rewrite_gate_blocker, str) and rewrite_gate_blocker:
        bits.append(f"rewrite_gate={rewrite_gate_blocker}")

    rewrite_status_ready = summary.get("rewrite_status_ready")
    rewrite_status_blocker = summary.get("rewrite_status_blocker")
    if rewrite_status_ready is True:
        bits.append("rewrite_status=ready")
    elif isinstance(rewrite_status_blocker, str) and rewrite_status_blocker:
        bits.append(f"rewrite_status={rewrite_status_blocker}")
    rewrite_next_gate = summary.get("next_gate_id")
    if isinstance(rewrite_next_gate, str) and rewrite_next_gate:
        bits.append(f"next_gate={rewrite_next_gate}")
    rewrite_next_blockers = summary.get("next_blockers")
    if isinstance(rewrite_next_blockers, list) and rewrite_next_blockers:
        first_blocker = rewrite_next_blockers[0]
        if isinstance(first_blocker, str) and first_blocker:
            bits.append(f"next_blocker={_short_label_value(first_blocker, limit=48)}")
    rewrite_next_diagnostics = summary.get("next_blocker_diagnostics")
    if isinstance(rewrite_next_diagnostics, list) and rewrite_next_diagnostics:
        first_diagnostic = rewrite_next_diagnostics[0]
        if isinstance(first_diagnostic, str) and first_diagnostic:
            bits.append(f"next_diag={_short_label_value(first_diagnostic)}")

    capture_rehearsal_ready = summary.get("capture_rehearsal_ready_for_sync")
    capture_rehearsal_blocker = summary.get("capture_rehearsal_blocker")
    if capture_rehearsal_ready is True:
        bits.append("capture_rehearsal=ready")
    elif isinstance(capture_rehearsal_blocker, str) and capture_rehearsal_blocker:
        bits.append(f"capture_rehearsal={capture_rehearsal_blocker}")

    capture_plan_ready_for_preflight = summary.get("capture_plan_ready_for_preflight")
    capture_plan_blocker = summary.get("capture_plan_blocker")
    if capture_plan_ready_for_preflight is True:
        bits.append("capture_plan=ready")
    elif isinstance(capture_plan_blocker, str) and capture_plan_blocker:
        bits.append(f"capture_plan={capture_plan_blocker}")

    capture_preflight_ready = summary.get("capture_plan_preflight_ready")
    capture_preflight_blocker = summary.get("capture_plan_preflight_blocker")
    if capture_preflight_ready is True:
        bits.append("capture_preflight=ready")
    elif isinstance(capture_preflight_blocker, str) and capture_preflight_blocker:
        bits.append(f"capture_preflight={capture_preflight_blocker}")

    capture_plan_ready = summary.get("capture_execution_plan_ready")
    capture_plan_blocker = summary.get("capture_execution_plan_blocker")
    if capture_plan_ready is True:
        bits.append("capture_execution_plan=ready")
    elif isinstance(capture_plan_blocker, str) and capture_plan_blocker:
        bits.append(f"capture_execution_plan={capture_plan_blocker}")

    bop_eval_ready = summary.get("ready_for_metrics")
    bop_eval_blocker = summary.get("bop_evaluation_report_blocker")
    if bop_eval_ready is True:
        bits.append("bop_eval=ready")
    elif isinstance(bop_eval_blocker, str) and bop_eval_blocker:
        bits.append(f"bop_eval={bop_eval_blocker}")

    bop_eval_plan_ready = summary.get("bop_evaluation_plan_ready_for_execution")
    bop_eval_plan_blocker = summary.get("bop_evaluation_plan_blocker")
    if bop_eval_plan_ready is True:
        bits.append("bop_eval_plan=ready")
    elif isinstance(bop_eval_plan_blocker, str) and bop_eval_plan_blocker:
        bits.append(f"bop_eval_plan={bop_eval_plan_blocker}")

    sync_ready = summary.get("sync_quality_ready_for_downstream")
    sync_blocker = summary.get("sync_quality_report_blocker")
    if sync_ready is True:
        bits.append("sync_quality=ready")
    elif isinstance(sync_blocker, str) and sync_blocker:
        bits.append(f"sync_quality={sync_blocker}")

    aruco_coverage_ready = summary.get("aruco_coverage_ready_for_downstream")
    aruco_coverage_blocker = summary.get("aruco_coverage_blocker")
    if aruco_coverage_ready is True:
        bits.append("aruco_coverage=ready")
    elif isinstance(aruco_coverage_blocker, str) and aruco_coverage_blocker:
        bits.append(f"aruco_coverage={aruco_coverage_blocker}")

    calibration_preflight_ready = summary.get(
        "calibration_preflight_ready_for_calibrated_stages"
    )
    calibration_preflight_blocker = summary.get("calibration_preflight_blocker")
    if calibration_preflight_ready is True:
        bits.append("calibration_preflight=ready")
    elif (
        isinstance(calibration_preflight_blocker, str)
        and calibration_preflight_blocker
    ):
        bits.append(f"calibration_preflight={calibration_preflight_blocker}")

    observations_ready = summary.get("calibration_observations_ready_for_solver")
    observations_blocker = summary.get("calibration_observations_blocker")
    if observations_ready is True:
        bits.append("calibration_observations=ready")
    elif isinstance(observations_blocker, str) and observations_blocker:
        bits.append(f"calibration_observations={observations_blocker}")

    solver_ready = summary.get("calibration_solver_ready_for_candidates")
    solver_blocker = summary.get("calibration_solver_blocker")
    if solver_ready is True:
        bits.append("calibration_solver=ready")
    elif isinstance(solver_blocker, str) and solver_blocker:
        bits.append(f"calibration_solver={solver_blocker}")

    candidates_ready = summary.get("calibration_candidates_ready_for_validation")
    candidates_blocker = summary.get("calibration_candidates_blocker")
    if candidates_ready is True:
        bits.append("calibration_candidates=ready")
    elif isinstance(candidates_blocker, str) and candidates_blocker:
        bits.append(f"calibration_candidates={candidates_blocker}")

    profile_collection_ready = summary.get(
        "calibration_profile_collection_ready_for_validation"
    )
    profile_collection_blocker = summary.get(
        "calibration_profile_collection_blocker"
    )
    if profile_collection_ready is True:
        bits.append("calibration_profiles=ready")
    elif isinstance(profile_collection_blocker, str) and profile_collection_blocker:
        bits.append(f"calibration_profiles={profile_collection_blocker}")

    validation_ready = summary.get("calibration_validation_ready_for_profiles")
    validation_blocker = summary.get("calibration_validation_blocker")
    if validation_ready is True:
        bits.append("calibration_validation=ready")
    elif isinstance(validation_blocker, str) and validation_blocker:
        bits.append(f"calibration_validation={validation_blocker}")

    bop_results_ready = summary.get("bop_result_export_ready_for_evaluation")
    bop_results_blocker = summary.get("bop_result_export_blocker")
    if bop_results_ready is True:
        bits.append("bop_results=ready")
    elif isinstance(bop_results_blocker, str) and bop_results_blocker:
        bits.append(f"bop_results={bop_results_blocker}")

    bop_export_ready = summary.get("bop_export_ready_for_results")
    bop_export_blocker = summary.get("bop_export_blocker")
    if bop_export_ready is True:
        bits.append("bop_export=ready")
    elif isinstance(bop_export_blocker, str) and bop_export_blocker:
        bits.append(f"bop_export={bop_export_blocker}")

    bop_targets_ready = summary.get("bop_targets_ready_for_evaluation")
    bop_targets_blocker = summary.get("bop_targets_blocker")
    if bop_targets_ready is True:
        bits.append("bop_targets=ready")
    elif isinstance(bop_targets_blocker, str) and bop_targets_blocker:
        bits.append(f"bop_targets={bop_targets_blocker}")

    bop_multiview_ready = summary.get("bop_multiview_targets_ready")
    bop_multiview_blocker = summary.get("bop_multiview_targets_blocker")
    if bop_multiview_ready is True:
        bits.append("bop_multiview=ready")
    elif isinstance(bop_multiview_blocker, str) and bop_multiview_blocker:
        bits.append(f"bop_multiview={bop_multiview_blocker}")

    bop_coco_ready = summary.get("bop_coco_annotations_ready")
    bop_coco_blocker = summary.get("bop_coco_annotations_blocker")
    if bop_coco_ready is True:
        bits.append("bop_coco=ready")
    elif isinstance(bop_coco_blocker, str) and bop_coco_blocker:
        bits.append(f"bop_coco={bop_coco_blocker}")

    bop_models_ready = summary.get("bop_models_info_ready")
    bop_models_blocker = summary.get("bop_models_info_blocker")
    if bop_models_ready is True:
        bits.append("bop_models=ready")
    elif isinstance(bop_models_blocker, str) and bop_models_blocker:
        bits.append(f"bop_models={bop_models_blocker}")

    metric_report_ready = summary.get("metric_report_ready_for_dashboard")
    metric_report_blocker = summary.get("metric_report_blocker")
    if metric_report_ready is True:
        bits.append("metric_report=ready")
    elif isinstance(metric_report_blocker, str) and metric_report_blocker:
        bits.append(f"metric_report={metric_report_blocker}")

    estimator_plan_ready = summary.get("estimator_plan_ready_for_jobs")
    estimator_plan_blocker = summary.get("estimator_plan_blocker")
    if estimator_plan_ready is True:
        bits.append("estimator_plan=ready")
    elif isinstance(estimator_plan_blocker, str) and estimator_plan_blocker:
        bits.append(f"estimator_plan={estimator_plan_blocker}")

    return " · ".join(bits)


def _utc_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).replace(microsecond=0).isoformat()


def _run_root(run_root: str | Path) -> Path:
    return Path(run_root).resolve()


def _relative_to(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ArtifactPathError(
            f"Artifact path is outside run root: {path}"
        ) from exc


def resolve_artifact_path(run_root: str | Path, artifact_path: str | Path) -> Path:
    root = _run_root(run_root)
    raw_path = Path(artifact_path)
    path = raw_path if raw_path.is_absolute() else root / raw_path
    resolved = path.resolve()
    _relative_to(resolved, root)
    return resolved


def _preview_type(path: Path, kind: str) -> str:
    if kind != "file":
        return kind
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix == ".json":
        return "json"
    if suffix in TEXT_SUFFIXES:
        return "text"
    return "binary"


def _safe_json(path: Path) -> object | None:
    try:
        return _json_if_present(path)
    except (OSError, json.JSONDecodeError):
        return None


def _preflight_queue_summary_fields(preflight: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "preflight_exists": preflight["exists"],
        "preflight_status": preflight["overall_status"],
        "preflight_matches_config": preflight["matches_config"],
        "preflight_ready_for_queue": preflight["ready_for_queue"],
        "preflight_queue_blocker": preflight["queue_blocker"],
    }
    if isinstance(preflight.get("error"), str):
        fields["preflight_error"] = preflight["error"]
    return fields


def _json_summary(path: Path) -> dict[str, Any] | None:
    value = _safe_json(path)
    if value is None:
        return None

    if isinstance(value, Mapping):
        if path.name in ACCURACY_ARTIFACTS:
            accuracy_summary = _accuracy_json_summary(value, source_name=path.name)
            if accuracy_summary is not None:
                return accuracy_summary
        if path.name == ALL_RESULTS_JSON:
            combined_summary = _combined_accuracy_json_summary(value)
            if combined_summary is not None:
                return combined_summary

        summary: dict[str, Any] = {
            "type": "json",
            "keys": sorted(str(key) for key in value.keys())[:20],
            "key_count": len(value),
        }
        schema_version = value.get("schema_version")
        if isinstance(schema_version, str):
            summary["schema_version"] = schema_version

        if path.name == REWRITE_GATE_REPORT:
            gate_summary = value.get("summary", {})
            next_blockers = value.get("next_blockers", [])
            checks = value.get("checks", [])
            overall_status = value.get("overall_status")
            gate_blocker = _rewrite_gate_blocker(overall_status)
            blocker_detail_summary = _rewrite_blocker_detail_summary(next_blockers)
            summary.update(
                {
                    "type": "rewrite_gate_report",
                    "gate_id": value.get("gate_id"),
                    "overall_status": overall_status,
                    "rewrite_gate_ready": gate_blocker is None,
                    "rewrite_gate_blocker": gate_blocker,
                    "ready_count": (
                        gate_summary.get("ready_count")
                        if isinstance(gate_summary, Mapping)
                        else None
                    ),
                    "blocked_count": (
                        gate_summary.get("blocked_count")
                        if isinstance(gate_summary, Mapping)
                        else None
                    ),
                    "check_count": (
                        gate_summary.get("check_count")
                        if isinstance(gate_summary, Mapping)
                        else len(checks) if isinstance(checks, list) else 0
                    ),
                    "next_blockers": _rewrite_blocker_names(next_blockers),
                    "next_blocker_messages": _rewrite_blocker_messages(
                        next_blockers
                    ),
                    "next_blocker_diagnostics": blocker_detail_summary[
                        "diagnostics"
                    ],
                    "next_blocker_hints": blocker_detail_summary["hints"],
                    "next_blocker_checks": blocker_detail_summary["blocked_checks"],
                }
            )
        elif path.name == REWRITE_STATUS_REPORT:
            status_summary = value.get("summary", {})
            gates = value.get("gates", [])
            next_gate = value.get("next_gate")
            next_actions = value.get("next_actions")
            next_blockers = value.get("next_blockers")
            overall_status = value.get("overall_status")
            blocker_detail_summary = _rewrite_blocker_detail_summary(next_blockers)
            first_action = (
                next_actions[0]
                if isinstance(next_actions, list)
                and next_actions
                and isinstance(next_actions[0], Mapping)
                else None
            )
            action_items = (
                [
                    action
                    for action in next_actions
                    if isinstance(action, Mapping)
                ][:5]
                if isinstance(next_actions, list)
                else []
            )
            blocked_gate_ids = [
                str(gate.get("gate_id"))
                for gate in gates
                if isinstance(gate, Mapping)
                and gate.get("overall_status") != "ready"
                and gate.get("gate_id")
            ]
            summary.update(
                {
                    "type": "rewrite_status_report",
                    "overall_status": overall_status,
                    "rewrite_status_ready": overall_status == "ready",
                    "rewrite_status_blocker": (
                        None
                        if overall_status == "ready"
                        else "blocked_rewrite_status"
                    ),
                    "gate_count": (
                        status_summary.get("gate_count")
                        if isinstance(status_summary, Mapping)
                        else len(gates) if isinstance(gates, list) else 0
                    ),
                    "ready_gate_count": (
                        status_summary.get("ready_gate_count")
                        if isinstance(status_summary, Mapping)
                        else None
                    ),
                    "blocked_gate_count": (
                        status_summary.get("blocked_gate_count")
                        if isinstance(status_summary, Mapping)
                        else None
                    ),
                    "ready_check_count": (
                        status_summary.get("ready_check_count")
                        if isinstance(status_summary, Mapping)
                        else None
                    ),
                    "check_count": (
                        status_summary.get("check_count")
                        if isinstance(status_summary, Mapping)
                        else None
                    ),
                    "blocked_gate_ids": blocked_gate_ids[:10],
                    "next_blockers": _rewrite_blocker_names(next_blockers),
                    "next_blocker_messages": _rewrite_blocker_messages(
                        next_blockers
                    ),
                    "next_blocker_diagnostics": blocker_detail_summary[
                        "diagnostics"
                    ],
                    "next_blocker_hints": blocker_detail_summary["hints"],
                    "next_blocker_checks": blocker_detail_summary["blocked_checks"],
                    "next_gate_id": (
                        next_gate.get("gate_id")
                        if isinstance(next_gate, Mapping)
                        and isinstance(next_gate.get("gate_id"), str)
                        else None
                    ),
                    "next_gate_run_root": (
                        next_gate.get("run_root")
                        if isinstance(next_gate, Mapping)
                        and isinstance(next_gate.get("run_root"), str)
                        else None
                    ),
                    "next_action_count": (
                        len(next_actions) if isinstance(next_actions, list) else 0
                    ),
                    "next_action_labels": [
                        str(action["label"])
                        for action in action_items
                        if isinstance(action.get("label"), str)
                    ],
                    "next_action_commands": [
                        [str(part) for part in action.get("command")]
                        for action in action_items
                        if isinstance(action.get("command"), list)
                    ],
                    "next_action_blocks_on": [
                        [
                            str(blocker)
                            for blocker in action.get("blocks_on")
                            if isinstance(blocker, str)
                        ]
                        for action in action_items
                        if isinstance(action.get("blocks_on"), list)
                    ],
                    "next_action_label": (
                        first_action.get("label")
                        if isinstance(first_action, Mapping)
                        and isinstance(first_action.get("label"), str)
                        else None
                    ),
                    "next_action_command": (
                        [
                            str(part)
                            for part in first_action.get("command")
                            if isinstance(part, str)
                        ]
                        if isinstance(first_action, Mapping)
                        and isinstance(first_action.get("command"), list)
                        else []
                    ),
                }
            )
        elif path.name == BOP_EXPORT_MANIFEST:
            exports = value.get("exports", [])
            object_models = value.get("object_models", [])
            export_count = len(exports) if isinstance(exports, list) else 0
            summary.update(
                {
                    "type": "bop_export_manifest",
                    "export_count": export_count,
                    "bop_export_ready_for_results": bool(export_count),
                    "bop_export_blocker": _bop_export_blocker(exports),
                    "sensors": [
                        export.get("sensor_name")
                        for export in exports
                        if isinstance(export, Mapping)
                        and isinstance(export.get("sensor_name"), str)
                    ]
                    if isinstance(exports, list)
                    else [],
                }
            )
            if isinstance(object_models, list):
                summary["object_model_count"] = len(object_models)
            summary["has_targets"] = bool(value.get("targets_path"))
            summary["has_multiview_targets"] = bool(
                value.get("multiview_targets_path")
            )
            summary["has_coco_annotations"] = bool(value.get("coco_annotations_path"))
        elif path.name == "models_info.json" and path.parent.name == MODELS_DIR:
            models_info_blocker = _bop_models_info_blocker(value)
            summary.update(
                {
                    "type": "bop_models_info",
                    "model_count": len(value),
                    "bop_models_info_ready": models_info_blocker is None,
                    "bop_models_info_blocker": models_info_blocker,
                    "object_ids": sorted(str(obj_id) for obj_id in value)[:20],
                }
            )
        elif path.name == BOP_COCO_ANNOTATIONS:
            images = value.get("images", [])
            annotations = value.get("annotations", [])
            categories = value.get("categories", [])
            coco_blocker = _bop_coco_annotations_blocker(
                value.get("schema_version"),
                images,
                categories,
                annotations,
            )
            summary.update(
                {
                    "type": "bop_coco_annotations",
                    "bop_coco_annotations_ready": coco_blocker is None,
                    "bop_coco_annotations_blocker": coco_blocker,
                    "image_count": len(images) if isinstance(images, list) else 0,
                    "annotation_count": (
                        len(annotations) if isinstance(annotations, list) else 0
                    ),
                    "category_count": (
                        len(categories) if isinstance(categories, list) else 0
                    ),
                }
            )
        elif path.name == BOP_MULTIVIEW_TARGETS:
            targets = value.get("targets")
            multiview_blocker = _bop_multiview_targets_blocker(
                value.get("schema_version"),
                targets,
            )
            summary.update(
                {
                    "type": "bop_multiview_targets",
                    "bop_multiview_targets_ready": multiview_blocker is None,
                    "bop_multiview_targets_blocker": multiview_blocker,
                    "split": value.get("split"),
                    "scene_count": value.get("scene_count"),
                    "object_count": value.get("object_count"),
                    "target_count": len(targets) if isinstance(targets, list) else 0,
                }
            )
        elif path.name == BOP_RESULT_EXPORT_MANIFEST:
            results = value.get("results", [])
            usable_result_count = 0
            if isinstance(results, list):
                for result in results:
                    if not isinstance(result, Mapping):
                        continue
                    result_path = result.get("path")
                    if isinstance(result_path, str) and Path(result_path).is_file():
                        usable_result_count += 1
            summary.update(
                {
                    "type": "bop_result_export_manifest",
                    "result_count": len(results) if isinstance(results, list) else 0,
                    "usable_result_count": usable_result_count,
                    "bop_result_export_ready_for_evaluation": (
                        usable_result_count > 0
                    ),
                    "bop_result_export_blocker": _bop_result_export_blocker(
                        results,
                        usable_result_count,
                    ),
                    "total_rows": (
                        sum(
                            int(result.get("row_count", 0))
                            for result in results
                            if isinstance(result, Mapping)
                        )
                        if isinstance(results, list)
                        else 0
                    ),
                }
            )
            if isinstance(value.get("dataset_name"), str):
                summary["dataset_name"] = value["dataset_name"]
        elif path.name == PIPELINE_SEQUENCE_PLAN:
            steps = value.get("steps")
            sequence_plan_blocker = _pipeline_sequence_plan_blocker(
                value.get("schema_version"),
                steps,
            )
            summary.update(
                {
                    "type": "pipeline_sequence_plan",
                    "sequence_id": value.get("sequence_id"),
                    "step_count": len(steps) if isinstance(steps, list) else 0,
                    "pipeline_sequence_plan_ready_for_queue": (
                        sequence_plan_blocker is None
                    ),
                    "pipeline_sequence_plan_blocker": sequence_plan_blocker,
                    "steps": [],
                    "plan_only": bool(value.get("plan_only", False)),
                    "resources": [],
                    "calibration_profile_steps": [],
                    "calibration_profile_paths": [],
                }
            )
            if isinstance(steps, list):
                step_mappings = [
                    step for step in steps if isinstance(step, Mapping)
                ]
                calibration_profile_steps = []
                calibration_profile_paths = set()
                for step in step_mappings:
                    options = step.get("options")
                    if not isinstance(options, Mapping):
                        continue
                    calibration_profiles = options.get("calibration_profiles")
                    if not isinstance(calibration_profiles, str):
                        continue
                    step_id = step.get("id")
                    if isinstance(step_id, str):
                        calibration_profile_steps.append(step_id)
                    calibration_profile_paths.add(calibration_profiles)
                resources = value.get("resources", [])
                summary.update(
                    {
                        "steps": [
                            step.get("id")
                            for step in step_mappings
                            if isinstance(step.get("id"), str)
                        ],
                        "resources": [
                            str(resource)
                            for resource in resources
                            if isinstance(resource, str)
                        ],
                        "calibration_profile_steps": calibration_profile_steps,
                        "calibration_profile_paths": sorted(
                            calibration_profile_paths
                        ),
                    }
                )
        elif path.name in ESTIMATOR_PLAN_ARTIFACTS:
            expected_estimator_id = ESTIMATOR_PLAN_ARTIFACTS[path.name]
            jobs = value.get("jobs")
            job_mappings = []
            if isinstance(jobs, list):
                job_mappings = [job for job in jobs if isinstance(job, Mapping)]
            command = value.get("command", [])
            estimator_id = value.get("estimator_id")
            if not isinstance(estimator_id, str):
                estimator_id = expected_estimator_id
            estimator_plan_blocker = _estimator_plan_blocker(
                jobs,
                estimator_id,
                expected_estimator_id,
            )
            summary.update(
                {
                    "type": "estimator_plan",
                    "estimator_id": estimator_id,
                    "dry_run": bool(value.get("dry_run", False)),
                    "object_id": value.get("object_id"),
                    "job_count": len(jobs) if isinstance(jobs, list) else 0,
                    "estimator_plan_ready_for_jobs": (
                        estimator_plan_blocker is None
                    ),
                    "estimator_plan_blocker": estimator_plan_blocker,
                    "sensor_names": [
                        job.get("sensor_name")
                        for job in job_mappings
                        if isinstance(job.get("sensor_name"), str)
                    ],
                    "object_names": sorted(
                        {
                            str(job.get("object_name"))
                            for job in job_mappings
                            if isinstance(job.get("object_name"), str)
                        }
                    ),
                    "command_uses_uv": (
                        isinstance(command, list)
                        and len(command) >= 2
                        and command[0] == "uv"
                        and command[1] == "run"
                    ),
                }
            )
            options = value.get("options")
            if isinstance(options, Mapping):
                summary["option_keys"] = sorted(str(key) for key in options.keys())
            for optional_key in (
                "result_id",
                "wrapper_script",
                "wrapper_exists",
                "foundationpose_folder",
                "no_tracking",
                "est_refine_iter",
                "track_refine_iter",
            ):
                if optional_key in value:
                    summary[optional_key] = value.get(optional_key)
        elif path.name == METRIC_REPORT_JSON:
            dashboard = value.get("dashboard")
            rows = value.get("rows", [])
            row_count = len(rows) if isinstance(rows, list) else 0
            metric_report_blocker = _metric_report_blocker(rows, dashboard)
            summary.update(
                {
                    "type": "metric_report",
                    "row_count": row_count,
                    "metric_report_ready_for_dashboard": (
                        metric_report_blocker is None
                    ),
                    "metric_report_blocker": metric_report_blocker,
                    "metric_artifact_count": (
                        dashboard.get("metric_artifact_count")
                        if isinstance(dashboard, Mapping)
                        else None
                    ),
                    "direct_method_count": (
                        dashboard.get("direct_method_count")
                        if isinstance(dashboard, Mapping)
                        else None
                    ),
                    "combined_group_count": (
                        dashboard.get("combined_group_count")
                        if isinstance(dashboard, Mapping)
                        else None
                    ),
                    "best_by_AP_p": (
                        dashboard.get("best_by_AP_p")
                        if isinstance(dashboard, Mapping)
                        else None
                    ),
                }
            )
        elif path.name == ARUCO_COVERAGE_REPORT:
            checks = value.get("checks", [])
            sensors = value.get("sensors", [])
            overall_status = value.get("overall_status")
            aruco_coverage_ready = overall_status in {"ok", "warning"}
            summary.update(
                {
                    "type": "aruco_coverage_report",
                    "overall_status": overall_status,
                    "aruco_coverage_ready_for_downstream": (
                        aruco_coverage_ready
                    ),
                    "aruco_coverage_blocker": (
                        None
                        if aruco_coverage_ready
                        else _aruco_coverage_report_blocker(overall_status)
                    ),
                    "sensor_count": value.get("sensor_count"),
                    "frame_count": value.get("frame_count"),
                    "detected_frame_count": value.get("detected_frame_count"),
                    "valid_pose_count": value.get("valid_pose_count"),
                    "detection_ratio": value.get("detection_ratio"),
                    "valid_pose_ratio": value.get("valid_pose_ratio"),
                    "min_marker_count": value.get("min_marker_count"),
                    "check_count": len(checks) if isinstance(checks, list) else 0,
                    "sensor_names": [
                        sensor.get("sensor_name")
                        for sensor in sensors
                        if isinstance(sensor, Mapping)
                        and isinstance(sensor.get("sensor_name"), str)
                    ],
                }
            )
        elif path.name == RUN_CONFIG:
            run_config_ready, run_config_blocker, run_config_error = (
                _run_config_readiness(value)
            )
            capture = value.get("capture")
            pipeline = value.get("pipeline")
            robot = value.get("robot_profile")
            calibration_profiles = value.get("calibration_profiles")
            sensors = (
                capture.get("sensors", [])
                if isinstance(capture, Mapping)
                else []
            )
            summary.update(
                {
                    "type": "run_config",
                    "run_name": value.get("run_name"),
                    "object_folder": value.get("object_folder"),
                    "run_config_ready_for_pipeline": run_config_ready,
                    "run_config_blocker": run_config_blocker,
                    "calibration_profiles": calibration_profiles,
                    "has_calibration_profiles": (
                        isinstance(calibration_profiles, str)
                        and bool(calibration_profiles.strip())
                    ),
                    "robot_mode": (
                        robot.get("mode")
                        if isinstance(robot, Mapping)
                        else None
                    ),
                    "sensor_count": len(sensors) if isinstance(sensors, list) else 0,
                    "sequence_id": (
                        pipeline.get("sequence_id")
                        if isinstance(pipeline, Mapping)
                        else None
                    ),
                    "plan_only": (
                        bool(pipeline.get("plan_only", False))
                        if isinstance(pipeline, Mapping)
                        else False
                    ),
                }
            )
            if run_config_error is not None:
                summary["run_config_error"] = run_config_error
            if run_config_ready:
                preflight = run_preflight_queue_summary(path.parent, value)
                summary.update(_preflight_queue_summary_fields(preflight))
            else:
                summary.update(
                    {
                        "preflight_exists": (path.parent / RUN_PREFLIGHT_REPORT).is_file(),
                        "preflight_status": None,
                        "preflight_matches_config": None,
                        "preflight_ready_for_queue": False,
                        "preflight_queue_blocker": run_config_blocker,
                    }
                )
        elif path.name == RUN_PREFLIGHT_REPORT:
            checks = value.get("checks", [])
            sequence_plan = value.get("sequence_plan")
            config = value.get("config")
            robot_status = value.get("robot_status")
            sensor_status = value.get("sensor_status")
            runtime_status = value.get("runtime_status")
            check_mappings = [
                check for check in checks if isinstance(check, Mapping)
            ] if isinstance(checks, list) else []
            check_status_counts: dict[str, int] = {}
            for check in check_mappings:
                status = check.get("status")
                if isinstance(status, str):
                    check_status_counts[status] = (
                        check_status_counts.get(status, 0) + 1
                    )
            robot_profile = (
                config.get("robot_profile")
                if isinstance(config, Mapping)
                else None
            )
            selected_robot_profile = (
                robot_status.get("selected_profile")
                if isinstance(robot_status, Mapping)
                else None
            )
            summary.update(
                {
                    "type": "run_preflight_report",
                    "overall_status": value.get("overall_status"),
                    "check_count": len(check_mappings),
                    "check_status_counts": check_status_counts,
                    "sequence_id": (
                        sequence_plan.get("sequence_id")
                        if isinstance(sequence_plan, Mapping)
                        else None
                    ),
                    "step_count": (
                        len(sequence_plan.get("steps", []))
                        if isinstance(sequence_plan, Mapping)
                        and isinstance(sequence_plan.get("steps"), list)
                        else 0
                    ),
                    "robot_mode": (
                        robot_profile.get("mode")
                        if isinstance(robot_profile, Mapping)
                        else None
                    ),
                    "selected_robot_mode": (
                        selected_robot_profile.get("mode")
                        if isinstance(selected_robot_profile, Mapping)
                        else None
                    ),
                    "sensor_status_included": sensor_status is not None,
                    "runtime_status_included": runtime_status is not None,
                }
            )
            if isinstance(sensor_status, Mapping):
                summary["total_connected_sensors"] = sensor_status.get(
                    "total_connected"
                )
                summary["all_expected_connected"] = sensor_status.get(
                    "all_expected_connected"
                )
            if isinstance(runtime_status, Mapping):
                summary["available_runtime_count"] = runtime_status.get(
                    "available_count"
                )
                summary["runtime_count"] = runtime_status.get("runtime_count")
            current_config = _safe_json(path.parent / RUN_CONFIG)
            if isinstance(current_config, Mapping):
                summary.update(
                    _preflight_queue_summary_fields(
                        run_preflight_queue_summary(path.parent, current_config)
                    )
                )
            else:
                summary.update(
                    {
                        "preflight_exists": True,
                        "preflight_status": value.get("overall_status"),
                        "preflight_matches_config": None,
                        "preflight_ready_for_queue": False,
                        "preflight_queue_blocker": "missing_run_config",
                    }
                )
        elif path.name == HARDWARE_STATUS_REPORT:
            checks = value.get("checks", [])
            robot_status = value.get("robot_status")
            sensor_status = value.get("sensor_status")
            runtime_status = value.get("runtime_status")
            overall_status = value.get("overall_status")
            hardware_status_blocker = _hardware_status_report_blocker(
                overall_status
            )
            selected = (
                robot_status.get("selected_profile")
                if isinstance(robot_status, Mapping)
                else None
            )
            summary.update(
                {
                    "type": "hardware_status_report",
                    "overall_status": overall_status,
                    "hardware_status_ready_for_capture": (
                        hardware_status_blocker is None
                    ),
                    "hardware_status_blocker": hardware_status_blocker,
                    "check_count": len(checks) if isinstance(checks, list) else 0,
                    "robot_mode": (
                        selected.get("mode") if isinstance(selected, Mapping) else None
                    ),
                    "total_connected_sensors": (
                        sensor_status.get("total_connected")
                        if isinstance(sensor_status, Mapping)
                        else None
                    ),
                    "all_expected_connected": (
                        sensor_status.get("all_expected_connected")
                        if isinstance(sensor_status, Mapping)
                        else None
                    ),
                    "available_runtime_count": (
                        runtime_status.get("available_count")
                        if isinstance(runtime_status, Mapping)
                        else None
                    ),
                    "runtime_count": (
                        runtime_status.get("runtime_count")
                        if isinstance(runtime_status, Mapping)
                        else None
                    ),
                }
            )
        elif path.name == CAPTURE_PLAN:
            commands = value.get("commands")
            sensors = value.get("sensors", [])
            robot = value.get("robot_profile")
            capture_plan_blocker = _capture_plan_blocker(
                value.get("schema_version"),
                commands,
            )
            summary.update(
                {
                    "type": "capture_plan",
                    "dry_run": bool(value.get("dry_run", False)),
                    "command_count": len(commands)
                    if isinstance(commands, list)
                    else 0,
                    "capture_plan_ready_for_preflight": (
                        capture_plan_blocker is None
                    ),
                    "capture_plan_blocker": capture_plan_blocker,
                    "sensor_count": len(sensors) if isinstance(sensors, list) else 0,
                    "robot_mode": (
                        robot.get("mode")
                        if isinstance(robot, Mapping)
                        else None
                    ),
                    "roles": [
                        command.get("role")
                        for command in (commands if isinstance(commands, list) else [])
                        if isinstance(command, Mapping)
                        and isinstance(command.get("role"), str)
                    ],
                }
            )
        elif path.name == CAPTURE_REHEARSAL_REPORT:
            status = value.get("status")
            raw_pose_count = value.get("raw_pose_count")
            capture_rehearsal_blocker = _capture_rehearsal_report_blocker(
                status,
                raw_pose_count,
            )
            summary.update(
                {
                    "type": "capture_rehearsal_report",
                    "status": status,
                    "mode": value.get("mode"),
                    "raw_pose_count": raw_pose_count,
                    "capture_rehearsal_ready_for_sync": (
                        capture_rehearsal_blocker is None
                    ),
                    "capture_rehearsal_blocker": capture_rehearsal_blocker,
                }
            )
        elif path.name == CAPTURE_PLAN_PREFLIGHT_REPORT:
            checks = value.get("checks", [])
            overall_status = value.get("overall_status")
            preflight_ready = overall_status in {"ok", "warning"}
            summary.update(
                {
                    "type": "capture_plan_preflight_report",
                    "overall_status": overall_status,
                    "capture_plan_preflight_ready": preflight_ready,
                    "capture_plan_preflight_blocker": (
                        None
                        if preflight_ready
                        else _capture_plan_preflight_blocker(overall_status)
                    ),
                    "check_count": len(checks) if isinstance(checks, list) else 0,
                }
            )
        elif path.name == CAPTURE_EXECUTION_PLAN:
            selected = value.get("selected_commands", [])
            skipped = value.get("skipped_commands", [])
            status = value.get("status")
            ready_to_execute = bool(value.get("ready_to_execute", False))
            problem_checks = _problem_checks_from_report(value)
            plan_summary = {
                "type": "capture_execution_plan",
                "status": status,
                "mode": value.get("mode"),
                "ready_to_execute": ready_to_execute,
                "capture_execution_plan_ready": ready_to_execute,
                "capture_execution_plan_blocker": (
                    _capture_execution_plan_blocker(status, ready_to_execute)
                ),
                "selected_count": len(selected) if isinstance(selected, list) else 0,
                "skipped_count": len(skipped) if isinstance(skipped, list) else 0,
                "selected_roles": value.get("selected_roles", []),
            }
            if problem_checks:
                plan_summary["blocked_checks"] = problem_checks
                plan_summary["blocked_check_messages"] = [
                    str(check["message"])
                    for check in problem_checks
                    if check.get("message")
                ]
            summary.update(plan_summary)
        elif path.name == CAPTURE_EXECUTION_REPORT:
            processes = value.get("processes", [])
            status = value.get("status")
            ready_for_downstream = status == "succeeded"
            process_status_counts: dict[str, int] = {}
            termination_reason_counts: dict[str, int] = {}
            elapsed_values: list[float] = []
            if isinstance(processes, list):
                for process in processes:
                    if isinstance(process, Mapping):
                        status = str(process.get("status", "unknown"))
                        process_status_counts[status] = (
                            process_status_counts.get(status, 0) + 1
                        )
                        reason = process.get("termination_reason")
                        if reason:
                            reason_key = str(reason)
                            termination_reason_counts[reason_key] = (
                                termination_reason_counts.get(reason_key, 0) + 1
                            )
                        elapsed = process.get("elapsed_s")
                        if isinstance(elapsed, (int, float)):
                            elapsed_values.append(float(elapsed))
            summary.update(
                {
                    "type": "capture_execution_report",
                    "status": status,
                    "ready_for_downstream": ready_for_downstream,
                    "capture_execution_report_blocker": (
                        None
                        if ready_for_downstream
                        else _capture_execution_report_blocker(status)
                    ),
                    "mode": value.get("mode"),
                    "raw_pose_count": value.get("raw_pose_count"),
                    "process_count": (
                        len(processes) if isinstance(processes, list) else 0
                    ),
                    "process_status_counts": process_status_counts,
                    "termination_reason_counts": termination_reason_counts,
                    "processes_with_timing": len(elapsed_values),
                    "max_process_elapsed_s": (
                        max(elapsed_values) if elapsed_values else None
                    ),
                }
            )
        elif path.name == CAPTURE_EXECUTION_STATUS:
            processes = value.get("processes", [])
            active_roles: list[str] = []
            process_status_counts: dict[str, int] = {}
            summary.update(
                {
                    "type": "capture_execution_status",
                    "status": value.get("status"),
                    "mode": value.get("mode"),
                    "active_process_count": value.get("active_process_count"),
                    "process_count": (
                        len(processes) if isinstance(processes, list) else 0
                    ),
                    "raw_pose_count": value.get("raw_pose_count"),
                    "selected_roles": value.get("selected_roles", []),
                }
            )
            if isinstance(processes, list):
                for process in processes:
                    if not isinstance(process, Mapping):
                        continue
                    status = str(process.get("status", "unknown"))
                    process_status_counts[status] = (
                        process_status_counts.get(status, 0) + 1
                    )
                    if process.get("active"):
                        active_roles.append(str(process.get("role") or ""))
            summary["active_roles"] = active_roles
            summary["process_status_counts"] = process_status_counts
        elif path.name == CALIBRATION_PREFLIGHT_REPORT:
            checks = value.get("checks", [])
            overall_status = value.get("overall_status")
            preflight_ready = overall_status in {"ok", "warning"}
            check_status_counts: dict[str, int] = {}
            if isinstance(checks, list):
                for check in checks:
                    if not isinstance(check, Mapping):
                        continue
                    status = str(check.get("status", "unknown"))
                    check_status_counts[status] = (
                        check_status_counts.get(status, 0) + 1
                    )
            matched_sensors = value.get("matched_sensors", [])
            matched_profile_ids = []
            if isinstance(matched_sensors, list):
                matched_profile_ids = sorted(
                    {
                        str(match.get("profile_id"))
                        for match in matched_sensors
                        if isinstance(match, Mapping)
                        and isinstance(match.get("profile_id"), str)
                    }
                )
            summary.update(
                {
                    "type": "calibration_preflight_report",
                    "overall_status": overall_status,
                    "calibration_preflight_ready_for_calibrated_stages": (
                        preflight_ready
                    ),
                    "calibration_preflight_blocker": (
                        None
                        if preflight_ready
                        else _calibration_preflight_blocker(overall_status)
                    ),
                    "profile_path": value.get("profile_path"),
                    "profile_count": value.get("profile_count"),
                    "sensor_count": value.get("sensor_count"),
                    "matched_sensor_count": value.get("matched_sensor_count"),
                    "check_count": len(checks) if isinstance(checks, list) else 0,
                    "check_status_counts": check_status_counts,
                    "matched_profile_ids": matched_profile_ids,
                    "require_valid": bool(value.get("require_valid", False)),
                    "min_observations": value.get("min_observations"),
                    "max_mean_reprojection_error_px": value.get(
                        "max_mean_reprojection_error_px"
                    ),
                }
            )
        elif path.name == CALIBRATION_OBSERVATIONS:
            checks = value.get("checks", [])
            overall_status = value.get("overall_status")
            observations_ready = overall_status in {"ok", "warning"}
            target = value.get("target")
            if not isinstance(target, Mapping):
                target = value.get("board")
            if not isinstance(target, Mapping):
                target = {}
            summary.update(
                {
                    "type": "calibration_observations",
                    "overall_status": overall_status,
                    "calibration_observations_ready_for_solver": (
                        observations_ready
                    ),
                    "calibration_observations_blocker": (
                        None
                        if observations_ready
                        else _calibration_observations_blocker(overall_status)
                    ),
                    "target_type": target.get("target_type"),
                    "dictionary": target.get("dictionary"),
                    "sensor_count": value.get("sensor_count"),
                    "frame_count": value.get("frame_count"),
                    "observation_count": value.get("observation_count"),
                    "rejected_count": value.get("rejected_count"),
                    "motion_count": value.get("motion_count"),
                    "check_count": len(checks) if isinstance(checks, list) else 0,
                }
            )
        elif path.name == CALIBRATION_CANDIDATES:
            checks = value.get("checks", [])
            overall_status = value.get("overall_status")
            candidates_ready = overall_status in {"ok", "warning"}
            summary.update(
                {
                    "type": "calibration_candidates",
                    "overall_status": overall_status,
                    "calibration_candidates_ready_for_validation": (
                        candidates_ready
                    ),
                    "calibration_candidates_blocker": (
                        None
                        if candidates_ready
                        else _calibration_candidates_blocker(overall_status)
                    ),
                    "sensor_count": value.get("sensor_count"),
                    "profile_count": value.get("profile_count"),
                    "candidate_count": value.get("candidate_count"),
                    "inlier_count": value.get("inlier_count"),
                    "outlier_count": value.get("outlier_count"),
                    "check_count": len(checks) if isinstance(checks, list) else 0,
                }
            )
        elif path.name == CALIBRATION_PROFILES_FROM_OBSERVATIONS:
            profiles = value.get("profiles", [])
            profile_collection_blocker = _calibration_profile_collection_blocker(
                value.get("schema_version"),
                profiles,
            )
            statuses = [
                profile.get("status")
                for profile in profiles
                if isinstance(profile, Mapping)
            ]
            summary.update(
                {
                    "type": "calibration_profiles_from_observations",
                    "profile_count": len(profiles) if isinstance(profiles, list) else 0,
                    "calibration_profile_collection_ready_for_validation": (
                        profile_collection_blocker is None
                    ),
                    "calibration_profile_collection_blocker": (
                        profile_collection_blocker
                    ),
                    "statuses": sorted({str(status) for status in statuses if status}),
                }
            )
        elif path.name == CALIBRATION_SOLVER_REPORT:
            checks = value.get("checks", [])
            overall_status = value.get("overall_status")
            solver_ready = overall_status in {"ok", "warning"}
            solver_summary = {
                "type": "calibration_solver_report",
                "overall_status": overall_status,
                "calibration_solver_ready_for_candidates": solver_ready,
                "calibration_solver_blocker": (
                    None
                    if solver_ready
                    else _calibration_solver_blocker(overall_status)
                ),
                "sensor_count": value.get("sensor_count"),
                "profile_count": value.get("profile_count"),
                "observation_count": value.get("observation_count"),
                "inlier_count": value.get("inlier_count"),
                "outlier_count": value.get("outlier_count"),
                "hand_eye_method": value.get("hand_eye_method"),
                "check_count": len(checks) if isinstance(checks, list) else 0,
            }
            if "holdout_fraction" in value:
                solver_summary["holdout_fraction"] = value.get("holdout_fraction")
            comparisons = value.get("method_comparisons", [])
            if "method_comparisons" in value and isinstance(comparisons, list):
                solver_summary["method_comparison_count"] = len(comparisons)
                comparison_statuses = [
                    comparison.get("status")
                    for comparison in comparisons
                    if isinstance(comparison, Mapping)
                ]
                solver_summary["method_comparison_statuses"] = sorted(
                    {
                        str(status)
                        for status in comparison_statuses
                        if status is not None
                    }
                )
            summary.update(
                solver_summary
            )
        elif path.name == CALIBRATION_PROFILES_SOLVED:
            profiles = value.get("profiles", [])
            profile_collection_blocker = _calibration_profile_collection_blocker(
                value.get("schema_version"),
                profiles,
            )
            statuses = [
                profile.get("status")
                for profile in profiles
                if isinstance(profile, Mapping)
            ]
            methods = [
                profile.get("method")
                for profile in profiles
                if isinstance(profile, Mapping)
            ]
            summary.update(
                {
                    "type": "calibration_profiles_solved",
                    "profile_count": len(profiles) if isinstance(profiles, list) else 0,
                    "calibration_profile_collection_ready_for_validation": (
                        profile_collection_blocker is None
                    ),
                    "calibration_profile_collection_blocker": (
                        profile_collection_blocker
                    ),
                    "statuses": sorted({str(status) for status in statuses if status}),
                    "methods": sorted({str(method) for method in methods if method}),
                }
            )
        elif path.name == CALIBRATION_VALIDATION_REPORT:
            checks = value.get("checks", [])
            promotion = value.get("promotion")
            overall_status = value.get("overall_status")
            validation_ready = overall_status in {"ok", "warning"}
            summary.update(
                {
                    "type": "calibration_validation_report",
                    "overall_status": overall_status,
                    "calibration_validation_ready_for_profiles": (
                        validation_ready
                    ),
                    "calibration_validation_blocker": (
                        None
                        if validation_ready
                        else _calibration_validation_blocker(overall_status)
                    ),
                    "profile_count": value.get("profile_count"),
                    "promotable_profile_count": value.get("promotable_profile_count"),
                    "candidate_count": value.get("candidate_count"),
                    "inlier_count": value.get("inlier_count"),
                    "outlier_count": value.get("outlier_count"),
                    "promoted": (
                        bool(promotion.get("promoted", False))
                        if isinstance(promotion, Mapping)
                        else False
                    ),
                    "check_count": len(checks) if isinstance(checks, list) else 0,
                }
            )
        elif path.name == SYNC_QUALITY_REPORT:
            checks = value.get("checks", [])
            sensors = value.get("sensors", [])
            overall_status = value.get("overall_status")
            sync_quality_ready = overall_status in {"ok", "warning"}
            summary.update(
                {
                    "type": "sync_quality_report",
                    "overall_status": overall_status,
                    "sync_quality_ready_for_downstream": sync_quality_ready,
                    "sync_quality_report_blocker": (
                        None
                        if sync_quality_ready
                        else _sync_quality_report_blocker(overall_status)
                    ),
                    "sensor_count": value.get("sensor_count"),
                    "total_frames": value.get("total_frames"),
                    "matched_frames": value.get("matched_frames"),
                    "dropped_frames": value.get("dropped_frames"),
                    "overall_match_ratio": value.get("overall_match_ratio"),
                    "check_count": len(checks) if isinstance(checks, list) else 0,
                    "sensor_names": [
                        sensor.get("sensor_name")
                        for sensor in sensors
                        if isinstance(sensor, Mapping)
                        and isinstance(sensor.get("sensor_name"), str)
                    ],
                }
            )
        elif path.name == BOP_EVALUATION_PLAN:
            result = value.get("result")
            command = value.get("command")
            environment = value.get("environment")
            plan_blocker = _bop_evaluation_plan_blocker(
                value.get("schema_version"),
                result,
                command,
                environment,
            )
            summary.update(
                {
                    "type": "bop_evaluation_plan",
                    "dry_run": bool(value.get("dry_run", False)),
                    "bop_evaluation_plan_ready_for_execution": (
                        plan_blocker is None
                    ),
                    "bop_evaluation_plan_blocker": plan_blocker,
                    "result_filename": (
                        result.get("filename")
                        if isinstance(result, Mapping)
                        else None
                    ),
                    "result_path": (
                        result.get("path") if isinstance(result, Mapping) else None
                    ),
                    "command_count": len(command) if isinstance(command, list) else 0,
                    "bop_path": (
                        environment.get("BOP_PATH")
                        if isinstance(environment, Mapping)
                        else None
                    ),
                }
            )
        elif path.name == BOP_EVALUATION_REPORT:
            result = value.get("result")
            status = value.get("status")
            checks = value.get("checks", [])
            critical_failed_check_count, critical_missing_check_count = (
                _bop_evaluation_critical_check_counts(checks)
            )
            ready_for_metrics = (
                status in {"planned", "succeeded"}
                and critical_failed_check_count == 0
                and critical_missing_check_count == 0
            )
            if status not in {"planned", "succeeded"}:
                evaluation_blocker = _bop_evaluation_report_blocker(status)
            elif critical_failed_check_count or critical_missing_check_count:
                evaluation_blocker = "failed_bop_evaluation_prerequisites"
            else:
                evaluation_blocker = _bop_evaluation_report_blocker(status)
            output_artifacts = value.get("output_artifacts", [])
            score_summary = value.get("score_summary", {})
            score_metrics = (
                score_summary.get("metrics", {})
                if isinstance(score_summary, Mapping)
                else {}
            )
            score_file_count = (
                score_summary.get("score_file_count")
                if isinstance(score_summary, Mapping)
                else None
            )
            summary.update(
                {
                    "type": "bop_evaluation_report",
                    "status": status,
                    "ready_for_metrics": ready_for_metrics,
                    "bop_evaluation_report_blocker": (
                        None
                        if ready_for_metrics
                        else evaluation_blocker
                    ),
                    "dry_run": bool(value.get("dry_run", False)),
                    "result_filename": (
                        result.get("filename")
                        if isinstance(result, Mapping)
                        else None
                    ),
                    "check_count": len(checks) if isinstance(checks, list) else 0,
                    "failed_check_count": (
                        sum(
                            1
                            for check in checks
                            if isinstance(check, Mapping)
                            and not bool(check.get("ok", False))
                        )
                        if isinstance(checks, list)
                        else 0
                    ),
                    "critical_failed_check_count": critical_failed_check_count,
                    "critical_missing_check_count": critical_missing_check_count,
                    "output_artifact_count": (
                        len(output_artifacts)
                        if isinstance(output_artifacts, list)
                        else 0
                    ),
                }
            )
            if isinstance(score_file_count, int):
                summary["score_file_count"] = score_file_count
            if isinstance(score_metrics, Mapping):
                summary["score_metric_count"] = len(score_metrics)
                summary["score_metrics"] = dict(score_metrics)
                if "bop19_average_recall" in score_metrics:
                    summary["bop19_average_recall"] = score_metrics[
                        "bop19_average_recall"
                    ]
        elif path.name == DATASET_MANIFEST:
            stages = value.get("stages", [])
            sensors = value.get("sensors", [])
            summary.update(
                {
                    "type": "dataset_manifest",
                    "run_id": value.get("run_id"),
                    "stage_count": len(stages) if isinstance(stages, list) else 0,
                    "sensor_count": len(sensors) if isinstance(sensors, list) else 0,
                }
            )
        return summary

    if isinstance(value, list):
        if path.name == BOP_TARGETS_BOP19:
            targets_blocker = _bop_targets_blocker(value)
            return {
                "type": "bop_targets_bop19",
                "item_count": len(value),
                "target_count": len(value),
                "bop_targets_ready_for_evaluation": targets_blocker is None,
                "bop_targets_blocker": targets_blocker,
                "scene_count": len(
                    {
                        int(target["scene_id"])
                        for target in value
                        if isinstance(target, Mapping)
                        and "scene_id" in target
                        and str(target["scene_id"]).lstrip("-").isdigit()
                    }
                ),
                "object_count": len(
                    {
                        int(target["obj_id"])
                        for target in value
                        if isinstance(target, Mapping)
                        and "obj_id" in target
                        and str(target["obj_id"]).lstrip("-").isdigit()
                    }
                ),
            }
        return {"type": "json_list", "item_count": len(value)}

    return {"type": "json_scalar", "value_type": type(value).__name__}


def _csv_summary(path: Path, *, max_header_columns: int = 50) -> dict[str, Any] | None:
    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            row_count = sum(1 for row in reader if row)
    except OSError:
        return None
    return {
        "type": "csv",
        "columns": header[:max_header_columns],
        "column_count": len(header),
        "row_count": row_count,
    }


def _image_summary(path: Path) -> dict[str, Any] | None:
    image = cv2.imread(path.as_posix(), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        height, width = image.shape
        channels = 1
    else:
        height, width, channels = image.shape
    return {
        "type": "image",
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
        "dtype": str(image.dtype),
    }


def _metric_number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _metric_value(value: object) -> float | list[float] | None:
    number = _metric_number(value)
    if number is not None:
        return number
    if isinstance(value, list):
        numbers = [
            item
            for item in (_metric_number(item) for item in value)
            if item is not None
        ]
        return numbers[:2] if numbers else None
    return None


def _is_motion_metrics(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return any(key in value for key in METRIC_KEYS)


def _motion_sample_count(value: Mapping[str, object]) -> int | None:
    counts = [
        len(samples)
        for key in RAW_SAMPLE_KEYS
        if isinstance((samples := value.get(key)), list)
    ]
    return max(counts) if counts else None


def _motion_metric_summary(value: Mapping[str, object]) -> dict[str, Any]:
    metrics = {}
    for key in METRIC_KEYS:
        metric_value = _metric_value(value.get(key))
        if metric_value is not None:
            metrics[key] = metric_value
    summary: dict[str, Any] = {"metrics": metrics}
    sample_count = _motion_sample_count(value)
    if sample_count is not None:
        summary["sample_count"] = sample_count
    return summary


def _direct_accuracy_methods(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return []

    methods = []
    for method_name, method_data in value.items():
        if not isinstance(method_data, Mapping):
            continue
        motion_items = {
            str(motion_name): motion_data
            for motion_name, motion_data in method_data.items()
            if _is_motion_metrics(motion_data)
        }
        if not motion_items:
            continue

        all_motions = None
        if isinstance(motion_items.get("all_motions"), Mapping):
            all_motions = _motion_metric_summary(motion_items["all_motions"])

        method_summary: dict[str, Any] = {
            "name": str(method_name),
            "motion_count": len(motion_items),
            "motions": sorted(motion_items.keys()),
        }
        if all_motions is not None:
            method_summary["all_motions"] = all_motions["metrics"]
            if "sample_count" in all_motions:
                method_summary["sample_count"] = all_motions["sample_count"]
        methods.append(method_summary)
    return methods


def _best_by_ap(methods: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    best = None
    for method in methods:
        all_motions = method.get("all_motions")
        if not isinstance(all_motions, Mapping):
            continue
        ap_p = _metric_number(all_motions.get("AP_p"))
        if ap_p is None:
            continue
        if best is None or ap_p < best["AP_p"]:
            best = {"method": method["name"], "AP_p": ap_p}
    return best


def _accuracy_json_summary(value: object, *, source_name: str) -> dict[str, Any] | None:
    methods = _direct_accuracy_methods(value)
    if not methods:
        return None

    summary: dict[str, Any] = {
        "type": "pose_accuracy_metrics",
        "source_name": source_name,
        "method_count": len(methods),
        "methods": methods,
    }
    best = _best_by_ap(methods)
    if best is not None:
        summary["best_by_AP_p"] = best
    return summary


def _combined_accuracy_groups(value: object) -> list[dict[str, Any]]:
    groups = []

    def walk(item: object, path: tuple[str, ...]) -> None:
        methods = _direct_accuracy_methods(item)
        if methods:
            group: dict[str, Any] = {
                "context": "/".join(path) if path else "",
                "method_count": len(methods),
                "methods": [method["name"] for method in methods],
            }
            best = _best_by_ap(methods)
            if best is not None:
                group["best_by_AP_p"] = best
            groups.append(group)
            return

        if isinstance(item, Mapping):
            for key, value in item.items():
                walk(value, (*path, str(key)))
        elif isinstance(item, list):
            for index, value in enumerate(item):
                walk(value, (*path, str(index)))

    walk(value, ())
    return groups


def _combined_accuracy_json_summary(value: object) -> dict[str, Any] | None:
    groups = _combined_accuracy_groups(value)
    if not groups:
        return None

    method_names = sorted(
        {
            method_name
            for group in groups
            for method_name in group.get("methods", [])
            if isinstance(method_name, str)
        }
    )
    summary: dict[str, Any] = {
        "type": "combined_pose_accuracy_metrics",
        "experiment_count": len(value) if isinstance(value, Mapping) else None,
        "result_group_count": len(groups),
        "method_count": sum(int(group["method_count"]) for group in groups),
        "methods": method_names,
        "groups": groups[:20],
    }
    candidates = [
        group["best_by_AP_p"]
        for group in groups
        if isinstance(group.get("best_by_AP_p"), Mapping)
    ]
    best = min(candidates, key=lambda item: item["AP_p"]) if candidates else None
    if best is not None:
        summary["best_by_AP_p"] = best
    return summary


def _bop_scene_summary(path: Path) -> dict[str, Any] | None:
    scene_camera = _safe_json(path / "scene_camera.json")
    scene_gt = _safe_json(path / "scene_gt.json")
    if not isinstance(scene_camera, Mapping):
        return None

    rgb_count = (
        len(list((path / RGB_DIR).glob("*.png"))) if (path / RGB_DIR).is_dir() else 0
    )
    depth_count = (
        len(list((path / DEPTH_DIR).glob("*.png"))) if (path / DEPTH_DIR).is_dir() else 0
    )
    annotation_count = 0
    if isinstance(scene_gt, Mapping):
        for annotations in scene_gt.values():
            if isinstance(annotations, list):
                annotation_count += len(annotations)

    return {
        "type": "bop_scene",
        "image_count": len(scene_camera),
        "rgb_count": rgb_count,
        "depth_count": depth_count,
        "annotation_count": annotation_count,
        "has_scene_gt_info": (path / "scene_gt_info.json").is_file(),
        "has_mask": (path / "mask").is_dir(),
        "has_mask_visib": (path / "mask_visib").is_dir(),
    }


def _int_string(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _sorted_image_keys(*mappings: object) -> list[str]:
    keys: set[str] = set()
    for mapping in mappings:
        if isinstance(mapping, Mapping):
            keys.update(str(key) for key in mapping.keys())
    return sorted(keys, key=_image_key_sort_key)


def _image_key_sort_key(item: str) -> tuple[bool, int, str]:
    value = _int_string(item)
    return (value is None, value or 0, item)


def _frame_file(root: Path, folder: Path, image_id: int) -> dict[str, Any]:
    path = folder / f"{image_id:06d}.png"
    return {
        "path": path.as_posix(),
        "relative_path": _relative_to(path, root),
        "relative_name": f"{folder.name}/{path.name}",
        "exists": path.is_file(),
        "summary": _image_summary(path) if path.is_file() else None,
    }


def _mask_files(folder: Path, image_id: int) -> list[str]:
    if not folder.is_dir():
        return []
    return [
        path.name
        for path in sorted(folder.glob(f"{image_id:06d}_*.png"))
        if path.is_file()
    ]


def _mask_file_artifacts(root: Path, folder: Path, image_id: int) -> list[dict[str, Any]]:
    if not folder.is_dir():
        return []
    artifacts = []
    for path in sorted(folder.glob(f"{image_id:06d}_*.png")):
        if not path.is_file():
            continue
        artifacts.append(
            {
                "name": path.name,
                "path": path.as_posix(),
                "relative_path": _relative_to(path, root),
                "summary": _image_summary(path),
            }
        )
    return artifacts


def _frame_map_for_key(frame_map: object, image_key: str) -> object | None:
    if not isinstance(frame_map, Mapping):
        return None
    return frame_map.get(image_key)


def _scene_info_for_folder(
    run_root: Path,
    scene_folder: Path,
) -> dict[str, Any] | None:
    relative_scene_folder = _relative_to(scene_folder.resolve(), run_root)
    for scene in _bop_scene_lookup(run_root).values():
        if scene.get("relative_scene_folder") == relative_scene_folder:
            return dict(scene)
    return None


def _bop_scene_lookup(run_root: Path, *, split: str | None = None) -> dict[int, dict[str, Any]]:
    manifest = _json_if_present(run_root / BOP_DIR / BOP_EXPORT_MANIFEST)
    if not isinstance(manifest, Mapping):
        return {}

    scenes: dict[int, dict[str, Any]] = {}
    for export in manifest.get("exports", []):
        if not isinstance(export, Mapping):
            continue
        export_split = export.get("split")
        if split is not None and export_split != split:
            continue
        scene_id = _int_string(export.get("scene_id"))
        if scene_id is None:
            continue
        scene_folder = export.get("scene_folder")
        scene_info: dict[str, Any] = {
            "scene_id": scene_id,
            "sensor_name": export.get("sensor_name"),
            "split": export_split,
            "scene_folder": scene_folder,
        }
        if isinstance(scene_folder, str):
            try:
                scene_path = Path(scene_folder)
                if not scene_path.is_absolute():
                    scene_path = run_root / scene_path
                scene_info["relative_scene_folder"] = _relative_to(
                    scene_path.resolve(),
                    run_root,
                )
            except ArtifactPathError:
                scene_info["relative_scene_folder"] = None
        scenes[scene_id] = scene_info
    return scenes


def _float_values(text: object, *, expected_count: int, row_number: int, field: str) -> list[float]:
    try:
        values = [float(value) for value in str(text).split()]
    except ValueError as exc:
        raise ValueError(f"BOP result row {row_number} has invalid {field}") from exc
    if len(values) != expected_count:
        raise ValueError(
            f"BOP result row {row_number} {field} must have {expected_count} values"
        )
    return values


def _bop_result_row(
    row: Mapping[str, str],
    *,
    row_number: int,
    scenes: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        scene_id = int(row["scene_id"])
        im_id = int(row["im_id"])
        obj_id = int(row["obj_id"])
        score = float(row["score"])
        time = float(row["time"])
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"BOP result row {row_number} contains invalid scalar values"
        ) from exc

    rotation = _float_values(
        row.get("R", ""),
        expected_count=9,
        row_number=row_number,
        field="rotation",
    )
    translation = _float_values(
        row.get("t", ""),
        expected_count=3,
        row_number=row_number,
        field="translation",
    )
    scene = scenes.get(scene_id)
    return {
        "row_number": row_number,
        "scene_id": scene_id,
        "im_id": im_id,
        "image_key": str(im_id),
        "obj_id": obj_id,
        "score": score,
        "R": rotation,
        "R_matrix": [
            rotation[0:3],
            rotation[3:6],
            rotation[6:9],
        ],
        "t": translation,
        "time": time,
        "scene": dict(scene) if isinstance(scene, Mapping) else None,
    }


def bop_result_detail(
    run_root: str | Path,
    result_path: str | Path,
    *,
    row_limit: int = 500,
) -> dict[str, Any]:
    """Return a safe row-level drill-down for a BOP19 result CSV."""

    if row_limit < 1:
        raise ValueError("row_limit must be at least 1")

    root = _run_root(run_root)
    path = resolve_artifact_path(root, result_path)
    metadata = validate_bop19_result_file(path)
    scenes = _bop_scene_lookup(root, split=metadata.split)

    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != BOP19_RESULT_HEADER:
            raise ValueError(
                f"BOP result CSV header must be {BOP19_RESULT_HEADER}: {path}"
            )
        for row_number, row in enumerate(reader, start=2):
            if len(rows) >= row_limit:
                break
            if not any(row.values()):
                continue
            rows.append(_bop_result_row(row, row_number=row_number, scenes=scenes))

    return {
        "type": "bop_result_detail",
        "result_path": path.as_posix(),
        "relative_path": _relative_to(path, root),
        "metadata": asdict(metadata),
        "row_count": metadata.row_count,
        "row_limit": row_limit,
        "rows": rows,
        "scene_count": len(scenes),
        "scenes": scenes,
    }


def bop_scene_detail(
    run_root: str | Path,
    scene_path: str | Path,
    *,
    frame_limit: int = 200,
) -> dict[str, Any]:
    """Return a safe frame-by-frame drill-down for one BOP scene folder."""

    if frame_limit < 1:
        raise ValueError("frame_limit must be at least 1")

    root = _run_root(run_root)
    scene_folder = resolve_artifact_path(root, scene_path)
    if not scene_folder.is_dir():
        raise FileNotFoundError(f"BOP scene folder not found: {scene_folder}")

    scene_camera = _safe_json(scene_folder / "scene_camera.json")
    scene_gt = _safe_json(scene_folder / "scene_gt.json")
    scene_gt_info = _safe_json(scene_folder / "scene_gt_info.json")
    frame_map = _safe_json(scene_folder / BOP_FRAME_MAP_JSON)
    if not isinstance(scene_camera, Mapping):
        raise ValueError(f"Missing or invalid scene_camera.json in {scene_folder}")

    summary = _bop_scene_summary(scene_folder) or {"type": "bop_scene"}
    image_keys = _sorted_image_keys(scene_camera, scene_gt, scene_gt_info)
    frames = []
    for image_key in image_keys[:frame_limit]:
        image_id = _int_string(image_key)
        if image_id is None:
            continue
        gt_annotations = (
            scene_gt.get(image_key)
            if isinstance(scene_gt, Mapping)
            else []
        )
        gt_info = (
            scene_gt_info.get(image_key)
            if isinstance(scene_gt_info, Mapping)
            else None
        )
        frames.append(
            {
                "image_key": image_key,
                "image_id": image_id,
                "rgb": _frame_file(root, scene_folder / RGB_DIR, image_id),
                "depth": _frame_file(root, scene_folder / DEPTH_DIR, image_id),
                "camera": scene_camera.get(image_key),
                "gt_count": (
                    len(gt_annotations) if isinstance(gt_annotations, list) else 0
                ),
                "gt": gt_annotations if isinstance(gt_annotations, list) else None,
                "gt_info": gt_info,
                "mask_files": _mask_files(scene_folder / "mask", image_id),
                "mask_artifacts": _mask_file_artifacts(
                    root,
                    scene_folder / "mask",
                    image_id,
                ),
                "mask_visib_files": _mask_files(scene_folder / "mask_visib", image_id),
                "mask_visib_artifacts": _mask_file_artifacts(
                    root,
                    scene_folder / "mask_visib",
                    image_id,
                ),
                "frame_map": _frame_map_for_key(frame_map, image_key),
            }
        )

    return {
        "type": "bop_scene_detail",
        "scene_path": scene_folder.as_posix(),
        "relative_path": _relative_to(scene_folder, root),
        "summary": summary,
        "frame_count": len(image_keys),
        "frame_limit": frame_limit,
        "frames": frames,
        "files": {
            "scene_camera": (scene_folder / "scene_camera.json").is_file(),
            "scene_gt": (scene_folder / "scene_gt.json").is_file(),
            "scene_gt_info": (scene_folder / "scene_gt_info.json").is_file(),
            "frame_map": (scene_folder / BOP_FRAME_MAP_JSON).is_file(),
            "rgb_dir": (scene_folder / RGB_DIR).is_dir(),
            "depth_dir": (scene_folder / DEPTH_DIR).is_dir(),
            "mask_dir": (scene_folder / "mask").is_dir(),
            "mask_visib_dir": (scene_folder / "mask_visib").is_dir(),
        },
    }


def _matching_bop_result_rows(
    *,
    run_root: Path,
    result_path: str | Path,
    scene_id: int,
    image_id: int,
    row_limit: int,
) -> dict[str, Any]:
    path = resolve_artifact_path(run_root, result_path)
    metadata = validate_bop19_result_file(path)
    scenes = _bop_scene_lookup(run_root, split=metadata.split)

    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != BOP19_RESULT_HEADER:
            raise ValueError(
                f"BOP result CSV header must be {BOP19_RESULT_HEADER}: {path}"
            )
        for row_number, row in enumerate(reader, start=2):
            if not any(row.values()):
                continue
            parsed = _bop_result_row(row, row_number=row_number, scenes=scenes)
            if parsed["scene_id"] != scene_id or parsed["im_id"] != image_id:
                continue
            rows.append(parsed)
            if len(rows) >= row_limit:
                break

    return {
        "path": path.as_posix(),
        "relative_path": _relative_to(path, run_root),
        "metadata": asdict(metadata),
        "row_count": metadata.row_count,
        "matching_row_count": len(rows),
        "row_limit": row_limit,
        "rows": rows,
    }


def _project_result_origin(
    camera: object,
    row: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(camera, Mapping):
        return None
    cam_k = camera.get("cam_K")
    translation = row.get("t")
    if not isinstance(cam_k, list | tuple) or len(cam_k) < 9:
        return None
    if not isinstance(translation, list | tuple) or len(translation) < 3:
        return None
    try:
        fx = float(cam_k[0])
        cx = float(cam_k[2])
        fy = float(cam_k[4])
        cy = float(cam_k[5])
        x = float(translation[0])
        y = float(translation[1])
        z = float(translation[2])
    except (TypeError, ValueError):
        return None
    if z <= 0:
        return None
    return {
        "u": (fx * x / z) + cx,
        "v": (fy * y / z) + cy,
        "depth": z,
        "source": "bop19_t_object_origin",
    }


def _bop_model_lookup(run_root: Path) -> dict[int, dict[str, Any]]:
    manifest = _json_if_present(run_root / BOP_DIR / BOP_EXPORT_MANIFEST)
    if not isinstance(manifest, Mapping):
        return {}

    models: dict[int, dict[str, Any]] = {}
    for model in manifest.get("object_models", []):
        if not isinstance(model, Mapping):
            continue
        obj_id = _int_string(model.get("obj_id"))
        bop_path = model.get("bop_path")
        if obj_id is None or not isinstance(bop_path, str):
            continue
        try:
            model_path = resolve_artifact_path(run_root, bop_path)
        except ArtifactPathError:
            continue
        models[obj_id] = {
            "obj_id": obj_id,
            "object_name": model.get("object_name"),
            "path": model_path,
            "relative_path": _relative_to(model_path, run_root),
        }
    return models


def _camera_intrinsics(camera: object) -> tuple[float, float, float, float] | None:
    if not isinstance(camera, Mapping):
        return None
    cam_k = camera.get("cam_K")
    if not isinstance(cam_k, list | tuple) or len(cam_k) < 9:
        return None
    try:
        return float(cam_k[0]), float(cam_k[4]), float(cam_k[2]), float(cam_k[5])
    except (TypeError, ValueError):
        return None


def _project_model_bbox(
    camera: object,
    row: Mapping[str, Any],
    model: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    intrinsics = _camera_intrinsics(camera)
    if intrinsics is None or not isinstance(model, Mapping):
        return None
    model_path = model.get("path")
    if not isinstance(model_path, Path) or not model_path.is_file():
        return None
    rotation = row.get("R")
    translation = row.get("t")
    if not isinstance(rotation, list | tuple) or len(rotation) != 9:
        return None
    if not isinstance(translation, list | tuple) or len(translation) != 3:
        return None
    try:
        r_m2c = np.asarray([float(value) for value in rotation], dtype=float).reshape(
            3,
            3,
        )
        t_m2c = np.asarray([float(value) for value in translation], dtype=float)
    except (TypeError, ValueError):
        return None

    try:
        vertices = mesh_vertices(model_path)
    except Exception:
        return None
    if vertices.size == 0:
        return None

    points_cam = vertices @ r_m2c.T + t_m2c
    z = points_cam[:, 2]
    visible = z > 0
    if not bool(np.any(visible)):
        return None

    fx, fy, cx, cy = intrinsics
    points_visible = points_cam[visible]
    u = (fx * points_visible[:, 0] / points_visible[:, 2]) + cx
    v = (fy * points_visible[:, 1] / points_visible[:, 2]) + cy
    x_min = float(np.min(u))
    y_min = float(np.min(v))
    x_max = float(np.max(u))
    y_max = float(np.max(v))
    return {
        "bbox": [
            x_min,
            y_min,
            float(x_max - x_min),
            float(y_max - y_min),
        ],
        "vertex_count": int(len(vertices)),
        "projected_vertex_count": int(np.count_nonzero(visible)),
        "model_relative_path": model.get("relative_path"),
        "object_name": model.get("object_name"),
        "source": "bop19_pose_model_vertices",
    }


def _add_projected_result_origins(
    result: dict[str, Any] | None,
    camera: object,
    *,
    models: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if result is None:
        return None
    rows = result.get("rows")
    if not isinstance(rows, list):
        return result
    projected_rows = []
    projected_count = 0
    projected_model_count = 0
    for row in rows:
        if not isinstance(row, Mapping):
            projected_rows.append(row)
            continue
        row_copy = dict(row)
        projection = _project_result_origin(camera, row_copy)
        row_copy["projected_origin"] = projection
        if projection is not None:
            projected_count += 1
        model_projection = None
        if models is not None:
            obj_id = _int_string(row_copy.get("obj_id"))
            if obj_id is not None:
                model_projection = _project_model_bbox(
                    camera,
                    row_copy,
                    models.get(obj_id),
                )
        row_copy["projected_model_bbox"] = model_projection
        if model_projection is not None:
            projected_model_count += 1
        projected_rows.append(row_copy)
    result_copy = dict(result)
    result_copy["rows"] = projected_rows
    result_copy["projected_origin_count"] = projected_count
    result_copy["projected_model_bbox_count"] = projected_model_count
    return result_copy


def bop_frame_detail(
    run_root: str | Path,
    scene_path: str | Path,
    *,
    image_id: int,
    result_path: str | Path | None = None,
    row_limit: int = 100,
) -> dict[str, Any]:
    """Return one BOP frame bundle for RGB/depth/mask/GT/result inspection."""

    if image_id < 0:
        raise ValueError("image_id must be non-negative")
    if row_limit < 1:
        raise ValueError("row_limit must be at least 1")

    root = _run_root(run_root)
    scene_folder = resolve_artifact_path(root, scene_path)
    if not scene_folder.is_dir():
        raise FileNotFoundError(f"BOP scene folder not found: {scene_folder}")

    scene_camera = _safe_json(scene_folder / "scene_camera.json")
    scene_gt = _safe_json(scene_folder / "scene_gt.json")
    scene_gt_info = _safe_json(scene_folder / "scene_gt_info.json")
    frame_map = _safe_json(scene_folder / BOP_FRAME_MAP_JSON)
    if not isinstance(scene_camera, Mapping):
        raise ValueError(f"Missing or invalid scene_camera.json in {scene_folder}")

    image_key = str(image_id)
    gt_annotations = (
        scene_gt.get(image_key)
        if isinstance(scene_gt, Mapping)
        else []
    )
    gt_info = (
        scene_gt_info.get(image_key)
        if isinstance(scene_gt_info, Mapping)
        else None
    )
    scene = _scene_info_for_folder(root, scene_folder)
    camera = scene_camera.get(image_key)
    result = None
    if result_path is not None:
        if not scene or _int_string(scene.get("scene_id")) is None:
            raise ValueError(
                "BOP export manifest must map the scene folder before result "
                "rows can be joined to a frame."
            )
        result = _matching_bop_result_rows(
            run_root=root,
            result_path=result_path,
            scene_id=int(scene["scene_id"]),
            image_id=image_id,
            row_limit=row_limit,
        )
        result = _add_projected_result_origins(
            result,
            camera,
            models=_bop_model_lookup(root),
        )

    return {
        "type": "bop_frame_detail",
        "scene_path": scene_folder.as_posix(),
        "relative_path": _relative_to(scene_folder, root),
        "scene": scene,
        "image_id": image_id,
        "image_key": image_key,
        "rgb": _frame_file(root, scene_folder / RGB_DIR, image_id),
        "depth": _frame_file(root, scene_folder / DEPTH_DIR, image_id),
        "camera": camera,
        "gt_count": len(gt_annotations) if isinstance(gt_annotations, list) else 0,
        "gt": gt_annotations if isinstance(gt_annotations, list) else None,
        "gt_info": gt_info,
        "mask_artifacts": _mask_file_artifacts(
            root,
            scene_folder / "mask",
            image_id,
        ),
        "mask_visib_artifacts": _mask_file_artifacts(
            root,
            scene_folder / "mask_visib",
            image_id,
        ),
        "frame_map": _frame_map_for_key(frame_map, image_key),
        "result": result,
    }


def _mask_paths(folder: Path, image_id: int) -> list[Path]:
    if not folder.is_dir():
        return []
    return [
        path
        for path in sorted(folder.glob(f"{image_id:06d}_*.png"))
        if path.is_file()
    ]


def _mask_to_bool(mask: np.ndarray, *, width: int, height: int) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def _apply_mask_overlay(
    image: np.ndarray,
    mask_path: Path,
    *,
    color: tuple[int, int, int],
    alpha: float,
) -> bool:
    mask = cv2.imread(mask_path.as_posix(), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return False
    height, width = image.shape[:2]
    mask_pixels = _mask_to_bool(mask, width=width, height=height)
    if not bool(mask_pixels.any()):
        return False
    overlay = image.copy()
    overlay[mask_pixels] = color
    blended = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)
    image[mask_pixels] = blended[mask_pixels]
    return True


def _bbox_values(value: object) -> tuple[int, int, int, int] | None:
    if not isinstance(value, list | tuple) or len(value) < 4:
        return None
    try:
        x, y, width, height = (int(round(float(item))) for item in value[:4])
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return x, y, width, height


def _draw_bbox(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    color: tuple[int, int, int],
    label: str,
) -> None:
    height, width = image.shape[:2]
    x, y, box_width, box_height = bbox
    x1 = max(0, min(width - 1, x))
    y1 = max(0, min(height - 1, y))
    x2 = max(0, min(width - 1, x + box_width - 1))
    y2 = max(0, min(height - 1, y + box_height - 1))
    if x2 < x1 or y2 < y1:
        return
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=1)
    if label:
        text_y = max(0, y1 - 4)
        cv2.putText(
            image,
            label,
            (x1, text_y if text_y > 0 else min(height - 1, y1 + 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )


def _draw_projection_marker(
    image: np.ndarray,
    projection: Mapping[str, Any],
    *,
    color: tuple[int, int, int],
    label: str,
) -> None:
    height, width = image.shape[:2]
    try:
        u = int(round(float(projection["u"])))
        v = int(round(float(projection["v"])))
    except (KeyError, TypeError, ValueError):
        return
    if u < 0 or v < 0 or u >= width or v >= height:
        return
    cv2.drawMarker(
        image,
        (u, v),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=9,
        thickness=1,
        line_type=cv2.LINE_AA,
    )
    if label:
        cv2.putText(
            image,
            label,
            (min(width - 1, u + 3), max(0, v - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )


def _draw_label_strip(image: np.ndarray, lines: list[str]) -> None:
    if not lines:
        return
    height, width = image.shape[:2]
    y = 12
    for line in lines[:6]:
        text = line[:80]
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            1,
        )
        box_bottom = min(height - 1, y + baseline)
        cv2.rectangle(
            image,
            (0, max(0, y - text_height - 2)),
            (min(width - 1, text_width + 4), box_bottom),
            (0, 0, 0),
            thickness=-1,
        )
        cv2.putText(
            image,
            text,
            (2, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += text_height + baseline + 6


def render_bop_frame_overlay_png(
    run_root: str | Path,
    scene_path: str | Path,
    *,
    image_id: int,
    result_path: str | Path | None = None,
    row_limit: int = 20,
    include_masks: bool = True,
    include_gt: bool = True,
    include_results: bool = True,
) -> bytes:
    """Render a BOP frame RGB overlay with masks, GT boxes, and result labels."""

    detail = bop_frame_detail(
        run_root,
        scene_path,
        image_id=image_id,
        result_path=result_path if include_results else None,
        row_limit=row_limit,
    )
    root = _run_root(run_root)
    scene_folder = resolve_artifact_path(root, scene_path)
    rgb_path = resolve_artifact_path(root, detail["rgb"]["relative_path"])
    if not rgb_path.is_file():
        raise FileNotFoundError(f"BOP RGB frame not found: {rgb_path}")

    image = cv2.imread(rgb_path.as_posix(), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"BOP RGB frame is not a readable image: {rgb_path}")
    if image.dtype != np.uint8:
        image = cv2.convertScaleAbs(image)

    if include_masks:
        mask_colors = (
            (0, 170, 255),
            (255, 120, 0),
            (180, 0, 255),
            (80, 220, 80),
        )
        for index, mask_path in enumerate(_mask_paths(scene_folder / "mask", image_id)):
            _apply_mask_overlay(
                image,
                mask_path,
                color=mask_colors[index % len(mask_colors)],
                alpha=0.28,
            )
        for index, mask_path in enumerate(
            _mask_paths(scene_folder / "mask_visib", image_id)
        ):
            _apply_mask_overlay(
                image,
                mask_path,
                color=(0, 255, 0),
                alpha=0.38 if index == 0 else 0.25,
            )

    if include_gt:
        gt_rows = detail.get("gt") if isinstance(detail.get("gt"), list) else []
        gt_info_rows = (
            detail.get("gt_info") if isinstance(detail.get("gt_info"), list) else []
        )
        for index, row in enumerate(gt_rows):
            if not isinstance(row, Mapping):
                continue
            info = gt_info_rows[index] if index < len(gt_info_rows) else {}
            if not isinstance(info, Mapping):
                info = {}
            bbox = _bbox_values(info.get("bbox_obj") or row.get("bbox_obj"))
            if bbox is None:
                continue
            label = f"gt obj {row.get('obj_id', '?')}"
            _draw_bbox(image, bbox, color=(0, 255, 255), label=label)

    lines = [
        f"{detail['relative_path']} image {detail['image_id']}",
    ]
    result = detail.get("result")
    if include_results and isinstance(result, Mapping):
        rows = result.get("rows") if isinstance(result.get("rows"), list) else []
        for row in rows[:5]:
            if not isinstance(row, Mapping):
                continue
            score = row.get("score")
            score_text = f"{float(score):.3f}" if isinstance(score, int | float) else score
            projection = row.get("projected_origin")
            if isinstance(projection, Mapping):
                _draw_projection_marker(
                    image,
                    projection,
                    color=(255, 0, 255),
                    label=f"est {row.get('obj_id', '?')}",
                )
            model_projection = row.get("projected_model_bbox")
            if isinstance(model_projection, Mapping):
                bbox = _bbox_values(model_projection.get("bbox"))
                if bbox is not None:
                    _draw_bbox(
                        image,
                        bbox,
                        color=(255, 0, 255),
                        label=f"est box {row.get('obj_id', '?')}",
                    )
            lines.append(f"est obj {row.get('obj_id', '?')} score {score_text}")
    _draw_label_strip(image, lines)

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to encode BOP frame overlay PNG")
    return encoded.tobytes()


def _directory_summary(path: Path) -> dict[str, Any] | None:
    bop_scene = _bop_scene_summary(path)
    if bop_scene is not None:
        return bop_scene
    return None


def _summary_for_path(path: Path, preview_type: str, kind: str) -> dict[str, Any] | None:
    if kind == "directory":
        return _directory_summary(path)
    if kind != "file":
        return None
    if preview_type == "json":
        return _json_summary(path)
    if path.suffix.lower() == ".csv":
        return _csv_summary(path)
    if preview_type == "image":
        return _image_summary(path)
    return None


def _record_for_path(
    *,
    run_root: Path,
    key: str,
    source: str,
    artifact_path: str | Path,
) -> ArtifactRecord:
    raw_path = Path(artifact_path)
    path = raw_path if raw_path.is_absolute() else run_root / raw_path
    try:
        resolved = path.resolve()
        relative_path = _relative_to(resolved, run_root)
    except ArtifactPathError:
        return ArtifactRecord(
            key=key,
            source=source,
            path=path.as_posix(),
            relative_path=None,
            kind="outside_run_root",
            exists=False,
            preview_type="outside_run_root",
        )

    if not resolved.exists():
        return ArtifactRecord(
            key=key,
            source=source,
            path=resolved.as_posix(),
            relative_path=relative_path,
            kind="missing",
            exists=False,
            preview_type="missing",
        )

    stat = resolved.stat()
    if resolved.is_dir():
        child_count = sum(1 for _ in resolved.iterdir())
        return ArtifactRecord(
            key=key,
            source=source,
            path=resolved.as_posix(),
            relative_path=relative_path,
            kind="directory",
            exists=True,
            preview_type="directory",
            modified_at=_utc_timestamp(stat.st_mtime),
            child_count=child_count,
            summary=_summary_for_path(resolved, "directory", "directory"),
        )

    kind = "file" if resolved.is_file() else "other"
    preview_type = _preview_type(resolved, kind)
    return ArtifactRecord(
        key=key,
        source=source,
        path=resolved.as_posix(),
        relative_path=relative_path,
        kind=kind,
        exists=True,
        preview_type=preview_type,
        size_bytes=stat.st_size,
        modified_at=_utc_timestamp(stat.st_mtime),
        summary=_summary_for_path(resolved, preview_type, kind),
    )


def _add_record(
    records: list[ArtifactRecord],
    *,
    run_root: Path,
    key: str,
    source: str,
    artifact_path: str | Path | None,
) -> None:
    if artifact_path in (None, ""):
        return
    records.append(
        _record_for_path(
            run_root=run_root,
            key=key,
            source=source,
            artifact_path=artifact_path,
        )
    )


def _json_if_present(path: Path) -> object | None:
    if not path.is_file():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _manifest_artifacts(run_root: Path) -> Iterable[tuple[str, str, str]]:
    try:
        manifest = load_run_manifest(run_root)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return []

    entries: list[tuple[str, str, str]] = []
    for key, value in dict(manifest.get("artifacts", {})).items():
        if isinstance(value, str):
            entries.append((key, "manifest.artifacts", value))

    for stage in manifest.get("stages", []):
        if not isinstance(stage, Mapping):
            continue
        stage_name = str(stage.get("name", "unknown_stage"))
        artifacts = stage.get("artifacts", {})
        if not isinstance(artifacts, Mapping):
            continue
        for key, value in artifacts.items():
            if isinstance(value, str):
                entries.append((str(key), f"stage:{stage_name}", value))
    return entries


def _known_run_artifacts(run_root: Path) -> Iterable[tuple[str, str, str]]:
    known = [
        DATASET_MANIFEST,
        REWRITE_GATE_REPORT,
        REWRITE_STATUS_REPORT,
        RUN_CONFIG,
        RUN_PREFLIGHT_REPORT,
        HARDWARE_STATUS_REPORT,
        CAPTURE_PLAN,
        CAPTURE_PLAN_PREFLIGHT_REPORT,
        CAPTURE_EXECUTION_PLAN,
        CAPTURE_EXECUTION_STATUS,
        CAPTURE_EXECUTION_REPORT,
        CAPTURE_REHEARSAL_REPORT,
        REALSENSE_CAPTURE_SMOKE_REPORT,
        CAPTURE_EXECUTION_LOGS_DIR,
        CALIBRATION_PREFLIGHT_REPORT,
        CALIBRATION_OBSERVATIONS,
        CALIBRATION_CANDIDATES,
        CALIBRATION_PROFILES_FROM_OBSERVATIONS,
        CALIBRATION_SOLVER_REPORT,
        CALIBRATION_PROFILES_SOLVED,
        CALIBRATION_VALIDATION_REPORT,
        SYNC_QUALITY_REPORT,
        ARUCO_COVERAGE_REPORT,
        FOUNDATIONPOSE_PLAN,
        MEGAPOSE_PLAN,
        SAM6D_PLAN,
        PIPELINE_SEQUENCE_PLAN,
        BOP_EVALUATION_PLAN,
        BOP_EVALUATION_REPORT,
        BOP_RESULT_EXPORT_MANIFEST,
        f"{RESULTS_DIR}/{METRICS_DIR}/{METRIC_REPORT_JSON}",
        f"{RESULTS_DIR}/{METRICS_DIR}/{METRIC_REPORT_CSV}",
        f"{RESULTS_DIR}/{METRICS_DIR}/{METRIC_REPORT_XLSX}",
        f"{BOP_DIR}/{BOP_EXPORT_MANIFEST}",
        f"{BOP_DIR}/{MODELS_DIR}/models_info.json",
        f"{BOP_DIR}/{BOP_TARGETS_BOP19}",
        f"{BOP_DIR}/{BOP_MULTIVIEW_TARGETS}",
        f"{BOP_DIR}/{BOP_COCO_ANNOTATIONS}",
        RESULTS_DIR,
        EVALUATION_DIR,
    ]
    return [
        (Path(path).name, "known", path)
        for path in known
        if (run_root / path).exists()
    ]


def _bop_export_artifacts(run_root: Path) -> Iterable[tuple[str, str, str]]:
    manifest_path = run_root / BOP_DIR / BOP_EXPORT_MANIFEST
    manifest = _json_if_present(manifest_path)
    if not isinstance(manifest, Mapping):
        return []

    entries: list[tuple[str, str, str]] = []
    targets_path = manifest.get("targets_path")
    if isinstance(targets_path, str):
        entries.append(("targets_path", "bop_export", targets_path))
    multiview_targets_path = manifest.get("multiview_targets_path")
    if isinstance(multiview_targets_path, str):
        entries.append(
            (
                "multiview_targets_path",
                "bop_export",
                multiview_targets_path,
            )
        )
    coco_annotations_path = manifest.get("coco_annotations_path")
    if isinstance(coco_annotations_path, str):
        entries.append(
            (
                "coco_annotations_path",
                "bop_export",
                coco_annotations_path,
            )
        )

    for export in manifest.get("exports", []):
        if not isinstance(export, Mapping):
            continue
        sensor_name = str(export.get("sensor_name", "sensor"))
        scene_folder = export.get("scene_folder")
        if isinstance(scene_folder, str):
            entries.append((f"{sensor_name}:scene_folder", "bop_export", scene_folder))
        artifacts = export.get("artifacts", {})
        if isinstance(artifacts, Mapping):
            for key, value in artifacts.items():
                if isinstance(value, str):
                    entries.append(
                        (f"{sensor_name}:{key}", "bop_export.scene", value)
                    )

    for model in manifest.get("object_models", []):
        if not isinstance(model, Mapping):
            continue
        object_name = str(model.get("object_name", "object"))
        bop_path = model.get("bop_path")
        if isinstance(bop_path, str):
            entries.append((f"{object_name}:bop_model", "bop_export.models", bop_path))
    return entries


def _bop_result_artifacts(run_root: Path) -> Iterable[tuple[str, str, str]]:
    manifest = _json_if_present(run_root / BOP_RESULT_EXPORT_MANIFEST)
    if not isinstance(manifest, Mapping):
        return []

    entries: list[tuple[str, str, str]] = []
    output_folder = manifest.get("output_folder")
    if isinstance(output_folder, str):
        entries.append(("output_folder", "bop_result_export", output_folder))
    for result in manifest.get("results", []):
        if not isinstance(result, Mapping):
            continue
        filename = str(result.get("filename", "result"))
        path = result.get("path")
        if isinstance(path, str):
            entries.append((filename, "bop_result_export.result", path))
    return entries


def _bop_evaluation_artifacts(run_root: Path) -> Iterable[tuple[str, str, str]]:
    plan = _json_if_present(run_root / BOP_EVALUATION_PLAN)
    entries: list[tuple[str, str, str]] = []
    if isinstance(plan, Mapping):
        for key in ("eval_path", "bop_root", "dataset_folder"):
            value = plan.get(key)
            if isinstance(value, str):
                entries.append((key, "bop_evaluation_plan", value))
        result = plan.get("result")
        if isinstance(result, Mapping):
            path = result.get("path")
            if isinstance(path, str):
                entries.append(("result_file", "bop_evaluation_plan", path))

    report = _json_if_present(run_root / BOP_EVALUATION_REPORT)
    if isinstance(report, Mapping):
        eval_path = report.get("eval_path")
        if isinstance(eval_path, str):
            entries.append(("eval_path", "bop_evaluation_report", eval_path))
        for output in report.get("output_artifacts", []):
            if not isinstance(output, Mapping):
                continue
            path = output.get("path")
            relative_path = output.get("relative_path")
            if isinstance(path, str):
                key = (
                    f"output:{relative_path}"
                    if isinstance(relative_path, str)
                    else "output"
                )
                entries.append((key, "bop_evaluation_report.output", path))
    return entries


def _metric_artifacts(run_root: Path) -> Iterable[tuple[str, str, str]]:
    metric_names = (*ACCURACY_ARTIFACTS, ALL_RESULTS_JSON)
    entries = []
    for name in metric_names:
        for path in sorted(run_root.rglob(name)):
            if not path.is_file():
                continue
            source = (
                "metrics.combined"
                if path.name == ALL_RESULTS_JSON
                else "metrics.legacy_pose"
            )
            entries.append((path.name, source, _relative_to(path.resolve(), run_root)))
    return entries


def _metric_records(records: Iterable[ArtifactRecord]) -> list[ArtifactRecord]:
    return [
        record
        for record in records
        if isinstance(record.summary, Mapping)
        and record.summary.get("type")
        in {"pose_accuracy_metrics", "combined_pose_accuracy_metrics"}
    ]


def _direct_metric_rows(records: Iterable[ArtifactRecord]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        summary = record.summary or {}
        if summary.get("type") != "pose_accuracy_metrics":
            continue
        methods = summary.get("methods", [])
        if not isinstance(methods, list):
            continue
        for method in methods:
            if not isinstance(method, Mapping):
                continue
            all_motions = method.get("all_motions")
            row: dict[str, Any] = {
                "artifact_key": record.key,
                "source": record.source,
                "relative_path": record.relative_path,
                "method": method.get("name"),
                "motion_count": method.get("motion_count"),
                "motions": method.get("motions", []),
                "sample_count": method.get("sample_count"),
                "all_motions": all_motions if isinstance(all_motions, Mapping) else {},
            }
            rows.append(row)
    return rows


def _combined_metric_rows(
    records: Iterable[ArtifactRecord],
    *,
    group_limit: int,
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        summary = record.summary or {}
        if summary.get("type") != "combined_pose_accuracy_metrics":
            continue
        groups = summary.get("groups", [])
        if not isinstance(groups, list):
            continue
        for group in groups[:group_limit]:
            if not isinstance(group, Mapping):
                continue
            rows.append(
                {
                    "artifact_key": record.key,
                    "source": record.source,
                    "relative_path": record.relative_path,
                    "context": group.get("context", ""),
                    "method_count": group.get("method_count"),
                    "methods": group.get("methods", []),
                    "best_by_AP_p": group.get("best_by_AP_p"),
                }
            )
    return rows


def _dashboard_best_candidate(row: Mapping[str, Any]) -> dict[str, Any] | None:
    best = row.get("best_by_AP_p")
    if isinstance(best, Mapping):
        ap_p = _metric_number(best.get("AP_p"))
        method = best.get("method")
    else:
        all_motions = row.get("all_motions")
        if not isinstance(all_motions, Mapping):
            return None
        ap_p = _metric_number(all_motions.get("AP_p"))
        method = row.get("method")
    if ap_p is None or not isinstance(method, str):
        return None
    candidate = {
        "method": method,
        "AP_p": ap_p,
        "relative_path": row.get("relative_path"),
    }
    if isinstance(row.get("context"), str):
        candidate["context"] = row["context"]
    return candidate


def _dashboard_best_by_ap(
    direct_rows: Iterable[Mapping[str, Any]],
    combined_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    candidates = [
        candidate
        for row in (*direct_rows, *combined_rows)
        if (candidate := _dashboard_best_candidate(row)) is not None
    ]
    return min(candidates, key=lambda item: item["AP_p"]) if candidates else None


def _bop_score_rows(records: Iterable[ArtifactRecord]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        summary = record.summary or {}
        if summary.get("type") != "bop_evaluation_report":
            continue
        if summary.get("ready_for_metrics") is not True:
            continue
        metrics = summary.get("score_metrics")
        if not isinstance(metrics, Mapping):
            continue
        rows.append(
            {
                "artifact_key": record.key,
                "source": record.source,
                "relative_path": record.relative_path,
                "result_filename": summary.get("result_filename"),
                "status": summary.get("status"),
                "score_file_count": summary.get("score_file_count", 0),
                "score_metric_count": summary.get("score_metric_count", 0),
                "metrics": dict(metrics),
            }
        )
    return rows


def _dashboard_best_bop19_average_recall(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    candidates = []
    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        recall = _metric_number(metrics.get("bop19_average_recall"))
        if recall is None:
            continue
        candidates.append(
            {
                "result_filename": row.get("result_filename"),
                "bop19_average_recall": recall,
                "relative_path": row.get("relative_path"),
            }
        )
    return (
        max(candidates, key=lambda item: item["bop19_average_recall"])
        if candidates
        else None
    )


def metric_dashboard_summary(
    run_root: str | Path,
    *,
    group_limit: int = 200,
) -> dict[str, Any]:
    """Return dashboard-ready legacy metric summaries for one run."""

    if group_limit < 1:
        raise ValueError("group_limit must be at least 1")

    root = _run_root(run_root)
    if not root.exists():
        raise FileNotFoundError(f"Run root not found: {root}")

    records = collect_run_artifacts(root)
    metric_records = _metric_records(records)
    direct_rows = _direct_metric_rows(metric_records)
    combined_rows = _combined_metric_rows(metric_records, group_limit=group_limit)
    bop_score_rows = _bop_score_rows(records)
    methods = sorted(
        {
            method
            for row in (*direct_rows, *combined_rows)
            for method in (
                row.get("methods", [])
                if isinstance(row.get("methods"), list)
                else [row.get("method")]
            )
            if isinstance(method, str)
        }
    )

    return {
        "type": "metric_dashboard",
        "run_root": root.as_posix(),
        "metric_artifact_count": len(metric_records),
        "direct_method_count": len(direct_rows),
        "combined_group_count": len(combined_rows),
        "method_count": len(methods),
        "methods": methods,
        "best_by_AP_p": _dashboard_best_by_ap(direct_rows, combined_rows),
        "bop_score_count": len(bop_score_rows),
        "best_bop19_average_recall": _dashboard_best_bop19_average_recall(
            bop_score_rows
        ),
        "bop_scores": bop_score_rows,
        "direct_methods": direct_rows,
        "combined_groups": combined_rows,
        "artifacts": [record.to_dict() for record in metric_records],
        "group_limit": group_limit,
    }


def collect_run_artifacts(run_root: str | Path) -> list[ArtifactRecord]:
    """Collect known and manifest-recorded artifacts for a run root."""

    root = _run_root(run_root)
    if not root.exists():
        raise FileNotFoundError(f"Run root not found: {root}")

    records: list[ArtifactRecord] = []
    for key, source, path in _known_run_artifacts(root):
        _add_record(records, run_root=root, key=key, source=source, artifact_path=path)
    for key, source, path in _manifest_artifacts(root):
        _add_record(records, run_root=root, key=key, source=source, artifact_path=path)
    for key, source, path in _bop_export_artifacts(root):
        _add_record(records, run_root=root, key=key, source=source, artifact_path=path)
    for key, source, path in _bop_result_artifacts(root):
        _add_record(records, run_root=root, key=key, source=source, artifact_path=path)
    for key, source, path in _bop_evaluation_artifacts(root):
        _add_record(records, run_root=root, key=key, source=source, artifact_path=path)
    for key, source, path in _metric_artifacts(root):
        _add_record(records, run_root=root, key=key, source=source, artifact_path=path)

    return sorted(
        records,
        key=lambda record: (
            record.source,
            record.key,
            record.relative_path or record.path,
        ),
    )


def _directory_listing(path: Path, *, limit: int) -> list[dict[str, Any]]:
    children = []
    for child in sorted(path.iterdir(), key=lambda item: item.name)[:limit]:
        stat = child.stat()
        children.append(
            {
                "name": child.name,
                "kind": "directory" if child.is_dir() else "file",
                "size_bytes": None if child.is_dir() else stat.st_size,
                "modified_at": _utc_timestamp(stat.st_mtime),
            }
        )
    return children


def _image_preview(path: Path, *, max_side: int = 160) -> dict[str, Any]:
    image = cv2.imread(path.as_posix(), cv2.IMREAD_UNCHANGED)
    if image is None:
        return {"type": "image", "readable": False}

    if image.ndim == 2:
        height, width = image.shape
    else:
        height, width = image.shape[:2]
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    scale = min(1.0, float(max_side) / max(width, height))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".png", image)
    thumbnail = (
        base64.b64encode(encoded.tobytes()).decode("ascii") if ok else None
    )
    return {
        "type": "image",
        "readable": True,
        "width": int(width),
        "height": int(height),
        "thumbnail_png_base64": thumbnail,
        "thumbnail_max_side": max_side,
    }


def preview_artifact(
    run_root: str | Path,
    artifact_path: str | Path,
    *,
    max_bytes: int = 16384,
    directory_limit: int = 100,
) -> dict[str, Any]:
    """Return a safe preview for an artifact under the run root."""

    root = _run_root(run_root)
    path = resolve_artifact_path(root, artifact_path)
    record = _record_for_path(
        run_root=root,
        key=Path(artifact_path).name,
        source="preview",
        artifact_path=path,
    )
    payload: dict[str, Any] = {"artifact": record.to_dict()}

    if not path.exists():
        payload["preview"] = None
        return payload

    if path.is_dir():
        payload["preview"] = {
            "type": "directory",
            "children": _directory_listing(path, limit=directory_limit),
            "limit": directory_limit,
        }
        return payload

    if not path.is_file():
        payload["preview"] = {"type": "other"}
        return payload

    preview_type = _preview_type(path, "file")
    if preview_type == "image":
        payload["preview"] = _image_preview(path)
        return payload

    if preview_type == "json" and path.stat().st_size <= max_bytes:
        payload["preview"] = {
            "type": "json",
            "value": _json_if_present(path),
        }
        return payload

    if preview_type in {"json", "text"}:
        with open(path, "rb") as f:
            raw = f.read(max_bytes + 1)
        truncated = len(raw) > max_bytes
        text = raw[:max_bytes].decode("utf-8", errors="replace")
        payload["preview"] = {
            "type": "text",
            "text": text,
            "truncated": truncated,
            "max_bytes": max_bytes,
        }
        return payload

    payload["preview"] = {"type": "binary"}
    return payload
