"""Artifact-driven next-step recommendations for transition UI runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from posetestbot.io.artifacts import (
    ACCURACY_ARUCO_HRC_HUB,
    ACCURACY_HRC_HUB,
    ALL_RESULTS_JSON,
    ARUCO_COVERAGE_REPORT,
    ARUCO_POSE_ESTIMATION,
    BOP_DIR,
    BOP_EVALUATION_REPORT,
    BOP_EXPORT_MANIFEST,
    BOP_RESULT_EXPORT_MANIFEST,
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
    FOUNDATIONPOSE_PLAN,
    METRIC_REPORT_CSV,
    METRIC_REPORT_JSON,
    METRIC_REPORT_XLSX,
    METRICS_DIR,
    MODELS_DIR,
    PIPELINE_SEQUENCE_PLAN,
    PROCESSED_DIR,
    RAW_ROBOT_EE_POSES,
    REWRITE_GATE_REPORT,
    REWRITE_STATUS_REPORT,
    RESULTS_DIR,
    RUN_CONFIG,
    RUN_PREFLIGHT_REPORT,
    SYNC_QUALITY_REPORT,
    SYNCHRONIZED_DIR,
)
from posetestbot.pipeline.preflight import run_preflight_queue_summary
from posetestbot.pipeline.rewrite_gate import (
    CALIBRATION_VALIDATION_GATE_ID,
    FOUNDATIONPOSE_RUNTIME_GATE_ID,
    FULL_CAPTURE_GATE_ID,
    build_calibration_validation_gate_report,
    build_foundationpose_runtime_gate_report,
    build_full_capture_gate_report,
    build_rewrite_status_report,
)
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


def _bop_evaluation_critical_check_counts(
    checks: object,
) -> tuple[int, int]:
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


def _capture_execution_report_summary(root: Path) -> dict[str, Any]:
    path = root / CAPTURE_EXECUTION_REPORT
    if not path.is_file():
        return {
            "status": None,
            "ready_for_downstream": False,
            "blocker": "missing_capture_execution_report",
        }
    report = _json_if_present(root / CAPTURE_EXECUTION_REPORT)
    if not isinstance(report, Mapping):
        return {
            "status": None,
            "ready_for_downstream": False,
            "blocker": "invalid_capture_execution_report",
        }
    status = report.get("status")
    if not isinstance(status, str):
        return {
            "status": None,
            "ready_for_downstream": False,
            "blocker": "invalid_capture_execution_report",
        }
    if status != "succeeded":
        return {
            "status": status,
            "ready_for_downstream": False,
            "blocker": "failed_capture_execution_report",
        }
    return {"status": status, "ready_for_downstream": True, "blocker": None}


def _run_config_summary(root: Path) -> dict[str, Any]:
    path = root / RUN_CONFIG
    if not path.is_file():
        return {
            "ready_for_pipeline": False,
            "blocker": "missing_run_config",
            "error": None,
        }
    value = _json_if_present(path)
    if not isinstance(value, Mapping):
        return {
            "ready_for_pipeline": False,
            "blocker": "invalid_run_config",
            "error": f"{RUN_CONFIG} is not a JSON object.",
        }
    try:
        validate_run_config(value)
    except Exception as exc:
        return {
            "ready_for_pipeline": False,
            "blocker": "invalid_run_config",
            "error": str(exc),
        }
    return {
        "ready_for_pipeline": True,
        "blocker": None,
        "error": None,
    }


def _capture_plan_summary(root: Path) -> dict[str, Any]:
    path = root / CAPTURE_PLAN
    if not path.is_file():
        return {
            "ready_for_preflight": False,
            "blocker": "missing_capture_plan",
            "command_count": 0,
        }
    plan = _json_if_present(path)
    if not isinstance(plan, Mapping):
        return {
            "ready_for_preflight": False,
            "blocker": "invalid_capture_plan",
            "command_count": 0,
        }
    if plan.get("schema_version") != "capture_plan.v1":
        return {
            "ready_for_preflight": False,
            "blocker": "invalid_capture_plan",
            "command_count": 0,
        }
    commands = plan.get("commands")
    if not isinstance(commands, list):
        return {
            "ready_for_preflight": False,
            "blocker": "invalid_capture_plan",
            "command_count": 0,
        }
    command_count = len(commands)
    if command_count == 0:
        return {
            "ready_for_preflight": False,
            "blocker": "empty_capture_plan",
            "command_count": command_count,
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
            "command_count": command_count,
        }
    return {
        "ready_for_preflight": True,
        "blocker": None,
        "command_count": command_count,
    }


def _pipeline_sequence_plan_summary(root: Path) -> dict[str, Any]:
    path = root / PIPELINE_SEQUENCE_PLAN
    if not path.is_file():
        return {
            "ready_for_queue": False,
            "blocker": "missing_pipeline_sequence_plan",
            "step_count": 0,
        }
    plan = _json_if_present(path)
    if not isinstance(plan, Mapping):
        return {
            "ready_for_queue": False,
            "blocker": "invalid_pipeline_sequence_plan",
            "step_count": 0,
        }
    if plan.get("schema_version") != "pipeline_sequence_plan.v1":
        return {
            "ready_for_queue": False,
            "blocker": "invalid_pipeline_sequence_plan",
            "step_count": 0,
        }
    steps = plan.get("steps")
    if not isinstance(steps, list):
        return {
            "ready_for_queue": False,
            "blocker": "invalid_pipeline_sequence_plan",
            "step_count": 0,
        }
    step_count = len(steps)
    if step_count == 0:
        return {
            "ready_for_queue": False,
            "blocker": "empty_pipeline_sequence_plan",
            "step_count": step_count,
        }
    for step in steps:
        if not isinstance(step, Mapping):
            return {
                "ready_for_queue": False,
                "blocker": "invalid_pipeline_sequence_plan",
                "step_count": step_count,
            }
        if not isinstance(step.get("id"), str):
            return {
                "ready_for_queue": False,
                "blocker": "invalid_pipeline_sequence_plan",
                "step_count": step_count,
            }
        if not isinstance(step.get("stage_id"), str):
            return {
                "ready_for_queue": False,
                "blocker": "invalid_pipeline_sequence_plan",
                "step_count": step_count,
            }
        command = step.get("command")
        if not isinstance(command, list) or not all(
            isinstance(item, str) for item in command
        ):
            return {
                "ready_for_queue": False,
                "blocker": "invalid_pipeline_sequence_plan",
                "step_count": step_count,
            }
    return {
        "ready_for_queue": True,
        "blocker": None,
        "step_count": step_count,
    }


def _capture_plan_preflight_summary(root: Path) -> dict[str, Any]:
    path = root / CAPTURE_PLAN_PREFLIGHT_REPORT
    if not path.is_file():
        return {
            "overall_status": None,
            "ready_for_execution_plan": False,
            "blocker": "missing_capture_plan_preflight",
        }
    report = _json_if_present(path)
    if not isinstance(report, Mapping):
        return {
            "overall_status": None,
            "ready_for_execution_plan": False,
            "blocker": "invalid_capture_plan_preflight",
        }
    overall_status = report.get("overall_status")
    if not isinstance(overall_status, str):
        return {
            "overall_status": None,
            "ready_for_execution_plan": False,
            "blocker": "invalid_capture_plan_preflight",
        }
    if overall_status == "error":
        return {
            "overall_status": overall_status,
            "ready_for_execution_plan": False,
            "blocker": "failed_capture_plan_preflight",
        }
    return {
        "overall_status": overall_status,
        "ready_for_execution_plan": True,
        "blocker": None,
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


def _capture_execution_plan_summary(root: Path) -> dict[str, Any]:
    path = root / CAPTURE_EXECUTION_PLAN
    if not path.is_file():
        return {
            "status": None,
            "ready_to_execute": False,
            "blocker": "missing_capture_execution_plan",
            "blocked_checks": [],
        }
    plan = _json_if_present(path)
    if not isinstance(plan, Mapping):
        return {
            "status": None,
            "ready_to_execute": False,
            "blocker": "invalid_capture_execution_plan",
            "blocked_checks": [],
        }
    ready_to_execute = plan.get("ready_to_execute")
    status = plan.get("status")
    blocked_checks = _problem_checks_from_report(plan)
    if ready_to_execute is True and status in {"ok", "warning"}:
        return {
            "status": status,
            "ready_to_execute": True,
            "blocker": None,
            "blocked_checks": [],
        }
    if isinstance(status, str):
        return {
            "status": status,
            "ready_to_execute": False,
            "blocker": "failed_capture_execution_plan",
            "blocked_checks": blocked_checks,
        }
    return {
        "status": None,
        "ready_to_execute": False,
        "blocker": "invalid_capture_execution_plan",
        "blocked_checks": blocked_checks,
    }


def _bop_evaluation_report_summary(root: Path) -> dict[str, Any]:
    path = root / BOP_EVALUATION_REPORT
    if not path.is_file():
        return {
            "status": None,
            "ready_for_metrics": False,
            "blocker": "missing_bop_evaluation_report",
            "score_metric_count": 0,
        }
    report = _json_if_present(path)
    if not isinstance(report, Mapping):
        return {
            "status": None,
            "ready_for_metrics": False,
            "blocker": "invalid_bop_evaluation_report",
            "score_metric_count": 0,
        }
    status = report.get("status")
    if not isinstance(status, str):
        return {
            "status": None,
            "ready_for_metrics": False,
            "blocker": "invalid_bop_evaluation_report",
            "score_metric_count": 0,
        }
    if status not in {"planned", "succeeded"}:
        return {
            "status": status,
            "ready_for_metrics": False,
            "blocker": "failed_bop_evaluation_report",
            "critical_failed_check_count": 0,
            "score_metric_count": 0,
        }
    checks = report.get("checks", [])
    critical_failed_check_count, critical_missing_check_count = (
        _bop_evaluation_critical_check_counts(checks)
    )
    if critical_failed_check_count or critical_missing_check_count:
        return {
            "status": status,
            "ready_for_metrics": False,
            "blocker": "failed_bop_evaluation_prerequisites",
            "critical_failed_check_count": critical_failed_check_count,
            "critical_missing_check_count": critical_missing_check_count,
            "score_metric_count": 0,
        }
    score_summary = report.get("score_summary")
    score_metrics = (
        score_summary.get("metrics", {})
        if isinstance(score_summary, Mapping)
        else {}
    )
    score_metric_count = len(score_metrics) if isinstance(score_metrics, Mapping) else 0
    return {
        "status": status,
        "ready_for_metrics": True,
        "blocker": None,
        "critical_failed_check_count": 0,
        "critical_missing_check_count": 0,
        "score_metric_count": score_metric_count,
    }


def _sync_quality_report_summary(root: Path) -> dict[str, Any]:
    path = root / SYNC_QUALITY_REPORT
    if not path.is_file():
        return {
            "overall_status": None,
            "ready_for_downstream": False,
            "blocker": "missing_sync_quality_report",
        }
    report = _json_if_present(path)
    if not isinstance(report, Mapping):
        return {
            "overall_status": None,
            "ready_for_downstream": False,
            "blocker": "invalid_sync_quality_report",
        }
    overall_status = report.get("overall_status")
    if not isinstance(overall_status, str):
        return {
            "overall_status": None,
            "ready_for_downstream": False,
            "blocker": "invalid_sync_quality_report",
        }
    if overall_status == "error":
        return {
            "overall_status": overall_status,
            "ready_for_downstream": False,
            "blocker": "failed_sync_quality_report",
        }
    return {
        "overall_status": overall_status,
        "ready_for_downstream": True,
        "blocker": None,
    }


def _aruco_coverage_report_summary(root: Path) -> dict[str, Any]:
    path = root / ARUCO_COVERAGE_REPORT
    if not path.is_file():
        return {
            "overall_status": None,
            "ready_for_downstream": False,
            "blocker": "missing_aruco_coverage_report",
        }
    report = _json_if_present(path)
    if not isinstance(report, Mapping):
        return {
            "overall_status": None,
            "ready_for_downstream": False,
            "blocker": "invalid_aruco_coverage_report",
        }
    overall_status = report.get("overall_status")
    if not isinstance(overall_status, str):
        return {
            "overall_status": None,
            "ready_for_downstream": False,
            "blocker": "invalid_aruco_coverage_report",
        }
    if overall_status == "error":
        return {
            "overall_status": overall_status,
            "ready_for_downstream": False,
            "blocker": "failed_aruco_coverage_report",
        }
    return {
        "overall_status": overall_status,
        "ready_for_downstream": True,
        "blocker": None,
    }


def _calibration_preflight_summary(root: Path) -> dict[str, Any]:
    path = root / CALIBRATION_PREFLIGHT_REPORT
    if not path.is_file():
        return {
            "overall_status": None,
            "ready_for_calibrated_stages": False,
            "blocker": "missing_calibration_preflight",
        }
    report = _json_if_present(path)
    if not isinstance(report, Mapping):
        return {
            "overall_status": None,
            "ready_for_calibrated_stages": False,
            "blocker": "invalid_calibration_preflight",
        }
    overall_status = report.get("overall_status")
    if not isinstance(overall_status, str):
        return {
            "overall_status": None,
            "ready_for_calibrated_stages": False,
            "blocker": "invalid_calibration_preflight",
        }
    if overall_status == "error":
        return {
            "overall_status": overall_status,
            "ready_for_calibrated_stages": False,
            "blocker": "failed_calibration_preflight",
        }
    return {
        "overall_status": overall_status,
        "ready_for_calibrated_stages": True,
        "blocker": None,
    }


def _calibration_observations_summary(root: Path) -> dict[str, Any]:
    path = root / CALIBRATION_OBSERVATIONS
    if not path.is_file():
        return {
            "overall_status": None,
            "ready_for_solver": False,
            "blocker": "missing_calibration_observations",
        }
    report = _json_if_present(path)
    if not isinstance(report, Mapping):
        return {
            "overall_status": None,
            "ready_for_solver": False,
            "blocker": "invalid_calibration_observations",
        }
    overall_status = report.get("overall_status")
    if not isinstance(overall_status, str):
        return {
            "overall_status": None,
            "ready_for_solver": False,
            "blocker": "invalid_calibration_observations",
        }
    if overall_status == "error":
        return {
            "overall_status": overall_status,
            "ready_for_solver": False,
            "blocker": "failed_calibration_observations",
        }
    return {
        "overall_status": overall_status,
        "ready_for_solver": True,
        "blocker": None,
    }


def _calibration_solver_summary(root: Path) -> dict[str, Any]:
    path = root / CALIBRATION_SOLVER_REPORT
    if not path.is_file():
        return {
            "overall_status": None,
            "ready_for_candidates": False,
            "blocker": "missing_calibration_solver",
        }
    report = _json_if_present(path)
    if not isinstance(report, Mapping):
        return {
            "overall_status": None,
            "ready_for_candidates": False,
            "blocker": "invalid_calibration_solver",
        }
    overall_status = report.get("overall_status")
    if not isinstance(overall_status, str):
        return {
            "overall_status": None,
            "ready_for_candidates": False,
            "blocker": "invalid_calibration_solver",
        }
    if overall_status == "error":
        return {
            "overall_status": overall_status,
            "ready_for_candidates": False,
            "blocker": "failed_calibration_solver",
        }
    return {
        "overall_status": overall_status,
        "ready_for_candidates": True,
        "blocker": None,
    }


def _calibration_candidates_summary(root: Path) -> dict[str, Any]:
    path = root / CALIBRATION_CANDIDATES
    if not path.is_file():
        return {
            "overall_status": None,
            "ready_for_validation": False,
            "blocker": "missing_calibration_candidates",
        }
    report = _json_if_present(path)
    if not isinstance(report, Mapping):
        return {
            "overall_status": None,
            "ready_for_validation": False,
            "blocker": "invalid_calibration_candidates",
        }
    overall_status = report.get("overall_status")
    if not isinstance(overall_status, str):
        return {
            "overall_status": None,
            "ready_for_validation": False,
            "blocker": "invalid_calibration_candidates",
        }
    if overall_status == "error":
        return {
            "overall_status": overall_status,
            "ready_for_validation": False,
            "blocker": "failed_calibration_candidates",
        }
    return {
        "overall_status": overall_status,
        "ready_for_validation": True,
        "blocker": None,
    }


def _calibration_validation_summary(root: Path) -> dict[str, Any]:
    path = root / CALIBRATION_VALIDATION_REPORT
    if not path.is_file():
        return {
            "overall_status": None,
            "ready_for_profiles": False,
            "blocker": "missing_calibration_validation",
        }
    report = _json_if_present(path)
    if not isinstance(report, Mapping):
        return {
            "overall_status": None,
            "ready_for_profiles": False,
            "blocker": "invalid_calibration_validation",
        }
    overall_status = report.get("overall_status")
    if not isinstance(overall_status, str):
        return {
            "overall_status": None,
            "ready_for_profiles": False,
            "blocker": "invalid_calibration_validation",
        }
    if overall_status == "error":
        return {
            "overall_status": overall_status,
            "ready_for_profiles": False,
            "blocker": "failed_calibration_validation",
        }
    return {
        "overall_status": overall_status,
        "ready_for_profiles": True,
        "blocker": None,
    }


def _bop_result_export_summary(root: Path) -> dict[str, Any]:
    path = root / BOP_RESULT_EXPORT_MANIFEST
    if not path.is_file():
        return {
            "ready_for_evaluation": False,
            "blocker": "missing_bop_result_export_manifest",
            "result_count": 0,
            "usable_result_count": 0,
        }
    manifest = _json_if_present(path)
    if not isinstance(manifest, Mapping):
        return {
            "ready_for_evaluation": False,
            "blocker": "invalid_bop_result_export_manifest",
            "result_count": 0,
            "usable_result_count": 0,
        }
    results = manifest.get("results")
    if not isinstance(results, list):
        return {
            "ready_for_evaluation": False,
            "blocker": "invalid_bop_result_export_manifest",
            "result_count": 0,
            "usable_result_count": 0,
        }
    result_count = len(results)
    usable_result_count = 0
    for result in results:
        if not isinstance(result, Mapping):
            continue
        result_path = result.get("path")
        if isinstance(result_path, str) and Path(result_path).is_file():
            usable_result_count += 1
    if usable_result_count == 0:
        return {
            "ready_for_evaluation": False,
            "blocker": "missing_bop_result_csv",
            "result_count": result_count,
            "usable_result_count": usable_result_count,
        }
    return {
        "ready_for_evaluation": True,
        "blocker": None,
        "result_count": result_count,
        "usable_result_count": usable_result_count,
    }


def _bop_export_summary(root: Path) -> dict[str, Any]:
    path = root / BOP_DIR / BOP_EXPORT_MANIFEST
    if not path.is_file():
        return {
            "ready_for_results": False,
            "blocker": "missing_bop_export_manifest",
            "export_count": 0,
        }
    manifest = _json_if_present(path)
    if not isinstance(manifest, Mapping):
        return {
            "ready_for_results": False,
            "blocker": "invalid_bop_export_manifest",
            "export_count": 0,
        }
    exports = manifest.get("exports")
    if not isinstance(exports, list):
        return {
            "ready_for_results": False,
            "blocker": "invalid_bop_export_manifest",
            "export_count": 0,
        }
    export_count = len(exports)
    if export_count == 0:
        return {
            "ready_for_results": False,
            "blocker": "empty_bop_export_manifest",
            "export_count": export_count,
        }
    return {
        "ready_for_results": True,
        "blocker": None,
        "export_count": export_count,
    }


def _bop_targets_summary(root: Path) -> dict[str, Any]:
    path = root / BOP_DIR / BOP_TARGETS_BOP19
    if not path.is_file():
        return {
            "ready_for_evaluation": False,
            "blocker": "missing_bop_targets",
            "target_count": 0,
        }
    targets = _json_if_present(path)
    if not isinstance(targets, list):
        return {
            "ready_for_evaluation": False,
            "blocker": "invalid_bop_targets",
            "target_count": 0,
        }
    target_count = len(targets)
    if target_count == 0:
        return {
            "ready_for_evaluation": False,
            "blocker": "empty_bop_targets",
            "target_count": target_count,
        }
    required_fields = ("scene_id", "im_id", "obj_id", "inst_count")
    for target in targets:
        if not isinstance(target, Mapping):
            return {
                "ready_for_evaluation": False,
                "blocker": "invalid_bop_targets",
                "target_count": target_count,
            }
        for field in required_fields:
            try:
                int(target[field])
            except (KeyError, TypeError, ValueError):
                return {
                    "ready_for_evaluation": False,
                    "blocker": "invalid_bop_targets",
                    "target_count": target_count,
                }
    return {
        "ready_for_evaluation": True,
        "blocker": None,
        "target_count": target_count,
    }


def _bop_object_model_summary(root: Path) -> dict[str, Any]:
    manifest_path = root / BOP_DIR / BOP_EXPORT_MANIFEST
    object_model_count = 0
    if manifest_path.is_file():
        manifest = _json_if_present(manifest_path)
        if isinstance(manifest, Mapping):
            object_models = manifest.get("object_models")
            if isinstance(object_models, list):
                for model in object_models:
                    if not isinstance(model, Mapping):
                        continue
                    object_name = model.get("object_name")
                    obj_id = model.get("obj_id")
                    if isinstance(object_name, str) and obj_id is not None:
                        try:
                            int(obj_id)
                        except (TypeError, ValueError):
                            continue
                        object_model_count += 1
    if object_model_count > 0:
        return {
            "ready_for_result_export": True,
            "blocker": None,
            "object_model_count": object_model_count,
        }

    models_info_path = root / BOP_DIR / MODELS_DIR / "models_info.json"
    if not models_info_path.is_file():
        return {
            "ready_for_result_export": False,
            "blocker": "missing_bop_object_models",
            "object_model_count": 0,
        }
    models_info = _json_if_present(models_info_path)
    if not isinstance(models_info, Mapping):
        return {
            "ready_for_result_export": False,
            "blocker": "invalid_bop_object_models",
            "object_model_count": 0,
        }
    for obj_id, model_info in models_info.items():
        if not isinstance(model_info, Mapping):
            continue
        source_name = model_info.get("source_name")
        if not isinstance(source_name, str):
            continue
        try:
            int(obj_id)
        except (TypeError, ValueError):
            continue
        object_model_count += 1
    if object_model_count == 0:
        return {
            "ready_for_result_export": False,
            "blocker": "missing_bop_object_models",
            "object_model_count": 0,
        }
    return {
        "ready_for_result_export": True,
        "blocker": None,
        "object_model_count": object_model_count,
    }


def _metric_report_summary(root: Path) -> dict[str, Any]:
    path = root / RESULTS_DIR / METRICS_DIR / METRIC_REPORT_JSON
    if not path.is_file():
        return {
            "ready_for_dashboard": False,
            "blocker": "missing_metric_report",
            "row_count": 0,
        }
    report = _json_if_present(path)
    if not isinstance(report, Mapping):
        return {
            "ready_for_dashboard": False,
            "blocker": "invalid_metric_report",
            "row_count": 0,
        }
    rows = report.get("rows")
    dashboard = report.get("dashboard")
    if not isinstance(rows, list) or not isinstance(dashboard, Mapping):
        return {
            "ready_for_dashboard": False,
            "blocker": "invalid_metric_report",
            "row_count": 0,
        }
    row_count = len(rows)
    if row_count == 0:
        return {
            "ready_for_dashboard": False,
            "blocker": "empty_metric_report",
            "row_count": row_count,
        }
    return {
        "ready_for_dashboard": True,
        "blocker": None,
        "row_count": row_count,
    }


def _rewrite_status_file_summary(
    root: Path,
    current_report: Mapping[str, Any],
) -> dict[str, Any]:
    path = root / REWRITE_STATUS_REPORT
    current_summary = current_report.get("summary")
    current_gates = current_report.get("gates")
    if not path.is_file():
        return {
            "ready_for_inspection": False,
            "blocker": "missing_rewrite_status_report",
        }
    report = _json_if_present(path)
    if not isinstance(report, Mapping):
        return {
            "ready_for_inspection": False,
            "blocker": "invalid_rewrite_status_report",
        }
    if report.get("schema_version") != current_report.get("schema_version"):
        return {
            "ready_for_inspection": False,
            "blocker": "stale_rewrite_status_report",
        }
    if report.get("overall_status") != current_report.get("overall_status"):
        return {
            "ready_for_inspection": False,
            "blocker": "stale_rewrite_status_report",
        }
    if report.get("summary") != current_summary:
        return {
            "ready_for_inspection": False,
            "blocker": "stale_rewrite_status_report",
        }
    if report.get("gate_run_roots") != current_report.get("gate_run_roots"):
        return {
            "ready_for_inspection": False,
            "blocker": "stale_rewrite_status_report",
        }
    if _gate_statuses(report.get("gates")) != _gate_statuses(current_gates):
        return {
            "ready_for_inspection": False,
            "blocker": "stale_rewrite_status_report",
        }
    return {
        "ready_for_inspection": True,
        "blocker": None,
    }


def _saved_rewrite_status_gate_run_roots(root: Path) -> dict[str, Path]:
    report = _json_if_present(root / REWRITE_STATUS_REPORT)
    if not isinstance(report, Mapping):
        return {}
    gate_run_roots = report.get("gate_run_roots")
    if not isinstance(gate_run_roots, Mapping):
        return {}
    parsed: dict[str, Path] = {}
    for gate_id, gate_root in gate_run_roots.items():
        if isinstance(gate_id, str) and isinstance(gate_root, str) and gate_root:
            parsed[gate_id] = Path(gate_root)
    return parsed


def _rewrite_status_gate_run_root_options(gate_run_roots: Mapping[str, Path]) -> list[str]:
    return [
        f"{gate_id}={gate_root.as_posix()}"
        for gate_id, gate_root in gate_run_roots.items()
    ]


def _rewrite_status_next_actions(
    report: Mapping[str, Any],
    *,
    limit: int = 3,
) -> list[Mapping[str, Any]]:
    next_actions = report.get("next_actions")
    if not isinstance(next_actions, list) or not next_actions:
        return []
    actions: list[Mapping[str, Any]] = []
    for action in next_actions:
        if not isinstance(action, Mapping):
            continue
        if not _rewrite_status_action_command(action):
            continue
        actions.append(action)
        if len(actions) >= limit:
            break
    return actions


def _rewrite_status_action_command(action: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(action, Mapping):
        return []
    command = action.get("command")
    if not isinstance(command, list):
        return []
    return [str(part) for part in command if isinstance(part, str)]


def _rewrite_status_first_blocker_context(
    report: Mapping[str, Any],
) -> tuple[str | None, str | None, str | None]:
    next_blockers = report.get("next_blockers")
    if not isinstance(next_blockers, list):
        return None, None, None
    for blocker in next_blockers:
        if not isinstance(blocker, Mapping):
            continue
        name = blocker.get("name")
        message = blocker.get("message")
        diagnostic_message = None
        details = blocker.get("details")
        if isinstance(details, Mapping):
            diagnostics = details.get("sensor_diagnostics")
            if isinstance(diagnostics, list):
                for diagnostic in diagnostics:
                    if isinstance(diagnostic, Mapping) and diagnostic.get("message"):
                        diagnostic_message = str(diagnostic["message"])
                        break
        return (
            str(name) if name else None,
            str(message) if message else None,
            diagnostic_message,
        )
    return None, None, None


def _gate_statuses(gates: object) -> dict[str, str]:
    if not isinstance(gates, list):
        return {}
    statuses: dict[str, str] = {}
    for gate in gates:
        if not isinstance(gate, Mapping):
            continue
        gate_id = gate.get("gate_id")
        overall_status = gate.get("overall_status")
        if isinstance(gate_id, str) and isinstance(overall_status, str):
            statuses[gate_id] = overall_status
    return statuses


def _estimator_plan_summary(
    run_root: Path,
    artifact_name: str,
    *,
    expected_estimator_id: str,
) -> dict[str, Any]:
    path = run_root / artifact_name
    if not path.is_file():
        return {
            "ready_for_jobs": False,
            "blocker": f"missing_{expected_estimator_id}_plan",
            "job_count": 0,
        }
    plan = _json_if_present(path)
    if not isinstance(plan, Mapping):
        return {
            "ready_for_jobs": False,
            "blocker": f"invalid_{expected_estimator_id}_plan",
            "job_count": 0,
        }
    estimator_id = plan.get("estimator_id", expected_estimator_id)
    if estimator_id != expected_estimator_id:
        return {
            "ready_for_jobs": False,
            "blocker": f"invalid_{expected_estimator_id}_plan",
            "job_count": 0,
        }
    jobs = plan.get("jobs")
    if not isinstance(jobs, list):
        return {
            "ready_for_jobs": False,
            "blocker": f"invalid_{expected_estimator_id}_plan",
            "job_count": 0,
        }
    job_count = len(jobs)
    if job_count == 0:
        return {
            "ready_for_jobs": False,
            "blocker": f"empty_{expected_estimator_id}_plan",
            "job_count": job_count,
        }
    return {
        "ready_for_jobs": True,
        "blocker": None,
        "job_count": job_count,
    }


def _has_raw_sensor_folders(run_root: Path) -> bool:
    if not run_root.is_dir():
        return False
    return any(
        child.is_dir() and child.name.startswith(RAW_SENSOR_PREFIXES)
        for child in run_root.iterdir()
    )


def _synchronized_root(run_root: Path) -> Path:
    return run_root / PROCESSED_DIR / SYNCHRONIZED_DIR


def _synchronized_sensor_dirs(run_root: Path) -> list[Path]:
    root = _synchronized_root(run_root)
    if not root.is_dir():
        return []
    return [child for child in sorted(root.iterdir()) if child.is_dir()]


def _has_aruco_outputs(run_root: Path) -> bool:
    return any(
        (sensor / ARUCO_POSE_ESTIMATION).is_file()
        for sensor in _synchronized_sensor_dirs(run_root)
    )


def _has_calibration_target_pose_outputs(run_root: Path) -> bool:
    return any(
        (sensor / artifact_name).is_file()
        for sensor in _synchronized_sensor_dirs(run_root)
        for artifact_name in CALIBRATION_TARGET_POSE_ARTIFACTS
    )


def _has_non_aruco_calibration_target_pose_outputs(run_root: Path) -> bool:
    return any(
        (sensor / artifact_name).is_file()
        for sensor in _synchronized_sensor_dirs(run_root)
        for artifact_name in CALIBRATION_TARGET_POSE_ARTIFACTS
        if artifact_name != ARUCO_POSE_ESTIMATION
    )


def _has_blenderproc_prepared(run_root: Path) -> bool:
    return any(
        (sensor / "blenderproc" / "objects.json").is_file()
        for sensor in _synchronized_sensor_dirs(run_root)
    )


def _has_estimator_outputs(
    run_root: Path,
    prefix: str,
    *,
    required_child: str | None = None,
    required_child_type: str = "any",
) -> bool:
    return any(
        child.is_dir() and child.name.startswith(prefix) and child.name.endswith("_output")
        and _required_child_exists(
            child,
            required_child=required_child,
            required_child_type=required_child_type,
        )
        for sensor in _synchronized_sensor_dirs(run_root)
        for child in sensor.iterdir()
    )


def _required_child_exists(
    output_folder: Path,
    *,
    required_child: str | None,
    required_child_type: str,
) -> bool:
    if required_child is None:
        return True
    path = output_folder / required_child
    if required_child_type == "file":
        return path.is_file()
    if required_child_type == "dir":
        return path.is_dir()
    if required_child_type == "any":
        return path.exists()
    raise ValueError(f"Unsupported required child type: {required_child_type}")


def _first_result_csv(run_root: Path) -> Path | None:
    manifest = _json_if_present(run_root / BOP_RESULT_EXPORT_MANIFEST)
    if isinstance(manifest, Mapping):
        for result in manifest.get("results", []):
            if not isinstance(result, Mapping):
                continue
            path = result.get("path")
            if not isinstance(path, str):
                continue
            result_path = Path(path)
            if result_path.is_file():
                return result_path

    result_root = run_root / RESULTS_DIR / BOP_DIR
    if not result_root.is_dir():
        return None
    for path in sorted(result_root.glob("*.csv")):
        if path.is_file():
            return path
    return None


def _has_legacy_metric_artifacts(run_root: Path) -> bool:
    if not run_root.exists():
        return False
    metric_names = {ACCURACY_HRC_HUB, ACCURACY_ARUCO_HRC_HUB, ALL_RESULTS_JSON}
    return any(path.is_file() and path.name in metric_names for path in run_root.rglob("*"))


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
    )


def _run_config_sequence_id(run_root: Path) -> str | None:
    value = _json_if_present(run_root / RUN_CONFIG)
    if not isinstance(value, Mapping):
        return None
    pipeline = value.get("pipeline")
    if not isinstance(pipeline, Mapping):
        return None
    sequence_id = pipeline.get("sequence_id")
    return sequence_id if isinstance(sequence_id, str) else None


def _calibration_profile_source(run_root: Path) -> str | None:
    value = _json_if_present(run_root / RUN_CONFIG)
    if isinstance(value, Mapping):
        configured = value.get("calibration_profiles")
        if isinstance(configured, str) and configured.strip():
            return configured
    default_path = run_root / CALIBRATION_PROFILES
    return CALIBRATION_PROFILES if default_path.is_file() else None


def _run_config_sequence_has_stage(run_root: Path, stage_id: str) -> bool:
    value = _json_if_present(run_root / RUN_CONFIG)
    if not isinstance(value, Mapping):
        return False
    try:
        plan = sequence_plan_from_run_config(value)
    except Exception:
        return False
    return any(step.stage_id == stage_id for step in plan.steps)


def _run_config_robot_mode(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    robot_profile = value.get("robot_profile")
    if isinstance(robot_profile, Mapping) and robot_profile.get("mode"):
        return str(robot_profile["mode"])
    legacy_robot = value.get("robot")
    if isinstance(legacy_robot, Mapping) and legacy_robot.get("mode"):
        return str(legacy_robot["mode"])
    return None


def build_pipeline_recommendations(run_root: str | Path) -> dict[str, Any]:
    """Return read-only next-step suggestions derived from run artifacts."""

    root = Path(run_root)
    synchronized_sensors = _synchronized_sensor_dirs(root)
    first_result_csv = _first_result_csv(root)
    run_config = _json_if_present(root / RUN_CONFIG)
    run_config_robot_mode = _run_config_robot_mode(run_config)
    run_config_targets_real_robot = run_config_robot_mode == "real"
    has_run_config = (root / RUN_CONFIG).is_file()
    run_config_summary = _run_config_summary(root)
    run_config_ready_for_pipeline = bool(run_config_summary["ready_for_pipeline"])
    run_config_blocker = run_config_summary["blocker"]
    run_config_error = run_config_summary["error"]
    has_run_preflight = (root / RUN_PREFLIGHT_REPORT).is_file()
    run_preflight_summary = (
        run_preflight_queue_summary(root, run_config)
        if run_config_ready_for_pipeline and isinstance(run_config, Mapping)
        else {
            "overall_status": None,
            "matches_config": None,
            "ready_for_queue": False,
            "queue_blocker": run_config_blocker,
        }
    )
    run_preflight_status = run_preflight_summary["overall_status"]
    run_preflight_matches_config = run_preflight_summary["matches_config"]
    run_preflight_ready_for_queue = bool(run_preflight_summary["ready_for_queue"])
    run_preflight_queue_blocker = run_preflight_summary["queue_blocker"]
    has_pipeline_sequence_plan = (root / PIPELINE_SEQUENCE_PLAN).is_file()
    pipeline_sequence_plan_summary = _pipeline_sequence_plan_summary(root)
    pipeline_sequence_plan_ready_for_queue = bool(
        pipeline_sequence_plan_summary["ready_for_queue"]
    )
    pipeline_sequence_plan_blocker = pipeline_sequence_plan_summary["blocker"]
    pipeline_sequence_plan_step_count = pipeline_sequence_plan_summary["step_count"]
    has_capture_plan = (root / CAPTURE_PLAN).is_file()
    capture_plan_summary = _capture_plan_summary(root)
    capture_plan_ready_for_preflight = bool(
        capture_plan_summary["ready_for_preflight"]
    )
    capture_plan_blocker = capture_plan_summary["blocker"]
    capture_plan_command_count = capture_plan_summary["command_count"]
    has_capture_preflight = (root / CAPTURE_PLAN_PREFLIGHT_REPORT).is_file()
    capture_plan_preflight_summary = _capture_plan_preflight_summary(root)
    capture_plan_preflight_status = capture_plan_preflight_summary["overall_status"]
    capture_plan_preflight_ready = bool(
        capture_plan_preflight_summary["ready_for_execution_plan"]
    )
    capture_plan_preflight_blocker = capture_plan_preflight_summary["blocker"]
    has_capture_execution_plan = (root / CAPTURE_EXECUTION_PLAN).is_file()
    capture_execution_plan_summary = _capture_execution_plan_summary(root)
    capture_execution_plan_status = capture_execution_plan_summary["status"]
    capture_execution_plan_ready = bool(
        capture_execution_plan_summary["ready_to_execute"]
    )
    capture_execution_plan_blocker = capture_execution_plan_summary["blocker"]
    capture_execution_plan_blocked_checks = capture_execution_plan_summary[
        "blocked_checks"
    ]
    has_capture_execution_report = (root / CAPTURE_EXECUTION_REPORT).is_file()
    capture_execution_report_summary = _capture_execution_report_summary(root)
    capture_execution_report_status = capture_execution_report_summary["status"]
    capture_execution_report_ready = bool(
        capture_execution_report_summary["ready_for_downstream"]
    )
    capture_execution_report_blocker = capture_execution_report_summary["blocker"]
    has_raw_poses = (root / RAW_ROBOT_EE_POSES).is_file()
    has_raw_sensor_folders = _has_raw_sensor_folders(root)
    has_sync = bool(synchronized_sensors)
    has_sync_quality = (root / SYNC_QUALITY_REPORT).is_file()
    sync_quality_summary = _sync_quality_report_summary(root)
    sync_quality_status = sync_quality_summary["overall_status"]
    sync_quality_ready_for_downstream = bool(
        sync_quality_summary["ready_for_downstream"]
    )
    sync_quality_blocker = sync_quality_summary["blocker"]
    has_aruco = _has_aruco_outputs(root)
    calibration_profile_source = _calibration_profile_source(root)
    has_calibration_profiles_configured = calibration_profile_source is not None
    has_calibration_preflight = (root / CALIBRATION_PREFLIGHT_REPORT).is_file()
    sequence_uses_calibration_preflight = _run_config_sequence_has_stage(
        root,
        "calibration_preflight",
    )
    calibration_preflight_expected = has_run_config and (
        has_calibration_profiles_configured
        or sequence_uses_calibration_preflight
    )
    calibration_preflight_summary = _calibration_preflight_summary(root)
    calibration_preflight_status = calibration_preflight_summary["overall_status"]
    calibration_preflight_ready_for_calibrated_stages = bool(
        calibration_preflight_summary["ready_for_calibrated_stages"]
    )
    calibration_preflight_blocker = calibration_preflight_summary["blocker"]
    calibration_preflight_blocks_calibrated_stages = (
        calibration_preflight_expected
        and not calibration_preflight_ready_for_calibrated_stages
    )
    has_calibration_target_pose_outputs = _has_calibration_target_pose_outputs(root)
    has_non_aruco_calibration_target_pose_outputs = (
        _has_non_aruco_calibration_target_pose_outputs(root)
    )
    has_aruco_coverage = (root / ARUCO_COVERAGE_REPORT).is_file()
    aruco_coverage_summary = _aruco_coverage_report_summary(root)
    aruco_coverage_status = aruco_coverage_summary["overall_status"]
    aruco_coverage_ready_for_downstream = bool(
        aruco_coverage_summary["ready_for_downstream"]
    )
    aruco_coverage_blocker = aruco_coverage_summary["blocker"]
    has_calibration_observations = (root / CALIBRATION_OBSERVATIONS).is_file()
    calibration_observations_summary = _calibration_observations_summary(root)
    calibration_observations_status = calibration_observations_summary[
        "overall_status"
    ]
    calibration_observations_ready_for_solver = bool(
        calibration_observations_summary["ready_for_solver"]
    )
    calibration_observations_blocker = calibration_observations_summary["blocker"]
    has_calibration_solver = (root / CALIBRATION_SOLVER_REPORT).is_file()
    calibration_solver_summary = _calibration_solver_summary(root)
    calibration_solver_status = calibration_solver_summary["overall_status"]
    calibration_solver_ready_for_candidates = bool(
        calibration_solver_summary["ready_for_candidates"]
    )
    calibration_solver_blocker = calibration_solver_summary["blocker"]
    has_calibration_candidates = (root / CALIBRATION_CANDIDATES).is_file()
    calibration_candidates_summary = _calibration_candidates_summary(root)
    calibration_candidates_status = calibration_candidates_summary["overall_status"]
    calibration_candidates_ready_for_validation = bool(
        calibration_candidates_summary["ready_for_validation"]
    )
    calibration_candidates_blocker = calibration_candidates_summary["blocker"]
    has_calibration_validation = (root / CALIBRATION_VALIDATION_REPORT).is_file()
    calibration_validation_summary = _calibration_validation_summary(root)
    calibration_validation_status = calibration_validation_summary["overall_status"]
    calibration_validation_ready_for_profiles = bool(
        calibration_validation_summary["ready_for_profiles"]
    )
    calibration_validation_blocker = calibration_validation_summary["blocker"]
    has_blenderproc_prepared = _has_blenderproc_prepared(root)
    has_bop_export = (root / BOP_DIR / BOP_EXPORT_MANIFEST).is_file()
    bop_export_summary = _bop_export_summary(root)
    bop_export_ready_for_results = bool(bop_export_summary["ready_for_results"])
    bop_export_blocker = bop_export_summary["blocker"]
    bop_export_count = bop_export_summary["export_count"]
    bop_object_model_summary = _bop_object_model_summary(root)
    bop_object_models_ready_for_result_export = bool(
        bop_object_model_summary["ready_for_result_export"]
    )
    bop_object_models_blocker = bop_object_model_summary["blocker"]
    bop_object_model_count = bop_object_model_summary["object_model_count"]
    has_bop_targets = (root / BOP_DIR / BOP_TARGETS_BOP19).is_file()
    bop_targets_summary = _bop_targets_summary(root)
    bop_targets_ready_for_evaluation = bool(
        bop_targets_summary["ready_for_evaluation"]
    )
    bop_targets_blocker = bop_targets_summary["blocker"]
    bop_targets_count = bop_targets_summary["target_count"]
    has_foundationpose_plan = (root / FOUNDATIONPOSE_PLAN).is_file()
    foundationpose_plan_summary = _estimator_plan_summary(
        root,
        FOUNDATIONPOSE_PLAN,
        expected_estimator_id="foundationpose",
    )
    foundationpose_plan_ready_for_jobs = bool(
        foundationpose_plan_summary["ready_for_jobs"]
    )
    foundationpose_plan_blocker = foundationpose_plan_summary["blocker"]
    foundationpose_plan_job_count = foundationpose_plan_summary["job_count"]
    has_foundationpose_outputs = _has_estimator_outputs(
        root,
        "foundationpose",
        required_child="ob_in_cam",
        required_child_type="dir",
    )
    has_megapose_outputs = _has_estimator_outputs(
        root,
        "megapose",
        required_child="megapose_poses.json",
        required_child_type="file",
    )
    has_sam6d_outputs = _has_estimator_outputs(
        root,
        "sam6d",
        required_child="detections_pem",
        required_child_type="dir",
    )
    has_bop_result_export = (root / BOP_RESULT_EXPORT_MANIFEST).is_file()
    bop_result_export_summary = _bop_result_export_summary(root)
    bop_result_export_ready_for_evaluation = bool(
        bop_result_export_summary["ready_for_evaluation"]
    )
    bop_result_export_blocker = bop_result_export_summary["blocker"]
    bop_result_export_result_count = bop_result_export_summary["result_count"]
    bop_result_export_usable_result_count = bop_result_export_summary[
        "usable_result_count"
    ]
    has_bop_evaluation = (root / BOP_EVALUATION_REPORT).is_file()
    bop_evaluation_report_summary = _bop_evaluation_report_summary(root)
    bop_evaluation_report_status = bop_evaluation_report_summary["status"]
    bop_evaluation_report_ready = bool(
        bop_evaluation_report_summary["ready_for_metrics"]
    )
    bop_evaluation_report_blocker = bop_evaluation_report_summary["blocker"]
    bop_evaluation_report_critical_failed_check_count = (
        bop_evaluation_report_summary.get("critical_failed_check_count", 0)
    )
    bop_evaluation_report_critical_missing_check_count = (
        bop_evaluation_report_summary.get("critical_missing_check_count", 0)
    )
    bop_evaluation_report_score_metric_count = int(
        bop_evaluation_report_summary.get("score_metric_count", 0)
    )
    has_bop_score_metrics = (
        bop_evaluation_report_ready
        and bop_evaluation_report_score_metric_count > 0
    )
    has_legacy_metrics = _has_legacy_metric_artifacts(root)
    has_metric_sources = has_legacy_metrics or has_bop_score_metrics
    has_metric_report = (root / RESULTS_DIR / METRICS_DIR / METRIC_REPORT_JSON).is_file()
    metric_report_summary = _metric_report_summary(root)
    metric_report_ready_for_dashboard = bool(
        metric_report_summary["ready_for_dashboard"]
    )
    metric_report_blocker = metric_report_summary["blocker"]
    metric_report_row_count = metric_report_summary["row_count"]
    full_capture_gate_report = (
        build_full_capture_gate_report(root)
        if run_config_targets_real_robot
        else None
    )
    full_capture_gate_status = (
        str(full_capture_gate_report["overall_status"])
        if full_capture_gate_report is not None
        else None
    )
    full_capture_gate_blocker_count = (
        int(full_capture_gate_report["summary"]["blocked_count"])
        if full_capture_gate_report is not None
        else 0
    )
    full_capture_gate_next_blockers = (
        [
            str(blocker["name"])
            for blocker in full_capture_gate_report["next_blockers"]
            if isinstance(blocker, Mapping) and blocker.get("name")
        ]
        if full_capture_gate_report is not None
        else []
    )
    bop_evaluation_value = _json_if_present(root / BOP_EVALUATION_REPORT)
    bop_evaluation_result = (
        bop_evaluation_value.get("result")
        if isinstance(bop_evaluation_value, Mapping)
        else None
    )
    bop_evaluation_method = (
        bop_evaluation_result.get("method")
        if isinstance(bop_evaluation_result, Mapping)
        else None
    )
    foundationpose_runtime_gate_expected = (
        has_foundationpose_plan
        or has_foundationpose_outputs
        or (has_bop_evaluation and bop_evaluation_method == "foundationpose")
    )
    foundationpose_runtime_gate_report = (
        build_foundationpose_runtime_gate_report(root)
        if foundationpose_runtime_gate_expected
        else None
    )
    foundationpose_runtime_gate_status = (
        str(foundationpose_runtime_gate_report["overall_status"])
        if foundationpose_runtime_gate_report is not None
        else None
    )
    foundationpose_runtime_gate_blocker_count = (
        int(foundationpose_runtime_gate_report["summary"]["blocked_count"])
        if foundationpose_runtime_gate_report is not None
        else 0
    )
    foundationpose_runtime_gate_next_blockers = (
        [
            str(blocker["name"])
            for blocker in foundationpose_runtime_gate_report["next_blockers"]
            if isinstance(blocker, Mapping) and blocker.get("name")
        ]
        if foundationpose_runtime_gate_report is not None
        else []
    )
    calibration_validation_gate_expected = (
        has_calibration_validation or (root / CALIBRATION_PROFILES).is_file()
    )
    calibration_validation_gate_report = (
        build_calibration_validation_gate_report(root)
        if calibration_validation_gate_expected
        else None
    )
    calibration_validation_gate_status = (
        str(calibration_validation_gate_report["overall_status"])
        if calibration_validation_gate_report is not None
        else None
    )
    calibration_validation_gate_blocker_count = (
        int(calibration_validation_gate_report["summary"]["blocked_count"])
        if calibration_validation_gate_report is not None
        else 0
    )
    calibration_validation_gate_next_blockers = (
        [
            str(blocker["name"])
            for blocker in calibration_validation_gate_report["next_blockers"]
            if isinstance(blocker, Mapping) and blocker.get("name")
        ]
        if calibration_validation_gate_report is not None
        else []
    )
    has_rewrite_status_report = (root / REWRITE_STATUS_REPORT).is_file()
    saved_rewrite_status_gate_run_roots = _saved_rewrite_status_gate_run_roots(root)
    rewrite_status_report = build_rewrite_status_report(
        root,
        gate_run_roots=saved_rewrite_status_gate_run_roots,
    )
    rewrite_status_summary = rewrite_status_report["summary"]
    rewrite_status_overall_status = str(rewrite_status_report["overall_status"])
    rewrite_status_ready_gate_count = int(rewrite_status_summary["ready_gate_count"])
    rewrite_status_gate_count = int(rewrite_status_summary["gate_count"])
    rewrite_status_ready_check_count = int(
        rewrite_status_summary["ready_check_count"]
    )
    rewrite_status_check_count = int(rewrite_status_summary["check_count"])
    rewrite_status_next_blockers = [
        str(blocker["name"])
        for blocker in rewrite_status_report["next_blockers"]
        if isinstance(blocker, Mapping) and blocker.get("name")
    ]
    rewrite_status_expected = any(
        (
            has_rewrite_status_report,
            has_run_config,
            has_capture_execution_report,
            has_raw_poses,
            has_raw_sensor_folders,
            has_sync,
            has_bop_export,
            has_bop_result_export,
            has_bop_evaluation,
            has_metric_report,
            full_capture_gate_report is not None,
            foundationpose_runtime_gate_expected,
            calibration_validation_gate_expected,
        )
    )
    rewrite_status_file_summary = (
        _rewrite_status_file_summary(root, rewrite_status_report)
        if rewrite_status_expected
        else {"ready_for_inspection": False, "blocker": None}
    )
    rewrite_status_ready_for_inspection = bool(
        rewrite_status_file_summary["ready_for_inspection"]
    )
    rewrite_status_blocker = rewrite_status_file_summary["blocker"]
    rewrite_status_next_actions = _rewrite_status_next_actions(
        rewrite_status_report
    )
    rewrite_status_next_action = (
        rewrite_status_next_actions[0] if rewrite_status_next_actions else None
    )
    rewrite_status_next_action_label = (
        str(rewrite_status_next_action.get("label"))
        if isinstance(rewrite_status_next_action, Mapping)
        and rewrite_status_next_action.get("label")
        else None
    )
    rewrite_status_next_action_command = _rewrite_status_action_command(
        rewrite_status_next_action
    )
    rewrite_status_next_action_labels = [
        str(action.get("label"))
        for action in rewrite_status_next_actions
        if action.get("label")
    ]
    rewrite_status_next_action_commands = [
        _rewrite_status_action_command(action)
        for action in rewrite_status_next_actions
    ]
    rewrite_status_has_guided_next_action = bool(
        rewrite_status_expected
        and rewrite_status_ready_for_inspection
        and rewrite_status_overall_status != "ready"
        and rewrite_status_next_actions
    )
    (
        rewrite_status_first_blocker,
        rewrite_status_first_blocker_message,
        rewrite_status_first_blocker_diagnostic,
    ) = _rewrite_status_first_blocker_context(rewrite_status_report)
    can_build_calibration_observations = has_calibration_target_pose_outputs and (
        (not has_aruco)
        or aruco_coverage_ready_for_downstream
        or has_non_aruco_calibration_target_pose_outputs
    )

    capture_preflight_options = (
        {"allow_real_robot": True}
        if run_config_targets_real_robot
        else {"no_sensors": True}
    )
    capture_execution_options = (
        {
            "mode": "full",
            "allow_cameras": True,
            "allow_real_robot": True,
            "include_sensors": True,
        }
        if run_config_targets_real_robot
        else {"mode": "pose_only_fake"}
    )
    capture_execution_kind = "full" if run_config_targets_real_robot else "fake"

    should_write_run_preflight = False
    recommendations: list[PipelineRecommendation] = []

    if not run_config_ready_for_pipeline:
        if not rewrite_status_has_guided_next_action:
            if run_config_blocker == "missing_run_config":
                run_config_reason = f"Missing {RUN_CONFIG}."
            else:
                run_config_reason = (
                    f"{RUN_CONFIG} is invalid and should be rewritten"
                    f": {run_config_error}"
                )
            recommendations.append(
                PipelineRecommendation(
                    id="create_run_config",
                    label=(
                        "Create run config" if not has_run_config else "Rewrite run config"
                    ),
                    description=(
                        "Write the fake-iiwa-first run_config.json before queueing "
                        "typed stages."
                    ),
                    reason=run_config_reason,
                    priority=10,
                    action_type="api",
                    command=[
                        "uv",
                        "run",
                        "python",
                        "scripts/create_run_config.py",
                        root.as_posix(),
                        "--sequence",
                        "sync_to_bop_dry_run",
                        "--print-sequence-plan",
                    ],
                    endpoint="/run-config",
                    method="POST",
                    expected_artifacts=[RUN_CONFIG],
                )
            )
    else:
        should_write_run_preflight = not run_preflight_ready_for_queue
        if should_write_run_preflight:
            if run_preflight_queue_blocker == "missing_preflight":
                run_preflight_reason = (
                    f"{RUN_CONFIG} exists but {RUN_PREFLIGHT_REPORT} is missing."
                )
            elif run_preflight_queue_blocker == "failed_preflight":
                run_preflight_reason = (
                    f"{RUN_PREFLIGHT_REPORT} has overall_status=error."
                )
            elif run_preflight_queue_blocker == "invalid_preflight":
                run_preflight_reason = (
                    f"{RUN_PREFLIGHT_REPORT} is invalid and should be rewritten."
                )
            else:
                run_preflight_reason = (
                    f"{RUN_PREFLIGHT_REPORT} does not match the current "
                    f"{RUN_CONFIG}."
                )
            recommendations.append(
                PipelineRecommendation(
                    id="write_run_preflight",
                    label=(
                        "Write run preflight"
                        if not has_run_preflight
                        else "Refresh run preflight"
                    ),
                    description=(
                        "Persist robot, sensor, runtime, sequence, and input "
                        "readiness checks before queueing the saved workflow."
                    ),
                    reason=run_preflight_reason,
                    priority=15,
                    action_type="api",
                    command=[
                        "uv",
                        "run",
                        "python",
                        "scripts/run_preflight.py",
                        root.as_posix(),
                        "--write",
                    ],
                    endpoint="/pipeline/preflight",
                    method="POST",
                    expected_artifacts=[RUN_PREFLIGHT_REPORT],
                    resources=["disk_io"],
                )
            )

        if (
            not should_write_run_preflight
            and not pipeline_sequence_plan_ready_for_queue
        ):
            sequence_id = _run_config_sequence_id(root)
            if pipeline_sequence_plan_blocker == "missing_pipeline_sequence_plan":
                pipeline_sequence_reason = (
                    f"{RUN_CONFIG} exists but {PIPELINE_SEQUENCE_PLAN} "
                    "has not been written."
                )
            else:
                pipeline_sequence_reason = (
                    f"{PIPELINE_SEQUENCE_PLAN} is not ready for queueing "
                    f"({pipeline_sequence_plan_blocker})."
                )
            recommendations.append(
                PipelineRecommendation(
                    id="queue_saved_sequence",
                    label="Queue saved sequence",
                    description=(
                        "Queue the sequence recorded in run_config.json through "
                        "the transition job runner."
                    ),
                    reason=pipeline_sequence_reason,
                    priority=18,
                    action_type="sequence",
                    command=[
                        "uv",
                        "run",
                        "python",
                        "scripts/run_pipeline_sequence.py",
                        root.as_posix(),
                        "--sequence",
                        sequence_id or "sync_to_bop_dry_run",
                        "--plan-only",
                    ],
                    endpoint="/pipeline/run-config",
                    method="POST",
                    sequence_id=sequence_id,
                    expected_artifacts=[PIPELINE_SEQUENCE_PLAN],
                    resources=["disk_io"],
                )
            )

        if not capture_plan_ready_for_preflight:
            if capture_plan_blocker == "missing_capture_plan":
                capture_plan_reason = f"{RUN_CONFIG} exists but {CAPTURE_PLAN} is missing."
            else:
                capture_plan_reason = (
                    f"{CAPTURE_PLAN} is not ready for preflight "
                    f"({capture_plan_blocker})."
                )
            recommendations.append(
                _stage_recommendation(
                    recommendation_id="write_capture_plan",
                    stage_id="capture_plan",
                    run_root=root,
                    label="Write capture plan",
                    description=(
                        "Create capture_plan.json with fake iiwa, pose receiver, "
                        "and configured sensor commands without starting them."
                    ),
                    reason=capture_plan_reason,
                    priority=20,
                    expected_artifacts=[CAPTURE_PLAN],
                    endpoint="/capture-plan",
                )
            )
        elif not capture_plan_preflight_ready:
            if capture_plan_preflight_blocker == "missing_capture_plan_preflight":
                capture_plan_preflight_reason = (
                    f"{CAPTURE_PLAN} exists but "
                    f"{CAPTURE_PLAN_PREFLIGHT_REPORT} is missing."
                )
            elif capture_plan_preflight_blocker == "invalid_capture_plan_preflight":
                capture_plan_preflight_reason = (
                    f"{CAPTURE_PLAN_PREFLIGHT_REPORT} is invalid and should be rebuilt."
                )
            else:
                capture_plan_preflight_reason = (
                    f"{CAPTURE_PLAN_PREFLIGHT_REPORT} has "
                    f"overall_status={capture_plan_preflight_status}."
                )
            recommendations.append(
                _stage_recommendation(
                    recommendation_id="preflight_capture_plan",
                    stage_id="capture_plan_preflight",
                    run_root=root,
                    label="Preflight capture plan",
                    description=(
                        "Check command shape, fake/real robot safety, scripts, "
                        "and optional sensor readiness before execution."
                    ),
                    reason=capture_plan_preflight_reason,
                    priority=30,
                    expected_artifacts=[CAPTURE_PLAN_PREFLIGHT_REPORT],
                    options=capture_preflight_options,
                    endpoint="/capture-plan/preflight",
                )
            )
        elif not capture_execution_plan_ready:
            if capture_execution_plan_blocker == "missing_capture_execution_plan":
                capture_execution_plan_reason = (
                    f"{CAPTURE_PLAN_PREFLIGHT_REPORT} exists but "
                    f"{CAPTURE_EXECUTION_PLAN} is missing."
                )
            elif capture_execution_plan_blocker == "invalid_capture_execution_plan":
                capture_execution_plan_reason = (
                    f"{CAPTURE_EXECUTION_PLAN} is invalid and should be rebuilt."
                )
            else:
                first_blocked_check = (
                    capture_execution_plan_blocked_checks[0]
                    if capture_execution_plan_blocked_checks
                    and isinstance(capture_execution_plan_blocked_checks[0], Mapping)
                    else None
                )
                capture_execution_plan_reason = (
                    (
                        f"{CAPTURE_EXECUTION_PLAN} is blocked by "
                        f"{first_blocked_check['name']}: "
                        f"{first_blocked_check['message']}."
                    )
                    if first_blocked_check
                    and first_blocked_check.get("name")
                    and first_blocked_check.get("message")
                    else (
                        f"{CAPTURE_EXECUTION_PLAN} has "
                        f"status={capture_execution_plan_status}."
                    )
                )
            recommendations.append(
                _stage_recommendation(
                    recommendation_id=f"plan_{capture_execution_kind}_capture_execution",
                    stage_id="capture_execution_plan",
                    run_root=root,
                    label=(
                        "Plan full capture execution"
                        if run_config_targets_real_robot
                        else "Plan fake capture execution"
                    ),
                    description=(
                        (
                            "Select real robot and camera commands from "
                            "capture_plan.json without starting processes."
                        )
                        if run_config_targets_real_robot
                        else (
                            "Select the safe pose-only fake iiwa command set from "
                            "capture_plan.json without starting processes."
                        )
                    ),
                    reason=capture_execution_plan_reason,
                    priority=40,
                    expected_artifacts=[CAPTURE_EXECUTION_PLAN],
                    options=capture_execution_options,
                    endpoint="/capture-plan/execution",
                )
            )
        elif not capture_execution_report_ready:
            if capture_execution_report_blocker == "missing_capture_execution_report":
                capture_execution_reason = (
                    f"{CAPTURE_EXECUTION_PLAN} exists but "
                    f"{CAPTURE_EXECUTION_REPORT} is missing."
                )
            elif capture_execution_report_blocker == "invalid_capture_execution_report":
                capture_execution_reason = (
                    f"{CAPTURE_EXECUTION_REPORT} is invalid and should be rerun."
                )
            else:
                capture_execution_reason = (
                    f"{CAPTURE_EXECUTION_REPORT} has "
                    f"status={capture_execution_report_status}."
                )
            recommendations.append(
                _stage_recommendation(
                    recommendation_id=f"run_{capture_execution_kind}_capture_execution",
                    stage_id="capture_execution",
                    run_root=root,
                    label=(
                        "Run full capture execution"
                        if run_config_targets_real_robot
                        else "Run fake capture execution"
                    ),
                    description=(
                        (
                            "Run supervised real robot plus camera capture with "
                            "process-group teardown evidence."
                        )
                        if run_config_targets_real_robot
                        else (
                            "Run the supervised fake iiwa plus pose receiver path. "
                            "Camera processes stay disabled by default."
                        )
                    ),
                    reason=capture_execution_reason,
                    priority=50,
                    expected_artifacts=[CAPTURE_EXECUTION_REPORT, RAW_ROBOT_EE_POSES],
                    options=capture_execution_options,
                )
            )

    capture_execution_blocks_sync = capture_execution_plan_ready and (
        not capture_execution_report_ready
        )

    if (
        run_config_targets_real_robot
        and capture_execution_report_ready
        and full_capture_gate_status != "ready"
    ):
        blockers = ", ".join(full_capture_gate_next_blockers) or "unknown"
        recommendations.append(
            _stage_recommendation(
                recommendation_id="audit_full_capture_gate",
                stage_id="rewrite_gate",
                run_root=root,
                label="Audit full capture gate",
                description=(
                    "Write rewrite_gate_report.json for the real full-capture "
                    "milestone before treating hardware capture as validated."
                ),
                reason=(
                    f"{FULL_CAPTURE_GATE_ID} is not ready "
                    f"({full_capture_gate_blocker_count} blocker(s): {blockers})."
                ),
                priority=55,
                expected_artifacts=[REWRITE_GATE_REPORT],
                options={"gate": FULL_CAPTURE_GATE_ID, "write": True},
            )
        )

    if rewrite_status_expected and not rewrite_status_ready_for_inspection:
        if rewrite_status_blocker == "missing_rewrite_status_report":
            rewrite_status_reason = (
                "Rewrite-relevant artifacts exist but "
                f"{REWRITE_STATUS_REPORT} is missing."
            )
        elif rewrite_status_blocker == "invalid_rewrite_status_report":
            rewrite_status_reason = (
                f"{REWRITE_STATUS_REPORT} is invalid and should be rebuilt."
            )
        else:
            rewrite_status_reason = (
                f"{REWRITE_STATUS_REPORT} is stale relative to current gate "
                "evidence."
            )
        rewrite_status_options: dict[str, Any] = {"write": True}
        if saved_rewrite_status_gate_run_roots:
            rewrite_status_options["gate_run_root"] = (
                _rewrite_status_gate_run_root_options(
                    saved_rewrite_status_gate_run_roots
                )
            )
        recommendations.append(
            _stage_recommendation(
                recommendation_id="write_rewrite_status",
                stage_id="rewrite_status",
                run_root=root,
                label=(
                    "Write rewrite status"
                    if not has_rewrite_status_report
                    else "Refresh rewrite status"
                ),
                description=(
                    "Summarize all rewrite gates in one status report before "
                    "treating scattered artifacts as progress evidence."
                ),
                reason=rewrite_status_reason,
                priority=58,
                expected_artifacts=[REWRITE_STATUS_REPORT],
                options=rewrite_status_options,
            )
        )
    elif (
        rewrite_status_has_guided_next_action
    ):
        reason_parts = []
        if rewrite_status_first_blocker:
            reason_parts.append(
                f"Next rewrite blocker: {rewrite_status_first_blocker}."
            )
        if rewrite_status_first_blocker_message:
            reason_parts.append(rewrite_status_first_blocker_message)
        if rewrite_status_first_blocker_diagnostic:
            reason_parts.append(rewrite_status_first_blocker_diagnostic)
        if not reason_parts:
            reason_parts.append(
                f"{REWRITE_STATUS_REPORT} is current but blocked."
            )
        for index, action in enumerate(rewrite_status_next_actions, start=1):
            action_label = (
                str(action.get("label"))
                if action.get("label")
                else "Follow rewrite status next action"
            )
            action_reason_parts: list[str] = []
            if action.get("reason"):
                action_reason_parts.append(str(action["reason"]))
            action_reason_parts.extend(reason_parts)
            recommendations.append(
                PipelineRecommendation(
                    id=(
                        "follow_rewrite_status_next_action"
                        if index == 1
                        else f"follow_rewrite_status_next_action_{index}"
                    ),
                    label=action_label,
                    description=(
                        "Run the next action recorded by the current "
                        f"rewrite_status_report.json ({index} of "
                        f"{len(rewrite_status_next_actions)})."
                    ),
                    reason=" ".join(action_reason_parts),
                    priority=8 + index,
                    action_type="command",
                    command=_rewrite_status_action_command(action),
                    blocks_on=[
                        str(blocker)
                        for blocker in action.get("blocks_on", [])
                        if isinstance(blocker, str)
                    ],
                )
            )

    if (
        (has_raw_poses or has_raw_sensor_folders)
        and not has_sync
        and not capture_execution_blocks_sync
    ):
        recommendations.append(
            _stage_recommendation(
                recommendation_id="sync_raw_capture",
                stage_id="sync_run",
                run_root=root,
                label="Run non-destructive sync",
                description=(
                    "Copy matched RGB-D frames into processed/synchronized "
                    "without changing raw capture folders."
                ),
                reason="Raw capture inputs are present but no synchronized sensor folders were found.",
                priority=60,
                expected_artifacts=[f"{PROCESSED_DIR}/{SYNCHRONIZED_DIR}/<sensor>"],
            )
        )

    if has_sync and not sync_quality_ready_for_downstream:
        if sync_quality_blocker == "missing_sync_quality_report":
            sync_quality_reason = (
                f"Synchronized sensors exist but {SYNC_QUALITY_REPORT} is missing."
            )
        elif sync_quality_blocker == "invalid_sync_quality_report":
            sync_quality_reason = (
                f"{SYNC_QUALITY_REPORT} is invalid and should be rewritten."
            )
        else:
            sync_quality_reason = (
                f"{SYNC_QUALITY_REPORT} has overall_status={sync_quality_status}."
            )
        recommendations.append(
            _stage_recommendation(
                recommendation_id="check_sync_quality",
                stage_id="sync_quality",
                run_root=root,
                label="Check sync quality",
                description="Aggregate per-sensor sync reports before ArUco, calibration, or BOP export.",
                reason=sync_quality_reason,
                priority=70,
                expected_artifacts=[SYNC_QUALITY_REPORT],
                endpoint="/sync/quality",
            )
        )

    if sync_quality_ready_for_downstream and not has_aruco:
        recommendations.append(
            _stage_recommendation(
                recommendation_id="run_aruco",
                stage_id="aruco",
                run_root=root,
                label="Run ArUco estimation",
                description="Detect ArUco target poses in synchronized frames.",
                reason=f"{SYNC_QUALITY_REPORT} exists but no synchronized {ARUCO_POSE_ESTIMATION} files were found.",
                priority=80,
                expected_artifacts=[f"{PROCESSED_DIR}/{SYNCHRONIZED_DIR}/<sensor>/{ARUCO_POSE_ESTIMATION}"],
            )
        )

    if (
        calibration_preflight_expected
        and not calibration_preflight_ready_for_calibrated_stages
    ):
        if calibration_preflight_blocker == "missing_calibration_preflight":
            calibration_preflight_reason = (
                f"Calibration profile preflight is expected but "
                f"{CALIBRATION_PREFLIGHT_REPORT} is missing."
            )
        elif calibration_preflight_blocker == "invalid_calibration_preflight":
            calibration_preflight_reason = (
                f"{CALIBRATION_PREFLIGHT_REPORT} is invalid and should be rebuilt."
            )
        else:
            calibration_preflight_reason = (
                f"{CALIBRATION_PREFLIGHT_REPORT} has "
                f"overall_status={calibration_preflight_status}."
            )
        recommendations.append(
            _stage_recommendation(
                recommendation_id="preflight_calibration_profiles",
                stage_id="calibration_preflight",
                run_root=root,
                label="Preflight calibration profiles",
                description=(
                    "Check configured calibration profile coverage, status, "
                    "and quality metrics before calibrated downstream stages."
                ),
                reason=calibration_preflight_reason,
                priority=75,
                expected_artifacts=[CALIBRATION_PREFLIGHT_REPORT],
                endpoint="/calibration/preflight",
            )
        )

    if has_aruco and not aruco_coverage_ready_for_downstream:
        if aruco_coverage_blocker == "missing_aruco_coverage_report":
            aruco_coverage_reason = (
                f"ArUco outputs exist but {ARUCO_COVERAGE_REPORT} is missing."
            )
        elif aruco_coverage_blocker == "invalid_aruco_coverage_report":
            aruco_coverage_reason = (
                f"{ARUCO_COVERAGE_REPORT} is invalid and should be rebuilt."
            )
        else:
            aruco_coverage_reason = (
                f"{ARUCO_COVERAGE_REPORT} has overall_status={aruco_coverage_status}."
            )
        recommendations.append(
            _stage_recommendation(
                recommendation_id="check_aruco_coverage",
                stage_id="aruco_coverage",
                run_root=root,
                label="Check ArUco coverage",
                description=(
                    "Summarize marker detections and valid pose coverage before "
                    "calibration or ArUco result export."
                ),
                reason=aruco_coverage_reason,
                priority=85,
                expected_artifacts=[ARUCO_COVERAGE_REPORT],
            )
        )

    if can_build_calibration_observations and not calibration_observations_ready_for_solver:
        if calibration_observations_blocker == "missing_calibration_observations":
            calibration_observations_reason = (
                "Calibration target pose outputs exist but "
                f"{CALIBRATION_OBSERVATIONS} is missing."
            )
        elif calibration_observations_blocker == "invalid_calibration_observations":
            calibration_observations_reason = (
                f"{CALIBRATION_OBSERVATIONS} is invalid and should be rebuilt."
            )
        else:
            calibration_observations_reason = (
                f"{CALIBRATION_OBSERVATIONS} has "
                f"overall_status={calibration_observations_status}."
            )
        recommendations.append(
            _stage_recommendation(
                recommendation_id="build_calibration_observations",
                stage_id="calibration_observations",
                run_root=root,
                label="Build calibration observations",
                description=(
                    "Extract solver-ready calibration target and robot-pose "
                    "observation pairs."
                ),
                reason=calibration_observations_reason,
                priority=90,
                expected_artifacts=[CALIBRATION_OBSERVATIONS],
                endpoint="/calibration/observations",
            )
        )
    elif (
        calibration_observations_ready_for_solver
        and not calibration_solver_ready_for_candidates
    ):
        if calibration_solver_blocker == "missing_calibration_solver":
            calibration_solver_reason = (
                f"{CALIBRATION_OBSERVATIONS} exists but "
                f"{CALIBRATION_SOLVER_REPORT} is missing."
            )
        elif calibration_solver_blocker == "invalid_calibration_solver":
            calibration_solver_reason = (
                f"{CALIBRATION_SOLVER_REPORT} is invalid and should be rebuilt."
            )
        else:
            calibration_solver_reason = (
                f"{CALIBRATION_SOLVER_REPORT} has "
                f"overall_status={calibration_solver_status}."
            )
        recommendations.append(
            _stage_recommendation(
                recommendation_id="solve_calibration_profiles",
                stage_id="calibration_solver",
                run_root=root,
                label="Solve calibration profiles",
                description=(
                    "Run the calibration solver to produce needs-validation "
                    "profiles from ArUco/robot observations."
                ),
                reason=calibration_solver_reason,
                priority=98,
                expected_artifacts=[CALIBRATION_SOLVER_REPORT],
                endpoint="/calibration/solver",
            )
        )
    elif (
        calibration_solver_ready_for_candidates
        and not calibration_candidates_ready_for_validation
    ):
        if calibration_candidates_blocker == "missing_calibration_candidates":
            calibration_candidates_reason = (
                f"{CALIBRATION_OBSERVATIONS} exists but "
                f"{CALIBRATION_CANDIDATES} is missing."
            )
        elif calibration_candidates_blocker == "invalid_calibration_candidates":
            calibration_candidates_reason = (
                f"{CALIBRATION_CANDIDATES} is invalid and should be rebuilt."
            )
        else:
            calibration_candidates_reason = (
                f"{CALIBRATION_CANDIDATES} has "
                f"overall_status={calibration_candidates_status}."
            )
        recommendations.append(
            _stage_recommendation(
                recommendation_id="build_calibration_candidates",
                stage_id="calibration_candidates",
                run_root=root,
                label="Build calibration candidates",
                description="Generate validation-gated calibration profile candidates.",
                reason=calibration_candidates_reason,
                priority=100,
                expected_artifacts=[CALIBRATION_CANDIDATES],
                endpoint="/calibration/candidates",
            )
        )
    elif (
        calibration_candidates_ready_for_validation
        and not calibration_validation_ready_for_profiles
    ):
        if calibration_validation_blocker == "missing_calibration_validation":
            calibration_validation_reason = (
                f"{CALIBRATION_CANDIDATES} exists but "
                f"{CALIBRATION_VALIDATION_REPORT} is missing."
            )
        elif calibration_validation_blocker == "invalid_calibration_validation":
            calibration_validation_reason = (
                f"{CALIBRATION_VALIDATION_REPORT} is invalid and should be rebuilt."
            )
        else:
            calibration_validation_reason = (
                f"{CALIBRATION_VALIDATION_REPORT} has "
                f"overall_status={calibration_validation_status}."
            )
        recommendations.append(
            _stage_recommendation(
                recommendation_id="validate_calibration_candidates",
                stage_id="calibration_validation",
                run_root=root,
                label="Validate calibration candidates",
                description="Check candidate inliers, residuals, and outlier ratio before promotion.",
                reason=calibration_validation_reason,
                priority=110,
                expected_artifacts=[CALIBRATION_VALIDATION_REPORT],
                endpoint="/calibration/validation",
            )
        )

    if (
        calibration_validation_gate_expected
        and has_calibration_validation
        and calibration_validation_gate_status != "ready"
    ):
        blockers = ", ".join(calibration_validation_gate_next_blockers) or "unknown"
        recommendations.append(
            _stage_recommendation(
                recommendation_id="audit_calibration_validation_gate",
                stage_id="rewrite_gate",
                run_root=root,
                label="Audit calibration validation gate",
                description=(
                    "Write rewrite_gate_report.json for the production "
                    "calibration milestone before treating calibration profiles "
                    "as validated."
                ),
                reason=(
                    f"{CALIBRATION_VALIDATION_GATE_ID} is not ready "
                    f"({calibration_validation_gate_blocker_count} blocker(s): "
                    f"{blockers})."
                ),
                priority=115,
                expected_artifacts=[REWRITE_GATE_REPORT],
                options={"gate": CALIBRATION_VALIDATION_GATE_ID, "write": True},
            )
        )

    if (
        has_sync
        and not has_blenderproc_prepared
        and not calibration_preflight_blocks_calibrated_stages
    ):
        recommendations.append(
            _stage_recommendation(
                recommendation_id="prepare_blenderproc",
                stage_id="blenderproc_prepare",
                run_root=root,
                label="Prepare BlenderProc inputs",
                description="Create per-sensor BlenderProc folders for rendering and estimator setup.",
                reason="Synchronized sensors exist but no blenderproc/objects.json files were found.",
                priority=120,
                expected_artifacts=[f"{PROCESSED_DIR}/{SYNCHRONIZED_DIR}/<sensor>/blenderproc"],
            )
        )
    elif has_blenderproc_prepared and not foundationpose_plan_ready_for_jobs:
        if foundationpose_plan_blocker == "missing_foundationpose_plan":
            foundationpose_plan_reason = (
                f"Prepared BlenderProc inputs exist but {FOUNDATIONPOSE_PLAN} "
                "is missing."
            )
        else:
            foundationpose_plan_reason = (
                f"{FOUNDATIONPOSE_PLAN} is not ready for FoundationPose jobs "
                f"({foundationpose_plan_blocker})."
            )
        recommendations.append(
            _stage_recommendation(
                recommendation_id="plan_foundationpose",
                stage_id="foundationpose",
                run_root=root,
                label="Plan FoundationPose",
                description="Write foundationpose_plan.json without starting Docker.",
                reason=foundationpose_plan_reason,
                priority=130,
                expected_artifacts=[FOUNDATIONPOSE_PLAN],
                options={"dry_run": True},
            )
        )

    if (
        has_sync
        and (
            not bop_export_ready_for_results
            or not bop_object_models_ready_for_result_export
            or not bop_targets_ready_for_evaluation
        )
        and not calibration_preflight_blocks_calibrated_stages
    ):
        if (
            not bop_export_ready_for_results
            and bop_export_blocker == "missing_bop_export_manifest"
        ):
            bop_export_reason = (
                f"Synchronized sensors exist but "
                f"{BOP_DIR}/{BOP_EXPORT_MANIFEST} is missing."
            )
        elif (
            not bop_export_ready_for_results
            and bop_export_blocker == "invalid_bop_export_manifest"
        ):
            bop_export_reason = (
                f"{BOP_DIR}/{BOP_EXPORT_MANIFEST} is invalid and should be rebuilt."
            )
        elif not bop_export_ready_for_results:
            bop_export_reason = (
                f"{BOP_DIR}/{BOP_EXPORT_MANIFEST} has no exported scenes."
            )
        elif bop_object_models_blocker == "missing_bop_object_models":
            bop_export_reason = (
                f"{BOP_DIR}/{BOP_EXPORT_MANIFEST} has no BOP object model "
                "metadata; rebuild the BOP export with model export enabled."
            )
        elif bop_object_models_blocker == "invalid_bop_object_models":
            bop_export_reason = (
                "BOP object model metadata is invalid; rebuild the BOP export "
                "with model export enabled."
            )
        elif bop_targets_blocker == "missing_bop_targets":
            bop_export_reason = (
                f"{BOP_DIR}/{BOP_TARGETS_BOP19} is missing and should be "
                "regenerated before BOP Toolkit evaluation."
            )
        elif bop_targets_blocker == "empty_bop_targets":
            bop_export_reason = (
                f"{BOP_DIR}/{BOP_TARGETS_BOP19} has no target rows and should "
                "be regenerated before BOP Toolkit evaluation."
            )
        else:
            bop_export_reason = (
                f"{BOP_DIR}/{BOP_TARGETS_BOP19} is invalid and should be "
                "regenerated before BOP Toolkit evaluation."
            )
        recommendations.append(
            _stage_recommendation(
                recommendation_id="export_bop_dataset",
                stage_id="bop_export",
                run_root=root,
                label="Export BOP dataset",
                description="Export synchronized frames, camera metadata, models, and targets into BOP layout.",
                reason=bop_export_reason,
                priority=140,
                expected_artifacts=[f"{BOP_DIR}/{BOP_EXPORT_MANIFEST}"],
            )
        )

    if (
        bop_export_ready_for_results
        and bop_object_models_ready_for_result_export
        and not bop_result_export_ready_for_evaluation
    ):
        available_result_sources = []
        if has_foundationpose_outputs:
            available_result_sources.append("foundationpose")
        if has_megapose_outputs:
            available_result_sources.append("megapose")
        if has_sam6d_outputs:
            available_result_sources.append("sam6d")
        if has_aruco and aruco_coverage_ready_for_downstream:
            available_result_sources.append("aruco")
        for source_index, result_source in enumerate(available_result_sources):
            if bop_result_export_blocker == "missing_bop_result_export_manifest":
                bop_result_export_reason = (
                    f"{BOP_DIR}/{BOP_EXPORT_MANIFEST} exists but "
                    f"{BOP_RESULT_EXPORT_MANIFEST} is missing."
                )
            elif bop_result_export_blocker == "invalid_bop_result_export_manifest":
                bop_result_export_reason = (
                    f"{BOP_RESULT_EXPORT_MANIFEST} is invalid and should be rebuilt."
                )
            else:
                bop_result_export_reason = (
                    f"{BOP_RESULT_EXPORT_MANIFEST} has no usable result CSV."
                )
            recommendations.append(
                _stage_recommendation(
                    recommendation_id=f"export_{result_source}_bop_results",
                    stage_id="bop_result_export",
                    run_root=root,
                    label=f"Export {result_source} BOP results",
                    description="Convert available estimator outputs into BOP19 result CSV files.",
                    reason=bop_result_export_reason,
                    priority=150 + source_index,
                    expected_artifacts=[BOP_RESULT_EXPORT_MANIFEST],
                    options={"source": result_source},
                )
            )

    if (
        bop_result_export_ready_for_evaluation
        and bop_targets_ready_for_evaluation
        and not bop_evaluation_report_ready
        and first_result_csv is not None
    ):
        if bop_evaluation_report_blocker == "missing_bop_evaluation_report":
            bop_evaluation_reason = (
                f"{BOP_RESULT_EXPORT_MANIFEST} exists but "
                f"{BOP_EVALUATION_REPORT} is missing."
            )
        elif bop_evaluation_report_blocker == "invalid_bop_evaluation_report":
            bop_evaluation_reason = (
                f"{BOP_EVALUATION_REPORT} is invalid and should be rewritten."
            )
        elif (
            bop_evaluation_report_blocker
            == "failed_bop_evaluation_prerequisites"
        ):
            bop_evaluation_reason = (
                f"{BOP_EVALUATION_REPORT} has "
                f"{bop_evaluation_report_critical_failed_check_count} failed "
                "and "
                f"{bop_evaluation_report_critical_missing_check_count} missing "
                "BOP evaluation prerequisite check(s)."
            )
        else:
            bop_evaluation_reason = (
                f"{BOP_EVALUATION_REPORT} has "
                f"status={bop_evaluation_report_status}."
            )
        recommendations.append(
            _stage_recommendation(
                recommendation_id="evaluate_bop_results",
                stage_id="bop_evaluation",
                run_root=root,
                label="Plan BOP Toolkit evaluation",
                description="Validate the BOP19 CSV and write a dry-run BOP Toolkit evaluation plan/report.",
                reason=bop_evaluation_reason,
                priority=160,
                expected_artifacts=[BOP_EVALUATION_REPORT],
                options={"result_file": first_result_csv.as_posix(), "dry_run": True},
            )
        )

    if (
        foundationpose_runtime_gate_expected
        and has_bop_evaluation
        and foundationpose_runtime_gate_status != "ready"
    ):
        blockers = ", ".join(foundationpose_runtime_gate_next_blockers) or "unknown"
        recommendations.append(
            _stage_recommendation(
                recommendation_id="audit_foundationpose_runtime_gate",
                stage_id="rewrite_gate",
                run_root=root,
                label="Audit FoundationPose runtime gate",
                description=(
                    "Write rewrite_gate_report.json for the real "
                    "FoundationPose-to-BOP scoring milestone before treating "
                    "estimator runtime execution as validated."
                ),
                reason=(
                    f"{FOUNDATIONPOSE_RUNTIME_GATE_ID} is not ready "
                    f"({foundationpose_runtime_gate_blocker_count} blocker(s): "
                    f"{blockers})."
                ),
                priority=165,
                expected_artifacts=[REWRITE_GATE_REPORT],
                options={"gate": FOUNDATIONPOSE_RUNTIME_GATE_ID, "write": True},
            )
        )

    if has_metric_sources and not metric_report_ready_for_dashboard:
        if metric_report_blocker == "missing_metric_report":
            metric_report_reason = (
                "Metric source artifacts exist but "
                f"{RESULTS_DIR}/{METRICS_DIR}/{METRIC_REPORT_JSON} is missing."
            )
        elif metric_report_blocker == "invalid_metric_report":
            metric_report_reason = (
                f"{RESULTS_DIR}/{METRICS_DIR}/{METRIC_REPORT_JSON} is invalid "
                "and should be rebuilt."
            )
        else:
            metric_report_reason = (
                f"{RESULTS_DIR}/{METRICS_DIR}/{METRIC_REPORT_JSON} has no "
                "dashboard rows."
            )
        recommendations.append(
            _stage_recommendation(
                recommendation_id="export_metric_reports",
                stage_id="metric_report_export",
                run_root=root,
                label="Export metric reports",
                description=(
                    "Write JSON, CSV, and XLSX reports from discovered legacy "
                    "accuracy/all_results and BOP Toolkit score artifacts."
                ),
                reason=metric_report_reason,
                priority=170,
                expected_artifacts=[
                    f"{RESULTS_DIR}/{METRICS_DIR}/{METRIC_REPORT_JSON}",
                    f"{RESULTS_DIR}/{METRICS_DIR}/{METRIC_REPORT_CSV}",
                    f"{RESULTS_DIR}/{METRICS_DIR}/{METRIC_REPORT_XLSX}",
                ],
            )
        )

    recommendations = sorted(recommendations, key=lambda item: item.priority)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "run_root": root.as_posix(),
        "facts": {
            "has_run_config": has_run_config,
            "run_config_ready_for_pipeline": run_config_ready_for_pipeline,
            "run_config_blocker": run_config_blocker,
            "run_config_error": run_config_error,
            "run_config_robot_mode": run_config_robot_mode,
            "run_config_targets_real_robot": run_config_targets_real_robot,
            "has_run_preflight": has_run_preflight,
            "run_preflight_status": run_preflight_status,
            "run_preflight_matches_config": run_preflight_matches_config,
            "run_preflight_ready_for_queue": (
                has_run_config and run_preflight_ready_for_queue
            ),
            "run_preflight_queue_blocker": run_preflight_queue_blocker,
            "has_pipeline_sequence_plan": has_pipeline_sequence_plan,
            "pipeline_sequence_plan_ready_for_queue": (
                pipeline_sequence_plan_ready_for_queue
            ),
            "pipeline_sequence_plan_blocker": pipeline_sequence_plan_blocker,
            "pipeline_sequence_plan_step_count": pipeline_sequence_plan_step_count,
            "has_capture_plan": has_capture_plan,
            "capture_plan_ready_for_preflight": capture_plan_ready_for_preflight,
            "capture_plan_blocker": capture_plan_blocker,
            "capture_plan_command_count": capture_plan_command_count,
            "has_capture_plan_preflight": has_capture_preflight,
            "capture_plan_preflight_status": capture_plan_preflight_status,
            "capture_plan_preflight_ready": capture_plan_preflight_ready,
            "capture_plan_preflight_blocker": capture_plan_preflight_blocker,
            "has_capture_execution_plan": has_capture_execution_plan,
            "capture_execution_plan_status": capture_execution_plan_status,
            "capture_execution_plan_ready": capture_execution_plan_ready,
            "capture_execution_plan_blocker": capture_execution_plan_blocker,
            "capture_execution_plan_blocked_checks": (
                capture_execution_plan_blocked_checks
            ),
            "has_capture_execution_report": has_capture_execution_report,
            "capture_execution_report_status": capture_execution_report_status,
            "capture_execution_report_ready": capture_execution_report_ready,
            "capture_execution_report_blocker": capture_execution_report_blocker,
            "capture_execution_blocks_sync": capture_execution_blocks_sync,
            "has_raw_robot_poses": has_raw_poses,
            "has_raw_sensor_folders": has_raw_sensor_folders,
            "synchronized_sensor_count": len(synchronized_sensors),
            "has_sync_quality": has_sync_quality,
            "sync_quality_status": sync_quality_status,
            "sync_quality_ready_for_downstream": sync_quality_ready_for_downstream,
            "sync_quality_blocker": sync_quality_blocker,
            "has_aruco_outputs": has_aruco,
            "calibration_profile_source": calibration_profile_source,
            "has_calibration_profiles_configured": has_calibration_profiles_configured,
            "has_calibration_preflight": has_calibration_preflight,
            "sequence_uses_calibration_preflight": sequence_uses_calibration_preflight,
            "calibration_preflight_expected": calibration_preflight_expected,
            "calibration_preflight_status": calibration_preflight_status,
            "calibration_preflight_ready_for_calibrated_stages": (
                calibration_preflight_ready_for_calibrated_stages
            ),
            "calibration_preflight_blocker": calibration_preflight_blocker,
            "calibration_preflight_blocks_calibrated_stages": (
                calibration_preflight_blocks_calibrated_stages
            ),
            "has_calibration_target_pose_outputs": has_calibration_target_pose_outputs,
            "has_non_aruco_calibration_target_pose_outputs": (
                has_non_aruco_calibration_target_pose_outputs
            ),
            "has_aruco_coverage": has_aruco_coverage,
            "aruco_coverage_status": aruco_coverage_status,
            "aruco_coverage_ready_for_downstream": aruco_coverage_ready_for_downstream,
            "aruco_coverage_blocker": aruco_coverage_blocker,
            "has_calibration_observations": has_calibration_observations,
            "calibration_observations_status": calibration_observations_status,
            "calibration_observations_ready_for_solver": (
                calibration_observations_ready_for_solver
            ),
            "calibration_observations_blocker": calibration_observations_blocker,
            "has_calibration_solver": has_calibration_solver,
            "calibration_solver_status": calibration_solver_status,
            "calibration_solver_ready_for_candidates": (
                calibration_solver_ready_for_candidates
            ),
            "calibration_solver_blocker": calibration_solver_blocker,
            "has_calibration_candidates": has_calibration_candidates,
            "calibration_candidates_status": calibration_candidates_status,
            "calibration_candidates_ready_for_validation": (
                calibration_candidates_ready_for_validation
            ),
            "calibration_candidates_blocker": calibration_candidates_blocker,
            "has_calibration_validation": has_calibration_validation,
            "calibration_validation_status": calibration_validation_status,
            "calibration_validation_ready_for_profiles": (
                calibration_validation_ready_for_profiles
            ),
            "calibration_validation_blocker": calibration_validation_blocker,
            "calibration_validation_gate_expected": (
                calibration_validation_gate_expected
            ),
            "calibration_validation_gate_status": (
                calibration_validation_gate_status
            ),
            "calibration_validation_gate_blocker_count": (
                calibration_validation_gate_blocker_count
            ),
            "calibration_validation_gate_next_blockers": (
                calibration_validation_gate_next_blockers
            ),
            "has_blenderproc_prepared": has_blenderproc_prepared,
            "has_bop_export": has_bop_export,
            "bop_export_ready_for_results": bop_export_ready_for_results,
            "bop_export_blocker": bop_export_blocker,
            "bop_export_count": bop_export_count,
            "bop_object_models_ready_for_result_export": (
                bop_object_models_ready_for_result_export
            ),
            "bop_object_models_blocker": bop_object_models_blocker,
            "bop_object_model_count": bop_object_model_count,
            "has_bop_targets": has_bop_targets,
            "bop_targets_ready_for_evaluation": (
                bop_targets_ready_for_evaluation
            ),
            "bop_targets_blocker": bop_targets_blocker,
            "bop_targets_count": bop_targets_count,
            "has_foundationpose_plan": has_foundationpose_plan,
            "foundationpose_plan_ready_for_jobs": foundationpose_plan_ready_for_jobs,
            "foundationpose_plan_blocker": foundationpose_plan_blocker,
            "foundationpose_plan_job_count": foundationpose_plan_job_count,
            "has_foundationpose_outputs": has_foundationpose_outputs,
            "has_megapose_outputs": has_megapose_outputs,
            "has_sam6d_outputs": has_sam6d_outputs,
            "has_bop_result_export": has_bop_result_export,
            "bop_result_export_ready_for_evaluation": (
                bop_result_export_ready_for_evaluation
            ),
            "bop_result_export_blocker": bop_result_export_blocker,
            "bop_result_export_result_count": bop_result_export_result_count,
            "bop_result_export_usable_result_count": (
                bop_result_export_usable_result_count
            ),
            "has_bop_evaluation": has_bop_evaluation,
            "bop_evaluation_report_status": bop_evaluation_report_status,
            "bop_evaluation_report_ready": bop_evaluation_report_ready,
            "bop_evaluation_report_blocker": bop_evaluation_report_blocker,
            "bop_evaluation_report_critical_failed_check_count": (
                bop_evaluation_report_critical_failed_check_count
            ),
            "bop_evaluation_report_critical_missing_check_count": (
                bop_evaluation_report_critical_missing_check_count
            ),
            "bop_evaluation_report_score_metric_count": (
                bop_evaluation_report_score_metric_count
            ),
            "has_bop_score_metrics": has_bop_score_metrics,
            "has_legacy_metrics": has_legacy_metrics,
            "has_metric_sources": has_metric_sources,
            "has_metric_report": has_metric_report,
            "metric_report_ready_for_dashboard": metric_report_ready_for_dashboard,
            "metric_report_blocker": metric_report_blocker,
            "metric_report_row_count": metric_report_row_count,
            "full_capture_gate_status": full_capture_gate_status,
            "full_capture_gate_blocker_count": full_capture_gate_blocker_count,
            "full_capture_gate_next_blockers": full_capture_gate_next_blockers,
            "has_rewrite_status_report": has_rewrite_status_report,
            "rewrite_status_expected": rewrite_status_expected,
            "rewrite_status_gate_run_roots": (
                rewrite_status_report.get("gate_run_roots")
            ),
            "rewrite_status_overall_status": rewrite_status_overall_status,
            "rewrite_status_ready_for_inspection": rewrite_status_ready_for_inspection,
            "rewrite_status_blocker": rewrite_status_blocker,
            "rewrite_status_ready_gate_count": rewrite_status_ready_gate_count,
            "rewrite_status_gate_count": rewrite_status_gate_count,
            "rewrite_status_ready_check_count": rewrite_status_ready_check_count,
            "rewrite_status_check_count": rewrite_status_check_count,
            "rewrite_status_next_blockers": rewrite_status_next_blockers,
            "rewrite_status_next_action_label": rewrite_status_next_action_label,
            "rewrite_status_next_action_command": rewrite_status_next_action_command,
            "rewrite_status_next_action_labels": rewrite_status_next_action_labels,
            "rewrite_status_next_action_commands": rewrite_status_next_action_commands,
            "rewrite_status_first_blocker": rewrite_status_first_blocker,
            "rewrite_status_first_blocker_message": (
                rewrite_status_first_blocker_message
            ),
            "rewrite_status_first_blocker_diagnostic": (
                rewrite_status_first_blocker_diagnostic
            ),
            "foundationpose_runtime_gate_expected": (
                foundationpose_runtime_gate_expected
            ),
            "foundationpose_runtime_gate_status": (
                foundationpose_runtime_gate_status
            ),
            "foundationpose_runtime_gate_blocker_count": (
                foundationpose_runtime_gate_blocker_count
            ),
            "foundationpose_runtime_gate_next_blockers": (
                foundationpose_runtime_gate_next_blockers
            ),
            "first_result_csv": first_result_csv.as_posix() if first_result_csv else None,
        },
        "recommendation_count": len(recommendations),
        "recommendations": [recommendation.to_dict() for recommendation in recommendations],
    }
