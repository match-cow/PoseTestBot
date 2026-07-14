"""Run-level preflight summaries before queueing PoseTestBot workflows."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from posetestbot.io.atomic import atomic_write_json
from posetestbot.io.artifacts import CALIBRATION_PROFILES, RUN_PREFLIGHT_REPORT
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    upsert_stage,
    write_run_manifest,
)
from posetestbot.pipeline.run_config import (
    load_run_config_for_run_root,
    sequence_plan_from_run_config,
)
from posetestbot.robot.status import collect_robot_status
from posetestbot.runtime.status import collect_runtime_status
from posetestbot.sensors.status import collect_sensor_status

SCHEMA_VERSION = "run_preflight.v1"
STAGE_RUNTIME_REQUIREMENTS = {
    "blenderproc_render": ("blenderproc",),
}


def _check(
    name: str,
    status: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "message": message,
        "details": dict(details or {}),
    }


def _overall_status(checks: list[Mapping[str, Any]]) -> str:
    statuses = {str(check.get("status")) for check in checks}
    if "error" in statuses:
        return "error"
    if "warning" in statuses:
        return "warning"
    return "ok"


def load_run_preflight_report(run_root: str | Path) -> dict[str, Any] | None:
    path = Path(run_root) / RUN_PREFLIGHT_REPORT
    if not path.is_file():
        return None
    with open(path, "r") as f:
        report = json.load(f)
    if not isinstance(report, dict):
        raise ValueError(f"{RUN_PREFLIGHT_REPORT} must contain a JSON object")
    return report


def preflight_config_matches(
    report: Mapping[str, Any],
    config: Mapping[str, Any],
) -> bool:
    return report.get("config") == config


def run_preflight_queue_summary(
    run_root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize whether a saved preflight snapshot allows sequence queueing."""

    path = Path(run_root) / RUN_PREFLIGHT_REPORT
    try:
        report = load_run_preflight_report(run_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "path": path.as_posix(),
            "exists": path.exists(),
            "overall_status": None,
            "matches_config": None,
            "ready_for_queue": False,
            "queue_blocker": "invalid_preflight",
            "error": str(exc),
        }
    if report is None:
        return {
            "path": path.as_posix(),
            "exists": False,
            "overall_status": None,
            "matches_config": None,
            "ready_for_queue": False,
            "queue_blocker": "missing_preflight",
        }

    matches_config = preflight_config_matches(report, config)
    overall_status = report.get("overall_status")
    if not isinstance(overall_status, str):
        overall_status = None
    if overall_status == "error":
        queue_blocker = "failed_preflight"
    elif not matches_config:
        queue_blocker = "stale_preflight"
    else:
        queue_blocker = None
    return {
        "path": path.as_posix(),
        "exists": True,
        "overall_status": overall_status,
        "matches_config": matches_config,
        "ready_for_queue": queue_blocker is None,
        "queue_blocker": queue_blocker,
    }


def _sensor_counts(config: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sensor in config["capture"]["sensors"]:
        sensor_type = str(sensor["sensor_type"])
        counts[sensor_type] = counts.get(sensor_type, 0) + 1
    return counts


def _non_dry_run_steps(plan) -> list[str]:
    return [
        step.id
        for step in plan.steps
        if step.options.get("dry_run") is False
    ]


def _runtime_lookup(runtimes: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    lookup: dict[str, Mapping[str, Any]] = {}
    for runtime in runtimes.get("runtimes", []):
        if not isinstance(runtime, Mapping):
            continue
        runtime_id = runtime.get("runtime_id")
        if isinstance(runtime_id, str):
            lookup[runtime_id] = runtime
    return lookup


def _runtime_requirements(plan, runtimes: Mapping[str, Any]) -> list[dict[str, Any]]:
    runtime_by_id = _runtime_lookup(runtimes)
    requirements: list[dict[str, Any]] = []
    for step in plan.steps:
        if step.options.get("dry_run") is not False:
            continue
        if step.stage_id == "blenderproc_render" and step.options.get("objectless") is True:
            # Objectless rendering writes a successful skipped plan without
            # invoking BlenderProc, so its executable is not a requirement.
            continue
        for runtime_id in STAGE_RUNTIME_REQUIREMENTS.get(step.stage_id, ()):
            runtime = runtime_by_id.get(runtime_id, {})
            requirements.append(
                {
                    "step_id": step.id,
                    "stage_id": step.stage_id,
                    "runtime_id": runtime_id,
                    "available": bool(runtime.get("available", False)),
                    "display_name": runtime.get("display_name", runtime_id),
                    "category": runtime.get("category"),
                    "required_for": runtime.get("required_for"),
                    "hint": runtime.get("hint"),
                }
            )
    return requirements


def _resolve_run_path(run_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else run_root / path


def _calibration_profile_inputs(
    *,
    config: Mapping[str, Any],
    plan,
    run_root: Path,
) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    configured = config.get("calibration_profiles")
    if isinstance(configured, str) and configured.strip():
        inputs.append(
            {"source": "run_config.calibration_profiles", "path": configured}
        )

    has_calibration_preflight = False
    for step in plan.steps:
        if step.stage_id == "calibration_preflight":
            has_calibration_preflight = True
        calibration_profiles = step.options.get("calibration_profiles")
        if isinstance(calibration_profiles, str) and calibration_profiles.strip():
            inputs.append(
                {
                    "source": f"{step.id}.calibration_profiles",
                    "path": calibration_profiles,
                }
            )

    if has_calibration_preflight and not any(
        item["source"] == "run_config.calibration_profiles" for item in inputs
    ):
        inputs.append(
            {
                "source": "calibration_preflight.default",
                "path": CALIBRATION_PROFILES,
            }
        )

    seen: set[tuple[str, str]] = set()
    resolved_inputs: list[dict[str, Any]] = []
    for item in inputs:
        key = (str(item["source"]), str(item["path"]))
        if key in seen:
            continue
        seen.add(key)
        resolved_path = _resolve_run_path(run_root, str(item["path"]))
        resolved_inputs.append(
            {
                "source": item["source"],
                "path": item["path"],
                "resolved_path": resolved_path.as_posix(),
                "exists": resolved_path.is_file(),
            }
        )
    return resolved_inputs


def build_run_preflight(
    run_root: str | Path,
    *,
    include_sensor_status: bool = True,
    include_runtime_status: bool = True,
    collect_robot: Callable[[], dict] = collect_robot_status,
    collect_sensors: Callable[[], dict] = collect_sensor_status,
    collect_runtimes: Callable[[], dict] = collect_runtime_status,
) -> dict[str, Any]:
    """Build a run-readiness summary without launching pipeline stages."""

    run_root_path = Path(run_root)
    config = load_run_config_for_run_root(run_root_path)
    plan = sequence_plan_from_run_config(config)
    robot = collect_robot()
    sensors = collect_sensors() if include_sensor_status else None
    runtimes = collect_runtimes() if include_runtime_status else None
    plan_only = bool(config["pipeline"].get("plan_only", True))
    non_dry_run_steps = _non_dry_run_steps(plan)

    checks = [
        _check(
            "run_root",
            "ok" if run_root_path.exists() else "warning",
            (
                f"Run root exists: {run_root_path}"
                if run_root_path.exists()
                else f"Run root will be created or is not present yet: {run_root_path}"
            ),
            details={"run_root": run_root_path.as_posix()},
        ),
        _check(
            "run_config",
            "ok",
            f"Loaded {config['schema_version']} for {len(config['capture']['sensors'])} sensor(s).",
            details={
                "robot_profile": "real",
                "sensor_counts": _sensor_counts(config),
            },
        ),
        _check(
            "sequence_plan",
            "ok",
            f"Built sequence {plan.sequence_id} with {len(plan.steps)} step(s).",
            details={
                "plan_only": plan.plan_only,
                "resources": plan.resources,
                "steps": [step.id for step in plan.steps],
                "non_dry_run_steps": non_dry_run_steps,
            },
        ),
        _check(
            "robot_profile",
            "ok" if robot["selected_profile"]["mode"] == "real" else "error",
            (
                "Run config and runtime status use the real robot profile."
                if robot["selected_profile"]["mode"] == "real"
                else "Runtime robot status did not select the real profile."
            ),
            details={
                "configured": "real",
                "selected": robot["selected_profile"]["mode"],
            },
        ),
    ]

    calibration_inputs = _calibration_profile_inputs(
        config=config,
        plan=plan,
        run_root=run_root_path,
    )
    if calibration_inputs:
        missing_calibration_inputs = [
            item for item in calibration_inputs if not item["exists"]
        ]
        if missing_calibration_inputs:
            calibration_status = "warning" if plan_only else "error"
            message = (
                f"{len(missing_calibration_inputs)} of "
                f"{len(calibration_inputs)} calibration profile input(s) "
                "are missing."
            )
            if plan_only:
                message += " Plan-only inspection can still proceed."
        else:
            calibration_status = "ok"
            message = (
                f"All {len(calibration_inputs)} calibration profile input(s) "
                "are present."
            )
        checks.append(
            _check(
                "calibration_profile_inputs",
                calibration_status,
                message,
                details={
                    "plan_only": plan_only,
                    "input_count": len(calibration_inputs),
                    "missing_count": len(missing_calibration_inputs),
                    "inputs": calibration_inputs,
                },
            )
        )

    if sensors is not None:
        expected_counts_requested = bool(sensors.get("expected_counts_requested")) or any(
            isinstance(family, Mapping) and family.get("expected_count") is not None
            for family in sensors.get("families", [])
        )
        checks.append(
            _check(
                "sensor_status",
                "ok"
                if sensors["all_expected_connected"] or not expected_counts_requested
                else "warning",
                (
                    f"Detected {sensors['total_connected']} connected sensor(s)."
                    if not expected_counts_requested
                    else (
                        f"Connected {sensors['total_connected']} sensor(s); requested sensor counts are satisfied."
                    )
                    if sensors["all_expected_connected"]
                    else (
                        f"Connected {sensors['total_connected']} sensor(s); "
                        "one or more requested sensor counts are missing or unchecked."
                    )
                ),
                details={
                    "total_connected": sensors["total_connected"],
                    "all_expected_connected": sensors["all_expected_connected"],
                    "expected_counts_requested": expected_counts_requested,
                },
            )
        )

    if runtimes is not None:
        requirements = _runtime_requirements(plan, runtimes)
        missing_requirements = [
            requirement
            for requirement in requirements
            if not requirement["available"]
        ]
        runtime_status = "ok" if runtimes["all_available"] else "warning"
        if missing_requirements and not plan_only:
            runtime_status = "error"
        checks.append(
            _check(
                "runtime_status",
                runtime_status,
                (
                    f"{runtimes['available_count']} of {runtimes['runtime_count']} external runtime(s) available."
                    + (
                        " Only selected non-dry-run stage requirements can block queueing."
                        if plan_only and runtime_status == "warning"
                        else ""
                    )
                ),
                details={
                    "available_count": runtimes["available_count"],
                    "runtime_count": runtimes["runtime_count"],
                    "all_available": runtimes["all_available"],
                    "plan_only": plan_only,
                    "missing_required_runtime_ids": sorted(
                        {
                            str(requirement["runtime_id"])
                            for requirement in missing_requirements
                        }
                    ),
                },
            )
        )
        if requirements:
            if missing_requirements:
                requirements_status = "warning" if plan_only else "error"
                message = (
                    f"{len(missing_requirements)} of {len(requirements)} "
                    "non-dry-run runtime requirement(s) are unavailable."
                )
                if plan_only:
                    message += " Plan-only inspection can still proceed."
            else:
                requirements_status = "ok"
                message = (
                    f"All {len(requirements)} non-dry-run runtime requirement(s) "
                    "are available."
                )
        else:
            requirements_status = "ok"
            message = "No non-dry-run external runtime requirements in the sequence."
        checks.append(
            _check(
                "runtime_requirements",
                requirements_status,
                message,
                details={
                    "plan_only": plan_only,
                    "requirement_count": len(requirements),
                    "missing_count": len(missing_requirements),
                    "requirements": requirements,
                    "missing_runtime_ids": sorted(
                        {
                            str(requirement["runtime_id"])
                            for requirement in missing_requirements
                        }
                    ),
                },
            )
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_root": run_root_path.as_posix(),
        "overall_status": _overall_status(checks),
        "checks": checks,
        "config": config,
        "sequence_plan": plan.to_dict(),
        "robot_status": robot,
        "sensor_status": sensors,
        "runtime_status": runtimes,
    }


def write_run_preflight_report(
    run_root: str | Path,
    report: Mapping[str, Any],
    *,
    filename: str = RUN_PREFLIGHT_REPORT,
) -> Path:
    path = Path(run_root) / filename
    return atomic_write_json(path, dict(report))


def write_run_preflight_with_manifest(
    run_root: str | Path,
    *,
    include_sensor_status: bool = True,
    include_runtime_status: bool = True,
    collect_robot: Callable[[], dict] = collect_robot_status,
    collect_sensors: Callable[[], dict] = collect_sensor_status,
    collect_runtimes: Callable[[], dict] = collect_runtime_status,
) -> tuple[Path, dict[str, Any]]:
    run_root_path = Path(run_root)
    manifest = load_or_create_run_manifest(run_root_path)
    upsert_stage(manifest, name="run_preflight", status="running")
    write_run_manifest(manifest, run_root_path)
    try:
        report = build_run_preflight(
            run_root_path,
            include_sensor_status=include_sensor_status,
            include_runtime_status=include_runtime_status,
            collect_robot=collect_robot,
            collect_sensors=collect_sensors,
            collect_runtimes=collect_runtimes,
        )
        path = write_run_preflight_report(run_root_path, report)
        manifest["robot_profile"] = dict(report["config"].get("robot_profile") or {})
        manifest["capture_config"] = dict(report["config"].get("capture") or {})
        upsert_stage(
            manifest,
            name="run_preflight",
            status="succeeded" if report["overall_status"] != "error" else "failed",
            artifacts={RUN_PREFLIGHT_REPORT: path},
            run_root=run_root_path,
            message=f"Run preflight status: {report['overall_status']}.",
        )
        write_run_manifest(manifest, run_root_path)
    except Exception as exc:
        upsert_stage(
            manifest,
            name="run_preflight",
            status="failed",
            message=str(exc),
        )
        write_run_manifest(manifest, run_root_path)
        raise
    return path, report
