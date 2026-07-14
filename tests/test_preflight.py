from __future__ import annotations

from pathlib import Path

from posetestbot.pipeline.preflight import (
    STAGE_RUNTIME_REQUIREMENTS,
    build_run_preflight,
    run_preflight_queue_summary,
    write_run_preflight_report,
)
from posetestbot.pipeline.run_config import create_run_config, write_run_config


def fake_robot_status() -> dict:
    return {
        "schema_version": "robot_status.v2",
        "selected_profile": {"mode": "real"},
    }


def fake_sensor_status() -> dict:
    return {
        "schema_version": "sensor_status.v1",
        "total_connected": 0,
        "all_expected_connected": False,
    }


def runtime_status(*, blenderproc_available: bool, zed_available: bool = False) -> dict:
    runtimes = [
        {
            "runtime_id": "blenderproc",
            "display_name": "BlenderProc",
            "category": "renderer",
            "required_for": "BlenderProc rendering",
            "available": blenderproc_available,
            "checks": [],
            "hint": None,
        },
        {
            "runtime_id": "zed_sdk_python",
            "display_name": "Stereolabs ZED SDK Python",
            "category": "camera_sdk",
            "required_for": "ZED capture",
            "available": zed_available,
            "checks": [],
            "hint": None,
        },
    ]
    available_count = sum(1 for item in runtimes if item["available"])
    return {
        "schema_version": "runtime_status.v1",
        "runtime_count": len(runtimes),
        "available_count": available_count,
        "all_available": available_count == len(runtimes),
        "runtimes": runtimes,
    }


def write_config(
    run_root: Path,
    *,
    plan_only: bool = False,
    options: dict | None = None,
    selected_objects: list[str] | None = None,
) -> None:
    config = create_run_config(
        run_root=run_root,
        sequence_id="sync_to_bop_dry_run",
        sequence_options=options or {},
        plan_only=plan_only,
        selected_objects=selected_objects,
    )
    write_run_config(run_root, config)


def test_runtime_requirements_are_acquisition_only() -> None:
    assert STAGE_RUNTIME_REQUIREMENTS == {"blenderproc_render": ("blenderproc",)}


def test_preflight_warns_for_optional_missing_runtime_without_required_stage(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    write_config(run_root, plan_only=False)

    report = build_run_preflight(
        run_root,
        include_sensor_status=False,
        collect_robot=fake_robot_status,
        collect_sensors=fake_sensor_status,
        collect_runtimes=lambda: runtime_status(blenderproc_available=False),
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert checks["runtime_status"]["status"] == "warning"
    assert checks["runtime_status"]["details"]["missing_required_runtime_ids"] == []
    assert checks["runtime_requirements"]["status"] == "ok"
    assert report["overall_status"] == "warning"


def test_preflight_errors_when_non_dry_run_blenderproc_is_missing(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_config(
        run_root,
        plan_only=False,
        options={"blenderproc_render": {"dry_run": False}},
    )

    report = build_run_preflight(
        run_root,
        include_sensor_status=False,
        collect_robot=fake_robot_status,
        collect_sensors=fake_sensor_status,
        collect_runtimes=lambda: runtime_status(blenderproc_available=False),
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert checks["runtime_status"]["status"] == "error"
    assert checks["runtime_status"]["details"]["missing_required_runtime_ids"] == [
        "blenderproc"
    ]
    assert checks["runtime_requirements"]["status"] == "error"
    assert checks["runtime_requirements"]["details"]["missing_runtime_ids"] == [
        "blenderproc"
    ]
    assert report["overall_status"] == "error"


def test_preflight_does_not_require_blenderproc_for_objectless_render(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    write_config(
        run_root,
        plan_only=False,
        options={"blenderproc_render": {"dry_run": False}},
        selected_objects=[],
    )

    report = build_run_preflight(
        run_root,
        include_sensor_status=False,
        collect_robot=fake_robot_status,
        collect_sensors=fake_sensor_status,
        collect_runtimes=lambda: runtime_status(blenderproc_available=False),
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert checks["runtime_status"]["status"] == "warning"
    assert checks["runtime_status"]["details"]["missing_required_runtime_ids"] == []
    assert checks["runtime_requirements"]["status"] == "ok"
    assert checks["runtime_requirements"]["details"]["requirement_count"] == 0
    assert report["overall_status"] == "warning"


def test_run_preflight_queue_summary_tracks_missing_ready_and_stale(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    config = create_run_config(run_root=run_root).to_dict()
    write_run_config(run_root, create_run_config(run_root=run_root))

    missing = run_preflight_queue_summary(run_root, config)
    assert missing["queue_blocker"] == "missing_preflight"

    report = {
        "schema_version": "run_preflight.v1",
        "overall_status": "warning",
        "config": config,
    }
    write_run_preflight_report(run_root, report)
    ready = run_preflight_queue_summary(run_root, config)
    assert ready["ready_for_queue"] is True
    assert ready["queue_blocker"] is None

    stale_config = dict(config)
    stale_config["object_folder"] = "other_models"
    stale = run_preflight_queue_summary(run_root, stale_config)
    assert stale["queue_blocker"] == "stale_preflight"
