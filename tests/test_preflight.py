from __future__ import annotations

from pathlib import Path

import pytest

from posetestbot.pipeline.preflight import (
    STAGE_RUNTIME_REQUIREMENTS,
    _calibration_arrangement_check,
    build_run_preflight,
    run_preflight_queue_summary,
    write_run_preflight_report,
)
from posetestbot.pipeline.run_config import create_run_config, write_run_config
from posetestbot.pipeline.run_config import SensorRunConfig
from posetestbot.robot.reference_frames import POSE_TEMPLATE_BASE_SUNRISE_PATH


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
    dataset_mode: str = "objectless",
) -> None:
    config = create_run_config(
        run_root=run_root,
        sequence_id="sync_to_bop_dry_run",
        sequence_options=options or {},
        plan_only=plan_only,
        dataset_mode=dataset_mode,
    )
    write_run_config(run_root, config)


def test_runtime_requirements_are_acquisition_only() -> None:
    assert STAGE_RUNTIME_REQUIREMENTS == {"blenderproc_render": ("blenderproc",)}


def test_run_preflight_counts_only_enabled_sensors(tmp_path: Path) -> None:
    run_root = tmp_path / "two-enabled-one-disabled"
    write_run_config(
        run_root,
        create_run_config(
            run_root=run_root,
            sensors=(
                SensorRunConfig("realsense_d435", "1", "First"),
                SensorRunConfig("realsense_d435", "2", "Second"),
                SensorRunConfig("realsense_d435", "3", "Offline", enabled=False),
            ),
            sequence_id="sync_to_bop_dry_run",
        ),
    )

    report = build_run_preflight(
        run_root,
        include_sensor_status=False,
        include_runtime_status=False,
        collect_robot=fake_robot_status,
    )

    check = next(item for item in report["checks"] if item["name"] == "run_config")
    assert check["details"]["configured_sensor_count"] == 3
    assert check["details"]["enabled_sensor_count"] == 2
    assert check["details"]["sensor_counts"] == {"realsense_d435": 2}


def test_calibration_arrangement_requires_one_mounting_group_and_matching_target_frame(
    tmp_path: Path,
) -> None:
    static_config = create_run_config(
        run_root=tmp_path / "static-calibration",
        sensors=tuple(
            SensorRunConfig(
                "realsense_d435",
                str(index),
                f"Static {index}",
                mounting_mode="static",
            )
            for index in range(1, 4)
        ),
        robot_pose_sunrise_reference_frame_path=(POSE_TEMPLATE_BASE_SUNRISE_PATH),
    ).to_dict()

    ready = _calibration_arrangement_check(
        static_config,
        {"placement_mode": "unknown", "mounting_frame": "robot_flange"},
    )
    wrong_target = _calibration_arrangement_check(
        static_config,
        {"placement_mode": "unknown", "mounting_frame": "template_base"},
    )
    legacy_target = _calibration_arrangement_check(
        static_config,
        {"placement_mode": "unknown", "mounting_frame": None},
    )
    wrong_reference_config = create_run_config(
        run_root=tmp_path / "wrong-reference",
        sensors=tuple(
            SensorRunConfig(
                "realsense_d435",
                str(index),
                f"Static {index}",
                mounting_mode="static",
            )
            for index in range(1, 4)
        ),
        robot_pose_sunrise_reference_frame_path="/PoseTestBot/TemplateBase",
    ).to_dict()
    wrong_reference = _calibration_arrangement_check(
        wrong_reference_config,
        {"placement_mode": "unknown", "mounting_frame": "robot_flange"},
    )
    mixed_config = create_run_config(
        run_root=tmp_path / "mixed-calibration",
        sensors=(
            SensorRunConfig(
                "realsense_d435",
                "static",
                "Static",
                mounting_mode="static",
            ),
            SensorRunConfig(
                "realsense_d435",
                "wrist",
                "Wrist",
                mounting_mode="eye_in_hand",
            ),
        ),
    ).to_dict()
    mixed = _calibration_arrangement_check(
        mixed_config,
        {"placement_mode": "unknown", "mounting_frame": "robot_flange"},
    )

    assert ready["status"] == "ok"
    assert ready["details"] == {
        "calibration_mode": "eye_to_hand",
        "camera_mounting_mode": "static",
        "target_mounting_frame": "robot_flange",
        "target_transform_state": "estimated",
        "placement_mode": "unknown",
    }
    assert wrong_target["status"] == "error"
    assert "requires the target mounted in robot_flange" in wrong_target["message"]
    assert legacy_target["status"] == "error"
    assert "predates explicit physical mounting" in legacy_target["message"]
    assert legacy_target["details"]["recorded_target_mounting_frame"] is None
    assert wrong_reference["status"] == "error"
    assert POSE_TEMPLATE_BASE_SUNRISE_PATH in wrong_reference["message"]
    assert wrong_reference["details"] == {
        "calibration_mode": "eye_to_hand",
        "camera_mounting_mode": "static",
        "required_sunrise_reference_frame_path": (POSE_TEMPLATE_BASE_SUNRISE_PATH),
        "configured_sunrise_reference_frame_path": "/PoseTestBot/TemplateBase",
    }
    assert mixed["status"] == "error"
    assert "separate runs" in mixed["message"]


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


def test_annotation_free_bop_sequence_rejects_blenderproc_render_options(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    with pytest.raises(
        ValueError,
        match="Unknown pipeline sequence option group.*blenderproc_render",
    ):
        write_config(
            run_root,
            plan_only=False,
            options={"blenderproc_render": {"dry_run": False}},
            dataset_mode="pose_template",
        )


def test_preflight_does_not_require_blenderproc_for_annotation_free_bop_export(
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
    assert checks["runtime_requirements"]["details"]["requirement_count"] == 0
    assert checks["sequence_plan"]["details"]["steps"] == [
        "sync_run",
        "sync_quality",
        "bop_export",
    ]
    assert report["overall_status"] == "warning"


def test_preflight_blocks_pose_template_gt_without_confirmed_selection(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "template-run"
    config = create_run_config(
        run_root=run_root,
        sequence_id="sync_to_bop_dry_run",
        dataset_mode="pose_template",
        plan_only=True,
    )
    write_run_config(run_root, config)

    report = build_run_preflight(
        run_root,
        include_sensor_status=False,
        collect_robot=fake_robot_status,
        collect_sensors=fake_sensor_status,
        collect_runtimes=lambda: runtime_status(blenderproc_available=False),
    )

    check = next(
        item for item in report["checks"] if item["name"] == "pose_template_selection"
    )
    assert check["status"] == "error"
    assert "dataset export requires a valid confirmed pose template" in check["message"]


def test_run_preflight_queue_summary_tracks_missing_ready_and_stale(
    tmp_path: Path,
) -> None:
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
    stale_config["dataset_mode"] = "pose_template"
    stale = run_preflight_queue_summary(run_root, stale_config)
    assert stale["queue_blocker"] == "stale_preflight"
