from __future__ import annotations

import json
from pathlib import Path

from posetestbot.io.artifacts import DATASET_MANIFEST, RUN_PREFLIGHT_REPORT
from posetestbot.pipeline.preflight import (
    build_run_preflight,
    run_preflight_queue_summary,
    write_run_preflight_with_manifest,
)
from posetestbot.pipeline.run_config import create_run_config, write_run_config


def fake_robot_status(mode: str = "fake") -> dict:
    return {
        "schema_version": "robot_status.v1",
        "generated_at": "2026-06-16T00:00:00+00:00",
        "selected_profile": {"mode": mode},
        "profiles": {},
        "fake_first": mode == "fake",
        "real_robot": {},
        "env_overrides": {},
        "command_protocols": ["legacy", "robot_command.v1"],
        "default_command_protocol": "legacy",
        "notes": [],
    }


def fake_sensor_status(ok: bool = True) -> dict:
    return {
        "schema_version": "sensor_status.v1",
        "generated_at": "2026-06-16T00:00:00+00:00",
        "families": [],
        "total_connected": 5 if ok else 3,
        "all_expected_connected": ok,
    }


def fake_runtime_status(ok: bool = True) -> dict:
    runtime_ids = (
        "blenderproc",
        "foundationpose",
        "megapose",
        "sam6d",
        "bop_toolkit",
        "zed_sdk_python",
    )
    runtimes = [
        {
            "runtime_id": runtime_id,
            "display_name": runtime_id,
            "category": "test",
            "required_for": f"{runtime_id} tests",
            "available": ok,
            "hint": None if ok else f"Install {runtime_id}.",
            "checks": [],
        }
        for runtime_id in runtime_ids
    ]
    return {
        "schema_version": "runtime_status.v1",
        "generated_at": "2026-06-16T00:00:00+00:00",
        "available_count": len(runtimes) if ok else 0,
        "runtime_count": len(runtimes),
        "all_available": ok,
        "runtimes": runtimes,
    }


def test_run_preflight_reports_ok_for_matching_dry_run_config(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, config)

    preflight = build_run_preflight(
        run_root,
        collect_robot=lambda: fake_robot_status("fake"),
        collect_sensors=lambda: fake_sensor_status(True),
        collect_runtimes=lambda: fake_runtime_status(True),
    )

    assert preflight["schema_version"] == "run_preflight.v1"
    assert preflight["overall_status"] == "ok"
    assert preflight["sequence_plan"]["sequence_id"] == "sync_aruco"
    assert preflight["config"]["robot_profile"]["mode"] == "fake"
    checks = {check["name"]: check for check in preflight["checks"]}
    assert checks["robot_mode"]["status"] == "ok"
    assert checks["sensor_status"]["status"] == "ok"
    assert checks["runtime_requirements"]["status"] == "ok"
    assert checks["runtime_requirements"]["details"]["requirement_count"] == 0


def test_run_preflight_writer_records_report_and_manifest_stage(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-write"
    config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, config)

    path, preflight = write_run_preflight_with_manifest(
        run_root,
        collect_robot=lambda: fake_robot_status("fake"),
        collect_sensors=lambda: fake_sensor_status(True),
        collect_runtimes=lambda: fake_runtime_status(True),
    )

    assert path == run_root / RUN_PREFLIGHT_REPORT
    assert path.is_file()
    assert json.loads(path.read_text())["overall_status"] == "ok"
    assert preflight["overall_status"] == "ok"
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(stage for stage in manifest["stages"] if stage["name"] == "run_preflight")
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][RUN_PREFLIGHT_REPORT] == RUN_PREFLIGHT_REPORT


def test_run_preflight_queue_summary_reports_missing_snapshot(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-missing-preflight"
    config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, config)

    summary = run_preflight_queue_summary(run_root, config.to_dict())

    assert summary == {
        "path": (run_root / RUN_PREFLIGHT_REPORT).as_posix(),
        "exists": False,
        "overall_status": None,
        "matches_config": None,
        "ready_for_queue": False,
        "queue_blocker": "missing_preflight",
    }


def test_run_preflight_queue_summary_reports_invalid_snapshot(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-invalid-preflight"
    config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, config)
    (run_root / RUN_PREFLIGHT_REPORT).write_text("[]\n")

    summary = run_preflight_queue_summary(run_root, config.to_dict())

    assert summary["path"] == (run_root / RUN_PREFLIGHT_REPORT).as_posix()
    assert summary["exists"] is True
    assert summary["overall_status"] is None
    assert summary["matches_config"] is None
    assert summary["ready_for_queue"] is False
    assert summary["queue_blocker"] == "invalid_preflight"
    assert RUN_PREFLIGHT_REPORT in summary["error"]


def test_run_preflight_queue_summary_reports_ready_warning_snapshot(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-warning-preflight"
    config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, config)
    (run_root / RUN_PREFLIGHT_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "run_preflight.v1",
                "overall_status": "warning",
                "config": config.to_dict(),
            }
        )
        + "\n"
    )

    summary = run_preflight_queue_summary(run_root, config.to_dict())

    assert summary["exists"] is True
    assert summary["overall_status"] == "warning"
    assert summary["matches_config"] is True
    assert summary["ready_for_queue"] is True
    assert summary["queue_blocker"] is None


def test_run_preflight_queue_summary_reports_failed_or_stale_snapshot(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-bad-preflight"
    original_config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, original_config)
    (run_root / RUN_PREFLIGHT_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "run_preflight.v1",
                "overall_status": "error",
                "config": original_config.to_dict(),
            }
        )
        + "\n"
    )

    failed = run_preflight_queue_summary(run_root, original_config.to_dict())

    assert failed["matches_config"] is True
    assert failed["ready_for_queue"] is False
    assert failed["queue_blocker"] == "failed_preflight"

    updated_config = create_run_config(
        run_root=run_root,
        sequence_id="sync_to_bop_dry_run",
    )
    write_run_config(run_root, updated_config)
    (run_root / RUN_PREFLIGHT_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "run_preflight.v1",
                "overall_status": "ok",
                "config": original_config.to_dict(),
            }
        )
        + "\n"
    )

    stale = run_preflight_queue_summary(run_root, updated_config.to_dict())

    assert stale["matches_config"] is False
    assert stale["ready_for_queue"] is False
    assert stale["queue_blocker"] == "stale_preflight"


def test_run_preflight_warns_for_mismatch_and_dry_run_runtime_gap(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, config)

    preflight = build_run_preflight(
        run_root,
        collect_robot=lambda: fake_robot_status("real"),
        collect_sensors=lambda: fake_sensor_status(False),
        collect_runtimes=lambda: fake_runtime_status(False),
    )

    assert preflight["overall_status"] == "warning"
    checks = {check["name"]: check for check in preflight["checks"]}
    assert checks["robot_mode"]["status"] == "warning"
    assert checks["sensor_status"]["status"] == "warning"
    assert checks["runtime_status"]["status"] == "warning"
    assert checks["runtime_requirements"]["status"] == "ok"


def test_run_preflight_reports_non_dry_run_runtime_steps(tmp_path: Path) -> None:
    run_root = tmp_path / "run-runtime"
    config = create_run_config(
        run_root=run_root,
        sequence_id="foundationpose_runtime_to_bop_eval",
    )
    write_run_config(run_root, config)

    preflight = build_run_preflight(
        run_root,
        collect_robot=lambda: fake_robot_status("fake"),
        collect_sensors=lambda: fake_sensor_status(True),
        collect_runtimes=lambda: fake_runtime_status(True),
    )

    checks = {check["name"]: check for check in preflight["checks"]}
    assert checks["sequence_plan"]["details"]["non_dry_run_steps"] == [
        "foundationpose",
        "bop_evaluation",
    ]
    assert checks["runtime_requirements"]["status"] == "ok"
    assert checks["runtime_requirements"]["details"]["missing_count"] == 0
    assert checks["runtime_requirements"]["details"]["requirements"] == [
        {
            "step_id": "foundationpose",
            "stage_id": "foundationpose",
            "runtime_id": "foundationpose",
            "available": True,
            "display_name": "foundationpose",
            "category": "test",
            "required_for": "foundationpose tests",
            "hint": None,
        },
        {
            "step_id": "bop_evaluation",
            "stage_id": "bop_evaluation",
            "runtime_id": "bop_toolkit",
            "available": True,
            "display_name": "bop_toolkit",
            "category": "test",
            "required_for": "bop_toolkit tests",
            "hint": None,
        },
    ]


def test_run_preflight_warns_for_missing_plan_only_calibration_profiles(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-calibrated"
    config = create_run_config(
        run_root=run_root,
        sequence_id="sync_to_bop_calibrated_dry_run",
        calibration_profiles="profiles/missing_calibration.json",
        plan_only=True,
    )
    write_run_config(run_root, config)

    preflight = build_run_preflight(
        run_root,
        collect_robot=lambda: fake_robot_status("fake"),
        collect_sensors=lambda: fake_sensor_status(True),
        collect_runtimes=lambda: fake_runtime_status(True),
    )

    assert preflight["overall_status"] == "warning"
    checks = {check["name"]: check for check in preflight["checks"]}
    calibration_check = checks["calibration_profile_inputs"]
    assert calibration_check["status"] == "warning"
    assert calibration_check["details"]["missing_count"] == 3
    assert {
        item["source"] for item in calibration_check["details"]["inputs"]
    } == {
        "run_config.calibration_profiles",
        "blenderproc_prepare.calibration_profiles",
        "bop_export.calibration_profiles",
    }
    assert all(
        item["resolved_path"].endswith("profiles/missing_calibration.json")
        for item in calibration_check["details"]["inputs"]
    )


def test_run_preflight_errors_for_missing_execution_calibration_profiles(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-calibrated"
    config = create_run_config(
        run_root=run_root,
        sequence_id="sync_to_bop_calibrated_dry_run",
        calibration_profiles="profiles/missing_calibration.json",
        plan_only=False,
    )
    write_run_config(run_root, config)

    preflight = build_run_preflight(
        run_root,
        collect_robot=lambda: fake_robot_status("fake"),
        collect_sensors=lambda: fake_sensor_status(True),
        collect_runtimes=lambda: fake_runtime_status(True),
    )

    assert preflight["overall_status"] == "error"
    checks = {check["name"]: check for check in preflight["checks"]}
    assert checks["calibration_profile_inputs"]["status"] == "error"


def test_run_preflight_warns_for_missing_plan_only_runtime_requirements(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-runtime"
    config = create_run_config(
        run_root=run_root,
        sequence_id="foundationpose_runtime_to_bop_eval",
        plan_only=True,
    )
    write_run_config(run_root, config)

    preflight = build_run_preflight(
        run_root,
        collect_robot=lambda: fake_robot_status("fake"),
        collect_sensors=lambda: fake_sensor_status(True),
        collect_runtimes=lambda: fake_runtime_status(False),
    )

    assert preflight["overall_status"] == "warning"
    checks = {check["name"]: check for check in preflight["checks"]}
    assert checks["runtime_requirements"]["status"] == "warning"
    assert checks["runtime_requirements"]["details"]["missing_runtime_ids"] == [
        "bop_toolkit",
        "foundationpose",
    ]


def test_run_preflight_errors_when_runtime_missing_for_execution(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    config = create_run_config(
        run_root=run_root,
        sequence_id="sync_aruco",
        plan_only=False,
    )
    write_run_config(run_root, config)

    preflight = build_run_preflight(
        run_root,
        collect_robot=lambda: fake_robot_status("fake"),
        collect_sensors=lambda: fake_sensor_status(True),
        collect_runtimes=lambda: fake_runtime_status(False),
    )

    assert preflight["overall_status"] == "error"
    checks = {check["name"]: check for check in preflight["checks"]}
    assert checks["runtime_status"]["status"] == "error"


def test_run_preflight_errors_for_missing_execution_runtime_requirements(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-runtime"
    config = create_run_config(
        run_root=run_root,
        sequence_id="foundationpose_runtime_to_bop_eval",
        plan_only=False,
    )
    write_run_config(run_root, config)

    preflight = build_run_preflight(
        run_root,
        collect_robot=lambda: fake_robot_status("fake"),
        collect_sensors=lambda: fake_sensor_status(True),
        collect_runtimes=lambda: fake_runtime_status(False),
    )

    assert preflight["overall_status"] == "error"
    checks = {check["name"]: check for check in preflight["checks"]}
    assert checks["runtime_requirements"]["status"] == "error"
    assert checks["runtime_requirements"]["details"]["missing_count"] == 2
