from __future__ import annotations

import json
import subprocess
from pathlib import Path

from posetestbot.io.artifacts import (
    BOP_DIR,
    BOP_EVALUATION_REPORT,
    BOP_EXPORT_MANIFEST,
    BOP_RESULT_EXPORT_MANIFEST,
    CALIBRATION_PROFILES,
    CALIBRATION_VALIDATION_REPORT,
    CAPTURE_EXECUTION_PLAN,
    CAPTURE_EXECUTION_REPORT,
    CAPTURE_PLAN,
    CAPTURE_PLAN_PREFLIGHT_REPORT,
    DEPTH_DIR,
    FRAME_METADATA_JSONL,
    FOUNDATIONPOSE_PLAN,
    HARDWARE_STATUS_REPORT,
    METRIC_REPORT_JSON,
    METRICS_DIR,
    PIPELINE_SEQUENCE_PLAN,
    RESULTS_DIR,
    REWRITE_GATE_REPORT,
    REWRITE_STATUS_REPORT,
    RGB_DIR,
    RUN_CONFIG,
    RUN_PREFLIGHT_REPORT,
    SYNTHETIC_RGBD_REPORT,
    SYNC_QUALITY_REPORT,
)
from posetestbot.pipeline.rewrite_gate import (
    CALIBRATION_VALIDATION_GATE_ID,
    FOUNDATIONPOSE_RUNTIME_GATE_ID,
    build_calibration_validation_gate_report,
    build_fake_end_to_end_gate_report,
    build_foundationpose_runtime_gate_report,
    build_full_capture_gate_report,
    build_rewrite_status_report,
    default_full_capture_run_root,
    write_fake_end_to_end_gate_report,
    write_rewrite_status_report,
)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def populate_ready_fake_gate(run_root: Path) -> None:
    write_json(
        run_root / RUN_CONFIG,
        {
            "schema_version": "run_config.v1",
            "robot": {"mode": "fake"},
            "pipeline": {"sequence_id": "fake_capture_execution"},
        },
    )
    write_json(run_root / RUN_PREFLIGHT_REPORT, {"overall_status": "warning"})
    write_json(
        run_root / CAPTURE_EXECUTION_REPORT,
        {
            "schema_version": "capture_execution_report.v1",
            "status": "succeeded",
            "selected_roles": ["robot_controller", "robot_pose_receiver"],
            "raw_pose_count": 2,
        },
    )
    write_json(
        run_root / SYNTHETIC_RGBD_REPORT,
        {
            "schema_version": "synthetic_rgbd_report.v1",
            "status": "succeeded",
            "frame_count": 2,
        },
    )
    write_json(run_root / SYNC_QUALITY_REPORT, {"overall_status": "ok"})
    write_json(
        run_root / BOP_DIR / BOP_EXPORT_MANIFEST,
        {
            "schema_version": "bop_export_manifest.v1",
            "exports": [{"sensor_name": "realsense_123", "scene_id": 1}],
        },
    )
    write_json(
        run_root / BOP_RESULT_EXPORT_MANIFEST,
        {
            "schema_version": "bop_result_export_manifest.v1",
            "results": [{"filename": "foundationpose_bop-test.csv", "row_count": 1}],
        },
    )
    write_json(
        run_root / BOP_EVALUATION_REPORT,
        {
            "schema_version": "bop_evaluation_report.v1",
            "status": "planned",
            "dry_run": True,
            "result": {"filename": "foundationpose_bop-test.csv"},
        },
    )
    write_json(
        run_root / RESULTS_DIR / METRICS_DIR / METRIC_REPORT_JSON,
        {"dashboard": {"methods": []}, "rows": [{"row_type": "bop_toolkit_score"}]},
    )


def populate_ready_full_capture_gate(run_root: Path) -> None:
    write_json(
        run_root / RUN_CONFIG,
        {
            "schema_version": "run_config.v1",
            "robot_profile": {"mode": "real"},
            "capture": {
                "sensors": [
                    {
                        "sensor_type": "realsense_d435",
                        "device_id": "123",
                        "enabled": True,
                    }
                ]
            },
        },
    )
    write_json(run_root / RUN_PREFLIGHT_REPORT, {"overall_status": "warning"})
    write_json(
        run_root / HARDWARE_STATUS_REPORT,
        {
            "overall_status": "warning",
            "robot_status": {"selected_profile": {"mode": "real"}},
        },
    )
    write_json(
        run_root / CAPTURE_PLAN,
        {
            "schema_version": "capture_plan.v1",
            "commands": [
                {
                    "role": "sensor_capture",
                    "name": "realsense_123",
                    "command": ["uv", "run", "python", "scripts/capture.py"],
                },
                {
                    "role": "robot_pose_receiver",
                    "name": "pose_receiver_udp_json",
                    "command": ["uv", "run", "python", "scripts/pose_receiver.py"],
                },
            ],
        },
    )
    write_json(
        run_root / CAPTURE_PLAN_PREFLIGHT_REPORT,
        {"overall_status": "warning"},
    )
    write_json(
        run_root / CAPTURE_EXECUTION_PLAN,
        {
            "schema_version": "capture_execution_plan.v1",
            "status": "ok",
            "ready_to_execute": True,
            "selected_roles": [
                "sensor_capture",
                "robot_pose_receiver",
            ],
        },
    )
    write_json(
        run_root / CAPTURE_EXECUTION_REPORT,
        {
            "schema_version": "capture_execution_report.v1",
            "status": "succeeded",
            "mode": "full",
            "allow_cameras": True,
            "raw_pose_count": 5,
            "capture_execution_plan": {
                "selected_roles": [
                    "robot_controller",
                    "sensor_capture",
                    "robot_pose_receiver",
                ]
            },
            "processes": [
                {"role": "robot_controller", "status": "succeeded"},
                {
                    "role": "sensor_capture",
                    "status": "stopped",
                    "started_at": "2026-06-19T00:00:00+00:00",
                    "ended_at": "2026-06-19T00:00:01+00:00",
                },
                {"role": "robot_pose_receiver", "status": "succeeded"},
            ],
        },
    )
    sensor_folder = run_root / "realsense_123"
    (sensor_folder / RGB_DIR).mkdir(parents=True)
    (sensor_folder / DEPTH_DIR).mkdir()
    (sensor_folder / RGB_DIR / "000001.png").write_bytes(b"rgb")
    (sensor_folder / DEPTH_DIR / "000001.png").write_bytes(b"depth")
    (sensor_folder / FRAME_METADATA_JSONL).write_text(
        json.dumps({"frame_id": "000001.png"}) + "\n"
    )


def populate_blocked_full_capture_with_sensor_diagnostics(run_root: Path) -> None:
    populate_ready_full_capture_gate(run_root)
    diagnostic_families = [
        {
            "sensor_type": "realsense_d435",
            "display_name": "Intel RealSense D435",
            "diagnostics": [
                {
                    "code": "discovery_error",
                    "severity": "error",
                    "message": "RealSense discovery failed.",
                    "hints": ["Check USB/udev access."],
                }
            ],
        },
        {
            "sensor_type": "oak_d_pro",
            "display_name": "Luxonis OAK-D Pro",
            "diagnostics": [
                {
                    "code": "expected_count_not_met",
                    "severity": "warning",
                    "message": "Connected 0 of expected 1 OAK-D Pro device(s).",
                }
            ],
        },
        {
            "sensor_type": "zed_2i",
            "display_name": "Stereolabs ZED 2i",
            "diagnostics": [
                {
                    "code": "sdk_unavailable",
                    "severity": "warning",
                    "message": "Python SDK module 'pyzed.sl' is not importable.",
                }
            ],
        },
    ]
    write_json(
        run_root / HARDWARE_STATUS_REPORT,
        {
            "overall_status": "error",
            "robot_status": {"selected_profile": {"mode": "real"}},
            "checks": [
                {
                    "name": "sensor:realsense_d435",
                    "status": "error",
                    "message": "Intel RealSense D435 discovery failed.",
                    "details": {
                        "error": "RuntimeError: could not initialize udev monitor"
                    },
                }
            ],
            "sensor_status": {"families": diagnostic_families},
        },
    )
    write_json(
        run_root / CAPTURE_PLAN_PREFLIGHT_REPORT,
        {
            "overall_status": "error",
            "checks": [
                    {
                        "name": "sensor:realsense_d435:123",
                        "status": "error",
                        "message": "Discovery error for realsense_d435.",
                        "details": {
                            "diagnostics": diagnostic_families[0]["diagnostics"]
                        },
                    }
                ],
                "sensor_status": {"families": diagnostic_families},
            },
        )


def populate_ready_foundationpose_runtime_gate(run_root: Path) -> None:
    output_folder = (
        run_root
        / "processed"
        / "synchronized"
        / "realsense_123"
        / "foundationpose_est5_track2_obj0_output"
    )
    ob_in_cam = output_folder / "ob_in_cam"
    ob_in_cam.mkdir(parents=True)
    (ob_in_cam / "000000.txt").write_text(
        "1 0 0 0\n0 1 0 0\n0 0 1 1\n0 0 0 1\n"
    )
    write_json(
        run_root / FOUNDATIONPOSE_PLAN,
        {
            "schema_version": "foundationpose_plan.v1",
            "dry_run": False,
            "jobs": [
                {
                    "sensor_name": "realsense_123",
                    "expected_output_folder": output_folder.as_posix(),
                }
            ],
        },
    )
    result_path = run_root / RESULTS_DIR / BOP_DIR / "foundationpose_bop-test.csv"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        "scene_id,im_id,obj_id,score,R,t,time\n"
        "1,0,1,1,1 0 0 0 1 0 0 0 1,0 0 1000,-1\n"
    )
    write_json(
        run_root / BOP_RESULT_EXPORT_MANIFEST,
        {
            "schema_version": "bop_result_export_manifest.v1",
            "source_type": "foundationpose",
            "results": [
                {
                    "filename": result_path.name,
                    "path": result_path.as_posix(),
                    "row_count": 1,
                    "source_outputs": [output_folder.as_posix()],
                }
            ],
        },
    )
    write_json(
        run_root / BOP_EVALUATION_REPORT,
        {
            "schema_version": "bop_evaluation_report.v1",
            "status": "succeeded",
            "dry_run": False,
            "result": {"filename": result_path.name, "method": "foundationpose"},
            "score_summary": {
                "score_file_count": 1,
                "metrics": {"bop19_average_recall": 0.75},
            },
        },
    )


def promoted_calibration_profile(*, status: str = "valid") -> dict:
    return {
        "schema_version": "calibration.v1",
        "profile_id": "realsense_123_static_valid",
        "sensor_id": "123",
        "sensor_type": "realsense_d435",
        "mounting_mode": "static",
        "rig_position": "static",
        "intrinsics": {
            "cam_K": [1.0, 0.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 1.0],
            "width": 1280,
            "height": 720,
            "distortion": [],
            "depth_scale_to_mm": 1.0,
        },
        "extrinsics": {
            "from": "camera",
            "to": "robot_base",
            "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "translation_mm": [1.0, 2.0, 3.0],
        },
        "target_type": "aruco_grid",
        "calibration_dataset_id": "calibration-run-1",
        "method": "fixture_validation",
        "status": status,
        "quality": {
            "num_observations": 8,
            "num_inliers": 7,
            "mean_reprojection_error_px": 0.4,
            "residual_translation_mm": 1.2,
            "residual_rotation_deg": 0.4,
        },
        "operator": "operator",
        "calibrated_at": "2026-06-19T00:00:00+00:00",
        "sync_delta_ms": 2.5,
        "metadata": {
            "validated_from_status": "needs_validation",
            "validation_schema_version": "calibration_validation.v1",
        },
    }


def populate_ready_calibration_validation_gate(run_root: Path) -> None:
    write_json(
        run_root / CALIBRATION_VALIDATION_REPORT,
        {
            "schema_version": "calibration_validation.v1",
            "overall_status": "ok",
            "profile_count": 1,
            "promotable_profile_count": 1,
            "promotion": {
                "requested": True,
                "promoted": True,
                "path": (run_root / CALIBRATION_PROFILES).as_posix(),
                "profile_count": 1,
            },
            "profiles": [
                {
                    "profile_id": "realsense_123_static_valid",
                    "sensor_id": "123",
                    "validation_status": "ok",
                    "promotable": True,
                }
            ],
        },
    )
    write_json(
        run_root / CALIBRATION_PROFILES,
        {
            "schema_version": "calibration.v1",
            "profiles": [promoted_calibration_profile()],
        },
    )


def test_fake_end_to_end_gate_reports_missing_blockers(tmp_path: Path) -> None:
    run_root = tmp_path / "run"

    report = build_fake_end_to_end_gate_report(run_root)

    assert report["overall_status"] == "blocked"
    assert report["summary"] == {
        "ready_count": 0,
        "blocked_count": 9,
        "check_count": 9,
    }
    assert [blocker["name"] for blocker in report["next_blockers"]] == [
        "run_config",
        "run_preflight",
        "capture_execution",
        "synthetic_rgbd_fixture",
        "sync_quality",
        "bop_export",
        "bop_result_export",
        "bop_evaluation",
        "metric_report",
    ]


def test_fake_end_to_end_gate_requires_execution_not_plan_only(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    populate_ready_fake_gate(run_root)
    write_json(
        run_root / CAPTURE_EXECUTION_REPORT,
        {
            "schema_version": "capture_execution_report.v1",
            "status": "planned",
            "selected_roles": ["robot_pose_receiver"],
            "raw_pose_count": 0,
        },
    )

    report = build_fake_end_to_end_gate_report(run_root)

    capture_check = next(
        check for check in report["checks"] if check["name"] == "capture_execution"
    )
    assert report["overall_status"] == "blocked"
    assert capture_check["status"] == "blocked"
    assert capture_check["details"]["raw_pose_count"] == 0


def test_fake_end_to_end_gate_accepts_roles_from_real_capture_report_shape(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    populate_ready_fake_gate(run_root)
    write_json(
        run_root / CAPTURE_EXECUTION_REPORT,
        {
            "schema_version": "capture_execution_report.v1",
            "status": "succeeded",
            "raw_pose_count": 3,
            "capture_execution_plan": {
                "selected_roles": ["robot_controller", "robot_pose_receiver"]
            },
            "processes": [
                {"role": "robot_controller"},
                {"role": "robot_pose_receiver"},
            ],
        },
    )

    report = build_fake_end_to_end_gate_report(run_root)

    assert report["overall_status"] == "ready"


def test_fake_end_to_end_gate_ready_and_writable(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    populate_ready_fake_gate(run_root)

    path, report = write_fake_end_to_end_gate_report(run_root)

    assert path == run_root / REWRITE_GATE_REPORT
    assert report["overall_status"] == "ready"
    written = json.loads(path.read_text())
    assert written["summary"]["ready_count"] == 9
    assert written["next_blockers"] == []


def test_rewrite_gate_cli_returns_nonzero_for_blocked_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = tmp_path / "blocked"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_rewrite_gate.py",
            run_root.as_posix(),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "rewrite_fake_end_to_end.v1: blocked" in result.stdout


def test_fake_e2e_smoke_cli_help_lists_gate_command() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_rewrite_fake_e2e_smoke.py",
            "--help",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "rewrite_gate_report.json" in result.stdout


def test_fake_e2e_smoke_cli_refuses_nonempty_run_without_overwrite(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = tmp_path / "existing"
    run_root.mkdir()
    (run_root / "keep.txt").write_text("do not remove")

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_rewrite_fake_e2e_smoke.py",
            run_root.as_posix(),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "Pass --overwrite" in result.stderr
    assert (run_root / "keep.txt").read_text() == "do not remove"


def test_full_capture_gate_rejects_fake_end_to_end_evidence(tmp_path: Path) -> None:
    run_root = tmp_path / "fake-run"
    populate_ready_fake_gate(run_root)

    report = build_full_capture_gate_report(run_root)

    assert report["overall_status"] == "blocked"
    blockers = {blocker["name"] for blocker in report["next_blockers"]}
    assert "run_config" in blockers
    assert "capture_execution" in blockers


def test_full_capture_gate_ready_with_full_capture_and_sensor_frames(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "real-run"
    populate_ready_full_capture_gate(run_root)

    report = build_full_capture_gate_report(run_root)

    assert report["overall_status"] == "ready"
    assert report["summary"] == {
        "ready_count": 8,
        "blocked_count": 0,
        "check_count": 8,
    }


def test_full_capture_gate_blocks_missing_sensor_frames(tmp_path: Path) -> None:
    run_root = tmp_path / "real-run-missing-frames"
    populate_ready_full_capture_gate(run_root)
    (run_root / "realsense_123" / FRAME_METADATA_JSONL).unlink()

    report = build_full_capture_gate_report(run_root)

    assert report["overall_status"] == "blocked"
    frame_check = next(
        check for check in report["checks"] if check["name"] == "sensor_frames:realsense_123"
    )
    assert frame_check["status"] == "blocked"
    assert frame_check["details"]["has_frame_metadata"] is False


def test_full_capture_gate_blocks_mismatched_rgb_depth_counts(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "real-run-mismatched-frames"
    populate_ready_full_capture_gate(run_root)
    (run_root / "realsense_123" / RGB_DIR / "000002.png").write_bytes(b"rgb")

    report = build_full_capture_gate_report(run_root)

    assert report["overall_status"] == "blocked"
    frame_check = next(
        check for check in report["checks"] if check["name"] == "sensor_frames:realsense_123"
    )
    assert frame_check["status"] == "blocked"
    assert frame_check["details"]["rgb_count"] == 2
    assert frame_check["details"]["depth_count"] == 1
    assert frame_check["details"]["frame_count_match"] is False


def test_full_capture_gate_explains_blocked_execution_plan_gate(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "real-run-blocked-execution-plan"
    populate_ready_full_capture_gate(run_root)
    write_json(
        run_root / CAPTURE_EXECUTION_PLAN,
        {
            "schema_version": "capture_execution_plan.v1",
            "status": "error",
            "ready_to_execute": False,
            "selected_roles": [
                "sensor_capture",
                "robot_pose_receiver",
            ],
            "gates": [
                {
                    "name": "capture_plan_preflight",
                    "status": "error",
                    "message": "Capture-plan preflight status is error.",
                }
            ],
        },
    )

    report = build_full_capture_gate_report(run_root)

    execution_plan_check = next(
        check
        for check in report["checks"]
        if check["name"] == "capture_execution_plan"
    )
    assert execution_plan_check["status"] == "blocked"
    assert execution_plan_check["message"] == (
        "capture_execution_plan.json is blocked by capture_plan_preflight: "
        "Capture-plan preflight status is error."
    )
    assert execution_plan_check["details"]["error_checks"] == [
        {
            "name": "capture_plan_preflight",
            "status": "error",
            "message": "Capture-plan preflight status is error.",
            "details": {},
        }
    ]


def test_full_capture_gate_blocks_fake_selected_robot_profile(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "real-run-fake-selected-profile"
    populate_ready_full_capture_gate(run_root)
    write_json(
        run_root / HARDWARE_STATUS_REPORT,
        {
            "overall_status": "warning",
            "robot_status": {"selected_profile": {"mode": "fake"}},
        },
    )

    report = build_full_capture_gate_report(run_root)

    assert report["overall_status"] == "blocked"
    hardware_check = next(
        check for check in report["checks"] if check["name"] == "hardware_status"
    )
    assert hardware_check["status"] == "blocked"
    assert hardware_check["details"]["selected_robot_mode"] == "fake"
    assert hardware_check["details"]["robot_mode_ok"] is False


def test_full_capture_gate_includes_prerequisite_diagnostics(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "real-run-diagnostics"
    populate_blocked_full_capture_with_sensor_diagnostics(run_root)

    report = build_full_capture_gate_report(run_root)

    hardware_check = next(
        check for check in report["checks"] if check["name"] == "hardware_status"
    )
    assert hardware_check["status"] == "blocked"
    assert hardware_check["details"]["error_checks"][0]["name"] == (
        "sensor:realsense_d435"
    )
    assert hardware_check["details"]["sensor_diagnostics"][0]["code"] == (
        "discovery_error"
    )
    preflight_check = next(
        check for check in report["checks"] if check["name"] == "capture_plan_preflight"
    )
    assert preflight_check["details"]["error_checks"][0]["name"] == (
        "sensor:realsense_d435:123"
    )
    assert report["next_blockers"][0]["details"]["sensor_diagnostics"][0][
        "sensor_type"
    ] == "realsense_d435"
    write_json(
        run_root / PIPELINE_SEQUENCE_PLAN,
        {
            "schema_version": "pipeline_sequence_plan.v1",
            "sequence_id": "real_full_capture_validation",
        },
    )
    status_report = build_rewrite_status_report(
        tmp_path / "status",
        gate_ids=("rewrite_full_capture.v1",),
        gate_run_roots={"rewrite_full_capture.v1": run_root},
    )
    assert status_report["next_blockers"][0]["details"]["sensor_diagnostics"][0][
        "code"
    ] == "discovery_error"
    assert status_report["next_actions"][0]["blocks_on"] == [
        "sensor:realsense_d435",
        "sensor:oak_d_pro",
        "sensor:zed_2i",
    ]


def test_rewrite_gate_cli_accepts_full_capture_gate(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = tmp_path / "real-run-cli"
    populate_ready_full_capture_gate(run_root)

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_rewrite_gate.py",
            run_root.as_posix(),
            "--gate",
            "rewrite_full_capture.v1",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "rewrite_full_capture.v1: ready (8/8 ready)" in result.stdout


def test_rewrite_gate_cli_prints_blocker_diagnostics(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = tmp_path / "real-run-cli-diagnostics"
    populate_blocked_full_capture_with_sensor_diagnostics(run_root)

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_rewrite_gate.py",
            run_root.as_posix(),
            "--gate",
            "rewrite_full_capture.v1",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "blocked check: sensor:realsense_d435" in result.stdout
    assert "diagnostic: RealSense discovery failed." in result.stdout
    assert "hint: Check USB/udev access." in result.stdout


def test_foundationpose_runtime_gate_ready_with_outputs_results_and_scores(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "foundationpose-runtime-ready"
    populate_ready_foundationpose_runtime_gate(run_root)

    report = build_foundationpose_runtime_gate_report(run_root)

    assert report["overall_status"] == "ready"
    assert report["gate_id"] == FOUNDATIONPOSE_RUNTIME_GATE_ID
    assert report["summary"] == {
        "ready_count": 3,
        "blocked_count": 0,
        "check_count": 3,
    }


def test_foundationpose_runtime_gate_blocks_dry_run_plan(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "foundationpose-runtime-dry-run"
    populate_ready_foundationpose_runtime_gate(run_root)
    plan = json.loads((run_root / FOUNDATIONPOSE_PLAN).read_text())
    plan["dry_run"] = True
    write_json(run_root / FOUNDATIONPOSE_PLAN, plan)

    report = build_foundationpose_runtime_gate_report(run_root)

    assert report["overall_status"] == "blocked"
    runtime_check = next(
        check for check in report["checks"] if check["name"] == "foundationpose_runtime"
    )
    assert runtime_check["status"] == "blocked"
    assert runtime_check["details"]["dry_run"] is True


def test_foundationpose_runtime_gate_blocks_dry_run_bop_evaluation(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "foundationpose-runtime-dry-run-eval"
    populate_ready_foundationpose_runtime_gate(run_root)
    evaluation = json.loads((run_root / BOP_EVALUATION_REPORT).read_text())
    evaluation["dry_run"] = True
    write_json(run_root / BOP_EVALUATION_REPORT, evaluation)

    report = build_foundationpose_runtime_gate_report(run_root)

    assert report["overall_status"] == "blocked"
    evaluation_check = next(
        check
        for check in report["checks"]
        if check["name"] == "foundationpose_bop_evaluation"
    )
    assert evaluation_check["status"] == "blocked"
    assert evaluation_check["details"]["dry_run"] is True


def test_rewrite_gate_cli_accepts_foundationpose_runtime_gate(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = tmp_path / "foundationpose-runtime-cli"
    populate_ready_foundationpose_runtime_gate(run_root)

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_rewrite_gate.py",
            run_root.as_posix(),
            "--gate",
            FOUNDATIONPOSE_RUNTIME_GATE_ID,
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )

    assert f"{FOUNDATIONPOSE_RUNTIME_GATE_ID}: ready (3/3 ready)" in result.stdout


def test_calibration_validation_gate_ready_with_promoted_valid_profiles(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "calibration-validation-ready"
    populate_ready_calibration_validation_gate(run_root)

    report = build_calibration_validation_gate_report(run_root)

    assert report["overall_status"] == "ready"
    assert report["gate_id"] == CALIBRATION_VALIDATION_GATE_ID
    assert report["summary"] == {
        "ready_count": 2,
        "blocked_count": 0,
        "check_count": 2,
    }


def test_calibration_validation_gate_blocks_unpromoted_validation(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "calibration-validation-unpromoted"
    populate_ready_calibration_validation_gate(run_root)
    validation = json.loads((run_root / CALIBRATION_VALIDATION_REPORT).read_text())
    validation["promotion"]["promoted"] = False
    validation["promotion"]["profile_count"] = 0
    write_json(run_root / CALIBRATION_VALIDATION_REPORT, validation)

    report = build_calibration_validation_gate_report(run_root)

    assert report["overall_status"] == "blocked"
    validation_check = next(
        check for check in report["checks"] if check["name"] == "calibration_validation"
    )
    assert validation_check["status"] == "blocked"
    assert validation_check["details"]["promotion_promoted"] is False


def test_calibration_validation_gate_blocks_needs_validation_profile(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "calibration-validation-needs-validation"
    populate_ready_calibration_validation_gate(run_root)
    write_json(
        run_root / CALIBRATION_PROFILES,
        {
            "schema_version": "calibration.v1",
            "profiles": [
                promoted_calibration_profile(status="needs_validation"),
            ],
        },
    )

    report = build_calibration_validation_gate_report(run_root)

    assert report["overall_status"] == "blocked"
    profiles_check = next(
        check for check in report["checks"] if check["name"] == "calibration_profiles"
    )
    assert profiles_check["status"] == "blocked"
    assert profiles_check["details"]["profiles"][0]["status"] == "needs_validation"


def test_rewrite_gate_cli_accepts_calibration_validation_gate(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = tmp_path / "calibration-validation-cli"
    populate_ready_calibration_validation_gate(run_root)

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_rewrite_gate.py",
            run_root.as_posix(),
            "--gate",
            CALIBRATION_VALIDATION_GATE_ID,
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )

    assert f"{CALIBRATION_VALIDATION_GATE_ID}: ready (2/2 ready)" in result.stdout


def test_rewrite_status_report_summarizes_all_gates(tmp_path: Path) -> None:
    run_root = tmp_path / "rewrite-status-run"
    full_capture_root = default_full_capture_run_root(run_root)
    populate_ready_fake_gate(run_root)

    report = build_rewrite_status_report(run_root)

    assert report["schema_version"] == "rewrite_status_report.v1"
    assert report["overall_status"] == "blocked"
    assert report["summary"]["gate_count"] == 4
    assert report["summary"]["ready_gate_count"] == 1
    assert report["summary"]["blocked_gate_count"] == 3
    assert [gate["gate_id"] for gate in report["gates"]] == [
        "rewrite_fake_end_to_end.v1",
        "rewrite_full_capture.v1",
        "rewrite_foundationpose_runtime.v1",
        "rewrite_calibration_validation.v1",
    ]
    assert report["gates"][0]["overall_status"] == "ready"
    assert report["next_gate"]["gate_id"] == "rewrite_full_capture.v1"
    assert report["next_gate"]["run_root"] == full_capture_root.as_posix()
    assert report["next_blockers"][0]["gate_id"] == "rewrite_full_capture.v1"
    assert report["gate_run_roots"]["rewrite_fake_end_to_end.v1"] == (
        run_root.as_posix()
    )
    assert report["gate_run_roots"]["rewrite_full_capture.v1"] == (
        full_capture_root.as_posix()
    )
    assert report["next_actions"][0] == {
        "gate_id": "rewrite_full_capture.v1",
        "label": "Create real lab run config",
        "command": [
            "uv",
            "run",
            "python",
            "scripts/create_run_config.py",
            full_capture_root.as_posix(),
            "--robot-mode",
            "real",
            "--sequence",
            "real_full_capture_validation",
            "--print-sequence-plan",
        ],
        "reason": (
            "The full-capture gate requires an intentional real robot profile "
            "with enabled lab sensors and the saved real full-capture "
            "validation sequence."
        ),
        "blocks_on": ["run_config"],
    }
    assert report["next_actions"][1]["label"] == (
        "Plan real full-capture validation sequence"
    )
    assert report["next_actions"][1]["command"] == [
        "uv",
        "run",
        "python",
        "scripts/run_pipeline_sequence.py",
        full_capture_root.as_posix(),
        "--sequence",
        "real_full_capture_validation",
        "--plan-only",
    ]
    assert len(report["next_actions"]) == 2


def test_rewrite_status_report_accepts_per_gate_run_roots(tmp_path: Path) -> None:
    status_root = tmp_path / "rewrite-status-root"
    fake_root = tmp_path / "fake-evidence"
    full_root = tmp_path / "full-capture-evidence"
    populate_ready_fake_gate(fake_root)
    populate_ready_full_capture_gate(full_root)

    report = build_rewrite_status_report(
        status_root,
        gate_run_roots={
            "rewrite_fake_end_to_end.v1": fake_root,
            "rewrite_full_capture.v1": full_root,
        },
    )

    assert report["overall_status"] == "blocked"
    assert report["summary"]["ready_gate_count"] == 2
    assert report["summary"]["blocked_gate_count"] == 2
    assert report["gate_run_roots"]["rewrite_fake_end_to_end.v1"] == (
        fake_root.as_posix()
    )
    assert report["gate_run_roots"]["rewrite_full_capture.v1"] == (
        full_root.as_posix()
    )
    gates = {gate["gate_id"]: gate for gate in report["gates"]}
    assert gates["rewrite_fake_end_to_end.v1"]["run_root"] == fake_root.as_posix()
    assert gates["rewrite_fake_end_to_end.v1"]["overall_status"] == "ready"
    assert gates["rewrite_full_capture.v1"]["run_root"] == full_root.as_posix()
    assert gates["rewrite_full_capture.v1"]["overall_status"] == "ready"
    assert report["next_gate"]["gate_id"] == "rewrite_foundationpose_runtime.v1"
    assert report["next_gate"]["run_root"] == status_root.as_posix()
    assert report["next_actions"][0]["command"] == [
        "uv",
        "run",
        "python",
        "scripts/run_foundationpose_stage.py",
        status_root.as_posix(),
    ]


def test_rewrite_status_next_action_skips_existing_full_capture_plan_for_preflight(
    tmp_path: Path,
) -> None:
    status_root = tmp_path / "rewrite-status-root"
    fake_root = tmp_path / "fake-evidence"
    full_root = tmp_path / "full-capture-evidence"
    populate_ready_fake_gate(fake_root)
    write_json(
        full_root / RUN_CONFIG,
        {
            "schema_version": "run_config.v1",
            "robot_profile": {"mode": "real"},
            "capture": {
                "sensors": [
                    {
                        "sensor_type": "realsense_d435",
                        "device_id": "123",
                        "enabled": True,
                    }
                ]
            },
        },
    )
    write_json(
        full_root / PIPELINE_SEQUENCE_PLAN,
        {
            "schema_version": "pipeline_sequence_plan.v1",
            "sequence_id": "real_full_capture_validation",
        },
    )

    report = build_rewrite_status_report(
        status_root,
        gate_run_roots={
            "rewrite_fake_end_to_end.v1": fake_root,
            "rewrite_full_capture.v1": full_root,
        },
    )

    assert report["next_gate"]["gate_id"] == "rewrite_full_capture.v1"
    assert report["next_actions"][0]["label"] == (
        "Write real run preflight"
    )
    assert report["next_actions"][0]["command"] == [
        "uv",
        "run",
        "python",
        "scripts/run_preflight.py",
        full_root.as_posix(),
        "--check",
        "--write",
    ]


def test_rewrite_status_next_action_stops_on_hardware_blocker(
    tmp_path: Path,
) -> None:
    status_root = tmp_path / "rewrite-status-root"
    fake_root = tmp_path / "fake-evidence"
    full_root = tmp_path / "full-capture-evidence"
    populate_ready_fake_gate(fake_root)
    write_json(
        full_root / RUN_CONFIG,
        {
            "schema_version": "run_config.v1",
            "robot_profile": {"mode": "real"},
            "capture": {
                "sensors": [
                    {
                        "sensor_type": "realsense_d435",
                        "device_id": "123",
                        "enabled": True,
                    }
                ]
            },
        },
    )
    write_json(
        full_root / PIPELINE_SEQUENCE_PLAN,
        {
            "schema_version": "pipeline_sequence_plan.v1",
            "sequence_id": "real_full_capture_validation",
        },
    )
    write_json(full_root / RUN_PREFLIGHT_REPORT, {"overall_status": "warning"})
    write_json(
        full_root / HARDWARE_STATUS_REPORT,
        {
            "overall_status": "error",
            "checks": [
                {"name": "sensor:realsense_d435", "status": "error"},
            ],
        },
    )

    report = build_rewrite_status_report(
        status_root,
        gate_run_roots={
            "rewrite_fake_end_to_end.v1": fake_root,
            "rewrite_full_capture.v1": full_root,
        },
    )

    assert report["next_gate"]["gate_id"] == "rewrite_full_capture.v1"
    assert report["next_actions"][0]["label"] == "Inspect sensor status"
    assert report["next_actions"][0]["command"] == [
        "uv",
        "run",
        "python",
        "scripts/sensor_status.py",
        "--json",
        "--check-expected",
    ]
    assert report["next_actions"][1]["label"] == (
        "Refresh hardware status after sensor fix"
    )
    assert report["next_actions"][1]["command"] == [
        "uv",
        "run",
        "python",
        "scripts/run_hardware_status_stage.py",
        full_root.as_posix(),
    ]


def test_rewrite_status_next_action_refreshes_fake_hardware_profile_as_real(
    tmp_path: Path,
) -> None:
    status_root = tmp_path / "rewrite-status-root"
    fake_root = tmp_path / "fake-evidence"
    full_root = tmp_path / "full-capture-evidence"
    populate_ready_fake_gate(fake_root)
    write_json(
        full_root / RUN_CONFIG,
        {
            "schema_version": "run_config.v1",
            "robot_profile": {"mode": "real"},
            "capture": {
                "sensors": [
                    {
                        "sensor_type": "realsense_d435",
                        "device_id": "123",
                        "enabled": True,
                    }
                ]
            },
        },
    )
    write_json(
        full_root / PIPELINE_SEQUENCE_PLAN,
        {
            "schema_version": "pipeline_sequence_plan.v1",
            "sequence_id": "real_full_capture_validation",
        },
    )
    write_json(full_root / RUN_PREFLIGHT_REPORT, {"overall_status": "warning"})
    write_json(
        full_root / HARDWARE_STATUS_REPORT,
        {
            "overall_status": "warning",
            "robot_status": {"selected_profile": {"mode": "fake"}},
        },
    )

    report = build_rewrite_status_report(
        status_root,
        gate_run_roots={
            "rewrite_fake_end_to_end.v1": fake_root,
            "rewrite_full_capture.v1": full_root,
        },
    )

    assert report["next_gate"]["gate_id"] == "rewrite_full_capture.v1"
    assert report["next_actions"][0]["label"] == (
        "Refresh hardware status from run config"
    )
    assert report["next_actions"][0]["command"] == [
        "uv",
        "run",
        "python",
        "scripts/run_hardware_status_stage.py",
        full_root.as_posix(),
    ]


def test_rewrite_status_next_action_writes_missing_hardware_snapshot(
    tmp_path: Path,
) -> None:
    status_root = tmp_path / "rewrite-status-root"
    fake_root = tmp_path / "fake-evidence"
    full_root = tmp_path / "full-capture-evidence"
    populate_ready_fake_gate(fake_root)
    write_json(
        full_root / RUN_CONFIG,
        {
            "schema_version": "run_config.v1",
            "robot_profile": {"mode": "real"},
            "capture": {
                "sensors": [
                    {
                        "sensor_type": "realsense_d435",
                        "device_id": "123",
                        "enabled": True,
                    }
                ]
            },
        },
    )
    write_json(
        full_root / PIPELINE_SEQUENCE_PLAN,
        {
            "schema_version": "pipeline_sequence_plan.v1",
            "sequence_id": "real_full_capture_validation",
        },
    )
    write_json(full_root / RUN_PREFLIGHT_REPORT, {"overall_status": "warning"})

    report = build_rewrite_status_report(
        status_root,
        gate_run_roots={
            "rewrite_fake_end_to_end.v1": fake_root,
            "rewrite_full_capture.v1": full_root,
        },
    )

    assert report["next_gate"]["gate_id"] == "rewrite_full_capture.v1"
    assert report["next_actions"][0]["label"] == "Write hardware status snapshot"
    assert report["next_actions"][0]["command"] == [
        "uv",
        "run",
        "python",
        "scripts/run_hardware_status_stage.py",
        full_root.as_posix(),
    ]


def test_write_rewrite_status_report(tmp_path: Path) -> None:
    run_root = tmp_path / "rewrite-status-write"
    populate_ready_fake_gate(run_root)

    path, report = write_rewrite_status_report(run_root)

    assert path == run_root / REWRITE_STATUS_REPORT
    assert report["summary"]["ready_gate_count"] == 1
    written = json.loads(path.read_text())
    assert written["schema_version"] == "rewrite_status_report.v1"


def test_rewrite_status_cli_writes_report(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = tmp_path / "rewrite-status-cli"
    full_capture_root = default_full_capture_run_root(run_root)
    populate_ready_fake_gate(run_root)

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_rewrite_status.py",
            run_root.as_posix(),
            "--write",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "rewrite milestones: blocked (1/4 gates ready" in result.stdout
    assert "next actions:" in result.stdout
    assert "1. Create real lab run config" in result.stdout
    assert (
        "uv run python scripts/create_run_config.py "
        f"{full_capture_root.as_posix()} --robot-mode real --sequence "
        "real_full_capture_validation --print-sequence-plan"
    ) in result.stdout
    assert (run_root / REWRITE_STATUS_REPORT).is_file()
    written = json.loads((run_root / REWRITE_STATUS_REPORT).read_text())
    assert written["gate_run_roots"]["rewrite_full_capture.v1"] == (
        full_capture_root.as_posix()
    )


def test_rewrite_status_cli_accepts_gate_run_root_overrides(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    status_root = tmp_path / "rewrite-status-cli-mixed"
    fake_root = tmp_path / "fake-evidence-cli"
    full_root = tmp_path / "full-capture-evidence-cli"
    populate_ready_fake_gate(fake_root)
    populate_ready_full_capture_gate(full_root)

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_rewrite_status.py",
            status_root.as_posix(),
            "--write",
            "--gate-run-root",
            f"rewrite_fake_end_to_end.v1={fake_root.as_posix()}",
            "--gate-run-root",
            f"rewrite_full_capture.v1={full_root.as_posix()}",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "rewrite milestones: blocked (2/4 gates ready" in result.stdout
    written = json.loads((status_root / REWRITE_STATUS_REPORT).read_text())
    assert written["summary"]["ready_gate_count"] == 2


def test_rewrite_status_cli_prints_multiple_next_actions_for_sensor_blocker(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    status_root = tmp_path / "rewrite-status-cli-sensor-blocked"
    fake_root = tmp_path / "fake-evidence-cli"
    full_root = tmp_path / "full-capture-evidence-cli"
    populate_ready_fake_gate(fake_root)
    write_json(
        full_root / RUN_CONFIG,
        {
            "schema_version": "run_config.v1",
            "robot_profile": {"mode": "real"},
            "capture": {
                "sensors": [
                    {
                        "sensor_type": "realsense_d435",
                        "device_id": "123",
                        "enabled": True,
                    }
                ]
            },
        },
    )
    write_json(
        full_root / PIPELINE_SEQUENCE_PLAN,
        {
            "schema_version": "pipeline_sequence_plan.v1",
            "sequence_id": "real_full_capture_validation",
        },
    )
    write_json(full_root / RUN_PREFLIGHT_REPORT, {"overall_status": "warning"})
    write_json(
        full_root / HARDWARE_STATUS_REPORT,
        {
            "overall_status": "error",
            "checks": [
                {"name": "sensor:realsense_d435", "status": "error"},
            ],
        },
    )

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_rewrite_status.py",
            status_root.as_posix(),
            "--gate-run-root",
            f"rewrite_fake_end_to_end.v1={fake_root.as_posix()}",
            "--gate-run-root",
            f"rewrite_full_capture.v1={full_root.as_posix()}",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "next actions:" in result.stdout
    assert "1. Inspect sensor status" in result.stdout
    assert "2. Refresh hardware status after sensor fix" in result.stdout
    assert (
        "uv run python scripts/run_hardware_status_stage.py "
        f"{full_root.as_posix()}"
    ) in result.stdout


def test_rewrite_status_cli_prints_next_blocker_diagnostics(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    status_root = tmp_path / "rewrite-status-cli-diagnostics"
    fake_root = tmp_path / "fake-evidence-cli-diagnostics"
    full_root = tmp_path / "full-capture-evidence-cli-diagnostics"
    populate_ready_fake_gate(fake_root)
    populate_blocked_full_capture_with_sensor_diagnostics(full_root)

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_rewrite_status.py",
            status_root.as_posix(),
            "--gate-run-root",
            f"rewrite_fake_end_to_end.v1={fake_root.as_posix()}",
            "--gate-run-root",
            f"rewrite_full_capture.v1={full_root.as_posix()}",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "next blockers:" in result.stdout
    assert "blocked check: sensor:realsense_d435" in result.stdout
    assert "diagnostic: RealSense discovery failed." in result.stdout
    assert "hint: Check USB/udev access." in result.stdout
