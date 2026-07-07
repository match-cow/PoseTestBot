from __future__ import annotations

import json
import subprocess
from pathlib import Path

from posetestbot.io.artifacts import (
    BOP_DIR,
    BOP_EXPORT_MANIFEST,
    BOP_TARGETS_BOP19,
    CALIBRATION_PROFILES,
    CALIBRATION_VALIDATION_REPORT,
    CAPTURE_EXECUTION_REPORT,
    DEPTH_DIR,
    HARDWARE_STATUS_REPORT,
    RGB_DIR,
    RUN_PREFLIGHT_REPORT,
    SYNTHETIC_RGBD_REPORT,
    SYNC_QUALITY_REPORT,
)
from posetestbot.pipeline.rewrite_gate import (
    BOP_EXPORT_READINESS_GATE_ID,
    CALIBRATION_VALIDATION_GATE_ID,
    FAKE_ACQUISITION_TO_BOP_GATE_ID,
    FULL_CAPTURE_GATE_ID,
    GATE_IDS,
    build_bop_export_readiness_gate_report,
    build_calibration_validation_gate_report,
    build_fake_end_to_end_gate_report,
    build_rewrite_status_report,
)
from posetestbot.pipeline.run_config import create_run_config, write_run_config


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def populate_bop_export(
    run_root: Path,
    *,
    with_targets: bool = True,
) -> None:
    scene = run_root / BOP_DIR / "realsense_123" / "test" / "000001"
    (scene / RGB_DIR).mkdir(parents=True)
    (scene / DEPTH_DIR).mkdir()
    (scene / RGB_DIR / "000000.png").write_bytes(b"rgb")
    (scene / DEPTH_DIR / "000000.png").write_bytes(b"depth")
    write_json(scene / "scene_camera.json", {"0": {"cam_K": [1, 0, 0, 0, 1, 0, 0, 0, 1]}})
    write_json(scene / "scene_gt.json", {"0": []})
    write_json(
        run_root / BOP_DIR / BOP_EXPORT_MANIFEST,
        {
            "schema_version": "bop_export_manifest.v1",
            "exports": [
                {
                    "sensor_name": "realsense_123",
                    "scene_id": 1,
                    "split": "test",
                    "scene_folder": scene.relative_to(run_root).as_posix(),
                }
            ],
            "object_models": [
                {
                    "object_name": "cube",
                    "obj_id": 1,
                    "bop_path": "bop/models/obj_000001.ply",
                }
            ],
        },
    )
    write_json(run_root / BOP_DIR / "models" / "models_info.json", {"1": {"diameter": 1}})
    if with_targets:
        write_json(
            run_root / BOP_DIR / BOP_TARGETS_BOP19,
            [{"scene_id": 1, "im_id": 0, "obj_id": 1, "inst_count": 1}],
        )


def populate_fake_acquisition_to_bop(run_root: Path) -> None:
    config = create_run_config(
        run_root=run_root,
        sequence_id="fake_capture_to_bop_dataset_dry_run",
    )
    write_run_config(run_root, config)
    write_json(
        run_root / RUN_PREFLIGHT_REPORT,
        {"schema_version": "run_preflight.v1", "overall_status": "warning", "config": config.to_dict()},
    )
    write_json(
        run_root / CAPTURE_EXECUTION_REPORT,
        {
            "schema_version": "capture_execution_report.v1",
            "status": "succeeded",
            "raw_pose_count": 3,
            "selected_roles": ["robot_controller", "robot_pose_receiver"],
        },
    )
    write_json(
        run_root / SYNTHETIC_RGBD_REPORT,
        {"schema_version": "synthetic_rgbd_fixture.v1", "status": "succeeded", "frame_count": 1},
    )
    write_json(
        run_root / SYNC_QUALITY_REPORT,
        {"schema_version": "sync_quality_report.v1", "overall_status": "ok"},
    )
    populate_bop_export(run_root)


def test_fake_acquisition_to_bop_gate_ready(tmp_path: Path) -> None:
    run_root = tmp_path / "fake-run"
    populate_fake_acquisition_to_bop(run_root)

    report = build_fake_end_to_end_gate_report(run_root)

    assert report["gate_id"] == FAKE_ACQUISITION_TO_BOP_GATE_ID
    assert report["overall_status"] == "ready"
    assert {check["name"] for check in report["checks"]} >= {
        "capture_execution",
        "synthetic_rgbd_fixture",
        "sync_quality",
        "bop_export",
        "bop_targets",
        "bop_models_info",
    }


def test_fake_acquisition_to_bop_gate_allows_empty_bop_targets(tmp_path: Path) -> None:
    run_root = tmp_path / "fake-run-empty-targets"
    populate_fake_acquisition_to_bop(run_root)
    write_json(run_root / BOP_DIR / BOP_TARGETS_BOP19, [])

    report = build_fake_end_to_end_gate_report(run_root)

    assert report["overall_status"] == "ready"
    targets = next(check for check in report["checks"] if check["name"] == "bop_targets")
    assert targets["details"]["target_count"] == 0


def test_bop_export_readiness_gate_blocks_missing_targets(tmp_path: Path) -> None:
    run_root = tmp_path / "bop-run"
    populate_bop_export(run_root, with_targets=False)

    report = build_bop_export_readiness_gate_report(run_root)

    assert report["gate_id"] == BOP_EXPORT_READINESS_GATE_ID
    assert report["overall_status"] == "blocked"
    blockers = {blocker["name"] for blocker in report["next_blockers"]}
    assert "bop_targets" in blockers


def test_calibration_validation_gate_ready_after_promotion(tmp_path: Path) -> None:
    run_root = tmp_path / "calibration-run"
    write_json(
        run_root / CALIBRATION_VALIDATION_REPORT,
        {
            "schema_version": "calibration_validation_report.v1",
            "overall_status": "ok",
            "profile_count": 1,
            "promotable_profile_count": 1,
            "promotion": {
                "requested": True,
                "promoted": True,
                "profile_count": 1,
                "path": "calibration_profiles.json",
            },
        },
    )
    write_json(
        run_root / CALIBRATION_PROFILES,
        {
            "schema_version": "calibration_profiles.v1",
            "profiles": [
                {
                    "profile_id": "profile-1",
                    "sensor_id": "realsense_123",
                    "sensor_type": "realsense_d435",
                    "status": "valid",
                    "quality": {
                        "num_inliers": 8,
                        "residual_translation_mm": 1.0,
                        "residual_rotation_deg": 0.5,
                    },
                }
            ],
        },
    )

    report = build_calibration_validation_gate_report(run_root)

    assert report["gate_id"] == CALIBRATION_VALIDATION_GATE_ID
    assert report["overall_status"] == "ready"


def test_rewrite_status_uses_acquisition_gate_ids(tmp_path: Path) -> None:
    run_root = tmp_path / "status-run"

    report = build_rewrite_status_report(run_root)

    gate_ids = [gate["gate_id"] for gate in report["gates"]]
    assert tuple(gate_ids) == GATE_IDS
    assert "rewrite_foundationpose_runtime.v1" not in gate_ids


def test_rewrite_gate_cli_accepts_bop_export_readiness_gate(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = tmp_path / "bop-cli"
    populate_bop_export(run_root)

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_rewrite_gate.py",
            run_root.as_posix(),
            "--gate",
            BOP_EXPORT_READINESS_GATE_ID,
            "--json",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["gate_id"] == BOP_EXPORT_READINESS_GATE_ID
    assert payload["overall_status"] == "ready"


def test_full_capture_gate_checks_real_hardware_snapshot(tmp_path: Path) -> None:
    run_root = tmp_path / "real-run"
    config = create_run_config(run_root=run_root, robot_mode="real")
    write_run_config(run_root, config)
    write_json(
        run_root / RUN_PREFLIGHT_REPORT,
        {"schema_version": "run_preflight.v1", "overall_status": "ok", "config": config.to_dict()},
    )
    write_json(
        run_root / HARDWARE_STATUS_REPORT,
        {
            "schema_version": "hardware_status_report.v1",
            "overall_status": "ok",
            "robot_status": {"selected_profile": {"mode": "fake"}},
        },
    )

    from posetestbot.pipeline.rewrite_gate import build_full_capture_gate_report

    report = build_full_capture_gate_report(run_root)

    hardware = next(check for check in report["checks"] if check["name"] == "hardware_status")
    assert report["gate_id"] == FULL_CAPTURE_GATE_ID
    assert hardware["status"] == "blocked"
    assert hardware["details"]["robot_mode_ok"] is False
