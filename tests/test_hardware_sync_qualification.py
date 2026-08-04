from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from posetestbot.io.artifacts import (
    CAPTURE_EXECUTION_LOGS_DIR,
    CAPTURE_EXECUTION_REPORT,
    CAPTURE_EXECUTION_STATUS,
    DATASET_MANIFEST,
    FRAME_METADATA_JSONL,
    HARDWARE_SYNC_QUALIFICATION,
    RAW_ROBOT_EE_POSES,
)
from posetestbot.pipeline.capture_plan_preflight import (
    build_capture_plan_preflight,
)
from posetestbot.pipeline.run_config import (
    SensorRunConfig,
    create_run_config,
    write_run_config,
)
from posetestbot.sensors.hardware_sync_qualification import (
    EVIDENCE_DIR,
    HardwareSyncQualificationError,
    hardware_sync_qualification_contract,
    record_hardware_sync_qualification,
    validate_hardware_sync_qualification,
)


def _hardware_config(run_root: Path, *, fps: int = 6):
    return create_run_config(
        run_root=run_root,
        fps=fps,
        sensors=(
            SensorRunConfig(
                "realsense_d435",
                "static",
                "Static",
                mounting_mode="static",
            ),
            SensorRunConfig(
                "realsense_d435",
                "hand",
                "Robot",
                mounting_mode="eye_in_hand",
            ),
        ),
        synchronization={
            "schema_version": "capture_synchronization.v1",
            "mode": "hardware_trigger",
            "implementation": "realsense_inter_cam_sync",
            "scope": "depth_exposure",
            "group_id": "mixed-rig",
            "master_sensor_key": "realsense_d435:static",
            "max_depth_timestamp_skew_ms": 2.0,
        },
    )


def _record(run_root: Path, evidence: Path) -> dict:
    return record_hardware_sync_qualification(
        run_root,
        operator="researcher@example.test",
        method="pulsed_light",
        observed_max_depth_timestamp_skew_ms=0.75,
        evidence_paths=[evidence],
        confirm_passed=True,
    )


def _sensor_status() -> dict:
    return {
        "schema_version": "sensor_status.v1",
        "families": [
            {
                "sensor_type": "realsense_d435",
                "display_name": "Intel RealSense D435",
                "sdk_module": "pyrealsense2",
                "sdk_available": True,
                "connected_count": 2,
                "devices": [
                    {
                        "sensor_type": "realsense_d435",
                        "device_id": "static",
                        "connected": True,
                        "capture_ready": True,
                        "metadata": {"usb_type_descriptor": "3.2"},
                    },
                    {
                        "sensor_type": "realsense_d435",
                        "device_id": "hand",
                        "connected": True,
                        "capture_ready": True,
                        "metadata": {"usb_type_descriptor": "3.2"},
                    },
                ],
            }
        ],
    }


def test_recorded_qualification_is_hash_verified_and_exactly_config_bound(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    config = _hardware_config(run_root)
    write_run_config(run_root, config)
    evidence = tmp_path / "physical pulse trace.csv"
    evidence.write_bytes(b"time_ns,static,hand\n0,1,1\n")

    artifact = _record(run_root, evidence)
    provenance = validate_hardware_sync_qualification(run_root)

    assert artifact["schema_version"] == "hardware_sync_qualification.v1"
    assert provenance["status"] == "passed"
    assert provenance["configuration_sha256"] == artifact["configuration_sha256"]
    assert provenance["rgb_exposure_hardware_synchronized"] is False
    copied = run_root / artifact["evidence"][0]["path"]
    assert copied.parent.parent.name == EVIDENCE_DIR
    assert copied.read_bytes() == evidence.read_bytes()
    assert artifact["evidence"][0]["sha256"] == hashlib.sha256(
        evidence.read_bytes()
    ).hexdigest()
    contract = hardware_sync_qualification_contract(config.to_dict())
    roles = {
        item["sensor_key"]: item["hardware_sync_role"]
        for item in contract["capture"]["sensors"]
    }
    assert roles == {
        "realsense_d435:static": "master",
        "realsense_d435:hand": "subordinate",
    }


def test_qualification_rejects_missing_confirmation_stale_config_and_tampering(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    config = _hardware_config(run_root)
    write_run_config(run_root, config)
    evidence = tmp_path / "trace.bin"
    evidence.write_bytes(b"physical trace")

    with pytest.raises(HardwareSyncQualificationError, match="confirm_passed"):
        record_hardware_sync_qualification(
            run_root,
            operator="operator",
            method="pulsed_light",
            observed_max_depth_timestamp_skew_ms=0.5,
            evidence_paths=[evidence],
            confirm_passed=False,
        )

    artifact = _record(run_root, evidence)
    changed_config = _hardware_config(run_root, fps=7)
    with pytest.raises(HardwareSyncQualificationError, match="stale"):
        validate_hardware_sync_qualification(
            run_root,
            run_config=changed_config.to_dict(),
        )

    changed_orientation = config.to_dict()
    changed_orientation["capture"]["sensors"][0]["inverted"] = True
    with pytest.raises(HardwareSyncQualificationError, match="stale"):
        validate_hardware_sync_qualification(
            run_root,
            run_config=changed_orientation,
        )

    copied = run_root / artifact["evidence"][0]["path"]
    copied.write_bytes(b"tampered")
    with pytest.raises(
        HardwareSyncQualificationError,
        match="size mismatch|SHA-256 mismatch",
    ):
        validate_hardware_sync_qualification(run_root)


def test_qualification_rejects_escaped_or_symlinked_evidence(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    write_run_config(run_root, _hardware_config(run_root))
    evidence = tmp_path / "trace.bin"
    evidence.write_bytes(b"trace")
    _record(run_root, evidence)
    artifact_path = run_root / HARDWARE_SYNC_QUALIFICATION
    artifact = json.loads(artifact_path.read_text())
    artifact["evidence"][0]["path"] = "../trace.bin"
    artifact_path.write_text(json.dumps(artifact))

    with pytest.raises(
        HardwareSyncQualificationError,
        match="managed run-relative",
    ):
        validate_hardware_sync_qualification(run_root)

    artifact = _record(run_root, evidence)
    copied = run_root / artifact["evidence"][0]["path"]
    copied.unlink()
    copied.symlink_to(evidence)
    with pytest.raises(HardwareSyncQualificationError, match="symbolic link"):
        validate_hardware_sync_qualification(run_root)


@pytest.mark.parametrize(
    "blocker",
    [
        "raw_metadata",
        "raw_rgb",
        "raw_depth",
        "raw_robot_pose",
        "capture_status",
        "capture_report",
        "capture_logs",
        "capture_manifest_stage",
    ],
)
def test_qualification_cannot_be_published_or_replaced_after_capture_evidence(
    tmp_path: Path,
    blocker: str,
) -> None:
    run_root = tmp_path / blocker
    write_run_config(run_root, _hardware_config(run_root))
    evidence = tmp_path / f"{blocker}.trace"
    evidence.write_text("physical timing evidence")
    _record(run_root, evidence)
    artifact_path = run_root / HARDWARE_SYNC_QUALIFICATION
    original_artifact = artifact_path.read_bytes()

    raw_sensor = run_root / "realsense_static"
    if blocker == "raw_metadata":
        raw_sensor.mkdir()
        (raw_sensor / FRAME_METADATA_JSONL).write_text("{}\n")
    elif blocker == "raw_rgb":
        (raw_sensor / "rgb").mkdir(parents=True)
    elif blocker == "raw_depth":
        (raw_sensor / "depth").mkdir(parents=True)
    elif blocker == "raw_robot_pose":
        (run_root / RAW_ROBOT_EE_POSES).write_text("{}\n")
    elif blocker == "capture_status":
        (run_root / CAPTURE_EXECUTION_STATUS).write_text("{}\n")
    elif blocker == "capture_report":
        (run_root / CAPTURE_EXECUTION_REPORT).write_text("{}\n")
    elif blocker == "capture_logs":
        (run_root / CAPTURE_EXECUTION_LOGS_DIR).mkdir()
    else:
        (run_root / DATASET_MANIFEST).write_text(
            json.dumps(
                {
                    "stages": [
                        {"name": "capture_execution", "status": "running"}
                    ]
                }
            )
        )

    with pytest.raises(
        HardwareSyncQualificationError,
        match="immutable once capture evidence exists",
    ):
        record_hardware_sync_qualification(
            run_root,
            operator="replacement",
            method="pulsed_light",
            observed_max_depth_timestamp_skew_ms=0.5,
            evidence_paths=[evidence],
            confirm_passed=True,
        )

    assert artifact_path.read_bytes() == original_artifact


def test_initial_qualification_publication_rejects_existing_raw_frames(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    write_run_config(run_root, _hardware_config(run_root))
    (run_root / "realsense_hand" / "rgb").mkdir(parents=True)
    (run_root / "realsense_hand" / "rgb" / "1.png").write_bytes(b"raw")
    evidence = tmp_path / "trace"
    evidence.write_text("physical timing evidence")

    with pytest.raises(
        HardwareSyncQualificationError,
        match="immutable once capture evidence exists",
    ):
        _record(run_root, evidence)

    assert not (run_root / HARDWARE_SYNC_QUALIFICATION).exists()


def test_capture_preflight_fails_closed_until_current_qualification_exists(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    write_run_config(run_root, _hardware_config(run_root))
    evidence = tmp_path / "trace.csv"
    evidence.write_text("physical evidence\n")

    missing_report = build_capture_plan_preflight(
        run_root,
        include_sensor_status=False,
        allow_real_robot=True,
    )
    missing_check = next(
        check
        for check in missing_report["checks"]
        if check["name"] == "hardware_sync_qualification"
    )
    assert missing_report["overall_status"] == "error"
    assert missing_check["status"] == "error"
    assert "does not exist" in missing_check["message"]

    _record(run_root, evidence)
    qualified_report = build_capture_plan_preflight(
        run_root,
        allow_real_robot=True,
        collect_sensors=_sensor_status,
    )
    qualified_check = next(
        check
        for check in qualified_report["checks"]
        if check["name"] == "hardware_sync_qualification"
    )
    assert qualified_report["overall_status"] == "ok"
    assert qualified_check["status"] == "ok"
    assert qualified_check["details"]["valid"] is True


def test_sync_cli_blocks_before_frame_processing_and_invalidates_stale_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "sync_run_non_destructive.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_sync_run_qualification_script",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    sync_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync_script)

    run_root = tmp_path / "run"
    write_run_config(run_root, _hardware_config(run_root))
    stale_groups = (
        run_root
        / "processed"
        / "synchronized"
        / "multiview_frame_groups.json"
    )
    stale_groups.parent.mkdir(parents=True)
    stale_groups.write_text("{}\n")
    synchronize_called = False

    def unexpected_synchronize(*_args, **_kwargs):
        nonlocal synchronize_called
        synchronize_called = True
        raise AssertionError("synchronization must not start")

    monkeypatch.setattr(
        sync_script,
        "parse_args",
        lambda: SimpleNamespace(
            run_root=run_root.as_posix(),
            output_root=None,
            sensor_folder=None,
            sync_delta=None,
            timestamp_source=None,
            robot_timestamp_source=None,
            no_copy=False,
        ),
    )
    monkeypatch.setattr(
        sync_script,
        "resolve_calibration_profile_sync_policy",
        lambda _root: None,
    )
    monkeypatch.setattr(
        sync_script,
        "validate_hardware_sync_qualification",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            HardwareSyncQualificationError("qualification is stale")
        ),
    )
    monkeypatch.setattr(
        sync_script,
        "synchronize_run",
        unexpected_synchronize,
    )

    with pytest.raises(
        HardwareSyncQualificationError,
        match="qualification is stale",
    ):
        sync_script.main()

    assert synchronize_called is False
    assert not stale_groups.exists()
    assert len(
        list(
            stale_groups.parent.glob(
                ".multiview_frame_groups.json.*.invalidated"
            )
        )
    ) == 1
