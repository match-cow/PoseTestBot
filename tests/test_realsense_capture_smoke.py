from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from posetestbot.io.artifacts import (
    DATASET_MANIFEST,
    FRAME_METADATA_JSONL,
    REALSENSE_CAPTURE_SMOKE_REPORT,
)
from posetestbot.pipeline.run_config import (
    create_run_config,
    sensor_config_from_token,
    write_run_config,
)
from posetestbot.sensors.contracts import CameraIntrinsics, SensorDeviceInfo, SensorType
from posetestbot.sensors.frame_writer import (
    write_legacy_camera_sidecars,
    write_legacy_rgbd_frame,
)
from posetestbot.sensors.realsense_smoke import (
    build_realsense_capture_smoke_report,
    write_realsense_capture_smoke_with_manifest,
)


SERIALS = ("825412070181", "033422071805", "923322072633")


def realsense_device(serial: str) -> SensorDeviceInfo:
    return SensorDeviceInfo(
        sensor_type=SensorType.REALSENSE_D435,
        device_id=serial,
        display_name=f"RealSense {serial}",
        metadata={"product_line": "D400"},
    )


def realsense_only_config(run_root: Path):
    return create_run_config(
        run_root=run_root,
        sensors=tuple(
            sensor_config_from_token(f"realsense:{serial}:static:RealSense {serial}")
            for serial in SERIALS
        ),
    )


def fake_capture(output_path, *, device_id, max_frames, fps, warmup_frames, preview, record):
    intrinsics = CameraIntrinsics(
        cam_k=(100.0, 0.0, 2.0, 0.0, 101.0, 2.0, 0.0, 0.0, 1.0),
        width=4,
        height=3,
        depth_scale_to_mm=1.0,
    )
    write_legacy_camera_sidecars(output_path, intrinsics)
    first_frame_id = None
    last_frame_id = None
    for index in range(max_frames):
        metadata = write_legacy_rgbd_frame(
            output_path,
            rgb_image=np.zeros((3, 4, 3), dtype=np.uint8),
            depth_image=np.ones((3, 4), dtype=np.uint16) * index,
            sensor_type=SensorType.REALSENSE_D435,
            sensor_id=device_id,
            frame_index=index,
            sensor_timestamp_ns=1_000 + index,
            host_received_timestamp_ns=2_000 + index,
            host_wall_timestamp_ns=1_700_000_000_000_000_000 + index * 1_000_000,
        )
        first_frame_id = first_frame_id or metadata["frame_id"]
        last_frame_id = metadata["frame_id"]
    return {
        "schema_version": "realsense_capture_summary.v1",
        "status": "succeeded",
        "sensor_id": device_id,
        "frame_count": max_frames,
        "fps": fps,
        "warmup_frames": warmup_frames,
        "preview": preview,
        "record": record,
        "first_frame_id": first_frame_id,
        "last_frame_id": last_frame_id,
    }


def test_realsense_capture_smoke_succeeds_with_three_mocked_devices(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    write_run_config(run_root, realsense_only_config(run_root))

    path, report = write_realsense_capture_smoke_with_manifest(
        run_root,
        max_frames=2,
        warmup_frames=1,
        discoverer=lambda: [realsense_device(serial) for serial in SERIALS],
        capture_func=fake_capture,
    )

    assert path == run_root / REALSENSE_CAPTURE_SMOKE_REPORT
    assert report["schema_version"] == "realsense_capture_smoke.v1"
    assert report["status"] == "succeeded"
    assert len(report["captures"]) == 3
    assert [capture["status"] for capture in report["captures"]] == [
        "succeeded",
        "succeeded",
        "succeeded",
    ]
    for serial in SERIALS:
        metadata = run_root / f"realsense_{serial}" / FRAME_METADATA_JSONL
        assert metadata.is_file()
        assert len(metadata.read_text().splitlines()) == 2

    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage for stage in manifest["stages"] if stage["name"] == "realsense_capture_smoke"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][REALSENSE_CAPTURE_SMOKE_REPORT] == (
        REALSENSE_CAPTURE_SMOKE_REPORT
    )
    assert [sensor["status"] for sensor in manifest["sensors"]] == [
        "captured",
        "captured",
        "captured",
    ]


def test_realsense_capture_smoke_fails_for_missing_visible_serial(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-missing"
    write_run_config(run_root, realsense_only_config(run_root))
    capture_calls = []

    report = build_realsense_capture_smoke_report(
        run_root,
        discoverer=lambda: [realsense_device(serial) for serial in SERIALS[:2]],
        capture_func=lambda *args, **kwargs: capture_calls.append((args, kwargs)),
    )

    assert report["status"] == "failed"
    checks = {check["name"]: check for check in report["checks"]}
    assert checks[f"visible_realsense:{SERIALS[2]}"]["status"] == "error"
    assert capture_calls == []


def test_realsense_capture_smoke_refuses_nonempty_output_folder(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-nonempty"
    write_run_config(run_root, realsense_only_config(run_root))
    sensor_folder = run_root / f"realsense_{SERIALS[0]}"
    sensor_folder.mkdir(parents=True)
    (sensor_folder / "old.txt").write_text("do not mix captures")

    report = build_realsense_capture_smoke_report(
        run_root,
        discoverer=lambda: [realsense_device(serial) for serial in SERIALS],
        capture_func=fake_capture,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "failed"
    assert checks[f"output_folder:realsense_{SERIALS[0]}"]["status"] == "error"
    assert report["captures"] == []


def test_realsense_capture_smoke_refuses_real_robot_profile(tmp_path: Path) -> None:
    run_root = tmp_path / "run-real-robot"
    config = create_run_config(
        run_root=run_root,
        robot_mode="real",
        sensors=tuple(
            sensor_config_from_token(f"realsense:{serial}:static:RealSense {serial}")
            for serial in SERIALS
        ),
    )
    write_run_config(run_root, config)

    report = build_realsense_capture_smoke_report(
        run_root,
        discoverer=lambda: [realsense_device(serial) for serial in SERIALS],
        capture_func=fake_capture,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "failed"
    assert checks["robot_profile_scope"]["status"] == "error"
    assert report["captures"] == []


def test_realsense_capture_smoke_records_one_camera_capture_failure(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-capture-failure"
    write_run_config(run_root, realsense_only_config(run_root))

    def failing_capture(output_path, *, device_id, **kwargs):
        if device_id == SERIALS[1]:
            raise RuntimeError("camera busy")
        return fake_capture(output_path, device_id=device_id, **kwargs)

    path, report = write_realsense_capture_smoke_with_manifest(
        run_root,
        max_frames=1,
        discoverer=lambda: [realsense_device(serial) for serial in SERIALS],
        capture_func=failing_capture,
    )

    assert path == run_root / REALSENSE_CAPTURE_SMOKE_REPORT
    assert report["status"] == "failed"
    assert [capture["status"] for capture in report["captures"]] == [
        "succeeded",
        "failed",
    ]
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage for stage in manifest["stages"] if stage["name"] == "realsense_capture_smoke"
    )
    assert stage["status"] == "failed"


def test_realsense_capture_smoke_cli_writes_failed_report_for_wrong_scope(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-cli"
    repo_root = Path(__file__).resolve().parents[1]
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("oak:auto:static:Cell OAK-D Pro"),),
    )
    write_run_config(run_root, config)

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_realsense_capture_smoke.py",
            run_root.as_posix(),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert f"Wrote {run_root / REALSENSE_CAPTURE_SMOKE_REPORT}" in result.stdout
    report = json.loads((run_root / REALSENSE_CAPTURE_SMOKE_REPORT).read_text())
    assert report["status"] == "failed"
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["realsense_only_config"]["status"] == "error"
