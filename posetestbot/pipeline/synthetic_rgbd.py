"""Synthetic RGB-D fixture capture for hardware-free pipeline validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from posetestbot.io.artifacts import (
    FRAME_METADATA_JSONL,
    RAW_ROBOT_EE_POSES,
    SYNTHETIC_RGBD_REPORT,
)
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    make_sensor_record,
    upsert_stage,
    write_run_manifest,
)
from posetestbot.sensors.contracts import CameraIntrinsics, MountingMode, SensorType
from posetestbot.sensors.frame_writer import (
    write_legacy_camera_sidecars,
    write_legacy_rgbd_frame,
)


SCHEMA_VERSION = "synthetic_rgbd_report.v1"
DEFAULT_SENSOR_FOLDER = "realsense_synthetic"
DEFAULT_SENSOR_ID = "synthetic"
DEFAULT_WIDTH = 64
DEFAULT_HEIGHT = 48
DEFAULT_SYNC_DELTA_MS = 100.0


def _read_json_object(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return loaded


def _robot_timestamp_ns(record: Mapping[str, Any]) -> int:
    if record.get("host_received_timestamp_ns") is not None:
        return int(record["host_received_timestamp_ns"])
    if record.get("host_wall_timestamp_ns") is not None:
        return int(record["host_wall_timestamp_ns"])
    return int(record["framename"]) * 1_000_000


def _sorted_pose_records(raw_poses: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for key, value in raw_poses.items():
        if not isinstance(value, Mapping):
            continue
        record = dict(value)
        record["pose_index"] = int(key)
        record["timestamp_ns"] = _robot_timestamp_ns(record)
        records.append(record)
    return sorted(records, key=lambda item: item["timestamp_ns"])


def _default_intrinsics(width: int, height: int) -> CameraIntrinsics:
    fx = float(width)
    fy = float(width)
    cx = float(width - 1) / 2.0
    cy = float(height - 1) / 2.0
    return CameraIntrinsics(
        cam_k=(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0),
        width=width,
        height=height,
        distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
        depth_scale_to_mm=1.0,
    )


def _synthetic_rgb(width: int, height: int, frame_index: int) -> np.ndarray:
    x = np.arange(width, dtype=np.uint16)[None, :]
    y = np.arange(height, dtype=np.uint16)[:, None]
    red = np.broadcast_to((x + frame_index * 17) % 255, (height, width))
    green = np.broadcast_to((y + frame_index * 23) % 255, (height, width))
    blue = np.full((height, width), (frame_index * 41) % 255, dtype=np.uint8)
    return np.stack(
        [
            blue,
            green.astype(np.uint8),
            red.astype(np.uint8),
        ],
        axis=2,
    )


def _synthetic_depth(width: int, height: int, frame_index: int) -> np.ndarray:
    base = np.arange(width * height, dtype=np.uint16).reshape(height, width)
    return (base % 1000) + np.uint16(500 + frame_index * 10)


def write_synthetic_rgbd_fixture(
    run_root: str | Path,
    *,
    sensor_folder_name: str = DEFAULT_SENSOR_FOLDER,
    sensor_id: str = DEFAULT_SENSOR_ID,
    frame_count: int | None = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    sync_delta_ms: float = DEFAULT_SYNC_DELTA_MS,
    include_end_motion: bool = False,
    overwrite: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Write a synthetic sensor folder aligned to existing raw robot poses."""

    root = Path(run_root)
    raw_pose_path = root / RAW_ROBOT_EE_POSES
    raw_poses = _read_json_object(raw_pose_path)
    records = _sorted_pose_records(raw_poses)
    if not include_end_motion:
        records = [record for record in records if record.get("motion") != "end"]
    if frame_count is not None:
        records = records[:frame_count]
    if not records:
        raise ValueError(
            f"No usable robot poses found in {raw_pose_path}; cannot write synthetic RGB-D."
        )

    sensor_folder = root / sensor_folder_name
    metadata_path = sensor_folder / FRAME_METADATA_JSONL
    if sensor_folder.exists() and any(sensor_folder.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Synthetic sensor folder already exists: {sensor_folder}. "
            "Pass overwrite=True or --overwrite to replace fixture files."
        )
    if overwrite and sensor_folder.exists():
        for child in sensor_folder.iterdir():
            if child.is_dir():
                for grandchild in sorted(child.rglob("*"), reverse=True):
                    if grandchild.is_file():
                        grandchild.unlink()
                    elif grandchild.is_dir():
                        grandchild.rmdir()
                child.rmdir()
            else:
                child.unlink()

    intrinsics = _default_intrinsics(width, height)
    write_legacy_camera_sidecars(
        sensor_folder,
        intrinsics,
        include_distortion_in_cam_k=True,
    )

    metadata_records = []
    timestamp_offset_ns = int(sync_delta_ms * 1_000_000)
    for frame_index, pose_record in enumerate(records):
        robot_timestamp_ns = int(pose_record["timestamp_ns"])
        frame_timestamp_ns = robot_timestamp_ns + timestamp_offset_ns
        metadata = write_legacy_rgbd_frame(
            sensor_folder,
            rgb_image=_synthetic_rgb(width, height, frame_index),
            depth_image=_synthetic_depth(width, height, frame_index),
            sensor_type=SensorType.REALSENSE_D435,
            sensor_id=sensor_id,
            frame_index=frame_index,
            sensor_timestamp_ns=frame_timestamp_ns,
            host_received_timestamp_ns=frame_timestamp_ns,
            host_wall_timestamp_ns=frame_timestamp_ns,
            frame_stem=str(frame_timestamp_ns // 1_000_000),
            extra_metadata={
                "synthetic": True,
                "source_robot_pose_index": pose_record["pose_index"],
                "source_robot_motion": pose_record.get("motion"),
                "expected_sync_delta_ms": sync_delta_ms,
            },
        )
        metadata_records.append(metadata)

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "succeeded",
        "sensor_folder": sensor_folder.relative_to(root).as_posix(),
        "sensor_type": SensorType.REALSENSE_D435.value,
        "sensor_id": sensor_id,
        "frame_count": len(metadata_records),
        "width": width,
        "height": height,
        "expected_sync_delta_ms": sync_delta_ms,
        "raw_robot_pose_artifact": RAW_ROBOT_EE_POSES,
        "frame_metadata_artifact": metadata_path.relative_to(root).as_posix(),
    }
    report_path = root / SYNTHETIC_RGBD_REPORT
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    manifest = load_or_create_run_manifest(root)
    sensors = [
        sensor
        for sensor in manifest.get("sensors", [])
        if sensor.get("folder") != sensor_folder_name
    ]
    sensors.append(
        make_sensor_record(
            sensor_type=SensorType.REALSENSE_D435,
            device_id=sensor_id,
            folder=sensor_folder,
            run_root=root,
            display_name="Synthetic RealSense fixture",
            mounting_mode=MountingMode.STATIC,
            status="synthetic",
            metadata={
                "fixture": True,
                "expected_sync_delta_ms": sync_delta_ms,
                "frame_count": len(metadata_records),
            },
        )
    )
    manifest["sensors"] = sensors
    upsert_stage(
        manifest,
        name="synthetic_rgbd_fixture",
        status="succeeded",
        artifacts={
            SYNTHETIC_RGBD_REPORT: report_path,
            FRAME_METADATA_JSONL: metadata_path,
            sensor_folder_name: sensor_folder,
        },
        run_root=root,
        message=f"Wrote {len(metadata_records)} synthetic RGB-D frame(s).",
    )
    write_run_manifest(manifest, root)

    return report_path, report
