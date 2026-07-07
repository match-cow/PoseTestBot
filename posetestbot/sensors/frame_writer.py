"""Shared RGB-D frame writing helpers for capture adapters."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import cv2

from posetestbot.io.artifacts import (
    CAMERA_DATA_JSON,
    CAMERA_JSON,
    CAM_K,
    DEPTH_DIR,
    DEPTH_SCALE,
    FRAME_METADATA_JSONL,
    RGB_DIR,
)
from posetestbot.sensors.contracts import AlignedRgbdFrame, CameraIntrinsics, SensorType


SCHEMA_VERSION = "frame_metadata.v1"


def ensure_legacy_rgbd_folders(output_path: str | Path) -> Path:
    """Create the legacy capture folder shape used by later acquisition stages."""

    output = Path(output_path)
    (output / RGB_DIR).mkdir(parents=True, exist_ok=True)
    (output / DEPTH_DIR).mkdir(parents=True, exist_ok=True)
    return output


def append_frame_metadata(output_path: str | Path, metadata: Mapping[str, Any]) -> Path:
    """Append one compact JSONL metadata record for a captured frame."""

    output = Path(output_path)
    metadata_path = output / FRAME_METADATA_JSONL
    with open(metadata_path, "a") as f:
        f.write(json.dumps(dict(metadata), separators=(",", ":")) + "\n")
    return metadata_path


def frame_stem_from_host_wall_ns(host_wall_timestamp_ns: int) -> str:
    """Return the legacy millisecond timestamp filename stem."""

    return str(int(round(host_wall_timestamp_ns / 1_000_000)))


def _sensor_type_value(sensor_type: SensorType | str) -> str:
    return sensor_type.value if isinstance(sensor_type, SensorType) else str(sensor_type)


def _write_png(path: Path, image: Any) -> None:
    if not cv2.imwrite(path.as_posix(), image):
        raise OSError(f"Failed to write image: {path}")


def write_legacy_camera_sidecars(
    output_path: str | Path,
    intrinsics: CameraIntrinsics,
    *,
    include_distortion_in_cam_k: bool = False,
) -> dict[str, Path]:
    """Write legacy camera sidecars shared by calibration and BOP export stages."""

    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    cam_k = [float(value) for value in intrinsics.cam_k]
    matrix_rows = intrinsics.as_matrix_rows()

    cam_k_path = output / CAM_K
    with open(cam_k_path, "w") as f:
        for row in matrix_rows:
            f.write(f"{row[0]} {row[1]} {row[2]}\n")
        if include_distortion_in_cam_k and intrinsics.distortion:
            f.write(" ".join(str(float(value)) for value in intrinsics.distortion))
            f.write("\n")

    depth_scale_path = output / DEPTH_SCALE
    with open(depth_scale_path, "w") as f:
        f.write(f"{float(intrinsics.depth_scale_to_mm)}\n")

    camera_json_path = output / CAMERA_JSON
    with open(camera_json_path, "w") as f:
        json.dump(
            {
                "cam_K": cam_k,
                "depth_scale": float(intrinsics.depth_scale_to_mm),
            },
            f,
            indent=4,
        )

    camera_data_path = output / CAMERA_DATA_JSON
    with open(camera_data_path, "w") as f:
        json.dump(
            {
                "K": [[float(value) for value in row] for row in matrix_rows],
                "resolution": [int(intrinsics.height), int(intrinsics.width)],
            },
            f,
        )

    return {
        CAM_K: cam_k_path,
        DEPTH_SCALE: depth_scale_path,
        CAMERA_JSON: camera_json_path,
        CAMERA_DATA_JSON: camera_data_path,
    }


def write_legacy_rgbd_frame(
    output_path: str | Path,
    *,
    rgb_image: Any,
    depth_image: Any,
    sensor_type: SensorType | str,
    sensor_id: str,
    frame_index: int,
    sensor_timestamp_ns: int | None,
    host_received_timestamp_ns: int,
    host_wall_timestamp_ns: int | None = None,
    depth_sensor_timestamp_ns: int | None = None,
    frame_stem: str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one RGB-D pair and append its metadata sidecar record."""

    output = ensure_legacy_rgbd_folders(output_path)
    wall_timestamp = host_wall_timestamp_ns if host_wall_timestamp_ns else time.time_ns()
    stem = frame_stem or frame_stem_from_host_wall_ns(wall_timestamp)
    frame_id = f"{stem}.png"
    rgb_path = output / RGB_DIR / frame_id
    depth_path = output / DEPTH_DIR / frame_id

    _write_png(rgb_path, rgb_image)
    _write_png(depth_path, depth_image)

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sensor_type": _sensor_type_value(sensor_type),
        "sensor_id": sensor_id,
        "frame_index": int(frame_index),
        "frame_id": frame_id,
        "rgb_path": f"{RGB_DIR}/{frame_id}",
        "depth_path": f"{DEPTH_DIR}/{frame_id}",
        "sensor_timestamp_ns": sensor_timestamp_ns,
        "host_received_timestamp_ns": int(host_received_timestamp_ns),
        "host_wall_timestamp_ns": int(wall_timestamp),
    }
    if depth_sensor_timestamp_ns is not None:
        metadata["depth_sensor_timestamp_ns"] = int(depth_sensor_timestamp_ns)
    if extra_metadata:
        metadata.update(dict(extra_metadata))

    append_frame_metadata(output, metadata)
    return metadata


def write_aligned_rgbd_frame(
    output_path: str | Path,
    frame: AlignedRgbdFrame,
    *,
    host_wall_timestamp_ns: int | None = None,
    frame_stem: str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write an `AlignedRgbdFrame` through the legacy capture folder contract."""

    metadata = dict(frame.exposure_metadata)
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    return write_legacy_rgbd_frame(
        output_path,
        rgb_image=frame.rgb_image,
        depth_image=frame.depth_image_aligned_to_rgb,
        sensor_type=frame.sensor_type,
        sensor_id=frame.sensor_id,
        frame_index=frame.frame_index,
        sensor_timestamp_ns=frame.sensor_timestamp_ns,
        host_received_timestamp_ns=frame.host_received_timestamp_ns,
        host_wall_timestamp_ns=host_wall_timestamp_ns,
        frame_stem=frame_stem,
        extra_metadata=metadata,
    )
