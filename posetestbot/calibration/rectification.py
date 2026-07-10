"""Transactional non-destructive RGB/aligned-depth rectification."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from posetestbot.calibration.intrinsics import select_intrinsic_profile, sensor_intrinsic_identity
from posetestbot.io.atomic import atomic_write_json, atomic_write_text, replace_directory
from posetestbot.io.artifacts import (
    CAMERA_DATA_JSON,
    CAMERA_RECTIFICATION_REPORT,
    CAM_K,
    DEPTH_DIR,
    DEPTH_SCALE,
    FRAME_METADATA_JSONL,
    MATCH_ROBOT_EE_POSES,
    PROCESSED_DIR,
    RGB_DIR,
    SYNCHRONIZED_DIR,
)


SCHEMA_VERSION = "camera_rectification.v1"
RECTIFIED_DIR = "rectified"


def _pairs(sensor_folder: Path) -> list[tuple[Path, Path]]:
    rgb = {path.name: path for path in (sensor_folder / RGB_DIR).glob("*.png")}
    depth = {path.name: path for path in (sensor_folder / DEPTH_DIR).glob("*.png")}
    if not rgb or set(rgb) != set(depth):
        raise ValueError(f"RGB/depth filenames must be non-empty and identical: {sensor_folder}")
    return [(rgb[name], depth[name]) for name in sorted(rgb)]


def _maps(profile: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    native = profile["native"]
    rectified = profile["rectified"]
    image_size = tuple(int(item) for item in profile["resolution"])
    native_k = np.asarray(native["cam_K"], dtype=float).reshape(3, 3)
    distortion = np.asarray(native["distortion"], dtype=float).reshape(5)
    rectified_k = np.asarray(rectified["cam_K"], dtype=float).reshape(3, 3)
    map_x, map_y = cv2.initUndistortRectifyMap(
        native_k,
        distortion,
        None,
        rectified_k,
        image_size,
        cv2.CV_32FC1,
    )
    return map_x, map_y, rectified_k


def _copy_metadata(source: Path, destination: Path, profile_id: str) -> int:
    if not source.is_file():
        return 0
    records = []
    for line_number, line in enumerate(source.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"Frame metadata line {line_number} must be an object")
        record = dict(value)
        record["derivation"] = {
            "operation": "alpha0_camera_rectification",
            "intrinsic_profile_id": profile_id,
            "source_metadata": source.as_posix(),
            "rgb_interpolation": "linear",
            "depth_interpolation": "nearest",
            "invalid_depth_value": 0,
        }
        records.append(record)
    atomic_write_text(
        destination,
        "".join(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n" for record in records),
    )
    return len(records)


def _write_sidecars(
    destination: Path,
    profile: Mapping[str, Any],
    rectified_k: np.ndarray,
    *,
    source_sensor: Path,
) -> None:
    depth_scale = float(profile["depth"]["scale_to_mm"])
    matrix_rows = rectified_k.tolist()
    atomic_write_text(
        destination / CAM_K,
        "".join(" ".join(str(float(item)) for item in row) + "\n" for row in matrix_rows)
        + "0.0 0.0 0.0 0.0 0.0\n",
    )
    atomic_write_text(destination / DEPTH_SCALE, f"{depth_scale}\n")
    atomic_write_json(
        destination / CAMERA_DATA_JSON,
        {
            "K": matrix_rows,
            "resolution": [int(profile["resolution"][1]), int(profile["resolution"][0])],
            "distortion": [0.0] * 5,
            "projection": "rectified_alpha0",
            "valid_roi": profile["rectified"]["valid_roi"],
            "intrinsic_profile_id": profile["profile_id"],
            "source_sensor_folder": source_sensor.as_posix(),
            "depth_alignment": profile["depth"]["alignment"],
        },
    )


def rectify_sensor_folder(
    source_sensor: str | Path,
    destination_sensor: str | Path,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Rectify a sensor into an empty staging folder."""

    source = Path(source_sensor)
    destination = Path(destination_sensor)
    sensor_id, orientation, image_size = sensor_intrinsic_identity(source)
    expected = (str(profile["sensor_id"]), str(profile["orientation"]), tuple(profile["resolution"]))
    actual = (sensor_id, orientation, image_size)
    if actual != expected:
        raise ValueError(
            "Intrinsic profile serial/resolution/orientation mismatch: "
            f"captured={actual}, profile={expected}"
        )
    pairs = _pairs(source)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / RGB_DIR).mkdir()
    (destination / DEPTH_DIR).mkdir()
    map_x, map_y, rectified_k = _maps(profile)
    for rgb_path, depth_path in pairs:
        rgb = cv2.imread(rgb_path.as_posix(), cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(depth_path.as_posix(), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            raise ValueError(f"Unreadable RGB-D frame pair: {rgb_path.name}")
        if rgb.shape[:2] != depth.shape or (rgb.shape[1], rgb.shape[0]) != image_size:
            raise ValueError(f"RGB-D dimensions do not match intrinsic profile: {rgb_path.name}")
        rectified_rgb = cv2.remap(
            rgb,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        rectified_depth = cv2.remap(
            depth,
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if not cv2.imwrite((destination / RGB_DIR / rgb_path.name).as_posix(), rectified_rgb):
            raise OSError(f"Failed to write rectified RGB: {rgb_path.name}")
        if not cv2.imwrite((destination / DEPTH_DIR / depth_path.name).as_posix(), rectified_depth):
            raise OSError(f"Failed to write rectified depth: {depth_path.name}")

    for artifact in (MATCH_ROBOT_EE_POSES,):
        source_path = source / artifact
        if source_path.is_file():
            shutil.copy2(source_path, destination / artifact)
    metadata_count = _copy_metadata(
        source / FRAME_METADATA_JSONL,
        destination / FRAME_METADATA_JSONL,
        str(profile["profile_id"]),
    )
    _write_sidecars(destination, profile, rectified_k, source_sensor=source)
    atomic_write_json(
        destination / "rectification_provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "source_sensor_folder": source.as_posix(),
            "intrinsic_profile_id": profile["profile_id"],
            "projection": "rectified_alpha0",
            "rgb_interpolation": "linear",
            "depth_interpolation": "nearest",
            "invalid_depth_value": 0,
            "frame_count": len(pairs),
        },
    )
    return {
        "sensor_name": source.name,
        "sensor_id": sensor_id,
        "orientation": orientation,
        "resolution": list(image_size),
        "profile_id": profile["profile_id"],
        "frame_count": len(pairs),
        "metadata_record_count": metadata_count,
        "source": source.as_posix(),
        "output": destination.as_posix(),
    }


def rectify_run(
    run_root: str | Path,
    profiles: Sequence[Mapping[str, Any]],
    *,
    input_root: str | Path | None = None,
    output_root: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Build every rectified sensor in one staging tree, then promote it atomically."""

    root = Path(run_root)
    source_root = Path(input_root) if input_root else root / PROCESSED_DIR / SYNCHRONIZED_DIR
    destination_root = Path(output_root) if output_root else root / PROCESSED_DIR / RECTIFIED_DIR
    if destination_root.exists() and not overwrite:
        raise FileExistsError(f"Rectified output already exists: {destination_root}")
    sensors = [
        path
        for path in sorted(source_root.iterdir())
        if path.is_dir() and (path / RGB_DIR).is_dir() and (path / DEPTH_DIR).is_dir()
    ] if source_root.is_dir() else []
    if not sensors:
        raise FileNotFoundError(f"No synchronized RGB-D sensor folders: {source_root}")
    staging = destination_root.with_name(f".{destination_root.name}.{uuid.uuid4().hex}.tmp")
    staging.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        for sensor in sensors:
            sensor_id, orientation, resolution = sensor_intrinsic_identity(sensor)
            profile = select_intrinsic_profile(
                profiles,
                sensor_id=sensor_id,
                resolution=resolution,
                orientation=orientation,
            )
            records.append(rectify_sensor_folder(sensor, staging / sensor.name, profile))
        replace_directory(staging, destination_root)
        for record in records:
            record["output"] = (destination_root / str(record["sensor_name"])).as_posix()
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_root": root.as_posix(),
        "source_root": source_root.as_posix(),
        "output_root": destination_root.as_posix(),
        "projection": "rectified_alpha0",
        "sensor_count": len(records),
        "frame_count": sum(int(item["frame_count"]) for item in records),
        "sensors": records,
    }
    report_path = atomic_write_json(root / CAMERA_RECTIFICATION_REPORT, report)
    return report_path, report
