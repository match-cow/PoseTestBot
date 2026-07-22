"""Calibration observation extraction from synchronized ArUco results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from posetestbot.io.atomic import atomic_write_json
from posetestbot.calibration.profiles import sensor_identity_from_folder_name
from posetestbot.calibration.targets import (
    normalize_calibration_target_spec,
    target_identity,
    validate_target_identity,
)
from posetestbot.io.artifacts import (
    ARUCO_POSE_ESTIMATION,
    CALIBRATION_OBSERVATIONS,
    CALIBRATION_TARGET_POSE_ESTIMATION,
    CHARUCO_POSE_ESTIMATION,
    CHECKERBOARD_POSE_ESTIMATION,
    PROCESSED_DIR,
    SYNCHRONIZED_DIR,
)
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    upsert_stage,
    write_run_manifest,
)
from posetestbot.pipeline.run_config import load_run_config_for_run_root
from posetestbot.pipeline.sensor_selection import filter_enabled_sensor_folders
from posetestbot.sensors.registry import sensor_folder_name


SCHEMA_VERSION = "calibration_observations.v1"
TARGET_SOURCE_FILES = {
    "aruco_grid": (ARUCO_POSE_ESTIMATION, CALIBRATION_TARGET_POSE_ESTIMATION),
    "charuco": (
        CHARUCO_POSE_ESTIMATION,
        CALIBRATION_TARGET_POSE_ESTIMATION,
        ARUCO_POSE_ESTIMATION,
    ),
    "checkerboard": (
        CHECKERBOARD_POSE_ESTIMATION,
        CALIBRATION_TARGET_POSE_ESTIMATION,
        ARUCO_POSE_ESTIMATION,
    ),
}
TARGET_POSE_KEYS = {
    "aruco_grid": ("aruco_pose_estimation", "target_pose_estimation"),
    "charuco": (
        "charuco_pose_estimation",
        "target_pose_estimation",
        "aruco_pose_estimation",
    ),
    "checkerboard": (
        "checkerboard_pose_estimation",
        "target_pose_estimation",
        "aruco_pose_estimation",
    ),
}


def _generated_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check(
    name: str,
    status: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "message": message,
        "details": dict(details or {}),
    }


def _overall_status(checks: list[Mapping[str, Any]]) -> str:
    statuses = {str(check.get("status")) for check in checks}
    if "error" in statuses:
        return "error"
    if "warning" in statuses:
        return "warning"
    return "ok"


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(
            f"Calibration observation source must be a JSON object: {path}"
        )
    return value


def discover_aruco_outputs(run_root: str | Path) -> list[Path]:
    root = Path(run_root)
    aruco_root = root / PROCESSED_DIR / SYNCHRONIZED_DIR
    if not aruco_root.is_dir():
        return []
    folders = filter_enabled_sensor_folders(
        root,
        (path for path in sorted(aruco_root.iterdir()) if path.is_dir()),
    )
    return [
        folder / ARUCO_POSE_ESTIMATION
        for folder in folders
        if (folder / ARUCO_POSE_ESTIMATION).is_file()
    ]


def discover_calibration_pose_outputs(
    run_root: str | Path,
    *,
    target_type: str = "aruco_grid",
) -> list[Path]:
    root = Path(run_root)
    synchronized_root = root / PROCESSED_DIR / SYNCHRONIZED_DIR
    if not synchronized_root.is_dir():
        return []
    filenames = TARGET_SOURCE_FILES.get(target_type, TARGET_SOURCE_FILES["aruco_grid"])
    folders = filter_enabled_sensor_folders(
        root,
        (path for path in sorted(synchronized_root.iterdir()) if path.is_dir()),
    )
    paths: list[Path] = []
    seen: set[Path] = set()
    for filename in filenames:
        for folder in folders:
            path = folder / filename
            if not path.is_file():
                continue
            if path in seen:
                continue
            paths.append(path)
            seen.add(path)
    return paths


def _run_config_sensor_map(run_root: Path) -> dict[str, dict[str, Any]]:
    try:
        config = load_run_config_for_run_root(run_root)
    except (FileNotFoundError, ValueError):
        return {}

    mapped: dict[str, dict[str, Any]] = {}
    for sensor in config.get("capture", {}).get("sensors", []):
        if not isinstance(sensor, Mapping):
            continue
        try:
            folder = sensor_folder_name(
                sensor["sensor_type"],
                str(sensor["device_id"]),
            )
        except (KeyError, ValueError):
            continue
        mapped[folder] = dict(sensor)
    return mapped


def _sensor_metadata(
    sensor_name: str, config_sensors: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    sensor_type, device_id = sensor_identity_from_folder_name(sensor_name)
    config = dict(config_sensors.get(sensor_name, {}))
    return {
        "sensor_name": sensor_name,
        "sensor_type": (
            str(
                config.get("sensor_type")
                or (sensor_type.value if sensor_type else "unknown")
            )
        ),
        "device_id": str(config.get("device_id") or device_id),
        "mounting_mode": config.get("mounting_mode"),
        "display_name": config.get("display_name"),
        "calibration_profile_id": config.get("calibration_profile_id"),
    }


def _vector(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _pose_source(
    frame: Mapping[str, Any],
    *,
    target_type: str,
) -> tuple[str | None, Mapping[str, Any] | None]:
    for key in TARGET_POSE_KEYS.get(target_type, TARGET_POSE_KEYS["aruco_grid"]):
        value = frame.get(key)
        if isinstance(value, Mapping):
            return key, value
    return None, None


def _feature_count(pose: Mapping[str, Any]) -> int:
    for key in (
        "feature_count",
        "marker_count",
        "corner_count",
        "len_ids",
        "len_corners",
    ):
        if key not in pose or pose.get(key) is None:
            continue
        try:
            return int(pose.get(key) or 0)
        except (TypeError, ValueError):
            continue
    ids = pose.get("ids")
    if isinstance(ids, list):
        return len(ids)
    corners = pose.get("corners")
    if isinstance(corners, list):
        return len(corners)
    return 0


def _rejection_reason(
    frame: Mapping[str, Any],
    *,
    min_marker_count: int,
    target_type: str,
) -> str | None:
    _source_key, pose = _pose_source(frame, target_type=target_type)
    if not isinstance(pose, Mapping):
        return "missing_target_pose_estimation"
    feature_count = _feature_count(pose)
    if feature_count < min_marker_count:
        return (
            "insufficient_markers"
            if target_type == "aruco_grid"
            else "insufficient_target_features"
        )
    if _vector(pose.get("rvec")) is None:
        return "invalid_rvec"
    if _vector(pose.get("tvec")) is None:
        return "invalid_tvec"
    if not isinstance(frame.get("robot_ee_pose"), Mapping):
        return "missing_robot_ee_pose"
    return None


def _observation(
    *,
    sensor: Mapping[str, Any],
    frame_id: str,
    frame: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    source_key, pose = _pose_source(frame, target_type=str(target["target_type"]))
    if pose is None or source_key is None:
        raise ValueError("frame does not contain a usable target pose estimation")
    validate_target_identity(
        pose.get("target"), target, label=f"Pose evidence for {frame_id}"
    )
    return {
        "observation_id": f"{sensor['sensor_name']}:{frame_id}",
        "sensor_name": sensor["sensor_name"],
        "sensor_type": sensor["sensor_type"],
        "device_id": sensor["device_id"],
        "mounting_mode": sensor.get("mounting_mode"),
        "frame_id": frame_id,
        "motion": frame.get("motion"),
        "image_frame": frame.get("image_frame"),
        "source_rgb": frame.get("source_rgb"),
        "source_depth": frame.get("source_depth"),
        "synchronized_rgb": frame.get("synchronized_rgb"),
        "synchronized_depth": frame.get("synchronized_depth"),
        "nearest_robot_delta_ns": frame.get("nearest_robot_delta_ns"),
        "robot_ee_pose": dict(frame["robot_ee_pose"]),
        "target_type": target["target_type"],
        **target_identity(target),
        "target_pose_source": source_key,
        "target_to_camera": {
            "rotation_vector_rodrigues": _vector(pose.get("rvec")),
            "translation": _vector(pose.get("tvec")),
            "unit": target["unit"],
            "convention": "opencv_solvepnp_object_to_camera",
        },
        "feature_count": _feature_count(pose),
        "marker_count": _feature_count(pose),
    }


def _sensor_summary(
    *,
    sensor: Mapping[str, Any],
    source_path: Path,
    root: Path,
    frame_count: int,
    observation_count: int,
    rejected_count: int,
    motions: set[str],
) -> dict[str, Any]:
    return {
        **dict(sensor),
        "calibration_pose_file": _relative(source_path, root),
        "aruco_pose_file": _relative(source_path, root),
        "source_filename": source_path.name,
        "frame_count": frame_count,
        "observation_count": observation_count,
        "rejected_count": rejected_count,
        "motions": sorted(motions),
    }


def build_calibration_observations(
    run_root: str | Path,
    *,
    min_marker_count: int = 4,
    min_observations: int = 6,
    aruco_paths: list[str | Path] | None = None,
    target: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build solver-ready calibration observations from target-pose outputs."""

    if min_marker_count < 1:
        raise ValueError("min_marker_count must be at least 1")
    if min_observations < 0:
        raise ValueError("min_observations cannot be negative")
    target_spec = normalize_calibration_target_spec(target)

    root = Path(run_root)
    paths = (
        [Path(path) for path in aruco_paths]
        if aruco_paths is not None
        else discover_calibration_pose_outputs(
            root,
            target_type=str(target_spec["target_type"]),
        )
    )
    config_sensors = _run_config_sensor_map(root)
    checks: list[dict[str, Any]] = []
    sensors: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    if not paths:
        checks.append(
            _check(
                "calibration_pose_outputs_present",
                "error",
                "No synchronized calibration target pose files were found.",
                details={
                    "expected_root": (
                        root / PROCESSED_DIR / SYNCHRONIZED_DIR
                    ).as_posix(),
                    "target_type": target_spec["target_type"],
                    "expected_filenames": list(
                        TARGET_SOURCE_FILES.get(
                            str(target_spec["target_type"]),
                            TARGET_SOURCE_FILES["aruco_grid"],
                        )
                    ),
                },
            )
        )
    else:
        checks.append(
            _check(
                "calibration_pose_outputs_present",
                "ok",
                f"Found {len(paths)} calibration target pose file(s).",
                details={
                    "file_count": len(paths),
                    "target_type": target_spec["target_type"],
                    "filenames": sorted({path.name for path in paths}),
                },
            )
        )

    for raw_path in paths:
        path = raw_path if raw_path.is_absolute() else root / raw_path
        sensor_name = path.parent.name
        sensor = _sensor_metadata(sensor_name, config_sensors)
        try:
            frames = _read_json(path)
        except Exception as exc:
            checks.append(
                _check(
                    f"aruco_output_load:{_relative(path, root)}",
                    "error",
                    (
                        "Could not load calibration target pose output "
                        f"{path}: {type(exc).__name__}: {exc}"
                    ),
                    details={"path": path.as_posix()},
                )
            )
            continue

        sensor_observations = 0
        sensor_rejections = 0
        motions: set[str] = set()
        for frame_id, frame in sorted(frames.items()):
            if not isinstance(frame, Mapping):
                rejected.append(
                    {
                        "sensor_name": sensor_name,
                        "frame_id": str(frame_id),
                        "reason": "invalid_frame_record",
                    }
                )
                sensor_rejections += 1
                continue
            motion = frame.get("motion")
            if isinstance(motion, str):
                motions.add(motion)
            source_key, pose = _pose_source(
                frame,
                target_type=str(target_spec["target_type"]),
            )
            reason = _rejection_reason(
                frame,
                min_marker_count=min_marker_count,
                target_type=str(target_spec["target_type"]),
            )
            if reason is not None:
                rejected.append(
                    {
                        "sensor_name": sensor_name,
                        "frame_id": str(frame_id),
                        "reason": reason,
                        "target_pose_source": source_key,
                        "marker_count": (
                            _feature_count(pose) if isinstance(pose, Mapping) else None
                        ),
                        "feature_count": (
                            _feature_count(pose) if isinstance(pose, Mapping) else None
                        ),
                    }
                )
                sensor_rejections += 1
                continue
            observations.append(
                _observation(
                    sensor=sensor,
                    frame_id=str(frame_id),
                    frame=frame,
                    target=target_spec,
                )
            )
            sensor_observations += 1

        sensors.append(
            _sensor_summary(
                sensor=sensor,
                source_path=path,
                root=root,
                frame_count=len(frames),
                observation_count=sensor_observations,
                rejected_count=sensor_rejections,
                motions=motions,
            )
        )
        checks.append(
            _check(
                f"calibration_observations:{sensor_name}",
                "ok" if sensor_observations >= min_observations else "warning",
                (
                    f"{sensor_name} has {sensor_observations} calibration observation(s)."
                    if sensor_observations >= min_observations
                    else (
                        f"{sensor_name} has {sensor_observations} calibration "
                        f"observation(s); recommended minimum is {min_observations}."
                    )
                ),
                details={
                    "sensor_name": sensor_name,
                    "observation_count": sensor_observations,
                    "min_observations": min_observations,
                    "rejected_count": sensor_rejections,
                    "source_filename": path.name,
                },
            )
        )

    if paths and not observations:
        checks.append(
            _check(
                "calibration_observations_present",
                "error",
                "No usable calibration observations were found.",
                details={
                    "min_marker_count": min_marker_count,
                    "target_type": target_spec["target_type"],
                },
            )
        )

    motion_names = sorted(
        {
            observation["motion"]
            for observation in observations
            if isinstance(observation.get("motion"), str)
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_root": root.as_posix(),
        "overall_status": _overall_status(checks),
        "target": dict(target_spec),
        "board": dict(target_spec),
        "min_marker_count": min_marker_count,
        "min_observations": min_observations,
        "source_file_count": len(paths),
        "sensor_count": len(sensors),
        "frame_count": sum(int(sensor["frame_count"]) for sensor in sensors),
        "observation_count": len(observations),
        "rejected_count": len(rejected),
        "motion_count": len(motion_names),
        "motions": motion_names,
        "checks": checks,
        "sensors": sensors,
        "observations": observations,
        "rejected": rejected,
    }


def calibration_observations_path(
    run_root: str | Path,
    *,
    output_root: str | Path | None = None,
) -> Path:
    destination = Path(output_root) if output_root is not None else Path(run_root)
    return destination / CALIBRATION_OBSERVATIONS


def write_calibration_observations(
    run_root: str | Path,
    report: Mapping[str, Any],
    *,
    output_root: str | Path | None = None,
) -> Path:
    path = calibration_observations_path(run_root, output_root=output_root)
    return atomic_write_json(path, dict(report))


def write_calibration_observations_with_manifest(
    run_root: str | Path,
    *,
    min_marker_count: int = 4,
    min_observations: int = 6,
    aruco_paths: list[str | Path] | None = None,
    output_root: str | Path | None = None,
    target: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    run_root_path = Path(run_root)
    manifest = load_or_create_run_manifest(run_root_path)
    upsert_stage(manifest, name="calibration_observations", status="running")
    write_run_manifest(manifest, run_root_path)
    try:
        report = build_calibration_observations(
            run_root_path,
            min_marker_count=min_marker_count,
            min_observations=min_observations,
            aruco_paths=aruco_paths,
            target=target,
        )
        path = write_calibration_observations(
            run_root_path,
            report,
            output_root=output_root,
        )
        upsert_stage(
            manifest,
            name="calibration_observations",
            status="succeeded" if report["overall_status"] != "error" else "failed",
            artifacts={CALIBRATION_OBSERVATIONS: path},
            run_root=run_root_path,
            message=f"Calibration observation status: {report['overall_status']}.",
        )
        write_run_manifest(manifest, run_root_path)
    except Exception as exc:
        upsert_stage(
            manifest,
            name="calibration_observations",
            status="failed",
            message=str(exc),
        )
        write_run_manifest(manifest, run_root_path)
        raise
    return path, report
