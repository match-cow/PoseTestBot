"""Generate validation-gated calibration profile candidates from observations."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

from posetestbot.calibration.observations import SCHEMA_VERSION as OBSERVATION_SCHEMA
from posetestbot.calibration.profiles import (
    CalibrationProfile,
    CalibrationQuality,
    CalibrationStatus,
    CalibrationTargetType,
    RigidTransform,
    TransformFrame,
    profile_from_dict,
    profile_to_dict,
    write_profile_collection,
)
from posetestbot.io.artifacts import (
    CALIBRATION_CANDIDATES,
    CALIBRATION_OBSERVATIONS,
    CALIBRATION_PROFILES_FROM_OBSERVATIONS,
    CAM_K,
    DEPTH_SCALE,
)
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    upsert_stage,
    write_run_manifest,
)
from posetestbot.sensors.contracts import CameraIntrinsics, MountingMode, SensorType


SCHEMA_VERSION = "calibration_candidates.v1"
DEFAULT_MAX_TRANSLATION_RESIDUAL_MM = 50.0
DEFAULT_MAX_ROTATION_RESIDUAL_DEG = 15.0
MAX_INLIER_REFINEMENT_ITERATIONS = 4
DEFAULT_TARGET_TO_REFERENCE = {
    "from": "calibration_target",
    "to": "robot_base",
    "rotation_quaternion_wxyz": [0.0, 1.0, 0.0, 0.0],
    "translation_mm": [-199.5, 137.0, 0.0],
    "unit": "mm",
    "source": "legacy_aruco_grid_template_default",
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


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Calibration candidate source must be a JSON object: {path}")
    return value


def _target_transform(value: Mapping[str, Any] | None = None) -> dict[str, Any]:
    data = dict(DEFAULT_TARGET_TO_REFERENCE if value is None else value)
    rotation = data.get("rotation_quaternion_wxyz")
    translation = data.get("translation_mm")
    if not isinstance(rotation, list) or len(rotation) != 4:
        raise ValueError("target rotation_quaternion_wxyz must have 4 values")
    if not isinstance(translation, list) or len(translation) != 3:
        raise ValueError("target translation_mm must have 3 values")
    data["rotation_quaternion_wxyz"] = [float(item) for item in rotation]
    data["translation_mm"] = [float(item) for item in translation]
    data["from"] = str(data.get("from", "calibration_target"))
    data["to"] = str(data.get("to", "robot_base"))
    data["unit"] = str(data.get("unit", "mm"))
    return data


def _transform_from_quaternion_translation(
    *,
    rotation_quaternion_wxyz: list[float],
    translation_mm: list[float],
) -> np.ndarray:
    return pt.transform_from(
        pr.matrix_from_quaternion(np.array(rotation_quaternion_wxyz, dtype=float)),
        np.array(translation_mm, dtype=float),
    )


def _robot_ee_to_reference(robot_ee_pose: Mapping[str, Any]) -> np.ndarray:
    try:
        rotation = pr.matrix_from_euler(
            np.array(
                [
                    float(robot_ee_pose["C"]),
                    float(robot_ee_pose["B"]),
                    float(robot_ee_pose["A"]),
                ]
            ),
            0,
            1,
            2,
            True,
        )
        translation = [
            float(robot_ee_pose["X"]),
            float(robot_ee_pose["Y"]),
            float(robot_ee_pose["Z"]),
        ]
    except KeyError as exc:
        raise ValueError("robot_ee_pose must include X, Y, Z, A, B, C") from exc
    return pt.transform_from(rotation, translation)


def _target_to_camera(observation: Mapping[str, Any]) -> np.ndarray:
    target = observation.get("target_to_camera")
    if not isinstance(target, Mapping):
        raise ValueError("observation is missing target_to_camera")
    rvec = target.get("rotation_vector_rodrigues")
    tvec = target.get("translation")
    if not isinstance(rvec, list) or len(rvec) != 3:
        raise ValueError("target_to_camera rotation_vector_rodrigues must have 3 values")
    if not isinstance(tvec, list) or len(tvec) != 3:
        raise ValueError("target_to_camera translation must have 3 values")
    return pt.transform_from(
        pr.matrix_from_compact_axis_angle(np.array(rvec, dtype=float)),
        np.array([float(item) for item in tvec], dtype=float),
    )


def _candidate_transform(
    observation: Mapping[str, Any],
    *,
    target_to_reference: np.ndarray,
    mounting_mode: MountingMode,
) -> np.ndarray:
    tm = TransformManager()
    tm.add_transform("calibration_target", "robot_base", target_to_reference)
    tm.add_transform("calibration_target", "camera", _target_to_camera(observation))
    if mounting_mode == MountingMode.EYE_IN_HAND:
        robot_pose = observation.get("robot_ee_pose")
        if not isinstance(robot_pose, Mapping):
            raise ValueError("eye-in-hand observations require robot_ee_pose")
        tm.add_transform(
            "end_effector",
            "robot_base",
            _robot_ee_to_reference(robot_pose),
        )
        return tm.get_transform("camera", "end_effector")
    return tm.get_transform("camera", "robot_base")


def _average_quaternions(quaternions: np.ndarray) -> np.ndarray:
    accumulator = np.zeros((4, 4), dtype=float)
    for quaternion in quaternions:
        q = np.array(quaternion, dtype=float)
        if q[0] < 0:
            q = -q
        accumulator += np.outer(q, q)
    accumulator /= len(quaternions)
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    return eigenvectors[:, np.argmax(eigenvalues)]


def _average_transform(transforms: list[np.ndarray]) -> np.ndarray:
    translations = []
    quaternions = []
    for transform in transforms:
        x, y, z, qw, qx, qy, qz = pt.pq_from_transform(transform)
        translations.append([x, y, z])
        quaternions.append([qw, qx, qy, qz])
    translation = np.mean(np.array(translations, dtype=float), axis=0)
    quaternion = _average_quaternions(np.array(quaternions, dtype=float))
    return pt.transform_from(pr.matrix_from_quaternion(quaternion), translation)


def _median_seed_transform(transforms: list[np.ndarray]) -> np.ndarray:
    translations = []
    quaternions = []
    for transform in transforms:
        x, y, z, qw, qx, qy, qz = pt.pq_from_transform(transform)
        translations.append([x, y, z])
        quaternions.append([qw, qx, qy, qz])
    translation = np.median(np.array(translations, dtype=float), axis=0)
    quaternion = _average_quaternions(np.array(quaternions, dtype=float))
    return pt.transform_from(pr.matrix_from_quaternion(quaternion), translation)


def _rotation_delta_deg(left: np.ndarray, right: np.ndarray) -> float:
    delta = left[:3, :3].T @ right[:3, :3]
    cosine = max(-1.0, min(1.0, (float(np.trace(delta)) - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def _residual_records(
    transforms: list[np.ndarray], average: np.ndarray
) -> list[dict[str, float]]:
    return [
        {
            "translation_mm": float(np.linalg.norm(transform[:3, 3] - average[:3, 3])),
            "rotation_deg": _rotation_delta_deg(average, transform),
        }
        for transform in transforms
    ]


def _residual_summary_from_records(
    records: list[Mapping[str, float]],
) -> dict[str, float]:
    translation = [float(record["translation_mm"]) for record in records]
    rotation = [float(record["rotation_deg"]) for record in records]
    return {
        "mean_translation_mm": float(np.mean(translation)) if translation else 0.0,
        "max_translation_mm": float(np.max(translation)) if translation else 0.0,
        "mean_rotation_deg": float(np.mean(rotation)) if rotation else 0.0,
        "max_rotation_deg": float(np.max(rotation)) if rotation else 0.0,
    }


def _within_residual_thresholds(
    record: Mapping[str, float],
    *,
    max_translation_residual_mm: float | None,
    max_rotation_residual_deg: float | None,
) -> bool:
    if (
        max_translation_residual_mm is not None
        and float(record["translation_mm"]) > max_translation_residual_mm
    ):
        return False
    if (
        max_rotation_residual_deg is not None
        and float(record["rotation_deg"]) > max_rotation_residual_deg
    ):
        return False
    return True


def _select_inliers(
    transforms: list[np.ndarray],
    *,
    max_translation_residual_mm: float | None,
    max_rotation_residual_deg: float | None,
) -> tuple[np.ndarray, list[dict[str, float]], list[bool]]:
    if max_translation_residual_mm is None and max_rotation_residual_deg is None:
        average = _average_transform(transforms)
        return average, _residual_records(transforms, average), [True] * len(transforms)

    average = _median_seed_transform(transforms)
    inliers: list[bool] | None = None
    residuals: list[dict[str, float]] = []
    for _iteration in range(MAX_INLIER_REFINEMENT_ITERATIONS):
        residuals = _residual_records(transforms, average)
        next_inliers = [
            _within_residual_thresholds(
                residual,
                max_translation_residual_mm=max_translation_residual_mm,
                max_rotation_residual_deg=max_rotation_residual_deg,
            )
            for residual in residuals
        ]
        if not any(next_inliers):
            return average, residuals, next_inliers
        if next_inliers == inliers:
            return average, residuals, next_inliers
        inliers = next_inliers
        average = _average_transform(
            [
                transform
                for transform, is_inlier in zip(transforms, inliers, strict=True)
                if is_inlier
            ]
        )

    residuals = _residual_records(transforms, average)
    final_inliers = [
        _within_residual_thresholds(
            residual,
            max_translation_residual_mm=max_translation_residual_mm,
            max_rotation_residual_deg=max_rotation_residual_deg,
        )
        for residual in residuals
    ]
    return average, residuals, final_inliers


def _safe_profile_id(sensor_name: str, mounting_mode: MountingMode) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", sensor_name).strip("_")
    return f"{slug}_{mounting_mode.value}_aruco_candidate"


def _intrinsics_from_sensor_folder(sensor_folder: Path) -> CameraIntrinsics:
    cam_k = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    if (sensor_folder / CAM_K).is_file():
        rows = []
        for line in (sensor_folder / CAM_K).read_text().splitlines()[:3]:
            rows.extend(float(item) for item in line.split())
        if len(rows) == 9:
            cam_k = tuple(rows)  # type: ignore[assignment]
    depth_scale = 1.0
    if (sensor_folder / DEPTH_SCALE).is_file():
        try:
            depth_scale = float((sensor_folder / DEPTH_SCALE).read_text().strip())
        except ValueError:
            depth_scale = 1.0
    return CameraIntrinsics(cam_k=cam_k, width=0, height=0, depth_scale_to_mm=depth_scale)


def _profile_from_average(
    *,
    sensor: Mapping[str, Any],
    average: np.ndarray,
    residuals: Mapping[str, float],
    observation_count: int,
    inlier_count: int,
    sensor_folder: Path,
    mounting_mode: MountingMode,
) -> CalibrationProfile:
    x, y, z, qw, qx, qy, qz = pt.pq_from_transform(average)
    to_frame = (
        TransformFrame.END_EFFECTOR
        if mounting_mode == MountingMode.EYE_IN_HAND
        else TransformFrame.ROBOT_BASE
    )
    sensor_name = str(sensor["sensor_name"])
    sensor_type = SensorType(str(sensor["sensor_type"]))
    device_id = str(sensor.get("device_id") or sensor_name)
    return CalibrationProfile(
        schema_version="calibration.v1",
        profile_id=_safe_profile_id(sensor_name, mounting_mode),
        sensor_id=device_id,
        sensor_type=sensor_type,
        mounting_mode=mounting_mode,
        rig_position="wrist" if mounting_mode == MountingMode.EYE_IN_HAND else "static",
        intrinsics=_intrinsics_from_sensor_folder(sensor_folder),
        extrinsics=RigidTransform(
            from_frame=TransformFrame.CAMERA,
            to_frame=to_frame,
            rotation_quaternion_wxyz=(float(qw), float(qx), float(qy), float(qz)),
            translation_mm=(float(x), float(y), float(z)),
        ),
        target_type=CalibrationTargetType.ARUCO_GRID,
        calibration_dataset_id=None,
        method="aruco_observation_transform_average",
        status=CalibrationStatus.NEEDS_VALIDATION,
        quality=CalibrationQuality(
            num_observations=observation_count,
            num_inliers=inlier_count,
            residual_translation_mm=float(residuals["mean_translation_mm"]),
            residual_rotation_deg=float(residuals["mean_rotation_deg"]),
            notes=(
                "Candidate generated from ArUco observations with residual "
                "threshold outlier filtering; validate before use."
            ),
        ),
        sync_delta_ms=None,
        metadata={
            "sensor_name": sensor_name,
            "candidate_source": CALIBRATION_OBSERVATIONS,
            "inlier_count": inlier_count,
            "outlier_count": observation_count - inlier_count,
            "max_residual_translation_mm": residuals["max_translation_mm"],
            "max_residual_rotation_deg": residuals["max_rotation_deg"],
        },
    )


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _observations_by_sensor(report: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for observation in report.get("observations", []):
        if not isinstance(observation, Mapping):
            continue
        sensor_name = str(observation.get("sensor_name", ""))
        if not sensor_name:
            continue
        grouped.setdefault(sensor_name, []).append(observation)
    return grouped


def _sensor_by_name(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    sensors = {}
    for sensor in report.get("sensors", []):
        if isinstance(sensor, Mapping) and sensor.get("sensor_name"):
            sensors[str(sensor["sensor_name"])] = sensor
    return sensors


def build_calibration_candidates(
    run_root: str | Path,
    *,
    observations_path: str | Path | None = None,
    min_observations: int = 6,
    target_to_reference: Mapping[str, Any] | None = None,
    max_translation_residual_mm: float | None = DEFAULT_MAX_TRANSLATION_RESIDUAL_MM,
    max_rotation_residual_deg: float | None = DEFAULT_MAX_ROTATION_RESIDUAL_DEG,
) -> dict[str, Any]:
    if min_observations < 1:
        raise ValueError("min_observations must be at least 1")
    if max_translation_residual_mm is not None and max_translation_residual_mm < 0:
        raise ValueError("max_translation_residual_mm must be greater than or equal to 0")
    if max_rotation_residual_deg is not None and max_rotation_residual_deg < 0:
        raise ValueError("max_rotation_residual_deg must be greater than or equal to 0")

    root = Path(run_root)
    source_path = (
        Path(observations_path)
        if observations_path is not None
        else root / CALIBRATION_OBSERVATIONS
    )
    if not source_path.is_absolute():
        source_path = root / source_path
    observations_report = _read_json(source_path)
    if observations_report.get("schema_version") != OBSERVATION_SCHEMA:
        raise ValueError(
            "Unsupported calibration observation schema: "
            f"{observations_report.get('schema_version')!r}"
        )

    target = _target_transform(target_to_reference)
    target_transform = _transform_from_quaternion_translation(
        rotation_quaternion_wxyz=target["rotation_quaternion_wxyz"],
        translation_mm=target["translation_mm"],
    )
    by_sensor = _observations_by_sensor(observations_report)
    sensor_metadata = _sensor_by_name(observations_report)
    checks: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    residuals_by_sensor: list[dict[str, Any]] = []
    total_inlier_count = 0
    total_outlier_count = 0

    if not by_sensor:
        checks.append(
            _check(
                "calibration_observations_present",
                "error",
                "No usable calibration observations were found.",
                details={"path": source_path.as_posix()},
            )
        )

    for sensor_name, observations in sorted(by_sensor.items()):
        sensor = dict(sensor_metadata.get(sensor_name, {}))
        sensor.setdefault("sensor_name", sensor_name)
        sensor.setdefault("sensor_type", observations[0].get("sensor_type"))
        sensor.setdefault("device_id", observations[0].get("device_id"))
        mounting_mode = MountingMode(
            str(sensor.get("mounting_mode") or MountingMode.EYE_IN_HAND.value)
        )
        if len(observations) < min_observations:
            checks.append(
                _check(
                    f"candidate_observations:{sensor_name}",
                    "warning",
                    (
                        f"{sensor_name} has {len(observations)} observation(s); "
                        f"recommended minimum is {min_observations}."
                    ),
                    details={
                        "sensor_name": sensor_name,
                        "observation_count": len(observations),
                        "min_observations": min_observations,
                    },
                )
            )
        else:
            checks.append(
                _check(
                    f"candidate_observations:{sensor_name}",
                    "ok",
                    f"{sensor_name} has {len(observations)} observation(s).",
                    details={
                        "sensor_name": sensor_name,
                        "observation_count": len(observations),
                    },
                )
            )

        transforms = []
        for observation in observations:
            transform = _candidate_transform(
                observation,
                target_to_reference=target_transform,
                mounting_mode=mounting_mode,
            )
            transforms.append(transform)

        average, residual_records, inlier_mask = _select_inliers(
            transforms,
            max_translation_residual_mm=max_translation_residual_mm,
            max_rotation_residual_deg=max_rotation_residual_deg,
        )
        inlier_transforms = [
            transform
            for transform, is_inlier in zip(transforms, inlier_mask, strict=True)
            if is_inlier
        ]
        inlier_residuals = [
            residual
            for residual, is_inlier in zip(residual_records, inlier_mask, strict=True)
            if is_inlier
        ]
        inlier_count = len(inlier_transforms)
        outlier_count = len(transforms) - inlier_count
        total_inlier_count += inlier_count
        total_outlier_count += outlier_count

        for observation, transform, residual, is_inlier in zip(
            observations, transforms, residual_records, inlier_mask, strict=True
        ):
            x, y, z, qw, qx, qy, qz = pt.pq_from_transform(transform)
            candidates.append(
                {
                    "observation_id": observation.get("observation_id"),
                    "sensor_name": sensor_name,
                    "mounting_mode": mounting_mode.value,
                    "inlier": bool(is_inlier),
                    "residual_translation_mm": float(residual["translation_mm"]),
                    "residual_rotation_deg": float(residual["rotation_deg"]),
                    "from": "camera",
                    "to": (
                        "end_effector"
                        if mounting_mode == MountingMode.EYE_IN_HAND
                        else "robot_base"
                    ),
                    "translation_mm": [float(x), float(y), float(z)],
                    "rotation_quaternion_wxyz": [
                        float(qw),
                        float(qx),
                        float(qy),
                        float(qz),
                    ],
                }
            )

        all_residuals = _residual_summary_from_records(residual_records)
        residuals = _residual_summary_from_records(inlier_residuals)
        if inlier_count == 0:
            checks.append(
                _check(
                    f"candidate_inliers:{sensor_name}",
                    "error",
                    (
                        f"{sensor_name} has no inlier calibration candidate "
                        "transforms after residual threshold filtering."
                    ),
                    details={
                        "sensor_name": sensor_name,
                        "observation_count": len(observations),
                        "max_translation_residual_mm": max_translation_residual_mm,
                        "max_rotation_residual_deg": max_rotation_residual_deg,
                    },
                )
            )
            residuals_by_sensor.append(
                {
                    "sensor_name": sensor_name,
                    "mounting_mode": mounting_mode.value,
                    "observation_count": len(observations),
                    "inlier_count": 0,
                    "outlier_count": outlier_count,
                    **dict(residuals),
                    "all_mean_translation_mm": all_residuals["mean_translation_mm"],
                    "all_max_translation_mm": all_residuals["max_translation_mm"],
                    "all_mean_rotation_deg": all_residuals["mean_rotation_deg"],
                    "all_max_rotation_deg": all_residuals["max_rotation_deg"],
                }
            )
            continue

        if inlier_count < min_observations:
            checks.append(
                _check(
                    f"candidate_inliers:{sensor_name}",
                    "warning",
                    (
                        f"{sensor_name} has {inlier_count} inlier candidate "
                        f"transform(s); recommended minimum is {min_observations}."
                    ),
                    details={
                        "sensor_name": sensor_name,
                        "inlier_count": inlier_count,
                        "outlier_count": outlier_count,
                        "min_observations": min_observations,
                    },
                )
            )
        else:
            checks.append(
                _check(
                    f"candidate_inliers:{sensor_name}",
                    "ok",
                    f"{sensor_name} has {inlier_count} inlier candidate transform(s).",
                    details={
                        "sensor_name": sensor_name,
                        "inlier_count": inlier_count,
                        "outlier_count": outlier_count,
                    },
                )
            )

        if outlier_count:
            checks.append(
                _check(
                    f"candidate_outliers:{sensor_name}",
                    "warning",
                    (
                        f"{sensor_name} rejected {outlier_count} candidate "
                        "transform(s) by residual threshold."
                    ),
                    details={
                        "sensor_name": sensor_name,
                        "outlier_count": outlier_count,
                        "max_translation_residual_mm": max_translation_residual_mm,
                        "max_rotation_residual_deg": max_rotation_residual_deg,
                    },
                )
            )

        sensor_folder = source_path.parent / "processed" / "synchronized" / sensor_name
        if not sensor_folder.is_dir():
            sensor_folder = root / "processed" / "synchronized" / sensor_name
        profile = _profile_from_average(
            sensor=sensor,
            average=average,
            residuals=residuals,
            observation_count=len(observations),
            inlier_count=inlier_count,
            sensor_folder=sensor_folder,
            mounting_mode=mounting_mode,
        )
        profiles.append(profile_to_dict(profile))
        residuals_by_sensor.append(
            {
                "sensor_name": sensor_name,
                "mounting_mode": mounting_mode.value,
                "observation_count": len(observations),
                "inlier_count": inlier_count,
                "outlier_count": outlier_count,
                **dict(residuals),
                "all_mean_translation_mm": all_residuals["mean_translation_mm"],
                "all_max_translation_mm": all_residuals["max_translation_mm"],
                "all_mean_rotation_deg": all_residuals["mean_rotation_deg"],
                "all_max_rotation_deg": all_residuals["max_rotation_deg"],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_root": root.as_posix(),
        "source_observations": _relative(source_path, root),
        "target_to_reference": target,
        "residual_thresholds": {
            "max_translation_mm": max_translation_residual_mm,
            "max_rotation_deg": max_rotation_residual_deg,
        },
        "overall_status": _overall_status(checks),
        "min_observations": min_observations,
        "sensor_count": len(by_sensor),
        "profile_count": len(profiles),
        "candidate_count": len(candidates),
        "inlier_count": total_inlier_count,
        "outlier_count": total_outlier_count,
        "checks": checks,
        "profiles": profiles,
        "candidates": candidates,
        "residuals": residuals_by_sensor,
    }


def calibration_candidates_path(run_root: str | Path) -> Path:
    return Path(run_root) / CALIBRATION_CANDIDATES


def calibration_profiles_from_observations_path(run_root: str | Path) -> Path:
    return Path(run_root) / CALIBRATION_PROFILES_FROM_OBSERVATIONS


def write_calibration_candidates(
    run_root: str | Path,
    report: Mapping[str, Any],
) -> tuple[Path, Path]:
    root = Path(run_root)
    report_path = calibration_candidates_path(root)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(dict(report), f, indent=2, sort_keys=True)
        f.write("\n")

    profiles = [profile_from_dict(profile) for profile in report.get("profiles", [])]
    profile_path = calibration_profiles_from_observations_path(root)
    write_profile_collection(profiles, profile_path)
    return report_path, profile_path


def write_calibration_candidates_with_manifest(
    run_root: str | Path,
    *,
    observations_path: str | Path | None = None,
    min_observations: int = 6,
    target_to_reference: Mapping[str, Any] | None = None,
    max_translation_residual_mm: float | None = DEFAULT_MAX_TRANSLATION_RESIDUAL_MM,
    max_rotation_residual_deg: float | None = DEFAULT_MAX_ROTATION_RESIDUAL_DEG,
) -> tuple[Path, Path, dict[str, Any]]:
    run_root_path = Path(run_root)
    manifest = load_or_create_run_manifest(run_root_path)
    upsert_stage(manifest, name="calibration_candidates", status="running")
    write_run_manifest(manifest, run_root_path)
    try:
        report = build_calibration_candidates(
            run_root_path,
            observations_path=observations_path,
            min_observations=min_observations,
            target_to_reference=target_to_reference,
            max_translation_residual_mm=max_translation_residual_mm,
            max_rotation_residual_deg=max_rotation_residual_deg,
        )
        report_path, profiles_path = write_calibration_candidates(run_root_path, report)
        upsert_stage(
            manifest,
            name="calibration_candidates",
            status="succeeded" if report["overall_status"] != "error" else "failed",
            artifacts={
                CALIBRATION_CANDIDATES: report_path,
                CALIBRATION_PROFILES_FROM_OBSERVATIONS: profiles_path,
            },
            run_root=run_root_path,
            message=f"Calibration candidate status: {report['overall_status']}.",
        )
        write_run_manifest(manifest, run_root_path)
    except Exception as exc:
        upsert_stage(
            manifest,
            name="calibration_candidates",
            status="failed",
            message=str(exc),
        )
        write_run_manifest(manifest, run_root_path)
        raise
    return report_path, profiles_path, report
