"""Calibration solvers built from synchronized ArUco observations."""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

from posetestbot.io.atomic import atomic_write_json
from posetestbot.calibration.candidates import (
    DEFAULT_MAX_ROTATION_RESIDUAL_DEG,
    DEFAULT_MAX_TRANSLATION_RESIDUAL_MM,
    _average_transform,
    _candidate_transform,
    _intrinsics_from_sensor_folder,
    _observations_by_sensor,
    _overall_status,
    _read_json,
    _relative,
    _residual_records,
    _residual_summary_from_records,
    _robot_ee_to_reference,
    _select_inliers,
    _sensor_by_name,
    _target_to_camera,
    _target_transform,
    _transform_from_quaternion_translation,
)
from posetestbot.calibration.legacy_static import require_legacy_static_known_target
from posetestbot.calibration.observations import SCHEMA_VERSION as OBSERVATION_SCHEMA
from posetestbot.calibration.targets import target_identity, validate_target_identity
from posetestbot.calibration.profiles import (
    SCHEMA_VERSION as PROFILE_SCHEMA_VERSION,
    CalibrationProfile,
    CalibrationQuality,
    CalibrationStatus,
    CalibrationTargetType,
    RigidTransform,
    TransformFrame,
    profile_from_dict,
    profile_to_dict,
    rectified_intrinsics_from_native,
    write_profile_collection,
)
from posetestbot.io.artifacts import (
    CALIBRATION_OBSERVATIONS,
    CALIBRATION_PROFILES_SOLVED,
    CALIBRATION_SOLVER_REPORT,
)
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    upsert_stage,
    write_run_manifest,
)
from posetestbot.sensors.contracts import MountingMode, SensorType


SCHEMA_VERSION = "calibration_solver.v1"
DEFAULT_HAND_EYE_METHOD = "tsai"
DEFAULT_HOLDOUT_FRACTION = 0.0
DEFAULT_COMPARE_HAND_EYE_METHODS = False
HAND_EYE_MIN_OBSERVATIONS = 3
HAND_EYE_METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
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


def _safe_profile_id(sensor_name: str, mounting_mode: MountingMode) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", sensor_name).strip("_")
    return f"{slug}_{mounting_mode.value}_aruco_solved"


def _sensor_folder(root: Path, source_path: Path, sensor_name: str) -> Path:
    source_sensor_folder = source_path.parent / "processed" / "synchronized" / sensor_name
    if source_sensor_folder.is_dir():
        return source_sensor_folder
    return root / "processed" / "synchronized" / sensor_name


def _profile_from_solution(
    *,
    sensor: Mapping[str, Any],
    solution: np.ndarray,
    mounting_mode: MountingMode,
    sensor_folder: Path,
    method: str,
    residuals: Mapping[str, float],
    observation_count: int,
    inlier_count: int,
    outlier_count: int,
    residual_frame: str,
    holdout_summary: Mapping[str, float] | None = None,
    holdout_count: int = 0,
    calibration_target: Mapping[str, Any] | None = None,
) -> CalibrationProfile:
    x, y, z, qw, qx, qy, qz = pt.pq_from_transform(solution)
    to_frame = (
        TransformFrame.END_EFFECTOR
        if mounting_mode == MountingMode.EYE_IN_HAND
        else TransformFrame.ROBOT_BASE
    )
    sensor_name = str(sensor["sensor_name"])
    sensor_type = SensorType(str(sensor["sensor_type"]))
    device_id = str(sensor.get("device_id") or sensor_name)
    intrinsics = _intrinsics_from_sensor_folder(sensor_folder)
    profile = CalibrationProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        profile_id=_safe_profile_id(sensor_name, mounting_mode),
        sensor_id=device_id,
        sensor_type=sensor_type,
        mounting_mode=mounting_mode,
        rig_position="wrist" if mounting_mode == MountingMode.EYE_IN_HAND else "static",
        intrinsics=intrinsics,
        rectified_intrinsics=rectified_intrinsics_from_native(intrinsics),
        extrinsics=RigidTransform(
            from_frame=TransformFrame.CAMERA,
            to_frame=to_frame,
            rotation_quaternion_wxyz=(float(qw), float(qx), float(qy), float(qz)),
            translation_mm=(float(x), float(y), float(z)),
        ),
        target_type=CalibrationTargetType.ARUCO_GRID,
        method=method,
        status=CalibrationStatus.NEEDS_VALIDATION,
        quality=CalibrationQuality(
            num_observations=observation_count,
            num_inliers=inlier_count,
            residual_translation_mm=float(residuals["mean_translation_mm"]),
            residual_rotation_deg=float(residuals["mean_rotation_deg"]),
            notes=(
                "Profile generated by calibration solver, residual-threshold "
                "inlier filtering, and optional held-out validation; validate "
                "before promotion."
            ),
        ),
        metadata={
            "sensor_name": sensor_name,
            "solver_source": CALIBRATION_OBSERVATIONS,
            "solver_report": CALIBRATION_SOLVER_REPORT,
            "solver_method": method,
            "residual_frame": residual_frame,
            "inlier_count": inlier_count,
            "outlier_count": outlier_count,
            "max_residual_translation_mm": residuals["max_translation_mm"],
            "max_residual_rotation_deg": residuals["max_rotation_deg"],
            "holdout_count": holdout_count,
            "calibration_target": dict(calibration_target or {}),
        },
    )
    if holdout_summary is not None:
        metadata = dict(profile.metadata)
        metadata.update(
            {
                "holdout_mean_residual_translation_mm": holdout_summary[
                    "mean_translation_mm"
                ],
                "holdout_max_residual_translation_mm": holdout_summary[
                    "max_translation_mm"
                ],
                "holdout_mean_residual_rotation_deg": holdout_summary[
                    "mean_rotation_deg"
                ],
                "holdout_max_residual_rotation_deg": holdout_summary[
                    "max_rotation_deg"
                ],
            }
        )
        profile = replace(profile, metadata=metadata)
    return profile


def _opencv_hand_eye_method(name: str) -> int:
    try:
        return HAND_EYE_METHODS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(HAND_EYE_METHODS))
        raise ValueError(f"hand_eye_method must be one of: {choices}") from exc


def _solve_hand_eye(
    observations: list[Mapping[str, Any]],
    *,
    method: str,
) -> np.ndarray:
    rotations_gripper_to_base = []
    translations_gripper_to_base = []
    rotations_target_to_camera = []
    translations_target_to_camera = []
    for observation in observations:
        robot_pose = observation.get("robot_ee_pose")
        if not isinstance(robot_pose, Mapping):
            raise ValueError("eye-in-hand observations require robot_ee_pose")
        gripper_to_base = _robot_ee_to_reference(robot_pose)
        target_to_camera = _target_to_camera(observation)
        rotations_gripper_to_base.append(gripper_to_base[:3, :3])
        translations_gripper_to_base.append(gripper_to_base[:3, 3])
        rotations_target_to_camera.append(target_to_camera[:3, :3])
        translations_target_to_camera.append(target_to_camera[:3, 3])

    rotation, translation = cv2.calibrateHandEye(
        rotations_gripper_to_base,
        translations_gripper_to_base,
        rotations_target_to_camera,
        translations_target_to_camera,
        method=_opencv_hand_eye_method(method),
    )
    return pt.transform_from(
        np.asarray(rotation, dtype=float),
        np.asarray(translation, dtype=float).reshape(3),
    )


def _target_to_reference_estimate(
    observation: Mapping[str, Any],
    camera_to_end_effector: np.ndarray,
) -> np.ndarray:
    robot_pose = observation.get("robot_ee_pose")
    if not isinstance(robot_pose, Mapping):
        raise ValueError("eye-in-hand observations require robot_ee_pose")
    tm = TransformManager()
    tm.add_transform("calibration_target", "camera", _target_to_camera(observation))
    tm.add_transform("camera", "end_effector", camera_to_end_effector)
    tm.add_transform("end_effector", "robot_base", _robot_ee_to_reference(robot_pose))
    return tm.get_transform("calibration_target", "robot_base")


def _hand_eye_solution(
    observations: list[Mapping[str, Any]],
    *,
    method: str,
    max_translation_residual_mm: float | None,
    max_rotation_residual_deg: float | None,
) -> tuple[np.ndarray, list[dict[str, float]], list[bool]]:
    if len(observations) < HAND_EYE_MIN_OBSERVATIONS:
        raise ValueError(
            "eye-in-hand hand-eye calibration requires at least "
            f"{HAND_EYE_MIN_OBSERVATIONS} observations"
        )
    solution = _solve_hand_eye(observations, method=method)
    target_estimates = [
        _target_to_reference_estimate(observation, solution)
        for observation in observations
    ]
    _average, residuals, inliers = _select_inliers(
        target_estimates,
        max_translation_residual_mm=max_translation_residual_mm,
        max_rotation_residual_deg=max_rotation_residual_deg,
    )
    inlier_observations = [
        observation
        for observation, is_inlier in zip(observations, inliers, strict=True)
        if is_inlier
    ]
    if 0 < len(inlier_observations) < len(observations):
        if len(inlier_observations) < HAND_EYE_MIN_OBSERVATIONS:
            return solution, residuals, inliers
        solution = _solve_hand_eye(inlier_observations, method=method)
        target_estimates = [
            _target_to_reference_estimate(observation, solution)
            for observation in observations
        ]
        _average, residuals, inliers = _select_inliers(
            target_estimates,
            max_translation_residual_mm=max_translation_residual_mm,
            max_rotation_residual_deg=max_rotation_residual_deg,
        )
    return solution, residuals, inliers


def _static_transforms(
    observations: list[Mapping[str, Any]],
    *,
    target_transform: np.ndarray,
) -> list[np.ndarray]:
    return [
        _candidate_transform(
            observation,
            target_to_reference=target_transform,
            mounting_mode=MountingMode.STATIC,
        )
        for observation in observations
    ]


def _static_solution(
    observations: list[Mapping[str, Any]],
    *,
    target_transform: np.ndarray,
    max_translation_residual_mm: float | None,
    max_rotation_residual_deg: float | None,
) -> tuple[np.ndarray, list[dict[str, float]], list[bool]]:
    transforms = _static_transforms(observations, target_transform=target_transform)
    return _select_inliers(
        transforms,
        max_translation_residual_mm=max_translation_residual_mm,
        max_rotation_residual_deg=max_rotation_residual_deg,
    )


def _summarize_observation_solution(
    *,
    observations: list[Mapping[str, Any]],
    residuals: list[Mapping[str, float]],
    inliers: list[bool],
    split: str = "train",
) -> list[dict[str, Any]]:
    records = []
    for observation, residual, inlier in zip(observations, residuals, inliers, strict=True):
        records.append(
            {
                "observation_id": observation.get("observation_id"),
                "sensor_name": observation.get("sensor_name"),
                "frame_id": observation.get("frame_id") or observation.get("frame_key"),
                "motion": observation.get("motion"),
                "split": split,
                "inlier": bool(inlier),
                "residual_translation_mm": float(residual["translation_mm"]),
                "residual_rotation_deg": float(residual["rotation_deg"]),
            }
        )
    return records


def _split_train_holdout(
    observations: list[Mapping[str, Any]],
    *,
    holdout_fraction: float,
    min_train_count: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    if holdout_fraction <= 0:
        return observations, []
    holdout_count = max(1, int(round(len(observations) * holdout_fraction)))
    holdout_count = min(holdout_count, len(observations))
    if len(observations) - holdout_count < min_train_count:
        holdout_count = max(0, len(observations) - min_train_count)
    if holdout_count <= 0:
        return observations, []
    return observations[:-holdout_count], observations[-holdout_count:]


def _hand_eye_holdout_residuals(
    *,
    solution: np.ndarray,
    train_observations: list[Mapping[str, Any]],
    train_inliers: list[bool],
    holdout_observations: list[Mapping[str, Any]],
) -> list[dict[str, float]]:
    inlier_targets = [
        _target_to_reference_estimate(observation, solution)
        for observation, is_inlier in zip(train_observations, train_inliers, strict=True)
        if is_inlier
    ]
    if not inlier_targets:
        return []
    reference = _average_transform(inlier_targets)
    holdout_targets = [
        _target_to_reference_estimate(observation, solution)
        for observation in holdout_observations
    ]
    return _residual_records(holdout_targets, reference)


def _static_holdout_residuals(
    *,
    solution: np.ndarray,
    holdout_observations: list[Mapping[str, Any]],
    target_transform: np.ndarray,
) -> list[dict[str, float]]:
    transforms = _static_transforms(holdout_observations, target_transform=target_transform)
    return _residual_records(transforms, solution)


def _holdout_status(
    summary: Mapping[str, float],
    *,
    max_translation_residual_mm: float | None,
    max_rotation_residual_deg: float | None,
) -> str:
    if (
        max_translation_residual_mm is not None
        and float(summary["max_translation_mm"]) > max_translation_residual_mm
    ):
        return "warning"
    if (
        max_rotation_residual_deg is not None
        and float(summary["max_rotation_deg"]) > max_rotation_residual_deg
    ):
        return "warning"
    return "ok"


def _method_label(mounting_mode: MountingMode, method: str) -> str:
    if mounting_mode == MountingMode.EYE_IN_HAND:
        return f"opencv_calibrateHandEye_{method}"
    return "static_target_reference_transform_average"


def _method_comparison_record(
    *,
    sensor_name: str,
    mounting_mode: MountingMode,
    method: str,
    train_observations: list[Mapping[str, Any]],
    holdout_observations: list[Mapping[str, Any]],
    target_transform: np.ndarray,
    max_translation_residual_mm: float | None,
    max_rotation_residual_deg: float | None,
    selected: bool,
) -> dict[str, Any]:
    method_name = _method_label(mounting_mode, method)
    try:
        if mounting_mode == MountingMode.EYE_IN_HAND:
            solution, residuals, inliers = _hand_eye_solution(
                train_observations,
                method=method,
                max_translation_residual_mm=max_translation_residual_mm,
                max_rotation_residual_deg=max_rotation_residual_deg,
            )
        else:
            solution, residuals, inliers = _static_solution(
                train_observations,
                target_transform=target_transform,
                max_translation_residual_mm=max_translation_residual_mm,
                max_rotation_residual_deg=max_rotation_residual_deg,
            )
    except (cv2.error, ValueError) as exc:
        return {
            "sensor_name": sensor_name,
            "mounting_mode": mounting_mode.value,
            "method": method_name,
            "status": "error",
            "selected": selected,
            "train_observation_count": len(train_observations),
            "holdout_observation_count": len(holdout_observations),
            "error": str(exc),
        }

    inlier_count = sum(1 for inlier in inliers if inlier)
    outlier_count = len(train_observations) - inlier_count
    inlier_residuals = [
        residual
        for residual, inlier in zip(residuals, inliers, strict=True)
        if inlier
    ]
    residual_summary = _residual_summary_from_records(inlier_residuals)
    holdout_summary: dict[str, float] | None = None
    holdout_check_status: str | None = None
    if holdout_observations and inlier_count > 0:
        if mounting_mode == MountingMode.EYE_IN_HAND:
            holdout_residuals = _hand_eye_holdout_residuals(
                solution=solution,
                train_observations=train_observations,
                train_inliers=inliers,
                holdout_observations=holdout_observations,
            )
        else:
            holdout_residuals = _static_holdout_residuals(
                solution=solution,
                holdout_observations=holdout_observations,
                target_transform=target_transform,
            )
        holdout_summary = _residual_summary_from_records(holdout_residuals)
        holdout_check_status = _holdout_status(
            holdout_summary,
            max_translation_residual_mm=max_translation_residual_mm,
            max_rotation_residual_deg=max_rotation_residual_deg,
        )

    status = "ok"
    if inlier_count == 0:
        status = "error"
    elif (
        holdout_check_status == "warning"
        or outlier_count > 0
    ):
        status = "warning"

    return {
        "sensor_name": sensor_name,
        "mounting_mode": mounting_mode.value,
        "method": method_name,
        "status": status,
        "selected": selected,
        "train_observation_count": len(train_observations),
        "holdout_observation_count": len(holdout_observations),
        "inlier_count": inlier_count,
        "outlier_count": outlier_count,
        "residuals": residual_summary,
        "holdout_residuals": holdout_summary,
        "holdout_status": holdout_check_status,
    }


def _method_comparisons(
    *,
    sensor_name: str,
    mounting_mode: MountingMode,
    selected_hand_eye_method: str,
    train_observations: list[Mapping[str, Any]],
    holdout_observations: list[Mapping[str, Any]],
    target_transform: np.ndarray,
    max_translation_residual_mm: float | None,
    max_rotation_residual_deg: float | None,
) -> list[dict[str, Any]]:
    if mounting_mode == MountingMode.EYE_IN_HAND:
        methods = sorted(HAND_EYE_METHODS)
    else:
        methods = [selected_hand_eye_method]
    return [
        _method_comparison_record(
            sensor_name=sensor_name,
            mounting_mode=mounting_mode,
            method=method,
            train_observations=train_observations,
            holdout_observations=holdout_observations,
            target_transform=target_transform,
            max_translation_residual_mm=max_translation_residual_mm,
            max_rotation_residual_deg=max_rotation_residual_deg,
            selected=method == selected_hand_eye_method,
        )
        for method in methods
    ]


def build_calibration_solver(
    run_root: str | Path,
    *,
    observations_path: str | Path | None = None,
    min_observations: int = 6,
    target_to_reference: Mapping[str, Any] | None = None,
    hand_eye_method: str = DEFAULT_HAND_EYE_METHOD,
    max_translation_residual_mm: float | None = DEFAULT_MAX_TRANSLATION_RESIDUAL_MM,
    max_rotation_residual_deg: float | None = DEFAULT_MAX_ROTATION_RESIDUAL_DEG,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    compare_hand_eye_methods: bool = DEFAULT_COMPARE_HAND_EYE_METHODS,
) -> dict[str, Any]:
    """Solve calibration profiles from calibration observations."""

    if min_observations < 1:
        raise ValueError("min_observations must be at least 1")
    if max_translation_residual_mm is not None and max_translation_residual_mm < 0:
        raise ValueError("max_translation_residual_mm must be greater than or equal to 0")
    if max_rotation_residual_deg is not None and max_rotation_residual_deg < 0:
        raise ValueError("max_rotation_residual_deg must be greater than or equal to 0")
    if not 0 <= holdout_fraction < 1:
        raise ValueError("holdout_fraction must be greater than or equal to 0 and less than 1")

    _opencv_hand_eye_method(hand_eye_method)
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
    require_legacy_static_known_target(
        root,
        observations_report,
        target_to_reference=target_to_reference,
        stage_label="Legacy calibration solver",
    )
    calibration_target = observations_report.get("target")
    calibration_target_evidence = (
        target_identity(calibration_target)
        if isinstance(calibration_target, Mapping)
        else {}
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
    solutions: list[dict[str, Any]] = []
    residual_records: list[dict[str, Any]] = []
    method_comparison_records: list[dict[str, Any]] = []
    total_observation_count = 0
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

    for sensor_name, raw_observations in sorted(by_sensor.items()):
        observations = list(raw_observations)
        if isinstance(calibration_target, Mapping):
            for observation in observations:
                validate_target_identity(
                    observation,
                    calibration_target,
                    label=f"Calibration observation {observation.get('observation_id', '')}",
                )
        total_observation_count += len(observations)
        sensor = dict(sensor_metadata.get(sensor_name, {}))
        sensor.setdefault("sensor_name", sensor_name)
        sensor.setdefault("sensor_type", observations[0].get("sensor_type"))
        sensor.setdefault("device_id", observations[0].get("device_id"))
        mounting_mode = MountingMode(
            str(sensor.get("mounting_mode") or MountingMode.EYE_IN_HAND.value)
        )
        min_train_count = (
            HAND_EYE_MIN_OBSERVATIONS
            if mounting_mode == MountingMode.EYE_IN_HAND
            else 1
        )
        train_observations, holdout_observations = _split_train_holdout(
            observations,
            holdout_fraction=holdout_fraction,
            min_train_count=min_train_count,
        )
        recommended_status = "ok" if len(observations) >= min_observations else "warning"
        checks.append(
            _check(
                f"solver_observations:{sensor_name}",
                recommended_status,
                (
                    f"{sensor_name} has {len(observations)} observation(s)."
                    if recommended_status == "ok"
                    else (
                        f"{sensor_name} has {len(observations)} observation(s); "
                        f"recommended minimum is {min_observations}."
                    )
                ),
                details={
                    "sensor_name": sensor_name,
                    "observation_count": len(observations),
                    "min_observations": min_observations,
                },
            )
        )
        if holdout_fraction > 0 and not holdout_observations:
            checks.append(
                _check(
                    f"solver_holdout:{sensor_name}",
                    "warning",
                    (
                        f"{sensor_name} does not have enough observations for "
                        "a held-out validation split."
                    ),
                    details={
                        "sensor_name": sensor_name,
                        "observation_count": len(observations),
                        "requested_holdout_fraction": holdout_fraction,
                        "min_train_count": min_train_count,
                    },
                )
            )

        try:
            if mounting_mode == MountingMode.EYE_IN_HAND:
                solution, residuals, inliers = _hand_eye_solution(
                    train_observations,
                    method=hand_eye_method,
                    max_translation_residual_mm=max_translation_residual_mm,
                    max_rotation_residual_deg=max_rotation_residual_deg,
                )
                method = _method_label(mounting_mode, hand_eye_method)
                residual_frame = "target_to_robot_base_consistency"
            else:
                solution, residuals, inliers = _static_solution(
                    train_observations,
                    target_transform=target_transform,
                    max_translation_residual_mm=max_translation_residual_mm,
                    max_rotation_residual_deg=max_rotation_residual_deg,
                )
                method = _method_label(mounting_mode, hand_eye_method)
                residual_frame = "camera_to_robot_base_consistency"
        except (cv2.error, ValueError) as exc:
            checks.append(
                _check(
                    f"solver_method:{sensor_name}",
                    "error",
                    f"Calibration solver failed for {sensor_name}: {exc}",
                    details={
                        "sensor_name": sensor_name,
                        "mounting_mode": mounting_mode.value,
                        "hand_eye_method": hand_eye_method,
                    },
                )
            )
            continue

        inlier_count = sum(1 for inlier in inliers if inlier)
        outlier_count = len(train_observations) - inlier_count
        total_inlier_count += inlier_count
        total_outlier_count += outlier_count
        if inlier_count == 0:
            checks.append(
                _check(
                    f"solver_inliers:{sensor_name}",
                    "error",
                    f"{sensor_name} has no inlier observations after solver filtering.",
                    details={
                        "sensor_name": sensor_name,
                        "observation_count": len(train_observations),
                        "max_translation_residual_mm": max_translation_residual_mm,
                        "max_rotation_residual_deg": max_rotation_residual_deg,
                    },
                )
            )
            continue

        inlier_residuals = [
            residual
            for residual, inlier in zip(residuals, inliers, strict=True)
            if inlier
        ]
        residual_summary = _residual_summary_from_records(inlier_residuals)
        all_residual_summary = _residual_summary_from_records(residuals)
        inlier_status = "ok" if inlier_count >= min_observations else "warning"
        checks.append(
            _check(
                f"solver_inliers:{sensor_name}",
                inlier_status,
                (
                    f"{sensor_name} has {inlier_count} inlier observation(s)."
                    if inlier_status == "ok"
                    else (
                        f"{sensor_name} has {inlier_count} inlier observation(s); "
                        f"recommended minimum is {min_observations}."
                    )
                ),
                details={
                    "sensor_name": sensor_name,
                    "inlier_count": inlier_count,
                    "outlier_count": outlier_count,
                    "train_observation_count": len(train_observations),
                    "holdout_observation_count": len(holdout_observations),
                    "min_observations": min_observations,
                },
            )
        )
        if outlier_count:
            checks.append(
                _check(
                    f"solver_outliers:{sensor_name}",
                    "warning",
                    f"{sensor_name} rejected {outlier_count} observation(s).",
                    details={
                        "sensor_name": sensor_name,
                        "outlier_count": outlier_count,
                    },
                )
        )

        holdout_residuals: list[dict[str, float]] = []
        holdout_summary: dict[str, float] | None = None
        holdout_check_status = None
        if holdout_observations:
            if mounting_mode == MountingMode.EYE_IN_HAND:
                holdout_residuals = _hand_eye_holdout_residuals(
                    solution=solution,
                    train_observations=train_observations,
                    train_inliers=inliers,
                    holdout_observations=holdout_observations,
                )
            else:
                holdout_residuals = _static_holdout_residuals(
                    solution=solution,
                    holdout_observations=holdout_observations,
                    target_transform=target_transform,
                )
            holdout_summary = _residual_summary_from_records(holdout_residuals)
            holdout_check_status = _holdout_status(
                holdout_summary,
                max_translation_residual_mm=max_translation_residual_mm,
                max_rotation_residual_deg=max_rotation_residual_deg,
            )
            checks.append(
                _check(
                    f"solver_holdout:{sensor_name}",
                    holdout_check_status,
                    (
                        f"{sensor_name} held-out residuals are within thresholds."
                        if holdout_check_status == "ok"
                        else (
                            f"{sensor_name} held-out residuals exceeded a "
                            "configured residual threshold."
                        )
                    ),
                    details={
                        "sensor_name": sensor_name,
                        "holdout_observation_count": len(holdout_observations),
                        **holdout_summary,
                    },
                )
            )

        comparison_records: list[dict[str, Any]] = []
        if compare_hand_eye_methods:
            comparison_records = _method_comparisons(
                sensor_name=sensor_name,
                mounting_mode=mounting_mode,
                selected_hand_eye_method=hand_eye_method,
                train_observations=train_observations,
                holdout_observations=holdout_observations,
                target_transform=target_transform,
                max_translation_residual_mm=max_translation_residual_mm,
                max_rotation_residual_deg=max_rotation_residual_deg,
            )
            method_comparison_records.extend(comparison_records)
            comparison_statuses = {
                str(record.get("status"))
                for record in comparison_records
            }
            checks.append(
                _check(
                    f"solver_method_comparison:{sensor_name}",
                    (
                        "warning"
                        if comparison_statuses - {"ok"}
                        else "ok"
                    ),
                    (
                        f"{sensor_name} method comparison evaluated "
                        f"{len(comparison_records)} method(s)."
                    ),
                    details={
                        "sensor_name": sensor_name,
                        "method_count": len(comparison_records),
                        "statuses": sorted(comparison_statuses),
                    },
                )
            )

        profile = _profile_from_solution(
            sensor=sensor,
            solution=solution,
            mounting_mode=mounting_mode,
            sensor_folder=_sensor_folder(root, source_path, sensor_name),
            method=method,
            residuals=residual_summary,
            observation_count=len(observations),
            inlier_count=inlier_count,
            outlier_count=outlier_count,
            residual_frame=residual_frame,
            holdout_summary=holdout_summary,
            holdout_count=len(holdout_observations),
            calibration_target=calibration_target_evidence,
        )
        profiles.append(profile_to_dict(profile))
        x, y, z, qw, qx, qy, qz = pt.pq_from_transform(solution)
        solutions.append(
            {
                "sensor_name": sensor_name,
                "mounting_mode": mounting_mode.value,
                "method": method,
                "observation_count": len(observations),
                "train_observation_count": len(train_observations),
                "holdout_observation_count": len(holdout_observations),
                "inlier_count": inlier_count,
                "outlier_count": outlier_count,
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
                "residuals": {
                    **dict(residual_summary),
                    "all_mean_translation_mm": all_residual_summary[
                        "mean_translation_mm"
                    ],
                    "all_max_translation_mm": all_residual_summary[
                        "max_translation_mm"
                    ],
                    "all_mean_rotation_deg": all_residual_summary["mean_rotation_deg"],
                    "all_max_rotation_deg": all_residual_summary["max_rotation_deg"],
                },
                "holdout_residuals": holdout_summary,
                "holdout_status": holdout_check_status,
                "method_comparisons": comparison_records,
            }
        )
        residual_records.extend(
            _summarize_observation_solution(
                observations=train_observations,
                residuals=residuals,
                inliers=inliers,
                split="train",
            )
        )
        if holdout_residuals:
            residual_records.extend(
                _summarize_observation_solution(
                    observations=holdout_observations,
                    residuals=holdout_residuals,
                    inliers=[True] * len(holdout_observations),
                    split="holdout",
                )
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_root": root.as_posix(),
        "source_observations": _relative(source_path, root),
        "calibration_target": calibration_target_evidence,
        "target_to_reference": target,
        "hand_eye_method": hand_eye_method,
        "compare_hand_eye_methods": compare_hand_eye_methods,
        "residual_thresholds": {
            "max_translation_mm": max_translation_residual_mm,
            "max_rotation_deg": max_rotation_residual_deg,
        },
        "holdout_fraction": holdout_fraction,
        "overall_status": _overall_status(checks),
        "min_observations": min_observations,
        "sensor_count": len(by_sensor),
        "profile_count": len(profiles),
        "observation_count": total_observation_count,
        "candidate_count": total_observation_count,
        "inlier_count": total_inlier_count,
        "outlier_count": total_outlier_count,
        "checks": checks,
        "profiles": profiles,
        "solutions": solutions,
        "residuals": residual_records,
        "method_comparisons": method_comparison_records,
    }


def calibration_solver_report_path(
    run_root: str | Path,
    *,
    output_root: str | Path | None = None,
) -> Path:
    destination = Path(output_root) if output_root is not None else Path(run_root)
    return destination / CALIBRATION_SOLVER_REPORT


def calibration_profiles_solved_path(
    run_root: str | Path,
    *,
    output_root: str | Path | None = None,
) -> Path:
    destination = Path(output_root) if output_root is not None else Path(run_root)
    return destination / CALIBRATION_PROFILES_SOLVED


def write_calibration_solver(
    run_root: str | Path,
    report: Mapping[str, Any],
    *,
    output_root: str | Path | None = None,
) -> tuple[Path, Path]:
    root = Path(run_root)
    report_path = calibration_solver_report_path(root, output_root=output_root)
    atomic_write_json(report_path, dict(report))

    profiles = [profile_from_dict(profile) for profile in report.get("profiles", [])]
    profiles_path = calibration_profiles_solved_path(
        root,
        output_root=output_root,
    )
    write_profile_collection(profiles, profiles_path)
    return report_path, profiles_path


def write_calibration_solver_with_manifest(
    run_root: str | Path,
    *,
    observations_path: str | Path | None = None,
    min_observations: int = 6,
    target_to_reference: Mapping[str, Any] | None = None,
    hand_eye_method: str = DEFAULT_HAND_EYE_METHOD,
    max_translation_residual_mm: float | None = DEFAULT_MAX_TRANSLATION_RESIDUAL_MM,
    max_rotation_residual_deg: float | None = DEFAULT_MAX_ROTATION_RESIDUAL_DEG,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    compare_hand_eye_methods: bool = DEFAULT_COMPARE_HAND_EYE_METHODS,
    output_root: str | Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    run_root_path = Path(run_root)
    manifest = load_or_create_run_manifest(run_root_path)
    upsert_stage(manifest, name="calibration_solver", status="running")
    write_run_manifest(manifest, run_root_path)
    try:
        report = build_calibration_solver(
            run_root_path,
            observations_path=observations_path,
            min_observations=min_observations,
            target_to_reference=target_to_reference,
            hand_eye_method=hand_eye_method,
            max_translation_residual_mm=max_translation_residual_mm,
            max_rotation_residual_deg=max_rotation_residual_deg,
            holdout_fraction=holdout_fraction,
            compare_hand_eye_methods=compare_hand_eye_methods,
        )
        report_path, profiles_path = write_calibration_solver(
            run_root_path,
            report,
            output_root=output_root,
        )
        upsert_stage(
            manifest,
            name="calibration_solver",
            status="succeeded" if report["overall_status"] != "error" else "failed",
            artifacts={
                CALIBRATION_SOLVER_REPORT: report_path,
                CALIBRATION_PROFILES_SOLVED: profiles_path,
            },
            run_root=run_root_path,
            message=f"Calibration solver status: {report['overall_status']}.",
        )
        write_run_manifest(manifest, run_root_path)
    except Exception as exc:
        upsert_stage(
            manifest,
            name="calibration_solver",
            status="failed",
            message=str(exc),
        )
        write_run_manifest(manifest, run_root_path)
        raise
    return report_path, profiles_path, report
