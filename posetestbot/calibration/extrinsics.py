"""Explicit unknown-target, known-target, and comparison extrinsic solving."""

from __future__ import annotations

import math
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from pytransform3d import transformations as pt

from posetestbot.calibration.candidates import (
    _average_transform,
    _candidate_transform,
    _intrinsics_from_sensor_folder,
    _observations_by_sensor,
    _read_json,
    _residual_summary_from_records,
    _select_inliers,
    _sensor_by_name,
    _target_transform,
    _transform_from_quaternion_translation,
)
from posetestbot.calibration.frame_graph import resolve_profile_transform
from posetestbot.calibration.intrinsics import (
    load_intrinsic_profile_collection,
    select_intrinsic_profile,
    sensor_intrinsic_identity,
)
from posetestbot.calibration.observations import SCHEMA_VERSION as OBSERVATION_SCHEMA
from posetestbot.calibration.profiles import (
    SCHEMA_VERSION as PROFILE_SCHEMA,
    CalibrationProfile,
    CalibrationQuality,
    CalibrationStatus,
    CalibrationTargetType,
    RigidTransform,
    TransformFrame,
    profile_to_dict,
    profile_from_dict,
    rectified_intrinsics_from_native,
    write_profile_collection,
)
from posetestbot.calibration.solver import (
    DEFAULT_HAND_EYE_METHOD,
    _hand_eye_solution,
    _sensor_folder,
    _target_to_reference_estimate,
)
from posetestbot.io.atomic import atomic_write_json
from posetestbot.io.artifacts import (
    CALIBRATION_OBSERVATIONS,
    CALIBRATION_PROFILES_SOLVED,
    CALIBRATION_SOLVER_REPORT,
    INTRINSIC_CALIBRATION_PROFILES,
)
from posetestbot.io.manifest import load_or_create_run_manifest, upsert_stage, write_run_manifest
from posetestbot.sensors.contracts import CameraIntrinsics, MountingMode, SensorType


SCHEMA_VERSION = "calibration_solver.v2"
MODES = ("hand_eye_unknown_target", "known_target", "compare")
DEFAULT_MIN_INLIERS = 6
DEFAULT_MAX_MEAN_TRANSLATION_MM = 10.0
DEFAULT_MAX_MEAN_ROTATION_DEG = 5.0
DEFAULT_MAX_OUTLIER_RATIO = 0.25
DEFAULT_MAX_CROSS_TRANSLATION_MM = 10.0
DEFAULT_MAX_CROSS_ROTATION_DEG = 5.0


def _generated_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rotation_delta_deg(left: np.ndarray, right: np.ndarray) -> float:
    delta = left[:3, :3].T @ right[:3, :3]
    cosine = max(-1.0, min(1.0, (float(np.trace(delta)) - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def _profile_id(sensor_name: str, mounting_mode: MountingMode, mode: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", sensor_name).strip("_")
    return f"{slug}_{mounting_mode.value}_{mode}"


def _profile(
    *,
    sensor: Mapping[str, Any],
    sensor_folder: Path,
    mounting_mode: MountingMode,
    solution: np.ndarray,
    mode: str,
    residual_summary: Mapping[str, float],
    observation_count: int,
    inlier_count: int,
    fixed_transforms: list[Mapping[str, Any]],
    intrinsic_profile: Mapping[str, Any] | None = None,
) -> CalibrationProfile:
    x, y, z, qw, qx, qy, qz = pt.pq_from_transform(solution)
    if intrinsic_profile is None:
        native = _intrinsics_from_sensor_folder(sensor_folder)
        rectified = rectified_intrinsics_from_native(native)
        rectified_roi = None
        intrinsic_metadata: dict[str, Any] = {
            "intrinsic_source": "legacy_camera_sidecars"
        }
    else:
        native_value = intrinsic_profile["native"]
        rectified_value = intrinsic_profile["rectified"]
        depth_scale = float(intrinsic_profile["depth"]["scale_to_mm"])
        native = CameraIntrinsics(
            cam_k=tuple(float(item) for item in native_value["cam_K"]),
            width=int(native_value["width"]),
            height=int(native_value["height"]),
            distortion=tuple(float(item) for item in native_value["distortion"]),
            depth_scale_to_mm=depth_scale,
        )
        rectified = CameraIntrinsics(
            cam_k=tuple(float(item) for item in rectified_value["cam_K"]),
            width=int(rectified_value["width"]),
            height=int(rectified_value["height"]),
            distortion=tuple(float(item) for item in rectified_value["distortion"]),
            depth_scale_to_mm=depth_scale,
        )
        rectified_roi = tuple(int(item) for item in rectified_value["valid_roi"])
        intrinsic_metadata = {
            "intrinsic_profile_id": intrinsic_profile["profile_id"],
            "intrinsic_source": intrinsic_profile["source"],
            "projection_provenance": {
                "native": intrinsic_profile["source"],
                "rectified": {
                    "algorithm": "opencv_alpha0_same_resolution",
                    "valid_roi": list(rectified_roi),
                },
                "depth": intrinsic_profile["depth"],
            },
        }
    profile = CalibrationProfile(
        schema_version=PROFILE_SCHEMA,
        profile_id=_profile_id(str(sensor["sensor_name"]), mounting_mode, mode),
        sensor_id=str(sensor.get("device_id") or sensor["sensor_name"]),
        sensor_type=SensorType(str(sensor["sensor_type"])),
        mounting_mode=mounting_mode,
        rig_position="wrist" if mounting_mode == MountingMode.EYE_IN_HAND else "static",
        intrinsics=native,
        rectified_intrinsics=rectified,
        rectified_valid_roi=rectified_roi,
        extrinsics=RigidTransform(
            from_frame=TransformFrame.CAMERA,
            to_frame=(
                TransformFrame.ROBOT_FLANGE
                if mounting_mode == MountingMode.EYE_IN_HAND
                else TransformFrame.TEMPLATE_BASE
            ),
            rotation_quaternion_wxyz=(float(qw), float(qx), float(qy), float(qz)),
            translation_mm=(float(x), float(y), float(z)),
        ),
        target_type=CalibrationTargetType.ARUCO_GRID,
        method=mode,
        status=CalibrationStatus.NEEDS_VALIDATION,
        quality=CalibrationQuality(
            num_observations=observation_count,
            num_inliers=inlier_count,
            residual_translation_mm=float(residual_summary["mean_translation_mm"]),
            residual_rotation_deg=float(residual_summary["mean_rotation_deg"]),
        ),
        metadata={
            "sensor_name": sensor["sensor_name"],
            "solver_mode": mode,
            "outlier_count": observation_count - inlier_count,
            "residual_frame": (
                "aruco_grid_to_template_base_consistency"
                if mode == "hand_eye_unknown_target"
                else "camera_extrinsic_consistency"
            ),
            "max_residual_translation_mm": residual_summary["max_translation_mm"],
            "max_residual_rotation_deg": residual_summary["max_rotation_deg"],
            **intrinsic_metadata,
        },
    )
    if mounting_mode == MountingMode.EYE_IN_HAND and fixed_transforms:
        try:
            camera_to_tcp = resolve_profile_transform(
                profile,
                "tcp",
                fixed_transforms=fixed_transforms,
            )
        except ValueError:
            pass
        else:
            tx, ty, tz, tqw, tqx, tqy, tqz = pt.pq_from_transform(camera_to_tcp)
            metadata = dict(profile.metadata)
            metadata["derived_camera_to_tcp"] = {
                "from": "camera",
                "to": "tcp",
                "rotation_quaternion_wxyz": [tqw, tqx, tqy, tqz],
                "translation_mm": [tx, ty, tz],
                "fixed_transform_provenance": [dict(item) for item in fixed_transforms],
            }
            profile = replace(profile, metadata=metadata)
    return profile


def _known_solution(
    observations: list[Mapping[str, Any]],
    *,
    mounting_mode: MountingMode,
    target_to_template_base: np.ndarray,
    max_translation_mm: float | None,
    max_rotation_deg: float | None,
) -> tuple[np.ndarray, list[dict[str, float]], list[bool]]:
    transforms = [
        _candidate_transform(
            observation,
            target_to_reference=target_to_template_base,
            mounting_mode=mounting_mode,
        )
        for observation in observations
    ]
    return _select_inliers(
        transforms,
        max_translation_residual_mm=max_translation_mm,
        max_rotation_residual_deg=max_rotation_deg,
    )


def _unknown_solution(
    observations: list[Mapping[str, Any]],
    *,
    hand_eye_method: str,
    max_translation_mm: float | None,
    max_rotation_deg: float | None,
) -> tuple[np.ndarray, list[dict[str, float]], list[bool], np.ndarray]:
    solution, residuals, inliers = _hand_eye_solution(
        observations,
        method=hand_eye_method,
        max_translation_residual_mm=max_translation_mm,
        max_rotation_residual_deg=max_rotation_deg,
    )
    estimates = [
        _target_to_reference_estimate(observation, solution)
        for observation, keep in zip(observations, inliers, strict=True)
        if keep
    ]
    if not estimates:
        raise ValueError("Unknown-target solve produced no inlier target estimates")
    return solution, residuals, inliers, _average_transform(estimates)


def _target_edge(target: Mapping[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    placement = target.get("placement")
    if not isinstance(placement, Mapping):
        raise ValueError("known_target/compare requires calibration_target placement")
    if placement.get("from") != "aruco_grid" or placement.get("to") != "template_base":
        raise ValueError("Target placement must map aruco_grid to template_base")
    normalized = _target_transform(
        {
            "from": "aruco_grid",
            "to": "template_base",
            "rotation_quaternion_wxyz": list(placement.get("rotation_quaternion_wxyz", [])),
            "translation_mm": list(placement.get("translation_mm", [])),
            "unit": "mm",
            "source": placement.get("source"),
        }
    )
    return normalized, _transform_from_quaternion_translation(
        rotation_quaternion_wxyz=normalized["rotation_quaternion_wxyz"],
        translation_mm=normalized["translation_mm"],
    )


def build_grid_extrinsic_solver(
    run_root: str | Path,
    *,
    target: Mapping[str, Any],
    mode: str,
    observations_path: str | Path | None = None,
    hand_eye_method: str = DEFAULT_HAND_EYE_METHOD,
    min_inliers: int = DEFAULT_MIN_INLIERS,
    max_mean_translation_mm: float | None = DEFAULT_MAX_MEAN_TRANSLATION_MM,
    max_mean_rotation_deg: float | None = DEFAULT_MAX_MEAN_ROTATION_DEG,
    max_outlier_ratio: float | None = DEFAULT_MAX_OUTLIER_RATIO,
    max_cross_translation_mm: float | None = DEFAULT_MAX_CROSS_TRANSLATION_MM,
    max_cross_rotation_deg: float | None = DEFAULT_MAX_CROSS_ROTATION_DEG,
    fixed_transforms: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError("mode must be one of: " + ", ".join(MODES))
    root = Path(run_root)
    source_path = Path(observations_path) if observations_path else root / CALIBRATION_OBSERVATIONS
    if not source_path.is_absolute():
        source_path = root / source_path
    observation_report = _read_json(source_path)
    if observation_report.get("schema_version") != OBSERVATION_SCHEMA:
        raise ValueError("Unsupported calibration observation schema")
    fixed = list(fixed_transforms or [])
    intrinsic_profiles_path = root / INTRINSIC_CALIBRATION_PROFILES
    intrinsic_profiles = (
        load_intrinsic_profile_collection(intrinsic_profiles_path)
        if intrinsic_profiles_path.is_file()
        else []
    )
    placement_record: dict[str, Any] | None = None
    target_matrix: np.ndarray | None = None
    if mode in {"known_target", "compare"}:
        placement_record, target_matrix = _target_edge(target)

    profiles: list[CalibrationProfile] = []
    solutions: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    sensors = _sensor_by_name(observation_report)
    for sensor_name, observations in sorted(_observations_by_sensor(observation_report).items()):
        sensor = dict(sensors.get(sensor_name, {}))
        sensor.setdefault("sensor_name", sensor_name)
        sensor.setdefault("sensor_type", observations[0].get("sensor_type"))
        sensor.setdefault("device_id", observations[0].get("device_id"))
        mounting_mode = MountingMode(str(sensor.get("mounting_mode") or "eye_in_hand"))
        sensor_folder = _sensor_folder(root, source_path, sensor_name)
        selected_intrinsic = None
        if intrinsic_profiles:
            sensor_id, orientation, resolution = sensor_intrinsic_identity(sensor_folder)
            selected_intrinsic = select_intrinsic_profile(
                intrinsic_profiles,
                sensor_id=sensor_id,
                resolution=resolution,
                orientation=orientation,
            )
        requested_modes = (
            ["hand_eye_unknown_target", "known_target"] if mode == "compare" else [mode]
        )
        solved: dict[str, np.ndarray] = {}
        for solve_mode in requested_modes:
            if solve_mode == "hand_eye_unknown_target" and mounting_mode == MountingMode.STATIC:
                checks.append(
                    {
                        "name": f"observability:{sensor_name}",
                        "status": "error",
                        "message": "Unknown-target robot-relative calibration is unobservable for static cameras.",
                    }
                )
                continue
            try:
                if solve_mode == "hand_eye_unknown_target":
                    solution, residuals, inliers, target_estimate = _unknown_solution(
                        observations,
                        hand_eye_method=hand_eye_method,
                        max_translation_mm=max_mean_translation_mm,
                        max_rotation_deg=max_mean_rotation_deg,
                    )
                else:
                    assert target_matrix is not None
                    solution, residuals, inliers = _known_solution(
                        observations,
                        mounting_mode=mounting_mode,
                        target_to_template_base=target_matrix,
                        max_translation_mm=max_mean_translation_mm,
                        max_rotation_deg=max_mean_rotation_deg,
                    )
                    target_estimate = target_matrix
            except (ValueError, cv2.error) as exc:
                checks.append(
                    {
                        "name": f"solve:{sensor_name}:{solve_mode}",
                        "status": "error",
                        "message": str(exc),
                    }
                )
                continue
            inlier_records = [
                residual for residual, keep in zip(residuals, inliers, strict=True) if keep
            ]
            summary = _residual_summary_from_records(inlier_records)
            inlier_count = sum(bool(item) for item in inliers)
            outlier_ratio = (len(observations) - inlier_count) / len(observations)
            failures = []
            if inlier_count < min_inliers:
                failures.append(f"{inlier_count} inliers is below {min_inliers}")
            if max_mean_translation_mm is not None and summary["mean_translation_mm"] > max_mean_translation_mm:
                failures.append("mean translation residual exceeds threshold")
            if max_mean_rotation_deg is not None and summary["mean_rotation_deg"] > max_mean_rotation_deg:
                failures.append("mean rotation residual exceeds threshold")
            if max_outlier_ratio is not None and outlier_ratio > max_outlier_ratio:
                failures.append("outlier ratio exceeds threshold")
            checks.append(
                {
                    "name": f"solve:{sensor_name}:{solve_mode}",
                    "status": "error" if failures else "ok",
                    "message": "; ".join(failures) if failures else f"Solved {solve_mode} with {inlier_count} inliers.",
                }
            )
            profile = _profile(
                sensor=sensor,
                sensor_folder=sensor_folder,
                mounting_mode=mounting_mode,
                solution=solution,
                mode=solve_mode,
                residual_summary=summary,
                observation_count=len(observations),
                inlier_count=inlier_count,
                fixed_transforms=fixed,
                intrinsic_profile=selected_intrinsic,
            )
            profiles.append(profile)
            solved[solve_mode] = solution
            tx, ty, tz, qw, qx, qy, qz = pt.pq_from_transform(solution)
            solutions.append(
                {
                    "sensor_name": sensor_name,
                    "mounting_mode": mounting_mode.value,
                    "mode": solve_mode,
                    "profile_id": profile.profile_id,
                    "transform": {
                        "from": "camera",
                        "to": profile.extrinsics.to_frame.value,
                        "rotation_quaternion_wxyz": [qw, qx, qy, qz],
                        "translation_mm": [tx, ty, tz],
                    },
                    "target_to_template_base_estimate": target_estimate.tolist(),
                    "inlier_count": inlier_count,
                    "outlier_ratio": outlier_ratio,
                    "residuals": summary,
                }
            )

        if mode == "compare" and len(solved) == 2:
            unknown = solved["hand_eye_unknown_target"]
            known = solved["known_target"]
            translation = float(np.linalg.norm(unknown[:3, 3] - known[:3, 3]))
            rotation = _rotation_delta_deg(unknown, known)
            blocked = (
                (max_cross_translation_mm is not None and translation > max_cross_translation_mm)
                or (max_cross_rotation_deg is not None and rotation > max_cross_rotation_deg)
            )
            comparison = {
                "sensor_name": sensor_name,
                "translation_disagreement_mm": translation,
                "rotation_disagreement_deg": rotation,
                "status": "error" if blocked else "ok",
                "thresholds": {
                    "translation_mm": max_cross_translation_mm,
                    "rotation_deg": max_cross_rotation_deg,
                },
                "gate_override": {
                    "translation_changed": max_cross_translation_mm != DEFAULT_MAX_CROSS_TRANSLATION_MM,
                    "rotation_changed": max_cross_rotation_deg != DEFAULT_MAX_CROSS_ROTATION_DEG,
                    "disabled": max_cross_translation_mm is None or max_cross_rotation_deg is None,
                },
            }
            comparisons.append(comparison)
            checks.append(
                {
                    "name": f"cross_method_agreement:{sensor_name}",
                    "status": comparison["status"],
                    "message": (
                        f"Cross-method disagreement is {translation:.3f} mm / {rotation:.3f} deg."
                    ),
                }
            )

    statuses = {str(item["status"]) for item in checks}
    overall_status = "error" if "error" in statuses else "ok"
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_root": root.as_posix(),
        "overall_status": overall_status,
        "mode": mode,
        "target": dict(target),
        "target_to_template_base": placement_record,
        "fixed_transforms": [dict(item) for item in fixed],
        "sensor_count": len(_observations_by_sensor(observation_report)),
        "profile_count": len(profiles),
        "observation_count": sum(
            len(items) for items in _observations_by_sensor(observation_report).values()
        ),
        "candidate_count": sum(profile.quality.num_observations for profile in profiles),
        "inlier_count": sum(profile.quality.num_inliers for profile in profiles),
        "outlier_count": sum(
            profile.quality.num_observations - profile.quality.num_inliers for profile in profiles
        ),
        "thresholds": {
            "min_inliers": min_inliers,
            "max_mean_translation_mm": max_mean_translation_mm,
            "max_mean_rotation_deg": max_mean_rotation_deg,
            "max_outlier_ratio": max_outlier_ratio,
            "max_cross_translation_mm": max_cross_translation_mm,
            "max_cross_rotation_deg": max_cross_rotation_deg,
        },
        "profiles": [profile_to_dict(profile) for profile in profiles],
        "solutions": solutions,
        "comparisons": comparisons,
        "checks": checks,
    }


def write_grid_extrinsic_solver_with_manifest(
    run_root: str | Path,
    **kwargs: Any,
) -> tuple[Path, Path, dict[str, Any]]:
    root = Path(run_root)
    manifest = load_or_create_run_manifest(root)
    upsert_stage(manifest, name="calibration_solver", status="running")
    write_run_manifest(manifest, root)
    try:
        report = build_grid_extrinsic_solver(root, **kwargs)
        profiles = [profile_from_dict(item) for item in report["profiles"]]
        profiles_path = write_profile_collection(profiles, root / CALIBRATION_PROFILES_SOLVED)
        report_path = atomic_write_json(root / CALIBRATION_SOLVER_REPORT, report)
        upsert_stage(
            manifest,
            name="calibration_solver",
            status="succeeded" if report["overall_status"] != "error" else "failed",
            artifacts={
                CALIBRATION_SOLVER_REPORT: report_path,
                CALIBRATION_PROFILES_SOLVED: profiles_path,
            },
            run_root=root,
        )
        write_run_manifest(manifest, root)
        return report_path, profiles_path, report
    except Exception as exc:
        upsert_stage(manifest, name="calibration_solver", status="failed", message=str(exc))
        write_run_manifest(manifest, root)
        raise
