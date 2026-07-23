"""Intent-level, immutable calibration attempts and explicit promotion."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from posetestbot.aruco.grid import _matched_points, detect_sensor_folder
from posetestbot.calibration.attempt_solver import (
    DEFAULT_MAX_MEAN_ROTATION_DEG,
    DEFAULT_MAX_MEAN_TRANSLATION_MM,
    DEFAULT_MAX_OUTLIER_RATIO,
    DEFAULT_MAX_PNP_ALL_POINT_MEAN_ERROR_PX,
    DEFAULT_MAX_OBSERVATIONS_PER_MOTION,
    DEFAULT_MIN_INLIERS,
    DEFAULT_MIN_PNP_COMMON_INLIERS,
    DEFAULT_MIN_PNP_COMMON_INLIER_RATIO,
    DEFAULT_MIN_PNP_GRID_COLUMNS,
    DEFAULT_MIN_PNP_GRID_ROWS,
    DEFAULT_MIN_PNP_SUPPORTED_CORNERS_PER_MARKER,
    DEFAULT_MIN_PNP_SUPPORTED_MARKERS,
    DEFAULT_MIN_ROTATION_AXIS_ANGLE_DEG,
    DEFAULT_MIN_ROTATION_AXIS_SINGULAR_RATIO,
    EXTRINSIC_METHOD_ORDER,
    PNP_METHOD_ORDER,
    evaluate_extrinsic_candidate,
    rank_candidates,
    solve_planar_pnp_candidates,
    transform_from_record,
    transform_residual,
)
from posetestbot.calibration.intrinsics import (
    DEFAULT_MAX_RMS_PX,
    DEFAULT_MAX_VIEW_ERROR_PX,
    DEFAULT_MIN_ACCEPTED_VIEWS,
    DEFAULT_MIN_COVERAGE_CELLS,
    IntrinsicCalibrationError,
    _view_points,
    calibrate_intrinsic_profile,
    factory_intrinsic_profile,
    load_intrinsic_profile_collection,
    projection_is_opencv_compatible,
    select_intrinsic_profile,
    sensor_intrinsic_identity,
    write_intrinsic_profile_collection,
)
from posetestbot.calibration.profiles import (
    SCHEMA_VERSION as PROFILE_SCHEMA_VERSION,
    CalibrationProfile,
    CalibrationQuality,
    CalibrationStatus,
    CalibrationTargetType,
    RigidTransform,
    TransformFrame,
    load_profile_collection,
    profile_to_dict,
    rectified_intrinsics_from_native,
    write_profile_collection,
)
from posetestbot.calibration.target_library import (
    LIBRARY_DIRECTORY,
    CalibrationTargetConflict,
    default_target_library_root,
    list_target_bundles,
    replacement_blockers,
    validate_run_target_selection,
    validate_target_bundle,
)
from posetestbot.calibration.targets import (
    normalize_calibration_target_spec,
    opencv_grid_board,
    target_identity,
    validate_target_identity,
)
from posetestbot.calibration.time_offset import (
    DEFAULT_POLICY as DEFAULT_SYNCHRONIZATION_POLICY,
    DEFAULT_REFERENCE_PNP_METHOD,
    IMPLEMENTATION_REVISION as TIME_OFFSET_IMPLEMENTATION_REVISION,
    POLICIES as SYNCHRONIZATION_POLICIES,
    SCHEMA_VERSION as TIME_OFFSET_SEARCH_SCHEMA_VERSION,
    SUPPORTED_IMPLEMENTATION_REVISIONS as TIME_OFFSET_SUPPORTED_REVISIONS,
    apply_sensor_time_offset,
    estimate_sensor_time_offset,
    failed_sensor_result,
    fixed_zero_sensor_result,
    offset_values as time_offset_values,
    search_configuration as time_offset_search_configuration,
    sign_convention as time_offset_sign_convention,
)
from posetestbot.io.atomic import atomic_write_json
from posetestbot.io.artifacts import (
    ARUCO_DETECTIONS,
    ARUCO_POSE_ESTIMATION,
    CALIBRATION_CANDIDATES,
    CALIBRATION_OBSERVATIONS,
    CALIBRATION_PROFILES,
    CALIBRATION_PROFILES_FROM_OBSERVATIONS,
    CALIBRATION_PROFILES_SOLVED,
    CALIBRATION_SOLVER_REPORT,
    CALIBRATION_TARGET,
    CALIBRATION_VALIDATION_REPORT,
    DATASET_MANIFEST,
    DEPTH_DIR,
    FRAME_METADATA_JSONL,
    INTRINSIC_COMPARISON,
    INTRINSIC_CALIBRATION_PROFILES,
    MATCH_ROBOT_EE_POSES,
    RAW_ROBOT_EE_POSES,
    RGB_DIR,
    RUN_CONFIG,
    SYNC_QUALITY_REPORT,
    TIME_OFFSET_SEARCH,
)
from posetestbot.io.manifest import (
    discover_sensor_records,
    load_or_create_run_manifest,
    upsert_stage,
)
from posetestbot.pipeline.run_config import (
    load_run_config_for_run_root,
    run_config_lock,
    validate_run_config,
)
from posetestbot.sensors.contracts import CameraIntrinsics, MountingMode, SensorType
from posetestbot.sync.non_destructive import (
    indexed_robot_poses,
    load_frame_metadata,
    load_robot_poses,
    synchronize_run,
)
from posetestbot.sync.quality import build_sync_quality_report


ATTEMPT_SCHEMA_VERSION = "calibration_attempt.v1"
REQUEST_SCHEMA_VERSION = "calibration_attempt_request.v1"
PROGRESS_SCHEMA_VERSION = "calibration_attempt_progress.v1"
PROMOTION_SCHEMA_VERSION = "calibration_attempt_promotion.v1"
PROMOTION_REQUEST_SCHEMA_VERSION = "calibration_promotion_request.v1"
ATTEMPT_DIRECTORY = Path("processed") / "calibration"
REQUEST_FILE = "request.json"
PROGRESS_FILE = "progress.json"
PNP_CANDIDATES_FILE = "pnp_candidates.json"
EXTRINSIC_CANDIDATES_FILE = "extrinsic_candidates.json"
RANKING_FILE = "ranking.json"
CHECKS_FILE = "checks.json"
CANDIDATE_PROFILES_FILE = "candidate_profiles.json"
PROMOTION_REQUEST_FILE = "promotion_request.json"
PROMOTION_FILE = "promotion.json"
TARGET_BUNDLE_DIRECTORY = "target_bundle"
ATTEMPT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
ATTEMPT_MIN_MOTION_POSES = 4
ATTEMPT_MIN_TRANSLATION_SPAN_MM = 20.0
ATTEMPT_MIN_ROTATION_SPAN_DEG = 5.0
ATTEMPT_MIN_TARGET_MARKER_COVERAGE_RATIO = 0.5
ATTEMPT_MIN_TARGET_ROW_COVERAGE_RATIO = 0.6
ATTEMPT_MIN_TARGET_COLUMN_COVERAGE_RATIO = 0.6
ATTEMPT_SYNC_DELTA_MS = 0.0
ATTEMPT_MAX_NEAREST_POSE_DELTA_MS = 20.0
ATTEMPT_TIMESTAMP_SOURCE = "sensor"
ATTEMPT_ROBOT_TIMESTAMP_SOURCE = "host_wall"
ATTEMPT_REALSENSE_TIMESTAMP_DOMAIN = "global_time"
ATTEMPT_INTRINSIC_MIN_HOLDOUT_VIEWS = 5
ATTEMPT_INTRINSIC_MAX_TRAINING_VIEWS = 45
ATTEMPT_INTRINSIC_MAX_HOLDOUT_VIEWS = 15
ATTEMPT_INTRINSIC_HOLDOUT_FRACTION = 0.10
ATTEMPT_INTRINSIC_TEMPORAL_GUARD_VIEWS = 5
ATTEMPT_INTRINSIC_DESCRIPTOR_CORNER_SCALE = 0.03
ATTEMPT_INTRINSIC_DESCRIPTOR_GUARD_DISTANCE = 1.0
ATTEMPT_INTRINSIC_MIN_ABSOLUTE_IMPROVEMENT_PX = 0.05
ATTEMPT_INTRINSIC_MIN_RELATIVE_IMPROVEMENT = 0.05
ATTEMPT_INTRINSIC_MAX_FOCAL_DELTA_RATIO = 0.10
ATTEMPT_INTRINSIC_MAX_PRINCIPAL_DELTA_RATIO = 0.05
ATTEMPT_INTRINSIC_MAX_ASPECT_DELTA_RATIO = 0.05
# Candidate score is normalized by the 10 mm / 5 degree residual gates. This
# band is one percent of that combined acceptance budget (0.1 mm if translation
# alone changes, or 0.05 degree if rotation alone changes).
JOINT_INDIVIDUAL_SCORE_EQUIVALENCE_TOLERANCE = 0.01
# Six decimals in normalized closure space correspond to 0.00001 mm for a
# translation-only difference or 0.000005 degree for a rotation-only
# difference. Both are far below physical calibration significance, so smaller
# solver differences intentionally fall through to canonical method ordering.
JOINT_RANKING_NUMERIC_DECIMALS = 6
JOINT_CLOSURE_SCORE_EQUIVALENCE_TOLERANCE = 10 ** (-JOINT_RANKING_NUMERIC_DECIMALS)
PROMOTION_TRANSFORM_TOLERANCE_MM = 1e-6
# Reconstructing a JSON quaternion into a matrix can introduce roughly
# 2e-6 degrees of acos round-off even when both records describe one transform.
PROMOTION_TRANSFORM_TOLERANCE_DEG = 1e-5
DEFAULT_INTRINSICS_POLICY = "compare_factory_opencv"
INTRINSICS_POLICIES = {
    "compare_factory_opencv": (
        "Compare captured factory intrinsics with a gated OpenCV calibration"
    ),
    "reuse_compatible_or_factory": (
        "Reuse an exact compatible profile, otherwise captured factory intrinsics"
    ),
}
PHASES = (
    ("prepare_data", "Prepare data"),
    ("estimate_target_poses", "Estimate target poses"),
    ("estimate_time_offsets", "Estimate time alignment"),
    ("compare_robot_camera_solutions", "Compare robot-camera solutions"),
    ("validate_and_rank", "Validate and rank"),
)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _attempt_timestamp_policy(
    sensors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    per_sensor: dict[str, dict[str, Any]] = {}
    for sensor in sensors:
        sensor_key = str(sensor.get("sensor_key") or sensor.get("folder"))
        if str(sensor.get("sensor_type")) == SensorType.REALSENSE_D435.value:
            selected = {
                "frame_timestamp_source": ATTEMPT_TIMESTAMP_SOURCE,
                "robot_timestamp_source": ATTEMPT_ROBOT_TIMESTAMP_SOURCE,
                "required_frame_timestamp_domain": (ATTEMPT_REALSENSE_TIMESTAMP_DOMAIN),
                "timestamp_fallback_allowed": False,
            }
        else:
            selected = {
                "frame_timestamp_source": "host_received",
                "robot_timestamp_source": "host_received",
                "required_frame_timestamp_domain": None,
                "timestamp_fallback_allowed": False,
            }
        per_sensor[sensor_key] = selected
    frame_sources = {item["frame_timestamp_source"] for item in per_sensor.values()}
    robot_sources = {item["robot_timestamp_source"] for item in per_sensor.values()}
    required_domains = {
        item["required_frame_timestamp_domain"] for item in per_sensor.values()
    }
    return {
        "frame_timestamp_source": (
            next(iter(frame_sources)) if len(frame_sources) == 1 else "per_sensor"
        ),
        "robot_timestamp_source": (
            next(iter(robot_sources)) if len(robot_sources) == 1 else "per_sensor"
        ),
        "required_frame_timestamp_domain": (
            next(iter(required_domains)) if len(required_domains) == 1 else "per_sensor"
        ),
        "timestamp_fallback_allowed": False,
        "per_sensor": per_sensor,
    }


def _timestamp_policy_for_sensor(
    policy: Mapping[str, Any], sensor: Mapping[str, Any]
) -> dict[str, Any]:
    sensor_key = str(sensor.get("sensor_key") or sensor.get("folder"))
    per_sensor = policy.get("per_sensor")
    if isinstance(per_sensor, Mapping):
        selected = per_sensor.get(sensor_key)
        if isinstance(selected, Mapping):
            return dict(selected)
    return {
        key: policy.get(key)
        for key in (
            "frame_timestamp_source",
            "robot_timestamp_source",
            "required_frame_timestamp_domain",
            "timestamp_fallback_allowed",
            "per_sensor",
        )
    }


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _is_contained(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _calibration_timestamp_preflight(
    run_root: Path,
    sensors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail closed unless selected cameras provide one coherent timebase."""

    policy = _attempt_timestamp_policy(sensors)
    errors: list[str] = []
    evidence: list[dict[str, Any]] = []
    robot_paths: set[Path] = set()
    for sensor in sensors:
        sensor_policy = _timestamp_policy_for_sensor(policy, sensor)
        required_domain = sensor_policy["required_frame_timestamp_domain"]
        if required_domain is None:
            continue
        sensor_key = str(sensor.get("sensor_key") or sensor.get("folder"))
        folder = run_root / str(sensor.get("folder", ""))
        metadata_path = folder / FRAME_METADATA_JSONL
        if not _is_contained(metadata_path, run_root):
            errors.append(f"{sensor_key}: frame metadata escapes the run root")
            continue
        try:
            records = load_frame_metadata(folder)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{sensor_key}: invalid frame timestamp evidence: {exc}")
            continue
        if not records:
            errors.append(f"{sensor_key}: frame timestamp evidence is empty")
            continue
        missing_sensor_timestamp = sum(
            item.get("sensor_timestamp_ns") is None for item in records
        )
        domains: dict[str, int] = {}
        for item in records:
            domain = str(item.get("color_timestamp_domain") or "missing")
            domains[domain] = domains.get(domain, 0) + 1
        if missing_sensor_timestamp:
            errors.append(
                f"{sensor_key}: {missing_sensor_timestamp} frame(s) lack "
                "sensor_timestamp_ns"
            )
        if set(domains) != {required_domain}:
            errors.append(
                f"{sensor_key}: RealSense color timestamps must all use "
                f"{required_domain}; observed {domains}"
            )
        evidence.append(
            {
                "sensor_key": sensor_key,
                "frame_metadata_path": _relative(metadata_path, run_root),
                "frame_count": len(records),
                "sensor_timestamp_missing_count": missing_sensor_timestamp,
                "color_timestamp_domain_counts": domains,
            }
        )
        robot_path = run_root / str(sensor.get("robot_pose_path") or RAW_ROBOT_EE_POSES)
        if not _is_contained(robot_path, run_root):
            errors.append(
                f"{sensor_key}: robot timestamp evidence escapes the run root"
            )
        else:
            robot_paths.add(robot_path)

    robot_evidence: list[dict[str, Any]] = []
    for robot_path in sorted(robot_paths):
        try:
            poses = _read_json(robot_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid robot timestamp evidence: {exc}")
            continue
        missing_host_wall = sum(
            not isinstance(item, Mapping) or item.get("host_wall_timestamp_ns") is None
            for item in poses.values()
        )
        if not poses:
            errors.append("robot timestamp evidence is empty")
        if missing_host_wall:
            errors.append(
                f"{_relative(robot_path, run_root)}: {missing_host_wall} robot "
                "pose(s) lack host_wall_timestamp_ns"
            )
        robot_evidence.append(
            {
                "path": _relative(robot_path, run_root),
                "pose_count": len(poses),
                "host_wall_timestamp_missing_count": missing_host_wall,
            }
        )
    if errors:
        raise ValueError(
            "Strict calibration timestamp preflight failed: " + "; ".join(errors)
        )
    return {
        **policy,
        "sensors": evidence,
        "robot_pose_artifacts": robot_evidence,
    }


def validate_attempt_id(attempt_id: str) -> str:
    value = str(attempt_id).strip().lower()
    if not ATTEMPT_ID_PATTERN.fullmatch(value):
        raise ValueError("attempt_id must contain 32 lowercase hexadecimal characters")
    return value


def calibration_attempt_root(run_root: str | Path, attempt_id: str) -> Path:
    return Path(run_root) / ATTEMPT_DIRECTORY / validate_attempt_id(attempt_id)


def _attempt_artifact_reference(attempt_id: str, filename: str) -> str:
    return (ATTEMPT_DIRECTORY / validate_attempt_id(attempt_id) / filename).as_posix()


def _sensor_key(sensor_type: str, device_id: str) -> str:
    return f"{sensor_type}:{device_id}"


def _manifest_sensor_records(root: Path) -> dict[str, dict[str, Any]]:
    path = root / DATASET_MANIFEST
    if not _is_contained(path, root) or not path.is_file():
        return {}
    try:
        manifest = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    result = {}
    for raw in manifest.get("sensors", []):
        if not isinstance(raw, Mapping):
            continue
        folder = str(raw.get("folder", ""))
        if folder:
            result[folder] = dict(raw)
    return result


def discover_calibration_cameras(run_root: str | Path) -> list[dict[str, Any]]:
    """Return captured camera identities without opening hardware."""

    root = Path(run_root)
    config = load_run_config_for_run_root(root)
    configured = [
        dict(item)
        for item in config.get("capture", {}).get("sensors", [])
        if isinstance(item, Mapping)
    ]
    enabled_configured = [
        item for item in configured if item.get("enabled", True) is True
    ]
    manifest_records = _manifest_sensor_records(root)
    cameras: list[dict[str, Any]] = []
    for discovered in discover_sensor_records(root):
        folder = str(discovered["folder"])
        record = {**dict(discovered), **manifest_records.get(folder, {})}
        sensor_type = str(record.get("sensor_type", ""))
        device_id = str(record.get("device_id", ""))
        candidate_folder = root / folder
        contained = _is_contained(candidate_folder, root)
        folder_path = candidate_folder.resolve() if contained else candidate_folder
        matching_config = next(
            (
                item
                for item in configured
                if str(item.get("sensor_type")) == sensor_type
                and str(item.get("device_id")) == device_id
            ),
            None,
        )
        if (
            matching_config is not None
            and matching_config.get("enabled", True) is not True
        ):
            continue
        if matching_config is None:
            same_family = [
                item
                for item in enabled_configured
                if str(item.get("sensor_type")) == sensor_type
            ]
            if len(same_family) == 1:
                matching_config = same_family[0]
                device_id = str(matching_config.get("device_id") or device_id)
        key = _sensor_key(sensor_type, device_id)
        errors = []
        if not sensor_type or not device_id:
            errors.append("missing stable sensor type/device identity")
        if not contained:
            errors.append("captured sensor folder escapes the run root")
        rgb = folder_path / RGB_DIR
        depth = folder_path / DEPTH_DIR
        if contained:
            rgb_contained = _is_contained(rgb, root)
            depth_contained = _is_contained(depth, root)
            if not rgb_contained or not rgb.is_dir() or not any(rgb.glob("*.png")):
                errors.append("missing captured RGB frames")
            if (
                not depth_contained
                or not depth.is_dir()
                or not any(depth.glob("*.png"))
            ):
                errors.append("missing captured depth frames")
            if (
                not _is_contained(folder_path / FRAME_METADATA_JSONL, root)
                or not (folder_path / FRAME_METADATA_JSONL).is_file()
            ):
                errors.append("missing frame timestamp evidence")
        sensor_robot_path = folder_path / RAW_ROBOT_EE_POSES
        robot_path = (
            sensor_robot_path
            if contained
            and _is_contained(sensor_robot_path, root)
            and sensor_robot_path.is_file()
            else root / RAW_ROBOT_EE_POSES
        )
        if not _is_contained(robot_path, root) or not robot_path.is_file():
            errors.append(f"missing {RAW_ROBOT_EE_POSES}")
        else:
            try:
                robot = _read_json(robot_path)
                if not robot:
                    errors.append("robot-pose evidence is empty")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"invalid robot-pose evidence: {exc}")
        cameras.append(
            {
                "sensor_key": key,
                "sensor_type": sensor_type,
                "device_id": device_id,
                "sensor_name": folder_path.name,
                "folder": folder,
                "display_name": str(
                    (matching_config or {}).get("display_name")
                    or record.get("display_name")
                    or folder_path.name
                ),
                "current_mounting_mode": (matching_config or {}).get("mounting_mode"),
                "data_ready": not errors,
                "errors": errors,
                "frame_metadata": (
                    contained
                    and _is_contained(folder_path / FRAME_METADATA_JSONL, root)
                    and (folder_path / FRAME_METADATA_JSONL).is_file()
                ),
                "robot_pose_path": _relative(robot_path, root),
            }
        )
    identity_counts: dict[str, int] = {}
    for camera in cameras:
        sensor_key = str(camera["sensor_key"])
        identity_counts[sensor_key] = identity_counts.get(sensor_key, 0) + 1
    for camera in cameras:
        if identity_counts[str(camera["sensor_key"])] <= 1:
            continue
        camera["errors"] = [
            *camera["errors"],
            "duplicate stable sensor identity",
        ]
        camera["data_ready"] = False
    return cameras


def _saved_targets(run_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "target_id": item.get("target_id"),
            "display_name": item.get("display_name") or item.get("target_id"),
            "created_at": item.get("created_at"),
            "geometry_sha256": item.get("geometry_sha256"),
            "valid": item.get("valid", False),
            "error": item.get("error"),
            "selected": item.get("selected", False),
            "target": item.get("target"),
        }
        for item in list_target_bundles(
            library_root=default_target_library_root(), run_root=run_root
        )
    ]


def list_calibration_attempts(run_root: str | Path) -> list[dict[str, Any]]:
    root = Path(run_root)
    parent = root / ATTEMPT_DIRECTORY
    if not parent.is_dir():
        return []
    records = []
    for child in parent.iterdir():
        if not child.is_dir() or not ATTEMPT_ID_PATTERN.fullmatch(child.name):
            continue
        try:
            request_value = _read_json(child / REQUEST_FILE)
            progress = _read_json(child / PROGRESS_FILE)
            _validate_attempt_identity(root, child.name, request_value, progress)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        ranking = (
            _read_json(child / RANKING_FILE)
            if (child / RANKING_FILE).is_file()
            else None
        )
        promotion = (
            _read_json(child / PROMOTION_FILE)
            if (child / PROMOTION_FILE).is_file()
            else None
        )
        records.append(
            {
                "attempt_id": child.name,
                "created_at": request_value.get("created_at"),
                "mode": request_value.get("mode"),
                "sensor_keys": request_value.get("sensor_keys", []),
                "target_id": request_value.get("target_id"),
                "status": progress.get("status"),
                "recommended_camera_count": (
                    int(ranking.get("recommended_camera_count", 0)) if ranking else 0
                ),
                "promotion": promotion,
            }
        )
    return sorted(
        records, key=lambda item: str(item.get("created_at", "")), reverse=True
    )


def calibration_setup(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root)
    cameras = discover_calibration_cameras(root)
    attempts = list_calibration_attempts(root)
    return {
        "schema_version": "calibration_setup.v1",
        "run_root": root.as_posix(),
        "cameras": [item for item in cameras if item["data_ready"]],
        "unavailable_cameras": [item for item in cameras if not item["data_ready"]],
        "saved_targets": _saved_targets(root),
        "modes": [
            {
                "id": "eye_in_hand",
                "label": "Robot-mounted camera (eye-in-hand)",
                "primary_transform": "camera → robot_flange",
                "target_mounting": "stationary relative to template_base",
            },
            {
                "id": "eye_to_hand",
                "label": "Static camera (eye-to-hand)",
                "primary_transform": "camera → template_base",
                "target_mounting": "rigidly attached to robot_flange",
            },
        ],
        "solver": {
            "policies": [
                {
                    "id": "auto_compare",
                    "label": "Auto compare — recommended",
                }
            ],
            "default_policy": "auto_compare",
            "default_pnp_methods": list(PNP_METHOD_ORDER),
            "pnp_methods": list(PNP_METHOD_ORDER),
            "default_extrinsic_methods": list(EXTRINSIC_METHOD_ORDER),
            "extrinsic_methods": list(EXTRINSIC_METHOD_ORDER),
            "intrinsics_policy": DEFAULT_INTRINSICS_POLICY,
            "intrinsics_policies": [
                {"id": policy_id, "label": label}
                for policy_id, label in INTRINSICS_POLICIES.items()
            ],
            "synchronization": {
                "default_policy": "auto_offset",
                "policies": [
                    {
                        "id": "auto_offset",
                        "label": "Auto-estimate robot-pose offset — recommended",
                        "description": (
                            "Estimate effective per-camera latency with "
                            "motion-disjoint cross-validation."
                        ),
                    },
                    {
                        "id": "fixed_zero",
                        "label": "Use captured timestamps (0 ms)",
                        "description": (
                            "Pair camera and robot evidence without an inferred "
                            "time offset."
                        ),
                    },
                ],
                "search": time_offset_search_configuration(),
                "sign_convention": time_offset_sign_convention(),
            },
            "thresholds": {
                "min_inliers": 6,
                "min_pnp_common_inliers": DEFAULT_MIN_PNP_COMMON_INLIERS,
                "min_pnp_common_inlier_ratio": (DEFAULT_MIN_PNP_COMMON_INLIER_RATIO),
                "max_pnp_all_point_mean_reprojection_error_px": (
                    DEFAULT_MAX_PNP_ALL_POINT_MEAN_ERROR_PX
                ),
                "min_pnp_supported_markers": (DEFAULT_MIN_PNP_SUPPORTED_MARKERS),
                "min_pnp_supported_corners_per_marker": (
                    DEFAULT_MIN_PNP_SUPPORTED_CORNERS_PER_MARKER
                ),
                "min_pnp_grid_rows": DEFAULT_MIN_PNP_GRID_ROWS,
                "min_pnp_grid_columns": DEFAULT_MIN_PNP_GRID_COLUMNS,
                "min_target_marker_coverage_ratio": (
                    ATTEMPT_MIN_TARGET_MARKER_COVERAGE_RATIO
                ),
                "min_target_row_coverage_ratio": (
                    ATTEMPT_MIN_TARGET_ROW_COVERAGE_RATIO
                ),
                "min_target_column_coverage_ratio": (
                    ATTEMPT_MIN_TARGET_COLUMN_COVERAGE_RATIO
                ),
                "min_accepted_views": DEFAULT_MIN_ACCEPTED_VIEWS,
                "min_coverage_cells": DEFAULT_MIN_COVERAGE_CELLS,
                "max_per_view_reprojection_error_px": DEFAULT_MAX_VIEW_ERROR_PX,
                "max_intrinsic_rms_reprojection_error_px": DEFAULT_MAX_RMS_PX,
                "min_motion_poses": ATTEMPT_MIN_MOTION_POSES,
                "min_translation_span_mm": ATTEMPT_MIN_TRANSLATION_SPAN_MM,
                "min_rotation_span_deg": ATTEMPT_MIN_ROTATION_SPAN_DEG,
                "min_rotation_axis_angle_deg": (DEFAULT_MIN_ROTATION_AXIS_ANGLE_DEG),
                "min_rotation_axis_second_to_first_ratio": (
                    DEFAULT_MIN_ROTATION_AXIS_SINGULAR_RATIO
                ),
                "max_observations_per_motion": (DEFAULT_MAX_OBSERVATIONS_PER_MOTION),
                "max_nearest_pose_delta_ms": (ATTEMPT_MAX_NEAREST_POSE_DELTA_MS),
                "max_mean_translation_mm": 10.0,
                "max_mean_rotation_deg": 5.0,
                "max_outlier_ratio": 0.25,
                "joint_individual_score_equivalence_tolerance": (
                    JOINT_INDIVIDUAL_SCORE_EQUIVALENCE_TOLERANCE
                ),
            },
        },
        "latest_attempt": attempts[0] if attempts else None,
    }


def _target_selection(bundle: Mapping[str, Any]) -> dict[str, Any]:
    files = bundle.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("Calibration-target bundle file evidence is invalid")
    return {
        "target_id": bundle["target_id"],
        "bundle_path": f"{LIBRARY_DIRECTORY}/{bundle['target_id']}",
        "source_sha256": files["source"]["sha256"],
        "spec_sha256": files["target"]["sha256"],
        "pdf_sha256": files["pdf"]["sha256"],
        "configuration_sha256": bundle["configuration_sha256"],
        "geometry_sha256": bundle["geometry_sha256"],
        "placement": {"mode": "unknown"},
    }


def validate_attempt_request(
    run_root: str | Path,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(run_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Run root not found: {root}")
    mode = str(value.get("mode", ""))
    if mode not in {"eye_in_hand", "eye_to_hand"}:
        raise ValueError("mode must be eye_in_hand or eye_to_hand")
    raw_keys = value.get("sensor_keys")
    if not isinstance(raw_keys, list) or not raw_keys:
        raise ValueError("sensor_keys must be a non-empty list")
    sensor_keys = [str(item) for item in raw_keys]
    if len(sensor_keys) != len(set(sensor_keys)):
        raise ValueError("sensor_keys must not contain duplicates")
    cameras = {item["sensor_key"]: item for item in discover_calibration_cameras(root)}
    unknown = sorted(set(sensor_keys) - cameras.keys())
    if unknown:
        raise ValueError("Unknown sensor key(s): " + ", ".join(unknown))
    unavailable = [key for key in sensor_keys if not cameras[key]["data_ready"]]
    if unavailable:
        messages = [
            f"{key}: {', '.join(cameras[key]['errors'])}" for key in unavailable
        ]
        raise ValueError("Selected cameras are not data-ready: " + "; ".join(messages))
    selected_cameras = [cameras[key] for key in sensor_keys]
    expected_mounting_mode = "eye_in_hand" if mode == "eye_in_hand" else "static"
    mounting_mismatches = [
        str(camera["sensor_key"])
        for camera in selected_cameras
        if str(camera.get("current_mounting_mode") or "") != expected_mounting_mode
    ]
    if mounting_mismatches:
        raise ValueError(
            f"{mode} calibration requires cameras configured as "
            f"{expected_mounting_mode}; update run setup or remove: "
            + ", ".join(mounting_mismatches)
        )
    timestamp_policy = _calibration_timestamp_preflight(root, selected_cameras)
    target_id = str(value.get("target_id", ""))
    try:
        selected_target = validate_run_target_selection(root)
    except FileNotFoundError as exc:
        raise ValueError(
            "Select the exact printed calibration grid in workflow step 2 before analysis"
        ) from exc
    active_id = str(selected_target["target_id"])
    if active_id != target_id:
        blockers = [
            item
            for item in replacement_blockers(root)
            if not item.startswith(f"{ATTEMPT_DIRECTORY.as_posix()}/")
        ]
        if blockers:
            raise CalibrationTargetConflict(
                "The calibration target conflicts with existing target-dependent artifacts; create a new run.",
                blockers=blockers,
            )
        raise CalibrationTargetConflict(
            "The requested grid is not the grid selected in workflow step 2; change the run selection first.",
            blockers=[],
        )
    run_target_library = root / LIBRARY_DIRECTORY
    bundle = validate_target_bundle(
        run_target_library / target_id,
        library_root=run_target_library,
    )
    solver_policy = str(value.get("solver_policy", "auto_compare"))
    if solver_policy != "auto_compare":
        raise ValueError("solver_policy must be auto_compare")
    intrinsics_policy = str(value.get("intrinsics_policy", DEFAULT_INTRINSICS_POLICY))
    if intrinsics_policy not in INTRINSICS_POLICIES:
        raise ValueError(
            "intrinsics_policy must be one of: " + ", ".join(INTRINSICS_POLICIES)
        )
    synchronization_policy = str(value.get("synchronization_policy", "auto_offset"))
    if synchronization_policy not in SYNCHRONIZATION_POLICIES:
        raise ValueError(
            "synchronization_policy must be one of: "
            + ", ".join(SYNCHRONIZATION_POLICIES)
        )
    pnp_methods = value.get("pnp_methods", list(PNP_METHOD_ORDER))
    extrinsic_methods = value.get("extrinsic_methods", list(EXTRINSIC_METHOD_ORDER))
    if not isinstance(pnp_methods, list) or not pnp_methods:
        raise ValueError("pnp_methods must be a non-empty list")
    if not isinstance(extrinsic_methods, list) or not extrinsic_methods:
        raise ValueError("extrinsic_methods must be a non-empty list")
    pnp_methods = [str(item).upper() for item in pnp_methods]
    extrinsic_methods = [str(item).lower() for item in extrinsic_methods]
    if len(pnp_methods) != len(set(pnp_methods)):
        raise ValueError("pnp_methods must not contain duplicates")
    if len(extrinsic_methods) != len(set(extrinsic_methods)):
        raise ValueError("extrinsic_methods must not contain duplicates")
    unsupported_pnp = sorted(set(pnp_methods) - set(PNP_METHOD_ORDER))
    unsupported_extrinsic = sorted(set(extrinsic_methods) - set(EXTRINSIC_METHOD_ORDER))
    if unsupported_pnp:
        raise ValueError(
            "Unsupported board-level PnP method(s): " + ", ".join(unsupported_pnp)
        )
    if unsupported_extrinsic:
        raise ValueError(
            "Unsupported extrinsic method(s): " + ", ".join(unsupported_extrinsic)
        )
    if (
        synchronization_policy == "auto_offset"
        and DEFAULT_REFERENCE_PNP_METHOD not in pnp_methods
    ):
        raise ValueError(
            "auto_offset synchronization requires the fixed reference PnP "
            f"method {DEFAULT_REFERENCE_PNP_METHOD}"
        )
    return {
        "mode": mode,
        "sensor_keys": sensor_keys,
        "sensors": selected_cameras,
        "timestamp_policy": timestamp_policy,
        "target_id": target_id,
        "target": normalize_calibration_target_spec(bundle["target"]),
        "target_bundle": {
            "target_id": bundle["target_id"],
            "display_name": bundle.get("display_name"),
            "configuration_sha256": bundle["configuration_sha256"],
            "geometry_sha256": bundle["geometry_sha256"],
            "files": bundle["files"],
            "selection": _target_selection(bundle),
            "source_path": bundle["bundle_path"],
        },
        "target_mounting": {
            "from": "aruco_grid",
            "to": "template_base" if mode == "eye_in_hand" else "robot_flange",
            "state": "estimated",
        },
        "solver_policy": solver_policy,
        "pnp_methods": pnp_methods,
        "extrinsic_methods": extrinsic_methods,
        "intrinsics_policy": intrinsics_policy,
        "synchronization_policy": synchronization_policy,
        "synchronization_search": time_offset_search_configuration(),
        "synchronization_implementation_revision": (
            TIME_OFFSET_IMPLEMENTATION_REVISION
        ),
    }


def _initial_progress(attempt_id: str) -> dict[str, Any]:
    return {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "status": "queued",
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "current_phase": None,
        "message": "Calibration attempt queued.",
        "phases": [
            {"id": phase_id, "label": label, "status": "pending"}
            for phase_id, label in PHASES
        ],
    }


def _validate_attempt_identity(
    run_root: Path,
    attempt_id: str,
    request_value: Mapping[str, Any],
    progress: Mapping[str, Any],
) -> None:
    if request_value.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise ValueError("Unsupported calibration attempt request schema")
    if progress.get("schema_version") != PROGRESS_SCHEMA_VERSION:
        raise ValueError("Unsupported calibration attempt progress schema")
    if (
        request_value.get("attempt_id") != attempt_id
        or progress.get("attempt_id") != attempt_id
    ):
        raise ValueError("Calibration attempt identity does not match its directory")
    recorded_root = Path(str(request_value.get("run_root", ""))).resolve()
    if recorded_root != run_root.resolve():
        raise ValueError("Calibration attempt belongs to a different run root")


def create_calibration_attempt(
    run_root: str | Path,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(run_root)
    normalized = validate_attempt_request(root, value)
    attempt_id = uuid.uuid4().hex
    destination = calibration_attempt_root(root, attempt_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{attempt_id}.{uuid.uuid4().hex}.tmp"
    staging.mkdir(parents=False, exist_ok=False)
    created_at = utc_now_iso()
    request_value = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "run_root": root.as_posix(),
        "created_at": created_at,
        **normalized,
    }
    source_bundle = Path(str(normalized["target_bundle"]["source_path"]))
    try:
        shutil.copytree(source_bundle, staging / TARGET_BUNDLE_DIRECTORY)
        atomic_write_json(staging / REQUEST_FILE, request_value)
        atomic_write_json(staging / PROGRESS_FILE, _initial_progress(attempt_id))
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return request_value


def _update_progress(
    attempt_root: Path,
    *,
    status: str | None = None,
    phase: str | None = None,
    phase_status: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    progress = _read_json(attempt_root / PROGRESS_FILE)
    recorded_phases = {
        str(item.get("id")): item
        for item in progress.get("phases", [])
        if isinstance(item, Mapping) and item.get("id")
    }
    progress["phases"] = [
        {
            **dict(recorded_phases.get(phase_id, {})),
            "id": phase_id,
            "label": label,
            "status": str(recorded_phases.get(phase_id, {}).get("status") or "pending"),
        }
        for phase_id, label in PHASES
    ]
    if status is not None:
        progress["status"] = status
    if phase is not None:
        progress["current_phase"] = phase
        for item in progress["phases"]:
            if item["id"] == phase and phase_status is not None:
                item["status"] = phase_status
    if message is not None:
        progress["message"] = message
    progress["updated_at"] = utc_now_iso()
    atomic_write_json(attempt_root / PROGRESS_FILE, progress)
    return progress


def record_attempt_job(
    run_root: str | Path,
    attempt_id: str,
    *,
    job_id: str,
    kind: str,
) -> None:
    attempt_root = calibration_attempt_root(run_root, attempt_id)
    if kind == "calculation":
        progress = _read_json(attempt_root / PROGRESS_FILE)
        progress["job_id"] = job_id
        progress["updated_at"] = utc_now_iso()
        atomic_write_json(attempt_root / PROGRESS_FILE, progress)
        return
    if kind == "promotion":
        promotion = _read_json(attempt_root / PROMOTION_FILE)
        promotion["job_id"] = job_id
        atomic_write_json(attempt_root / PROMOTION_FILE, promotion)
        return
    raise ValueError("Calibration attempt job kind must be calculation or promotion")


def record_attempt_job_submission_failure(
    run_root: str | Path,
    attempt_id: str,
    *,
    kind: str,
    error: Exception,
) -> None:
    """Make a synchronous queue failure visible without losing the attempt."""

    attempt_root = calibration_attempt_root(run_root, attempt_id)
    message = f"{type(error).__name__}: {error}"
    if kind == "calculation":
        progress = _read_json(attempt_root / PROGRESS_FILE)
        progress.update(
            {
                "status": "failed",
                "updated_at": utc_now_iso(),
                "message": message,
                "failure_stage": "job_submission",
            }
        )
        atomic_write_json(attempt_root / PROGRESS_FILE, progress)
        return
    if kind == "promotion":
        promotion = _read_json(attempt_root / PROMOTION_FILE)
        promotion.update(
            {
                "status": "failed",
                "ended_at": utc_now_iso(),
                "error": message,
                "failure_stage": "job_submission",
            }
        )
        atomic_write_json(attempt_root / PROMOTION_FILE, promotion)
        return
    raise ValueError("Calibration attempt job kind must be calculation or promotion")


def _intrinsic_deltas(
    factory: Mapping[str, Any],
    manual: Mapping[str, Any],
) -> dict[str, Any]:
    factory_native = factory["native"]
    manual_native = manual["native"]
    factory_k = np.asarray(factory_native["cam_K"], dtype=float).reshape(3, 3)
    manual_k = np.asarray(manual_native["cam_K"], dtype=float).reshape(3, 3)
    matrix_delta = manual_k - factory_k
    factory_model = str(factory_native.get("distortion_model", "brown_conrady"))
    manual_model = str(manual_native.get("distortion_model", "brown_conrady"))
    distortion_comparable = factory_model == manual_model
    distortion_delta: np.ndarray | None = None
    if distortion_comparable:
        distortion_delta = np.asarray(
            manual_native["distortion"], dtype=float
        ) - np.asarray(factory_native["distortion"], dtype=float)
    return {
        "manual_minus_factory_cam_K": matrix_delta.reshape(-1).tolist(),
        "max_abs_cam_K_delta": float(np.max(np.abs(matrix_delta))),
        "focal_length_delta_px": [
            float(manual_k[0, 0] - factory_k[0, 0]),
            float(manual_k[1, 1] - factory_k[1, 1]),
        ],
        "principal_point_delta_px": [
            float(manual_k[0, 2] - factory_k[0, 2]),
            float(manual_k[1, 2] - factory_k[1, 2]),
        ],
        "factory_distortion_model": factory_model,
        "manual_distortion_model": manual_model,
        "distortion_coefficients_comparable": distortion_comparable,
        "manual_minus_factory_distortion": (
            distortion_delta.tolist() if distortion_delta is not None else None
        ),
        "max_abs_distortion_delta": (
            float(np.max(np.abs(distortion_delta)))
            if distortion_delta is not None
            else None
        ),
    }


def _intrinsic_natural_key(value: str) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", value)
        if part
    )


def _intrinsic_descriptor_distance(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> float:
    return float(
        np.linalg.norm(
            np.asarray(first["descriptor"], dtype=float)
            - np.asarray(second["descriptor"], dtype=float)
        )
    )


def _intrinsic_quality_key(record: Mapping[str, Any]) -> tuple[int, float, int]:
    return (
        int(record["matched_corner_count"]),
        float(record["target_hull_area_ratio"]),
        -int(record["chronological_index"]),
    )


def _intrinsic_maximin_selection(
    candidates: Sequence[Mapping[str, Any]],
    count: int,
    *,
    seeds: Sequence[Mapping[str, Any]] = (),
) -> list[Mapping[str, Any]]:
    selected = list(dict.fromkeys(str(item["frame"]) for item in seeds))
    by_name = {str(item["frame"]): item for item in [*seeds, *candidates]}
    while len(selected) < count:
        available = [item for item in candidates if str(item["frame"]) not in selected]
        if not available:
            break
        references = [by_name[name] for name in selected]

        def selection_key(item: Mapping[str, Any]) -> tuple[float, int, float, int]:
            minimum_distance = (
                min(
                    _intrinsic_descriptor_distance(item, reference)
                    for reference in references
                )
                if references
                else math.inf
            )
            return (minimum_distance, *_intrinsic_quality_key(item))

        selected.append(str(max(available, key=selection_key)["frame"]))
    return [by_name[name] for name in selected[:count]]


def _intrinsic_views_are_separated(
    candidate: Mapping[str, Any],
    references: Sequence[Mapping[str, Any]],
    *,
    temporal_guard: int,
    descriptor_guard: float,
) -> bool:
    for reference in references:
        if (
            abs(
                int(candidate["chronological_index"])
                - int(reference["chronological_index"])
            )
            <= temporal_guard
        ):
            return False
        if (
            descriptor_guard > 0.0
            and _intrinsic_descriptor_distance(candidate, reference) < descriptor_guard
        ):
            return False
    return True


def _intrinsic_guarded_holdouts(
    candidates: Sequence[Mapping[str, Any]],
    count: int,
    *,
    protected_training: Sequence[Mapping[str, Any]],
    temporal_guard: int,
    descriptor_guard: float,
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    while len(selected) < count:
        references = [*protected_training, *selected]
        available = [
            item
            for item in candidates
            if str(item["frame"]) not in {str(value["frame"]) for value in selected}
            and _intrinsic_views_are_separated(
                item,
                references,
                temporal_guard=temporal_guard,
                descriptor_guard=descriptor_guard,
            )
        ]
        if not available:
            break

        def selection_key(item: Mapping[str, Any]) -> tuple[float, int, float, int]:
            minimum_distance = (
                min(
                    _intrinsic_descriptor_distance(item, reference)
                    for reference in references
                )
                if references
                else math.inf
            )
            return (minimum_distance, *_intrinsic_quality_key(item))

        selected.append(max(available, key=selection_key))
    return selected


def _intrinsic_detection_split(
    detections: Mapping[str, Any],
    target: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw_frames = detections.get("frames")
    if not isinstance(raw_frames, Mapping):
        raise ValueError("ArUco detections require a frames object")
    frames = {str(name): frame for name, frame in raw_frames.items()}
    if len(frames) != len(raw_frames):
        raise ValueError("ArUco detection frame names must be unique strings")
    image_size = detections.get("image_size")
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or any(float(value) <= 0.0 for value in image_size)
    ):
        raise ValueError(
            "ArUco detections require a positive [width, height] image_size"
        )
    width, height = (float(value) for value in image_size)
    _dictionary, board = opencv_grid_board(target)
    board_points = np.concatenate(
        [
            np.asarray(points, dtype=np.float64).reshape(-1, 3)
            for points in board.getObjPoints()
        ]
    )
    minimum = board_points[:, :2].min(axis=0)
    maximum = board_points[:, :2].max(axis=0)
    board_corners = np.asarray(
        [
            minimum,
            [maximum[0], minimum[1]],
            maximum,
            [minimum[0], maximum[1]],
        ],
        dtype=np.float64,
    ).reshape(1, 4, 2)

    records: list[dict[str, Any]] = []
    unusable_views: list[dict[str, str]] = []
    ordered_names = sorted(frames, key=_intrinsic_natural_key)
    for chronological_index, name in enumerate(ordered_names):
        frame = frames[name]
        if not isinstance(frame, Mapping):
            unusable_views.append({"frame": name, "reason": "invalid_detection_record"})
            continue
        points = _view_points(frame, board)
        if points is None:
            unusable_views.append(
                {"frame": name, "reason": "insufficient_matched_grid_corners"}
            )
            continue
        object_points, image_points = points
        try:
            homography, _mask = cv2.findHomography(
                np.asarray(object_points[:, :2], dtype=np.float64),
                np.asarray(image_points, dtype=np.float64),
                method=0,
            )
            if homography is None:
                raise ValueError("homography fit returned no matrix")
            projected = cv2.perspectiveTransform(
                board_corners,
                np.asarray(homography, dtype=np.float64),
            ).reshape(4, 2)
            if not np.all(np.isfinite(projected)):
                raise ValueError("projected board corners are non-finite")
        except (cv2.error, ValueError, TypeError) as exc:
            unusable_views.append(
                {"frame": name, "reason": f"projective_descriptor_unavailable: {exc}"}
            )
            continue
        centroid = np.asarray(image_points, dtype=float).mean(axis=0)
        hull = cv2.convexHull(np.asarray(image_points, dtype=np.float32))
        normalized_corners = projected / np.asarray([width, height])
        descriptor = (
            normalized_corners.reshape(-1) / ATTEMPT_INTRINSIC_DESCRIPTOR_CORNER_SCALE
        )
        records.append(
            {
                "frame": name,
                "chronological_index": chronological_index,
                "coverage_cell": _coverage_cell(centroid.tolist(), image_size),
                "matched_corner_count": len(image_points),
                "target_hull_area_ratio": float(
                    cv2.contourArea(hull) / (width * height)
                ),
                "projected_board_corners_normalized": (
                    normalized_corners.reshape(-1).astype(float).tolist()
                ),
                "descriptor": descriptor.astype(float).tolist(),
            }
        )

    by_cell: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        cell = record["coverage_cell"]
        if cell is not None:
            by_cell.setdefault(int(cell), []).append(record)
    represented_cells = sorted(by_cell)
    protected_training = [
        max(by_cell[cell], key=_intrinsic_quality_key) for cell in represented_cells
    ]
    protected_names = {str(item["frame"]) for item in protected_training}
    holdout_count = min(
        ATTEMPT_INTRINSIC_MAX_HOLDOUT_VIEWS,
        max(
            ATTEMPT_INTRINSIC_MIN_HOLDOUT_VIEWS,
            math.ceil(len(records) * ATTEMPT_INTRINSIC_HOLDOUT_FRACTION),
        ),
        max(0, len(records) - DEFAULT_MIN_ACCEPTED_VIEWS),
    )
    holdout_candidates = [
        item for item in records if str(item["frame"]) not in protected_names
    ]
    minimum_training_count = min(
        DEFAULT_MIN_ACCEPTED_VIEWS,
        max(0, len(records) - holdout_count),
    )
    required_training_cells = min(
        DEFAULT_MIN_COVERAGE_CELLS,
        len(represented_cells),
    )
    descriptor_options = (1.0, 0.75, 0.5, 0.0)
    guard_options = [
        (temporal_guard, descriptor_guard)
        for temporal_guard in range(ATTEMPT_INTRINSIC_TEMPORAL_GUARD_VIEWS, -1, -1)
        for descriptor_guard in descriptor_options
    ]
    guard_options.sort(
        key=lambda item: (
            (ATTEMPT_INTRINSIC_TEMPORAL_GUARD_VIEWS - item[0])
            / max(1, ATTEMPT_INTRINSIC_TEMPORAL_GUARD_VIEWS)
            + ATTEMPT_INTRINSIC_DESCRIPTOR_GUARD_DISTANCE
            - item[1],
            -item[0],
            -item[1],
        )
    )
    selected_holdouts: list[Mapping[str, Any]] = []
    training_pool: list[Mapping[str, Any]] = []
    effective_temporal_guard = 0
    effective_descriptor_guard = 0.0
    for temporal_guard, descriptor_guard in guard_options:
        candidate_holdouts = _intrinsic_guarded_holdouts(
            holdout_candidates,
            holdout_count,
            protected_training=protected_training,
            temporal_guard=temporal_guard,
            descriptor_guard=descriptor_guard,
        )
        if len(candidate_holdouts) != holdout_count:
            continue
        holdout_names = {str(item["frame"]) for item in candidate_holdouts}
        candidate_training_pool = [
            item
            for item in records
            if str(item["frame"]) not in holdout_names
            and _intrinsic_views_are_separated(
                item,
                candidate_holdouts,
                temporal_guard=temporal_guard,
                descriptor_guard=descriptor_guard,
            )
        ]
        available_cells = {
            int(item["coverage_cell"])
            for item in candidate_training_pool
            if item["coverage_cell"] is not None
        }
        if (
            len(candidate_training_pool) < minimum_training_count
            or len(available_cells) < required_training_cells
        ):
            continue
        selected_holdouts = candidate_holdouts
        training_pool = candidate_training_pool
        effective_temporal_guard = temporal_guard
        effective_descriptor_guard = descriptor_guard
        break
    if not training_pool and records:
        raise ValueError(
            "Intrinsic split could not preserve the minimum training/holdout evidence"
        )

    training_seed_names = {
        str(item["frame"]) for item in protected_training if item in training_pool
    }
    training_seeds = [
        item for item in training_pool if str(item["frame"]) in training_seed_names
    ]
    selected_training = _intrinsic_maximin_selection(
        training_pool,
        min(ATTEMPT_INTRINSIC_MAX_TRAINING_VIEWS, len(training_pool)),
        seeds=training_seeds,
    )
    selected_training.sort(key=lambda item: int(item["chronological_index"]))
    selected_holdouts.sort(key=lambda item: int(item["chronological_index"]))
    training_names = [str(item["frame"]) for item in selected_training]
    holdout_names = [str(item["frame"]) for item in selected_holdouts]
    selected_names = set(training_names + holdout_names)

    omitted_views: list[dict[str, Any]] = []
    for record in records:
        name = str(record["frame"])
        if name in selected_names:
            continue
        reasons = []
        if any(
            abs(
                int(record["chronological_index"]) - int(holdout["chronological_index"])
            )
            <= effective_temporal_guard
            for holdout in selected_holdouts
        ):
            reasons.append("holdout_temporal_guard")
        if effective_descriptor_guard > 0.0 and any(
            _intrinsic_descriptor_distance(record, holdout) < effective_descriptor_guard
            for holdout in selected_holdouts
        ):
            reasons.append("holdout_descriptor_guard")
        if not reasons:
            reasons.append("training_diversity_cap")
        omitted_views.append({"frame": name, "reasons": reasons})
    correlated_omissions = sum(
        1
        for item in omitted_views
        if any(str(reason).startswith("holdout_") for reason in item["reasons"])
    )

    def evidence(
        values: Sequence[Mapping[str, Any]],
        split_name: str,
    ) -> list[dict[str, Any]]:
        return [
            {
                "frame": item["frame"],
                "split": split_name,
                "chronological_index": item["chronological_index"],
                "coverage_cell": item["coverage_cell"],
                "matched_corner_count": item["matched_corner_count"],
                "target_hull_area_ratio": item["target_hull_area_ratio"],
                "projected_board_corners_normalized": item[
                    "projected_board_corners_normalized"
                ],
            }
            for item in values
        ]

    training_cells = sorted(
        {
            int(item["coverage_cell"])
            for item in selected_training
            if item["coverage_cell"] is not None
        }
    )
    holdout_cells = sorted(
        {
            int(item["coverage_cell"])
            for item in selected_holdouts
            if item["coverage_cell"] is not None
        }
    )
    training = {
        **dict(detections),
        "frames": {name: frames[name] for name in training_names},
    }
    holdout = {
        **dict(detections),
        "frames": {name: frames[name] for name in holdout_names},
    }
    split = {
        "strategy": "deterministic_projective_maximin_guarded_views_v2",
        "usable_view_count": len(records),
        "unusable_view_count": len(unusable_views),
        "unusable_views": unusable_views,
        "selected_usable_view_count": len(selected_names),
        "omitted_usable_view_count": len(omitted_views),
        "omitted_correlated_view_count": correlated_omissions,
        "omitted_views": omitted_views,
        "max_optimization_views": (
            ATTEMPT_INTRINSIC_MAX_TRAINING_VIEWS + ATTEMPT_INTRINSIC_MAX_HOLDOUT_VIEWS
        ),
        "max_training_views": ATTEMPT_INTRINSIC_MAX_TRAINING_VIEWS,
        "max_holdout_views": ATTEMPT_INTRINSIC_MAX_HOLDOUT_VIEWS,
        "training_usable_view_count": len(training_names),
        "heldout_usable_view_count": len(holdout_names),
        "training_views": training_names,
        "heldout_views": holdout_names,
        "represented_coverage_cells": represented_cells,
        "training_coverage_cells": training_cells,
        "heldout_coverage_cells": holdout_cells,
        "selected_view_evidence": [
            *evidence(selected_training, "training"),
            *evidence(selected_holdouts, "holdout"),
        ],
        "descriptor": {
            "method": "planar_homography_projected_board_corners_v1",
            "dimension": 8,
            "normalized_corner_coordinate_scale": (
                ATTEMPT_INTRINSIC_DESCRIPTOR_CORNER_SCALE
            ),
            "factory_intrinsics_used": False,
        },
        "holdout_guard": {
            "requested_temporal_radius_views": (ATTEMPT_INTRINSIC_TEMPORAL_GUARD_VIEWS),
            "effective_temporal_radius_views": effective_temporal_guard,
            "requested_descriptor_distance": (
                ATTEMPT_INTRINSIC_DESCRIPTOR_GUARD_DISTANCE
            ),
            "effective_descriptor_distance": effective_descriptor_guard,
            "relaxed_for_minimum_split_feasibility": (
                effective_temporal_guard < ATTEMPT_INTRINSIC_TEMPORAL_GUARD_VIEWS
                or effective_descriptor_guard
                < ATTEMPT_INTRINSIC_DESCRIPTOR_GUARD_DISTANCE
            ),
        },
        "thresholds": {
            "min_training_views": DEFAULT_MIN_ACCEPTED_VIEWS,
            "min_heldout_views": ATTEMPT_INTRINSIC_MIN_HOLDOUT_VIEWS,
            "min_training_coverage_cells": DEFAULT_MIN_COVERAGE_CELLS,
            "holdout_fraction_before_caps": ATTEMPT_INTRINSIC_HOLDOUT_FRACTION,
        },
    }
    return training, holdout, split


def _intrinsic_holdout_evaluation(
    profile: Mapping[str, Any],
    detections: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    native = profile.get("native")
    if not isinstance(native, Mapping):
        return {
            "status": "unavailable",
            "comparable": False,
            "reason": "missing_native_projection",
        }
    if not projection_is_opencv_compatible(native):
        return {
            "status": "unavailable",
            "comparable": False,
            "reason": "distortion_model_is_not_forward_opencv_compatible",
            "distortion_model": native.get("distortion_model"),
        }
    matrix = np.asarray(native["cam_K"], dtype=float).reshape(3, 3)
    distortion = np.asarray(native["distortion"], dtype=float).reshape(-1)
    _dictionary, board = opencv_grid_board(target)
    frames = detections.get("frames")
    if not isinstance(frames, Mapping):
        raise ValueError("Held-out detections require a frames object")
    per_view: dict[str, float] = {}
    failures: list[dict[str, str]] = []
    squared_errors: list[float] = []
    for frame_name, frame in sorted(frames.items()):
        if not isinstance(frame, Mapping):
            failures.append(
                {"frame": str(frame_name), "reason": "invalid_detection_record"}
            )
            continue
        points = _view_points(frame, board)
        if points is None:
            failures.append(
                {"frame": str(frame_name), "reason": "insufficient_grid_points"}
            )
            continue
        object_points, image_points = points
        try:
            success, rvec, tvec = cv2.solvePnP(
                np.asarray(object_points, dtype=np.float64),
                np.asarray(image_points, dtype=np.float64),
                matrix,
                distortion,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                raise ValueError("solvePnP returned no pose")
            projected, _ = cv2.projectPoints(
                object_points,
                rvec,
                tvec,
                matrix,
                distortion,
            )
            errors = np.linalg.norm(
                projected.reshape(-1, 2) - image_points.reshape(-1, 2),
                axis=1,
            )
            if not np.all(np.isfinite(errors)):
                raise ValueError("non-finite reprojection error")
        except (cv2.error, ValueError, TypeError) as exc:
            failures.append({"frame": str(frame_name), "reason": str(exc)})
            continue
        squared_errors.extend(np.square(errors).astype(float).tolist())
        per_view[str(frame_name)] = float(np.sqrt(np.mean(np.square(errors))))
    rms = float(np.sqrt(np.mean(squared_errors))) if squared_errors else None
    max_view = max(per_view.values(), default=None)
    enough_views = len(per_view) >= ATTEMPT_INTRINSIC_MIN_HOLDOUT_VIEWS
    accepted = (
        enough_views
        and rms is not None
        and rms <= DEFAULT_MAX_RMS_PX
        and max_view is not None
        and max_view <= DEFAULT_MAX_VIEW_ERROR_PX
    )
    return {
        "status": "accepted" if accepted else "rejected",
        "comparable": enough_views and rms is not None,
        "profile_id": profile.get("profile_id"),
        "evaluated_view_count": len(per_view),
        "rms_reprojection_error_px": rms,
        "max_view_reprojection_error_px": max_view,
        "per_view_rms_reprojection_error_px": per_view,
        "failures": failures,
        "thresholds": {
            "min_heldout_views": ATTEMPT_INTRINSIC_MIN_HOLDOUT_VIEWS,
            "max_rms_reprojection_error_px": DEFAULT_MAX_RMS_PX,
            "max_view_reprojection_error_px": DEFAULT_MAX_VIEW_ERROR_PX,
        },
    }


def _manual_intrinsic_plausibility(
    factory: Mapping[str, Any],
    manual: Mapping[str, Any],
) -> dict[str, Any]:
    factory_native = factory["native"]
    manual_native = manual["native"]
    factory_k = np.asarray(factory_native["cam_K"], dtype=float).reshape(3, 3)
    manual_k = np.asarray(manual_native["cam_K"], dtype=float).reshape(3, 3)
    distortion = np.asarray(manual_native["distortion"], dtype=float).reshape(-1)
    width, height = (float(value) for value in manual["resolution"])
    fx, fy, cx, cy = (
        float(manual_k[0, 0]),
        float(manual_k[1, 1]),
        float(manual_k[0, 2]),
        float(manual_k[1, 2]),
    )
    factory_fx, factory_fy = float(factory_k[0, 0]), float(factory_k[1, 1])
    focal_delta_ratio = max(
        abs(fx - factory_fx) / factory_fx,
        abs(fy - factory_fy) / factory_fy,
    )
    principal_delta_ratio = max(
        abs(cx - float(factory_k[0, 2])) / width,
        abs(cy - float(factory_k[1, 2])) / height,
    )
    aspect_delta_ratio = abs((fx / fy) / (factory_fx / factory_fy) - 1.0)
    distortion_limits = np.asarray([1.0, 3.0, 0.05, 0.05, 5.0])
    checks = {
        "finite_parameters": bool(
            np.all(np.isfinite(manual_k)) and np.all(np.isfinite(distortion))
        ),
        "positive_focal_lengths": fx > 0.0 and fy > 0.0,
        "principal_point_inside_image": 0.0 <= cx < width and 0.0 <= cy < height,
        "focal_delta_ratio": focal_delta_ratio
        <= ATTEMPT_INTRINSIC_MAX_FOCAL_DELTA_RATIO,
        "principal_delta_ratio": principal_delta_ratio
        <= ATTEMPT_INTRINSIC_MAX_PRINCIPAL_DELTA_RATIO,
        "pixel_aspect_delta_ratio": aspect_delta_ratio
        <= ATTEMPT_INTRINSIC_MAX_ASPECT_DELTA_RATIO,
        "distortion_magnitude": distortion.size == 5
        and bool(np.all(np.abs(distortion) <= distortion_limits)),
    }
    return {
        "status": "accepted" if all(checks.values()) else "rejected",
        "checks": checks,
        "metrics": {
            "focal_delta_ratio": float(focal_delta_ratio),
            "principal_delta_ratio": float(principal_delta_ratio),
            "pixel_aspect_delta_ratio": float(aspect_delta_ratio),
            "absolute_distortion": np.abs(distortion).astype(float).tolist(),
        },
        "thresholds": {
            "max_focal_delta_ratio": ATTEMPT_INTRINSIC_MAX_FOCAL_DELTA_RATIO,
            "max_principal_delta_ratio": (ATTEMPT_INTRINSIC_MAX_PRINCIPAL_DELTA_RATIO),
            "max_pixel_aspect_delta_ratio": (ATTEMPT_INTRINSIC_MAX_ASPECT_DELTA_RATIO),
            "max_absolute_distortion": distortion_limits.tolist(),
        },
    }


def _intrinsic_projection_evidence(profile: Mapping[str, Any]) -> dict[str, Any]:
    native = profile.get("native")
    compatible = isinstance(native, Mapping) and projection_is_opencv_compatible(native)
    evidence = {
        "profile_id": profile.get("profile_id"),
        "opencv_projection_compatible": compatible,
        "distortion_model": (
            native.get("distortion_model") if isinstance(native, Mapping) else None
        ),
        "reason": (
            None
            if compatible
            else (
                "distortion_model_is_not_forward_opencv_compatible"
                if isinstance(native, Mapping)
                else "missing_native_projection"
            )
        ),
    }
    source = profile.get("source")
    compatibility_basis = (
        source.get("opencv_projection_compatibility_basis")
        if isinstance(source, Mapping)
        else None
    )
    if compatible and compatibility_basis:
        evidence["compatibility_basis"] = compatibility_basis
    return evidence


def _intrinsics_for_sensors(
    run_root: Path,
    attempt_root: Path,
    synchronized: Mapping[str, Path],
    request_value: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    existing_path = run_root / INTRINSIC_CALIBRATION_PROFILES
    existing = (
        load_intrinsic_profile_collection(existing_path)
        if existing_path.is_file()
        else []
    )
    profiles: list[dict[str, Any]] = []
    by_sensor: dict[str, dict[str, Any]] = {}
    unusable_sensor_keys: list[str] = []
    comparisons = {
        "schema_version": "intrinsic_comparison.v1",
        "generated_at": utc_now_iso(),
        "attempt_id": request_value["attempt_id"],
        "policy": request_value["intrinsics_policy"],
        "thresholds": {
            "min_accepted_views": DEFAULT_MIN_ACCEPTED_VIEWS,
            "min_coverage_cells": DEFAULT_MIN_COVERAGE_CELLS,
            "max_per_view_reprojection_error_px": DEFAULT_MAX_VIEW_ERROR_PX,
            "max_rms_reprojection_error_px": DEFAULT_MAX_RMS_PX,
            "min_heldout_views": ATTEMPT_INTRINSIC_MIN_HOLDOUT_VIEWS,
            "max_training_views": ATTEMPT_INTRINSIC_MAX_TRAINING_VIEWS,
            "max_holdout_views": ATTEMPT_INTRINSIC_MAX_HOLDOUT_VIEWS,
            "min_absolute_heldout_improvement_px": (
                ATTEMPT_INTRINSIC_MIN_ABSOLUTE_IMPROVEMENT_PX
            ),
            "min_relative_heldout_improvement": (
                ATTEMPT_INTRINSIC_MIN_RELATIVE_IMPROVEMENT
            ),
        },
        "sensors": [],
    }
    for sensor_key, folder in synchronized.items():
        sensor_id, orientation, resolution = sensor_intrinsic_identity(folder)
        factory = factory_intrinsic_profile(folder)
        factory_projection = _intrinsic_projection_evidence(factory)
        candidates = [factory]
        selected: dict[str, Any] | None = None
        existing_projection: dict[str, Any] | None = None
        unusable_projection: dict[str, Any] | None = None
        manual: dict[str, Any] | None = None
        manual_failure: dict[str, Any] | None = None
        comparison_split: dict[str, Any] | None = None
        factory_evaluation: dict[str, Any] | None = None
        manual_evaluation: dict[str, Any] | None = None
        manual_plausibility: dict[str, Any] | None = None
        selection_gates: dict[str, bool] | None = None
        improvement: dict[str, float] | None = None
        if request_value["intrinsics_policy"] == "compare_factory_opencv":
            detections = detect_sensor_folder(
                folder,
                request_value["target"],
                output_path=folder / ARUCO_DETECTIONS,
            )
            try:
                training_detections, holdout_detections, comparison_split = (
                    _intrinsic_detection_split(
                        detections,
                        request_value["target"],
                    )
                )
                if (
                    comparison_split["heldout_usable_view_count"]
                    < ATTEMPT_INTRINSIC_MIN_HOLDOUT_VIEWS
                    or comparison_split["training_usable_view_count"]
                    < DEFAULT_MIN_ACCEPTED_VIEWS
                ):
                    raise ValueError(
                        "Intrinsic comparison requires at least "
                        f"{DEFAULT_MIN_ACCEPTED_VIEWS} training and "
                        f"{ATTEMPT_INTRINSIC_MIN_HOLDOUT_VIEWS} held-out views"
                    )
                manual = calibrate_intrinsic_profile(
                    folder,
                    training_detections,
                    request_value["target"],
                )
                candidates.append(manual)
                factory_evaluation = _intrinsic_holdout_evaluation(
                    factory,
                    holdout_detections,
                    request_value["target"],
                )
                manual_evaluation = _intrinsic_holdout_evaluation(
                    manual,
                    holdout_detections,
                    request_value["target"],
                )
                manual_plausibility = _manual_intrinsic_plausibility(
                    factory,
                    manual,
                )
            except IntrinsicCalibrationError as exc:
                manual_failure = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "quality": exc.report,
                }
            except (cv2.error, ValueError, TypeError) as exc:
                manual_failure = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            if (
                factory_evaluation is not None
                and manual_evaluation is not None
                and factory_evaluation.get("comparable")
                and manual_evaluation.get("comparable")
            ):
                factory_rms = float(factory_evaluation["rms_reprojection_error_px"])
                manual_rms = float(manual_evaluation["rms_reprojection_error_px"])
                absolute_improvement = factory_rms - manual_rms
                relative_improvement = (
                    absolute_improvement / factory_rms if factory_rms > 0.0 else 0.0
                )
                improvement = {
                    "absolute_rms_reprojection_error_px": float(absolute_improvement),
                    "relative_rms_reprojection_error": float(relative_improvement),
                }
            factory_projection_unavailable = not bool(
                factory_projection["opencv_projection_compatible"]
            )
            selection_gates = {
                "manual_training_quality": bool(
                    manual is not None
                    and manual.get("quality", {}).get("status") == "accepted"
                ),
                "manual_parameter_plausibility": bool(
                    manual_plausibility is not None
                    and manual_plausibility.get("status") == "accepted"
                ),
                "manual_heldout_absolute_quality": bool(
                    manual_evaluation is not None
                    and manual_evaluation.get("status") == "accepted"
                ),
                "factory_heldout_comparable": bool(
                    factory_evaluation is not None
                    and factory_evaluation.get("comparable")
                ),
                "minimum_absolute_improvement": bool(
                    improvement is not None
                    and improvement["absolute_rms_reprojection_error_px"]
                    >= ATTEMPT_INTRINSIC_MIN_ABSOLUTE_IMPROVEMENT_PX
                ),
                "minimum_relative_improvement": bool(
                    improvement is not None
                    and improvement["relative_rms_reprojection_error"]
                    >= ATTEMPT_INTRINSIC_MIN_RELATIVE_IMPROVEMENT
                ),
                "factory_projection_unavailable": factory_projection_unavailable,
            }
            manual_proven = (
                selection_gates["manual_training_quality"]
                and selection_gates["manual_parameter_plausibility"]
                and selection_gates["manual_heldout_absolute_quality"]
                and factory_projection_unavailable
            )
            if manual is not None and manual_proven:
                selected = {
                    **manual,
                    "attempt_intrinsics_source": (
                        "opencv_manual_factory_projection_unavailable"
                    ),
                }
                selection_reason = (
                    "manual_opencv_passed_training_heldout_and_plausibility_"
                    "gates_while_factory_projection_was_unavailable"
                )
            else:
                selected = {
                    **factory,
                    "attempt_intrinsics_source": (
                        "factory_unusable_manual_not_accepted"
                        if factory_projection_unavailable
                        else "factory_compatible_default_comparison_only"
                    ),
                }
                selection_reason = (
                    (
                        "factory_projection_unusable_and_manual_opencv_did_"
                        "not_pass_all_activation_gates"
                    )
                    if factory_projection_unavailable
                    else (
                        "compatible_factory_intrinsics_retained_by_policy;_"
                        "manual_opencv_result_is_comparison_only"
                    )
                )
        else:
            try:
                existing_profile = select_intrinsic_profile(
                    existing,
                    sensor_id=sensor_id,
                    orientation=orientation,
                    resolution=resolution,
                )
            except ValueError:
                existing_profile = None
            if existing_profile is not None:
                existing_projection = _intrinsic_projection_evidence(existing_profile)
                existing_candidate = {
                    **existing_profile,
                    "attempt_intrinsics_source": (
                        "compatible_existing"
                        if existing_projection["opencv_projection_compatible"]
                        else "existing_projection_unusable"
                    ),
                }
                candidates.append(existing_candidate)
                if existing_projection["opencv_projection_compatible"]:
                    selected = existing_candidate
                    selection_reason = "exact_compatible_existing_profile"
            if selected is None and factory_projection["opencv_projection_compatible"]:
                selected = {
                    **factory,
                    "attempt_intrinsics_source": (
                        "factory_capture_sidecars_existing_projection_unusable"
                        if existing_profile is not None
                        else "factory_capture_sidecars"
                    ),
                }
                selection_reason = (
                    "exact_existing_profile_projection_unusable;_"
                    "compatible_factory_capture_sidecars_selected"
                    if existing_profile is not None
                    else "no_exact_compatible_existing_profile"
                )
            elif selected is None:
                selection_reason = (
                    "exact_existing_and_factory_projections_are_unusable"
                    if existing_profile is not None
                    else "no_exact_existing_profile_and_factory_projection_is_unusable"
                )

        selected_projection = (
            _intrinsic_projection_evidence(selected) if selected is not None else None
        )
        if (
            selected_projection is None
            or not selected_projection["opencv_projection_compatible"]
        ):
            unusable_projection = {
                "reason": "no_opencv_compatible_intrinsic_projection",
                "factory": factory_projection,
                "existing": existing_projection,
                "selected": selected_projection,
            }
            unusable_sensor_keys.append(sensor_key)
            selected = None
        else:
            profiles.append(selected)
            by_sensor[sensor_key] = selected
        manual_selected = (
            manual is not None
            and selected is not None
            and selected["profile_id"] == manual.get("profile_id")
            and selected.get("attempt_intrinsics_source")
            == "opencv_manual_factory_projection_unavailable"
        )
        selection_status = (
            "unusable"
            if unusable_projection is not None
            else (
                "manual_selected"
                if manual_selected
                else (
                    "existing_selected"
                    if selected is not None
                    and selected.get("attempt_intrinsics_source")
                    == "compatible_existing"
                    else "factory_selected"
                )
            )
        )
        comparisons["sensors"].append(
            {
                "sensor_key": sensor_key,
                "sensor_id": sensor_id,
                "resolution": list(resolution),
                "orientation": orientation,
                "status": selection_status,
                "selected_profile_id": (
                    selected["profile_id"] if selected is not None else None
                ),
                "selection_reason": selection_reason,
                "factory_profile_id": factory["profile_id"],
                "factory_projection": factory_projection,
                "existing_projection": existing_projection,
                "unusable_projection": unusable_projection,
                "manual_profile_id": manual.get("profile_id") if manual else None,
                "manual_failure": manual_failure,
                "comparison_split": comparison_split,
                "factory_heldout_evaluation": factory_evaluation,
                "manual_heldout_evaluation": manual_evaluation,
                "manual_plausibility": manual_plausibility,
                "heldout_improvement": improvement,
                "selection_gates": selection_gates,
                "deltas": (
                    _intrinsic_deltas(factory, manual) if manual is not None else None
                ),
                "candidates": candidates,
            }
        )
    atomic_write_json(attempt_root / INTRINSIC_COMPARISON, comparisons)
    if unusable_sensor_keys:
        raise ValueError(
            "No OpenCV-compatible intrinsic projection is available for: "
            + ", ".join(sorted(unusable_sensor_keys))
            + f"; see {INTRINSIC_COMPARISON} for preserved projection evidence"
        )
    return profiles, by_sensor


def _prepare_attempt_data(
    run_root: Path,
    attempt_root: Path,
    request_value: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    timestamp_policy = _calibration_timestamp_preflight(
        run_root, request_value["sensors"]
    )
    recorded_timestamp_policy = request_value.get("timestamp_policy")
    if isinstance(recorded_timestamp_policy, Mapping):
        policy_keys = (
            "frame_timestamp_source",
            "robot_timestamp_source",
            "required_frame_timestamp_domain",
            "timestamp_fallback_allowed",
        )
        for key in policy_keys:
            if recorded_timestamp_policy.get(key) != timestamp_policy.get(key):
                raise ValueError(
                    "Recorded calibration timestamp policy no longer matches "
                    f"the selected sensor timebase: {key}"
                )
        recorded_per_sensor = recorded_timestamp_policy.get("per_sensor")
        current_per_sensor = timestamp_policy.get("per_sensor")
        expected_sensor_keys = {
            str(sensor.get("sensor_key") or sensor.get("folder"))
            for sensor in request_value["sensors"]
        }
        if (
            not isinstance(recorded_per_sensor, Mapping)
            or not isinstance(current_per_sensor, Mapping)
            or set(recorded_per_sensor) != expected_sensor_keys
            or set(current_per_sensor) != expected_sensor_keys
        ):
            raise ValueError(
                "Recorded calibration timestamp policy no longer matches the "
                "selected per-sensor timebases"
            )
        for sensor_key in sorted(expected_sensor_keys):
            recorded_sensor_policy = recorded_per_sensor.get(sensor_key)
            current_sensor_policy = current_per_sensor.get(sensor_key)
            if not isinstance(recorded_sensor_policy, Mapping) or not isinstance(
                current_sensor_policy, Mapping
            ):
                raise ValueError(
                    "Recorded calibration timestamp policy lacks per-sensor "
                    f"evidence for {sensor_key}"
                )
            for key in policy_keys:
                if recorded_sensor_policy.get(key) != current_sensor_policy.get(key):
                    raise ValueError(
                        "Recorded calibration timestamp policy no longer matches "
                        f"{sensor_key}: {key}"
                    )
    # Retain the zero-offset image/PnP workspace separately.  The accepted
    # per-sensor offsets are materialized later under processed/synchronized;
    # reusing that folder would delete the detections and source-frame mapping
    # that make the search reproducible.
    output_root = attempt_root / "processed" / "preparation_synchronized"
    synchronized: dict[str, Path] = {}
    sync_reports = []
    selected_by_path = {
        (run_root / str(sensor["folder"])).resolve(): sensor
        for sensor in request_value["sensors"]
    }
    required_frame_sources: dict[str, str] = {}
    required_robot_sources: dict[str, str] = {}
    for sensor_path, sensor in selected_by_path.items():
        sensor_policy = _timestamp_policy_for_sensor(timestamp_policy, sensor)
        sensor_name = str(sensor.get("sensor_name") or sensor_path.name)
        required_frame_sources[sensor_name] = str(
            sensor_policy["frame_timestamp_source"]
        )
        required_robot_sources[sensor_name] = str(
            sensor_policy["robot_timestamp_source"]
        )
        results = synchronize_run(
            run_root,
            sensor_folders=[sensor_path],
            output_root=output_root,
            sync_delta=ATTEMPT_SYNC_DELTA_MS,
            timestamp_source=sensor_policy["frame_timestamp_source"],
            robot_timestamp_source=sensor_policy["robot_timestamp_source"],
            max_nearest_pose_delta_ms=ATTEMPT_MAX_NEAREST_POSE_DELTA_MS,
        )
        for result in results:
            selected_sensor = selected_by_path[Path(result.sensor_folder).resolve()]
            sensor_key = str(selected_sensor["sensor_key"])
            synchronized[sensor_key] = Path(result.output_folder).resolve()
            # synchronize_run mirrors the caller's path style. Normalizing here is
            # essential because build_sync_quality_report interprets explicit
            # relative report paths relative to run_root.
            sync_reports.append(Path(result.report_path).resolve())
    sync_quality = build_sync_quality_report(
        run_root,
        report_paths=sync_reports,
        max_nearest_pose_delta_ms=ATTEMPT_MAX_NEAREST_POSE_DELTA_MS,
        require_timestamp_source=required_frame_sources,
        require_robot_timestamp_source=required_robot_sources,
    )
    sync_quality["calibration_attempt_policy"] = {
        "sync_delta_ms": ATTEMPT_SYNC_DELTA_MS,
        **timestamp_policy,
        "max_nearest_pose_delta_ms": ATTEMPT_MAX_NEAREST_POSE_DELTA_MS,
        "historical_per_sensor_offsets_allowed": False,
    }
    sensor_summaries = sync_quality.get("sensors")
    sync_delta_checks: list[dict[str, Any]] = []
    if not isinstance(sensor_summaries, list) or len(sensor_summaries) != len(
        synchronized
    ):
        sync_delta_checks.append(
            {
                "name": "calibration_sync_delta_evidence",
                "status": "error",
                "message": "Sync delta evidence is missing for selected cameras.",
                "details": {
                    "expected_sensor_count": len(synchronized),
                    "actual_sensor_count": (
                        len(sensor_summaries)
                        if isinstance(sensor_summaries, list)
                        else 0
                    ),
                },
            }
        )
    else:
        for sensor in sensor_summaries:
            sensor_name = (
                str(sensor.get("sensor_name"))
                if isinstance(sensor, Mapping)
                else "unknown"
            )
            actual_value = (
                sensor.get("sync_delta_ms") if isinstance(sensor, Mapping) else None
            )
            try:
                actual_delta_ms = float(actual_value)
            except (TypeError, ValueError):
                actual_delta_ms = math.nan
            delta_ok = math.isfinite(actual_delta_ms) and math.isclose(
                actual_delta_ms,
                ATTEMPT_SYNC_DELTA_MS,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            sync_delta_checks.append(
                {
                    "name": f"calibration_sync_delta:{sensor_name}",
                    "status": "ok" if delta_ok else "error",
                    "message": (
                        f"{sensor_name} used the required zero sync offset."
                        if delta_ok
                        else (
                            f"{sensor_name} sync offset {actual_value!r} ms "
                            "does not equal the required 0.0 ms."
                        )
                    ),
                    "details": {
                        "actual_sync_delta_ms": actual_value,
                        "required_sync_delta_ms": ATTEMPT_SYNC_DELTA_MS,
                    },
                }
            )
    quality_checks = sync_quality.get("checks")
    if not isinstance(quality_checks, list):
        quality_checks = []
        sync_quality["checks"] = quality_checks
    quality_checks.extend(sync_delta_checks)
    atomic_write_json(attempt_root / SYNC_QUALITY_REPORT, sync_quality)
    checks = quality_checks
    blocking_checks = [
        check
        for check in checks
        if isinstance(check, Mapping)
        and (
            check.get("status") == "error"
            or (
                str(check.get("name", "")).startswith("sync_nearest_pose_delta:")
                and check.get("status") != "ok"
            )
        )
    ]
    timestamp_checks = [
        check
        for check in checks
        if isinstance(check, Mapping)
        and str(check.get("name", "")).startswith("sync_timestamp_source:")
    ]
    robot_timestamp_checks = [
        check
        for check in checks
        if isinstance(check, Mapping)
        and str(check.get("name", "")).startswith("sync_robot_timestamp_source:")
    ]
    nearest_checks = [
        check
        for check in checks
        if isinstance(check, Mapping)
        and str(check.get("name", "")).startswith("sync_nearest_pose_delta:")
    ]
    if (
        blocking_checks
        or len(timestamp_checks) != len(synchronized)
        or len(robot_timestamp_checks) != len(synchronized)
        or len(nearest_checks) != len(synchronized)
    ):
        names = [str(check.get("name")) for check in blocking_checks]
        raise ValueError(
            "Selected-camera synchronization quality failed strict "
            "eye-in-hand policy" + (f": {', '.join(names)}" if names else "")
        )
    profiles, by_sensor = _intrinsics_for_sensors(
        run_root,
        attempt_root,
        synchronized,
        request_value,
    )
    write_intrinsic_profile_collection(
        profiles,
        attempt_root / INTRINSIC_CALIBRATION_PROFILES,
    )
    return synchronized, by_sensor


def _projection(profile: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    native = profile.get("native")
    if not isinstance(native, Mapping):
        raise ValueError("Intrinsic profile has no native projection")
    if not projection_is_opencv_compatible(native):
        raise ValueError(
            "Intrinsic SDK distortion model is not a supported forward OpenCV projection"
        )
    return (
        np.asarray(native["cam_K"], dtype=float).reshape(3, 3),
        np.asarray(native["distortion"], dtype=float).reshape(-1),
    )


def _pose_vectors(
    transform_value: Mapping[str, Any],
) -> tuple[list[float], list[float]]:
    transform = transform_from_record(transform_value)
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    return (
        np.asarray(rvec, dtype=float).reshape(3).tolist(),
        np.asarray(transform[:3, 3], dtype=float).reshape(3).tolist(),
    )


def _coverage_cell(
    centroid: Any,
    image_size: Any,
) -> int | None:
    if (
        not isinstance(centroid, list)
        or len(centroid) != 2
        or not isinstance(image_size, list)
        or len(image_size) != 2
    ):
        return None
    width, height = (float(value) for value in image_size)
    x, y = (float(value) for value in centroid)
    if (
        width <= 0.0
        or height <= 0.0
        or not all(np.isfinite(value) for value in (x, y, width, height))
    ):
        return None
    column = min(2, max(0, int(x * 3.0 / width)))
    row = min(2, max(0, int(y * 3.0 / height)))
    return column + 3 * row


def _pnp_point_marker_metadata(
    detection: Mapping[str, Any],
    target: Mapping[str, Any],
    point_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    ids = detection.get("ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError("PnP detection requires marker IDs")
    marker_positions = {
        int(marker["id"]): divmod(index, int(target["grid_size"][0]))
        for index, marker in enumerate(target["markers"])
    }
    try:
        point_marker_ids = np.repeat(
            np.asarray([int(value) for value in ids], dtype=np.int64), 4
        )
        point_grid_indices = np.repeat(
            np.asarray(
                [marker_positions[int(value)] for value in ids],
                dtype=np.int64,
            ),
            4,
            axis=0,
        )
    except KeyError as exc:
        raise ValueError(f"Detection includes marker outside target: {exc}") from exc
    if len(point_marker_ids) != point_count:
        raise ValueError("PnP marker metadata does not align with matched points")
    return point_marker_ids, point_grid_indices


def _estimate_target_poses(
    attempt_root: Path,
    request_value: Mapping[str, Any],
    synchronized: Mapping[str, Path],
    intrinsics: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, list[dict[str, Any]]]]]:
    target = normalize_calibration_target_spec(request_value["target"])
    _dictionary, board = opencv_grid_board(target)
    evidence = {
        "schema_version": "calibration_pnp_candidates.v1",
        "attempt_id": request_value["attempt_id"],
        "target": target_identity(target),
        "methods": list(request_value["pnp_methods"]),
        "sensors": [],
    }
    observations: dict[str, dict[str, list[dict[str, Any]]]] = {}
    sensor_metadata = {
        str(item["sensor_key"]): item for item in request_value["sensors"]
    }
    for sensor_key, folder in synchronized.items():
        detection_path = folder / ARUCO_DETECTIONS
        if detection_path.is_file():
            detections = _read_json(detection_path)
            validate_target_identity(
                detections.get("target"), target, label="ArUco detections"
            )
        else:
            detections = detect_sensor_folder(
                folder,
                target,
                output_path=detection_path,
            )
        matrix, distortion = _projection(intrinsics[sensor_key])
        matched = _read_json(folder / MATCH_ROBOT_EE_POSES)
        frames = []
        method_observations = {method: [] for method in request_value["pnp_methods"]}
        compatibility_output: dict[str, Any] = {}
        for frame_id, detection in sorted(detections.get("frames", {}).items()):
            frame_record: dict[str, Any] = {
                "frame_id": frame_id,
                "source_frame_id": (
                    matched.get(frame_id, {}).get("source_frame_id")
                    if isinstance(matched.get(frame_id), Mapping)
                    else None
                ),
                "marker_count": int(detection.get("marker_count", 0)),
                "image_centroid_px": detection.get("image_centroid_px"),
                "image_coverage_cell": _coverage_cell(
                    detection.get("image_centroid_px"),
                    detections.get("image_size"),
                ),
                "status": "rejected",
                "candidates": [],
                "failures": [],
            }
            matched_pose = matched.get(frame_id)
            if not isinstance(matched_pose, Mapping):
                frame_record["failures"].append({"reason": "missing_robot_pose"})
                frames.append(frame_record)
                continue
            points = _matched_points(detection, board)
            if points is None or int(detection.get("marker_count", 0)) < 4:
                frame_record["failures"].append(
                    {"reason": "insufficient_board_markers"}
                )
                frames.append(frame_record)
                continue
            try:
                point_marker_ids, point_grid_indices = _pnp_point_marker_metadata(
                    detection,
                    target,
                    len(points[0]),
                )
                solved = solve_planar_pnp_candidates(
                    points[0],
                    points[1],
                    matrix,
                    distortion,
                    methods=request_value["pnp_methods"],
                    point_marker_ids=point_marker_ids,
                    point_grid_indices=point_grid_indices,
                )
            except (cv2.error, ValueError, TypeError) as exc:
                frame_record["failures"].append({"reason": str(exc)})
                frames.append(frame_record)
                continue
            frame_record.update(
                {
                    "status": "ok" if solved["selected"] else "rejected",
                    "common_inlier_indices": solved["common_inlier_indices"],
                    "common_inlier_count": solved["common_inlier_count"],
                    "common_inlier_ratio": solved["common_inlier_ratio"],
                    "correspondence_count": solved["correspondence_count"],
                    "supported_marker_ids": solved["supported_marker_ids"],
                    "supported_marker_count": solved["supported_marker_count"],
                    "supported_marker_corner_counts": solved[
                        "supported_marker_corner_counts"
                    ],
                    "supported_grid_rows": solved["supported_grid_rows"],
                    "supported_grid_columns": solved["supported_grid_columns"],
                    "quality_thresholds": solved["thresholds"],
                    "candidates": solved["candidates"],
                    "failures": solved["failures"],
                }
            )
            for method, selected in solved["selected"].items():
                method_observations[method].append(
                    {
                        "observation_id": f"{sensor_key}:{method}:{frame_id}",
                        "frame_id": frame_id,
                        "source_frame_id": matched_pose.get("source_frame_id"),
                        "image_timestamp_ns": matched_pose.get("image_timestamp_ns"),
                        "initial_matched_robot_pose_index": matched_pose.get(
                            "matched_robot_pose_index"
                        ),
                        "initial_robot_timestamp_ns": matched_pose.get(
                            "robot_timestamp_ns"
                        ),
                        "initial_nearest_robot_delta_ns": matched_pose.get(
                            "nearest_robot_delta_ns"
                        ),
                        "motion": matched_pose.get("motion"),
                        "robot_ee_pose": dict(matched_pose["robot_ee_pose"]),
                        "target_to_camera": selected["transform"],
                        "mean_reprojection_error_px": selected[
                            "mean_reprojection_error_px"
                        ],
                        "pnp_common_inlier_count": solved["common_inlier_count"],
                        "pnp_common_inlier_ratio": solved["common_inlier_ratio"],
                        "pnp_correspondence_count": solved["correspondence_count"],
                        "pnp_supported_marker_ids": solved["supported_marker_ids"],
                        "pnp_supported_grid_rows": solved["supported_grid_rows"],
                        "pnp_supported_grid_columns": solved["supported_grid_columns"],
                        "all_point_mean_reprojection_error_px": selected[
                            "all_point_mean_reprojection_error_px"
                        ],
                        "image_centroid_px": detection.get("image_centroid_px"),
                        "image_coverage_cell": _coverage_cell(
                            detection.get("image_centroid_px"),
                            detections.get("image_size"),
                        ),
                    }
                )
            preferred = solved["selected"].get("ITERATIVE") or next(
                iter(solved["selected"].values()), None
            )
            if preferred is not None:
                rvec, tvec = _pose_vectors(preferred["transform"])
                compatibility_output[frame_id] = {
                    **dict(matched_pose),
                    "aruco_pose_estimation": {
                        "schema_version": "aruco_pose_estimation.v2",
                        "rvec": rvec,
                        "tvec": tvec,
                        "len_ids": int(detection.get("marker_count", 0)),
                        "pnp_inlier_indices": solved["common_inlier_indices"],
                        "pnp_inlier_count": solved["common_inlier_count"],
                        "pnp_inlier_ratio": solved["common_inlier_ratio"],
                        "mean_reprojection_error_px": preferred[
                            "mean_reprojection_error_px"
                        ],
                        "max_reprojection_error_px": preferred[
                            "max_reprojection_error_px"
                        ],
                        "all_point_mean_reprojection_error_px": preferred[
                            "all_point_mean_reprojection_error_px"
                        ],
                        "target": target_identity(target),
                    },
                }
            frames.append(frame_record)
        target_marker_count = len(target["markers"])
        target_columns, target_rows = (int(value) for value in target["grid_size"])
        accepted_frames = [item for item in frames if item["status"] == "ok"]
        accepted_marker_ids = sorted(
            {
                int(marker_id)
                for item in accepted_frames
                for marker_id in item.get("supported_marker_ids", [])
            }
        )
        accepted_grid_rows = sorted(
            {
                int(row)
                for item in accepted_frames
                for row in item.get("supported_grid_rows", [])
            }
        )
        accepted_grid_columns = sorted(
            {
                int(column)
                for item in accepted_frames
                for column in item.get("supported_grid_columns", [])
            }
        )
        dataset_support_thresholds = {
            "min_target_markers": math.ceil(
                target_marker_count * ATTEMPT_MIN_TARGET_MARKER_COVERAGE_RATIO
            ),
            "min_target_rows": math.ceil(
                target_rows * ATTEMPT_MIN_TARGET_ROW_COVERAGE_RATIO
            ),
            "min_target_columns": math.ceil(
                target_columns * ATTEMPT_MIN_TARGET_COLUMN_COVERAGE_RATIO
            ),
            "min_target_marker_coverage_ratio": (
                ATTEMPT_MIN_TARGET_MARKER_COVERAGE_RATIO
            ),
            "min_target_row_coverage_ratio": (ATTEMPT_MIN_TARGET_ROW_COVERAGE_RATIO),
            "min_target_column_coverage_ratio": (
                ATTEMPT_MIN_TARGET_COLUMN_COVERAGE_RATIO
            ),
        }
        dataset_support_ok = (
            len(accepted_marker_ids) >= dataset_support_thresholds["min_target_markers"]
            and len(accepted_grid_rows) >= dataset_support_thresholds["min_target_rows"]
            and len(accepted_grid_columns)
            >= dataset_support_thresholds["min_target_columns"]
        )
        dataset_marker_support = {
            "status": "ok" if dataset_support_ok else "error",
            "accepted_marker_ids": accepted_marker_ids,
            "accepted_marker_count": len(accepted_marker_ids),
            "accepted_grid_rows": accepted_grid_rows,
            "accepted_grid_columns": accepted_grid_columns,
            "target_marker_count": target_marker_count,
            "target_row_count": target_rows,
            "target_column_count": target_columns,
            "thresholds": dataset_support_thresholds,
        }
        if not dataset_support_ok:
            # Retain frame-level PnP evidence but do not permit a small fixed
            # patch of the board to reach the hand-eye solver.
            method_observations = {
                method: [] for method in request_value["pnp_methods"]
            }
        atomic_write_json(folder / ARUCO_POSE_ESTIMATION, compatibility_output)
        observations[sensor_key] = method_observations
        evidence["sensors"].append(
            {
                **sensor_metadata[sensor_key],
                "frame_count": len(frames),
                "solved_frame_count": sum(
                    1 for item in frames if item["status"] == "ok"
                ),
                "dataset_marker_support": dataset_marker_support,
                "accepted_coverage_cells": sorted(
                    {
                        int(item["image_coverage_cell"])
                        for item in frames
                        if item["status"] == "ok"
                        and item.get("image_coverage_cell") is not None
                    }
                ),
                "frames": frames,
            }
        )
    atomic_write_json(attempt_root / PNP_CANDIDATES_FILE, evidence)
    return evidence, observations


def _calibration_observation_report(
    request_value: Mapping[str, Any],
    observations: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    flat = []
    sensors = []
    sensor_metadata = {
        str(item["sensor_key"]): item for item in request_value["sensors"]
    }
    for sensor_key, by_method in observations.items():
        count = sum(len(items) for items in by_method.values())
        sensors.append(
            {
                **sensor_metadata[sensor_key],
                "observation_count": count,
                "pnp_method_counts": {
                    method: len(items) for method, items in by_method.items()
                },
            }
        )
        for method, items in by_method.items():
            for item in items:
                flat.append(
                    {
                        **dict(item),
                        "sensor_name": sensor_metadata[sensor_key]["sensor_name"],
                        "sensor_type": sensor_metadata[sensor_key]["sensor_type"],
                        "device_id": sensor_metadata[sensor_key]["device_id"],
                        "mounting_mode": (
                            "eye_in_hand"
                            if request_value["mode"] == "eye_in_hand"
                            else "static"
                        ),
                        "pnp_method": method,
                    }
                )
    return {
        "schema_version": "calibration_observations.v1",
        "generated_at": utc_now_iso(),
        "run_root": request_value["run_root"],
        "attempt_id": request_value["attempt_id"],
        "overall_status": "ok" if flat else "error",
        "target": request_value["target"],
        "board": request_value["target"],
        "sensor_count": len(sensors),
        "observation_count": len(flat),
        "sensors": sensors,
        "observations": flat,
        "checks": [],
        "rejected": [],
    }


def _estimate_and_apply_time_offsets(
    run_root: Path,
    attempt_root: Path,
    request_value: Mapping[str, Any],
    observations: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, list[dict[str, Any]]]],
]:
    policy = str(
        request_value.get(
            "synchronization_policy",
            DEFAULT_SYNCHRONIZATION_POLICY,
        )
    )
    timestamp_policy = request_value.get("timestamp_policy")
    search = request_value.get("synchronization_search")
    search_configuration = (
        dict(search)
        if isinstance(search, Mapping)
        else time_offset_search_configuration()
    )
    implementation_revision = str(
        request_value.get(
            "synchronization_implementation_revision",
            TIME_OFFSET_IMPLEMENTATION_REVISION,
        )
    )
    if implementation_revision not in TIME_OFFSET_SUPPORTED_REVISIONS:
        raise ValueError(
            "Unsupported calibration time-offset implementation revision: "
            f"{implementation_revision}"
        )
    max_nearest_pose_delta_ms = float(search_configuration["max_nearest_pose_delta_ms"])
    if not math.isfinite(max_nearest_pose_delta_ms) or max_nearest_pose_delta_ms <= 0.0:
        raise ValueError(
            "Calibration time-offset max nearest-pose delta must be positive"
        )
    sensor_metadata = {
        str(item["sensor_key"]): item for item in request_value["sensors"]
    }
    adjusted: dict[str, dict[str, list[dict[str, Any]]]] = {}
    sensor_results: list[dict[str, Any]] = []
    failed: list[str] = []
    for sensor_key in request_value["sensor_keys"]:
        by_method = observations[sensor_key]
        reference_observations = list(by_method.get(DEFAULT_REFERENCE_PNP_METHOD, ()))
        sensor = sensor_metadata[sensor_key]
        if policy == "fixed_zero":
            sensor_result = fixed_zero_sensor_result(
                sensor_key=sensor_key,
                observation_count=max(
                    (len(items) for items in by_method.values()),
                    default=0,
                ),
            )
            adjusted[sensor_key] = {
                method: [dict(item) for item in items]
                for method, items in by_method.items()
            }
        else:
            try:
                if not reference_observations:
                    raise ValueError(
                        f"{sensor_key}: auto-sync reference observations are missing"
                    )
                sensor_policy = _timestamp_policy_for_sensor(
                    timestamp_policy
                    if isinstance(timestamp_policy, Mapping)
                    else _attempt_timestamp_policy(request_value["sensors"]),
                    sensor,
                )
                sensor_folder = run_root / str(sensor["folder"])
                robot_records = indexed_robot_poses(
                    load_robot_poses(run_root, sensor_folder),
                    timestamp_source=str(sensor_policy["robot_timestamp_source"]),
                )
                sensor_result, _reference_adjusted = estimate_sensor_time_offset(
                    reference_observations,
                    sensor_key=sensor_key,
                    robot_records=robot_records,
                    mode=str(request_value["mode"]),
                    offsets_ms=time_offset_values(
                        float(
                            search_configuration["minimum_robot_pose_time_offset_ms"]
                        ),
                        float(
                            search_configuration["maximum_robot_pose_time_offset_ms"]
                        ),
                        float(search_configuration["step_ms"]),
                    ),
                    methods=tuple(
                        str(item)
                        for item in search_configuration["reference_extrinsic_methods"]
                    ),
                    max_nearest_pose_delta_ms=max_nearest_pose_delta_ms,
                    max_observations_per_motion=int(
                        search_configuration["max_observations_per_motion"]
                    ),
                    max_search_motions=int(
                        search_configuration["maximum_search_motion_count"]
                    ),
                    min_motions_per_fold=int(
                        search_configuration[
                            "minimum_motion_count_per_cross_validation_fold"
                        ]
                    ),
                    min_absolute_improvement_mm=float(
                        search_configuration[
                            "minimum_absolute_cross_validated_improvement_mm"
                        ]
                    ),
                    min_relative_improvement=float(
                        search_configuration[
                            "minimum_relative_cross_validated_improvement"
                        ]
                    ),
                    max_rotation_degradation_deg=float(
                        search_configuration[
                            "maximum_cross_validated_rotation_degradation_deg"
                        ]
                    ),
                    minimum_offset_stability_ms=float(
                        search_configuration["minimum_offset_stability_ms"]
                    ),
                )
                selected_offset_ms = float(
                    sensor_result["selected_robot_pose_time_offset_ms"]
                )
                adjusted[sensor_key] = {
                    method: apply_sensor_time_offset(
                        items,
                        robot_records=robot_records,
                        robot_pose_time_offset_ms=selected_offset_ms,
                        max_nearest_pose_delta_ms=max_nearest_pose_delta_ms,
                    )
                    for method, items in by_method.items()
                }
            except Exception as exc:
                sensor_result = failed_sensor_result(
                    sensor_key=sensor_key,
                    observation_count=len(reference_observations),
                    error=exc,
                )
                adjusted[sensor_key] = {}
            if sensor_result["status"] == "failed":
                failed.append(sensor_key)
        sensor_result["display_name"] = sensor.get("display_name")
        sensor_result["sensor_name"] = sensor.get("sensor_name")
        sensor_results.append(sensor_result)

    report = {
        "schema_version": TIME_OFFSET_SEARCH_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "attempt_id": request_value["attempt_id"],
        "policy": policy,
        "implementation_revision": implementation_revision,
        "offset_kind": "effective_capture_and_pose_pipeline_latency",
        "sign_convention": time_offset_sign_convention(),
        "search": search_configuration,
        "status": "failed" if failed else "complete",
        "sensor_count": len(sensor_results),
        "failed_sensor_keys": failed,
        "sensors": sensor_results,
    }
    atomic_write_json(attempt_root / TIME_OFFSET_SEARCH, report)
    if failed:
        raise ValueError("Auto-sync evidence failed closed for: " + ", ".join(failed))
    return report, adjusted


def _materialize_authoritative_synchronization(
    run_root: Path,
    attempt_root: Path,
    request_value: Mapping[str, Any],
    time_offset_search: Mapping[str, Any],
    observations: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> tuple[
    dict[str, Path],
    dict[str, dict[str, list[dict[str, Any]]]],
]:
    timestamp_policy = _calibration_timestamp_preflight(
        run_root, request_value["sensors"]
    )
    result_by_sensor = {
        str(item["sensor_key"]): item
        for item in time_offset_search.get("sensors", [])
        if isinstance(item, Mapping) and item.get("sensor_key")
    }
    expected_sensor_keys = {str(item) for item in request_value["sensor_keys"]}
    if set(result_by_sensor) != expected_sensor_keys:
        raise ValueError("Time-offset evidence does not cover every selected sensor")
    try:
        max_nearest_pose_delta_ms = float(
            time_offset_search["search"]["max_nearest_pose_delta_ms"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Time-offset evidence lacks a valid max nearest-pose delta"
        ) from exc
    if not math.isfinite(max_nearest_pose_delta_ms) or max_nearest_pose_delta_ms <= 0.0:
        raise ValueError("Time-offset evidence max nearest-pose delta must be positive")

    output_root = attempt_root / "processed" / "synchronized"
    synchronized: dict[str, Path] = {}
    sync_reports: list[Path] = []
    required_frame_sources: dict[str, str] = {}
    required_robot_sources: dict[str, str] = {}
    expected_by_sensor_name: dict[str, float] = {}
    for sensor in request_value["sensors"]:
        sensor_key = str(sensor["sensor_key"])
        sensor_path = run_root / str(sensor["folder"])
        sensor_policy = _timestamp_policy_for_sensor(timestamp_policy, sensor)
        sensor_name = str(sensor.get("sensor_name") or sensor_path.name)
        selected_sync_delta_ms = float(
            result_by_sensor[sensor_key]["selected_sync_delta_ms"]
        )
        required_frame_sources[sensor_name] = str(
            sensor_policy["frame_timestamp_source"]
        )
        required_robot_sources[sensor_name] = str(
            sensor_policy["robot_timestamp_source"]
        )
        expected_by_sensor_name[sensor_name] = selected_sync_delta_ms
        results = synchronize_run(
            run_root,
            sensor_folders=[sensor_path],
            output_root=output_root,
            sync_delta=selected_sync_delta_ms,
            timestamp_source=sensor_policy["frame_timestamp_source"],
            robot_timestamp_source=sensor_policy["robot_timestamp_source"],
            copy_files=False,
            max_nearest_pose_delta_ms=max_nearest_pose_delta_ms,
        )
        if len(results) != 1:
            raise ValueError(
                f"Authoritative synchronization returned no result for {sensor_key}"
            )
        result = results[0]
        synchronized[sensor_key] = Path(result.output_folder).resolve()
        sync_reports.append(Path(result.report_path).resolve())

    sync_quality = build_sync_quality_report(
        run_root,
        report_paths=sync_reports,
        max_nearest_pose_delta_ms=max_nearest_pose_delta_ms,
        require_timestamp_source=required_frame_sources,
        require_robot_timestamp_source=required_robot_sources,
    )
    sync_quality["calibration_attempt_policy"] = {
        "purpose": "authoritative_calibration_solver_pairing",
        "synchronization_policy": time_offset_search["policy"],
        "time_offset_search": _attempt_artifact_reference(
            str(request_value["attempt_id"]),
            TIME_OFFSET_SEARCH,
        ),
        "sign_convention": time_offset_search["sign_convention"],
        **timestamp_policy,
        "per_sensor_offsets": {
            sensor_key: {
                "robot_pose_time_offset_ms": float(
                    value["selected_robot_pose_time_offset_ms"]
                ),
                "sync_delta_ms": float(value["selected_sync_delta_ms"]),
                "status": value["status"],
            }
            for sensor_key, value in result_by_sensor.items()
        },
        "max_nearest_pose_delta_ms": max_nearest_pose_delta_ms,
        "historical_per_sensor_offsets_allowed": False,
        "auto_estimated_per_sensor_offsets": (
            time_offset_search["policy"] == "auto_offset"
        ),
    }
    checks = sync_quality.get("checks")
    if not isinstance(checks, list):
        checks = []
        sync_quality["checks"] = checks
    summaries = sync_quality.get("sensors")
    observed_names: set[str] = set()
    if isinstance(summaries, list):
        for summary in summaries:
            if not isinstance(summary, Mapping):
                continue
            sensor_name = str(summary.get("sensor_name") or "")
            observed_names.add(sensor_name)
            expected = expected_by_sensor_name.get(sensor_name)
            try:
                actual = float(summary["sync_delta_ms"])
            except (KeyError, TypeError, ValueError):
                actual = None
            matched = (
                expected is not None
                and actual is not None
                and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9)
            )
            checks.append(
                {
                    "name": f"calibration_authoritative_sync_delta:{sensor_name}",
                    "status": "ok" if matched else "error",
                    "message": (
                        f"{sensor_name} authoritative sync delta is {actual:g} ms."
                        if matched
                        else (
                            f"{sensor_name} authoritative sync delta "
                            f"{actual!r} does not match {expected!r} ms."
                        )
                    ),
                    "details": {
                        "actual_sync_delta_ms": actual,
                        "expected_sync_delta_ms": expected,
                    },
                }
            )
    missing_names = sorted(set(expected_by_sensor_name) - observed_names)
    for sensor_name in missing_names:
        checks.append(
            {
                "name": f"calibration_authoritative_sync_delta:{sensor_name}",
                "status": "error",
                "message": "Authoritative sync-delta evidence is missing.",
            }
        )
    statuses = {str(item.get("status")) for item in checks if isinstance(item, Mapping)}
    sync_quality["overall_status"] = (
        "error" if "error" in statuses else "warning" if "warning" in statuses else "ok"
    )
    atomic_write_json(attempt_root / SYNC_QUALITY_REPORT, sync_quality)
    blocking = [
        item
        for item in checks
        if isinstance(item, Mapping)
        and (
            item.get("status") == "error"
            or (
                str(item.get("name", "")).startswith("sync_nearest_pose_delta:")
                and item.get("status") != "ok"
            )
        )
    ]
    if blocking:
        raise ValueError(
            "Authoritative calibration synchronization failed: "
            + ", ".join(str(item.get("name")) for item in blocking)
        )

    remapped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for sensor_key, by_method in observations.items():
        matched = _read_json(synchronized[sensor_key] / MATCH_ROBOT_EE_POSES)
        source_matches: dict[str, tuple[str, Mapping[str, Any]]] = {}
        for final_frame_id, value in matched.items():
            if not isinstance(value, Mapping):
                continue
            source_frame_id = str(value.get("source_frame_id") or "")
            if not source_frame_id:
                continue
            if source_frame_id in source_matches:
                raise ValueError(
                    f"{sensor_key}: duplicate authoritative source frame "
                    f"{source_frame_id}"
                )
            source_matches[source_frame_id] = (final_frame_id, value)
        remapped[sensor_key] = {}
        selected = result_by_sensor[sensor_key]
        for method, items in by_method.items():
            remapped_items = []
            for item in items:
                source_frame_id = str(item.get("source_frame_id") or "")
                final_match = source_matches.get(source_frame_id)
                if final_match is None:
                    continue
                final_frame_id, match = final_match
                if int(item["image_timestamp_ns"]) != int(match["image_timestamp_ns"]):
                    raise ValueError(
                        f"{sensor_key}: authoritative timestamp changed for source "
                        f"frame {source_frame_id}"
                    )
                remapped_items.append(
                    {
                        **dict(item),
                        "observation_id": (f"{sensor_key}:{method}:{final_frame_id}"),
                        "frame_id": final_frame_id,
                        "source_frame_id": source_frame_id,
                        "motion": match["motion"],
                        "robot_ee_pose": dict(match["robot_ee_pose"]),
                        "image_timestamp_ns": match["image_timestamp_ns"],
                        "robot_pose_time_offset_ms": float(
                            selected["selected_robot_pose_time_offset_ms"]
                        ),
                        "sync_delta_ms": float(selected["selected_sync_delta_ms"]),
                        "timestamp_alignment": {
                            "frame_timestamp_ns": match["image_timestamp_ns"],
                            "robot_pose_query_timestamp_ns": match[
                                "delayed_timestamp_ns"
                            ],
                            "robot_pose_time_offset_ms": float(
                                selected["selected_robot_pose_time_offset_ms"]
                            ),
                            "sync_delta_ms": float(selected["selected_sync_delta_ms"]),
                            "matched_robot_pose_index": match[
                                "matched_robot_pose_index"
                            ],
                            "robot_timestamp_ns": match["robot_timestamp_ns"],
                            "nearest_robot_delta_ns": match["nearest_robot_delta_ns"],
                            "source": _attempt_artifact_reference(
                                str(request_value["attempt_id"]),
                                TIME_OFFSET_SEARCH,
                            ),
                        },
                    }
                )
            remapped[sensor_key][method] = remapped_items
    return synchronized, remapped


def _compare_solutions(
    attempt_root: Path,
    request_value: Mapping[str, Any],
    observations: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    time_offset_search: Mapping[str, Any],
) -> list[dict[str, Any]]:
    alignment_by_sensor = {
        str(item["sensor_key"]): item
        for item in time_offset_search["sensors"]
        if isinstance(item, Mapping)
    }
    candidates = []
    for sensor_key in request_value["sensor_keys"]:
        for pnp_method in request_value["pnp_methods"]:
            method_observations = observations[sensor_key][pnp_method]
            for extrinsic_method in request_value["extrinsic_methods"]:
                candidate = evaluate_extrinsic_candidate(
                    method_observations,
                    mode=request_value["mode"],
                    pnp_method=pnp_method,
                    extrinsic_method=extrinsic_method,
                    sensor_key=sensor_key,
                    min_accepted_views=DEFAULT_MIN_ACCEPTED_VIEWS,
                    min_coverage_cells=DEFAULT_MIN_COVERAGE_CELLS,
                    min_motion_poses=ATTEMPT_MIN_MOTION_POSES,
                    min_translation_span_mm=(ATTEMPT_MIN_TRANSLATION_SPAN_MM),
                    min_rotation_span_deg=ATTEMPT_MIN_ROTATION_SPAN_DEG,
                )
                candidate["synchronization"] = {
                    "policy": time_offset_search["policy"],
                    "status": alignment_by_sensor[sensor_key]["status"],
                    "robot_pose_time_offset_ms": float(
                        alignment_by_sensor[sensor_key][
                            "selected_robot_pose_time_offset_ms"
                        ]
                    ),
                    "sync_delta_ms": float(
                        alignment_by_sensor[sensor_key]["selected_sync_delta_ms"]
                    ),
                    "source": _attempt_artifact_reference(
                        str(request_value["attempt_id"]),
                        TIME_OFFSET_SEARCH,
                    ),
                }
                candidates.append(candidate)
    report = {
        "schema_version": "calibration_extrinsic_candidates.v1",
        "generated_at": utc_now_iso(),
        "attempt_id": request_value["attempt_id"],
        "mode": request_value["mode"],
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    atomic_write_json(attempt_root / EXTRINSIC_CANDIDATES_FILE, report)
    return candidates


def _camera_intrinsics(profile: Mapping[str, Any]) -> CameraIntrinsics:
    native = profile["native"]
    return CameraIntrinsics(
        cam_k=tuple(float(item) for item in native["cam_K"]),
        width=int(native["width"]),
        height=int(native["height"]),
        distortion=tuple(float(item) for item in native["distortion"]),
        depth_scale_to_mm=float(profile["depth"]["scale_to_mm"]),
        distortion_model=str(native.get("distortion_model", "brown_conrady")),
        projection_source=str(
            profile.get("attempt_intrinsics_source")
            or profile.get("source", {}).get("camera_projection")
            or "attempt_intrinsic_profile"
        ),
    )


def _candidate_profile(
    candidate: Mapping[str, Any],
    *,
    request_value: Mapping[str, Any],
    sensor: Mapping[str, Any],
    intrinsic_profile: Mapping[str, Any],
) -> CalibrationProfile:
    transform = candidate["primary_transform"]
    quaternion = tuple(float(item) for item in transform["rotation_quaternion_wxyz"])
    translation = tuple(float(item) for item in transform["translation_mm"])
    mode = str(request_value["mode"])
    raw_timestamp_policy = request_value.get("timestamp_policy")
    timestamp_policy = (
        dict(raw_timestamp_policy)
        if isinstance(raw_timestamp_policy, Mapping)
        else _attempt_timestamp_policy(request_value["sensors"])
    )
    sensor_timestamp_policy = _timestamp_policy_for_sensor(timestamp_policy, sensor)
    raw_synchronization = candidate.get("synchronization")
    synchronization = (
        dict(raw_synchronization)
        if isinstance(raw_synchronization, Mapping)
        else {
            "policy": DEFAULT_SYNCHRONIZATION_POLICY,
            "status": "fixed_zero",
            "robot_pose_time_offset_ms": 0.0,
            "sync_delta_ms": ATTEMPT_SYNC_DELTA_MS,
            "source": None,
        }
    )
    selected_sync_delta_ms = float(synchronization["sync_delta_ms"])
    mounting = (
        MountingMode.EYE_IN_HAND if mode == "eye_in_hand" else MountingMode.STATIC
    )
    safe_sensor = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sensor["device_id"]))
    safe_method = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        f"{candidate['pnp_method']}_{candidate['extrinsic_method']}",
    )
    profile_id = (
        f"{safe_sensor}_{mode}_{safe_method}_{str(request_value['attempt_id'])[:8]}"
    )
    intrinsics = _camera_intrinsics(intrinsic_profile)
    return CalibrationProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        profile_id=profile_id,
        sensor_id=str(sensor["device_id"]),
        sensor_type=SensorType(str(sensor["sensor_type"])),
        mounting_mode=mounting,
        rig_position="wrist" if mode == "eye_in_hand" else "static",
        intrinsics=intrinsics,
        rectified_intrinsics=rectified_intrinsics_from_native(intrinsics),
        extrinsics=RigidTransform(
            from_frame=TransformFrame.CAMERA,
            to_frame=(
                TransformFrame.ROBOT_FLANGE
                if mode == "eye_in_hand"
                else TransformFrame.TEMPLATE_BASE
            ),
            rotation_quaternion_wxyz=quaternion,  # type: ignore[arg-type]
            translation_mm=translation,  # type: ignore[arg-type]
        ),
        target_type=CalibrationTargetType.ARUCO_GRID,
        calibration_dataset_id=str(request_value["attempt_id"]),
        sync_delta_ms=selected_sync_delta_ms,
        method=f"auto_compare:{candidate['pnp_method']}+{candidate['extrinsic_method']}",
        status=CalibrationStatus.NEEDS_VALIDATION,
        quality=CalibrationQuality(
            num_observations=int(candidate["observation_count"]),
            num_inliers=int(candidate["inlier_count"]),
            mean_reprojection_error_px=candidate.get("mean_reprojection_error_px"),
            residual_translation_mm=float(
                candidate["held_out_residuals"]["mean_translation_mm"]
            ),
            residual_rotation_deg=float(
                candidate["held_out_residuals"]["mean_rotation_deg"]
            ),
            notes="Deterministic leave-one-pose-out calibration attempt candidate.",
        ),
        metadata={
            "sensor_name": sensor["sensor_name"],
            "sensor_key": sensor["sensor_key"],
            "attempt_id": request_value["attempt_id"],
            "candidate_id": candidate["candidate_id"],
            "solver_policy": request_value["solver_policy"],
            "pnp_method": candidate["pnp_method"],
            "extrinsic_method": candidate["extrinsic_method"],
            "target_id": request_value["target_id"],
            "target_mounting": request_value["target_mounting"],
            "companion_transform": candidate["companion_transform"],
            "held_out_residuals": candidate["held_out_residuals"],
            "outlier_count": candidate["outlier_count"],
            "outlier_ratio": candidate["outlier_ratio"],
            "intrinsic_profile_id": intrinsic_profile["profile_id"],
            "intrinsics_policy": request_value["intrinsics_policy"],
            "synchronization": {
                **synchronization,
                "sync_delta_ms": selected_sync_delta_ms,
                "timestamp_source": sensor_timestamp_policy["frame_timestamp_source"],
                "frame_timestamp_source": sensor_timestamp_policy[
                    "frame_timestamp_source"
                ],
                "robot_timestamp_source": sensor_timestamp_policy[
                    "robot_timestamp_source"
                ],
                "required_frame_timestamp_domain": sensor_timestamp_policy.get(
                    "required_frame_timestamp_domain"
                ),
                "timestamp_fallback_allowed": False,
                "max_nearest_pose_delta_ms": (ATTEMPT_MAX_NEAREST_POSE_DELTA_MS),
                "historical_per_sensor_offsets_allowed": False,
                "auto_estimated_per_sensor_offset": (
                    synchronization.get("policy") == "auto_offset"
                ),
                "sensor_key": sensor["sensor_key"],
                "quality_report": (
                    f"processed/calibration/{request_value['attempt_id']}/"
                    f"{SYNC_QUALITY_REPORT}"
                ),
            },
        },
    )


def _joint_companion_frame(
    request_value: Mapping[str, Any],
) -> dict[str, str] | None:
    """Return the shared estimated companion frame when joint ranking applies."""

    sensor_keys = request_value.get("sensor_keys")
    target_mounting = request_value.get("target_mounting")
    if (
        not isinstance(sensor_keys, Sequence)
        or isinstance(sensor_keys, (str, bytes))
        or len(sensor_keys) < 2
        or not isinstance(target_mounting, Mapping)
        or target_mounting.get("state") != "estimated"
    ):
        return None
    from_frame = str(target_mounting.get("from") or "").strip()
    to_frame = str(target_mounting.get("to") or "").strip()
    if not from_frame or not to_frame:
        return None
    return {"from": from_frame, "to": to_frame, "state": "estimated"}


def _algorithm_pair_sort_key(pair: tuple[str, str]) -> tuple[Any, ...]:
    pnp_order = {name: index for index, name in enumerate(PNP_METHOD_ORDER)}
    extrinsic_order = {name: index for index, name in enumerate(EXTRINSIC_METHOD_ORDER)}
    return (
        pnp_order.get(pair[0], len(pnp_order)),
        extrinsic_order.get(pair[1], len(extrinsic_order)),
        pair[0],
        pair[1],
    )


def _joint_algorithm_pairs(
    request_value: Mapping[str, Any],
    ranked_by_sensor: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[tuple[str, str]]:
    raw_pnp = request_value.get("pnp_methods")
    raw_extrinsic = request_value.get("extrinsic_methods")
    if (
        isinstance(raw_pnp, Sequence)
        and not isinstance(raw_pnp, (str, bytes))
        and raw_pnp
        and isinstance(raw_extrinsic, Sequence)
        and not isinstance(raw_extrinsic, (str, bytes))
        and raw_extrinsic
    ):
        pairs = {
            (str(pnp_method), str(extrinsic_method))
            for pnp_method in raw_pnp
            for extrinsic_method in raw_extrinsic
        }
    else:
        pairs = {
            (str(candidate.get("pnp_method")), str(candidate.get("extrinsic_method")))
            for candidates in ranked_by_sensor.values()
            for candidate in candidates
            if candidate.get("pnp_method") and candidate.get("extrinsic_method")
        }
    return sorted(pairs, key=_algorithm_pair_sort_key)


def _candidate_float(candidate: Mapping[str, Any], key: str) -> float | None:
    try:
        value = float(candidate[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _joint_bundle_record(
    *,
    sensor_keys: Sequence[str],
    pair: tuple[str, str],
    ranked_by_sensor: Mapping[str, Sequence[Mapping[str, Any]]],
    companion_frame: Mapping[str, str],
) -> dict[str, Any]:
    pnp_method, extrinsic_method = pair
    bundle_id = f"{pnp_method}|{extrinsic_method}"
    matches_by_sensor = {
        sensor_key: [
            candidate
            for candidate in ranked_by_sensor.get(sensor_key, [])
            if str(candidate.get("pnp_method")) == pnp_method
            and str(candidate.get("extrinsic_method")) == extrinsic_method
        ]
        for sensor_key in sensor_keys
    }
    candidate_options = {
        sensor_key: sorted(str(item.get("candidate_id")) for item in matches)
        for sensor_key, matches in matches_by_sensor.items()
    }
    selected = {
        sensor_key: matches[0]
        for sensor_key, matches in matches_by_sensor.items()
        if len(matches) == 1
    }
    candidate_ids = {
        sensor_key: str(candidate["candidate_id"])
        for sensor_key, candidate in selected.items()
    }
    passing_count = sum(
        candidate.get("status") == "passing" for candidate in selected.values()
    )
    scores = [_candidate_float(candidate, "score") for candidate in selected.values()]
    scores_valid = len(scores) == len(sensor_keys) and all(
        value is not None for value in scores
    )
    numeric_scores = [float(value) for value in scores if value is not None]
    aggregate_score = sum(numeric_scores) if scores_valid else None
    mean_score = (
        aggregate_score / len(sensor_keys) if aggregate_score is not None else None
    )
    reprojection_values = [
        value
        for candidate in selected.values()
        if (value := _candidate_float(candidate, "mean_reprojection_error_px"))
        is not None
    ]
    mean_reprojection_error_px = (
        sum(reprojection_values) / len(reprojection_values)
        if len(reprojection_values) == len(sensor_keys)
        else None
    )
    total_inlier_count = sum(
        int(candidate.get("inlier_count", 0)) for candidate in selected.values()
    )

    transforms: dict[str, np.ndarray] = {}
    transform_errors: dict[str, str] = {}
    for sensor_key, candidate in selected.items():
        raw_transform = candidate.get("companion_transform")
        if not isinstance(raw_transform, Mapping):
            transform_errors[sensor_key] = "companion transform is missing"
            continue
        actual_frame = {
            "from": str(raw_transform.get("from") or ""),
            "to": str(raw_transform.get("to") or ""),
        }
        expected_frame = {
            "from": companion_frame["from"],
            "to": companion_frame["to"],
        }
        if actual_frame != expected_frame:
            transform_errors[sensor_key] = (
                f"companion frame {actual_frame!r} does not match {expected_frame!r}"
            )
            continue
        try:
            transforms[sensor_key] = transform_from_record(raw_transform)
        except (TypeError, ValueError) as exc:
            transform_errors[sensor_key] = str(exc)

    pairwise_residuals: list[dict[str, Any]] = []
    for left_index, left_sensor_key in enumerate(sensor_keys):
        for right_sensor_key in sensor_keys[left_index + 1 :]:
            if left_sensor_key not in transforms or right_sensor_key not in transforms:
                continue
            residual = transform_residual(
                transforms[left_sensor_key], transforms[right_sensor_key]
            )
            pairwise_residuals.append(
                {
                    "left_sensor_key": left_sensor_key,
                    "right_sensor_key": right_sensor_key,
                    "left_candidate_id": candidate_ids[left_sensor_key],
                    "right_candidate_id": candidate_ids[right_sensor_key],
                    "translation_mm": residual["translation_mm"],
                    "rotation_deg": residual["rotation_deg"],
                    "status": (
                        "ok"
                        if residual["translation_mm"] <= DEFAULT_MAX_MEAN_TRANSLATION_MM
                        and residual["rotation_deg"] <= DEFAULT_MAX_MEAN_ROTATION_DEG
                        else "error"
                    ),
                }
            )
    expected_pair_count = len(sensor_keys) * (len(sensor_keys) - 1) // 2
    max_translation_mm = (
        max(item["translation_mm"] for item in pairwise_residuals)
        if pairwise_residuals
        else None
    )
    max_rotation_deg = (
        max(item["rotation_deg"] for item in pairwise_residuals)
        if pairwise_residuals
        else None
    )
    normalized_companion_closure_score = (
        max_translation_mm / DEFAULT_MAX_MEAN_TRANSLATION_MM
        + max_rotation_deg / DEFAULT_MAX_MEAN_ROTATION_DEG
        if max_translation_mm is not None and max_rotation_deg is not None
        else None
    )
    presence_ok = len(selected) == len(sensor_keys)
    passing_ok = passing_count == len(sensor_keys)
    transform_ok = (
        len(transforms) == len(sensor_keys)
        and len(pairwise_residuals) == expected_pair_count
    )
    translation_ok = (
        transform_ok
        and max_translation_mm is not None
        and max_translation_mm <= DEFAULT_MAX_MEAN_TRANSLATION_MM
    )
    rotation_ok = (
        transform_ok
        and max_rotation_deg is not None
        and max_rotation_deg <= DEFAULT_MAX_MEAN_ROTATION_DEG
    )
    checks = [
        {
            "name": "joint_candidate_presence",
            "status": "ok" if presence_ok else "error",
            "actual": len(selected),
            "threshold": len(sensor_keys),
        },
        {
            "name": "joint_individual_candidate_validation",
            "status": "ok" if passing_ok else "error",
            "actual": passing_count,
            "threshold": len(sensor_keys),
        },
        {
            "name": "joint_individual_score_validity",
            "status": "ok" if scores_valid else "error",
            "actual": len(numeric_scores),
            "threshold": len(sensor_keys),
        },
        {
            "name": "joint_companion_transform_validity",
            "status": "ok" if transform_ok else "error",
            "actual": len(transforms),
            "threshold": len(sensor_keys),
            "errors": transform_errors,
        },
        {
            "name": "joint_companion_translation_consistency",
            "status": "ok" if translation_ok else "error",
            "actual": max_translation_mm,
            "threshold": DEFAULT_MAX_MEAN_TRANSLATION_MM,
            "unit": "mm",
        },
        {
            "name": "joint_companion_rotation_consistency",
            "status": "ok" if rotation_ok else "error",
            "actual": max_rotation_deg,
            "threshold": DEFAULT_MAX_MEAN_ROTATION_DEG,
            "unit": "deg",
        },
    ]
    passing = (
        presence_ok
        and passing_ok
        and scores_valid
        and transform_ok
        and translation_ok
        and rotation_ok
    )
    return {
        "bundle_id": bundle_id,
        "pnp_method": pnp_method,
        "extrinsic_method": extrinsic_method,
        "algorithms": [pnp_method, extrinsic_method],
        "sensor_keys": list(sensor_keys),
        "candidate_ids": candidate_ids,
        "candidate_options": candidate_options,
        "status": "passing" if passing else "failed",
        "aggregate_score": aggregate_score,
        "mean_score": mean_score,
        "mean_reprojection_error_px": mean_reprojection_error_px,
        "total_inlier_count": total_inlier_count,
        "companion_frame": dict(companion_frame),
        "pairwise_companion_residuals": pairwise_residuals,
        "max_pairwise_companion_translation_mm": max_translation_mm,
        "max_pairwise_companion_rotation_deg": max_rotation_deg,
        "normalized_companion_closure_score": (normalized_companion_closure_score),
        "checks": checks,
    }


def _joint_consistency_ranking(
    request_value: Mapping[str, Any],
    ranked_by_sensor: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any] | None:
    companion_frame = _joint_companion_frame(request_value)
    if companion_frame is None:
        return None
    sensor_keys = [str(item) for item in request_value["sensor_keys"]]
    bundles = [
        _joint_bundle_record(
            sensor_keys=sensor_keys,
            pair=pair,
            ranked_by_sensor=ranked_by_sensor,
            companion_frame=companion_frame,
        )
        for pair in _joint_algorithm_pairs(request_value, ranked_by_sensor)
    ]
    passing_mean_scores = [
        float(bundle["mean_score"])
        for bundle in bundles
        if bundle.get("status") == "passing" and bundle.get("mean_score") is not None
    ]
    best_individual_score = min(passing_mean_scores, default=None)
    for bundle in bundles:
        mean_score = bundle.get("mean_score")
        score_delta = (
            float(mean_score) - best_individual_score
            if mean_score is not None and best_individual_score is not None
            else None
        )
        quality_equivalent = bool(
            bundle.get("status") == "passing"
            and score_delta is not None
            and score_delta <= JOINT_INDIVIDUAL_SCORE_EQUIVALENCE_TOLERANCE + 1e-12
        )
        bundle["individual_score_delta_from_best"] = score_delta
        bundle["individual_score_equivalent_to_best"] = quality_equivalent
    equivalent_closure_scores = [
        float(bundle["normalized_companion_closure_score"])
        for bundle in bundles
        if bundle["individual_score_equivalent_to_best"]
        and bundle.get("normalized_companion_closure_score") is not None
    ]
    best_equivalent_closure_score = min(equivalent_closure_scores, default=None)
    for bundle in bundles:
        closure_score = bundle.get("normalized_companion_closure_score")
        closure_delta = (
            float(closure_score) - best_equivalent_closure_score
            if bundle["individual_score_equivalent_to_best"]
            and closure_score is not None
            and best_equivalent_closure_score is not None
            else None
        )
        bundle["closure_score_delta_from_best_equivalent"] = closure_delta
        bundle["closure_score_equivalent_to_best"] = bool(
            closure_delta is not None
            and closure_delta <= JOINT_CLOSURE_SCORE_EQUIVALENCE_TOLERANCE + 1e-12
        )

    def ranking_number(value: Any) -> float:
        return (
            round(float(value), JOINT_RANKING_NUMERIC_DECIMALS)
            if value is not None
            else math.inf
        )

    def bundle_sort_key(bundle: Mapping[str, Any]) -> tuple[Any, ...]:
        mean_score = bundle.get("mean_score")
        aggregate_score = bundle.get("aggregate_score")
        closure_score = bundle.get("normalized_companion_closure_score")
        reprojection = bundle.get("mean_reprojection_error_px")
        passing = bundle.get("status") == "passing"
        quality_equivalent = bool(bundle.get("individual_score_equivalent_to_best"))
        closure_equivalent = bool(bundle.get("closure_score_equivalent_to_best"))
        algorithm_key = _algorithm_pair_sort_key(
            (str(bundle["pnp_method"]), str(bundle["extrinsic_method"]))
        )
        quality_key = (
            ranking_number(mean_score),
            ranking_number(aggregate_score),
            ranking_number(reprojection),
            -int(bundle.get("total_inlier_count", 0)),
        )
        if passing and quality_equivalent and closure_equivalent:
            return (0, 0, 0, *algorithm_key, *quality_key, str(bundle["bundle_id"]))
        if passing and quality_equivalent:
            return (
                0,
                0,
                1,
                ranking_number(closure_score),
                *quality_key,
                *algorithm_key,
                str(bundle["bundle_id"]),
            )
        return (
            0 if passing else 1,
            1,
            0,
            *quality_key,
            ranking_number(closure_score),
            *algorithm_key,
            str(bundle["bundle_id"]),
        )

    bundles = [dict(item) for item in sorted(bundles, key=bundle_sort_key)]
    for index, bundle in enumerate(bundles, start=1):
        bundle["rank"] = index
        bundle["recommended"] = index == 1 and bundle["status"] == "passing"
    recommendation = next((bundle for bundle in bundles if bundle["recommended"]), None)
    return {
        "required": True,
        "status": "passing" if recommendation else "failed",
        "sensor_keys": sensor_keys,
        "sensor_count": len(sensor_keys),
        "companion_frame": companion_frame,
        "thresholds": {
            "max_pairwise_companion_translation_mm": (DEFAULT_MAX_MEAN_TRANSLATION_MM),
            "max_pairwise_companion_rotation_deg": DEFAULT_MAX_MEAN_ROTATION_DEG,
            "individual_score_equivalence_tolerance": (
                JOINT_INDIVIDUAL_SCORE_EQUIVALENCE_TOLERANCE
            ),
            "closure_score_equivalence_tolerance": (
                JOINT_CLOSURE_SCORE_EQUIVALENCE_TOLERANCE
            ),
        },
        "ranking_policy": {
            "individual_quality_metric": "mean_score",
            "best_individual_score": best_individual_score,
            "individual_score_equivalence_tolerance": (
                JOINT_INDIVIDUAL_SCORE_EQUIVALENCE_TOLERANCE
            ),
            "equivalent_quality_ordering_metric": (
                "normalized_companion_closure_score"
            ),
            "normalized_companion_closure_score_definition": (
                "max_pairwise_translation_mm/max_translation_mm + "
                "max_pairwise_rotation_deg/max_rotation_deg"
            ),
            "numeric_round_decimals": JOINT_RANKING_NUMERIC_DECIMALS,
            "closure_score_equivalence_tolerance": (
                JOINT_CLOSURE_SCORE_EQUIVALENCE_TOLERANCE
            ),
            "closure_equivalent_ordering": "canonical_algorithm_order",
            "outside_equivalence_band_ordering_metric": "mean_score",
        },
        "bundle_count": len(bundles),
        "passing_bundle_count": sum(
            bundle["status"] == "passing" for bundle in bundles
        ),
        "recommended_bundle_id": (
            recommendation["bundle_id"] if recommendation else None
        ),
        "recommendation": recommendation,
        "bundles": bundles,
    }


def _validate_and_rank(
    attempt_root: Path,
    request_value: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    intrinsics: Mapping[str, Mapping[str, Any]],
    *,
    time_offset_search: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw_timestamp_policy = request_value.get("timestamp_policy")
    timestamp_policy = (
        dict(raw_timestamp_policy)
        if isinstance(raw_timestamp_policy, Mapping)
        else _attempt_timestamp_policy(request_value["sensors"])
    )
    if time_offset_search is None:
        time_offset_search = {
            "policy": "fixed_zero",
            "sensors": [
                fixed_zero_sensor_result(
                    sensor_key=str(sensor_key),
                    observation_count=0,
                )
                for sensor_key in request_value["sensor_keys"]
            ],
        }
    sensor_metadata = {
        str(item["sensor_key"]): item for item in request_value["sensors"]
    }
    alignment_by_sensor = {
        str(item["sensor_key"]): item
        for item in time_offset_search["sensors"]
        if isinstance(item, Mapping)
    }
    profiles: list[CalibrationProfile] = []
    results = []
    all_checks = []
    ranked_by_sensor: dict[str, list[dict[str, Any]]] = {}
    for sensor_key in request_value["sensor_keys"]:
        ranked = rank_candidates(
            [item for item in candidates if item["sensor_key"] == sensor_key]
        )
        ranked_by_sensor[str(sensor_key)] = ranked
        for item in ranked:
            if "primary_transform" in item:
                profile = _candidate_profile(
                    item,
                    request_value=request_value,
                    sensor=sensor_metadata[sensor_key],
                    intrinsic_profile=intrinsics[sensor_key],
                )
                profiles.append(profile)
                item["profile_id"] = profile.profile_id
            all_checks.extend(
                {**dict(check), "candidate_id": item["candidate_id"]}
                for check in item.get("checks", [])
            )

    joint_consistency = _joint_consistency_ranking(request_value, ranked_by_sensor)
    if joint_consistency is not None:
        for ranked in ranked_by_sensor.values():
            for item in ranked:
                item["recommended"] = False
                item.pop("joint_bundle_id", None)
                item.pop("recommendation_basis", None)
        joint_recommendation = joint_consistency.get("recommendation")
        if isinstance(joint_recommendation, Mapping):
            selected_candidate_ids = set(
                joint_recommendation.get("candidate_ids", {}).values()
            )
            for ranked in ranked_by_sensor.values():
                for item in ranked:
                    if item["candidate_id"] in selected_candidate_ids:
                        item["recommended"] = True
                        item["joint_bundle_id"] = joint_recommendation["bundle_id"]
                        item["recommendation_basis"] = (
                            "multi_camera_companion_consistency"
                        )
        for bundle in joint_consistency["bundles"]:
            all_checks.extend(
                {
                    **dict(check),
                    "scope": "multi_camera_bundle",
                    "bundle_id": bundle["bundle_id"],
                }
                for check in bundle.get("checks", [])
            )

    for sensor_key in request_value["sensor_keys"]:
        ranked = ranked_by_sensor[str(sensor_key)]
        recommendation = next(
            (item for item in ranked if item.get("recommended")), None
        )
        results.append(
            {
                **sensor_metadata[sensor_key],
                "status": "passing" if recommendation else "failed",
                "recommended_candidate_id": (
                    recommendation["candidate_id"] if recommendation else None
                ),
                "recommended_profile_id": (
                    recommendation.get("profile_id") if recommendation else None
                ),
                "recommendation": recommendation,
                "time_offset_search": alignment_by_sensor[sensor_key],
                "candidates": ranked,
            }
        )
    write_profile_collection(profiles, attempt_root / CANDIDATE_PROFILES_FILE)
    ranking = {
        "schema_version": "calibration_ranking.v1",
        "generated_at": utc_now_iso(),
        "attempt_id": request_value["attempt_id"],
        "mode": request_value["mode"],
        "status": (
            "complete"
            if all(item["status"] == "passing" for item in results)
            else "partial"
            if any(item["status"] == "passing" for item in results)
            else "failed"
        ),
        "recommended_camera_count": sum(
            1 for item in results if item["status"] == "passing"
        ),
        "failed_camera_count": sum(1 for item in results if item["status"] == "failed"),
        "thresholds": {
            "min_inliers": 6,
            "min_accepted_views": DEFAULT_MIN_ACCEPTED_VIEWS,
            "min_coverage_cells": DEFAULT_MIN_COVERAGE_CELLS,
            "max_per_view_reprojection_error_px": (DEFAULT_MAX_VIEW_ERROR_PX),
            "max_intrinsic_rms_reprojection_error_px": DEFAULT_MAX_RMS_PX,
            "min_motion_poses": ATTEMPT_MIN_MOTION_POSES,
            "min_translation_span_mm": ATTEMPT_MIN_TRANSLATION_SPAN_MM,
            "min_rotation_span_deg": ATTEMPT_MIN_ROTATION_SPAN_DEG,
            "min_rotation_axis_angle_deg": (DEFAULT_MIN_ROTATION_AXIS_ANGLE_DEG),
            "min_rotation_axis_second_to_first_ratio": (
                DEFAULT_MIN_ROTATION_AXIS_SINGULAR_RATIO
            ),
            "max_observations_per_motion": (DEFAULT_MAX_OBSERVATIONS_PER_MOTION),
            "max_nearest_pose_delta_ms": (ATTEMPT_MAX_NEAREST_POSE_DELTA_MS),
            "timestamp_source": timestamp_policy["frame_timestamp_source"],
            "robot_timestamp_source": timestamp_policy["robot_timestamp_source"],
            "synchronization_policy": time_offset_search["policy"],
            "sync_delta_ms": (
                0.0
                if all(
                    math.isclose(
                        float(item["selected_sync_delta_ms"]),
                        0.0,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    for item in time_offset_search["sensors"]
                )
                else "per_sensor"
            ),
            "per_sensor_sync_delta_ms": {
                str(item["sensor_key"]): float(item["selected_sync_delta_ms"])
                for item in time_offset_search["sensors"]
            },
            "max_mean_translation_mm": 10.0,
            "max_mean_rotation_deg": 5.0,
            "max_outlier_ratio": 0.25,
        },
        "results": results,
    }
    if joint_consistency is not None:
        ranking["multi_camera_consistency"] = joint_consistency
    atomic_write_json(attempt_root / RANKING_FILE, ranking)
    atomic_write_json(
        attempt_root / CHECKS_FILE,
        {
            "schema_version": "calibration_attempt_checks.v1",
            "attempt_id": request_value["attempt_id"],
            "checks": all_checks,
        },
    )
    return ranking


def run_calibration_attempt(run_root: str | Path, attempt_id: str) -> dict[str, Any]:
    root = Path(run_root)
    attempt_root = calibration_attempt_root(root, attempt_id)
    request_value = _read_json(attempt_root / REQUEST_FILE)
    initial_progress = _read_json(attempt_root / PROGRESS_FILE)
    _validate_attempt_identity(root, attempt_id, request_value, initial_progress)
    if initial_progress.get("status") != "queued":
        raise ValueError(
            "Calibration attempts are immutable and may only be calculated once"
        )
    try:
        _update_progress(
            attempt_root,
            status="running",
            phase="prepare_data",
            phase_status="running",
            message="Synchronizing the selected camera subset.",
        )
        synchronized, intrinsics = _prepare_attempt_data(
            root, attempt_root, request_value
        )
        _update_progress(
            attempt_root,
            phase="prepare_data",
            phase_status="complete",
            message="Selected camera data and compatible intrinsics are ready.",
        )
        _update_progress(
            attempt_root,
            phase="estimate_target_poses",
            phase_status="running",
            message="Comparing planar target-pose estimates.",
        )
        _pnp, observations = _estimate_target_poses(
            attempt_root,
            request_value,
            synchronized,
            intrinsics,
        )
        _update_progress(
            attempt_root,
            phase="estimate_target_poses",
            phase_status="complete",
            message="Target poses were estimated with the shared robust mask.",
        )
        _update_progress(
            attempt_root,
            phase="estimate_time_offsets",
            phase_status="running",
            message=(
                "Estimating effective camera-to-robot latency on fixed "
                "motion-disjoint evidence."
            ),
        )
        time_offset_search, adjusted_observations = _estimate_and_apply_time_offsets(
            root,
            attempt_root,
            request_value,
            observations,
        )
        _authoritative_synchronized, observations = (
            _materialize_authoritative_synchronization(
                root,
                attempt_root,
                request_value,
                time_offset_search,
                adjusted_observations,
            )
        )
        observation_report = _calibration_observation_report(
            request_value, observations
        )
        observation_report["time_offset_search"] = _attempt_artifact_reference(
            str(request_value["attempt_id"]),
            TIME_OFFSET_SEARCH,
        )
        observation_report["synchronization_policy"] = time_offset_search["policy"]
        atomic_write_json(attempt_root / CALIBRATION_OBSERVATIONS, observation_report)
        _update_progress(
            attempt_root,
            phase="estimate_time_offsets",
            phase_status="complete",
            message="Authoritative camera/robot time alignment is ready.",
        )
        _update_progress(
            attempt_root,
            phase="compare_robot_camera_solutions",
            phase_status="running",
            message="Evaluating every compatible PnP/extrinsic combination.",
        )
        candidates = _compare_solutions(
            attempt_root,
            request_value,
            observations,
            time_offset_search=time_offset_search,
        )
        _update_progress(
            attempt_root,
            phase="compare_robot_camera_solutions",
            phase_status="complete",
            message="Robot-camera solver comparison is complete.",
        )
        _update_progress(
            attempt_root,
            phase="validate_and_rank",
            phase_status="running",
            message="Applying validation gates and deterministic ranking.",
        )
        ranking = _validate_and_rank(
            attempt_root,
            request_value,
            candidates,
            intrinsics,
            time_offset_search=time_offset_search,
        )
        _update_progress(
            attempt_root,
            status="complete",
            phase="validate_and_rank",
            phase_status="complete",
            message="Calibration calculations are complete and awaiting review.",
        )
        return ranking
    except Exception as exc:
        progress = _read_json(attempt_root / PROGRESS_FILE)
        current = progress.get("current_phase")
        _update_progress(
            attempt_root,
            status="failed",
            phase=str(current) if current else None,
            phase_status="failed" if current else None,
            message=f"{type(exc).__name__}: {exc}",
        )
        raise


def load_calibration_attempt(run_root: str | Path, attempt_id: str) -> dict[str, Any]:
    root = Path(run_root)
    attempt_root = calibration_attempt_root(root, attempt_id)
    if not attempt_root.is_dir():
        raise FileNotFoundError(f"Calibration attempt not found: {attempt_id}")
    request_value = _read_json(attempt_root / REQUEST_FILE)
    progress = _read_json(attempt_root / PROGRESS_FILE)
    _validate_attempt_identity(root, attempt_id, request_value, progress)
    ranking = (
        _read_json(attempt_root / RANKING_FILE)
        if (attempt_root / RANKING_FILE).is_file()
        else None
    )
    promotion = (
        _read_json(attempt_root / PROMOTION_FILE)
        if (attempt_root / PROMOTION_FILE).is_file()
        else None
    )
    intrinsic_comparison = (
        _read_json(attempt_root / INTRINSIC_COMPARISON)
        if (attempt_root / INTRINSIC_COMPARISON).is_file()
        else None
    )
    time_offset_search = (
        _read_json(attempt_root / TIME_OFFSET_SEARCH)
        if (attempt_root / TIME_OFFSET_SEARCH).is_file()
        else None
    )
    return {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "run_root": root.as_posix(),
        "request": request_value,
        "progress": progress,
        "results": ranking,
        "intrinsic_comparison": intrinsic_comparison,
        "time_offset_search": time_offset_search,
        "promotion": promotion,
        "artifacts": {
            name: _relative(attempt_root / name, root)
            for name in (
                REQUEST_FILE,
                PROGRESS_FILE,
                SYNC_QUALITY_REPORT,
                TIME_OFFSET_SEARCH,
                INTRINSIC_COMPARISON,
                INTRINSIC_CALIBRATION_PROFILES,
                PNP_CANDIDATES_FILE,
                CALIBRATION_OBSERVATIONS,
                EXTRINSIC_CANDIDATES_FILE,
                RANKING_FILE,
                CHECKS_FILE,
                CANDIDATE_PROFILES_FILE,
                PROMOTION_FILE,
            )
            if (attempt_root / name).exists()
        },
    }


def _optional_floats_match(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(left_value)
        and math.isfinite(right_value)
        and math.isclose(left_value, right_value, rel_tol=0.0, abs_tol=1e-9)
    )


def _revalidate_joint_promotion(
    attempt: Mapping[str, Any],
    selections: Mapping[str, str],
    *,
    expected_bundle_id: str | None = None,
) -> dict[str, Any] | None:
    request_value = attempt.get("request")
    if not isinstance(request_value, Mapping):
        raise ValueError("Calibration attempt request evidence is missing")
    companion_frame = _joint_companion_frame(request_value)
    if companion_frame is None:
        if expected_bundle_id is not None:
            raise ValueError(
                "Single-camera promotion unexpectedly names a multi-camera bundle"
            )
        return None

    sensor_keys = [str(item) for item in request_value["sensor_keys"]]
    if set(selections) != set(sensor_keys):
        raise ValueError(
            "Multi-camera promotion must select every jointly ranked sensor"
        )
    ranking = attempt.get("results")
    if not isinstance(ranking, Mapping):
        raise ValueError("Calibration ranking evidence is missing")
    consistency = ranking.get("multi_camera_consistency")
    if not isinstance(consistency, Mapping) or consistency.get("required") is not True:
        raise ValueError("Multi-camera consistency evidence is missing")
    if consistency.get("companion_frame") != companion_frame:
        raise ValueError("Multi-camera companion-frame evidence is inconsistent")
    if [str(item) for item in consistency.get("sensor_keys", [])] != sensor_keys:
        raise ValueError("Multi-camera sensor-order evidence is inconsistent")
    thresholds = consistency.get("thresholds")
    if (
        not isinstance(thresholds, Mapping)
        or not _optional_floats_match(
            thresholds.get("max_pairwise_companion_translation_mm"),
            DEFAULT_MAX_MEAN_TRANSLATION_MM,
        )
        or not _optional_floats_match(
            thresholds.get("max_pairwise_companion_rotation_deg"),
            DEFAULT_MAX_MEAN_ROTATION_DEG,
        )
        or not _optional_floats_match(
            thresholds.get("individual_score_equivalence_tolerance"),
            JOINT_INDIVIDUAL_SCORE_EQUIVALENCE_TOLERANCE,
        )
        or not _optional_floats_match(
            thresholds.get("closure_score_equivalence_tolerance"),
            JOINT_CLOSURE_SCORE_EQUIVALENCE_TOLERANCE,
        )
    ):
        raise ValueError("Multi-camera consistency thresholds are invalid")

    results = {
        str(item.get("sensor_key")): item
        for item in ranking.get("results", [])
        if isinstance(item, Mapping) and item.get("sensor_key") is not None
    }
    selected_candidates: dict[str, Mapping[str, Any]] = {}
    for sensor_key in sensor_keys:
        result = results.get(sensor_key)
        if not isinstance(result, Mapping):
            raise ValueError(f"Multi-camera ranking result is missing for {sensor_key}")
        matches = [
            item
            for item in result.get("candidates", [])
            if isinstance(item, Mapping)
            and str(item.get("candidate_id")) == selections[sensor_key]
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Multi-camera candidate evidence is ambiguous for {sensor_key}"
            )
        selected_candidates[sensor_key] = matches[0]

    algorithm_pairs = {
        (
            str(candidate.get("pnp_method")),
            str(candidate.get("extrinsic_method")),
        )
        for candidate in selected_candidates.values()
    }
    if len(algorithm_pairs) != 1:
        raise ValueError(
            "Multi-camera promotion selections must use one common algorithm bundle"
        )
    pair = next(iter(algorithm_pairs))
    recalculated = _joint_bundle_record(
        sensor_keys=sensor_keys,
        pair=pair,
        ranked_by_sensor={
            sensor_key: [candidate]
            for sensor_key, candidate in selected_candidates.items()
        },
        companion_frame=companion_frame,
    )
    if recalculated["status"] != "passing":
        raise ValueError(
            "Selected multi-camera bundle no longer satisfies consistency gates"
        )

    recorded_matches = [
        bundle
        for bundle in consistency.get("bundles", [])
        if isinstance(bundle, Mapping)
        and bundle.get("candidate_ids") == dict(selections)
    ]
    if len(recorded_matches) != 1:
        raise ValueError("Selections do not match one recorded multi-camera bundle")
    recorded = recorded_matches[0]
    if (
        recorded.get("status") != "passing"
        or recorded.get("bundle_id") != recalculated["bundle_id"]
        or recorded.get("pnp_method") != pair[0]
        or recorded.get("extrinsic_method") != pair[1]
    ):
        raise ValueError("Recorded multi-camera bundle is not promotable")
    if expected_bundle_id is not None and recorded.get("bundle_id") != str(
        expected_bundle_id
    ):
        raise ValueError("Promotion request names a different multi-camera bundle")

    numeric_fields = (
        "aggregate_score",
        "mean_score",
        "max_pairwise_companion_translation_mm",
        "max_pairwise_companion_rotation_deg",
        "normalized_companion_closure_score",
    )
    if any(
        not _optional_floats_match(recorded.get(field), recalculated.get(field))
        for field in numeric_fields
    ):
        raise ValueError("Recorded multi-camera bundle summary is inconsistent")
    recorded_residuals = {
        (
            str(item.get("left_sensor_key")),
            str(item.get("right_sensor_key")),
            str(item.get("left_candidate_id")),
            str(item.get("right_candidate_id")),
        ): item
        for item in recorded.get("pairwise_companion_residuals", [])
        if isinstance(item, Mapping)
    }
    recalculated_residuals = {
        (
            str(item.get("left_sensor_key")),
            str(item.get("right_sensor_key")),
            str(item.get("left_candidate_id")),
            str(item.get("right_candidate_id")),
        ): item
        for item in recalculated["pairwise_companion_residuals"]
    }
    if recorded_residuals.keys() != recalculated_residuals.keys():
        raise ValueError("Recorded multi-camera pairwise evidence is incomplete")
    for key, recalculated_residual in recalculated_residuals.items():
        recorded_residual = recorded_residuals[key]
        if (
            recorded_residual.get("status") != recalculated_residual["status"]
            or not _optional_floats_match(
                recorded_residual.get("translation_mm"),
                recalculated_residual["translation_mm"],
            )
            or not _optional_floats_match(
                recorded_residual.get("rotation_deg"),
                recalculated_residual["rotation_deg"],
            )
        ):
            raise ValueError("Recorded multi-camera pairwise evidence is inconsistent")
    return dict(recorded)


def _promotion_selections(
    attempt: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> dict[str, str]:
    ranking = attempt.get("results")
    if not isinstance(ranking, Mapping):
        raise ValueError("Calibration calculations are not complete")
    explicit = overrides is not None
    supplied = {str(key): str(value) for key, value in (overrides or {}).items()}
    results = {
        str(item["sensor_key"]): item
        for item in ranking.get("results", [])
        if isinstance(item, Mapping)
    }
    unknown = sorted(set(supplied) - results.keys())
    if unknown:
        raise ValueError("Unknown promotion sensor key(s): " + ", ".join(unknown))
    selected = {}
    for sensor_key, result in results.items():
        candidate_id = (
            supplied.get(sensor_key)
            if explicit
            else result.get("recommended_candidate_id")
        )
        if not candidate_id:
            continue
        candidates = {
            str(item["candidate_id"]): item
            for item in result.get("candidates", [])
            if isinstance(item, Mapping)
        }
        candidate = candidates.get(str(candidate_id))
        if candidate is None:
            raise ValueError(
                f"Candidate {candidate_id!r} does not belong to {sensor_key}"
            )
        if candidate.get("status") != "passing":
            raise ValueError(f"Candidate {candidate_id!r} did not pass validation")
        selected[sensor_key] = str(candidate_id)
    if not selected:
        raise ValueError("No passing camera recommendations are available to promote")
    _revalidate_joint_promotion(attempt, selected)
    return selected


def _promotion_time_offset_evidence(
    attempt: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    request_value = attempt.get("request")
    if not isinstance(request_value, Mapping):
        raise ValueError("Calibration attempt request evidence is missing")
    has_explicit_policy = "synchronization_policy" in request_value
    policy = str(
        request_value.get("synchronization_policy", DEFAULT_SYNCHRONIZATION_POLICY)
    )
    report = attempt.get("time_offset_search")
    if report is None:
        if not has_explicit_policy:
            return {}
        raise ValueError("Calibration time-offset promotion evidence is missing")
    attempt_id = str(request_value.get("attempt_id") or attempt.get("attempt_id") or "")
    expected = {str(item) for item in request_value.get("sensor_keys", [])}
    raw_sensors = report.get("sensors", []) if isinstance(report, Mapping) else []
    recorded_search = request_value.get("synchronization_search")
    recorded_revision = request_value.get("synchronization_implementation_revision")
    if (
        not isinstance(report, Mapping)
        or not isinstance(recorded_search, Mapping)
        or not isinstance(recorded_revision, str)
        or recorded_revision not in TIME_OFFSET_SUPPORTED_REVISIONS
        or report.get("schema_version") != TIME_OFFSET_SEARCH_SCHEMA_VERSION
        or report.get("policy") != policy
        or report.get("status") != "complete"
        or report.get("attempt_id") != attempt_id
        or report.get("implementation_revision") != recorded_revision
        or report.get("offset_kind") != "effective_capture_and_pose_pipeline_latency"
        or report.get("sign_convention") != time_offset_sign_convention()
        or report.get("search") != dict(recorded_search)
        or report.get("failed_sensor_keys") != []
        or report.get("sensor_count") != len(expected)
        or not isinstance(raw_sensors, list)
        or len(raw_sensors) != len(expected)
    ):
        raise ValueError("Calibration time-offset promotion evidence is invalid")
    sensors = {
        str(item.get("sensor_key")): item
        for item in raw_sensors
        if isinstance(item, Mapping) and item.get("sensor_key")
    }
    if set(sensors) != expected:
        raise ValueError(
            "Calibration time-offset promotion evidence does not cover every sensor"
        )
    search_grid = time_offset_values(
        float(recorded_search["minimum_robot_pose_time_offset_ms"]),
        float(recorded_search["maximum_robot_pose_time_offset_ms"]),
        float(recorded_search["step_ms"]),
    )
    minimum_offset = min(search_grid)
    maximum_offset = max(search_grid)
    required_auto_checks = {
        "fixed_full_range_observation_set",
        "cross_validation_offset_stability",
        "reference_method_sensitivity",
        "search_optimum_not_at_boundary",
        "cross_validated_translation_improvement",
        "cross_validated_rotation_guard",
        "zero_offset_identifiability",
    }
    for sensor_key, item in sensors.items():
        status = str(item.get("status") or "")
        valid_statuses = (
            {"applied", "kept_zero"} if policy == "auto_offset" else {"fixed_zero"}
        )
        if status not in valid_statuses:
            raise ValueError(
                f"Calibration time-offset evidence is not promotable for {sensor_key}"
            )
        try:
            operator_offset = float(item["selected_robot_pose_time_offset_ms"])
            sync_delta = float(item["selected_sync_delta_ms"])
            candidate_offset = float(item["candidate_robot_pose_time_offset_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Calibration time-offset evidence is invalid for {sensor_key}"
            ) from exc
        if (
            not math.isfinite(operator_offset)
            or not math.isfinite(sync_delta)
            or not math.isfinite(candidate_offset)
            or not math.isclose(
                sync_delta,
                -operator_offset,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(
                f"Calibration time-offset sign evidence is inconsistent for {sensor_key}"
            )
        checks = item.get("checks")
        if not isinstance(checks, list):
            raise ValueError(
                f"Calibration time-offset checks are missing for {sensor_key}"
            )
        check_by_name = {
            str(check.get("name")): check
            for check in checks
            if isinstance(check, Mapping)
        }
        if policy == "auto_offset":
            if not required_auto_checks.issubset(check_by_name) or any(
                check.get("status") == "error" for check in check_by_name.values()
            ):
                raise ValueError(
                    f"Auto-sync checks are not promotable for {sensor_key}"
                )
            if status == "applied":
                if (
                    math.isclose(operator_offset, 0.0, rel_tol=0.0, abs_tol=1e-9)
                    or not math.isclose(
                        candidate_offset,
                        operator_offset,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    or item.get("boundary_hit") is not False
                    or not (minimum_offset < operator_offset < maximum_offset)
                    or not any(
                        math.isclose(
                            operator_offset,
                            value,
                            rel_tol=0.0,
                            abs_tol=1e-9,
                        )
                        for value in search_grid
                    )
                ):
                    raise ValueError(
                        f"Applied auto-sync offset is invalid for {sensor_key}"
                    )
            elif (
                not math.isclose(operator_offset, 0.0, rel_tol=0.0, abs_tol=1e-9)
                or not math.isclose(
                    candidate_offset,
                    0.0,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or check_by_name["zero_offset_identifiability"].get("status") != "ok"
            ):
                raise ValueError(
                    f"Zero-offset auto-sync evidence is invalid for {sensor_key}"
                )
        elif (
            checks
            or not math.isclose(operator_offset, 0.0, rel_tol=0.0, abs_tol=1e-9)
            or not math.isclose(candidate_offset, 0.0, rel_tol=0.0, abs_tol=1e-9)
        ):
            raise ValueError(
                f"Fixed-zero synchronization evidence is invalid for {sensor_key}"
            )
    _promotion_time_offset_artifact_bindings(
        attempt,
        request_value,
        sensors,
    )
    return sensors


def _promotion_time_offset_artifact_bindings(
    attempt: Mapping[str, Any],
    request_value: Mapping[str, Any],
    sensors: Mapping[str, Mapping[str, Any]],
) -> None:
    """Bind selected offsets to authoritative sync and solver observations."""

    run_root = Path(str(attempt["run_root"]))
    attempt_id = str(request_value["attempt_id"])
    attempt_root = calibration_attempt_root(run_root, attempt_id)
    source_reference = _attempt_artifact_reference(attempt_id, TIME_OFFSET_SEARCH)
    quality = _read_json(attempt_root / SYNC_QUALITY_REPORT)
    checks = quality.get("checks")
    if (
        quality.get("overall_status") == "error"
        or not isinstance(checks, list)
        or any(
            isinstance(item, Mapping) and item.get("status") == "error"
            for item in checks
        )
    ):
        raise ValueError("Authoritative synchronization quality is not promotable")
    policy = quality.get("calibration_attempt_policy")
    offsets = policy.get("per_sensor_offsets") if isinstance(policy, Mapping) else None
    if (
        not isinstance(policy, Mapping)
        or policy.get("synchronization_policy")
        != request_value["synchronization_policy"]
        or policy.get("time_offset_search") != source_reference
        or not isinstance(offsets, Mapping)
        or set(offsets) != set(sensors)
    ):
        raise ValueError("Authoritative synchronization provenance is inconsistent")

    sensor_metadata = {
        str(item["sensor_key"]): item
        for item in request_value.get("sensors", [])
        if isinstance(item, Mapping) and item.get("sensor_key")
    }
    summaries = {
        str(item.get("sensor_name")): item
        for item in quality.get("sensors", [])
        if isinstance(item, Mapping) and item.get("sensor_name")
    }
    for sensor_key, alignment in sensors.items():
        recorded = offsets.get(sensor_key)
        metadata = sensor_metadata.get(sensor_key)
        summary = (
            summaries.get(str(metadata.get("sensor_name")))
            if isinstance(metadata, Mapping)
            else None
        )
        if (
            not isinstance(recorded, Mapping)
            or recorded.get("status") != alignment.get("status")
            or not _optional_floats_match(
                recorded.get("robot_pose_time_offset_ms"),
                alignment.get("selected_robot_pose_time_offset_ms"),
            )
            or not _optional_floats_match(
                recorded.get("sync_delta_ms"),
                alignment.get("selected_sync_delta_ms"),
            )
            or not isinstance(summary, Mapping)
            or not _optional_floats_match(
                summary.get("sync_delta_ms"),
                alignment.get("selected_sync_delta_ms"),
            )
        ):
            raise ValueError(
                f"Authoritative synchronization offset is inconsistent for {sensor_key}"
            )

    observations = _read_json(attempt_root / CALIBRATION_OBSERVATIONS)
    if observations.get("time_offset_search") != source_reference:
        raise ValueError("Calibration observations reference invalid timing evidence")
    identity_to_sensor = {
        (str(item.get("sensor_type")), str(item.get("device_id"))): sensor_key
        for sensor_key, item in sensor_metadata.items()
    }
    observation_counts = {sensor_key: 0 for sensor_key in sensors}
    for observation in observations.get("observations", []):
        if not isinstance(observation, Mapping):
            continue
        sensor_key = identity_to_sensor.get(
            (
                str(observation.get("sensor_type")),
                str(observation.get("device_id")),
            )
        )
        if sensor_key not in sensors:
            continue
        alignment = sensors[sensor_key]
        timestamp_alignment = observation.get("timestamp_alignment")
        if (
            not isinstance(timestamp_alignment, Mapping)
            or timestamp_alignment.get("source") != source_reference
            or not _optional_floats_match(
                observation.get("robot_pose_time_offset_ms"),
                alignment.get("selected_robot_pose_time_offset_ms"),
            )
            or not _optional_floats_match(
                observation.get("sync_delta_ms"),
                alignment.get("selected_sync_delta_ms"),
            )
            or not _optional_floats_match(
                timestamp_alignment.get("robot_pose_time_offset_ms"),
                alignment.get("selected_robot_pose_time_offset_ms"),
            )
            or not _optional_floats_match(
                timestamp_alignment.get("sync_delta_ms"),
                alignment.get("selected_sync_delta_ms"),
            )
        ):
            raise ValueError(
                f"Calibration observation timing is inconsistent for {sensor_key}"
            )
        observation_counts[sensor_key] += 1
    missing = sorted(
        sensor_key for sensor_key, count in observation_counts.items() if count == 0
    )
    if missing:
        raise ValueError(
            "Calibration observation timing evidence is missing for: "
            + ", ".join(missing)
        )


def create_promotion_request(
    run_root: str | Path,
    attempt_id: str,
    *,
    selections: Mapping[str, Any] | None = None,
    operator: str | None = None,
) -> dict[str, Any]:
    root = Path(run_root)
    attempt = load_calibration_attempt(root, attempt_id)
    if attempt["progress"].get("status") != "complete":
        raise ValueError("Calibration attempt is not complete")
    prior_promotion = attempt.get("promotion")
    if (
        isinstance(prior_promotion, Mapping)
        and prior_promotion.get("status") != "failed"
    ):
        raise ValueError("Calibration attempt already has promotion evidence")
    _promotion_time_offset_evidence(attempt)
    selected = _promotion_selections(attempt, selections)
    joint_bundle = _revalidate_joint_promotion(attempt, selected)
    value = {
        "schema_version": PROMOTION_REQUEST_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "run_root": root.as_posix(),
        "created_at": utc_now_iso(),
        "operator": str(operator).strip() if operator else None,
        "selections": selected,
        "joint_bundle_id": (
            joint_bundle["bundle_id"] if joint_bundle is not None else None
        ),
        "previous_failure": (
            dict(prior_promotion) if isinstance(prior_promotion, Mapping) else None
        ),
    }
    attempt_root = calibration_attempt_root(root, attempt_id)
    atomic_write_json(attempt_root / PROMOTION_REQUEST_FILE, value)
    atomic_write_json(
        attempt_root / PROMOTION_FILE,
        {
            "schema_version": PROMOTION_SCHEMA_VERSION,
            "attempt_id": attempt_id,
            "status": "queued",
            "requested_at": value["created_at"],
            "selections": selected,
            "joint_bundle_id": value["joint_bundle_id"],
            "operator": value["operator"],
        },
    )
    return value


def _profile_slot(profile: CalibrationProfile) -> tuple[str, str]:
    return profile.sensor_type.value, profile.sensor_id


def _validate_promotion_request_identity(
    run_root: Path,
    attempt_id: str,
    promotion_request: Mapping[str, Any],
    promotion_status: Mapping[str, Any],
) -> None:
    if promotion_request.get("schema_version") != PROMOTION_REQUEST_SCHEMA_VERSION:
        raise ValueError("Unsupported calibration promotion request schema")
    if promotion_status.get("schema_version") != PROMOTION_SCHEMA_VERSION:
        raise ValueError("Unsupported calibration promotion status schema")
    if (
        promotion_request.get("attempt_id") != attempt_id
        or promotion_status.get("attempt_id") != attempt_id
    ):
        raise ValueError("Calibration promotion identity does not match its attempt")
    recorded_root = Path(str(promotion_request.get("run_root", ""))).resolve()
    if recorded_root != run_root.resolve():
        raise ValueError("Calibration promotion request belongs to a different run")
    request_selections = promotion_request.get("selections")
    status_selections = promotion_status.get("selections")
    if (
        not isinstance(request_selections, Mapping)
        or not request_selections
        or dict(request_selections) != status_selections
    ):
        raise ValueError(
            "Calibration promotion request/status selections are inconsistent"
        )
    if promotion_request.get("joint_bundle_id") != promotion_status.get(
        "joint_bundle_id"
    ):
        raise ValueError(
            "Calibration promotion request/status bundle identity is inconsistent"
        )


def _promotion_count(value: Any, *, label: str, candidate_id: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Candidate {candidate_id!r} has invalid {label} evidence")
    return value


def _promotion_ratio(value: Any, *, label: str, candidate_id: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Candidate {candidate_id!r} has invalid {label} evidence")
    try:
        ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Candidate {candidate_id!r} has invalid {label} evidence"
        ) from exc
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError(f"Candidate {candidate_id!r} has invalid {label} evidence")
    return ratio


def _promotion_outlier_evidence(
    candidate: Mapping[str, Any],
    profile: CalibrationProfile,
    *,
    candidate_id: str,
) -> tuple[float, float]:
    """Revalidate the exact full-input outlier policy used by ranking."""

    validation = candidate.get("full_input_validation")
    if not isinstance(validation, Mapping):
        raise ValueError(
            f"Candidate {candidate_id!r} lacks full-input outlier evidence"
        )
    per_motion = validation.get("per_motion")
    if not isinstance(per_motion, Mapping) or not per_motion:
        raise ValueError(
            f"Candidate {candidate_id!r} lacks per-motion outlier evidence"
        )

    total_observations = 0
    total_inliers = 0
    total_outliers = 0
    motion_ratios: list[float] = []
    repeated_motion_ratios: list[float] = []
    for pose_key, raw_motion in per_motion.items():
        if not isinstance(raw_motion, Mapping):
            raise ValueError(
                f"Candidate {candidate_id!r} has invalid motion {pose_key!r} evidence"
            )
        observation_count = _promotion_count(
            raw_motion.get("observation_count"),
            label=f"motion {pose_key!r} observation_count",
            candidate_id=candidate_id,
        )
        inlier_count = _promotion_count(
            raw_motion.get("inlier_count"),
            label=f"motion {pose_key!r} inlier_count",
            candidate_id=candidate_id,
        )
        outlier_count = _promotion_count(
            raw_motion.get("outlier_count"),
            label=f"motion {pose_key!r} outlier_count",
            candidate_id=candidate_id,
        )
        if observation_count <= 0 or inlier_count + outlier_count != observation_count:
            raise ValueError(
                f"Candidate {candidate_id!r} has inconsistent motion "
                f"{pose_key!r} counts"
            )
        ratio = _promotion_ratio(
            raw_motion.get("outlier_ratio"),
            label=f"motion {pose_key!r} outlier_ratio",
            candidate_id=candidate_id,
        )
        expected_ratio = outlier_count / observation_count
        if not math.isclose(ratio, expected_ratio, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"Candidate {candidate_id!r} has inconsistent motion "
                f"{pose_key!r} outlier ratio"
            )
        total_observations += observation_count
        total_inliers += inlier_count
        total_outliers += outlier_count
        motion_ratios.append(ratio)
        if observation_count >= 4:
            repeated_motion_ratios.append(ratio)

    balanced_ratio = sum(motion_ratios) / len(motion_ratios)
    repeated_motion_ratio = max(repeated_motion_ratios, default=0.0)
    recorded_balanced_ratio = _promotion_ratio(
        validation.get("motion_balanced_outlier_ratio"),
        label="motion_balanced_outlier_ratio",
        candidate_id=candidate_id,
    )
    recorded_repeated_motion_ratio = _promotion_ratio(
        validation.get("max_repeated_motion_outlier_ratio"),
        label="max_repeated_motion_outlier_ratio",
        candidate_id=candidate_id,
    )
    candidate_ratio = _promotion_ratio(
        candidate.get("outlier_ratio"),
        label="candidate outlier_ratio",
        candidate_id=candidate_id,
    )
    profile_ratio = _promotion_ratio(
        profile.metadata.get("outlier_ratio"),
        label="profile outlier_ratio",
        candidate_id=candidate_id,
    )
    ratios = (
        recorded_balanced_ratio,
        candidate_ratio,
        profile_ratio,
    )
    if any(
        not math.isclose(value, balanced_ratio, rel_tol=0.0, abs_tol=1e-12)
        for value in ratios
    ) or not math.isclose(
        recorded_repeated_motion_ratio,
        repeated_motion_ratio,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"Candidate {candidate_id!r} has inconsistent aggregate outlier evidence"
        )

    candidate_observations = _promotion_count(
        candidate.get("observation_count"),
        label="candidate observation_count",
        candidate_id=candidate_id,
    )
    candidate_inliers = _promotion_count(
        candidate.get("inlier_count"),
        label="candidate inlier_count",
        candidate_id=candidate_id,
    )
    candidate_outliers = _promotion_count(
        candidate.get("outlier_count"),
        label="candidate outlier_count",
        candidate_id=candidate_id,
    )
    profile_outliers = _promotion_count(
        profile.metadata.get("outlier_count"),
        label="profile outlier_count",
        candidate_id=candidate_id,
    )
    if (
        candidate_observations != total_observations
        or candidate_inliers != total_inliers
        or candidate_outliers != total_outliers
        or profile.quality.num_observations != total_observations
        or profile.quality.num_inliers != total_inliers
        or profile_outliers != total_outliers
    ):
        raise ValueError(
            f"Candidate {candidate_id!r} has inconsistent full-input outlier counts"
        )
    raw_ratio = _promotion_ratio(
        candidate.get("raw_outlier_ratio"),
        label="raw_outlier_ratio",
        candidate_id=candidate_id,
    )
    if not math.isclose(
        raw_ratio,
        total_outliers / total_observations,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"Candidate {candidate_id!r} has inconsistent raw outlier evidence"
        )

    checks = candidate.get("checks")
    if not isinstance(checks, list):
        raise ValueError(f"Candidate {candidate_id!r} lacks validation checks")
    expected_checks = {
        "outlier_ratio": balanced_ratio,
        "full_input_repeated_motion_outlier_ratio": repeated_motion_ratio,
    }
    for check_name, expected_actual in expected_checks.items():
        matches = [
            check
            for check in checks
            if isinstance(check, Mapping) and check.get("name") == check_name
        ]
        if len(matches) != 1 or matches[0].get("status") != "ok":
            raise ValueError(
                f"Candidate {candidate_id!r} lacks passing {check_name} evidence"
            )
        actual = _promotion_ratio(
            matches[0].get("actual"),
            label=f"{check_name} check actual",
            candidate_id=candidate_id,
        )
        threshold = _promotion_ratio(
            matches[0].get("threshold"),
            label=f"{check_name} check threshold",
            candidate_id=candidate_id,
        )
        if not math.isclose(
            actual, expected_actual, rel_tol=0.0, abs_tol=1e-12
        ) or not math.isclose(
            threshold,
            DEFAULT_MAX_OUTLIER_RATIO,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Candidate {candidate_id!r} has inconsistent {check_name} check"
            )
    return balanced_ratio, repeated_motion_ratio


def _promotion_transform_records_match(
    ranked: Any,
    profiled: Any,
    *,
    label: str,
    candidate_id: str,
) -> None:
    """Require the profile to preserve the exact ranked transform evidence."""

    if not isinstance(ranked, Mapping) or not isinstance(profiled, Mapping):
        raise ValueError(f"Candidate {candidate_id!r} lacks {label} transform evidence")
    ranked_frames = (str(ranked.get("from")), str(ranked.get("to")))
    profiled_frames = (str(profiled.get("from")), str(profiled.get("to")))
    if ranked_frames != profiled_frames:
        raise ValueError(
            f"Candidate profile {candidate_id!r} has inconsistent {label} frames"
        )
    try:
        residual = transform_residual(
            transform_from_record(ranked),
            transform_from_record(profiled),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Candidate {candidate_id!r} has invalid {label} transform evidence"
        ) from exc
    if (
        residual["translation_mm"] > PROMOTION_TRANSFORM_TOLERANCE_MM
        or residual["rotation_deg"] > PROMOTION_TRANSFORM_TOLERANCE_DEG
    ):
        raise ValueError(
            f"Candidate profile {candidate_id!r} does not match ranked {label} "
            "transform evidence"
        )


def _promotion_transform_evidence(
    candidate: Mapping[str, Any],
    profile: CalibrationProfile,
    *,
    candidate_id: str,
) -> None:
    primary = {
        "from": profile.extrinsics.from_frame.value,
        "to": profile.extrinsics.to_frame.value,
        "rotation_quaternion_wxyz": list(profile.extrinsics.rotation_quaternion_wxyz),
        "translation_mm": list(profile.extrinsics.translation_mm),
    }
    _promotion_transform_records_match(
        candidate.get("primary_transform"),
        primary,
        label="primary",
        candidate_id=candidate_id,
    )
    _promotion_transform_records_match(
        candidate.get("companion_transform"),
        profile.metadata.get("companion_transform"),
        label="companion",
        candidate_id=candidate_id,
    )


def _selected_profiles(
    attempt_root: Path,
    attempt: Mapping[str, Any],
    request_value: Mapping[str, Any],
    promotion_request: Mapping[str, Any],
) -> list[CalibrationProfile]:
    time_offset_by_sensor = _promotion_time_offset_evidence(attempt)
    time_offset_source = _attempt_artifact_reference(
        str(request_value["attempt_id"]),
        TIME_OFFSET_SEARCH,
    )
    joint_bundle = _revalidate_joint_promotion(
        attempt,
        {
            str(sensor_key): str(candidate_id)
            for sensor_key, candidate_id in promotion_request["selections"].items()
        },
        expected_bundle_id=(
            str(promotion_request["joint_bundle_id"])
            if promotion_request.get("joint_bundle_id") is not None
            else None
        ),
    )
    if joint_bundle is not None and promotion_request.get("joint_bundle_id") is None:
        raise ValueError("Promotion request lacks its multi-camera bundle identity")
    profiles = load_profile_collection(attempt_root / CANDIDATE_PROFILES_FILE)
    by_candidate = {
        str(profile.metadata.get("candidate_id")): profile for profile in profiles
    }
    timestamp = utc_now_iso()
    selected = []
    for sensor_key, candidate_id in promotion_request["selections"].items():
        profile = by_candidate.get(str(candidate_id))
        if profile is None:
            raise ValueError(f"Candidate profile not found: {candidate_id}")
        if profile.metadata.get("sensor_key") != sensor_key:
            raise ValueError(
                f"Candidate profile {candidate_id!r} does not belong to {sensor_key}"
            )
        result = next(
            item
            for item in attempt["results"]["results"]
            if item["sensor_key"] == sensor_key
        )
        candidate = next(
            item
            for item in result["candidates"]
            if item["candidate_id"] == candidate_id
        )
        if candidate["status"] != "passing":
            raise ValueError(f"Candidate no longer passes validation: {candidate_id}")
        alignment = time_offset_by_sensor.get(str(sensor_key))
        expected_sync_delta_ms = (
            float(alignment["selected_sync_delta_ms"])
            if alignment is not None
            else None
        )
        candidate_synchronization = candidate.get("synchronization")
        profile_synchronization = profile.metadata.get("synchronization")
        if alignment is not None and (
            not isinstance(candidate_synchronization, Mapping)
            or not isinstance(profile_synchronization, Mapping)
            or candidate_synchronization.get("source") != time_offset_source
            or profile_synchronization.get("source") != time_offset_source
            or candidate_synchronization.get("policy")
            != request_value["synchronization_policy"]
            or profile_synchronization.get("policy")
            != request_value["synchronization_policy"]
            or candidate_synchronization.get("status") != alignment.get("status")
            or profile_synchronization.get("status") != alignment.get("status")
            or not _optional_floats_match(
                candidate_synchronization.get("robot_pose_time_offset_ms"),
                alignment.get("selected_robot_pose_time_offset_ms"),
            )
            or not _optional_floats_match(
                profile_synchronization.get("robot_pose_time_offset_ms"),
                alignment.get("selected_robot_pose_time_offset_ms"),
            )
            or not _optional_floats_match(
                candidate_synchronization.get("sync_delta_ms"),
                expected_sync_delta_ms,
            )
            or not _optional_floats_match(
                profile_synchronization.get("sync_delta_ms"),
                expected_sync_delta_ms,
            )
        ):
            raise ValueError(
                f"Candidate {candidate_id!r} has inconsistent auto-sync provenance"
            )
        if alignment is not None and not _optional_floats_match(
            profile.sync_delta_ms, expected_sync_delta_ms
        ):
            raise ValueError(
                f"Candidate {candidate_id!r} profile sync delta is inconsistent"
            )
        _promotion_transform_evidence(
            candidate,
            profile,
            candidate_id=candidate_id,
        )
        inlier_count = profile.quality.num_inliers
        outlier_ratio, repeated_motion_outlier_ratio = _promotion_outlier_evidence(
            candidate,
            profile,
            candidate_id=candidate_id,
        )
        if (
            inlier_count < DEFAULT_MIN_INLIERS
            or profile.quality.residual_translation_mm is None
            or profile.quality.residual_translation_mm > DEFAULT_MAX_MEAN_TRANSLATION_MM
            or profile.quality.residual_rotation_deg is None
            or profile.quality.residual_rotation_deg > DEFAULT_MAX_MEAN_ROTATION_DEG
            or outlier_ratio > DEFAULT_MAX_OUTLIER_RATIO
            or repeated_motion_outlier_ratio > DEFAULT_MAX_OUTLIER_RATIO
        ):
            raise ValueError(
                f"Candidate no longer satisfies promotion gates: {candidate_id}"
            )
        metadata = dict(profile.metadata)
        metadata.update(
            {
                "promotion_attempt_id": request_value["attempt_id"],
                "promotion_candidate_id": candidate_id,
                "promotion_solver_provenance": {
                    "solver_policy": request_value["solver_policy"],
                    "pnp_method": candidate["pnp_method"],
                    "extrinsic_method": candidate["extrinsic_method"],
                },
                "promotion_synchronization_provenance": (
                    {
                        "source": time_offset_source,
                        "status": alignment["status"],
                        "robot_pose_time_offset_ms": alignment[
                            "selected_robot_pose_time_offset_ms"
                        ],
                        "sync_delta_ms": alignment["selected_sync_delta_ms"],
                    }
                    if alignment is not None
                    else {
                        "source": "historical_fixed_zero",
                        "sync_delta_ms": profile.sync_delta_ms,
                    }
                ),
                "promotion_multi_camera_bundle_id": (
                    joint_bundle["bundle_id"] if joint_bundle is not None else None
                ),
                "promoted_at": timestamp,
                "promoted_by": promotion_request.get("operator"),
            }
        )
        selected.append(
            replace(
                profile,
                status=CalibrationStatus.VALID,
                calibrated_at=timestamp,
                operator=promotion_request.get("operator") or profile.operator,
                metadata=metadata,
            )
        )
    return selected


def _canonical_reports(
    attempt_root: Path,
    attempt: Mapping[str, Any],
    selected_profiles: Sequence[CalibrationProfile],
    promotion_request: Mapping[str, Any],
    *,
    canonical_profile_count: int,
) -> dict[str, dict[str, Any]]:
    extrinsic = _read_json(attempt_root / EXTRINSIC_CANDIDATES_FILE)
    ranking = attempt["results"]
    all_profile_values = [profile_to_dict(profile) for profile in selected_profiles]
    selected_ids = {profile.profile_id for profile in selected_profiles}
    selected_candidate_ids = set(promotion_request["selections"].values())
    selected_candidates = [
        item
        for item in extrinsic["candidates"]
        if item.get("candidate_id") in selected_candidate_ids
    ]
    selected_results = [
        {
            **dict(result),
            "candidates": [
                item
                for item in result.get("candidates", [])
                if item.get("candidate_id") in selected_candidate_ids
            ],
        }
        for result in ranking["results"]
        if result.get("sensor_key") in promotion_request["selections"]
    ]
    time_offset_search = attempt.get("time_offset_search")
    synchronization_summary = (
        {
            "policy": time_offset_search.get("policy"),
            "source": (
                f"processed/calibration/{attempt['attempt_id']}/{TIME_OFFSET_SEARCH}"
            ),
            "sensors": [
                {
                    "sensor_key": item.get("sensor_key"),
                    "status": item.get("status"),
                    "robot_pose_time_offset_ms": item.get(
                        "selected_robot_pose_time_offset_ms"
                    ),
                    "sync_delta_ms": item.get("selected_sync_delta_ms"),
                }
                for item in time_offset_search.get("sensors", [])
                if isinstance(item, Mapping)
                and item.get("sensor_key") in promotion_request["selections"]
            ],
        }
        if isinstance(time_offset_search, Mapping)
        else {
            "policy": "fixed_zero",
            "source": "historical_attempt_without_time_offset_search",
            "sensors": [],
        }
    )
    selected_checks = [
        item
        for item in _read_json(attempt_root / CHECKS_FILE)["checks"]
        if item.get("candidate_id") in selected_candidate_ids
        or (
            promotion_request.get("joint_bundle_id") is not None
            and item.get("bundle_id") == promotion_request.get("joint_bundle_id")
        )
    ]
    candidate_report = {
        "schema_version": "calibration_candidates.v1",
        "generated_at": utc_now_iso(),
        "run_root": attempt["run_root"],
        "attempt_id": attempt["attempt_id"],
        "overall_status": "ok",
        "candidate_count": len(selected_candidates),
        "inlier_count": sum(
            profile.quality.num_inliers for profile in selected_profiles
        ),
        "outlier_count": sum(
            profile.quality.num_observations - profile.quality.num_inliers
            for profile in selected_profiles
        ),
        "profiles": all_profile_values,
        "candidates": selected_candidates,
        "comparisons": selected_results,
        "synchronization": synchronization_summary,
        "checks": selected_checks,
    }
    multi_camera_consistency = ranking.get("multi_camera_consistency")
    if isinstance(multi_camera_consistency, Mapping):
        selected_bundle_id = promotion_request.get("joint_bundle_id")
        selected_bundle = next(
            (
                dict(bundle)
                for bundle in multi_camera_consistency.get("bundles", [])
                if isinstance(bundle, Mapping)
                and bundle.get("bundle_id") == selected_bundle_id
            ),
            None,
        )
        candidate_report["multi_camera_consistency"] = {
            **dict(multi_camera_consistency),
            "bundles": [selected_bundle] if selected_bundle is not None else [],
            "recommendation": selected_bundle,
        }
    solver_report = {
        "schema_version": "calibration_solver.v2",
        "generated_at": utc_now_iso(),
        "run_root": attempt["run_root"],
        "attempt_id": attempt["attempt_id"],
        "overall_status": "ok",
        "mode": attempt["request"]["mode"],
        "profile_count": len(all_profile_values),
        "candidate_count": len(selected_candidates),
        "profiles": all_profile_values,
        "solutions": selected_candidates,
        "comparisons": selected_results,
        "synchronization": synchronization_summary,
        "checks": candidate_report["checks"],
    }
    if "multi_camera_consistency" in candidate_report:
        solver_report["multi_camera_consistency"] = candidate_report[
            "multi_camera_consistency"
        ]
    validation_report = {
        "schema_version": "calibration_validation.v1",
        "generated_at": utc_now_iso(),
        "run_root": attempt["run_root"],
        "attempt_id": attempt["attempt_id"],
        "overall_status": "ok",
        "profile_count": len(selected_profiles),
        "promotable_profile_count": len(selected_profiles),
        "selection": {
            "requested": dict(promotion_request["selections"]),
            "selected_profile_ids": sorted(selected_ids),
            "explicit_selection_required": True,
            "joint_bundle_id": promotion_request.get("joint_bundle_id"),
        },
        "synchronization": synchronization_summary,
        "promotion": {
            "requested": True,
            "promoted": True,
            "path": CALIBRATION_PROFILES,
            "profile_count": canonical_profile_count,
            "promoted_profile_ids": sorted(selected_ids),
        },
        "profiles": [
            {
                "profile_id": profile.profile_id,
                "sensor_id": profile.sensor_id,
                "sensor_type": profile.sensor_type.value,
                "mounting_mode": profile.mounting_mode.value,
                "validation_status": "ok",
                "selected": True,
                "promotable": True,
                "num_observations": profile.quality.num_observations,
                "num_inliers": profile.quality.num_inliers,
                "residual_translation_mm": profile.quality.residual_translation_mm,
                "residual_rotation_deg": profile.quality.residual_rotation_deg,
            }
            for profile in selected_profiles
        ],
        "checks": [],
    }
    return {
        CALIBRATION_CANDIDATES: candidate_report,
        CALIBRATION_SOLVER_REPORT: solver_report,
        CALIBRATION_VALIDATION_REPORT: validation_report,
    }


def _transactional_replace(
    run_root: Path,
    promotions: Sequence[tuple[Path, Path]],
) -> None:
    backup_root = run_root / f".calibration-promotion-backup-{uuid.uuid4().hex}"
    backup_root.mkdir(parents=False, exist_ok=False)
    installed: list[Path] = []
    backups: list[tuple[Path, Path]] = []
    try:
        for index, (source, destination) in enumerate(promotions):
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                backup = backup_root / str(index)
                os.replace(destination, backup)
                backups.append((backup, destination))
            os.replace(source, destination)
            installed.append(destination)
    except Exception:
        for destination in reversed(installed):
            if destination.is_dir():
                shutil.rmtree(destination)
            elif destination.exists():
                destination.unlink()
        for backup, destination in reversed(backups):
            os.replace(backup, destination)
        raise
    finally:
        if backup_root.exists():
            shutil.rmtree(backup_root)


def promote_calibration_attempt(
    run_root: str | Path, attempt_id: str
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    with run_config_lock(root):
        return _promote_calibration_attempt_locked(root, attempt_id)


def _promote_calibration_attempt_locked(
    run_root: str | Path, attempt_id: str
) -> dict[str, Any]:
    root = Path(run_root)
    attempt_root = calibration_attempt_root(root, attempt_id)
    attempt = load_calibration_attempt(root, attempt_id)
    request_value = attempt["request"]
    promotion_request = _read_json(attempt_root / PROMOTION_REQUEST_FILE)
    promotion_path = attempt_root / PROMOTION_FILE
    current = _read_json(promotion_path)
    _validate_promotion_request_identity(
        root,
        attempt_id,
        promotion_request,
        current,
    )
    current.update({"status": "running", "started_at": utc_now_iso()})
    atomic_write_json(promotion_path, current)
    staging = root / f".calibration-promotion-{attempt_id}-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        current_config = load_run_config_for_run_root(root)
        current_target = current_config.get("calibration_target")
        current_target_id = (
            current_target.get("target_id")
            if isinstance(current_target, Mapping)
            else None
        )
        if (
            current_target_id is not None
            and current_target_id != request_value["target_id"]
        ):
            blockers = [
                item
                for item in replacement_blockers(root)
                if not item.startswith(f"{ATTEMPT_DIRECTORY.as_posix()}/")
            ]
            if blockers:
                raise CalibrationTargetConflict(
                    "Canonical target-dependent artifacts changed after this attempt was created.",
                    blockers=blockers,
                )
        selected_profiles = _selected_profiles(
            attempt_root, attempt, request_value, promotion_request
        )
        existing_path = root / CALIBRATION_PROFILES
        existing = (
            load_profile_collection(existing_path) if existing_path.is_file() else []
        )
        promoted_slots = {_profile_slot(profile) for profile in selected_profiles}
        preserved = [
            profile
            for profile in existing
            if _profile_slot(profile) not in promoted_slots
        ]
        merged = [*preserved, *selected_profiles]
        write_profile_collection(merged, staging / CALIBRATION_PROFILES)
        write_profile_collection(
            list(selected_profiles),
            staging / CALIBRATION_PROFILES_FROM_OBSERVATIONS,
        )
        write_profile_collection(
            list(selected_profiles),
            staging / CALIBRATION_PROFILES_SOLVED,
        )
        selected_sensor_keys = set(promotion_request["selections"])
        selected_candidate_ids = set(promotion_request["selections"].values())
        selected_pnp_methods = {}
        for result in attempt["results"]["results"]:
            sensor_key = result.get("sensor_key")
            if sensor_key not in selected_sensor_keys:
                continue
            selected_candidate_id = promotion_request["selections"][sensor_key]
            selected_candidate = next(
                item
                for item in result["candidates"]
                if item["candidate_id"] == selected_candidate_id
            )
            selected_pnp_methods[sensor_key] = selected_candidate["pnp_method"]
        observations_report = _read_json(attempt_root / CALIBRATION_OBSERVATIONS)
        selected_observations = []
        for item in observations_report.get("observations", []):
            if not isinstance(item, Mapping):
                continue
            sensor_key = _sensor_key(
                str(item.get("sensor_type")), str(item.get("device_id"))
            )
            if sensor_key in selected_sensor_keys and item.get(
                "pnp_method"
            ) == selected_pnp_methods.get(sensor_key):
                selected_observations.append(dict(item))
        observations_report["observations"] = selected_observations
        observations_report["observation_count"] = len(selected_observations)
        observations_report["sensors"] = [
            item
            for item in observations_report.get("sensors", [])
            if item.get("sensor_key") in selected_sensor_keys
        ]
        observations_report["sensor_count"] = len(observations_report["sensors"])
        observations_report["promoted_candidate_ids"] = sorted(selected_candidate_ids)
        atomic_write_json(staging / CALIBRATION_OBSERVATIONS, observations_report)
        attempt_intrinsics = load_intrinsic_profile_collection(
            attempt_root / INTRINSIC_CALIBRATION_PROFILES
        )
        existing_intrinsics_path = root / INTRINSIC_CALIBRATION_PROFILES
        existing_intrinsics = (
            load_intrinsic_profile_collection(existing_intrinsics_path)
            if existing_intrinsics_path.is_file()
            else []
        )
        selected_sensor_ids = {profile.sensor_id for profile in selected_profiles}
        promoted_intrinsics = [
            item
            for item in attempt_intrinsics
            if str(item.get("sensor_id")) in selected_sensor_ids
        ]
        promoted_intrinsic_keys = {
            (
                str(item["sensor_id"]),
                tuple(item["resolution"]),
                str(item["orientation"]),
            )
            for item in promoted_intrinsics
        }
        preserved_intrinsics = [
            item
            for item in existing_intrinsics
            if (
                str(item["sensor_id"]),
                tuple(item["resolution"]),
                str(item["orientation"]),
            )
            not in promoted_intrinsic_keys
        ]
        write_intrinsic_profile_collection(
            [*preserved_intrinsics, *promoted_intrinsics],
            staging / INTRINSIC_CALIBRATION_PROFILES,
        )
        reports = _canonical_reports(
            attempt_root,
            attempt,
            selected_profiles,
            promotion_request,
            canonical_profile_count=len(merged),
        )
        for filename, report in reports.items():
            atomic_write_json(staging / filename, report)
        target = dict(request_value["target"])
        target.pop("placement", None)
        atomic_write_json(staging / CALIBRATION_TARGET, target)
        updated_config = dict(current_config)
        updated_config["calibration_target"] = request_value["target_bundle"][
            "selection"
        ]
        capture = dict(updated_config["capture"])
        sensors = []
        selected_by_identity = {
            (profile.sensor_type.value, profile.sensor_id): profile
            for profile in selected_profiles
        }
        for raw_sensor in capture["sensors"]:
            sensor = dict(raw_sensor)
            profile = selected_by_identity.get(
                (str(sensor.get("sensor_type")), str(sensor.get("device_id")))
            )
            if profile is not None:
                sensor["mounting_mode"] = profile.mounting_mode.value
                sensor["calibration_profile_id"] = profile.profile_id
            sensors.append(sensor)
        capture["sensors"] = sensors
        updated_config["capture"] = capture
        updated_config["calibration_profiles"] = CALIBRATION_PROFILES
        validate_run_config(updated_config)
        atomic_write_json(staging / RUN_CONFIG, updated_config)
        manifest = load_or_create_run_manifest(root)
        canonical_artifacts = {
            filename: root / filename
            for filename in (
                CALIBRATION_TARGET,
                INTRINSIC_CALIBRATION_PROFILES,
                CALIBRATION_OBSERVATIONS,
                CALIBRATION_CANDIDATES,
                CALIBRATION_PROFILES_FROM_OBSERVATIONS,
                CALIBRATION_SOLVER_REPORT,
                CALIBRATION_PROFILES_SOLVED,
                CALIBRATION_VALIDATION_REPORT,
                CALIBRATION_PROFILES,
                RUN_CONFIG,
            )
        }
        upsert_stage(
            manifest,
            name="calibration_attempt_promotion",
            status="succeeded",
            artifacts=canonical_artifacts,
            run_root=root,
            message=f"Promoted calibration attempt {attempt_id}.",
        )
        atomic_write_json(staging / DATASET_MANIFEST, manifest)
        bundle_stage = staging / TARGET_BUNDLE_DIRECTORY
        shutil.copytree(attempt_root / TARGET_BUNDLE_DIRECTORY, bundle_stage)
        promotions = [
            (staging / filename, root / filename)
            for filename in (
                CALIBRATION_TARGET,
                INTRINSIC_CALIBRATION_PROFILES,
                CALIBRATION_OBSERVATIONS,
                CALIBRATION_CANDIDATES,
                CALIBRATION_PROFILES_FROM_OBSERVATIONS,
                CALIBRATION_SOLVER_REPORT,
                CALIBRATION_PROFILES_SOLVED,
                CALIBRATION_VALIDATION_REPORT,
                CALIBRATION_PROFILES,
                RUN_CONFIG,
                DATASET_MANIFEST,
            )
        ]
        promotions.append(
            (
                bundle_stage,
                root / LIBRARY_DIRECTORY / str(request_value["target_id"]),
            )
        )
        _transactional_replace(root, promotions)
        promoted = {
            "schema_version": PROMOTION_SCHEMA_VERSION,
            "attempt_id": attempt_id,
            "status": "promoted",
            "requested_at": promotion_request["created_at"],
            "promoted_at": utc_now_iso(),
            "operator": promotion_request.get("operator"),
            "selections": dict(promotion_request["selections"]),
            "joint_bundle_id": promotion_request.get("joint_bundle_id"),
            "promoted_profile_ids": [
                profile.profile_id for profile in selected_profiles
            ],
            "preserved_profile_ids": [profile.profile_id for profile in preserved],
            "canonical_artifacts": sorted(canonical_artifacts),
        }
        atomic_write_json(promotion_path, promoted)
        return promoted
    except Exception as exc:
        failed = _read_json(promotion_path)
        failed.update(
            {
                "status": "failed",
                "ended_at": utc_now_iso(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_write_json(promotion_path, failed)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
