"""Intent-level, immutable calibration attempts and explicit promotion."""

from __future__ import annotations

import json
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
    DEFAULT_MIN_INLIERS,
    EXTRINSIC_METHOD_ORDER,
    PNP_METHOD_ORDER,
    evaluate_extrinsic_candidate,
    rank_candidates,
    solve_planar_pnp_candidates,
    transform_from_record,
)
from posetestbot.calibration.intrinsics import (
    factory_intrinsic_profile,
    load_intrinsic_profile_collection,
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
    validate_target_bundle,
)
from posetestbot.calibration.targets import (
    normalize_calibration_target_spec,
    opencv_grid_board,
    target_identity,
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
    INTRINSIC_CALIBRATION_PROFILES,
    MATCH_ROBOT_EE_POSES,
    RAW_ROBOT_EE_POSES,
    RGB_DIR,
    RUN_CONFIG,
    SYNC_QUALITY_REPORT,
)
from posetestbot.io.manifest import (
    discover_sensor_records,
    load_or_create_run_manifest,
    upsert_stage,
)
from posetestbot.pipeline.run_config import (
    load_run_config_for_run_root,
    validate_run_config,
)
from posetestbot.sensors.contracts import CameraIntrinsics, MountingMode, SensorType
from posetestbot.sync.non_destructive import synchronize_run
from posetestbot.sync.quality import build_sync_quality_report


ATTEMPT_SCHEMA_VERSION = "calibration_attempt.v1"
REQUEST_SCHEMA_VERSION = "calibration_attempt_request.v1"
PROGRESS_SCHEMA_VERSION = "calibration_attempt_progress.v1"
PROMOTION_SCHEMA_VERSION = "calibration_attempt_promotion.v1"
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
PHASES = (
    ("prepare_data", "Prepare data"),
    ("estimate_target_poses", "Estimate target poses"),
    ("compare_robot_camera_solutions", "Compare robot-camera solutions"),
    ("validate_and_rank", "Validate and rank"),
)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


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


def validate_attempt_id(attempt_id: str) -> str:
    value = str(attempt_id).strip().lower()
    if not ATTEMPT_ID_PATTERN.fullmatch(value):
        raise ValueError("attempt_id must contain 32 lowercase hexadecimal characters")
    return value


def calibration_attempt_root(run_root: str | Path, attempt_id: str) -> Path:
    return Path(run_root) / ATTEMPT_DIRECTORY / validate_attempt_id(attempt_id)


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
        if matching_config is None:
            same_family = [
                item
                for item in configured
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
                (
                    not _is_contained(folder_path / FRAME_METADATA_JSONL, root)
                    or not (folder_path / FRAME_METADATA_JSONL).is_file()
                )
                and not rgb.is_dir()
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
                "current_mounting_mode": (matching_config or {}).get(
                    "mounting_mode"
                ),
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
        ranking = _read_json(child / RANKING_FILE) if (child / RANKING_FILE).is_file() else None
        promotion = _read_json(child / PROMOTION_FILE) if (child / PROMOTION_FILE).is_file() else None
        records.append(
            {
                "attempt_id": child.name,
                "created_at": request_value.get("created_at"),
                "mode": request_value.get("mode"),
                "sensor_keys": request_value.get("sensor_keys", []),
                "target_id": request_value.get("target_id"),
                "status": progress.get("status"),
                "recommended_camera_count": (
                    int(ranking.get("recommended_camera_count", 0))
                    if ranking
                    else 0
                ),
                "promotion": promotion,
            }
        )
    return sorted(records, key=lambda item: str(item.get("created_at", "")), reverse=True)


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
            "intrinsics_policy": "reuse_compatible_or_factory",
            "thresholds": {
                "min_inliers": 6,
                "max_mean_translation_mm": 10.0,
                "max_mean_rotation_deg": 5.0,
                "max_outlier_ratio": 0.25,
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
    config = load_run_config_for_run_root(root)
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
    target_id = str(value.get("target_id", ""))
    bundle = validate_target_bundle(
        default_target_library_root() / target_id,
        library_root=default_target_library_root(),
    )
    active = config.get("calibration_target")
    active_id = active.get("target_id") if isinstance(active, Mapping) else None
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
    solver_policy = str(value.get("solver_policy", "auto_compare"))
    if solver_policy != "auto_compare":
        raise ValueError("solver_policy must be auto_compare")
    intrinsics_policy = str(
        value.get("intrinsics_policy", "reuse_compatible_or_factory")
    )
    if intrinsics_policy != "reuse_compatible_or_factory":
        raise ValueError(
            "intrinsics_policy must be reuse_compatible_or_factory"
        )
    pnp_methods = value.get("pnp_methods", list(PNP_METHOD_ORDER))
    extrinsic_methods = value.get(
        "extrinsic_methods", list(EXTRINSIC_METHOD_ORDER)
    )
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
    unsupported_extrinsic = sorted(
        set(extrinsic_methods) - set(EXTRINSIC_METHOD_ORDER)
    )
    if unsupported_pnp:
        raise ValueError(
            "Unsupported board-level PnP method(s): " + ", ".join(unsupported_pnp)
        )
    if unsupported_extrinsic:
        raise ValueError(
            "Unsupported extrinsic method(s): "
            + ", ".join(unsupported_extrinsic)
        )
    return {
        "mode": mode,
        "sensor_keys": sensor_keys,
        "sensors": [cameras[key] for key in sensor_keys],
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
    if request_value.get("attempt_id") != attempt_id or progress.get(
        "attempt_id"
    ) != attempt_id:
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


def _intrinsics_for_sensors(
    run_root: Path,
    synchronized: Mapping[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    existing_path = run_root / INTRINSIC_CALIBRATION_PROFILES
    existing = (
        load_intrinsic_profile_collection(existing_path)
        if existing_path.is_file()
        else []
    )
    profiles = []
    by_sensor = {}
    for sensor_key, folder in synchronized.items():
        sensor_id, orientation, resolution = sensor_intrinsic_identity(folder)
        try:
            selected = select_intrinsic_profile(
                existing,
                sensor_id=sensor_id,
                orientation=orientation,
                resolution=resolution,
            )
            selected = {**selected, "attempt_intrinsics_source": "compatible_existing"}
        except ValueError:
            selected = {
                **factory_intrinsic_profile(folder),
                "attempt_intrinsics_source": "factory_capture_sidecars",
            }
        profiles.append(selected)
        by_sensor[sensor_key] = selected
    return profiles, by_sensor


def _prepare_attempt_data(
    run_root: Path,
    attempt_root: Path,
    request_value: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    output_root = attempt_root / "processed" / "synchronized"
    synchronized: dict[str, Path] = {}
    sync_reports = []
    selected_by_path = {
        (run_root / str(sensor["folder"])).resolve(): str(sensor["sensor_key"])
        for sensor in request_value["sensors"]
    }
    results = synchronize_run(
        run_root,
        sensor_folders=list(selected_by_path),
        output_root=output_root,
    )
    for result in results:
        sensor_key = selected_by_path[Path(result.sensor_folder).resolve()]
        synchronized[sensor_key] = Path(result.output_folder)
        sync_reports.append(Path(result.report_path))
    sync_quality = build_sync_quality_report(
        run_root,
        report_paths=sync_reports,
    )
    atomic_write_json(attempt_root / SYNC_QUALITY_REPORT, sync_quality)
    if sync_quality["overall_status"] == "error":
        raise ValueError("Selected-camera synchronization quality failed")
    profiles, by_sensor = _intrinsics_for_sensors(run_root, synchronized)
    write_intrinsic_profile_collection(
        profiles,
        attempt_root / INTRINSIC_CALIBRATION_PROFILES,
    )
    return synchronized, by_sensor


def _projection(profile: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    native = profile.get("native")
    if not isinstance(native, Mapping):
        raise ValueError("Intrinsic profile has no native projection")
    return (
        np.asarray(native["cam_K"], dtype=float).reshape(3, 3),
        np.asarray(native["distortion"], dtype=float).reshape(-1),
    )


def _pose_vectors(transform_value: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    transform = transform_from_record(transform_value)
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    return (
        np.asarray(rvec, dtype=float).reshape(3).tolist(),
        np.asarray(transform[:3, 3], dtype=float).reshape(3).tolist(),
    )


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
        detections = detect_sensor_folder(
            folder,
            target,
            output_path=folder / ARUCO_DETECTIONS,
        )
        matrix, distortion = _projection(intrinsics[sensor_key])
        matched = _read_json(folder / MATCH_ROBOT_EE_POSES)
        frames = []
        method_observations = {
            method: [] for method in request_value["pnp_methods"]
        }
        compatibility_output: dict[str, Any] = {}
        for frame_id, detection in sorted(detections.get("frames", {}).items()):
            frame_record: dict[str, Any] = {
                "frame_id": frame_id,
                "marker_count": int(detection.get("marker_count", 0)),
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
                solved = solve_planar_pnp_candidates(
                    points[0],
                    points[1],
                    matrix,
                    distortion,
                    methods=request_value["pnp_methods"],
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
                    "candidates": solved["candidates"],
                    "failures": solved["failures"],
                }
            )
            for method, selected in solved["selected"].items():
                method_observations[method].append(
                    {
                        "observation_id": f"{sensor_key}:{method}:{frame_id}",
                        "frame_id": frame_id,
                        "motion": matched_pose.get("motion"),
                        "robot_ee_pose": dict(matched_pose["robot_ee_pose"]),
                        "target_to_camera": selected["transform"],
                        "mean_reprojection_error_px": selected[
                            "mean_reprojection_error_px"
                        ],
                        "pnp_common_inlier_count": solved["common_inlier_count"],
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
                        "mean_reprojection_error_px": preferred[
                            "mean_reprojection_error_px"
                        ],
                        "max_reprojection_error_px": preferred[
                            "max_reprojection_error_px"
                        ],
                        "target": target_identity(target),
                    },
                }
            frames.append(frame_record)
        atomic_write_json(folder / ARUCO_POSE_ESTIMATION, compatibility_output)
        observations[sensor_key] = method_observations
        evidence["sensors"].append(
            {
                **sensor_metadata[sensor_key],
                "frame_count": len(frames),
                "solved_frame_count": sum(
                    1 for item in frames if item["status"] == "ok"
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


def _compare_solutions(
    attempt_root: Path,
    request_value: Mapping[str, Any],
    observations: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> list[dict[str, Any]]:
    candidates = []
    for sensor_key in request_value["sensor_keys"]:
        for pnp_method in request_value["pnp_methods"]:
            method_observations = observations[sensor_key][pnp_method]
            for extrinsic_method in request_value["extrinsic_methods"]:
                candidates.append(
                    evaluate_extrinsic_candidate(
                        method_observations,
                        mode=request_value["mode"],
                        pnp_method=pnp_method,
                        extrinsic_method=extrinsic_method,
                        sensor_key=sensor_key,
                    )
                )
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
    mounting = MountingMode.EYE_IN_HAND if mode == "eye_in_hand" else MountingMode.STATIC
    safe_sensor = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sensor["device_id"]))
    safe_method = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        f"{candidate['pnp_method']}_{candidate['extrinsic_method']}",
    )
    profile_id = f"{safe_sensor}_{mode}_{safe_method}_{str(request_value['attempt_id'])[:8]}"
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
        method=f"auto_compare:{candidate['pnp_method']}+{candidate['extrinsic_method']}",
        status=CalibrationStatus.NEEDS_VALIDATION,
        quality=CalibrationQuality(
            num_observations=int(candidate["observation_count"]),
            num_inliers=int(candidate["inlier_count"]),
            mean_reprojection_error_px=candidate.get(
                "mean_reprojection_error_px"
            ),
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
        },
    )


def _validate_and_rank(
    attempt_root: Path,
    request_value: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    intrinsics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    sensor_metadata = {
        str(item["sensor_key"]): item for item in request_value["sensors"]
    }
    profiles: list[CalibrationProfile] = []
    results = []
    all_checks = []
    for sensor_key in request_value["sensor_keys"]:
        ranked = rank_candidates(
            [item for item in candidates if item["sensor_key"] == sensor_key]
        )
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
        "failed_camera_count": sum(
            1 for item in results if item["status"] == "failed"
        ),
        "thresholds": {
            "min_inliers": 6,
            "max_mean_translation_mm": 10.0,
            "max_mean_rotation_deg": 5.0,
            "max_outlier_ratio": 0.25,
        },
        "results": results,
    }
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
        observation_report = _calibration_observation_report(
            request_value, observations
        )
        atomic_write_json(
            attempt_root / CALIBRATION_OBSERVATIONS, observation_report
        )
        _update_progress(
            attempt_root,
            phase="estimate_target_poses",
            phase_status="complete",
            message="Target poses were estimated with the shared robust mask.",
        )
        _update_progress(
            attempt_root,
            phase="compare_robot_camera_solutions",
            phase_status="running",
            message="Evaluating every compatible PnP/extrinsic combination.",
        )
        candidates = _compare_solutions(
            attempt_root, request_value, observations
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
    ranking = _read_json(attempt_root / RANKING_FILE) if (attempt_root / RANKING_FILE).is_file() else None
    promotion = _read_json(attempt_root / PROMOTION_FILE) if (attempt_root / PROMOTION_FILE).is_file() else None
    return {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "run_root": root.as_posix(),
        "request": request_value,
        "progress": progress,
        "results": ranking,
        "promotion": promotion,
        "artifacts": {
            name: _relative(attempt_root / name, root)
            for name in (
                REQUEST_FILE,
                PROGRESS_FILE,
                SYNC_QUALITY_REPORT,
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
            raise ValueError(
                f"Candidate {candidate_id!r} did not pass validation"
            )
        selected[sensor_key] = str(candidate_id)
    if not selected:
        raise ValueError("No passing camera recommendations are available to promote")
    return selected


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
    if isinstance(prior_promotion, Mapping) and prior_promotion.get(
        "status"
    ) != "failed":
        raise ValueError("Calibration attempt already has promotion evidence")
    selected = _promotion_selections(attempt, selections)
    value = {
        "schema_version": "calibration_promotion_request.v1",
        "attempt_id": attempt_id,
        "run_root": root.as_posix(),
        "created_at": utc_now_iso(),
        "operator": str(operator).strip() if operator else None,
        "selections": selected,
        "previous_failure": (
            dict(prior_promotion)
            if isinstance(prior_promotion, Mapping)
            else None
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
            "operator": value["operator"],
        },
    )
    return value


def _profile_slot(profile: CalibrationProfile) -> tuple[str, str]:
    return profile.sensor_type.value, profile.sensor_id


def _selected_profiles(
    attempt_root: Path,
    attempt: Mapping[str, Any],
    request_value: Mapping[str, Any],
    promotion_request: Mapping[str, Any],
) -> list[CalibrationProfile]:
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
        observation_count = profile.quality.num_observations
        inlier_count = profile.quality.num_inliers
        outlier_count = int(
            profile.metadata.get(
                "outlier_count",
                max(0, observation_count - inlier_count),
            )
        )
        outlier_ratio = (
            outlier_count / observation_count if observation_count else 1.0
        )
        if (
            inlier_count < DEFAULT_MIN_INLIERS
            or profile.quality.residual_translation_mm is None
            or profile.quality.residual_translation_mm
            > DEFAULT_MAX_MEAN_TRANSLATION_MM
            or profile.quality.residual_rotation_deg is None
            or profile.quality.residual_rotation_deg
            > DEFAULT_MAX_MEAN_ROTATION_DEG
            or outlier_ratio > DEFAULT_MAX_OUTLIER_RATIO
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
    selected_checks = [
        item
        for item in _read_json(attempt_root / CHECKS_FILE)["checks"]
        if item.get("candidate_id") in selected_candidate_ids
    ]
    candidate_report = {
        "schema_version": "calibration_candidates.v1",
        "generated_at": utc_now_iso(),
        "run_root": attempt["run_root"],
        "attempt_id": attempt["attempt_id"],
        "overall_status": "ok",
        "candidate_count": len(selected_candidates),
        "inlier_count": sum(profile.quality.num_inliers for profile in selected_profiles),
        "outlier_count": sum(
            profile.quality.num_observations - profile.quality.num_inliers
            for profile in selected_profiles
        ),
        "profiles": all_profile_values,
        "candidates": selected_candidates,
        "comparisons": selected_results,
        "checks": selected_checks,
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
        "checks": candidate_report["checks"],
    }
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
        },
        "promotion": {
            "requested": True,
            "promoted": True,
            "path": CALIBRATION_PROFILES,
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


def promote_calibration_attempt(run_root: str | Path, attempt_id: str) -> dict[str, Any]:
    root = Path(run_root)
    attempt_root = calibration_attempt_root(root, attempt_id)
    attempt = load_calibration_attempt(root, attempt_id)
    request_value = attempt["request"]
    promotion_request = _read_json(attempt_root / PROMOTION_REQUEST_FILE)
    promotion_path = attempt_root / PROMOTION_FILE
    current = _read_json(promotion_path)
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
        existing = load_profile_collection(existing_path) if existing_path.is_file() else []
        promoted_slots = {_profile_slot(profile) for profile in selected_profiles}
        preserved = [
            profile for profile in existing if _profile_slot(profile) not in promoted_slots
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
            if (
                sensor_key in selected_sensor_keys
                and item.get("pnp_method") == selected_pnp_methods.get(sensor_key)
            ):
                selected_observations.append(dict(item))
        observations_report["observations"] = selected_observations
        observations_report["observation_count"] = len(selected_observations)
        observations_report["sensors"] = [
            item
            for item in observations_report.get("sensors", [])
            if item.get("sensor_key") in selected_sensor_keys
        ]
        observations_report["sensor_count"] = len(observations_report["sensors"])
        observations_report["promoted_candidate_ids"] = sorted(
            selected_candidate_ids
        )
        atomic_write_json(
            staging / CALIBRATION_OBSERVATIONS, observations_report
        )
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
            attempt_root, attempt, selected_profiles, promotion_request
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
                root
                / LIBRARY_DIRECTORY
                / str(request_value["target_id"]),
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
