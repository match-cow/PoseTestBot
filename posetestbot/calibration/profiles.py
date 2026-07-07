"""Typed calibration profile contract for rewrite-era sensor calibration."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping

from posetestbot.sensors.contracts import CameraIntrinsics, MountingMode, SensorType

SCHEMA_VERSION = "calibration.v1"


class CalibrationTargetType(StrEnum):
    CHARUCO = "charuco"
    ARUCO_GRID = "aruco_grid"
    CHECKERBOARD = "checkerboard"
    UNKNOWN = "unknown"


class CalibrationStatus(StrEnum):
    VALID = "valid"
    NEEDS_VALIDATION = "needs_validation"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class TransformFrame(StrEnum):
    CAMERA = "camera"
    END_EFFECTOR = "end_effector"
    ROBOT_BASE = "robot_base"
    CELL_WORLD = "cell_world"


@dataclass(frozen=True)
class RigidTransform:
    """Rigid transform with quaternion order matching the baseline: w, x, y, z."""

    from_frame: TransformFrame
    to_frame: TransformFrame
    rotation_quaternion_wxyz: tuple[float, float, float, float]
    translation_mm: tuple[float, float, float]

    def validate_for_mounting_mode(self, mounting_mode: MountingMode) -> None:
        if self.from_frame != TransformFrame.CAMERA:
            raise ValueError("Calibration extrinsics must use from_frame='camera'")
        if mounting_mode == MountingMode.EYE_IN_HAND:
            expected = TransformFrame.END_EFFECTOR
        else:
            expected = TransformFrame.ROBOT_BASE
        if self.to_frame != expected and not (
            mounting_mode == MountingMode.STATIC
            and self.to_frame == TransformFrame.CELL_WORLD
        ):
            raise ValueError(
                f"{mounting_mode.value} calibration must transform camera to "
                f"{expected.value}"
            )
        if len(self.rotation_quaternion_wxyz) != 4:
            raise ValueError("rotation_quaternion_wxyz must have 4 values")
        if len(self.translation_mm) != 3:
            raise ValueError("translation_mm must have 3 values")


@dataclass(frozen=True)
class CalibrationQuality:
    num_observations: int = 0
    num_inliers: int = 0
    mean_reprojection_error_px: float | None = None
    max_reprojection_error_px: float | None = None
    residual_translation_mm: float | None = None
    residual_rotation_deg: float | None = None
    notes: str | None = None

    def validate(self) -> None:
        if self.num_observations < 0:
            raise ValueError("num_observations cannot be negative")
        if self.num_inliers < 0:
            raise ValueError("num_inliers cannot be negative")
        if self.num_inliers > self.num_observations and self.num_observations:
            raise ValueError("num_inliers cannot exceed num_observations")


@dataclass(frozen=True)
class CalibrationProfile:
    schema_version: str
    profile_id: str
    sensor_id: str
    sensor_type: SensorType
    mounting_mode: MountingMode
    rig_position: str
    intrinsics: CameraIntrinsics
    extrinsics: RigidTransform
    target_type: CalibrationTargetType = CalibrationTargetType.UNKNOWN
    calibration_dataset_id: str | None = None
    method: str | None = None
    status: CalibrationStatus = CalibrationStatus.NEEDS_VALIDATION
    quality: CalibrationQuality = field(default_factory=CalibrationQuality)
    operator: str | None = None
    calibrated_at: str | None = None
    sync_delta_ms: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported calibration schema: {self.schema_version!r}")
        if not self.profile_id:
            raise ValueError("profile_id is required")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.profile_id):
            raise ValueError(
                "profile_id may only contain letters, numbers, underscore, dot, or dash"
            )
        if not self.sensor_id:
            raise ValueError("sensor_id is required")
        if self.intrinsics.width < 0 or self.intrinsics.height < 0:
            raise ValueError("intrinsics width/height cannot be negative")
        if len(self.intrinsics.cam_k) != 9:
            raise ValueError("intrinsics.cam_k must have 9 values")
        self.extrinsics.validate_for_mounting_mode(self.mounting_mode)
        self.quality.validate()


def _enum_value(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    return value


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return _enum_value(value)


def profile_to_dict(profile: CalibrationProfile) -> dict[str, Any]:
    profile.validate()
    return {
        "schema_version": profile.schema_version,
        "profile_id": profile.profile_id,
        "sensor_id": profile.sensor_id,
        "sensor_type": profile.sensor_type.value,
        "mounting_mode": profile.mounting_mode.value,
        "rig_position": profile.rig_position,
        "intrinsics": {
            "cam_K": list(profile.intrinsics.cam_k),
            "width": profile.intrinsics.width,
            "height": profile.intrinsics.height,
            "distortion": list(profile.intrinsics.distortion),
            "depth_scale_to_mm": profile.intrinsics.depth_scale_to_mm,
        },
        "extrinsics": {
            "from": profile.extrinsics.from_frame.value,
            "to": profile.extrinsics.to_frame.value,
            "rotation_quaternion_wxyz": list(
                profile.extrinsics.rotation_quaternion_wxyz
            ),
            "translation_mm": list(profile.extrinsics.translation_mm),
        },
        "target_type": profile.target_type.value,
        "calibration_dataset_id": profile.calibration_dataset_id,
        "method": profile.method,
        "status": profile.status.value,
        "quality": _jsonable(profile.quality),
        "operator": profile.operator,
        "calibrated_at": profile.calibrated_at,
        "sync_delta_ms": profile.sync_delta_ms,
        "metadata": dict(profile.metadata),
    }


def profile_from_dict(value: Mapping[str, Any]) -> CalibrationProfile:
    intrinsics = value["intrinsics"]
    extrinsics = value["extrinsics"]
    quality = value.get("quality", {})
    profile = CalibrationProfile(
        schema_version=str(value.get("schema_version")),
        profile_id=str(value["profile_id"]),
        sensor_id=str(value["sensor_id"]),
        sensor_type=SensorType(value["sensor_type"]),
        mounting_mode=MountingMode(value["mounting_mode"]),
        rig_position=str(value.get("rig_position", "")),
        intrinsics=CameraIntrinsics(
            cam_k=tuple(float(item) for item in intrinsics["cam_K"]),
            width=int(intrinsics.get("width", 0)),
            height=int(intrinsics.get("height", 0)),
            distortion=tuple(float(item) for item in intrinsics.get("distortion", [])),
            depth_scale_to_mm=float(intrinsics.get("depth_scale_to_mm", 1.0)),
        ),
        extrinsics=RigidTransform(
            from_frame=TransformFrame(extrinsics["from"]),
            to_frame=TransformFrame(extrinsics["to"]),
            rotation_quaternion_wxyz=tuple(
                float(item) for item in extrinsics["rotation_quaternion_wxyz"]
            ),
            translation_mm=tuple(float(item) for item in extrinsics["translation_mm"]),
        ),
        target_type=CalibrationTargetType(value.get("target_type", "unknown")),
        calibration_dataset_id=value.get("calibration_dataset_id"),
        method=value.get("method"),
        status=CalibrationStatus(value.get("status", "needs_validation")),
        quality=CalibrationQuality(
            num_observations=int(quality.get("num_observations", 0)),
            num_inliers=int(quality.get("num_inliers", 0)),
            mean_reprojection_error_px=quality.get("mean_reprojection_error_px"),
            max_reprojection_error_px=quality.get("max_reprojection_error_px"),
            residual_translation_mm=quality.get("residual_translation_mm"),
            residual_rotation_deg=quality.get("residual_rotation_deg"),
            notes=quality.get("notes"),
        ),
        operator=value.get("operator"),
        calibrated_at=value.get("calibrated_at"),
        sync_delta_ms=value.get("sync_delta_ms"),
        metadata=dict(value.get("metadata", {})),
    )
    profile.validate()
    return profile


def load_profile(path: str | Path) -> CalibrationProfile:
    with open(path, "r") as f:
        return profile_from_dict(json.load(f))


def write_profile(profile: CalibrationProfile, path: str | Path) -> Path:
    profile.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(profile_to_dict(profile), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def legacy_sensor_type(sensor_key: str) -> SensorType:
    key = sensor_key.lower()
    if key.startswith("realsense"):
        return SensorType.REALSENSE_D435
    if key.startswith("luxonis") or key.startswith("oak"):
        return SensorType.OAK_D_PRO
    if key.startswith("zed"):
        return SensorType.ZED_2I
    raise ValueError(f"Cannot infer sensor type from legacy key {sensor_key!r}")


def migrate_legacy_camera_ee_profiles(
    camera_ee_transform: Mapping[str, Mapping[str, Any]],
    *,
    sync_deltas_ms: Mapping[str, float] | None = None,
    intrinsics_by_sensor: Mapping[str, CameraIntrinsics] | None = None,
) -> list[CalibrationProfile]:
    profiles: list[CalibrationProfile] = []
    sync_deltas_ms = sync_deltas_ms or {}
    intrinsics_by_sensor = intrinsics_by_sensor or {}
    for sensor_key, transform in sorted(camera_ee_transform.items()):
        sensor_type = legacy_sensor_type(sensor_key)
        profile_id = f"{sensor_type.value}_{sensor_key}_eye_in_hand_wrist_legacy"
        intrinsics = intrinsics_by_sensor.get(
            sensor_key,
            CameraIntrinsics(
                cam_k=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                width=0,
                height=0,
            ),
        )
        profile = CalibrationProfile(
            schema_version=SCHEMA_VERSION,
            profile_id=profile_id,
            sensor_id=sensor_key,
            sensor_type=sensor_type,
            mounting_mode=MountingMode.EYE_IN_HAND,
            rig_position="wrist",
            intrinsics=intrinsics,
            extrinsics=RigidTransform(
                from_frame=TransformFrame.CAMERA,
                to_frame=TransformFrame.END_EFFECTOR,
                rotation_quaternion_wxyz=tuple(
                    float(item) for item in transform["quaternion"]
                ),
                translation_mm=tuple(float(item) for item in transform["position"]),
            ),
            target_type=CalibrationTargetType.ARUCO_GRID,
            method="legacy_camera_ee_transform",
            status=CalibrationStatus.NEEDS_VALIDATION,
            quality=CalibrationQuality(notes="Migrated from camera_ee_transform.json"),
            sync_delta_ms=sync_deltas_ms.get(sensor_key),
            metadata={"legacy_sensor_key": sensor_key},
        )
        profile.validate()
        profiles.append(profile)
    return profiles


def write_profile_collection(profiles: list[CalibrationProfile], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for profile in profiles:
        profile.validate()
    with open(path, "w") as f:
        json.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "profiles": [profile_to_dict(profile) for profile in profiles],
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")
    return path


def load_profile_collection(path: str | Path) -> list[CalibrationProfile]:
    with open(path, "r") as f:
        value = json.load(f)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported calibration collection schema: {value!r}")
    return [profile_from_dict(item) for item in value.get("profiles", [])]


def legacy_sensor_key_for_type(sensor_type: SensorType) -> str:
    if sensor_type == SensorType.REALSENSE_D435:
        return "realsense"
    if sensor_type == SensorType.OAK_D_PRO:
        return "luxonis"
    if sensor_type == SensorType.ZED_2I:
        return "zed_2i"
    return sensor_type.value


def sensor_identity_from_folder_name(sensor_name: str) -> tuple[SensorType | None, str]:
    name = sensor_name.lower()
    if name.startswith("realsense_"):
        return SensorType.REALSENSE_D435, sensor_name[len("realsense_") :]
    if name.startswith("realsense"):
        return SensorType.REALSENSE_D435, sensor_name
    if name.startswith("luxonis_"):
        return SensorType.OAK_D_PRO, sensor_name[len("luxonis_") :]
    if name.startswith("oak_"):
        return SensorType.OAK_D_PRO, sensor_name[len("oak_") :]
    if name.startswith("luxonis") or name.startswith("oak"):
        return SensorType.OAK_D_PRO, sensor_name
    if name.startswith("zed_2i_"):
        return SensorType.ZED_2I, sensor_name[len("zed_2i_") :]
    if name.startswith("zed_"):
        return SensorType.ZED_2I, sensor_name[len("zed_") :]
    if name.startswith("zed"):
        return SensorType.ZED_2I, sensor_name
    return None, sensor_name


def _profile_match_score(profile: CalibrationProfile, sensor_name: str) -> int | None:
    sensor_type, device_id = sensor_identity_from_folder_name(sensor_name)
    if sensor_type is not None and profile.sensor_type != sensor_type:
        return None

    legacy_key = legacy_sensor_key_for_type(profile.sensor_type)
    legacy_metadata_key = profile.metadata.get("legacy_sensor_key")
    candidates = {
        profile.sensor_id: 90,
        profile.profile_id: 80,
        str(legacy_metadata_key): 70 if legacy_metadata_key else 0,
        legacy_key: 60,
    }
    if profile.sensor_id == sensor_name:
        return 100
    if profile.sensor_id == device_id:
        return 95
    if sensor_name in candidates:
        return candidates[sensor_name]
    if device_id in candidates:
        return candidates[device_id]
    return candidates.get(legacy_key) if profile.sensor_id == legacy_key else None


def select_profile_for_sensor(
    profiles: Iterable[CalibrationProfile],
    sensor_name: str,
    *,
    mounting_mode: MountingMode | None = None,
) -> CalibrationProfile:
    matches = []
    for profile in profiles:
        if mounting_mode is not None and profile.mounting_mode != mounting_mode:
            continue
        score = _profile_match_score(profile, sensor_name)
        if score is not None:
            matches.append((score, profile))

    if not matches:
        mode = f" {mounting_mode.value}" if mounting_mode else ""
        raise KeyError(f"No{mode} calibration profile matches {sensor_name!r}")

    matches.sort(key=lambda item: item[0], reverse=True)
    if len(matches) > 1 and matches[0][0] == matches[1][0]:
        profile_ids = ", ".join(profile.profile_id for _, profile in matches)
        raise ValueError(
            f"Ambiguous calibration profiles for {sensor_name!r}: {profile_ids}"
        )
    return matches[0][1]


def select_eye_in_hand_profile_for_sensor(
    profiles: Iterable[CalibrationProfile], sensor_name: str
) -> CalibrationProfile:
    return select_profile_for_sensor(
        profiles, sensor_name, mounting_mode=MountingMode.EYE_IN_HAND
    )


def legacy_camera_ee_transform_from_profile(profile: CalibrationProfile) -> dict[str, list[float]]:
    profile.validate()
    if profile.mounting_mode != MountingMode.EYE_IN_HAND:
        raise ValueError(
            f"Profile {profile.profile_id!r} is {profile.mounting_mode.value}, "
            "not eye_in_hand"
        )
    return {
        "quaternion": list(profile.extrinsics.rotation_quaternion_wxyz),
        "position": list(profile.extrinsics.translation_mm),
    }


def legacy_camera_ee_transform_map_from_profiles(
    profiles: Iterable[CalibrationProfile], sensor_names: Iterable[str]
) -> dict[str, dict[str, list[float]]]:
    profile_list = list(profiles)
    transform_map = {}
    for sensor_name in sensor_names:
        profile = select_eye_in_hand_profile_for_sensor(profile_list, sensor_name)
        transform_map[sensor_name] = legacy_camera_ee_transform_from_profile(profile)
    return transform_map


def blenderproc_camera_transform_from_profile(
    profile: CalibrationProfile,
) -> dict[str, object]:
    """Return a BlenderProc-prep transform entry for eye-in-hand or static cameras."""

    profile.validate()
    return {
        "quaternion": list(profile.extrinsics.rotation_quaternion_wxyz),
        "position": list(profile.extrinsics.translation_mm),
        "mounting_mode": profile.mounting_mode.value,
        "from": profile.extrinsics.from_frame.value,
        "to": profile.extrinsics.to_frame.value,
        "profile_id": profile.profile_id,
    }


def blenderproc_camera_transform_map_from_profiles(
    profiles: Iterable[CalibrationProfile], sensor_names: Iterable[str]
) -> dict[str, dict[str, object]]:
    profile_list = list(profiles)
    transform_map = {}
    for sensor_name in sensor_names:
        profile = select_profile_for_sensor(profile_list, sensor_name)
        transform_map[sensor_name] = blenderproc_camera_transform_from_profile(profile)
    return transform_map


def write_legacy_camera_ee_transform_map(
    transform_map: Mapping[str, Mapping[str, object]], path: str | Path
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(transform_map, f, indent=2, sort_keys=True)
        f.write("\n")
    return path
