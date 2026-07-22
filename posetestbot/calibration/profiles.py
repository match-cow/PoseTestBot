"""Typed calibration profile contract for rewrite-era sensor calibration."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from posetestbot.calibration.intrinsics import projection_is_opencv_compatible
from posetestbot.io.atomic import atomic_write_json
from posetestbot.sensors.contracts import CameraIntrinsics, MountingMode, SensorType

SCHEMA_VERSION = "calibration.v2"
LEGACY_SCHEMA_VERSION = "calibration.v1"
QUATERNION_NORM_TOLERANCE = 1e-3


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
    ROBOT_FLANGE = "robot_flange"
    TEMPLATE_BASE = "template_base"
    ARUCO_GRID = "aruco_grid"
    TCP = "tcp"
    PHYSICAL_ROBOT_BASE = "physical_robot_base"
    # Source-compatible aliases. calibration.v1 values are converted by the loader.
    END_EFFECTOR = "robot_flange"
    ROBOT_BASE = "template_base"
    CELL_WORLD = "template_base"


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
            expected = TransformFrame.ROBOT_FLANGE
        else:
            expected = TransformFrame.TEMPLATE_BASE
        if self.to_frame != expected:
            raise ValueError(
                f"{mounting_mode.value} calibration must transform camera to "
                f"{expected.value}"
            )
        if len(self.rotation_quaternion_wxyz) != 4:
            raise ValueError("rotation_quaternion_wxyz must have 4 values")
        if len(self.translation_mm) != 3:
            raise ValueError("translation_mm must have 3 values")
        values = (*self.rotation_quaternion_wxyz, *self.translation_mm)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Calibration extrinsics must contain only finite values")
        quaternion_norm = math.sqrt(
            sum(float(value) ** 2 for value in self.rotation_quaternion_wxyz)
        )
        if abs(quaternion_norm - 1.0) > QUATERNION_NORM_TOLERANCE:
            raise ValueError(
                "rotation_quaternion_wxyz must be normalized to unit length"
            )


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
        if self.num_inliers > self.num_observations:
            raise ValueError("num_inliers cannot exceed num_observations")
        for name in (
            "mean_reprojection_error_px",
            "max_reprojection_error_px",
            "residual_translation_mm",
            "residual_rotation_deg",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"quality.{name} must be finite and nonnegative")


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
    rectified_intrinsics: CameraIntrinsics | None = None
    rectified_valid_roi: tuple[int, int, int, int] | None = None
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
        if not self.rig_position:
            raise ValueError("rig_position is required")
        if len(self.intrinsics.cam_k) != 9:
            raise ValueError("intrinsics.cam_k must have 9 values")
        intrinsic_values = (*self.intrinsics.cam_k, *self.intrinsics.distortion)
        if not all(math.isfinite(float(value)) for value in intrinsic_values):
            raise ValueError("Calibration intrinsics must contain only finite values")
        if (
            not math.isfinite(float(self.intrinsics.depth_scale_to_mm))
            or self.intrinsics.depth_scale_to_mm <= 0
        ):
            raise ValueError("intrinsics.depth_scale_to_mm must be finite and positive")
        legacy_unvalidated = (
            self.status == CalibrationStatus.NEEDS_VALIDATION
            and self.method == "legacy_camera_ee_transform"
            and bool(self.metadata.get("legacy_sensor_key"))
        )
        if legacy_unvalidated:
            if self.intrinsics.width < 0 or self.intrinsics.height < 0:
                raise ValueError("legacy intrinsics dimensions cannot be negative")
        else:
            if self.intrinsics.width <= 0 or self.intrinsics.height <= 0:
                raise ValueError("intrinsics width and height must be positive")
            if self.intrinsics.cam_k[0] <= 0 or self.intrinsics.cam_k[4] <= 0:
                raise ValueError("intrinsics focal lengths fx and fy must be positive")
        if not all(
            math.isclose(float(value), expected, abs_tol=1e-9)
            for value, expected in zip(self.intrinsics.cam_k[6:9], (0.0, 0.0, 1.0))
        ):
            raise ValueError("intrinsics.cam_k bottom row must be [0, 0, 1]")
        if self.rectified_intrinsics is not None:
            rectified = self.rectified_intrinsics
            if (
                rectified.width != self.intrinsics.width
                or rectified.height != self.intrinsics.height
            ):
                raise ValueError(
                    "rectified intrinsics must preserve native output resolution"
                )
            if len(rectified.cam_k) != 9:
                raise ValueError("rectified_intrinsics.cam_k must have 9 values")
            if any(
                not math.isclose(float(value), 0.0, abs_tol=1e-12)
                for value in rectified.distortion
            ):
                raise ValueError("rectified intrinsics must have zero distortion")
        if self.rectified_valid_roi is not None:
            if len(self.rectified_valid_roi) != 4 or any(
                int(value) < 0 for value in self.rectified_valid_roi
            ):
                raise ValueError(
                    "rectified valid ROI must contain four nonnegative integers"
                )
            x, y, width, height = self.rectified_valid_roi
            if x + width > self.intrinsics.width or y + height > self.intrinsics.height:
                raise ValueError(
                    "rectified valid ROI must fit within output resolution"
                )
        if self.sync_delta_ms is not None and not math.isfinite(
            float(self.sync_delta_ms)
        ):
            raise ValueError("sync_delta_ms must be finite")
        self.extrinsics.validate_for_mounting_mode(self.mounting_mode)
        self.quality.validate()
        if self.status == CalibrationStatus.VALID and self.quality.num_inliers <= 0:
            raise ValueError(
                "valid calibration profiles must record at least one inlier"
            )


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


def _intrinsics_to_dict(intrinsics: CameraIntrinsics) -> dict[str, Any]:
    distortion = [float(item) for item in intrinsics.distortion[:5]]
    distortion.extend([0.0] * (5 - len(distortion)))
    return {
        "cam_K": list(intrinsics.cam_k),
        "width": intrinsics.width,
        "height": intrinsics.height,
        "distortion_model": intrinsics.distortion_model,
        "distortion": distortion,
        "depth_scale_to_mm": intrinsics.depth_scale_to_mm,
        "projection_source": intrinsics.projection_source,
    }


def _intrinsics_from_dict(value: Mapping[str, Any]) -> CameraIntrinsics:
    distortion = [float(item) for item in value.get("distortion", [])[:5]]
    distortion.extend([0.0] * (5 - len(distortion)))
    return CameraIntrinsics(
        cam_k=tuple(float(item) for item in value["cam_K"]),
        width=int(value.get("width", 0)),
        height=int(value.get("height", 0)),
        distortion=tuple(distortion),
        depth_scale_to_mm=float(value.get("depth_scale_to_mm", 1.0)),
        distortion_model=str(value.get("distortion_model", "brown_conrady")),
        projection_source=(
            str(value["projection_source"])
            if value.get("projection_source") is not None
            else None
        ),
    )


def rectified_projection_from_native(
    intrinsics: CameraIntrinsics,
) -> tuple[CameraIntrinsics | None, tuple[int, int, int, int] | None]:
    if intrinsics.width <= 0 or intrinsics.height <= 0:
        return None, None
    if not projection_is_opencv_compatible(
        {
            "distortion_model": intrinsics.distortion_model,
            "distortion": list(intrinsics.distortion),
        }
    ):
        return None, None
    matrix = np.asarray(intrinsics.cam_k, dtype=float).reshape(3, 3)
    distortion = np.zeros(5, dtype=float)
    source = np.asarray(intrinsics.distortion, dtype=float).reshape(-1)
    distortion[: min(5, source.size)] = source[:5]
    rectified, roi = cv2.getOptimalNewCameraMatrix(
        matrix,
        distortion,
        (intrinsics.width, intrinsics.height),
        0.0,
        (intrinsics.width, intrinsics.height),
    )
    return (
        CameraIntrinsics(
            cam_k=tuple(float(item) for item in rectified.reshape(-1)),
            width=intrinsics.width,
            height=intrinsics.height,
            distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
            depth_scale_to_mm=intrinsics.depth_scale_to_mm,
            distortion_model="brown_conrady",
            projection_source=(
                f"rectified_alpha0_from:{intrinsics.projection_source}"
                if intrinsics.projection_source
                else "rectified_alpha0"
            ),
        ),
        tuple(int(item) for item in roi),
    )


def rectified_intrinsics_from_native(
    intrinsics: CameraIntrinsics,
) -> CameraIntrinsics | None:
    return rectified_projection_from_native(intrinsics)[0]


def _transform_frame(value: Any, *, source_schema: str) -> TransformFrame:
    name = str(value)
    if source_schema == LEGACY_SCHEMA_VERSION:
        name = {
            "end_effector": TransformFrame.ROBOT_FLANGE.value,
            "robot_base": TransformFrame.TEMPLATE_BASE.value,
            "cell_world": TransformFrame.TEMPLATE_BASE.value,
        }.get(name, name)
    return TransformFrame(name)


def profile_to_dict(profile: CalibrationProfile) -> dict[str, Any]:
    profile.validate()
    derived_rectified, derived_roi = rectified_projection_from_native(
        profile.intrinsics
    )
    rectified_intrinsics = profile.rectified_intrinsics or derived_rectified
    rectified_roi = profile.rectified_valid_roi or derived_roi
    rectified_value = (
        _intrinsics_to_dict(rectified_intrinsics)
        if rectified_intrinsics is not None
        else None
    )
    if rectified_value is not None:
        rectified_value.update(
            {"alpha": 0.0, "valid_roi": list(rectified_roi or (0, 0, 0, 0))}
        )
    return {
        "schema_version": profile.schema_version,
        "profile_id": profile.profile_id,
        "sensor_id": profile.sensor_id,
        "sensor_type": profile.sensor_type.value,
        "mounting_mode": profile.mounting_mode.value,
        "rig_position": profile.rig_position,
        "intrinsics": {
            "native": _intrinsics_to_dict(profile.intrinsics),
            "rectified": rectified_value,
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
        "projection_provenance": dict(
            profile.metadata.get(
                "projection_provenance",
                {
                    "native": "captured_or_calibrated_color_projection",
                    "rectified": "opencv_alpha0_same_resolution",
                    "depth_scale": "factory_sdk_not_recalibrated",
                    "depth_alignment": "capture_adapter_sdk_depth_to_color",
                },
            )
        ),
    }


def profile_from_dict(value: Mapping[str, Any]) -> CalibrationProfile:
    source_schema = str(value.get("schema_version"))
    if source_schema not in {SCHEMA_VERSION, LEGACY_SCHEMA_VERSION}:
        raise ValueError(f"Unsupported calibration schema: {source_schema!r}")
    raw_intrinsics = value["intrinsics"]
    if not isinstance(raw_intrinsics, Mapping):
        raise ValueError("Calibration intrinsics must be an object")
    if source_schema == LEGACY_SCHEMA_VERSION:
        intrinsics = raw_intrinsics
        rectified_intrinsics = None
    else:
        intrinsics = raw_intrinsics.get("native")
        rectified_intrinsics = raw_intrinsics.get("rectified")
        if not isinstance(intrinsics, Mapping):
            raise ValueError("calibration.v2 intrinsics.native must be an object")
    extrinsics = value["extrinsics"]
    quality = value.get("quality", {})
    native_intrinsics = _intrinsics_from_dict(intrinsics)
    normalized_rectified = (
        _intrinsics_from_dict(rectified_intrinsics)
        if isinstance(rectified_intrinsics, Mapping)
        else rectified_intrinsics_from_native(native_intrinsics)
    )
    rectified_roi = (
        tuple(int(item) for item in rectified_intrinsics.get("valid_roi", []))
        if isinstance(rectified_intrinsics, Mapping)
        and isinstance(rectified_intrinsics.get("valid_roi"), list)
        and len(rectified_intrinsics["valid_roi"]) == 4
        else None
    )
    profile = CalibrationProfile(
        schema_version=SCHEMA_VERSION,
        profile_id=str(value["profile_id"]),
        sensor_id=str(value["sensor_id"]),
        sensor_type=SensorType(value["sensor_type"]),
        mounting_mode=MountingMode(value["mounting_mode"]),
        rig_position=str(value.get("rig_position", "")),
        intrinsics=native_intrinsics,
        rectified_intrinsics=normalized_rectified,
        rectified_valid_roi=rectified_roi,
        extrinsics=RigidTransform(
            from_frame=_transform_frame(
                extrinsics["from"], source_schema=source_schema
            ),
            to_frame=_transform_frame(extrinsics["to"], source_schema=source_schema),
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
        metadata={
            **dict(value.get("metadata", {})),
            **(
                {"projection_provenance": dict(value["projection_provenance"])}
                if isinstance(value.get("projection_provenance"), Mapping)
                else {}
            ),
        },
    )
    profile.validate()
    return profile


def load_profile(path: str | Path) -> CalibrationProfile:
    with open(path, "r") as f:
        return profile_from_dict(json.load(f))


def write_profile(profile: CalibrationProfile, path: str | Path) -> Path:
    profile.validate()
    path = Path(path)
    return atomic_write_json(path, profile_to_dict(profile))


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


def write_profile_collection(
    profiles: list[CalibrationProfile], path: str | Path
) -> Path:
    path = Path(path)
    validate_profile_collection(profiles)
    return atomic_write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "profiles": [profile_to_dict(profile) for profile in profiles],
        },
    )


def load_profile_collection(path: str | Path) -> list[CalibrationProfile]:
    with open(path, "r") as f:
        value = json.load(f)
    if not isinstance(value, Mapping):
        raise ValueError("Calibration profile collection must be a JSON object")
    if value.get("schema_version") not in {SCHEMA_VERSION, LEGACY_SCHEMA_VERSION}:
        raise ValueError(f"Unsupported calibration collection schema: {value!r}")
    raw_profiles = value.get("profiles", [])
    if not isinstance(raw_profiles, list):
        raise ValueError("Calibration profile collection profiles must be a list")
    profiles = [profile_from_dict(item) for item in raw_profiles]
    validate_profile_collection(profiles)
    return profiles


def validate_profile_collection(profiles: Iterable[CalibrationProfile]) -> None:
    profile_list = list(profiles)
    seen_ids: set[str] = set()
    valid_slots: dict[tuple[SensorType, str, MountingMode, str], str] = {}
    for profile in profile_list:
        profile.validate()
        if profile.profile_id in seen_ids:
            raise ValueError(f"Duplicate calibration profile_id: {profile.profile_id}")
        seen_ids.add(profile.profile_id)
        if profile.status != CalibrationStatus.VALID:
            continue
        slot = (
            profile.sensor_type,
            profile.sensor_id,
            profile.mounting_mode,
            profile.rig_position,
        )
        existing = valid_slots.get(slot)
        if existing is not None:
            raise ValueError(
                "Multiple valid calibration profiles occupy the same sensor/mount/rig "
                f"slot: {existing}, {profile.profile_id}"
            )
        valid_slots[slot] = profile.profile_id


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
    required_statuses: set[CalibrationStatus] | None = None,
) -> CalibrationProfile:
    matches = []
    for profile in profiles:
        if mounting_mode is not None and profile.mounting_mode != mounting_mode:
            continue
        if required_statuses is not None and profile.status not in required_statuses:
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


def select_valid_profile_for_sensor(
    profiles: Iterable[CalibrationProfile],
    sensor_name: str,
    *,
    mounting_mode: MountingMode | None = None,
) -> CalibrationProfile:
    return select_profile_for_sensor(
        profiles,
        sensor_name,
        mounting_mode=mounting_mode,
        required_statuses={CalibrationStatus.VALID},
    )


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
        profile = select_valid_profile_for_sensor(profile_list, sensor_name)
        transform_map[sensor_name] = blenderproc_camera_transform_from_profile(profile)
    return transform_map
