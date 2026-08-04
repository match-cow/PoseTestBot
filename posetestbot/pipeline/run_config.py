"""Versioned run configuration for PoseTestBot pipeline jobs."""

from __future__ import annotations

import json
import math
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import fcntl

from posetestbot.io.atomic import atomic_write_json
from posetestbot.config import (
    DEFAULT_CAPTURE_VELOCITY_M_S,
    RobotProfile,
    robot_profile,
)
from posetestbot.io.artifacts import CALIBRATION_PROFILE_SELECTION, RUN_CONFIG
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    upsert_stage,
    write_run_manifest,
)
from posetestbot.pipeline.sequences import (
    PipelineSequenceSpec,
    build_sequence_job,
    build_sequence_plan,
)
from posetestbot.pipeline.stages import PipelineStageSpec
from posetestbot.robot.reference_frames import (
    normalize_sunrise_reference_frame_path,
)
from posetestbot.sensors.contracts import MountingMode, SensorType


LEGACY_SCHEMA_VERSION = "run_config.v1"
PREVIOUS_SCHEMA_VERSION = "run_config.v2"
SCHEMA_VERSION = "run_config.v3"
SUPPORTED_SCHEMA_VERSIONS = {
    LEGACY_SCHEMA_VERSION,
    PREVIOUS_SCHEMA_VERSION,
    SCHEMA_VERSION,
}
CAPTURE_SYNCHRONIZATION_SCHEMA_VERSION = "capture_synchronization.v1"
CAPTURE_SYNCHRONIZATION_MODES = {"timestamp_aligned", "hardware_trigger"}
HARDWARE_TRIGGER_IMPLEMENTATION = "realsense_inter_cam_sync"
HARDWARE_TRIGGER_SCOPE = "depth_exposure"
DEFAULT_MAX_DEPTH_TIMESTAMP_SKEW_MS = 2.0
MAX_DEPTH_TIMESTAMP_SKEW_MS = 5.0
_SAFE_SYNC_GROUP_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_SAFE_HARDWARE_DEVICE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
DATASET_MODES = {"objectless", "pose_template"}
CALIBRATION_PROFILE_OPTION_STAGES = ("blenderproc_prepare", "bop_export")
INTRINSIC_CALIBRATION_PROFILE_OPTION_STAGES = ("camera_rectification",)
DATASET_MODE_OPTION_STAGES = (
    "blenderproc_prepare",
    "blenderproc_render",
    "bop_export",
)
EXECUTION_GATE_OPTION_KEYS = frozenset({"allow_cameras", "allow_real_robot"})
RUN_CONFIG_LOCK = ".run_config.lock"

_RUN_CONFIG_LOCK = threading.RLock()
_RUN_CONFIG_LOCK_STATE = threading.local()

LAB_REALSENSE_SERIALS = (
    "825412070181",
    "033422071805",
    "923322072633",
)

SENSOR_TYPE_ALIASES = {
    "realsense": SensorType.REALSENSE_D435,
    "realsense_d435": SensorType.REALSENSE_D435,
    "d435": SensorType.REALSENSE_D435,
    "luxonis": SensorType.OAK_D_PRO,
    "oak": SensorType.OAK_D_PRO,
    "oak_d": SensorType.OAK_D_PRO,
    "oak_d_pro": SensorType.OAK_D_PRO,
    "zed": SensorType.ZED_2I,
    "zed2i": SensorType.ZED_2I,
    "zed_2i": SensorType.ZED_2I,
}


@contextmanager
def run_config_lock(run_root: str | Path):
    """Serialize run-config transactions across threads and processes.

    Pose-template selection performs a read-modify-write of ``run_config.json``.
    Keeping that transaction on the same lock as ordinary config replacement
    prevents either writer from observing or promoting an intermediate version.
    """

    with _RUN_CONFIG_LOCK:
        root = Path(run_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        lock_path = root / RUN_CONFIG_LOCK
        held = getattr(_RUN_CONFIG_LOCK_STATE, "locks", None)
        if held is None:
            held = {}
            _RUN_CONFIG_LOCK_STATE.locks = held
        depth = int(held.get(lock_path, 0))
        if depth:
            held[lock_path] = depth + 1
            try:
                yield root
            finally:
                held[lock_path] -= 1
            return
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            with os.fdopen(descriptor, "a+b", closefd=False) as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                held[lock_path] = 1
                try:
                    yield root
                finally:
                    held.pop(lock_path, None)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@dataclass(frozen=True)
class SensorRunConfig:
    """One intended sensor participant in a run."""

    sensor_type: str
    device_id: str
    display_name: str
    mounting_mode: str = MountingMode.EYE_IN_HAND.value
    enabled: bool = True
    calibration_profile_id: str | None = None
    inverted: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    operator_alias: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        operator_alias = (
            self.operator_alias.strip() if self.operator_alias is not None else None
        )
        if not operator_alias:
            data.pop("operator_alias")
        else:
            data["operator_alias"] = operator_alias
            data["display_name"] = operator_alias
        return data


@dataclass(frozen=True)
class CaptureSynchronizationConfig:
    """Cross-camera acquisition timing requested for one run."""

    schema_version: str = CAPTURE_SYNCHRONIZATION_SCHEMA_VERSION
    mode: str = "timestamp_aligned"
    implementation: str | None = None
    scope: str | None = None
    group_id: str | None = None
    master_sensor_key: str | None = None
    max_depth_timestamp_skew_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "mode": self.mode,
        }
        if self.mode == "hardware_trigger":
            result.update(
                {
                    "implementation": self.implementation,
                    "scope": self.scope,
                    "group_id": self.group_id,
                    "master_sensor_key": self.master_sensor_key,
                    "max_depth_timestamp_skew_ms": (self.max_depth_timestamp_skew_ms),
                }
            )
        return result


@dataclass(frozen=True)
class CaptureRunConfig:
    """Capture defaults shared by hardware adapters."""

    resolution: str = "720p"
    fps: int = 6
    velocity_m_s: float = DEFAULT_CAPTURE_VELOCITY_M_S
    sensors: tuple[SensorRunConfig, ...] = ()
    synchronization: CaptureSynchronizationConfig = field(
        default_factory=CaptureSynchronizationConfig
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sensors"] = [sensor.to_dict() for sensor in self.sensors]
        data["synchronization"] = self.synchronization.to_dict()
        return data


@dataclass(frozen=True)
class PipelineRunConfig:
    """Default sequence and options for one configured run."""

    sequence_id: str = "real_full_capture_validation"
    plan_only: bool = True
    options: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FixedFrameTransform:
    """One operator-supplied fixed edge in the run frame graph."""

    from_frame: str
    to_frame: str
    rotation_quaternion_wxyz: tuple[float, float, float, float]
    translation_mm: tuple[float, float, float]
    source: str = "operator_configured"

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_frame,
            "to": self.to_frame,
            "rotation_quaternion_wxyz": list(self.rotation_quaternion_wxyz),
            "translation_mm": list(self.translation_mm),
            "source": self.source,
        }


@dataclass(frozen=True)
class RunFramesConfig:
    robot_pose: Mapping[str, str] = field(
        default_factory=lambda: {
            "from": "robot_flange",
            "to": "template_base",
            "convention": "kuka_abc_radians",
        }
    )
    dataset_reference_frame: str = "template_base"
    fixed_transforms: tuple[FixedFrameTransform, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_pose": dict(self.robot_pose),
            "dataset_reference_frame": self.dataset_reference_frame,
            "fixed_transforms": [item.to_dict() for item in self.fixed_transforms],
        }


@dataclass(frozen=True)
class PoseTestBotRunConfig:
    """Top-level versioned run configuration artifact."""

    schema_version: str
    run_name: str
    run_root: str
    robot_profile: RobotProfile
    capture: CaptureRunConfig
    frames: RunFramesConfig = field(default_factory=RunFramesConfig)
    dataset_mode: str = "objectless"
    pose_template: Mapping[str, Any] | None = None
    calibration_profiles: str | None = None
    intrinsic_calibration_profiles: str | None = None
    calibration_profile_selection: Mapping[str, Any] | None = None
    calibration_target: Mapping[str, Any] | None = None
    pipeline: PipelineRunConfig = field(default_factory=PipelineRunConfig)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "run_name": self.run_name,
            "run_root": self.run_root,
            "robot_profile": asdict(self.robot_profile),
            "capture": self.capture.to_dict(),
            "frames": self.frames.to_dict(),
            "calibration_profiles": self.calibration_profiles,
            "calibration_target": (
                dict(self.calibration_target)
                if self.calibration_target is not None
                else None
            ),
            "pipeline": self.pipeline.to_dict(),
        }
        if self.intrinsic_calibration_profiles is not None:
            result["intrinsic_calibration_profiles"] = (
                self.intrinsic_calibration_profiles
            )
        if self.calibration_profile_selection is not None:
            result["calibration_profile_selection"] = dict(
                self.calibration_profile_selection
            )
        if self.schema_version in {PREVIOUS_SCHEMA_VERSION, SCHEMA_VERSION}:
            result["dataset_mode"] = self.dataset_mode
            result["pose_template"] = (
                dict(self.pose_template) if self.pose_template is not None else None
            )
        return result


def normalize_sensor_type(value: str) -> SensorType:
    key = value.strip().lower().replace("-", "_")
    try:
        return SENSOR_TYPE_ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(sorted(SENSOR_TYPE_ALIASES))
        raise ValueError(
            f"Unknown sensor type {value!r}; use one of: {choices}"
        ) from exc


def normalize_mounting_mode(value: str) -> MountingMode:
    key = value.strip().lower().replace("-", "_")
    try:
        return MountingMode(key)
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in MountingMode)
        raise ValueError(
            f"Unknown mounting mode {value!r}; use one of: {choices}"
        ) from exc


def normalize_inverted(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    key = str(value).strip().lower().replace("-", "_")
    if key in {"1", "true", "yes", "y", "inverted", "upside_down"}:
        return True
    if key in {"0", "false", "no", "n", "normal", "upright", ""}:
        return False
    raise ValueError(
        "Unknown sensor orientation "
        f"{value!r}; use inverted, normal, true, false, 1, or 0"
    )


def normalize_sensor_enabled(value: Any) -> bool:
    """Return a sensor participation flag without truthy-string coercion."""

    if isinstance(value, bool):
        return value
    raise ValueError("Sensor enabled must be a literal JSON boolean")


def normalize_operator_alias(value: Any) -> str | None:
    """Return a trimmed, optional operator-facing camera alias."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Sensor operator_alias must be a string or null")
    return value.strip() or None


def capture_synchronization_from_mapping(
    value: Mapping[str, Any] | CaptureSynchronizationConfig | None,
) -> CaptureSynchronizationConfig:
    """Normalize one strict camera synchronization policy."""

    if value is None:
        return CaptureSynchronizationConfig()
    if isinstance(value, CaptureSynchronizationConfig):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("capture.synchronization must be a JSON object")
    schema_version = str(value.get("schema_version", ""))
    if schema_version != CAPTURE_SYNCHRONIZATION_SCHEMA_VERSION:
        raise ValueError(
            "capture.synchronization.schema_version must be "
            f"{CAPTURE_SYNCHRONIZATION_SCHEMA_VERSION}"
        )
    mode = str(value.get("mode", ""))
    if mode not in CAPTURE_SYNCHRONIZATION_MODES:
        raise ValueError(
            "capture.synchronization.mode must be one of: "
            + ", ".join(sorted(CAPTURE_SYNCHRONIZATION_MODES))
        )
    base_keys = {"schema_version", "mode"}
    if mode == "timestamp_aligned":
        unexpected = sorted(set(value) - base_keys)
        if unexpected:
            raise ValueError(
                "timestamp_aligned capture synchronization does not accept: "
                + ", ".join(unexpected)
            )
        return CaptureSynchronizationConfig()

    allowed_keys = base_keys | {
        "implementation",
        "scope",
        "group_id",
        "master_sensor_key",
        "max_depth_timestamp_skew_ms",
    }
    unexpected = sorted(set(value) - allowed_keys)
    if unexpected:
        raise ValueError(
            "hardware_trigger capture synchronization contains unknown fields: "
            + ", ".join(unexpected)
        )
    implementation = str(value.get("implementation", ""))
    if implementation != HARDWARE_TRIGGER_IMPLEMENTATION:
        raise ValueError(
            "hardware_trigger capture synchronization implementation must be "
            f"{HARDWARE_TRIGGER_IMPLEMENTATION}"
        )
    scope = str(value.get("scope", ""))
    if scope != HARDWARE_TRIGGER_SCOPE:
        raise ValueError(
            "hardware_trigger capture synchronization scope must be "
            f"{HARDWARE_TRIGGER_SCOPE}"
        )
    group_id = str(value.get("group_id", ""))
    if not _SAFE_SYNC_GROUP_ID.fullmatch(group_id):
        raise ValueError(
            "hardware_trigger capture synchronization group_id must contain "
            "1-64 letters, digits, '.', '_', or '-', and start with a letter or digit"
        )
    master_sensor_key = str(value.get("master_sensor_key", ""))
    if not master_sensor_key:
        raise ValueError(
            "hardware_trigger capture synchronization master_sensor_key is required"
        )
    skew_value = value.get(
        "max_depth_timestamp_skew_ms",
        DEFAULT_MAX_DEPTH_TIMESTAMP_SKEW_MS,
    )
    if isinstance(skew_value, bool):
        raise ValueError(
            "hardware_trigger max_depth_timestamp_skew_ms must be a finite "
            "positive number"
        )
    try:
        max_skew = float(skew_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "hardware_trigger max_depth_timestamp_skew_ms must be a finite "
            "positive number"
        ) from exc
    if (
        not math.isfinite(max_skew)
        or max_skew <= 0
        or max_skew > MAX_DEPTH_TIMESTAMP_SKEW_MS
    ):
        raise ValueError(
            "hardware_trigger max_depth_timestamp_skew_ms must be a finite "
            f"positive number no greater than {MAX_DEPTH_TIMESTAMP_SKEW_MS}"
        )
    return CaptureSynchronizationConfig(
        mode=mode,
        implementation=implementation,
        scope=scope,
        group_id=group_id,
        master_sensor_key=master_sensor_key,
        max_depth_timestamp_skew_ms=max_skew,
    )


def validate_capture_synchronization(
    synchronization: Mapping[str, Any] | CaptureSynchronizationConfig | None,
    sensors: list[Mapping[str, Any]] | tuple[SensorRunConfig, ...],
) -> CaptureSynchronizationConfig:
    """Validate synchronization against the exact enabled camera set."""

    policy = capture_synchronization_from_mapping(synchronization)
    if policy.mode == "timestamp_aligned":
        return policy

    enabled: dict[str, Mapping[str, Any] | SensorRunConfig] = {}
    mounting_modes: set[str] = set()
    for index, raw_sensor in enumerate(sensors):
        sensor = (
            raw_sensor.to_dict()
            if isinstance(raw_sensor, SensorRunConfig)
            else raw_sensor
        )
        if not isinstance(sensor, Mapping):
            raise ValueError(f"Run config sensor {index} must be an object")
        if not normalize_sensor_enabled(sensor.get("enabled", True)):
            continue
        sensor_type = normalize_sensor_type(str(sensor.get("sensor_type", ""))).value
        device_id = str(sensor.get("device_id", "")).strip()
        sensor_key = f"{sensor_type}:{device_id}"
        if sensor_type != SensorType.REALSENSE_D435.value:
            raise ValueError(
                "hardware_trigger capture synchronization supports enabled "
                "RealSense D435 cameras only; OAK-D Pro and ZED 2i are unsupported"
            )
        if device_id.lower() in {"", "auto", "default"}:
            raise ValueError(
                "hardware_trigger capture synchronization requires exact "
                f"RealSense device IDs; got {device_id!r}"
            )
        if not _SAFE_HARDWARE_DEVICE_ID.fullmatch(device_id):
            raise ValueError(
                "hardware_trigger RealSense device IDs must contain 1-128 "
                "letters, digits, '.', '_', or '-', and start with a letter "
                "or digit"
            )
        if sensor_key in enabled:
            raise ValueError(
                f"hardware_trigger capture synchronization duplicates {sensor_key}"
            )
        enabled[sensor_key] = raw_sensor
        mounting_modes.add(
            normalize_mounting_mode(str(sensor.get("mounting_mode", ""))).value
        )
    if len(enabled) < 2:
        raise ValueError(
            "hardware_trigger capture synchronization requires at least two "
            "enabled exact-ID RealSense D435 cameras"
        )
    if mounting_modes != {
        MountingMode.EYE_IN_HAND.value,
        MountingMode.STATIC.value,
    }:
        raise ValueError(
            "hardware_trigger research capture requires both static and "
            "eye_in_hand enabled cameras"
        )
    if policy.master_sensor_key not in enabled:
        raise ValueError(
            "hardware_trigger master_sensor_key must exactly match one enabled "
            "RealSense D435 sensor key"
        )
    return policy


def _validate_sensor_orientation(sensor_type: SensorType | str, inverted: bool) -> None:
    normalized = (
        sensor_type if isinstance(sensor_type, SensorType) else SensorType(sensor_type)
    )
    if inverted and normalized != SensorType.REALSENSE_D435:
        raise ValueError("Sensor inverted=true is only supported for RealSense D435")


def _required_mounting_mode(
    value: Any,
    *,
    default_mounting_mode: str | None,
) -> str:
    """Resolve an explicitly authored camera mount without guessing one."""

    candidate = value
    if candidate is None or not str(candidate).strip():
        candidate = default_mounting_mode
    if candidate is None or not str(candidate).strip():
        raise ValueError(
            "Sensor mounting_mode is required; set it per sensor or provide "
            "an explicit default mounting mode"
        )
    return normalize_mounting_mode(str(candidate)).value


def sensor_config_from_token(
    token: str,
    *,
    default_mounting_mode: str | None = None,
) -> SensorRunConfig:
    """Parse ``sensor_type:device_id[:mounting_mode[:display_name[:orientation]]]``."""

    parts = token.split(":")
    if len(parts) < 2 or len(parts) > 5:
        raise ValueError(
            "Sensor entries must look like "
            "sensor_type:device_id[:mounting_mode[:display_name[:orientation]]]"
        )
    sensor_type = normalize_sensor_type(parts[0])
    device_id = parts[1].strip()
    if not device_id:
        raise ValueError("Sensor device_id must not be empty")
    mounting_mode = _required_mounting_mode(
        parts[2] if len(parts) >= 3 else None,
        default_mounting_mode=default_mounting_mode,
    )
    operator_alias = parts[3].strip() if len(parts) >= 4 and parts[3].strip() else None
    display_name = operator_alias or f"{sensor_type.value}:{device_id}"
    inverted = normalize_inverted(parts[4]) if len(parts) == 5 else False
    _validate_sensor_orientation(sensor_type, inverted)
    return SensorRunConfig(
        sensor_type=sensor_type.value,
        device_id=device_id,
        display_name=display_name,
        mounting_mode=mounting_mode,
        inverted=inverted,
        operator_alias=operator_alias,
    )


def sensor_config_from_mapping(
    value: Mapping[str, Any],
    *,
    default_mounting_mode: str | None = None,
) -> SensorRunConfig:
    sensor_type = normalize_sensor_type(str(value.get("sensor_type", "")))
    device_id = str(value.get("device_id", "")).strip()
    if not device_id:
        raise ValueError("Sensor device_id must not be empty")
    mounting_mode = _required_mounting_mode(
        value.get("mounting_mode"),
        default_mounting_mode=default_mounting_mode,
    )
    operator_alias = normalize_operator_alias(value.get("operator_alias"))
    display_name = str(
        operator_alias
        or value.get("display_name")
        or f"{sensor_type.value}:{device_id}"
    ).strip()
    metadata = value.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("Sensor metadata must be a JSON object")
    calibration_profile_id = value.get("calibration_profile_id")
    if calibration_profile_id is not None:
        calibration_profile_id = str(calibration_profile_id)
    inverted = normalize_inverted(value.get("inverted", False))
    _validate_sensor_orientation(sensor_type, inverted)
    return SensorRunConfig(
        sensor_type=sensor_type.value,
        device_id=device_id,
        display_name=display_name,
        mounting_mode=mounting_mode,
        enabled=normalize_sensor_enabled(value.get("enabled", True)),
        calibration_profile_id=calibration_profile_id,
        inverted=inverted,
        metadata=dict(metadata),
        operator_alias=operator_alias,
    )


def sensor_configs_from_values(
    values: list[Any] | None,
    *,
    default_mounting_mode: str | None = None,
) -> tuple[SensorRunConfig, ...]:
    normalized_default = (
        normalize_mounting_mode(default_mounting_mode).value
        if default_mounting_mode is not None
        else None
    )
    if values is None:
        mode = _required_mounting_mode(
            None,
            default_mounting_mode=normalized_default,
        )
        return default_lab_sensors(mounting_mode=mode)
    if not values:
        raise ValueError("At least one sensor entry is required")
    sensors = []
    for value in values:
        if isinstance(value, str):
            sensors.append(
                sensor_config_from_token(
                    value,
                    default_mounting_mode=normalized_default,
                )
            )
        elif isinstance(value, Mapping):
            sensors.append(
                sensor_config_from_mapping(
                    value,
                    default_mounting_mode=normalized_default,
                )
            )
        else:
            raise ValueError("Sensor entries must be strings or JSON objects")
    return tuple(sensors)


def sensor_configs_from_status(
    sensor_status: Mapping[str, Any],
    *,
    default_mounting_mode: str | None = None,
) -> tuple[SensorRunConfig, ...]:
    """Build run-config sensor entries from discovered status devices."""

    sensors: list[SensorRunConfig] = []
    mode = (
        normalize_mounting_mode(default_mounting_mode).value
        if default_mounting_mode is not None
        else None
    )
    for family in sensor_status.get("families", []):
        if not isinstance(family, Mapping):
            continue
        for device in family.get("devices", []):
            if not isinstance(device, Mapping) or not device.get("connected", True):
                continue
            sensor_type = normalize_sensor_type(str(device.get("sensor_type", "")))
            device_id = str(device.get("device_id", "")).strip()
            if not device_id:
                continue
            operator_alias = normalize_operator_alias(device.get("alias"))
            effective_display_name = str(
                device.get("effective_display_name") or ""
            ).strip()
            discovered_display_name = str(device.get("display_name") or "").strip()
            if (
                operator_alias is None
                and effective_display_name
                and effective_display_name != discovered_display_name
            ):
                operator_alias = effective_display_name
            display_name = str(
                operator_alias
                or effective_display_name
                or discovered_display_name
                or f"{sensor_type.value}:{device_id}"
            )
            mounting_mode = _required_mounting_mode(
                device.get("mounting_mode"),
                default_mounting_mode=mode,
            )
            inverted = normalize_inverted(device.get("inverted", False))
            _validate_sensor_orientation(sensor_type, inverted)
            metadata = device.get("metadata", {})
            if not isinstance(metadata, Mapping):
                metadata = {}
            sensors.append(
                SensorRunConfig(
                    sensor_type=sensor_type.value,
                    device_id=device_id,
                    display_name=display_name,
                    mounting_mode=mounting_mode,
                    inverted=inverted,
                    metadata=dict(metadata),
                    operator_alias=operator_alias,
                )
            )
    return tuple(sensors)


def default_lab_sensors(
    *,
    mounting_mode: str = MountingMode.EYE_IN_HAND.value,
) -> tuple[SensorRunConfig, ...]:
    mode = normalize_mounting_mode(mounting_mode).value
    sensors = [
        SensorRunConfig(
            sensor_type=SensorType.REALSENSE_D435.value,
            device_id=serial,
            display_name=f"RealSense D435 {serial}",
            mounting_mode=mode,
            metadata={"lab_profile": "current_posetestbot"},
        )
        for serial in LAB_REALSENSE_SERIALS
    ]
    sensors.extend(
        [
            SensorRunConfig(
                sensor_type=SensorType.OAK_D_PRO.value,
                device_id="auto",
                display_name="Luxonis OAK-D Pro",
                mounting_mode=mode,
                metadata={"lab_profile": "current_posetestbot"},
            ),
            SensorRunConfig(
                sensor_type=SensorType.ZED_2I.value,
                device_id="auto",
                display_name="Stereolabs ZED 2i",
                mounting_mode=mode,
                metadata={"lab_profile": "current_posetestbot"},
            ),
        ]
    )
    return tuple(sensors)


def create_run_config(
    *,
    run_root: str | Path,
    run_name: str | None = None,
    resolution: str = "720p",
    fps: int = 6,
    velocity_m_s: float = DEFAULT_CAPTURE_VELOCITY_M_S,
    sensors: tuple[SensorRunConfig, ...] | None = None,
    dataset_mode: str | None = None,
    pose_template: Mapping[str, Any] | None = None,
    calibration_profiles: str | None = None,
    intrinsic_calibration_profiles: str | None = None,
    calibration_profile_selection: Mapping[str, Any] | None = None,
    calibration_target: Mapping[str, Any] | None = None,
    sequence_id: str = "real_full_capture_validation",
    sequence_options: Mapping[str, Any] | None = None,
    plan_only: bool = True,
    fixed_transforms: tuple[FixedFrameTransform, ...] = (),
    robot_pose_sunrise_reference_frame_path: str | None = None,
    synchronization: (Mapping[str, Any] | CaptureSynchronizationConfig | None) = None,
) -> PoseTestBotRunConfig:
    run_root_path = Path(run_root)
    sensor_configs = sensors if sensors is not None else default_lab_sensors()
    inferred_mode = dataset_mode or "objectless"
    if inferred_mode not in DATASET_MODES:
        raise ValueError(
            "dataset_mode must be one of: " + ", ".join(sorted(DATASET_MODES))
        )
    synchronization_config = validate_capture_synchronization(
        synchronization,
        sensor_configs,
    )
    robot_pose = {
        "from": "robot_flange",
        "to": "template_base",
        "convention": "kuka_abc_radians",
    }
    if robot_pose_sunrise_reference_frame_path is not None:
        robot_pose["sunrise_reference_frame_path"] = (
            normalize_sunrise_reference_frame_path(
                robot_pose_sunrise_reference_frame_path
            )
        )
    config = PoseTestBotRunConfig(
        schema_version=SCHEMA_VERSION,
        run_name=run_name or run_root_path.name,
        run_root=run_root_path.as_posix(),
        robot_profile=robot_profile().with_overrides(
            cartesian_velocity_m_s=velocity_m_s,
        ),
        capture=CaptureRunConfig(
            resolution=resolution,
            fps=fps,
            velocity_m_s=velocity_m_s,
            sensors=tuple(sensor_configs),
            synchronization=synchronization_config,
        ),
        frames=RunFramesConfig(
            robot_pose=robot_pose,
            fixed_transforms=fixed_transforms,
        ),
        dataset_mode=inferred_mode,
        pose_template=dict(pose_template) if pose_template is not None else None,
        calibration_profiles=calibration_profiles,
        intrinsic_calibration_profiles=intrinsic_calibration_profiles,
        calibration_profile_selection=(
            dict(calibration_profile_selection)
            if calibration_profile_selection is not None
            else None
        ),
        calibration_target=(
            dict(calibration_target) if calibration_target is not None else None
        ),
        pipeline=PipelineRunConfig(
            sequence_id=sequence_id,
            plan_only=plan_only,
            options=dict(sequence_options or {}),
        ),
    )
    validate_run_config(config.to_dict())
    return config


def _persisted_execution_gate(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in EXECUTION_GATE_OPTION_KEYS:
                return key
            nested = _persisted_execution_gate(item)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = _persisted_execution_gate(item)
            if nested is not None:
                return nested
    return None


def fixed_transform_from_mapping(value: Mapping[str, Any]) -> FixedFrameTransform:
    quaternion = value.get("rotation_quaternion_wxyz")
    translation = value.get("translation_mm")
    if not isinstance(quaternion, (list, tuple)) or len(quaternion) != 4:
        raise ValueError("Fixed transform rotation_quaternion_wxyz must have 4 values")
    if not isinstance(translation, (list, tuple)) or len(translation) != 3:
        raise ValueError("Fixed transform translation_mm must have 3 values")
    return FixedFrameTransform(
        from_frame=str(value.get("from", "")),
        to_frame=str(value.get("to", "")),
        rotation_quaternion_wxyz=tuple(float(item) for item in quaternion),
        translation_mm=tuple(float(item) for item in translation),
        source=str(value.get("source") or "operator_configured"),
    )


def validate_run_config(value: Mapping[str, Any]) -> None:
    schema = value.get("schema_version")
    if schema not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            "Run config schema_version must be one of: "
            + ", ".join(sorted(SUPPORTED_SCHEMA_VERSIONS))
        )
    if schema in {PREVIOUS_SCHEMA_VERSION, SCHEMA_VERSION}:
        retired_fields = sorted({"object_folder", "selected_objects"} & value.keys())
        if retired_fields:
            raise ValueError(
                "Run config contains retired legacy object-registry fields: "
                + ", ".join(retired_fields)
            )
        dataset_mode = value.get("dataset_mode")
        if dataset_mode not in DATASET_MODES:
            raise ValueError(
                "Run config dataset_mode must be one of: "
                + ", ".join(sorted(DATASET_MODES))
            )
        pose_template = value.get("pose_template")
        if pose_template is not None:
            if not isinstance(pose_template, Mapping):
                raise ValueError("Run config pose_template must be an object or null")
            if (
                pose_template.get("selection_artifact")
                != "pose_template_selection.json"
            ):
                raise ValueError(
                    "Run config pose_template.selection_artifact must be "
                    "pose_template_selection.json"
                )

    robot = value.get("robot_profile")
    if not isinstance(robot, Mapping):
        raise ValueError("Run config robot_profile must be an object")
    if robot.get("mode") != "real":
        raise ValueError("Run config robot_profile.mode must be 'real'")

    capture = value.get("capture")
    if not isinstance(capture, Mapping):
        raise ValueError("Run config capture must be an object")
    if int(capture.get("fps", 0)) <= 0:
        raise ValueError("Run config capture.fps must be positive")
    if not str(capture.get("resolution", "")).strip():
        raise ValueError("Run config capture.resolution must not be empty")
    try:
        velocity_m_s = float(
            capture.get(
                "velocity_m_s",
                robot.get("cartesian_velocity_m_s"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Run config capture.velocity_m_s must be a finite positive number"
        ) from exc
    if not math.isfinite(velocity_m_s) or velocity_m_s <= 0.0:
        raise ValueError(
            "Run config capture.velocity_m_s must be a finite positive number"
        )
    calibration_target = value.get("calibration_target")
    intrinsic_profiles = value.get("intrinsic_calibration_profiles")
    if intrinsic_profiles is not None and (
        not isinstance(intrinsic_profiles, str) or not intrinsic_profiles.strip()
    ):
        raise ValueError(
            "Run config intrinsic_calibration_profiles must be a non-empty path or null"
        )
    calibration_selection = value.get("calibration_profile_selection")
    if calibration_selection is not None:
        if not isinstance(calibration_selection, Mapping):
            raise ValueError(
                "Run config calibration_profile_selection must be an object or null"
            )
        if (
            calibration_selection.get("selection_artifact")
            != CALIBRATION_PROFILE_SELECTION
        ):
            raise ValueError(
                "Run config calibration_profile_selection.selection_artifact must be "
                f"{CALIBRATION_PROFILE_SELECTION}"
            )
        digest = str(calibration_selection.get("bundle_sha256", ""))
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(
                "Run config calibration_profile_selection.bundle_sha256 must be a SHA-256 digest"
            )
        if not isinstance(value.get("calibration_profiles"), str) or not isinstance(
            intrinsic_profiles, str
        ):
            raise ValueError(
                "Run config calibration selection requires both calibration profile paths"
            )
    if calibration_target is not None:
        if not isinstance(calibration_target, Mapping):
            raise ValueError("Run config calibration_target must be an object or null")
        required_target_fields = {
            "target_id",
            "bundle_path",
            "source_sha256",
            "spec_sha256",
            "pdf_sha256",
            "configuration_sha256",
            "geometry_sha256",
            "placement",
        }
        missing_target_fields = sorted(
            required_target_fields - calibration_target.keys()
        )
        if missing_target_fields:
            raise ValueError(
                "Run config calibration_target is missing: "
                + ", ".join(missing_target_fields)
            )
        if not str(calibration_target.get("target_id", "")).strip():
            raise ValueError(
                "Run config calibration_target.target_id must not be empty"
            )
        bundle_path = Path(str(calibration_target.get("bundle_path", "")))
        if bundle_path.is_absolute() or ".." in bundle_path.parts:
            raise ValueError(
                "Run config calibration_target.bundle_path must be run-relative"
            )
        for hash_key in (
            "source_sha256",
            "spec_sha256",
            "pdf_sha256",
            "configuration_sha256",
            "geometry_sha256",
        ):
            digest = str(calibration_target.get(hash_key, ""))
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(
                    f"Run config calibration_target.{hash_key} must be a SHA-256 digest"
                )
        placement = calibration_target.get("placement")
        if not isinstance(placement, Mapping) or placement.get("mode") not in {
            "unknown",
            "template_base_identity",
            "posegridgen_board_to_base",
        }:
            raise ValueError("Run config calibration_target placement mode is invalid")
        mounting_frame = placement.get("mounting_frame")
        if "mounting_frame" in placement and mounting_frame not in {
            "robot_flange",
            "template_base",
        }:
            raise ValueError(
                "Run config calibration_target placement mounting_frame must be "
                "robot_flange or template_base"
            )
        if (
            placement.get("mode") != "unknown"
            and mounting_frame is not None
            and mounting_frame != "template_base"
        ):
            raise ValueError(
                "Run config known calibration-target placement requires "
                "mounting_frame=template_base"
            )
    if schema in {PREVIOUS_SCHEMA_VERSION, SCHEMA_VERSION}:
        if (
            value["dataset_mode"] == "objectless"
            and value.get("pose_template") is not None
        ):
            raise ValueError("Objectless run config cannot reference a pose template")
        if (
            value["dataset_mode"] != "pose_template"
            and value.get("pose_template") is not None
        ):
            raise ValueError(
                "Only pose_template dataset mode may reference a pose template"
            )

    frames = value.get("frames")
    if frames is not None:
        if not isinstance(frames, Mapping):
            raise ValueError("Run config frames must be an object")
        robot_pose = frames.get("robot_pose")
        if (
            not isinstance(robot_pose, Mapping)
            or robot_pose.get("from") != "robot_flange"
            or robot_pose.get("to") != "template_base"
        ):
            raise ValueError(
                "Run config frames.robot_pose must map robot_flange to template_base"
            )
        if robot_pose.get("convention") != "kuka_abc_radians":
            raise ValueError(
                "Run config robot pose convention must be kuka_abc_radians"
            )
        if "sunrise_reference_frame_path" in robot_pose:
            try:
                normalize_sunrise_reference_frame_path(
                    robot_pose.get("sunrise_reference_frame_path")
                )
            except ValueError as exc:
                raise ValueError(
                    "Run config frames.robot_pose.sunrise_reference_frame_path "
                    f"is invalid: {exc}"
                ) from exc
        if frames.get("dataset_reference_frame") != "template_base":
            raise ValueError("Run config dataset_reference_frame must be template_base")
        fixed_transforms = frames.get("fixed_transforms", [])
        if not isinstance(fixed_transforms, list):
            raise ValueError("Run config frames.fixed_transforms must be a list")
        for index, transform in enumerate(fixed_transforms):
            if not isinstance(transform, Mapping):
                raise ValueError(f"Fixed transform {index} must be an object")
            if not str(transform.get("from", "")) or not str(transform.get("to", "")):
                raise ValueError(f"Fixed transform {index} requires from/to endpoints")
            quaternion = transform.get("rotation_quaternion_wxyz")
            translation = transform.get("translation_mm")
            if not isinstance(quaternion, list) or len(quaternion) != 4:
                raise ValueError(
                    f"Fixed transform {index} quaternion must have 4 values"
                )
            if not isinstance(translation, list) or len(translation) != 3:
                raise ValueError(
                    f"Fixed transform {index} translation must have 3 values"
                )
            values = [float(item) for item in [*quaternion, *translation]]
            if not all(math.isfinite(item) for item in values):
                raise ValueError(f"Fixed transform {index} must be finite")
            if not math.isclose(
                sum(float(item) ** 2 for item in quaternion), 1.0, abs_tol=1e-3
            ):
                raise ValueError(
                    f"Fixed transform {index} quaternion must be normalized"
                )

    sensors = capture.get("sensors")
    if not isinstance(sensors, list) or not sensors:
        raise ValueError("Run config capture.sensors must be a non-empty list")
    enabled_sensor_count = 0
    for index, sensor in enumerate(sensors):
        if not isinstance(sensor, Mapping):
            raise ValueError(f"Run config sensor {index} must be an object")
        sensor_type = normalize_sensor_type(str(sensor.get("sensor_type", "")))
        normalize_mounting_mode(str(sensor.get("mounting_mode", "")))
        if not str(sensor.get("device_id", "")).strip():
            raise ValueError(f"Run config sensor {index} device_id must not be empty")
        try:
            normalize_operator_alias(sensor.get("operator_alias"))
        except ValueError as exc:
            raise ValueError(
                f"Run config sensor {index} operator_alias must be a string or null"
            ) from exc
        try:
            enabled = normalize_sensor_enabled(sensor.get("enabled", True))
        except ValueError as exc:
            raise ValueError(
                f"Run config sensor {index} enabled must be a boolean"
            ) from exc
        enabled_sensor_count += int(enabled)
        inverted = normalize_inverted(sensor.get("inverted", False))
        _validate_sensor_orientation(sensor_type, inverted)
    if enabled_sensor_count == 0:
        raise ValueError("Run config must enable at least one capture sensor")
    synchronization = capture.get("synchronization")
    if schema == SCHEMA_VERSION and synchronization is None:
        raise ValueError("run_config.v3 requires capture.synchronization")
    if schema in {LEGACY_SCHEMA_VERSION, PREVIOUS_SCHEMA_VERSION}:
        legacy_policy = capture_synchronization_from_mapping(synchronization)
        if legacy_policy.mode != "timestamp_aligned":
            raise ValueError(
                f"{schema} cannot claim hardware_trigger capture synchronization; "
                f"use {SCHEMA_VERSION}"
            )
    else:
        validate_capture_synchronization(synchronization, sensors)

    pipeline = value.get("pipeline")
    if not isinstance(pipeline, Mapping):
        raise ValueError("Run config pipeline must be an object")
    sequence_id = str(pipeline.get("sequence_id", ""))
    if not sequence_id:
        raise ValueError("Run config pipeline.sequence_id must not be empty")
    options = pipeline.get("options", {})
    if not isinstance(options, Mapping):
        raise ValueError("Run config pipeline.options must be an object")
    persisted_gate = _persisted_execution_gate(options)
    if persisted_gate is not None:
        raise ValueError(
            f"Run config pipeline.options must not persist execution gate: "
            f"{persisted_gate}"
        )
    build_sequence_plan(
        sequence_id=sequence_id,
        run_root=str(value.get("run_root", ".")),
        options=options,
        plan_only=bool(pipeline.get("plan_only", True)),
    )


def write_run_config(run_root: str | Path, config: PoseTestBotRunConfig) -> Path:
    with run_config_lock(run_root) as root:
        return atomic_write_json(root / RUN_CONFIG, config.to_dict())


def write_run_config_with_manifest(
    run_root: str | Path,
    config: PoseTestBotRunConfig,
) -> Path:
    run_root_path = Path(run_root)
    path = write_run_config(run_root_path, config)
    manifest = load_or_create_run_manifest(run_root_path)
    upsert_stage(
        manifest,
        name="run_config",
        status="succeeded",
        artifacts={RUN_CONFIG: path},
        run_root=run_root_path,
        message=(
            "Created run config for "
            f"{len(config.capture.sensors)} sensor(s), "
            "real robot profile, "
            f"sequence {config.pipeline.sequence_id}."
        ),
    )
    write_run_manifest(manifest, run_root_path)
    return path


def load_run_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Run config must be a JSON object: {path}")
    if "frames" not in value:
        value["frames"] = RunFramesConfig().to_dict()
        warnings = value.setdefault("warnings", [])
        if not isinstance(warnings, list):
            raise ValueError("Run config warnings must be a list")
        warnings.append(
            {
                "code": "legacy_frames_inferred",
                "message": (
                    "Run config omitted frames; inferred robot_flange -> "
                    "template_base with kuka_abc_radians. Rewrite the config "
                    "to make frame semantics explicit."
                ),
            }
        )
    capture = value.get("capture")
    if isinstance(capture, Mapping):
        sensors = capture.get("sensors")
        if isinstance(sensors, list):
            for sensor in sensors:
                if isinstance(sensor, dict):
                    sensor.setdefault("enabled", True)
                    operator_alias = normalize_operator_alias(
                        sensor.get("operator_alias")
                    )
                    if operator_alias is not None:
                        sensor["operator_alias"] = operator_alias
                        sensor["display_name"] = operator_alias
        if (
            value.get("schema_version")
            in {
                LEGACY_SCHEMA_VERSION,
                PREVIOUS_SCHEMA_VERSION,
            }
            and "synchronization" not in capture
        ):
            if not isinstance(capture, dict):
                capture = dict(capture)
                value["capture"] = capture
            capture["synchronization"] = CaptureSynchronizationConfig().to_dict()
            warnings = value.setdefault("warnings", [])
            if not isinstance(warnings, list):
                raise ValueError("Run config warnings must be a list")
            warnings.append(
                {
                    "code": "legacy_capture_synchronization_inferred",
                    "message": (
                        "Legacy run config omitted capture synchronization; "
                        "inferred timestamp_aligned. It is not hardware-sync evidence."
                    ),
                }
            )
    value.setdefault("calibration_target", None)
    if value.get("schema_version") in {PREVIOUS_SCHEMA_VERSION, SCHEMA_VERSION}:
        value.setdefault("pose_template", None)
    validate_run_config(value)
    value.setdefault("dataset_mode", "objectless")
    return value


def load_run_config_for_run_root(run_root: str | Path) -> dict[str, Any]:
    run_root_path = Path(run_root)
    config = load_run_config(run_root_path / RUN_CONFIG)
    config_run_root = Path(str(config["run_root"])).resolve()
    if config_run_root != run_root_path.resolve():
        raise ValueError(
            "Run config run_root does not match requested run_root: "
            f"{config['run_root']} != {run_root_path.as_posix()}"
        )
    return config


def _sequence_options_with_run_config_defaults(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    pipeline = config["pipeline"]
    options = {
        str(key): dict(value)
        for key, value in dict(pipeline.get("options", {})).items()
    }
    plan = build_sequence_plan(
        sequence_id=str(pipeline["sequence_id"]),
        run_root=str(config["run_root"]),
        options=options,
        plan_only=bool(pipeline.get("plan_only", True)),
    )
    available_groups = {step.id for step in plan.steps}
    available_groups.update(step.stage_id for step in plan.steps)
    calibration_profiles = config.get("calibration_profiles")
    if isinstance(calibration_profiles, str) and calibration_profiles.strip():
        for group_name in CALIBRATION_PROFILE_OPTION_STAGES:
            if group_name not in available_groups:
                continue
            group_options = dict(options.get(group_name, {}))
            group_options.setdefault("calibration_profiles", calibration_profiles)
            options[group_name] = group_options

    intrinsic_profiles = config.get("intrinsic_calibration_profiles")
    if isinstance(intrinsic_profiles, str) and intrinsic_profiles.strip():
        for group_name in INTRINSIC_CALIBRATION_PROFILE_OPTION_STAGES:
            if group_name not in available_groups:
                continue
            group_options = dict(options.get(group_name, {}))
            group_options.setdefault("intrinsic_profiles", intrinsic_profiles)
            options[group_name] = group_options

    for group_name in DATASET_MODE_OPTION_STAGES:
        if group_name not in available_groups:
            continue
        group_options = dict(options.get(group_name, {}))
        if (
            "objectless" not in group_options
            and config.get("dataset_mode") == "objectless"
        ):
            group_options["objectless"] = True
        options[group_name] = group_options
    return options


def sequence_plan_from_run_config(config: Mapping[str, Any]):
    validate_run_config(config)
    pipeline = config["pipeline"]
    return build_sequence_plan(
        sequence_id=str(pipeline["sequence_id"]),
        run_root=str(config["run_root"]),
        options=_sequence_options_with_run_config_defaults(config),
        plan_only=bool(pipeline.get("plan_only", True)),
    )


def build_sequence_job_from_run_config(
    config: Mapping[str, Any],
    *,
    sequence_registry: Mapping[str, PipelineSequenceSpec] | None = None,
    stage_registry: Mapping[str, PipelineStageSpec] | None = None,
):
    validate_run_config(config)
    pipeline = config["pipeline"]
    return build_sequence_job(
        sequence_id=str(pipeline["sequence_id"]),
        run_root=str(config["run_root"]),
        options=_sequence_options_with_run_config_defaults(config),
        plan_only=bool(pipeline.get("plan_only", True)),
        sequence_registry=sequence_registry,
        stage_registry=stage_registry,
    )
