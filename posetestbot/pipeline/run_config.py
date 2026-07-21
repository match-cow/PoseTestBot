"""Versioned run configuration for PoseTestBot pipeline jobs."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from posetestbot.io.atomic import atomic_write_json
from posetestbot.config import RobotProfile, robot_profile
from posetestbot.io.artifacts import RUN_CONFIG
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
from posetestbot.sensors.contracts import MountingMode, SensorType


LEGACY_SCHEMA_VERSION = "run_config.v1"
SCHEMA_VERSION = "run_config.v2"
DATASET_MODES = {"objectless", "pose_template"}
CALIBRATION_PROFILE_OPTION_STAGES = ("blenderproc_prepare", "bop_export")
DATASET_MODE_OPTION_STAGES = (
    "blenderproc_prepare",
    "blenderproc_render",
    "bop_export",
)

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CaptureRunConfig:
    """Capture defaults shared by hardware adapters."""

    resolution: str = "720p"
    fps: int = 6
    velocity_m_s: float = 0.2
    sensors: tuple[SensorRunConfig, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sensors"] = [sensor.to_dict() for sensor in self.sensors]
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
                dict(self.calibration_target) if self.calibration_target is not None else None
            ),
            "pipeline": self.pipeline.to_dict(),
        }
        if self.schema_version == SCHEMA_VERSION:
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
        raise ValueError(f"Unknown sensor type {value!r}; use one of: {choices}") from exc


def normalize_mounting_mode(value: str) -> MountingMode:
    key = value.strip().lower().replace("-", "_")
    try:
        return MountingMode(key)
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in MountingMode)
        raise ValueError(f"Unknown mounting mode {value!r}; use one of: {choices}") from exc


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


def _validate_sensor_orientation(sensor_type: SensorType | str, inverted: bool) -> None:
    normalized = sensor_type if isinstance(sensor_type, SensorType) else SensorType(sensor_type)
    if inverted and normalized != SensorType.REALSENSE_D435:
        raise ValueError("Sensor inverted=true is only supported for RealSense D435")


def sensor_config_from_token(
    token: str,
    *,
    default_mounting_mode: str = MountingMode.EYE_IN_HAND.value,
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
    mounting_mode = normalize_mounting_mode(
        parts[2] if len(parts) >= 3 and parts[2] else default_mounting_mode
    )
    display_name = (
        parts[3].strip()
        if len(parts) >= 4 and parts[3].strip()
        else f"{sensor_type.value}:{device_id}"
    )
    inverted = normalize_inverted(parts[4]) if len(parts) == 5 else False
    _validate_sensor_orientation(sensor_type, inverted)
    return SensorRunConfig(
        sensor_type=sensor_type.value,
        device_id=device_id,
        display_name=display_name,
        mounting_mode=mounting_mode.value,
        inverted=inverted,
    )


def sensor_config_from_mapping(
    value: Mapping[str, Any],
    *,
    default_mounting_mode: str = MountingMode.EYE_IN_HAND.value,
) -> SensorRunConfig:
    sensor_type = normalize_sensor_type(str(value.get("sensor_type", "")))
    device_id = str(value.get("device_id", "")).strip()
    if not device_id:
        raise ValueError("Sensor device_id must not be empty")
    mounting_mode = normalize_mounting_mode(
        str(value.get("mounting_mode") or default_mounting_mode)
    ).value
    display_name = str(
        value.get("display_name") or f"{sensor_type}:{device_id}"
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
        enabled=bool(value.get("enabled", True)),
        calibration_profile_id=calibration_profile_id,
        inverted=inverted,
        metadata=dict(metadata),
    )


def sensor_configs_from_values(
    values: list[Any] | None,
    *,
    default_mounting_mode: str = MountingMode.EYE_IN_HAND.value,
) -> tuple[SensorRunConfig, ...]:
    if not values:
        return default_lab_sensors(mounting_mode=default_mounting_mode)
    sensors = []
    for value in values:
        if isinstance(value, str):
            sensors.append(
                sensor_config_from_token(
                    value,
                    default_mounting_mode=default_mounting_mode,
                )
            )
        elif isinstance(value, Mapping):
            sensors.append(
                sensor_config_from_mapping(
                    value,
                    default_mounting_mode=default_mounting_mode,
                )
            )
        else:
            raise ValueError("Sensor entries must be strings or JSON objects")
    return tuple(sensors)


def sensor_configs_from_status(
    sensor_status: Mapping[str, Any],
    *,
    default_mounting_mode: str = MountingMode.EYE_IN_HAND.value,
) -> tuple[SensorRunConfig, ...]:
    """Build run-config sensor entries from discovered status devices."""

    sensors: list[SensorRunConfig] = []
    mode = normalize_mounting_mode(default_mounting_mode).value
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
            display_name = str(
                device.get("effective_display_name")
                or device.get("alias")
                or device.get("display_name")
                or f"{sensor_type.value}:{device_id}"
            )
            mounting_mode = normalize_mounting_mode(
                str(device.get("mounting_mode") or mode)
            ).value
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
    velocity_m_s: float = 0.2,
    sensors: tuple[SensorRunConfig, ...] | None = None,
    dataset_mode: str | None = None,
    pose_template: Mapping[str, Any] | None = None,
    calibration_profiles: str | None = None,
    calibration_target: Mapping[str, Any] | None = None,
    sequence_id: str = "real_full_capture_validation",
    sequence_options: Mapping[str, Any] | None = None,
    plan_only: bool = True,
    fixed_transforms: tuple[FixedFrameTransform, ...] = (),
) -> PoseTestBotRunConfig:
    run_root_path = Path(run_root)
    sensor_configs = sensors if sensors is not None else default_lab_sensors()
    inferred_mode = dataset_mode or "objectless"
    if inferred_mode not in DATASET_MODES:
        raise ValueError("dataset_mode must be one of: " + ", ".join(sorted(DATASET_MODES)))
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
        ),
        frames=RunFramesConfig(fixed_transforms=fixed_transforms),
        dataset_mode=inferred_mode,
        pose_template=dict(pose_template) if pose_template is not None else None,
        calibration_profiles=calibration_profiles,
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
    if schema not in {SCHEMA_VERSION, LEGACY_SCHEMA_VERSION}:
        raise ValueError(
            f"Run config schema_version must be {SCHEMA_VERSION!r} or "
            f"{LEGACY_SCHEMA_VERSION!r}"
        )
    if schema == SCHEMA_VERSION:
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
            if pose_template.get("selection_artifact") != "pose_template_selection.json":
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
    calibration_target = value.get("calibration_target")
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
        missing_target_fields = sorted(required_target_fields - calibration_target.keys())
        if missing_target_fields:
            raise ValueError(
                "Run config calibration_target is missing: "
                + ", ".join(missing_target_fields)
            )
        if not str(calibration_target.get("target_id", "")).strip():
            raise ValueError("Run config calibration_target.target_id must not be empty")
        bundle_path = Path(str(calibration_target.get("bundle_path", "")))
        if bundle_path.is_absolute() or ".." in bundle_path.parts:
            raise ValueError("Run config calibration_target.bundle_path must be run-relative")
        for hash_key in (
            "source_sha256",
            "spec_sha256",
            "pdf_sha256",
            "configuration_sha256",
            "geometry_sha256",
        ):
            digest = str(calibration_target.get(hash_key, ""))
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
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
    if schema == SCHEMA_VERSION:
        if value["dataset_mode"] == "objectless" and value.get("pose_template") is not None:
            raise ValueError("Objectless run config cannot reference a pose template")
        if value["dataset_mode"] != "pose_template" and value.get("pose_template") is not None:
            raise ValueError("Only pose_template dataset mode may reference a pose template")

    frames = value.get("frames")
    if frames is not None:
        if not isinstance(frames, Mapping):
            raise ValueError("Run config frames must be an object")
        robot_pose = frames.get("robot_pose")
        if not isinstance(robot_pose, Mapping) or robot_pose.get("from") != "robot_flange" or robot_pose.get("to") != "template_base":
            raise ValueError(
                "Run config frames.robot_pose must map robot_flange to template_base"
            )
        if robot_pose.get("convention") != "kuka_abc_radians":
            raise ValueError("Run config robot pose convention must be kuka_abc_radians")
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
                raise ValueError(f"Fixed transform {index} quaternion must have 4 values")
            if not isinstance(translation, list) or len(translation) != 3:
                raise ValueError(f"Fixed transform {index} translation must have 3 values")
            values = [float(item) for item in [*quaternion, *translation]]
            if not all(math.isfinite(item) for item in values):
                raise ValueError(f"Fixed transform {index} must be finite")
            if not math.isclose(sum(float(item) ** 2 for item in quaternion), 1.0, abs_tol=1e-3):
                raise ValueError(f"Fixed transform {index} quaternion must be normalized")

    sensors = capture.get("sensors")
    if not isinstance(sensors, list) or not sensors:
        raise ValueError("Run config capture.sensors must be a non-empty list")
    for index, sensor in enumerate(sensors):
        if not isinstance(sensor, Mapping):
            raise ValueError(f"Run config sensor {index} must be an object")
        sensor_type = normalize_sensor_type(str(sensor.get("sensor_type", "")))
        normalize_mounting_mode(str(sensor.get("mounting_mode", "")))
        if not str(sensor.get("device_id", "")).strip():
            raise ValueError(f"Run config sensor {index} device_id must not be empty")
        inverted = normalize_inverted(sensor.get("inverted", False))
        _validate_sensor_orientation(sensor_type, inverted)

    pipeline = value.get("pipeline")
    if not isinstance(pipeline, Mapping):
        raise ValueError("Run config pipeline must be an object")
    sequence_id = str(pipeline.get("sequence_id", ""))
    if not sequence_id:
        raise ValueError("Run config pipeline.sequence_id must not be empty")
    options = pipeline.get("options", {})
    if not isinstance(options, Mapping):
        raise ValueError("Run config pipeline.options must be an object")
    build_sequence_plan(
        sequence_id=sequence_id,
        run_root=str(value.get("run_root", ".")),
        options=options,
        plan_only=bool(pipeline.get("plan_only", True)),
    )


def write_run_config(run_root: str | Path, config: PoseTestBotRunConfig) -> Path:
    path = Path(run_root) / RUN_CONFIG
    return atomic_write_json(path, config.to_dict())


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
    value.setdefault("calibration_target", None)
    if value.get("schema_version") == SCHEMA_VERSION:
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

    for group_name in DATASET_MODE_OPTION_STAGES:
        if group_name not in available_groups:
            continue
        group_options = dict(options.get(group_name, {}))
        if "objectless" not in group_options and config.get("dataset_mode") == "objectless":
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
