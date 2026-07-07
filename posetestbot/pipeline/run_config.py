"""Versioned run configuration for PoseTestBot pipeline jobs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

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


SCHEMA_VERSION = "run_config.v1"
CALIBRATION_PROFILE_OPTION_STAGES = ("blenderproc_prepare", "bop_export")

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

    sequence_id: str = "sync_to_bop_dry_run"
    plan_only: bool = True
    options: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PoseTestBotRunConfig:
    """Top-level versioned run configuration artifact."""

    schema_version: str
    run_name: str
    run_root: str
    robot_profile: RobotProfile
    capture: CaptureRunConfig
    object_folder: str = "object_models"
    calibration_profiles: str | None = None
    pipeline: PipelineRunConfig = field(default_factory=PipelineRunConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_name": self.run_name,
            "run_root": self.run_root,
            "robot_profile": asdict(self.robot_profile),
            "capture": self.capture.to_dict(),
            "object_folder": self.object_folder,
            "calibration_profiles": self.calibration_profiles,
            "pipeline": self.pipeline.to_dict(),
        }


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


def sensor_config_from_token(
    token: str,
    *,
    default_mounting_mode: str = MountingMode.EYE_IN_HAND.value,
) -> SensorRunConfig:
    """Parse ``sensor_type:device_id[:mounting_mode[:display_name]]``."""

    parts = token.split(":")
    if len(parts) < 2 or len(parts) > 4:
        raise ValueError(
            "Sensor entries must look like "
            "sensor_type:device_id[:mounting_mode[:display_name]]"
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
        if len(parts) == 4 and parts[3].strip()
        else f"{sensor_type.value}:{device_id}"
    )
    return SensorRunConfig(
        sensor_type=sensor_type.value,
        device_id=device_id,
        display_name=display_name,
        mounting_mode=mounting_mode.value,
    )


def sensor_config_from_mapping(
    value: Mapping[str, Any],
    *,
    default_mounting_mode: str = MountingMode.EYE_IN_HAND.value,
) -> SensorRunConfig:
    sensor_type = normalize_sensor_type(str(value.get("sensor_type", ""))).value
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
    return SensorRunConfig(
        sensor_type=sensor_type,
        device_id=device_id,
        display_name=display_name,
        mounting_mode=mounting_mode,
        enabled=bool(value.get("enabled", True)),
        calibration_profile_id=calibration_profile_id,
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
    robot_mode: str = "fake",
    resolution: str = "720p",
    fps: int = 6,
    velocity_m_s: float = 0.2,
    sensors: tuple[SensorRunConfig, ...] | None = None,
    object_folder: str = "object_models",
    calibration_profiles: str | None = None,
    sequence_id: str = "sync_to_bop_dry_run",
    sequence_options: Mapping[str, Any] | None = None,
    plan_only: bool = True,
) -> PoseTestBotRunConfig:
    run_root_path = Path(run_root)
    sensor_configs = sensors if sensors is not None else default_lab_sensors()
    config = PoseTestBotRunConfig(
        schema_version=SCHEMA_VERSION,
        run_name=run_name or run_root_path.name,
        run_root=run_root_path.as_posix(),
        robot_profile=robot_profile(robot_mode).with_overrides(
            cartesian_velocity_m_s=velocity_m_s,
        ),
        capture=CaptureRunConfig(
            resolution=resolution,
            fps=fps,
            velocity_m_s=velocity_m_s,
            sensors=tuple(sensor_configs),
        ),
        object_folder=object_folder,
        calibration_profiles=calibration_profiles,
        pipeline=PipelineRunConfig(
            sequence_id=sequence_id,
            plan_only=plan_only,
            options=dict(sequence_options or {}),
        ),
    )
    validate_run_config(config.to_dict())
    return config


def validate_run_config(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Run config schema_version must be {SCHEMA_VERSION!r}")

    capture = value.get("capture")
    if not isinstance(capture, Mapping):
        raise ValueError("Run config capture must be an object")
    if int(capture.get("fps", 0)) <= 0:
        raise ValueError("Run config capture.fps must be positive")
    if not str(capture.get("resolution", "")).strip():
        raise ValueError("Run config capture.resolution must not be empty")
    if not str(value.get("object_folder", "")).strip():
        raise ValueError("Run config object_folder must not be empty")

    sensors = capture.get("sensors")
    if not isinstance(sensors, list) or not sensors:
        raise ValueError("Run config capture.sensors must be a non-empty list")
    for index, sensor in enumerate(sensors):
        if not isinstance(sensor, Mapping):
            raise ValueError(f"Run config sensor {index} must be an object")
        normalize_sensor_type(str(sensor.get("sensor_type", "")))
        normalize_mounting_mode(str(sensor.get("mounting_mode", "")))
        if not str(sensor.get("device_id", "")).strip():
            raise ValueError(f"Run config sensor {index} device_id must not be empty")

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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


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
            f"robot mode {config.robot_profile.mode}, "
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
    validate_run_config(value)
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
    calibration_profiles = config.get("calibration_profiles")
    if not isinstance(calibration_profiles, str) or not calibration_profiles.strip():
        return options

    plan = build_sequence_plan(
        sequence_id=str(pipeline["sequence_id"]),
        run_root=str(config["run_root"]),
        options=options,
        plan_only=bool(pipeline.get("plan_only", True)),
    )
    available_groups = {step.id for step in plan.steps}
    available_groups.update(step.stage_id for step in plan.steps)
    for group_name in CALIBRATION_PROFILE_OPTION_STAGES:
        if group_name not in available_groups:
            continue
        group_options = dict(options.get(group_name, {}))
        group_options.setdefault("calibration_profiles", calibration_profiles)
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
