"""Capture command planning from versioned run configs."""

from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from posetestbot.io.atomic import atomic_write_json
from posetestbot.config import DEFAULT_ROBOT_PORT, DEFAULT_RECEIVER_PORT
from posetestbot.io.artifacts import CAPTURE_PLAN
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    make_sensor_record,
    set_manifest_sensors,
    upsert_stage,
    write_run_manifest,
)
from posetestbot.pipeline.run_config import normalize_inverted, validate_run_config
from posetestbot.sensors.registry import (
    build_sensor_capture_command,
    sensor_folder_name,
)


SCHEMA_VERSION = "capture_plan.v1"


@dataclass(frozen=True)
class CaptureCommandPlan:
    """One command an operator or orchestration layer can start."""

    role: str
    name: str
    command: list[str]
    startup_order: int
    working_directory: str = "."
    description: str = ""
    resources: tuple[str, ...] = ()
    output_folder: str | None = None
    sensor_type: str | None = None
    device_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["resources"] = list(self.resources)
        data["command_text"] = shlex.join(self.command)
        return data


@dataclass(frozen=True)
class CapturePlan:
    """Versioned capture plan artifact."""

    schema_version: str
    run_root: str
    run_config: str
    dry_run: bool
    robot_profile: Mapping[str, Any]
    capture: Mapping[str, Any]
    sensors: list[Mapping[str, Any]]
    commands: list[CaptureCommandPlan]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_root": self.run_root,
            "run_config": self.run_config,
            "dry_run": self.dry_run,
            "robot_profile": dict(self.robot_profile),
            "capture": dict(self.capture),
            "sensors": [dict(sensor) for sensor in self.sensors],
            "commands": [command.to_dict() for command in self.commands],
            "notes": list(self.notes),
        }


def _sensor_folder_name(sensor_type: str, device_id: str) -> str:
    return sensor_folder_name(sensor_type, device_id)


def _robot_int(robot: Mapping[str, Any], key: str, default: int) -> int:
    value = robot.get(key, default)
    return int(value)


def _robot_float(robot: Mapping[str, Any], key: str, default: float) -> float:
    value = robot.get(key, default)
    return float(value)


def _run_config_relative_path(run_root: Path, run_config_path: str | Path | None) -> str:
    if run_config_path is None:
        return "run_config.json"
    path = Path(run_config_path)
    try:
        return path.resolve().relative_to(run_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build_capture_plan(
    config: Mapping[str, Any],
    *,
    run_config_path: str | Path | None = None,
    max_frames: int | None = None,
    warmup_frames: int | None = None,
    robot_ip: str | None = None,
    receiver_ip: str | None = None,
    robot_port: int | None = None,
    receiver_port: int | None = None,
    fake_controller_duration_s: float | None = None,
    fake_controller_sample_ms: float | None = None,
    fake_controller_startup_delay_s: float | None = None,
) -> CapturePlan:
    """Build a non-executing capture command plan from ``run_config.json`` data."""

    validate_run_config(config)
    run_root = Path(str(config["run_root"]))
    robot = dict(config.get("robot_profile") or {})
    capture = dict(config["capture"])
    resolution = str(capture["resolution"])
    fps = int(capture["fps"])
    velocity = float(
        capture.get(
            "velocity_m_s",
            _robot_float(robot, "cartesian_velocity_m_s", 0.2),
        )
    )

    if max_frames is not None and max_frames < 0:
        raise ValueError("max_frames must be greater than or equal to 0")
    if warmup_frames is not None and warmup_frames < 0:
        raise ValueError("warmup_frames must be greater than or equal to 0")
    if fake_controller_duration_s is not None and fake_controller_duration_s < 0:
        raise ValueError(
            "fake_controller_duration_s must be greater than or equal to 0"
        )
    if fake_controller_sample_ms is not None and fake_controller_sample_ms <= 0:
        raise ValueError("fake_controller_sample_ms must be greater than 0")
    if (
        fake_controller_startup_delay_s is not None
        and fake_controller_startup_delay_s < 0
    ):
        raise ValueError(
            "fake_controller_startup_delay_s must be greater than or equal to 0"
        )

    mode = str(robot.get("mode") or "fake")
    resolved_robot_ip = str(robot_ip or robot.get("robot_ip") or "127.0.0.1")
    resolved_receiver_ip = str(
        receiver_ip or robot.get("receiver_ip") or "127.0.0.1"
    )
    command_port = int(
        robot_port
        if robot_port is not None
        else _robot_int(robot, "command_port", DEFAULT_ROBOT_PORT)
    )
    resolved_receiver_port = int(
        receiver_port
        if receiver_port is not None
        else _robot_int(robot, "receiver_port", DEFAULT_RECEIVER_PORT)
    )

    commands: list[CaptureCommandPlan] = []
    notes: list[str] = [
        "This plan is non-executing; start the commands intentionally when ready.",
        "Sensor capture commands are long-running and should be stopped after the pose receiver exits.",
    ]

    if mode == "fake":
        fake_controller_command = [
            "uv",
            "run",
            "python",
            "iiwa/fake_iiwa_controller.py",
            "--bind-ip",
            resolved_robot_ip,
            "--robot-port",
            str(command_port),
            "--receiver-ip",
            resolved_receiver_ip,
            "--receiver-port",
            str(resolved_receiver_port),
        ]
        if fake_controller_duration_s is not None:
            fake_controller_command.extend(
                ["--duration", str(fake_controller_duration_s)]
            )
        if fake_controller_sample_ms is not None:
            fake_controller_command.extend(
                ["--sample-ms", str(fake_controller_sample_ms)]
            )
        if fake_controller_startup_delay_s is not None:
            fake_controller_command.extend(
                ["--startup-delay", str(fake_controller_startup_delay_s)]
            )
        fake_controller_command.append("--once")

        commands.append(
            CaptureCommandPlan(
                role="robot_controller",
                name="fake_iiwa_controller",
                startup_order=10,
                command=fake_controller_command,
                description="Start before the pose receiver when using fake iiwa mode.",
                resources=("robot_command",),
            )
        )
    else:
        notes.append(
            "Real iiwa mode targets "
            f"{resolved_robot_ip}:{command_port}; verify the robot app is ready."
        )

    sensor_records: list[Mapping[str, Any]] = []
    enabled_sensors = [
        sensor for sensor in capture.get("sensors", []) if bool(sensor.get("enabled", True))
    ]
    for index, sensor in enumerate(enabled_sensors):
        sensor_type = str(sensor["sensor_type"])
        device_id = str(sensor["device_id"])
        inverted = normalize_inverted(sensor.get("inverted", False))
        folder_name = _sensor_folder_name(sensor_type, device_id)
        output_folder = run_root / folder_name
        command = build_sensor_capture_command(
            sensor_type=sensor_type,
            device_id=device_id,
            output_folder=output_folder.as_posix(),
            fps=fps,
            resolution=resolution,
            max_frames=max_frames,
            warmup_frames=warmup_frames,
            inverted=inverted,
        )

        sensor_records.append(
            make_sensor_record(
                sensor_type=sensor_type,
                device_id=device_id,
                folder=output_folder,
                run_root=run_root,
                display_name=str(sensor.get("display_name") or folder_name),
                mounting_mode=str(sensor.get("mounting_mode") or ""),
                status="planned",
                metadata={
                    "capture_plan_index": index,
                    "calibration_profile_id": sensor.get("calibration_profile_id"),
                    "inverted": inverted,
                    "image_rotation_degrees": 180 if inverted else 0,
                    "configured_metadata": dict(sensor.get("metadata") or {}),
                },
            )
        )
        commands.append(
            CaptureCommandPlan(
                role="sensor_capture",
                name=folder_name,
                startup_order=20,
                command=command,
                description="Start before the pose receiver to avoid missing early robot motion.",
                resources=("camera", "disk_io"),
                output_folder=output_folder.as_posix(),
                sensor_type=sensor_type,
                device_id=device_id,
            )
        )

    receiver_command = [
        "uv",
        "run",
        "python",
        "scripts/pose_receiver_udp_json.py",
        run_root.as_posix(),
        "--capture_vel",
        str(velocity),
        "--robot_mode",
        mode,
        "--ip",
        resolved_receiver_ip,
        "--port",
        str(resolved_receiver_port),
        "--ip_robot",
        resolved_robot_ip,
        "--port_robot",
        str(command_port),
    ]
    commands.append(
        CaptureCommandPlan(
            role="robot_pose_receiver",
            name="pose_receiver_udp_json",
            startup_order=30,
            command=receiver_command,
            description="Bind the receiver socket, send robot start, and record raw_robot_ee_poses.json.",
            resources=("robot_command", "disk_io"),
            output_folder=run_root.as_posix(),
        )
    )

    capture_summary = {
        "resolution": resolution,
        "fps": fps,
        "velocity_m_s": velocity,
        "sensor_count": len(capture.get("sensors", [])),
        "enabled_sensor_count": len(enabled_sensors),
        "max_frames": max_frames,
        "warmup_frames": warmup_frames,
    }

    return CapturePlan(
        schema_version=SCHEMA_VERSION,
        run_root=run_root.as_posix(),
        run_config=_run_config_relative_path(run_root, run_config_path),
        dry_run=True,
        robot_profile=robot,
        capture=capture_summary,
        sensors=sensor_records,
        commands=commands,
        notes=notes,
    )


def capture_plan_path(run_root: str | Path) -> Path:
    return Path(run_root) / CAPTURE_PLAN


def load_capture_plan(run_root: str | Path) -> dict[str, Any]:
    path = capture_plan_path(run_root)
    with open(path, "r") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Capture plan must be a JSON object: {path}")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported capture plan schema: {value.get('schema_version')!r}"
        )
    return value


def write_capture_plan(run_root: str | Path, plan: CapturePlan) -> Path:
    path = capture_plan_path(run_root)
    return atomic_write_json(path, plan.to_dict())


def write_capture_plan_with_manifest(
    run_root: str | Path,
    config: Mapping[str, Any],
    *,
    run_config_path: str | Path | None = None,
    max_frames: int | None = None,
    warmup_frames: int | None = None,
) -> tuple[Path, CapturePlan]:
    """Write ``capture_plan.json`` and record the planning stage in the manifest."""

    run_root_path = Path(run_root)
    manifest = load_or_create_run_manifest(
        run_root_path,
        robot_profile=config.get("robot_profile"),
        capture_config=config.get("capture"),
    )
    manifest["robot_profile"] = dict(config.get("robot_profile") or {})
    manifest["capture_config"] = dict(config.get("capture") or {})
    upsert_stage(manifest, name="capture_plan", status="running")
    write_run_manifest(manifest, run_root_path)

    try:
        plan = build_capture_plan(
            config,
            run_config_path=run_config_path,
            max_frames=max_frames,
            warmup_frames=warmup_frames,
        )
        path = write_capture_plan(run_root_path, plan)
        set_manifest_sensors(manifest, plan.sensors)
        upsert_stage(
            manifest,
            name="capture_plan",
            status="succeeded",
            artifacts={CAPTURE_PLAN: path},
            run_root=run_root_path,
            message=(
                "Planned "
                f"{len(plan.sensors)} sensor capture command(s) and "
                f"{len(plan.commands)} total command(s)."
            ),
        )
        write_run_manifest(manifest, run_root_path)
    except Exception as exc:
        upsert_stage(manifest, name="capture_plan", status="failed", message=str(exc))
        write_run_manifest(manifest, run_root_path)
        raise

    return path, plan
