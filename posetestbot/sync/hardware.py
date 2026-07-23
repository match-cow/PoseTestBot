"""Fail-closed grouping of hardware-synchronized RealSense depth frames.

The capture adapters own hardware configuration and record its evidence in
``frame_metadata.jsonl``.  This module does not configure or open cameras.  It
intersects already synchronized, robot-pose-matched sensor folders into
complete multiview groups while preserving every raw and per-sensor derived
frame.
"""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from posetestbot.io.atomic import atomic_write_json
from posetestbot.io.artifacts import (
    CAPTURE_EXECUTION_REPORT,
    DEPTH_DIR,
    FRAME_METADATA_JSONL,
    MATCH_ROBOT_EE_POSES,
    MULTIVIEW_FRAME_GROUPS,
    PROCESSED_DIR,
    RGB_DIR,
    RUN_CONFIG,
    SYNCHRONIZED_DIR,
)
from posetestbot.sensors.hardware_sync_qualification import (
    HardwareSyncQualificationError,
    validate_hardware_sync_qualification,
)
from posetestbot.sensors.registry import sensor_folder_name


SCHEMA_VERSION = "hardware_sync_frame_groups.v1"
CONFIG_SCHEMA_VERSION = "capture_synchronization.v1"
SUPPORTED_MODE = "hardware_trigger"
SUPPORTED_IMPLEMENTATION = "realsense_inter_cam_sync"
SUPPORTED_SCOPE = "depth_exposure"
SUPPORTED_SENSOR_TYPE = "realsense_d435"
REQUIRED_TIMESTAMP_DOMAIN = "global_time"
MAX_SUPPORTED_SKEW_MS = 5.0
CONTENT_PROVENANCE_SCHEMA_VERSION = "hardware_sync_content_provenance.v1"
RUN_CONTRACT_SCHEMA_VERSION = "hardware_sync_run_contract.v1"
CONTENT_DIGEST_ALGORITHM = "sha256"
CAPTURE_REPORT_SCHEMA_VERSION = "capture_execution_report.v1"


class HardwareSyncEvidenceError(ValueError):
    """Raised when configuration or captured evidence cannot prove hardware sync."""


@dataclass(frozen=True)
class _SensorSpec:
    sensor_key: str
    sensor_type: str
    device_id: str
    sensor_folder_name: str
    mounting_mode: str
    role: str


@dataclass(frozen=True)
class _FrameEvidence:
    sensor: _SensorSpec
    synchronized_frame_index: int
    synchronized_frame_id: str
    synchronized_rgb_path: str
    synchronized_depth_path: str
    source_frame_index: int
    source_frame_id: str
    source_rgb_path: str
    source_depth_path: str
    depth_sensor_timestamp_ns: int
    depth_frame_number: int
    matched_robot_pose: dict[str, Any]


def hardware_sync_frame_groups_path(run_root: str | Path) -> Path:
    """Return the canonical run-owned multiview frame-group artifact path."""

    return (
        Path(run_root)
        / PROCESSED_DIR
        / SYNCHRONIZED_DIR
        / MULTIVIEW_FRAME_GROUPS
    )


def _read_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise HardwareSyncEvidenceError(
            f"Invalid JSON in {path}: {exc.msg}"
        ) from exc


def _canonical_sha256(value: Any) -> str:
    """Hash one JSON value with a stable, whitespace-independent encoding."""

    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HardwareSyncEvidenceError(
            f"Cannot calculate deterministic content provenance: {exc}"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _file_attestation(path: Path, run_root: Path) -> dict[str, Any]:
    """Return deterministic content evidence for one run-owned input file."""

    if not path.is_file():
        raise FileNotFoundError(f"Content-provenance input does not exist: {path}")
    digest = hashlib.sha256()
    byte_count = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    return {
        "path": _run_relative(path, run_root),
        "size_bytes": byte_count,
        "sha256": digest.hexdigest(),
    }


def _required_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HardwareSyncEvidenceError(f"{field} must be a JSON object")
    return value


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HardwareSyncEvidenceError(f"{field} must be a non-empty string")
    return value


def _required_int(
    value: Any,
    field: str,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HardwareSyncEvidenceError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise HardwareSyncEvidenceError(f"{field} must be at least {minimum}")
    return value


def _run_relative(path: Path, run_root: Path) -> str:
    try:
        return path.resolve().relative_to(run_root.resolve()).as_posix()
    except ValueError as exc:
        raise HardwareSyncEvidenceError(
            f"Artifact path escapes the run root: {path}"
        ) from exc


def _validated_frame_file(
    *,
    base_folder: Path,
    relative_value: Any,
    expected_directory: str,
    frame_id: str,
    field: str,
) -> str:
    relative_text = _required_text(relative_value, field)
    relative = Path(relative_text)
    if relative.is_absolute():
        raise HardwareSyncEvidenceError(f"{field} must be relative")
    resolved = (base_folder / relative).resolve()
    try:
        descendant = resolved.relative_to(base_folder.resolve())
    except ValueError as exc:
        raise HardwareSyncEvidenceError(
            f"{field} escapes {base_folder}: {relative_text}"
        ) from exc
    if not descendant.parts or descendant.parts[0] != expected_directory:
        raise HardwareSyncEvidenceError(
            f"{field} must be below {expected_directory}/"
        )
    if descendant.name != frame_id:
        raise HardwareSyncEvidenceError(
            f"{field} filename {descendant.name!r} does not match "
            f"frame id {frame_id!r}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"Referenced frame file does not exist: {resolved}")
    return descendant.as_posix()


def _load_run_config(
    run_root: Path,
    supplied: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if supplied is not None:
        return _required_mapping(supplied, "run_config")
    path = run_root / RUN_CONFIG
    if not path.is_file():
        raise FileNotFoundError(f"Run configuration does not exist: {path}")
    return _required_mapping(_read_json(path), "run_config")


def _configured_sensors(
    run_config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[_SensorSpec]]:
    capture = _required_mapping(run_config.get("capture"), "run_config.capture")
    synchronization = _required_mapping(
        capture.get("synchronization"),
        "run_config.capture.synchronization",
    )
    required_values = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "mode": SUPPORTED_MODE,
        "implementation": SUPPORTED_IMPLEMENTATION,
        "scope": SUPPORTED_SCOPE,
    }
    for field, expected in required_values.items():
        actual = synchronization.get(field)
        if actual != expected:
            raise HardwareSyncEvidenceError(
                f"run_config.capture.synchronization.{field} must be "
                f"{expected!r}; got {actual!r}"
            )

    group_id = _required_text(
        synchronization.get("group_id"),
        "run_config.capture.synchronization.group_id",
    )
    master_sensor_key = _required_text(
        synchronization.get("master_sensor_key"),
        "run_config.capture.synchronization.master_sensor_key",
    )
    raw_skew_ms = synchronization.get("max_depth_timestamp_skew_ms")
    if isinstance(raw_skew_ms, bool) or not isinstance(raw_skew_ms, int | float):
        raise HardwareSyncEvidenceError(
            "run_config.capture.synchronization."
            "max_depth_timestamp_skew_ms must be a finite non-negative number"
        )
    skew_ms = float(raw_skew_ms)
    if (
        not math.isfinite(skew_ms)
        or skew_ms <= 0
        or skew_ms > MAX_SUPPORTED_SKEW_MS
    ):
        raise HardwareSyncEvidenceError(
            "run_config.capture.synchronization."
            "max_depth_timestamp_skew_ms must be a finite positive number no "
            f"greater than {MAX_SUPPORTED_SKEW_MS}"
        )
    skew_ns = int(round(skew_ms * 1_000_000))

    raw_sensors = capture.get("sensors")
    if not isinstance(raw_sensors, Sequence) or isinstance(
        raw_sensors, str | bytes
    ):
        raise HardwareSyncEvidenceError(
            "run_config.capture.sensors must be a JSON list"
        )
    enabled: list[tuple[str, str, str, str]] = []
    seen_keys: set[str] = set()
    seen_folders: set[str] = set()
    for index, raw_sensor in enumerate(raw_sensors):
        sensor = _required_mapping(
            raw_sensor, f"run_config.capture.sensors[{index}]"
        )
        enabled_value = sensor.get("enabled", True)
        if not isinstance(enabled_value, bool):
            raise HardwareSyncEvidenceError(
                f"run_config.capture.sensors[{index}].enabled must be a boolean"
            )
        if not enabled_value:
            continue
        sensor_type = _required_text(
            sensor.get("sensor_type"),
            f"run_config.capture.sensors[{index}].sensor_type",
        )
        if sensor_type != SUPPORTED_SENSOR_TYPE:
            raise HardwareSyncEvidenceError(
                f"{SUPPORTED_IMPLEMENTATION} only supports enabled "
                f"{SUPPORTED_SENSOR_TYPE} sensors; got {sensor_type!r}"
            )
        device_id = _required_text(
            sensor.get("device_id"),
            f"run_config.capture.sensors[{index}].device_id",
        )
        if (
            not device_id[0].isalnum()
            or len(device_id) > 128
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                for character in device_id
            )
        ):
            raise HardwareSyncEvidenceError(
                "hardware-trigger RealSense device IDs must be safe exact IDs"
            )
        mounting_mode = _required_text(
            sensor.get("mounting_mode"),
            f"run_config.capture.sensors[{index}].mounting_mode",
        )
        if mounting_mode not in {"eye_in_hand", "static"}:
            raise HardwareSyncEvidenceError(
                f"Unsupported mounting mode for {sensor_type}:{device_id}: "
                f"{mounting_mode!r}"
            )
        sensor_key = f"{sensor_type}:{device_id}"
        folder_name = sensor_folder_name(sensor_type, device_id)
        if sensor_key in seen_keys:
            raise HardwareSyncEvidenceError(
                f"Duplicate enabled sensor key: {sensor_key}"
            )
        if folder_name in seen_folders:
            raise HardwareSyncEvidenceError(
                f"Duplicate enabled sensor folder: {folder_name}"
            )
        seen_keys.add(sensor_key)
        seen_folders.add(folder_name)
        enabled.append((sensor_key, sensor_type, device_id, mounting_mode))

    if len(enabled) < 2:
        raise HardwareSyncEvidenceError(
            "Hardware-trigger frame grouping requires at least two enabled sensors"
        )
    if master_sensor_key not in seen_keys:
        raise HardwareSyncEvidenceError(
            "run_config.capture.synchronization.master_sensor_key must identify "
            f"one enabled sensor; got {master_sensor_key!r}"
        )

    ordered = sorted(enabled, key=lambda item: item[0] != master_sensor_key)
    specs = [
        _SensorSpec(
            sensor_key=key,
            sensor_type=sensor_type,
            device_id=device_id,
            sensor_folder_name=sensor_folder_name(sensor_type, device_id),
            mounting_mode=mounting_mode,
            role="master" if key == master_sensor_key else "subordinate",
        )
        for key, sensor_type, device_id, mounting_mode in ordered
    ]
    return (
        {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "mode": SUPPORTED_MODE,
            "implementation": SUPPORTED_IMPLEMENTATION,
            "scope": SUPPORTED_SCOPE,
            "group_id": group_id,
            "master_sensor_key": master_sensor_key,
            "max_depth_timestamp_skew_ms": skew_ms,
            "max_depth_timestamp_skew_ns": skew_ns,
        },
        specs,
    )


def _hardware_sync_run_contract(
    run_config: Mapping[str, Any],
    configuration: Mapping[str, Any],
    sensors: Sequence[_SensorSpec],
) -> dict[str, Any]:
    """Normalize the run-config fields that define this hardware capture."""

    capture = _required_mapping(run_config.get("capture"), "run_config.capture")
    resolution = _required_text(
        capture.get("resolution"), "run_config.capture.resolution"
    )
    fps = _required_int(
        capture.get("fps"),
        "run_config.capture.fps",
        minimum=1,
    )
    raw_sensors = capture.get("sensors")
    if not isinstance(raw_sensors, Sequence) or isinstance(
        raw_sensors, str | bytes
    ):
        raise HardwareSyncEvidenceError(
            "run_config.capture.sensors must be a JSON list"
        )
    raw_by_key: dict[str, Mapping[str, Any]] = {}
    for index, raw_sensor in enumerate(raw_sensors):
        sensor = _required_mapping(
            raw_sensor, f"run_config.capture.sensors[{index}]"
        )
        if sensor.get("enabled", True) is False:
            continue
        sensor_type = _required_text(
            sensor.get("sensor_type"),
            f"run_config.capture.sensors[{index}].sensor_type",
        )
        device_id = _required_text(
            sensor.get("device_id"),
            f"run_config.capture.sensors[{index}].device_id",
        )
        raw_by_key[f"{sensor_type}:{device_id}"] = sensor

    contract_sensors: list[dict[str, Any]] = []
    for sensor in sensors:
        raw_sensor = raw_by_key.get(sensor.sensor_key)
        if raw_sensor is None:
            raise HardwareSyncEvidenceError(
                f"Hardware-sync sensor is absent from run config: {sensor.sensor_key}"
            )
        inverted = raw_sensor.get("inverted", False)
        if not isinstance(inverted, bool):
            raise HardwareSyncEvidenceError(
                f"run_config sensor {sensor.sensor_key} inverted must be a boolean"
            )
        contract_sensors.append(
            {
                "sensor_key": sensor.sensor_key,
                "sensor_type": sensor.sensor_type,
                "device_id": sensor.device_id,
                "mounting_mode": sensor.mounting_mode,
                "hardware_sync_role": sensor.role,
                "inverted": inverted,
            }
        )

    return {
        "schema_version": RUN_CONTRACT_SCHEMA_VERSION,
        "run_config_schema_version": _required_text(
            run_config.get("schema_version"), "run_config.schema_version"
        ),
        "capture": {
            "resolution": resolution,
            "fps": fps,
            "synchronization": {
                "schema_version": configuration["schema_version"],
                "mode": configuration["mode"],
                "implementation": configuration["implementation"],
                "scope": configuration["scope"],
                "group_id": configuration["group_id"],
                "master_sensor_key": configuration["master_sensor_key"],
                "max_depth_timestamp_skew_ms": configuration[
                    "max_depth_timestamp_skew_ms"
                ],
            },
            "sensors": contract_sensors,
        },
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Frame metadata does not exist: {path}")
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HardwareSyncEvidenceError(
                    f"Invalid JSON in {path} line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise HardwareSyncEvidenceError(
                    f"{path} line {line_number} must be a JSON object"
                )
            records.append(value)
    if not records:
        raise HardwareSyncEvidenceError(f"Frame metadata is empty: {path}")
    return records


def _matched_robot_pose(
    *,
    matched: Mapping[str, Any],
    record: Mapping[str, Any],
    frame_id: str,
    sensor_key: str,
) -> dict[str, Any]:
    raw = _required_mapping(
        matched.get(frame_id),
        f"{sensor_key} matched robot pose for {frame_id}",
    )
    metadata_index = _required_int(
        record.get("matched_robot_pose_index"),
        f"{sensor_key} frame {frame_id} matched_robot_pose_index",
        minimum=0,
    )
    pose_index = _required_int(
        raw.get("matched_robot_pose_index"),
        f"{sensor_key} matched pose {frame_id}.matched_robot_pose_index",
        minimum=0,
    )
    if pose_index != metadata_index:
        raise HardwareSyncEvidenceError(
            f"{sensor_key} frame {frame_id} matched robot pose index differs "
            "between frame metadata and match_robot_ee_poses.json"
        )
    if raw.get("source_frame_id") != record.get("source_frame_id"):
        raise HardwareSyncEvidenceError(
            f"{sensor_key} frame {frame_id} source_frame_id differs between "
            "frame metadata and match_robot_ee_poses.json"
        )
    robot_ee_pose = _required_mapping(
        raw.get("robot_ee_pose"),
        f"{sensor_key} matched pose {frame_id}.robot_ee_pose",
    )
    if not robot_ee_pose:
        raise HardwareSyncEvidenceError(
            f"{sensor_key} matched pose {frame_id}.robot_ee_pose must not be empty"
        )
    motion = _required_text(
        raw.get("motion"),
        f"{sensor_key} matched pose {frame_id}.motion",
    )
    robot_timestamp_ns = _required_int(
        raw.get("robot_timestamp_ns"),
        f"{sensor_key} matched pose {frame_id}.robot_timestamp_ns",
        minimum=0,
    )
    nearest_delta_ns = _required_int(
        raw.get("nearest_robot_delta_ns"),
        f"{sensor_key} matched pose {frame_id}.nearest_robot_delta_ns",
    )
    return {
        "matched_robot_pose_index": pose_index,
        "robot_timestamp_ns": robot_timestamp_ns,
        "nearest_robot_delta_ns": nearest_delta_ns,
        "motion": motion,
        "robot_ee_pose": dict(robot_ee_pose),
    }


def _load_sensor_frames(
    *,
    run_root: Path,
    sensor: _SensorSpec,
    configuration: Mapping[str, Any],
) -> tuple[list[_FrameEvidence], str, str]:
    synchronized_folder = (
        run_root
        / PROCESSED_DIR
        / SYNCHRONIZED_DIR
        / sensor.sensor_folder_name
    )
    metadata_path = synchronized_folder / FRAME_METADATA_JSONL
    matched_path = synchronized_folder / MATCH_ROBOT_EE_POSES
    if not matched_path.is_file():
        raise FileNotFoundError(
            f"Matched robot-pose artifact does not exist: {matched_path}"
        )
    raw_matched = _required_mapping(
        _read_json(matched_path),
        f"{sensor.sensor_key} {MATCH_ROBOT_EE_POSES}",
    )
    raw_folder = run_root / sensor.sensor_folder_name
    expected_option = 1 if sensor.role == "master" else 2
    records = _load_jsonl(metadata_path)
    frames: list[_FrameEvidence] = []
    seen_frame_ids: set[str] = set()
    seen_frame_indexes: set[int] = set()
    seen_source_frame_ids: set[str] = set()

    for record_number, record in enumerate(records, start=1):
        prefix = f"{sensor.sensor_key} metadata record {record_number}"
        if record.get("schema_version") != "frame_metadata.v1":
            raise HardwareSyncEvidenceError(
                f"{prefix} schema_version must be 'frame_metadata.v1'"
            )
        if record.get("sensor_type") != sensor.sensor_type:
            raise HardwareSyncEvidenceError(
                f"{prefix} sensor_type does not match run configuration"
            )
        if record.get("sensor_id") != sensor.device_id:
            raise HardwareSyncEvidenceError(
                f"{prefix} sensor_id does not match run configuration"
            )
        required_evidence = {
            "capture_group_id": configuration["group_id"],
            "hardware_sync_role": sensor.role,
            "hardware_sync_scope": SUPPORTED_SCOPE,
            "hardware_sync_transport": SUPPORTED_IMPLEMENTATION,
            "inter_cam_sync_mode_configured": expected_option,
            "inter_cam_sync_mode_readback": expected_option,
            "depth_timestamp_domain": REQUIRED_TIMESTAMP_DOMAIN,
        }
        for field, expected in required_evidence.items():
            actual = record.get(field)
            if actual != expected:
                raise HardwareSyncEvidenceError(
                    f"{prefix} {field} must be {expected!r}; got {actual!r}"
                )

        synchronized_frame_index = _required_int(
            record.get("frame_index"),
            f"{prefix} frame_index",
            minimum=0,
        )
        synchronized_frame_id = _required_text(
            record.get("frame_id"), f"{prefix} frame_id"
        )
        source_frame_index = _required_int(
            record.get("source_frame_index"),
            f"{prefix} source_frame_index",
            minimum=0,
        )
        source_frame_id = _required_text(
            record.get("source_frame_id"), f"{prefix} source_frame_id"
        )
        depth_sensor_timestamp_ns = _required_int(
            record.get("depth_sensor_timestamp_ns"),
            f"{prefix} depth_sensor_timestamp_ns",
            minimum=0,
        )
        depth_frame_number = _required_int(
            record.get("depth_frame_number"),
            f"{prefix} depth_frame_number",
            minimum=0,
        )
        if synchronized_frame_id in seen_frame_ids:
            raise HardwareSyncEvidenceError(
                f"Duplicate synchronized frame_id for {sensor.sensor_key}: "
                f"{synchronized_frame_id}"
            )
        if synchronized_frame_index in seen_frame_indexes:
            raise HardwareSyncEvidenceError(
                f"Duplicate synchronized frame_index for {sensor.sensor_key}: "
                f"{synchronized_frame_index}"
            )
        if source_frame_id in seen_source_frame_ids:
            raise HardwareSyncEvidenceError(
                f"Duplicate source_frame_id for {sensor.sensor_key}: {source_frame_id}"
            )
        seen_frame_ids.add(synchronized_frame_id)
        seen_frame_indexes.add(synchronized_frame_index)
        seen_source_frame_ids.add(source_frame_id)

        synchronized_rgb_path = _validated_frame_file(
            base_folder=synchronized_folder,
            relative_value=record.get("rgb_path"),
            expected_directory=RGB_DIR,
            frame_id=synchronized_frame_id,
            field=f"{prefix} rgb_path",
        )
        synchronized_depth_path = _validated_frame_file(
            base_folder=synchronized_folder,
            relative_value=record.get("depth_path"),
            expected_directory=DEPTH_DIR,
            frame_id=synchronized_frame_id,
            field=f"{prefix} depth_path",
        )
        source_rgb_path = _validated_frame_file(
            base_folder=raw_folder,
            relative_value=record.get("source_rgb_path"),
            expected_directory=RGB_DIR,
            frame_id=source_frame_id,
            field=f"{prefix} source_rgb_path",
        )
        source_depth_path = _validated_frame_file(
            base_folder=raw_folder,
            relative_value=record.get("source_depth_path"),
            expected_directory=DEPTH_DIR,
            frame_id=source_frame_id,
            field=f"{prefix} source_depth_path",
        )
        frames.append(
            _FrameEvidence(
                sensor=sensor,
                synchronized_frame_index=synchronized_frame_index,
                synchronized_frame_id=synchronized_frame_id,
                synchronized_rgb_path=synchronized_rgb_path,
                synchronized_depth_path=synchronized_depth_path,
                source_frame_index=source_frame_index,
                source_frame_id=source_frame_id,
                source_rgb_path=source_rgb_path,
                source_depth_path=source_depth_path,
                depth_sensor_timestamp_ns=depth_sensor_timestamp_ns,
                depth_frame_number=depth_frame_number,
                matched_robot_pose=_matched_robot_pose(
                    matched=raw_matched,
                    record=record,
                    frame_id=synchronized_frame_id,
                    sensor_key=sensor.sensor_key,
                ),
            )
        )

    frames.sort(
        key=lambda item: (
            item.depth_sensor_timestamp_ns,
            item.synchronized_frame_index,
            item.synchronized_frame_id,
        )
    )
    return (
        frames,
        _run_relative(metadata_path, run_root),
        _run_relative(matched_path, run_root),
    )


def _content_provenance(
    *,
    run_root: Path,
    run_config: Mapping[str, Any],
    configuration: Mapping[str, Any],
    sensors: Sequence[_SensorSpec],
    frames_by_sensor: Mapping[str, Sequence[_FrameEvidence]],
    metadata_paths: Mapping[str, str],
    matched_pose_paths: Mapping[str, str],
) -> dict[str, Any]:
    """Bind a group artifact to the exact configuration and input bytes."""

    contract = _hardware_sync_run_contract(
        run_config,
        configuration,
        sensors,
    )
    hardware_contract = {
        "schema_version": RUN_CONTRACT_SCHEMA_VERSION,
        "path": RUN_CONFIG,
        "sha256": _canonical_sha256(contract),
    }
    sensor_provenance: list[dict[str, Any]] = []
    for sensor in sensors:
        synchronized_folder = (
            run_root
            / PROCESSED_DIR
            / SYNCHRONIZED_DIR
            / sensor.sensor_folder_name
        )
        source_folder = run_root / sensor.sensor_folder_name
        referenced: dict[str, Path] = {}
        for frame in frames_by_sensor[sensor.sensor_key]:
            for path in (
                synchronized_folder / frame.synchronized_rgb_path,
                synchronized_folder / frame.synchronized_depth_path,
                source_folder / frame.source_rgb_path,
                source_folder / frame.source_depth_path,
            ):
                referenced[_run_relative(path, run_root)] = path
        frame_manifest = [
            _file_attestation(referenced[path], run_root)
            for path in sorted(referenced)
        ]
        base_sensor_provenance = {
            "sensor_key": sensor.sensor_key,
            "frame_metadata": _file_attestation(
                run_root / metadata_paths[sensor.sensor_key],
                run_root,
            ),
            "matched_robot_poses": _file_attestation(
                run_root / matched_pose_paths[sensor.sensor_key],
                run_root,
            ),
            "referenced_frames": {
                "file_count": len(frame_manifest),
                "total_size_bytes": sum(
                    int(item["size_bytes"]) for item in frame_manifest
                ),
                "manifest_sha256": _canonical_sha256(frame_manifest),
            },
        }
        sensor_provenance.append(
            {
                **base_sensor_provenance,
                "content_sha256": _canonical_sha256(base_sensor_provenance),
            }
        )

    base_provenance = {
        "schema_version": CONTENT_PROVENANCE_SCHEMA_VERSION,
        "digest_algorithm": CONTENT_DIGEST_ALGORITHM,
        "hardware_contract": hardware_contract,
        "sensors": sensor_provenance,
    }
    return {
        **base_provenance,
        "aggregate_sha256": _canonical_sha256(base_provenance),
    }


def _counter_discontinuities(
    frames: Sequence[_FrameEvidence],
) -> list[dict[str, Any]]:
    ordered = sorted(
        frames,
        key=lambda item: (
            item.synchronized_frame_index,
            item.depth_sensor_timestamp_ns,
        ),
    )
    discontinuities: list[dict[str, Any]] = []
    for previous, current in zip(ordered, ordered[1:]):
        delta = current.depth_frame_number - previous.depth_frame_number
        if delta == 1:
            continue
        kind = "gap" if delta > 1 else "duplicate" if delta == 0 else "regression"
        discontinuities.append(
            {
                "kind": kind,
                "previous_synchronized_frame_id": previous.synchronized_frame_id,
                "synchronized_frame_id": current.synchronized_frame_id,
                "previous_depth_frame_number": previous.depth_frame_number,
                "depth_frame_number": current.depth_frame_number,
                "counter_delta": delta,
                "missing_frame_count": max(delta - 1, 0),
                "previous_depth_sensor_timestamp_ns": (
                    previous.depth_sensor_timestamp_ns
                ),
                "depth_sensor_timestamp_ns": current.depth_sensor_timestamp_ns,
            }
        )
    return discontinuities


def _match_subordinate_to_master(
    master_frames: Sequence[_FrameEvidence],
    subordinate_frames: Sequence[_FrameEvidence],
    max_skew_ns: int,
) -> tuple[dict[int, _FrameEvidence], list[_FrameEvidence]]:
    """Return a chronological maximum-cardinality, one-to-one association."""

    matched: dict[int, _FrameEvidence] = {}
    unmatched: list[_FrameEvidence] = []
    master_index = 0
    subordinate_index = 0
    while master_index < len(master_frames) and subordinate_index < len(
        subordinate_frames
    ):
        master_timestamp = master_frames[master_index].depth_sensor_timestamp_ns
        subordinate_timestamp = (
            subordinate_frames[subordinate_index].depth_sensor_timestamp_ns
        )
        if subordinate_timestamp < master_timestamp - max_skew_ns:
            unmatched.append(subordinate_frames[subordinate_index])
            subordinate_index += 1
            continue
        if master_timestamp < subordinate_timestamp - max_skew_ns:
            master_index += 1
            continue
        matched[master_index] = subordinate_frames[subordinate_index]
        master_index += 1
        subordinate_index += 1
    unmatched.extend(subordinate_frames[subordinate_index:])
    return matched, unmatched


def _frame_reference(
    frame: _FrameEvidence,
    *,
    master_timestamp_ns: int | None,
) -> dict[str, Any]:
    skew_ns = (
        frame.depth_sensor_timestamp_ns - master_timestamp_ns
        if master_timestamp_ns is not None
        else None
    )
    return {
        "sensor_key": frame.sensor.sensor_key,
        "sensor_folder": (
            f"{PROCESSED_DIR}/{SYNCHRONIZED_DIR}/"
            f"{frame.sensor.sensor_folder_name}"
        ),
        "mounting_mode": frame.sensor.mounting_mode,
        "hardware_sync_role": frame.sensor.role,
        "synchronized_frame_index": frame.synchronized_frame_index,
        "synchronized_frame_id": frame.synchronized_frame_id,
        "synchronized_rgb_path": frame.synchronized_rgb_path,
        "synchronized_depth_path": frame.synchronized_depth_path,
        "source_frame_index": frame.source_frame_index,
        "source_frame_id": frame.source_frame_id,
        "source_sensor_folder": frame.sensor.sensor_folder_name,
        "source_rgb_path": frame.source_rgb_path,
        "source_depth_path": frame.source_depth_path,
        "depth_sensor_timestamp_ns": frame.depth_sensor_timestamp_ns,
        "depth_frame_number": frame.depth_frame_number,
        "depth_timestamp_domain": REQUIRED_TIMESTAMP_DOMAIN,
        "depth_timestamp_skew_ns": skew_ns,
        "abs_depth_timestamp_skew_ns": (
            abs(skew_ns) if skew_ns is not None else None
        ),
        "matched_robot_pose": dict(frame.matched_robot_pose),
    }


def _nearest_master_evidence(
    frame: _FrameEvidence,
    master_frames: Sequence[_FrameEvidence],
) -> dict[str, Any]:
    timestamps = [item.depth_sensor_timestamp_ns for item in master_frames]
    insertion = bisect_left(timestamps, frame.depth_sensor_timestamp_ns)
    candidates = [
        index
        for index in (insertion - 1, insertion)
        if 0 <= index < len(master_frames)
    ]
    nearest_index = min(
        candidates,
        key=lambda index: (
            abs(frame.depth_sensor_timestamp_ns - timestamps[index]),
            index,
        ),
    )
    nearest = master_frames[nearest_index]
    return {
        "reason": "no_master_within_max_depth_timestamp_skew",
        "nearest_master_ordinal": nearest_index,
        "nearest_master_synchronized_frame_id": nearest.synchronized_frame_id,
        "nearest_master_depth_sensor_timestamp_ns": (
            nearest.depth_sensor_timestamp_ns
        ),
        "nearest_master_depth_timestamp_skew_ns": (
            frame.depth_sensor_timestamp_ns - nearest.depth_sensor_timestamp_ns
        ),
    }


def _skew_statistics(values: Sequence[int]) -> dict[str, float | int | None]:
    absolute = [abs(value) for value in values]
    return {
        "matched_frame_count": len(values),
        "maximum_abs_depth_timestamp_skew_ns": max(absolute) if absolute else None,
        "mean_abs_depth_timestamp_skew_ns": (
            fmean(absolute) if absolute else None
        ),
        "minimum_depth_timestamp_skew_ns": min(values) if values else None,
        "maximum_depth_timestamp_skew_ns": max(values) if values else None,
    }


def _frame_group_timestamp_span_ns(
    frames: Mapping[str, Mapping[str, Any]],
) -> int:
    timestamps = [
        int(frame["depth_sensor_timestamp_ns"])
        for frame in frames.values()
    ]
    return max(timestamps) - min(timestamps)


def build_hardware_sync_frame_groups(
    run_root: str | Path,
    *,
    run_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build complete RealSense hardware-trigger frame groups without writing files."""

    root = Path(run_root)
    loaded_run_config = _load_run_config(root, run_config)
    configuration, sensors = _configured_sensors(loaded_run_config)
    frames_by_sensor: dict[str, list[_FrameEvidence]] = {}
    metadata_paths: dict[str, str] = {}
    matched_pose_paths: dict[str, str] = {}
    discontinuities: dict[str, list[dict[str, Any]]] = {}
    for sensor in sensors:
        frames, metadata_path, matched_path = _load_sensor_frames(
            run_root=root,
            sensor=sensor,
            configuration=configuration,
        )
        frames_by_sensor[sensor.sensor_key] = frames
        metadata_paths[sensor.sensor_key] = metadata_path
        matched_pose_paths[sensor.sensor_key] = matched_path
        discontinuities[sensor.sensor_key] = _counter_discontinuities(frames)

    master_sensor = sensors[0]
    master_frames = frames_by_sensor[master_sensor.sensor_key]
    max_skew_ns = int(configuration["max_depth_timestamp_skew_ns"])
    matches: dict[str, dict[int, _FrameEvidence]] = {}
    unmatched: dict[str, list[_FrameEvidence]] = {}
    for sensor in sensors[1:]:
        sensor_matches, sensor_unmatched = _match_subordinate_to_master(
            master_frames,
            frames_by_sensor[sensor.sensor_key],
            max_skew_ns,
        )
        matches[sensor.sensor_key] = sensor_matches
        unmatched[sensor.sensor_key] = sensor_unmatched

    complete_groups: list[dict[str, Any]] = []
    incomplete_groups: list[dict[str, Any]] = []
    complete_master_ordinals: set[int] = set()
    for master_ordinal, master_frame in enumerate(master_frames):
        associated: dict[str, _FrameEvidence] = {
            master_sensor.sensor_key: master_frame
        }
        missing: list[str] = []
        for sensor in sensors[1:]:
            matched = matches[sensor.sensor_key].get(master_ordinal)
            if matched is None:
                missing.append(sensor.sensor_key)
            else:
                associated[sensor.sensor_key] = matched
        frame_references = {
            sensor.sensor_key: _frame_reference(
                associated[sensor.sensor_key],
                master_timestamp_ns=master_frame.depth_sensor_timestamp_ns,
            )
            for sensor in sensors
            if sensor.sensor_key in associated
        }
        candidate_id = (
            f"{configuration['group_id']}:{master_ordinal:06d}"
        )
        observed_group_skews = [
            int(reference["abs_depth_timestamp_skew_ns"])
            for reference in frame_references.values()
        ]
        timestamp_span_ns = _frame_group_timestamp_span_ns(frame_references)
        span_exceeds_limit = (
            not missing and timestamp_span_ns > max_skew_ns
        )
        if missing or span_exceeds_limit:
            reason = (
                "missing_configured_sensor_frames"
                if missing
                else "depth_timestamp_span_exceeds_configured_maximum"
            )
            incomplete_groups.append(
                {
                    "frame_group_id": candidate_id,
                    "master_frame_ordinal": master_ordinal,
                    "capture_group_id": configuration["group_id"],
                    "master_sensor_key": master_sensor.sensor_key,
                    "depth_sensor_timestamp_ns": (
                        master_frame.depth_sensor_timestamp_ns
                    ),
                    "matched_sensor_keys": list(frame_references),
                    "missing_sensor_keys": missing,
                    "reason": reason,
                    "max_abs_depth_timestamp_skew_ns": max(
                        observed_group_skews
                    ),
                    "depth_timestamp_span_ns": timestamp_span_ns,
                    "frames": frame_references,
                }
            )
            continue
        complete_index = len(complete_groups)
        complete_master_ordinals.add(master_ordinal)
        complete_groups.append(
            {
                "frame_group_id": candidate_id,
                "frame_group_index": complete_index,
                "master_frame_ordinal": master_ordinal,
                "capture_group_id": configuration["group_id"],
                "master_sensor_key": master_sensor.sensor_key,
                "depth_sensor_timestamp_ns": (
                    master_frame.depth_sensor_timestamp_ns
                ),
                "matched_robot_pose": dict(master_frame.matched_robot_pose),
                "max_abs_depth_timestamp_skew_ns": max(observed_group_skews),
                "depth_timestamp_span_ns": timestamp_span_ns,
                "frames": frame_references,
            }
        )

    if not complete_groups:
        raise HardwareSyncEvidenceError(
            "Hardware synchronization evidence contains no complete multiview "
            "frame group within max_depth_timestamp_skew_ms"
        )

    unmatched_payload: dict[str, list[dict[str, Any]]] = {}
    for sensor in sensors[1:]:
        values = []
        for frame in unmatched[sensor.sensor_key]:
            values.append(
                {
                    **_frame_reference(frame, master_timestamp_ns=None),
                    **_nearest_master_evidence(frame, master_frames),
                }
            )
        unmatched_payload[sensor.sensor_key] = values

    per_sensor_skews: dict[str, dict[str, float | int | None]] = {
        master_sensor.sensor_key: _skew_statistics([0] * len(master_frames))
    }
    all_skews: list[int] = []
    for sensor in sensors[1:]:
        sensor_skews = [
            frame.depth_sensor_timestamp_ns
            - master_frames[master_ordinal].depth_sensor_timestamp_ns
            for master_ordinal, frame in sorted(matches[sensor.sensor_key].items())
        ]
        all_skews.extend(sensor_skews)
        per_sensor_skews[sensor.sensor_key] = _skew_statistics(sensor_skews)
    complete_skews = [
        int(reference["depth_timestamp_skew_ns"])
        for group in complete_groups
        for key, reference in group["frames"].items()
        if key != master_sensor.sensor_key
    ]

    mounting_modes: dict[str, list[str]] = {}
    for sensor in sensors:
        mounting_modes.setdefault(sensor.mounting_mode, []).append(sensor.sensor_key)

    sensor_payload: list[dict[str, Any]] = []
    for sensor in sensors:
        sensor_matches = (
            len(master_frames)
            if sensor.role == "master"
            else len(matches[sensor.sensor_key])
        )
        sensor_payload.append(
            {
                "sensor_key": sensor.sensor_key,
                "sensor_type": sensor.sensor_type,
                "device_id": sensor.device_id,
                "sensor_folder": (
                    f"{PROCESSED_DIR}/{SYNCHRONIZED_DIR}/"
                    f"{sensor.sensor_folder_name}"
                ),
                "mounting_mode": sensor.mounting_mode,
                "hardware_sync_role": sensor.role,
                "frame_metadata_path": metadata_paths[sensor.sensor_key],
                "matched_robot_poses_path": matched_pose_paths[sensor.sensor_key],
                "frame_count": len(frames_by_sensor[sensor.sensor_key]),
                "matched_to_master_frame_count": sensor_matches,
                "complete_group_frame_count": len(complete_groups),
                "incomplete_group_frame_count": (
                    len(incomplete_groups)
                    if sensor.role == "master"
                    else len(
                        set(matches[sensor.sensor_key])
                        - complete_master_ordinals
                    )
                ),
                "unmatched_frame_count": (
                    0
                    if sensor.role == "master"
                    else len(unmatched[sensor.sensor_key])
                ),
            }
        )

    total_counter_discontinuities = sum(
        len(values) for values in discontinuities.values()
    )
    total_unmatched = sum(len(values) for values in unmatched.values())
    overall_skew = _skew_statistics(all_skews)
    complete_skew = _skew_statistics(complete_skews)
    content_provenance = _content_provenance(
        run_root=root,
        run_config=loaded_run_config,
        configuration=configuration,
        sensors=sensors,
        frames_by_sensor=frames_by_sensor,
        metadata_paths=metadata_paths,
        matched_pose_paths=matched_pose_paths,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_config_path": RUN_CONFIG,
        "content_provenance": content_provenance,
        "group_id": configuration["group_id"],
        "mode": configuration["mode"],
        "implementation": configuration["implementation"],
        "scope": configuration["scope"],
        "master_sensor_key": configuration["master_sensor_key"],
        "max_depth_timestamp_skew_ms": (
            configuration["max_depth_timestamp_skew_ms"]
        ),
        "max_depth_timestamp_skew_ns": max_skew_ns,
        "sensor_order": [sensor.sensor_key for sensor in sensors],
        "sensors": sensor_payload,
        "mounting_modes": mounting_modes,
        "summary": {
            "sensor_count": len(sensors),
            "master_frame_count": len(master_frames),
            "complete_group_count": len(complete_groups),
            "incomplete_master_group_count": len(incomplete_groups),
            "unmatched_subordinate_frame_count": total_unmatched,
            "counter_discontinuity_count": total_counter_discontinuities,
        },
        "observed_skew": {
            "matched_subordinate_frame_count": overall_skew["matched_frame_count"],
            "maximum_abs_depth_timestamp_skew_ns": overall_skew[
                "maximum_abs_depth_timestamp_skew_ns"
            ],
            "mean_abs_depth_timestamp_skew_ns": overall_skew[
                "mean_abs_depth_timestamp_skew_ns"
            ],
            "complete_groups_maximum_abs_depth_timestamp_skew_ns": (
                complete_skew["maximum_abs_depth_timestamp_skew_ns"]
            ),
            "complete_groups_mean_abs_depth_timestamp_skew_ns": complete_skew[
                "mean_abs_depth_timestamp_skew_ns"
            ],
            "per_sensor": per_sensor_skews,
        },
        "counter_discontinuities": discontinuities,
        "incomplete_master_groups": incomplete_groups,
        "unmatched_subordinate_frames": unmatched_payload,
        "groups": complete_groups,
    }


def _validate_payload_folder(
    path_value: Any,
    run_root: Path,
    field: str,
) -> Path:
    relative_text = _required_text(path_value, field)
    relative = Path(relative_text)
    if relative.is_absolute():
        raise HardwareSyncEvidenceError(f"{field} must be run-relative")
    resolved = (run_root / relative).resolve()
    try:
        resolved.relative_to(run_root.resolve())
    except ValueError as exc:
        raise HardwareSyncEvidenceError(f"{field} escapes the run root") from exc
    if not resolved.is_dir():
        raise FileNotFoundError(f"Referenced sensor folder does not exist: {resolved}")
    return resolved


def _validate_payload_file(
    path_value: Any,
    *,
    base_folder: Path,
    expected_directory: str,
    field: str,
) -> None:
    relative_text = _required_text(path_value, field)
    relative = Path(relative_text)
    if relative.is_absolute():
        raise HardwareSyncEvidenceError(f"{field} must be sensor-folder-relative")
    resolved = (base_folder / relative).resolve()
    try:
        descendant = resolved.relative_to(base_folder.resolve())
    except ValueError as exc:
        raise HardwareSyncEvidenceError(
            f"{field} escapes its sensor folder"
        ) from exc
    if not descendant.parts or descendant.parts[0] != expected_directory:
        raise HardwareSyncEvidenceError(
            f"{field} must be below {expected_directory}/"
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"Referenced frame file does not exist: {resolved}")


def _required_sha256(value: Any, field: str) -> str:
    digest = _required_text(value, field)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise HardwareSyncEvidenceError(
            f"{field} must be a lowercase SHA-256 digest"
        )
    return digest


def _validate_hardware_sync_execution_binding_shape(
    value: Any,
) -> dict[str, Any]:
    binding = _required_mapping(
        value,
        "hardware_sync_execution_binding",
    )
    expected_fields = {
        "configuration_sha256",
        "qualification_artifact_sha256",
        "revalidated_immediately_before_receiver_spawn",
    }
    if set(binding) != expected_fields:
        raise HardwareSyncEvidenceError(
            "hardware_sync_execution_binding contains missing or unknown fields"
        )
    if binding.get("revalidated_immediately_before_receiver_spawn") is not True:
        raise HardwareSyncEvidenceError(
            "hardware_sync_execution_binding must prove immediate pre-receiver "
            "revalidation"
        )
    return {
        "configuration_sha256": _required_sha256(
            binding.get("configuration_sha256"),
            "hardware_sync_execution_binding.configuration_sha256",
        ),
        "qualification_artifact_sha256": _required_sha256(
            binding.get("qualification_artifact_sha256"),
            "hardware_sync_execution_binding.qualification_artifact_sha256",
        ),
        "revalidated_immediately_before_receiver_spawn": True,
    }


def capture_execution_hardware_sync_binding(
    run_root: str | Path,
    *,
    qualification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the successful capture report's immutable sync binding."""

    root = Path(run_root)
    current_qualification = validate_hardware_sync_qualification(root)
    if (
        qualification is not None
        and dict(qualification) != current_qualification
    ):
        raise HardwareSyncEvidenceError(
            "Supplied hardware-sync qualification does not exactly match the "
            "current validated qualification"
        )
    report_path = root / CAPTURE_EXECUTION_REPORT
    if report_path.is_symlink():
        raise HardwareSyncEvidenceError(
            "Capture execution report must not be a symbolic link"
        )
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Capture execution report does not exist: {report_path}"
        )
    report = _required_mapping(
        _read_json(report_path),
        "capture execution report",
    )
    if report.get("schema_version") != CAPTURE_REPORT_SCHEMA_VERSION:
        raise HardwareSyncEvidenceError(
            "Capture execution report schema_version must be "
            f"{CAPTURE_REPORT_SCHEMA_VERSION!r}"
        )
    if report.get("status") != "succeeded":
        raise HardwareSyncEvidenceError(
            "Authoritative hardware-sync groups require a succeeded capture "
            "execution report"
        )
    if report.get("mode") != "full":
        raise HardwareSyncEvidenceError(
            "Authoritative hardware-sync groups require a full capture report"
        )
    if (
        report.get("allow_cameras") is not True
        or report.get("allow_real_robot") is not True
    ):
        raise HardwareSyncEvidenceError(
            "Succeeded hardware-sync capture report must record both execution "
            "safety gates"
        )
    report_run_root = _required_text(
        report.get("run_root"),
        "capture_execution_report.run_root",
    )
    if Path(report_run_root).resolve() != root.resolve():
        raise HardwareSyncEvidenceError(
            "Capture execution report run_root does not match this run"
        )
    binding = _validate_hardware_sync_execution_binding_shape(
        report.get("hardware_sync_execution_binding")
    )
    if (
        binding["configuration_sha256"]
        != current_qualification.get("configuration_sha256")
    ):
        raise HardwareSyncEvidenceError(
            "Succeeded capture execution report hardware-sync configuration "
            "digest does not match the current qualification"
        )
    if (
        binding["qualification_artifact_sha256"]
        != current_qualification.get("artifact_sha256")
    ):
        raise HardwareSyncEvidenceError(
            "Succeeded capture execution report qualification hash does not "
            "match the current qualification"
        )
    return binding


def _validate_file_attestation(
    value: Any,
    *,
    field: str,
    expected_path: str,
) -> dict[str, Any]:
    attestation = _required_mapping(value, field)
    if set(attestation) != {"path", "size_bytes", "sha256"}:
        raise HardwareSyncEvidenceError(
            f"{field} must contain only path, size_bytes, and sha256"
        )
    path = _required_text(attestation.get("path"), f"{field}.path")
    if path != expected_path:
        raise HardwareSyncEvidenceError(
            f"{field}.path must be {expected_path!r}; got {path!r}"
        )
    size_bytes = _required_int(
        attestation.get("size_bytes"),
        f"{field}.size_bytes",
        minimum=0,
    )
    sha256 = _required_sha256(attestation.get("sha256"), f"{field}.sha256")
    return {
        "path": path,
        "size_bytes": size_bytes,
        "sha256": sha256,
    }


def _validate_content_provenance_shape(
    value: Any,
    *,
    sensor_order: Sequence[str],
    sensor_inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    provenance = _required_mapping(value, "content_provenance")
    required_fields = {
        "schema_version",
        "digest_algorithm",
        "hardware_contract",
        "sensors",
        "aggregate_sha256",
    }
    if set(provenance) != required_fields:
        raise HardwareSyncEvidenceError(
            "content_provenance contains missing or unknown fields"
        )
    if provenance.get("schema_version") != CONTENT_PROVENANCE_SCHEMA_VERSION:
        raise HardwareSyncEvidenceError(
            "content_provenance.schema_version must be "
            f"{CONTENT_PROVENANCE_SCHEMA_VERSION!r}"
        )
    if provenance.get("digest_algorithm") != CONTENT_DIGEST_ALGORITHM:
        raise HardwareSyncEvidenceError(
            "content_provenance.digest_algorithm must be 'sha256'"
        )
    hardware_contract_value = _required_mapping(
        provenance.get("hardware_contract"),
        "content_provenance.hardware_contract",
    )
    if set(hardware_contract_value) != {"schema_version", "path", "sha256"}:
        raise HardwareSyncEvidenceError(
            "content_provenance.hardware_contract contains invalid fields"
        )
    if (
        hardware_contract_value.get("schema_version")
        != RUN_CONTRACT_SCHEMA_VERSION
    ):
        raise HardwareSyncEvidenceError(
            "content_provenance.hardware_contract.schema_version must be "
            f"{RUN_CONTRACT_SCHEMA_VERSION!r}"
        )
    contract_path = _required_text(
        hardware_contract_value.get("path"),
        "content_provenance.hardware_contract.path",
    )
    if contract_path != RUN_CONFIG:
        raise HardwareSyncEvidenceError(
            f"content_provenance.hardware_contract.path must be {RUN_CONFIG!r}"
        )
    hardware_contract = {
        "schema_version": RUN_CONTRACT_SCHEMA_VERSION,
        "path": contract_path,
        "sha256": _required_sha256(
            hardware_contract_value.get("sha256"),
            "content_provenance.hardware_contract.sha256",
        ),
    }

    raw_sensors = provenance.get("sensors")
    if not isinstance(raw_sensors, list) or len(raw_sensors) != len(sensor_order):
        raise HardwareSyncEvidenceError(
            "content_provenance.sensors must exactly follow sensor_order"
        )
    validated_sensors: list[dict[str, Any]] = []
    for index, sensor_key in enumerate(sensor_order):
        field = f"content_provenance.sensors[{index}]"
        sensor = _required_mapping(raw_sensors[index], field)
        expected_fields = {
            "sensor_key",
            "frame_metadata",
            "matched_robot_poses",
            "referenced_frames",
            "content_sha256",
        }
        if set(sensor) != expected_fields:
            raise HardwareSyncEvidenceError(
                f"{field} contains missing or unknown fields"
            )
        if sensor.get("sensor_key") != sensor_key:
            raise HardwareSyncEvidenceError(
                "content_provenance.sensors must exactly follow sensor_order"
            )
        inventory = sensor_inventory[sensor_key]
        frame_metadata = _validate_file_attestation(
            sensor.get("frame_metadata"),
            field=f"{field}.frame_metadata",
            expected_path=_required_text(
                inventory.get("frame_metadata_path"),
                f"sensors[{index}].frame_metadata_path",
            ),
        )
        matched_robot_poses = _validate_file_attestation(
            sensor.get("matched_robot_poses"),
            field=f"{field}.matched_robot_poses",
            expected_path=_required_text(
                inventory.get("matched_robot_poses_path"),
                f"sensors[{index}].matched_robot_poses_path",
            ),
        )
        referenced_value = _required_mapping(
            sensor.get("referenced_frames"),
            f"{field}.referenced_frames",
        )
        if set(referenced_value) != {
            "file_count",
            "total_size_bytes",
            "manifest_sha256",
        }:
            raise HardwareSyncEvidenceError(
                f"{field}.referenced_frames contains invalid fields"
            )
        referenced_frames = {
            "file_count": _required_int(
                referenced_value.get("file_count"),
                f"{field}.referenced_frames.file_count",
                minimum=0,
            ),
            "total_size_bytes": _required_int(
                referenced_value.get("total_size_bytes"),
                f"{field}.referenced_frames.total_size_bytes",
                minimum=0,
            ),
            "manifest_sha256": _required_sha256(
                referenced_value.get("manifest_sha256"),
                f"{field}.referenced_frames.manifest_sha256",
            ),
        }
        frame_count = _required_int(
            inventory.get("frame_count"),
            f"sensors[{index}].frame_count",
            minimum=1,
        )
        if referenced_frames["file_count"] != frame_count * 4:
            raise HardwareSyncEvidenceError(
                f"{field}.referenced_frames.file_count must attest all four "
                "RGB-D source/synchronized files for every sensor frame"
            )
        base_sensor = {
            "sensor_key": sensor_key,
            "frame_metadata": frame_metadata,
            "matched_robot_poses": matched_robot_poses,
            "referenced_frames": referenced_frames,
        }
        content_sha256 = _required_sha256(
            sensor.get("content_sha256"),
            f"{field}.content_sha256",
        )
        if content_sha256 != _canonical_sha256(base_sensor):
            raise HardwareSyncEvidenceError(
                f"{field}.content_sha256 is inconsistent"
            )
        validated_sensors.append(
            {**base_sensor, "content_sha256": content_sha256}
        )

    base_provenance = {
        "schema_version": CONTENT_PROVENANCE_SCHEMA_VERSION,
        "digest_algorithm": CONTENT_DIGEST_ALGORITHM,
        "hardware_contract": hardware_contract,
        "sensors": validated_sensors,
    }
    aggregate_sha256 = _required_sha256(
        provenance.get("aggregate_sha256"),
        "content_provenance.aggregate_sha256",
    )
    if aggregate_sha256 != _canonical_sha256(base_provenance):
        raise HardwareSyncEvidenceError(
            "content_provenance.aggregate_sha256 is inconsistent"
        )
    return {
        **base_provenance,
        "aggregate_sha256": aggregate_sha256,
    }


def _current_authoritative_payload(run_root: Path) -> dict[str, Any]:
    """Rebuild the exact payload allowed at the durable artifact boundary."""

    run_config = _load_run_config(run_root, None)
    try:
        qualification = validate_hardware_sync_qualification(
            run_root,
            run_config=run_config,
        )
    except (FileNotFoundError, HardwareSyncQualificationError) as exc:
        raise HardwareSyncEvidenceError(
            "The hardware contract changed or a current physical hardware-sync "
            "qualification is otherwise invalid for authoritative frame groups: "
            f"{exc}"
        ) from exc
    canonical = build_hardware_sync_frame_groups(
        run_root,
        run_config=run_config,
    )
    try:
        final_qualification = validate_hardware_sync_qualification(run_root)
    except (FileNotFoundError, HardwareSyncQualificationError) as exc:
        raise HardwareSyncEvidenceError(
            "Hardware-sync qualification became invalid while rebuilding "
            f"authoritative frame groups: {exc}"
        ) from exc
    if final_qualification != qualification:
        raise HardwareSyncEvidenceError(
            "Hardware-sync qualification changed while rebuilding "
            "authoritative frame groups"
        )
    execution_binding = capture_execution_hardware_sync_binding(
        run_root,
        qualification=qualification,
    )
    return {
        **canonical,
        "hardware_sync_qualification": qualification,
        "hardware_sync_execution_binding": execution_binding,
    }


def validate_hardware_sync_frame_groups(
    value: Mapping[str, Any],
    *,
    run_root: str | Path | None = None,
) -> None:
    """Validate the durable complete-group boundary used by later consumers."""

    payload = _required_mapping(value, "hardware sync frame groups")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise HardwareSyncEvidenceError(
            f"Hardware sync frame groups schema_version must be {SCHEMA_VERSION!r}"
        )
    expected_top_level = {
        "mode": SUPPORTED_MODE,
        "implementation": SUPPORTED_IMPLEMENTATION,
        "scope": SUPPORTED_SCOPE,
    }
    for field, expected in expected_top_level.items():
        if payload.get(field) != expected:
            raise HardwareSyncEvidenceError(
                f"Hardware sync frame groups {field} must be {expected!r}"
            )
    group_id = _required_text(payload.get("group_id"), "group_id")
    master_sensor_key = _required_text(
        payload.get("master_sensor_key"), "master_sensor_key"
    )
    max_skew_ns = _required_int(
        payload.get("max_depth_timestamp_skew_ns"),
        "max_depth_timestamp_skew_ns",
        minimum=1,
    )
    raw_max_skew_ms = payload.get("max_depth_timestamp_skew_ms")
    if (
        isinstance(raw_max_skew_ms, bool)
        or not isinstance(raw_max_skew_ms, int | float)
    ):
        raise HardwareSyncEvidenceError(
            "max_depth_timestamp_skew_ms must be a finite positive number"
        )
    max_skew_ms = float(raw_max_skew_ms)
    if (
        not math.isfinite(max_skew_ms)
        or max_skew_ms <= 0
        or max_skew_ms > MAX_SUPPORTED_SKEW_MS
    ):
        raise HardwareSyncEvidenceError(
            "max_depth_timestamp_skew_ms must be a finite positive number no "
            f"greater than {MAX_SUPPORTED_SKEW_MS}"
        )
    if max_skew_ns != int(round(max_skew_ms * 1_000_000)):
        raise HardwareSyncEvidenceError(
            "max_depth_timestamp_skew_ns is inconsistent with "
            "max_depth_timestamp_skew_ms"
        )
    sensor_order_value = payload.get("sensor_order")
    if not isinstance(sensor_order_value, list) or len(sensor_order_value) < 2:
        raise HardwareSyncEvidenceError(
            "sensor_order must contain at least two sensor keys"
        )
    sensor_order = [
        _required_text(item, f"sensor_order[{index}]")
        for index, item in enumerate(sensor_order_value)
    ]
    if len(sensor_order) != len(set(sensor_order)):
        raise HardwareSyncEvidenceError("sensor_order must not contain duplicates")
    if sensor_order[0] != master_sensor_key:
        raise HardwareSyncEvidenceError(
            "sensor_order must place master_sensor_key first"
        )
    raw_sensors = payload.get("sensors")
    if not isinstance(raw_sensors, list) or len(raw_sensors) != len(sensor_order):
        raise HardwareSyncEvidenceError(
            "sensors must contain one inventory entry per sensor_order key"
        )
    sensor_inventory: dict[str, Mapping[str, Any]] = {}
    mounting_modes: set[str] = set()
    for sensor_index, raw_sensor in enumerate(raw_sensors):
        sensor = _required_mapping(
            raw_sensor,
            f"sensors[{sensor_index}]",
        )
        sensor_key = sensor_order[sensor_index]
        if sensor.get("sensor_key") != sensor_key:
            raise HardwareSyncEvidenceError(
                "sensors inventory must exactly follow sensor_order"
            )
        role = "master" if sensor_index == 0 else "subordinate"
        if sensor.get("hardware_sync_role") != role:
            raise HardwareSyncEvidenceError(
                f"sensors[{sensor_index}].hardware_sync_role must be {role!r}"
            )
        sensor_type = _required_text(
            sensor.get("sensor_type"),
            f"sensors[{sensor_index}].sensor_type",
        )
        device_id = _required_text(
            sensor.get("device_id"),
            f"sensors[{sensor_index}].device_id",
        )
        if sensor_key != f"{sensor_type}:{device_id}":
            raise HardwareSyncEvidenceError(
                f"sensors[{sensor_index}] identity does not match sensor_key"
            )
        sensor_folder = _required_text(
            sensor.get("sensor_folder"),
            f"sensors[{sensor_index}].sensor_folder",
        )
        if Path(sensor_folder).name != sensor_folder_name(sensor_type, device_id):
            raise HardwareSyncEvidenceError(
                f"sensors[{sensor_index}].sensor_folder does not match identity"
            )
        mounting_mode = _required_text(
            sensor.get("mounting_mode"),
            f"sensors[{sensor_index}].mounting_mode",
        )
        if mounting_mode not in {"static", "eye_in_hand"}:
            raise HardwareSyncEvidenceError(
                f"sensors[{sensor_index}].mounting_mode is unsupported"
            )
        mounting_modes.add(mounting_mode)
        sensor_inventory[sensor_key] = sensor
    if mounting_modes != {"static", "eye_in_hand"}:
        raise HardwareSyncEvidenceError(
            "hardware-sync sensor inventory must include static and eye_in_hand"
        )
    if payload.get("run_config_path") != RUN_CONFIG:
        raise HardwareSyncEvidenceError(
            f"run_config_path must be {RUN_CONFIG!r}"
        )
    _validate_hardware_sync_execution_binding_shape(
        payload.get("hardware_sync_execution_binding")
    )
    _validate_content_provenance_shape(
        payload.get("content_provenance"),
        sensor_order=sensor_order,
        sensor_inventory=sensor_inventory,
    )
    root = Path(run_root) if run_root is not None else None

    groups = payload.get("groups")
    if not isinstance(groups, list) or not groups:
        raise HardwareSyncEvidenceError(
            "Hardware sync frame groups must contain at least one complete group"
        )
    seen_group_ids: set[str] = set()
    seen_frames: dict[str, set[tuple[int, str]]] = {
        sensor_key: set() for sensor_key in sensor_order
    }
    previous_master_timestamp: int | None = None
    for group_index, raw_group in enumerate(groups):
        group = _required_mapping(raw_group, f"groups[{group_index}]")
        if group.get("frame_group_index") != group_index:
            raise HardwareSyncEvidenceError(
                f"groups[{group_index}].frame_group_index must be {group_index}"
            )
        frame_group_id = _required_text(
            group.get("frame_group_id"),
            f"groups[{group_index}].frame_group_id",
        )
        if frame_group_id in seen_group_ids:
            raise HardwareSyncEvidenceError(
                f"Duplicate frame_group_id: {frame_group_id}"
            )
        seen_group_ids.add(frame_group_id)
        master_ordinal = _required_int(
            group.get("master_frame_ordinal"),
            f"groups[{group_index}].master_frame_ordinal",
            minimum=0,
        )
        if frame_group_id != f"{group_id}:{master_ordinal:06d}":
            raise HardwareSyncEvidenceError(
                f"groups[{group_index}].frame_group_id is not stable"
            )
        if group.get("capture_group_id") != group_id:
            raise HardwareSyncEvidenceError(
                f"groups[{group_index}].capture_group_id does not match group_id"
            )
        if group.get("master_sensor_key") != master_sensor_key:
            raise HardwareSyncEvidenceError(
                f"groups[{group_index}].master_sensor_key is invalid"
            )
        master_timestamp = _required_int(
            group.get("depth_sensor_timestamp_ns"),
            f"groups[{group_index}].depth_sensor_timestamp_ns",
            minimum=0,
        )
        if (
            previous_master_timestamp is not None
            and master_timestamp <= previous_master_timestamp
        ):
            raise HardwareSyncEvidenceError(
                "complete groups must have strictly increasing master timestamps"
            )
        previous_master_timestamp = master_timestamp
        group_robot_pose = _required_mapping(
            group.get("matched_robot_pose"),
            f"groups[{group_index}].matched_robot_pose",
        )
        if not group_robot_pose:
            raise HardwareSyncEvidenceError(
                f"groups[{group_index}].matched_robot_pose must not be empty"
            )
        frames = _required_mapping(
            group.get("frames"), f"groups[{group_index}].frames"
        )
        if set(frames) != set(sensor_order):
            raise HardwareSyncEvidenceError(
                f"groups[{group_index}].frames must exactly cover sensor_order"
            )
        observed_abs: list[int] = []
        observed_timestamps: list[int] = []
        for sensor_key in sensor_order:
            frame = _required_mapping(
                frames[sensor_key],
                f"groups[{group_index}].frames[{sensor_key!r}]",
            )
            if frame.get("sensor_key") != sensor_key:
                raise HardwareSyncEvidenceError(
                    f"groups[{group_index}] frame sensor_key mismatch"
                )
            inventory = sensor_inventory[sensor_key]
            for field in (
                "sensor_folder",
                "mounting_mode",
                "hardware_sync_role",
            ):
                if frame.get(field) != inventory.get(field):
                    raise HardwareSyncEvidenceError(
                        f"groups[{group_index}] frame {field} differs from "
                        "sensor inventory"
                    )
            synchronized_frame_index = _required_int(
                frame.get("synchronized_frame_index"),
                (
                    f"groups[{group_index}].frames[{sensor_key!r}]."
                    "synchronized_frame_index"
                ),
                minimum=0,
            )
            synchronized_frame_id = _required_text(
                frame.get("synchronized_frame_id"),
                (
                    f"groups[{group_index}].frames[{sensor_key!r}]."
                    "synchronized_frame_id"
                ),
            )
            frame_identity = (
                synchronized_frame_index,
                synchronized_frame_id,
            )
            if frame_identity in seen_frames[sensor_key]:
                raise HardwareSyncEvidenceError(
                    f"groups[{group_index}] reuses a synchronized frame for "
                    f"{sensor_key}"
                )
            seen_frames[sensor_key].add(frame_identity)
            _required_int(
                frame.get("source_frame_index"),
                (
                    f"groups[{group_index}].frames[{sensor_key!r}]."
                    "source_frame_index"
                ),
                minimum=0,
            )
            source_frame_id = _required_text(
                frame.get("source_frame_id"),
                (
                    f"groups[{group_index}].frames[{sensor_key!r}]."
                    "source_frame_id"
                ),
            )
            _required_int(
                frame.get("depth_frame_number"),
                (
                    f"groups[{group_index}].frames[{sensor_key!r}]."
                    "depth_frame_number"
                ),
                minimum=0,
            )
            if frame.get("depth_timestamp_domain") != REQUIRED_TIMESTAMP_DOMAIN:
                raise HardwareSyncEvidenceError(
                    f"groups[{group_index}] frame depth_timestamp_domain must "
                    f"be {REQUIRED_TIMESTAMP_DOMAIN!r}"
                )
            timestamp = _required_int(
                frame.get("depth_sensor_timestamp_ns"),
                (
                    f"groups[{group_index}].frames[{sensor_key!r}]."
                    "depth_sensor_timestamp_ns"
                ),
                minimum=0,
            )
            skew = _required_int(
                frame.get("depth_timestamp_skew_ns"),
                (
                    f"groups[{group_index}].frames[{sensor_key!r}]."
                    "depth_timestamp_skew_ns"
                ),
            )
            absolute_skew = _required_int(
                frame.get("abs_depth_timestamp_skew_ns"),
                (
                    f"groups[{group_index}].frames[{sensor_key!r}]."
                    "abs_depth_timestamp_skew_ns"
                ),
                minimum=0,
            )
            if skew != timestamp - master_timestamp or absolute_skew != abs(skew):
                raise HardwareSyncEvidenceError(
                    f"groups[{group_index}] frame timestamp skew is inconsistent"
                )
            if absolute_skew > max_skew_ns:
                raise HardwareSyncEvidenceError(
                    f"groups[{group_index}] frame exceeds configured skew"
                )
            if sensor_key == master_sensor_key and skew != 0:
                raise HardwareSyncEvidenceError(
                    f"groups[{group_index}] master frame skew must be zero"
                )
            observed_timestamps.append(timestamp)
            matched_robot_pose = _required_mapping(
                frame.get("matched_robot_pose"),
                (
                    f"groups[{group_index}].frames[{sensor_key!r}]."
                    "matched_robot_pose"
                ),
            )
            if not matched_robot_pose:
                raise HardwareSyncEvidenceError(
                    f"groups[{group_index}] frame matched_robot_pose is empty"
                )
            if (
                sensor_key == master_sensor_key
                and dict(matched_robot_pose) != dict(group_robot_pose)
            ):
                raise HardwareSyncEvidenceError(
                    f"groups[{group_index}] master matched_robot_pose differs "
                    "from group matched_robot_pose"
                )
            path_frame_ids = {
                "synchronized_rgb_path": synchronized_frame_id,
                "synchronized_depth_path": synchronized_frame_id,
                "source_rgb_path": source_frame_id,
                "source_depth_path": source_frame_id,
            }
            for path_field, expected_frame_id in path_frame_ids.items():
                path_value = _required_text(
                    frame.get(path_field),
                    (
                        f"groups[{group_index}].frames[{sensor_key!r}]."
                        f"{path_field}"
                    ),
                )
                if Path(path_value).name != expected_frame_id:
                    raise HardwareSyncEvidenceError(
                        f"groups[{group_index}] {path_field} does not match "
                        "its frame id"
                    )
            if root is not None:
                synchronized_folder = _validate_payload_folder(
                    frame.get("sensor_folder"),
                    root,
                    (
                        f"groups[{group_index}].frames[{sensor_key!r}]."
                        "sensor_folder"
                    ),
                )
                source_folder = _validate_payload_folder(
                    frame.get("source_sensor_folder"),
                    root,
                    (
                        f"groups[{group_index}].frames[{sensor_key!r}]."
                        "source_sensor_folder"
                    ),
                )
                for path_field, base_folder, expected_directory in (
                    ("synchronized_rgb_path", synchronized_folder, RGB_DIR),
                    ("synchronized_depth_path", synchronized_folder, DEPTH_DIR),
                    ("source_rgb_path", source_folder, RGB_DIR),
                    ("source_depth_path", source_folder, DEPTH_DIR),
                ):
                    _validate_payload_file(
                        frame.get(path_field),
                        base_folder=base_folder,
                        expected_directory=expected_directory,
                        field=(
                            f"groups[{group_index}].frames[{sensor_key!r}]."
                            f"{path_field}"
                        ),
                    )
            observed_abs.append(absolute_skew)
        if group.get("max_abs_depth_timestamp_skew_ns") != max(observed_abs):
            raise HardwareSyncEvidenceError(
                f"groups[{group_index}].max_abs_depth_timestamp_skew_ns "
                "is inconsistent"
            )
        timestamp_span_ns = _required_int(
            group.get("depth_timestamp_span_ns"),
            f"groups[{group_index}].depth_timestamp_span_ns",
            minimum=0,
        )
        observed_span_ns = max(observed_timestamps) - min(observed_timestamps)
        if timestamp_span_ns != observed_span_ns:
            raise HardwareSyncEvidenceError(
                f"groups[{group_index}].depth_timestamp_span_ns is inconsistent"
            )
        if timestamp_span_ns > max_skew_ns:
            raise HardwareSyncEvidenceError(
                f"groups[{group_index}] depth timestamp span exceeds configured "
                "skew"
            )

    summary = _required_mapping(payload.get("summary"), "summary")
    if summary.get("complete_group_count") != len(groups):
        raise HardwareSyncEvidenceError(
            "summary.complete_group_count does not match groups"
        )
    if root is not None:
        expected = _current_authoritative_payload(root)
        expected_qualification = expected["hardware_sync_qualification"]
        if payload.get("hardware_sync_qualification") != expected_qualification:
            raise HardwareSyncEvidenceError(
                "hardware_sync_qualification does not exactly match; the exact "
                "current physical hardware-sync qualification is required"
            )
        if (
            payload.get("hardware_sync_execution_binding")
            != expected["hardware_sync_execution_binding"]
        ):
            raise HardwareSyncEvidenceError(
                "hardware_sync_execution_binding does not exactly match the "
                "succeeded capture execution report"
            )
        actual_core = {
            key: item
            for key, item in payload.items()
            if key
            not in {
                "hardware_sync_qualification",
                "hardware_sync_execution_binding",
            }
        }
        expected_core = {
            key: item
            for key, item in expected.items()
            if key
            not in {
                "hardware_sync_qualification",
                "hardware_sync_execution_binding",
            }
        }
        if actual_core != expected_core:
            raise HardwareSyncEvidenceError(
                "Hardware sync frame groups differ from the canonical groups "
                "rebuilt from current synchronized metadata, matched robot "
                "poses, RGB-D files, and run configuration"
            )


def write_hardware_sync_frame_groups(
    run_root: str | Path,
    value: Mapping[str, Any],
) -> Path:
    """Atomically write a validated multiview frame-group artifact."""

    root = Path(run_root)
    validate_hardware_sync_frame_groups(value, run_root=root)
    return atomic_write_json(hardware_sync_frame_groups_path(root), dict(value))


def load_hardware_sync_frame_groups(run_root: str | Path) -> dict[str, Any]:
    """Load and validate the run-owned multiview frame-group artifact."""

    root = Path(run_root)
    path = hardware_sync_frame_groups_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"Hardware sync frame groups do not exist: {path}")
    value = _required_mapping(_read_json(path), "hardware sync frame groups")
    result = dict(value)
    validate_hardware_sync_frame_groups(result, run_root=root)
    return result
