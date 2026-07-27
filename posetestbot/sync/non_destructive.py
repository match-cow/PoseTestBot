"""Non-destructive frame/robot-pose synchronization.

This module bridges the legacy timestamp-named frame folders to the rewrite's
manifest-backed storage. It copies synchronized frames into a derived folder and
keeps raw capture folders unchanged.
"""

from __future__ import annotations

import json
import math
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from posetestbot.io.atomic import (
    atomic_write_json,
    atomic_write_text,
    replace_directory,
)
from posetestbot.io.artifacts import (
    DEPTH_DIR,
    FRAME_METADATA_JSONL,
    LEGACY_SENSOR_METADATA_ARTIFACTS,
    MATCH_ROBOT_EE_POSES,
    PROCESSED_DIR,
    RGB_DIR,
    RAW_ROBOT_EE_POSES,
    SYNC_REPORT,
    SYNCHRONIZED_DIR,
)
from posetestbot.io.manifest import discover_sensor_records
from posetestbot.pipeline.sensor_selection import filter_enabled_sensor_folders


SCHEMA_VERSION = "sync_report.v3"
FRAME_TIMESTAMP_SOURCES = ("host_received", "host_wall", "sensor", "filename")
ROBOT_TIMESTAMP_SOURCES = ("host_received", "host_wall", "filename")
SUPPORTED_TIMESTAMP_PAIRS = {
    ("host_received", "host_received"),
    ("host_wall", "host_wall"),
    ("sensor", "host_wall"),
    ("filename", "host_wall"),
}


@dataclass(frozen=True)
class SyncResult:
    sensor_folder: str
    output_folder: str
    matched_poses_path: str
    report_path: str
    total_frames: int
    matched_frames: int
    dropped_frames: int


def _read_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_json(path, value)


def load_frame_metadata(sensor_folder: str | Path) -> list[dict[str, Any]]:
    folder = Path(sensor_folder)
    metadata_path = folder / FRAME_METADATA_JSONL
    if metadata_path.exists():
        records = []
        seen_frame_ids: set[str] = set()
        with open(metadata_path, "r") as f:
            for line_number, line in enumerate(f, start=1):
                stripped = line.strip()
                if stripped:
                    try:
                        record = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON in {metadata_path} line {line_number}: {exc.msg}"
                        ) from exc
                    if not isinstance(record, dict):
                        raise ValueError(
                            f"Frame metadata line {line_number} must be a JSON object"
                        )
                    frame_id = str(record.get("frame_id") or "")
                    if not frame_id:
                        raise ValueError(
                            f"Frame metadata line {line_number} is missing frame_id"
                        )
                    if frame_id in seen_frame_ids:
                        raise ValueError(f"Duplicate frame_id in metadata: {frame_id}")
                    seen_frame_ids.add(frame_id)
                    records.append(record)
        return records

    records = []
    for index, rgb_path in enumerate(sorted((folder / RGB_DIR).glob("*.png"))):
        frame_id = rgb_path.name
        try:
            timestamp_ms = int(rgb_path.stem)
        except ValueError as exc:
            raise ValueError(
                f"Legacy RGB filename is not a numeric timestamp: {rgb_path.name}"
            ) from exc
        records.append(
            {
                "schema_version": "frame_metadata.v1",
                "sensor_type": "legacy_unknown",
                "sensor_id": folder.name,
                "frame_index": index,
                "frame_id": frame_id,
                "rgb_path": f"{RGB_DIR}/{frame_id}",
                "depth_path": f"{DEPTH_DIR}/{frame_id}",
                "filename_timestamp_ns": timestamp_ms * 1_000_000,
            }
        )
    return records


def load_robot_poses(run_root: str | Path, sensor_folder: str | Path) -> dict[str, Any]:
    candidates = [
        Path(sensor_folder) / RAW_ROBOT_EE_POSES,
        Path(run_root) / RAW_ROBOT_EE_POSES,
    ]
    for candidate in candidates:
        if candidate.exists():
            return _read_json(candidate)
    raise FileNotFoundError(
        f"Could not find {RAW_ROBOT_EE_POSES} in {sensor_folder} or {run_root}"
    )


def resolve_timestamp_pair(
    frame_timestamp_source: str,
    robot_timestamp_source: str | None,
) -> tuple[str, str]:
    """Resolve one explicit, clock-compatible frame/robot timestamp pair."""

    if frame_timestamp_source not in FRAME_TIMESTAMP_SOURCES:
        raise ValueError(
            "timestamp_source must be host_received, host_wall, sensor, or filename"
        )
    if robot_timestamp_source is None:
        if frame_timestamp_source in {"host_received", "host_wall"}:
            robot_timestamp_source = frame_timestamp_source
        else:
            raise ValueError(
                f"timestamp_source={frame_timestamp_source!r} requires an explicit "
                "robot_timestamp_source"
            )
    if robot_timestamp_source not in ROBOT_TIMESTAMP_SOURCES:
        raise ValueError(
            "robot_timestamp_source must be host_received, host_wall, or filename"
        )
    if (frame_timestamp_source, robot_timestamp_source) not in (
        SUPPORTED_TIMESTAMP_PAIRS
    ):
        raise ValueError(
            "Frame/robot timestamp sources must share a clock domain; unsupported "
            f"pair: {frame_timestamp_source}->{robot_timestamp_source}"
        )
    return frame_timestamp_source, robot_timestamp_source


def robot_timestamp_ns(
    record: Mapping[str, Any], timestamp_source: str = "host_received"
) -> int:
    if timestamp_source == "host_received":
        value = record.get("host_received_timestamp_ns")
    elif timestamp_source == "host_wall":
        value = record.get("host_wall_timestamp_ns")
    elif timestamp_source == "filename":
        framename = record.get("framename")
        value = int(framename) * 1_000_000 if framename is not None else None
    else:
        raise ValueError(
            "robot timestamp source must be host_received, host_wall, or filename"
        )
    if value is None:
        raise ValueError(
            f"Robot pose is missing required {timestamp_source} timestamp evidence"
        )
    return int(value)


def resolve_frame_timestamp(
    record: Mapping[str, Any], timestamp_source: str
) -> tuple[int | None, str | None, bool]:
    if timestamp_source == "host_received":
        value = record.get("host_received_timestamp_ns")
    elif timestamp_source == "host_wall":
        value = record.get("host_wall_timestamp_ns")
    elif timestamp_source == "sensor":
        value = record.get("sensor_timestamp_ns")
    elif timestamp_source == "filename":
        value = record.get("filename_timestamp_ns")
        if value is None and record.get("frame_id"):
            try:
                value = int(Path(str(record["frame_id"])).stem) * 1_000_000
            except ValueError:
                value = None
    else:
        raise ValueError(
            "timestamp_source must be host_received, host_wall, sensor, or filename"
        )

    actual_source = timestamp_source if value is not None else None
    fallback = False
    if value is None and timestamp_source != "filename" and record.get("frame_id"):
        try:
            value = int(Path(str(record["frame_id"])).stem) * 1_000_000
        except ValueError:
            value = None
        else:
            actual_source = "filename"
            fallback = True

    return (
        (int(value), actual_source, fallback)
        if value is not None
        else (None, None, False)
    )


def frame_timestamp_ns(record: Mapping[str, Any], timestamp_source: str) -> int | None:
    """Compatibility wrapper returning only the resolved timestamp."""

    return resolve_frame_timestamp(record, timestamp_source)[0]


def indexed_robot_poses(
    raw_poses: Mapping[str, Any],
    *,
    timestamp_source: str = "host_received",
) -> list[dict[str, Any]]:
    if not isinstance(raw_poses, Mapping) or not raw_poses:
        raise ValueError("Raw robot pose artifact must be a non-empty JSON object")
    records = []
    for key, value in raw_poses.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"Robot pose {key!r} must be a JSON object")
        record = dict(value)
        try:
            record["pose_index"] = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Robot pose key must be numeric: {key!r}") from exc
        if not record.get("motion"):
            raise ValueError(f"Robot pose {key!r} is missing motion")
        if not isinstance(record.get("pose"), Mapping):
            raise ValueError(f"Robot pose {key!r} is missing pose coordinates")
        record["timestamp_ns"] = robot_timestamp_ns(record, timestamp_source)
        records.append(record)
    return sorted(records, key=lambda item: item["timestamp_ns"])


def motion_intervals(
    robot_records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    for record in robot_records:
        motion = str(record["motion"])
        timestamp = int(record["timestamp_ns"])
        if not intervals or intervals[-1]["motion"] != motion:
            intervals.append(
                {
                    "motion": motion,
                    "min_timestamp_ns": timestamp,
                    "max_timestamp_ns": timestamp,
                    "pose_count": 1,
                }
            )
        else:
            intervals[-1]["max_timestamp_ns"] = timestamp
            intervals[-1]["pose_count"] += 1
    return intervals


def motion_for_timestamp(
    timestamp_ns: int, intervals: Iterable[Mapping[str, Any]]
) -> str | None:
    for interval in intervals:
        if (
            int(interval["min_timestamp_ns"])
            <= timestamp_ns
            <= int(interval["max_timestamp_ns"])
        ):
            return str(interval["motion"])
    return None


def robot_pose_packet_loss(
    robot_records: Iterable[Mapping[str, Any]],
) -> tuple[bool, int]:
    """Return whether packet-loss evidence is complete and its recorded total."""

    audited = True
    total = 0
    found = False
    for record in robot_records:
        source_packet = record.get("source_packet")
        if not isinstance(source_packet, Mapping):
            audited = False
            continue
        value = source_packet.get("estimated_packets_lost")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            audited = False
            continue
        found = True
        total += value
    return audited and found, total


def closest_robot_pose(
    timestamp_ns: int, robot_records: list[dict[str, Any]]
) -> dict[str, Any]:
    return min(
        robot_records,
        key=lambda record: abs(int(record["timestamp_ns"]) - timestamp_ns),
    )


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _sensor_sync_keys(sensor_folder_name: str) -> tuple[str, ...]:
    name = sensor_folder_name.lower()
    exact = (
        (sensor_folder_name, name)
        if sensor_folder_name != name
        else (sensor_folder_name,)
    )
    if name.startswith("realsense"):
        aliases = ("realsense_d435", "realsense")
    elif name.startswith("luxonis") or name.startswith("oak"):
        aliases = ("oak_d_pro", "luxonis", "oak")
    elif name.startswith("zed_2i") or name.startswith("zed"):
        aliases = ("zed_2i", "zed")
    else:
        aliases = (name.split("_")[0],)
    return tuple(dict.fromkeys((*exact, *aliases)))


def resolve_sync_delta_ms(
    sensor_folder: str | Path, sync_delta: int | float | Mapping[str, Any] | None
) -> float:
    value: object = 100.0
    if sync_delta is not None:
        if isinstance(sync_delta, bool):
            raise ValueError(
                "Synchronization delta must be a finite number, not a boolean"
            )
        if isinstance(sync_delta, int | float):
            value = sync_delta
        elif isinstance(sync_delta, Mapping):
            for key in _sensor_sync_keys(Path(sensor_folder).name):
                if key in sync_delta:
                    value = sync_delta[key]
                    break
        else:
            raise ValueError("Synchronization delta must be a number or sensor mapping")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Synchronization delta must be numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError("Synchronization delta must be finite")
    return result


def resolve_max_nearest_pose_delta_ms(
    value: int | float | None,
) -> float | None:
    """Validate an optional strict nearest-pose matching threshold."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(
            "Maximum nearest-pose delta must be a finite non-negative number, "
            "not a boolean"
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Maximum nearest-pose delta must be numeric: {value!r}"
        ) from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(
            "Maximum nearest-pose delta must be finite and greater than or equal to 0"
        )
    return result


def _copy_frame_pair(
    *,
    sensor_folder: Path,
    output_folder: Path,
    frame_metadata: Mapping[str, Any],
    output_frame_id: str,
) -> tuple[Path, Path]:
    source_rgb = _resolve_source_frame_path(
        sensor_folder, frame_metadata.get("rgb_path"), RGB_DIR
    )
    source_depth = _resolve_source_frame_path(
        sensor_folder, frame_metadata.get("depth_path"), DEPTH_DIR
    )
    output_rgb = output_folder / RGB_DIR / output_frame_id
    output_depth = output_folder / DEPTH_DIR / output_frame_id
    output_rgb.parent.mkdir(parents=True, exist_ok=True)
    output_depth.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_rgb, output_rgb)
    shutil.copy2(source_depth, output_depth)
    return output_rgb, output_depth


def _resolve_source_frame_path(
    sensor_folder: Path, value: Any, expected_dir: str
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Frame metadata is missing {expected_dir} path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"Frame path must be relative: {value}")
    resolved = (sensor_folder / relative).resolve()
    sensor_resolved = sensor_folder.resolve()
    try:
        descendant = resolved.relative_to(sensor_resolved)
    except ValueError as exc:
        raise ValueError(f"Frame path escapes sensor folder: {value}") from exc
    if not descendant.parts or descendant.parts[0] != expected_dir:
        raise ValueError(f"Frame path must be below {expected_dir}/: {value}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Frame file does not exist: {resolved}")
    return resolved


def copy_sensor_metadata_artifacts(
    sensor_folder: Path, output_folder: Path
) -> list[str]:
    copied = []
    for artifact in LEGACY_SENSOR_METADATA_ARTIFACTS:
        if artifact == FRAME_METADATA_JSONL:
            continue
        source = sensor_folder / artifact
        if not source.exists():
            continue
        destination = output_folder / artifact
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(artifact)
    return copied


def _delta_stats(deltas_ns: list[int]) -> dict[str, float | int | None]:
    if not deltas_ns:
        return {
            "mean_abs_nearest_pose_delta_ns": None,
            "max_abs_nearest_pose_delta_ns": None,
        }
    abs_deltas = [abs(delta) for delta in deltas_ns]
    return {
        "mean_abs_nearest_pose_delta_ns": mean(abs_deltas),
        "max_abs_nearest_pose_delta_ns": max(abs_deltas),
    }


def synchronize_sensor_folder(
    sensor_folder: str | Path,
    *,
    run_root: str | Path | None = None,
    output_root: str | Path | None = None,
    sync_delta: int | float | Mapping[str, Any] | None = None,
    timestamp_source: str = "host_received",
    robot_timestamp_source: str | None = None,
    copy_files: bool = True,
    max_nearest_pose_delta_ms: int | float | None = None,
    required_frame_timestamp_domain: str | None = None,
    timestamp_fallback_allowed: bool = True,
    calibration_sync: Mapping[str, Any] | None = None,
) -> SyncResult:
    sensor_path = Path(sensor_folder)
    run_path = Path(run_root) if run_root is not None else sensor_path.parent
    timestamp_source, resolved_robot_timestamp_source = resolve_timestamp_pair(
        timestamp_source, robot_timestamp_source
    )
    nearest_pose_threshold_ms = resolve_max_nearest_pose_delta_ms(
        max_nearest_pose_delta_ms
    )
    nearest_pose_threshold_ns = (
        int(nearest_pose_threshold_ms * 1_000_000)
        if nearest_pose_threshold_ms is not None
        else None
    )
    if required_frame_timestamp_domain is not None and (
        not isinstance(required_frame_timestamp_domain, str)
        or not required_frame_timestamp_domain.strip()
    ):
        raise ValueError(
            "Required frame timestamp domain must be a non-empty string or null"
        )
    if not isinstance(timestamp_fallback_allowed, bool):
        raise ValueError("timestamp_fallback_allowed must be a boolean")
    if calibration_sync is not None and not isinstance(calibration_sync, Mapping):
        raise ValueError("calibration_sync provenance must be an object")
    output_base = (
        Path(output_root)
        if output_root is not None
        else run_path / PROCESSED_DIR / SYNCHRONIZED_DIR
    )
    output_folder = output_base / sensor_path.name
    output_base.mkdir(parents=True, exist_ok=True)
    staging_folder = output_base / f".{sensor_path.name}.{uuid.uuid4().hex}.tmp"
    staging_folder.mkdir(parents=False, exist_ok=False)

    try:
        frame_records = load_frame_metadata(sensor_path)
        if not frame_records:
            raise ValueError(f"No frame metadata or RGB frames found in {sensor_path}")
        raw_robot_poses = load_robot_poses(run_path, sensor_path)
        robot_records = indexed_robot_poses(
            raw_robot_poses,
            timestamp_source=resolved_robot_timestamp_source,
        )
        intervals = motion_intervals(robot_records)
        pose_packet_loss_audited, pose_packet_loss_count = robot_pose_packet_loss(
            robot_records
        )
        sensor_sync_delta_ms = resolve_sync_delta_ms(sensor_path, sync_delta)
        sync_delta_ns = int(sensor_sync_delta_ms * 1_000_000)

        resolved_records: list[tuple[int | None, str | None, bool, dict[str, Any]]] = []
        for frame_record in frame_records:
            _resolve_source_frame_path(
                sensor_path, frame_record.get("rgb_path"), RGB_DIR
            )
            _resolve_source_frame_path(
                sensor_path, frame_record.get("depth_path"), DEPTH_DIR
            )
            if (
                required_frame_timestamp_domain is not None
                and frame_record.get("color_timestamp_domain")
                != required_frame_timestamp_domain
            ):
                raise ValueError(
                    f"Frame {frame_record.get('frame_id')!r} in "
                    f"{sensor_path.name} has color timestamp domain "
                    f"{frame_record.get('color_timestamp_domain')!r}; required "
                    f"{required_frame_timestamp_domain!r}"
                )
            resolved = resolve_frame_timestamp(frame_record, timestamp_source)
            if not timestamp_fallback_allowed and (
                resolved[0] is None or resolved[1] != timestamp_source or resolved[2]
            ):
                raise ValueError(
                    f"Frame {frame_record.get('frame_id')!r} in "
                    f"{sensor_path.name} cannot prove required "
                    f"{timestamp_source!r} timing without fallback"
                )
            resolved_records.append((*resolved, frame_record))
        resolved_records.sort(
            key=lambda item: (item[0] is None, item[0] if item[0] is not None else 0)
        )

        matched: dict[str, Any] = {}
        derived_metadata: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        nearest_deltas_ns: list[int] = []
        timestamp_source_counts: dict[str, int] = {}
        timestamp_fallback_count = 0
        timestamp_missing_count = 0
        incompatible_timestamp_pair_count = 0
        outside_motion_interval_frame_count = 0
        eligible_in_motion_frames = 0
        nearest_pose_delta_rejection_count = 0
        previous_frame_timestamp_ns: int | None = None
        output_counter = 0
        copied_metadata_artifacts = (
            copy_sensor_metadata_artifacts(sensor_path, staging_folder)
            if copy_files
            else []
        )

        for timestamp_ns, actual_source, fallback, frame_record in resolved_records:
            if timestamp_ns is None or actual_source is None:
                timestamp_missing_count += 1
                dropped.append(
                    {
                        "frame_id": frame_record.get("frame_id"),
                        "reason": f"missing {timestamp_source} timestamp",
                    }
                )
                continue
            timestamp_source_counts[actual_source] = (
                timestamp_source_counts.get(actual_source, 0) + 1
            )
            if fallback:
                timestamp_fallback_count += 1
            if (actual_source, resolved_robot_timestamp_source) not in (
                SUPPORTED_TIMESTAMP_PAIRS
            ):
                incompatible_timestamp_pair_count += 1
                dropped.append(
                    {
                        "frame_id": frame_record.get("frame_id"),
                        "timestamp_ns": timestamp_ns,
                        "timestamp_source": actual_source,
                        "robot_timestamp_source": (resolved_robot_timestamp_source),
                        "reason": "frame/robot timestamp fallback clocks are incompatible",
                    }
                )
                continue

            delayed_timestamp_ns = timestamp_ns - sync_delta_ns
            motion = motion_for_timestamp(delayed_timestamp_ns, intervals)
            if motion is None:
                outside_motion_interval_frame_count += 1
                dropped.append(
                    {
                        "frame_id": frame_record.get("frame_id"),
                        "timestamp_ns": timestamp_ns,
                        "timestamp_source": actual_source,
                        "robot_timestamp_source": (resolved_robot_timestamp_source),
                        "delayed_timestamp_ns": delayed_timestamp_ns,
                        "reason": "outside robot motion intervals",
                    }
                )
                continue

            eligible_in_motion_frames += 1
            closest_pose = closest_robot_pose(delayed_timestamp_ns, robot_records)
            nearest_delta_ns = int(closest_pose["timestamp_ns"]) - delayed_timestamp_ns
            if (
                nearest_pose_threshold_ns is not None
                and abs(nearest_delta_ns) > nearest_pose_threshold_ns
            ):
                nearest_pose_delta_rejection_count += 1
                dropped.append(
                    {
                        "frame_id": frame_record.get("frame_id"),
                        "timestamp_ns": timestamp_ns,
                        "timestamp_source": actual_source,
                        "robot_timestamp_source": (resolved_robot_timestamp_source),
                        "delayed_timestamp_ns": delayed_timestamp_ns,
                        "motion": motion,
                        "matched_robot_pose_index": closest_pose["pose_index"],
                        "robot_timestamp_ns": int(closest_pose["timestamp_ns"]),
                        "nearest_robot_delta_ns": nearest_delta_ns,
                        "abs_nearest_robot_delta_ns": abs(nearest_delta_ns),
                        "max_nearest_pose_delta_ms": nearest_pose_threshold_ms,
                        "max_nearest_pose_delta_ns": nearest_pose_threshold_ns,
                        "reason": "nearest robot pose delta exceeds threshold",
                    }
                )
                continue
            nearest_deltas_ns.append(nearest_delta_ns)
            frame_delta_ns = (
                0
                if previous_frame_timestamp_ns is None
                else timestamp_ns - previous_frame_timestamp_ns
            )
            previous_frame_timestamp_ns = timestamp_ns

            output_frame_id = f"{output_counter:06d}.png"
            if copy_files:
                _copy_frame_pair(
                    sensor_folder=sensor_path,
                    output_folder=staging_folder,
                    frame_metadata=frame_record,
                    output_frame_id=output_frame_id,
                )
            synchronized_rgb = _relative_path(
                output_folder / RGB_DIR / output_frame_id, run_path
            )
            synchronized_depth = _relative_path(
                output_folder / DEPTH_DIR / output_frame_id, run_path
            )

            matched[output_frame_id] = {
                "motion": motion,
                "image_frame": timestamp_ns // 1_000_000,
                "image_timestamp_ns": timestamp_ns,
                "timestamp_source": actual_source,
                "timestamp_fallback": fallback,
                "robot_timestamp_source": resolved_robot_timestamp_source,
                "sensor_timestamp_ns": frame_record.get("sensor_timestamp_ns"),
                "host_received_timestamp_ns": frame_record.get(
                    "host_received_timestamp_ns"
                ),
                "host_wall_timestamp_ns": frame_record.get("host_wall_timestamp_ns"),
                "delayed_frame": delayed_timestamp_ns // 1_000_000,
                "delayed_timestamp_ns": delayed_timestamp_ns,
                "frame_delta": frame_delta_ns // 1_000_000,
                "frame_delta_ns": frame_delta_ns,
                "robot_frame": int(closest_pose["timestamp_ns"]) // 1_000_000,
                "robot_timestamp_ns": int(closest_pose["timestamp_ns"]),
                "nearest_robot_delta_ns": nearest_delta_ns,
                "matched_robot_pose_index": closest_pose["pose_index"],
                "source_frame_id": frame_record.get("frame_id"),
                "source_rgb": frame_record.get("rgb_path"),
                "source_depth": frame_record.get("depth_path"),
                "synchronized_rgb": synchronized_rgb,
                "synchronized_depth": synchronized_depth,
                "robot_ee_pose": closest_pose["pose"],
            }
            derived_record = dict(frame_record)
            derived_record.update(
                {
                    "frame_index": output_counter,
                    "frame_id": output_frame_id,
                    "rgb_path": f"{RGB_DIR}/{output_frame_id}",
                    "depth_path": f"{DEPTH_DIR}/{output_frame_id}",
                    "source_frame_index": frame_record.get("frame_index"),
                    "source_frame_id": frame_record.get("frame_id"),
                    "source_rgb_path": frame_record.get("rgb_path"),
                    "source_depth_path": frame_record.get("depth_path"),
                    "sync_requested_timestamp_source": timestamp_source,
                    "sync_timestamp_source": actual_source,
                    "sync_robot_timestamp_source": (resolved_robot_timestamp_source),
                    "sync_timestamp_fallback": fallback,
                    "sync_timestamp_ns": timestamp_ns,
                    "sync_delta_ms": sensor_sync_delta_ms,
                    "matched_robot_pose_index": closest_pose["pose_index"],
                    "nearest_robot_delta_ns": nearest_delta_ns,
                    "motion": motion,
                }
            )
            derived_metadata.append(derived_record)
            output_counter += 1

        if copy_files:
            metadata_text = "".join(
                json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
                for record in derived_metadata
            )
            atomic_write_text(staging_folder / FRAME_METADATA_JSONL, metadata_text)

        _write_json(staging_folder / MATCH_ROBOT_EE_POSES, matched)
        in_motion_exclusion_count = eligible_in_motion_frames - len(matched)
        unexplained_in_motion_exclusion_count = (
            in_motion_exclusion_count - nearest_pose_delta_rejection_count
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "sensor_folder": _relative_path(sensor_path, run_path),
            "output_folder": _relative_path(output_folder, run_path),
            "requested_timestamp_source": timestamp_source,
            "requested_frame_timestamp_source": timestamp_source,
            "timestamp_source": (
                timestamp_source if timestamp_fallback_count == 0 else "mixed"
            ),
            "frame_timestamp_source": (
                timestamp_source if timestamp_fallback_count == 0 else "mixed"
            ),
            "robot_timestamp_source": resolved_robot_timestamp_source,
            "timestamp_pair": {
                "frame_timestamp_source": (
                    timestamp_source if timestamp_fallback_count == 0 else "mixed"
                ),
                "requested_frame_timestamp_source": timestamp_source,
                "robot_timestamp_source": resolved_robot_timestamp_source,
            },
            "timestamp_pair_provenance_audited": True,
            "timestamp_source_counts": timestamp_source_counts,
            "timestamp_fallback_count": timestamp_fallback_count,
            "timestamp_missing_count": timestamp_missing_count,
            "incompatible_timestamp_pair_count": (incompatible_timestamp_pair_count),
            "sync_delta_ms": sensor_sync_delta_ms,
            "max_nearest_pose_delta_ms": nearest_pose_threshold_ms,
            "required_frame_timestamp_domain": required_frame_timestamp_domain,
            "timestamp_fallback_allowed": timestamp_fallback_allowed,
            "calibration_sync": (
                dict(calibration_sync) if calibration_sync is not None else None
            ),
            "nearest_pose_delta_rejection_count": (nearest_pose_delta_rejection_count),
            "total_frames": len(frame_records),
            "matched_frames": len(matched),
            "dropped_frames": len(dropped),
            "outside_motion_interval_frame_count": (
                outside_motion_interval_frame_count
            ),
            "eligible_in_motion_frames": eligible_in_motion_frames,
            "matched_eligible_frames": len(matched),
            "eligible_motion_coverage": (
                len(matched) / eligible_in_motion_frames
                if eligible_in_motion_frames
                else 0.0
            ),
            "in_motion_exclusion_count": in_motion_exclusion_count,
            "unexplained_in_motion_exclusion_count": (
                unexplained_in_motion_exclusion_count
            ),
            "robot_pose_packet_loss_audited": pose_packet_loss_audited,
            "robot_pose_packet_loss_count": (
                pose_packet_loss_count if pose_packet_loss_audited else None
            ),
            "motion_intervals": intervals,
            "dropped": dropped,
            "copied_metadata_artifacts": copied_metadata_artifacts,
            **_delta_stats(nearest_deltas_ns),
        }
        _write_json(staging_folder / SYNC_REPORT, report)
        replace_directory(staging_folder, output_folder)
    except Exception:
        if staging_folder.exists():
            shutil.rmtree(staging_folder)
        raise

    matched_path = output_folder / MATCH_ROBOT_EE_POSES
    report_path = output_folder / SYNC_REPORT

    return SyncResult(
        sensor_folder=sensor_path.as_posix(),
        output_folder=output_folder.as_posix(),
        matched_poses_path=matched_path.as_posix(),
        report_path=report_path.as_posix(),
        total_frames=len(frame_records),
        matched_frames=len(matched),
        dropped_frames=len(dropped),
    )


def sync_result_artifacts(result: SyncResult) -> dict[str, str]:
    result_dict = asdict(result)
    return {
        MATCH_ROBOT_EE_POSES: result_dict["matched_poses_path"],
        SYNC_REPORT: result_dict["report_path"],
    }


def synchronize_run(
    run_root: str | Path,
    *,
    sensor_folders: Sequence[str | Path] | None = None,
    output_root: str | Path | None = None,
    sync_delta: int | float | Mapping[str, Any] | None = None,
    timestamp_source: str = "host_received",
    robot_timestamp_source: str | None = None,
    copy_files: bool = True,
    max_nearest_pose_delta_ms: int | float | None = None,
    required_frame_timestamp_domain: str | None = None,
    timestamp_fallback_allowed: bool = True,
    calibration_sync: Mapping[str, Any] | None = None,
) -> list[SyncResult]:
    """Synchronize discovered sensors or an explicit contained subset.

    Omitting ``sensor_folders`` preserves the original run-wide behavior.
    Supplying it lets intent-level orchestration reuse the stage without
    allowing an unselected or out-of-run folder to enter the calculation.
    """

    run_path = Path(run_root)
    results = []
    if sensor_folders is None:
        selected = filter_enabled_sensor_folders(
            run_path,
            (
                run_path / str(sensor_record["folder"])
                for sensor_record in discover_sensor_records(run_path)
            ),
        )
    else:
        selected = []
        seen: set[Path] = set()
        for value in sensor_folders:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = run_path / candidate
            resolved = candidate.resolve()
            try:
                resolved.relative_to(run_path.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"Explicit sensor folder must remain below the run root: {value}"
                ) from exc
            if resolved in seen:
                raise ValueError(f"Explicit sensor folder is duplicated: {value}")
            seen.add(resolved)
            selected.append(resolved)

    for sensor_folder in selected:
        results.append(
            synchronize_sensor_folder(
                sensor_folder,
                run_root=run_path,
                output_root=output_root,
                sync_delta=sync_delta,
                timestamp_source=timestamp_source,
                robot_timestamp_source=robot_timestamp_source,
                copy_files=copy_files,
                max_nearest_pose_delta_ms=max_nearest_pose_delta_ms,
                required_frame_timestamp_domain=required_frame_timestamp_domain,
                timestamp_fallback_allowed=timestamp_fallback_allowed,
                calibration_sync=calibration_sync,
            )
        )

    return results
