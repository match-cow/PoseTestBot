"""Non-destructive frame/robot-pose synchronization.

This module bridges the legacy timestamp-named frame folders to the rewrite's
manifest-backed storage. It copies synchronized frames into a derived folder and
keeps raw capture folders unchanged.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def load_frame_metadata(sensor_folder: str | Path) -> list[dict[str, Any]]:
    folder = Path(sensor_folder)
    metadata_path = folder / FRAME_METADATA_JSONL
    if metadata_path.exists():
        records = []
        with open(metadata_path, "r") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
        return records

    records = []
    for index, rgb_path in enumerate(sorted((folder / RGB_DIR).glob("*.png"))):
        frame_id = rgb_path.name
        timestamp_ms = int(rgb_path.stem)
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


def robot_timestamp_ns(record: Mapping[str, Any]) -> int:
    if record.get("host_received_timestamp_ns") is not None:
        return int(record["host_received_timestamp_ns"])
    if record.get("host_wall_timestamp_ns") is not None:
        return int(record["host_wall_timestamp_ns"])
    return int(record["framename"]) * 1_000_000


def frame_timestamp_ns(record: Mapping[str, Any], timestamp_source: str) -> int | None:
    if timestamp_source == "host_received":
        value = record.get("host_received_timestamp_ns")
    elif timestamp_source == "host_wall":
        value = record.get("host_wall_timestamp_ns")
    elif timestamp_source == "sensor":
        value = record.get("sensor_timestamp_ns")
    elif timestamp_source == "filename":
        value = record.get("filename_timestamp_ns")
        if value is None and record.get("frame_id"):
            value = int(Path(str(record["frame_id"])).stem) * 1_000_000
    else:
        raise ValueError(
            "timestamp_source must be host_received, host_wall, sensor, or filename"
        )

    if value is None and timestamp_source != "filename" and record.get("frame_id"):
        value = int(Path(str(record["frame_id"])).stem) * 1_000_000

    return int(value) if value is not None else None


def indexed_robot_poses(raw_poses: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for key, value in raw_poses.items():
        record = dict(value)
        record["pose_index"] = int(key)
        record["timestamp_ns"] = robot_timestamp_ns(record)
        records.append(record)
    return sorted(records, key=lambda item: item["timestamp_ns"])


def motion_windows(robot_records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    windows: dict[str, dict[str, int]] = {}
    for record in robot_records:
        motion = str(record["motion"])
        timestamp = int(record["timestamp_ns"])
        if motion not in windows:
            windows[motion] = {"min_timestamp_ns": timestamp, "max_timestamp_ns": timestamp}
        else:
            windows[motion]["min_timestamp_ns"] = min(
                windows[motion]["min_timestamp_ns"], timestamp
            )
            windows[motion]["max_timestamp_ns"] = max(
                windows[motion]["max_timestamp_ns"], timestamp
            )
    return windows


def motion_for_timestamp(
    timestamp_ns: int, windows: Mapping[str, Mapping[str, int]]
) -> str | None:
    for motion, window in windows.items():
        if window["min_timestamp_ns"] <= timestamp_ns <= window["max_timestamp_ns"]:
            return motion
    return None


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
    if name.startswith("realsense"):
        return ("realsense_d435", "realsense")
    if name.startswith("luxonis") or name.startswith("oak"):
        return ("oak_d_pro", "luxonis", "oak")
    if name.startswith("zed_2i") or name.startswith("zed"):
        return ("zed_2i", "zed")
    return (name.split("_")[0],)


def resolve_sync_delta_ms(
    sensor_folder: str | Path, sync_delta: int | float | Mapping[str, Any] | None
) -> float:
    if sync_delta is None:
        return 100.0
    if isinstance(sync_delta, int | float):
        return float(sync_delta)

    for key in _sensor_sync_keys(Path(sensor_folder).name):
        if key in sync_delta:
            return float(sync_delta[key])
    return 100.0


def _copy_frame_pair(
    *,
    sensor_folder: Path,
    output_folder: Path,
    frame_metadata: Mapping[str, Any],
    output_frame_id: str,
) -> tuple[Path, Path]:
    source_rgb = sensor_folder / str(frame_metadata["rgb_path"])
    source_depth = sensor_folder / str(frame_metadata["depth_path"])
    output_rgb = output_folder / RGB_DIR / output_frame_id
    output_depth = output_folder / DEPTH_DIR / output_frame_id
    output_rgb.parent.mkdir(parents=True, exist_ok=True)
    output_depth.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_rgb, output_rgb)
    shutil.copy2(source_depth, output_depth)
    return output_rgb, output_depth


def copy_sensor_metadata_artifacts(sensor_folder: Path, output_folder: Path) -> list[str]:
    copied = []
    for artifact in LEGACY_SENSOR_METADATA_ARTIFACTS:
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
    copy_files: bool = True,
) -> SyncResult:
    sensor_path = Path(sensor_folder)
    run_path = Path(run_root) if run_root is not None else sensor_path.parent
    output_base = (
        Path(output_root)
        if output_root is not None
        else run_path / PROCESSED_DIR / SYNCHRONIZED_DIR
    )
    output_folder = output_base / sensor_path.name
    output_folder.mkdir(parents=True, exist_ok=True)

    frame_records = load_frame_metadata(sensor_path)
    raw_robot_poses = load_robot_poses(run_path, sensor_path)
    robot_records = indexed_robot_poses(raw_robot_poses)
    windows = motion_windows(robot_records)
    sensor_sync_delta_ms = resolve_sync_delta_ms(sensor_path, sync_delta)
    sync_delta_ns = int(sensor_sync_delta_ms * 1_000_000)

    matched: dict[str, Any] = {}
    dropped: list[dict[str, Any]] = []
    nearest_deltas_ns: list[int] = []
    previous_frame_timestamp_ns: int | None = None
    output_counter = 0
    copied_metadata_artifacts = (
        copy_sensor_metadata_artifacts(sensor_path, output_folder) if copy_files else []
    )

    for frame_record in sorted(
        frame_records,
        key=lambda record: frame_timestamp_ns(record, timestamp_source) or 0,
    ):
        timestamp_ns = frame_timestamp_ns(frame_record, timestamp_source)
        if timestamp_ns is None:
            dropped.append(
                {
                    "frame_id": frame_record.get("frame_id"),
                    "reason": f"missing {timestamp_source} timestamp",
                }
            )
            continue

        delayed_timestamp_ns = timestamp_ns - sync_delta_ns
        motion = motion_for_timestamp(delayed_timestamp_ns, windows)
        if motion is None:
            dropped.append(
                {
                    "frame_id": frame_record.get("frame_id"),
                    "timestamp_ns": timestamp_ns,
                    "delayed_timestamp_ns": delayed_timestamp_ns,
                    "reason": "outside robot motion windows",
                }
            )
            continue

        closest_pose = closest_robot_pose(delayed_timestamp_ns, robot_records)
        nearest_delta_ns = int(closest_pose["timestamp_ns"]) - delayed_timestamp_ns
        nearest_deltas_ns.append(nearest_delta_ns)
        frame_delta_ns = (
            0
            if previous_frame_timestamp_ns is None
            else timestamp_ns - previous_frame_timestamp_ns
        )
        previous_frame_timestamp_ns = timestamp_ns

        extension = Path(str(frame_record.get("frame_id", "frame.png"))).suffix or ".png"
        output_frame_id = f"{output_counter:06d}{extension}"
        if copy_files:
            output_rgb, output_depth = _copy_frame_pair(
                sensor_folder=sensor_path,
                output_folder=output_folder,
                frame_metadata=frame_record,
                output_frame_id=output_frame_id,
            )
            synchronized_rgb = _relative_path(output_rgb, run_path)
            synchronized_depth = _relative_path(output_depth, run_path)
        else:
            synchronized_rgb = f"{RGB_DIR}/{output_frame_id}"
            synchronized_depth = f"{DEPTH_DIR}/{output_frame_id}"

        matched[output_frame_id] = {
            "motion": motion,
            "image_frame": timestamp_ns // 1_000_000,
            "image_timestamp_ns": timestamp_ns,
            "timestamp_source": timestamp_source,
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
            "source_frame_id": frame_record.get("frame_id"),
            "source_rgb": frame_record.get("rgb_path"),
            "source_depth": frame_record.get("depth_path"),
            "synchronized_rgb": synchronized_rgb,
            "synchronized_depth": synchronized_depth,
            "robot_ee_pose": closest_pose["pose"],
        }
        output_counter += 1

    matched_path = output_folder / MATCH_ROBOT_EE_POSES
    report_path = output_folder / SYNC_REPORT
    report = {
        "schema_version": "sync_report.v1",
        "sensor_folder": _relative_path(sensor_path, run_path),
        "output_folder": _relative_path(output_folder, run_path),
        "timestamp_source": timestamp_source,
        "sync_delta_ms": sensor_sync_delta_ms,
        "total_frames": len(frame_records),
        "matched_frames": len(matched),
        "dropped_frames": len(dropped),
        "motion_windows": windows,
        "dropped": dropped,
        "copied_metadata_artifacts": copied_metadata_artifacts,
        **_delta_stats(nearest_deltas_ns),
    }

    _write_json(matched_path, matched)
    _write_json(report_path, report)

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
    output_root: str | Path | None = None,
    sync_delta: int | float | Mapping[str, Any] | None = None,
    timestamp_source: str = "host_received",
    copy_files: bool = True,
) -> list[SyncResult]:
    run_path = Path(run_root)
    results = []

    for sensor_record in discover_sensor_records(run_path):
        sensor_folder = run_path / str(sensor_record["folder"])
        results.append(
            synchronize_sensor_folder(
                sensor_folder,
                run_root=run_path,
                output_root=output_root,
                sync_delta=sync_delta,
                timestamp_source=timestamp_source,
                copy_files=copy_files,
            )
        )

    return results
