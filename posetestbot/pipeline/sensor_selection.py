"""Resolve enabled run-config sensors to their canonical capture folders."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from posetestbot.io.artifacts import RUN_CONFIG
from posetestbot.pipeline.run_config import load_run_config_for_run_root
from posetestbot.sensors.contracts import MountingMode
from posetestbot.sensors.registry import sensor_folder_name


def enabled_sensor_folder_names(
    run_root: str | Path,
) -> tuple[str, ...] | None:
    """Return enabled canonical folder names, or ``None`` for legacy runs.

    A missing run config predates the participation flag and therefore retains
    the historical discover-every-folder behavior.  Once a run config exists,
    it is authoritative and invalid configs fail through the normal loader.
    """

    root = Path(run_root)
    if not (root / RUN_CONFIG).is_file():
        return None
    config = load_run_config_for_run_root(root)
    return tuple(
        sensor_folder_name(str(sensor["sensor_type"]), str(sensor["device_id"]))
        for sensor in config["capture"]["sensors"]
        if sensor.get("enabled", True) is True
    )


def enabled_sensor_mounting_modes_by_folder(
    config: Mapping[str, Any] | None,
) -> dict[str, MountingMode] | None:
    """Return authoritative enabled sensor mounts from one run-config snapshot.

    ``None`` is reserved for legacy runs that have no run config.  Once a config
    exists, callers receive a complete folder mapping and must not silently fall
    back to mount-agnostic profile matching.
    """

    if config is None:
        return None
    capture = config.get("capture")
    if not isinstance(capture, Mapping):
        raise ValueError("run_config.capture must be an object")
    sensors = capture.get("sensors")
    if not isinstance(sensors, list):
        raise ValueError("run_config.capture.sensors must be an array")

    modes: dict[str, MountingMode] = {}
    for index, sensor in enumerate(sensors):
        if not isinstance(sensor, Mapping):
            raise ValueError(f"run_config.capture.sensors[{index}] must be an object")
        if sensor.get("enabled", True) is not True:
            continue
        folder = sensor_folder_name(
            str(sensor.get("sensor_type") or ""),
            str(sensor.get("device_id") or ""),
        )
        if folder in modes:
            raise ValueError(
                f"run_config.capture.sensors contains duplicate folder {folder!r}"
            )
        modes[folder] = MountingMode(str(sensor.get("mounting_mode") or ""))
    return modes


def filter_enabled_sensor_folders(
    run_root: str | Path,
    folders: Iterable[Path],
) -> list[Path]:
    """Filter discovered folders when run-config participation is available."""

    discovered = list(folders)
    enabled = enabled_sensor_folder_names(run_root)
    if enabled is None:
        return discovered
    enabled_names = set(enabled)
    return [folder for folder in discovered if folder.name in enabled_names]
