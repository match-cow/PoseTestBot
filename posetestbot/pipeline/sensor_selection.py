"""Resolve enabled run-config sensors to their canonical capture folders."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from posetestbot.io.artifacts import RUN_CONFIG
from posetestbot.pipeline.run_config import load_run_config_for_run_root
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
