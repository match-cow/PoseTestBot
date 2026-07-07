#!/usr/bin/env python3
"""Non-destructive synchronization CLI for PoseTestBot runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    upsert_stage,
    write_run_manifest,
)
from posetestbot.sync.non_destructive import (
    sync_result_artifacts,
    synchronize_sensor_folder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize one sensor folder into processed/synchronized without "
            "modifying raw rgb/depth frames."
        )
    )
    parser.add_argument("sensor_folder", help="Sensor folder containing rgb/depth frames.")
    parser.add_argument(
        "--run-root",
        default=None,
        help="Run root folder. Defaults to the parent of sensor_folder.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Derived sync output root. Defaults to <run-root>/processed/synchronized.",
    )
    parser.add_argument(
        "--sync-delta",
        default=None,
        help="Sync delta in ms, or a JSON file mapping sensor types to ms.",
    )
    parser.add_argument(
        "--timestamp-source",
        choices=("host_received", "host_wall", "sensor", "filename"),
        default="host_received",
        help="Timestamp source used for matching frames to robot poses.",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Write metadata only without copying rgb/depth frames.",
    )
    return parser.parse_args()


def load_sync_delta(value: str | None):
    if value is None:
        return None

    path = Path(value)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)

    return float(value)


def main() -> None:
    args = parse_args()
    sensor_folder = Path(args.sensor_folder)
    run_root = Path(args.run_root) if args.run_root else sensor_folder.parent
    sync_delta = load_sync_delta(args.sync_delta)

    manifest = load_or_create_run_manifest(run_root)
    stage_name = f"sync:{sensor_folder.name}"
    upsert_stage(manifest, name=stage_name, status="running")
    write_run_manifest(manifest, run_root)

    result = synchronize_sensor_folder(
        sensor_folder,
        run_root=run_root,
        output_root=args.output_root,
        sync_delta=sync_delta,
        timestamp_source=args.timestamp_source,
        copy_files=not args.no_copy,
    )

    upsert_stage(
        manifest,
        name=stage_name,
        status="succeeded",
        artifacts=sync_result_artifacts(result),
        run_root=run_root,
        message=(
            f"Matched {result.matched_frames}/{result.total_frames} frames; "
            f"dropped {result.dropped_frames}."
        ),
    )
    write_run_manifest(manifest, run_root)

    print(
        f"Matched {result.matched_frames}/{result.total_frames} frames "
        f"into {result.output_folder}"
    )


if __name__ == "__main__":
    main()

