#!/usr/bin/env python3
"""Synchronize every discovered sensor folder in a run non-destructively."""

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
    synchronize_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize every raw sensor folder in a run into "
            "processed/synchronized without modifying raw frames."
        )
    )
    parser.add_argument("run_root", help="Run root containing sensor folders.")
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
    run_root = Path(args.run_root)
    sync_delta = load_sync_delta(args.sync_delta)

    manifest = load_or_create_run_manifest(run_root)
    upsert_stage(manifest, name="sync_run", status="running")
    write_run_manifest(manifest, run_root)

    results = synchronize_run(
        run_root,
        output_root=args.output_root,
        sync_delta=sync_delta,
        timestamp_source=args.timestamp_source,
        copy_files=not args.no_copy,
    )

    for result in results:
        sensor_name = Path(result.sensor_folder).name
        upsert_stage(
            manifest,
            name=f"sync:{sensor_name}",
            status="succeeded",
            artifacts=sync_result_artifacts(result),
            run_root=run_root,
            message=(
                f"Matched {result.matched_frames}/{result.total_frames} frames; "
                f"dropped {result.dropped_frames}."
            ),
        )

    total_frames = sum(result.total_frames for result in results)
    matched_frames = sum(result.matched_frames for result in results)
    dropped_frames = sum(result.dropped_frames for result in results)
    upsert_stage(
        manifest,
        name="sync_run",
        status="succeeded",
        message=(
            f"Synchronized {len(results)} sensor(s): matched "
            f"{matched_frames}/{total_frames}, dropped {dropped_frames}."
        ),
    )
    write_run_manifest(manifest, run_root)

    print(
        f"Synchronized {len(results)} sensor(s): matched "
        f"{matched_frames}/{total_frames}, dropped {dropped_frames}."
    )


if __name__ == "__main__":
    main()

