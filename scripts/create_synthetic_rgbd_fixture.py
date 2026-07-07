#!/usr/bin/env python3
"""Create a synthetic RGB-D sensor folder aligned to raw robot poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.pipeline.synthetic_rgbd import (
    DEFAULT_HEIGHT,
    DEFAULT_SENSOR_FOLDER,
    DEFAULT_SENSOR_ID,
    DEFAULT_SYNC_DELTA_MS,
    DEFAULT_WIDTH,
    write_synthetic_rgbd_fixture,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write a small manifest-tracked synthetic RGB-D capture folder from "
            "an existing raw_robot_ee_poses.json artifact."
        )
    )
    parser.add_argument("run_root", help="Run root containing raw_robot_ee_poses.json.")
    parser.add_argument("--sensor-folder", default=DEFAULT_SENSOR_FOLDER)
    parser.add_argument("--sensor-id", default=DEFAULT_SENSOR_ID)
    parser.add_argument("--frame-count", type=int, default=None)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--sync-delta-ms", type=float, default=DEFAULT_SYNC_DELTA_MS)
    parser.add_argument("--include-end-motion", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the JSON report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path, report = write_synthetic_rgbd_fixture(
        Path(args.run_root),
        sensor_folder_name=args.sensor_folder,
        sensor_id=args.sensor_id,
        frame_count=args.frame_count,
        width=args.width,
        height=args.height,
        sync_delta_ms=args.sync_delta_ms,
        include_end_motion=args.include_end_motion,
        overwrite=args.overwrite,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"Wrote {report['frame_count']} synthetic RGB-D frame(s) to "
            f"{report['sensor_folder']}"
        )
        print(report_path)


if __name__ == "__main__":
    main()
