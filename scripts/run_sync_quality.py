#!/usr/bin/env python3
"""Aggregate non-destructive synchronization quality for a run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.sync.quality import (
    build_sync_quality_report,
    write_sync_quality_report_with_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check synchronized sync_report.json files for dropped frames, "
            "match ratio, timestamp source, and nearest-pose deltas."
        )
    )
    parser.add_argument("run_root", help="Run folder containing processed sync output.")
    parser.add_argument(
        "--min-match-ratio",
        type=float,
        default=0.8,
        help="Warn when a sensor's matched/total frame ratio is below this value.",
    )
    parser.add_argument(
        "--max-dropped-frames",
        type=int,
        help="Warn when a sensor dropped more than this many frames.",
    )
    parser.add_argument(
        "--max-nearest-pose-delta-ms",
        type=float,
        default=50.0,
        help="Warn when max nearest robot-pose delta is above this threshold.",
    )
    parser.add_argument(
        "--no-nearest-pose-threshold",
        action="store_true",
        help="Do not check nearest robot-pose delta.",
    )
    parser.add_argument(
        "--require-timestamp-source",
        choices=("host_received", "host_wall", "sensor", "filename"),
        help="Warn when a sync report used a different timestamp source.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print a report without writing sync_quality_report.json.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_delta = (
        None
        if args.no_nearest_pose_threshold
        else args.max_nearest_pose_delta_ms
    )
    report_args = {
        "min_match_ratio": args.min_match_ratio,
        "max_dropped_frames": args.max_dropped_frames,
        "max_nearest_pose_delta_ms": max_delta,
        "require_timestamp_source": args.require_timestamp_source,
    }

    if args.no_write:
        report = build_sync_quality_report(Path(args.run_root), **report_args)
        path = None
    else:
        path, report = write_sync_quality_report_with_manifest(
            Path(args.run_root),
            **report_args,
        )

    if path is not None:
        print(f"Wrote {path}")
    print(
        "Sync quality: "
        f"{report['overall_status']} "
        f"({report['matched_frames']}/{report['total_frames']} frames matched, "
        f"{report['sensor_count']} sensors)"
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))

    if report["overall_status"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
