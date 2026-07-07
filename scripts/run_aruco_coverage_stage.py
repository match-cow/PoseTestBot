#!/usr/bin/env python3
"""Summarize ArUco detection and valid-pose coverage for a run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.aruco.coverage import write_aruco_coverage_report_with_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build aruco_coverage_report.json from synchronized "
            "aruco_pose_estimation.json files."
        )
    )
    parser.add_argument("run_root", help="Run root containing synchronized ArUco outputs.")
    parser.add_argument(
        "--min-marker-count",
        type=int,
        default=4,
        help="Minimum marker count required for a frame to count as a valid pose.",
    )
    parser.add_argument(
        "--min-valid-pose-ratio",
        type=float,
        default=0.0,
        help="Minimum per-sensor valid-pose ratio before the sensor check is OK.",
    )
    parser.add_argument(
        "--aruco-pose-file",
        action="append",
        default=None,
        help=(
            "Specific aruco_pose_estimation.json file to summarize. May be "
            "repeated. Defaults to all synchronized sensor outputs."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full coverage report JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = write_aruco_coverage_report_with_manifest(
        Path(args.run_root),
        min_marker_count=args.min_marker_count,
        min_valid_pose_ratio=args.min_valid_pose_ratio,
        aruco_paths=args.aruco_pose_file,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "ArUco coverage "
            f"{report['overall_status']}: "
            f"{report['valid_pose_count']}/{report['frame_count']} valid pose frame(s)."
        )


if __name__ == "__main__":
    main()
