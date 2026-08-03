#!/usr/bin/env python3
"""Report and optionally retain iiwa pose-stream cadence evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.robot.cadence import (
    DEFAULT_MAXIMUM_GAP_MS,
    DEFAULT_MAXIMUM_P95_GAP_MS,
    DEFAULT_MINIMUM_MEDIAN_RATE_HZ,
    analyze_run_robot_pose_cadence,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--minimum-median-rate-hz",
        type=float,
        default=DEFAULT_MINIMUM_MEDIAN_RATE_HZ,
    )
    parser.add_argument(
        "--maximum-p95-gap-ms",
        type=float,
        default=DEFAULT_MAXIMUM_P95_GAP_MS,
    )
    parser.add_argument(
        "--maximum-gap-ms",
        type=float,
        default=DEFAULT_MAXIMUM_GAP_MS,
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write processed/robot_pose_cadence_report.json.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report, report_path = analyze_run_robot_pose_cadence(
        args.run_root,
        minimum_median_rate_hz=args.minimum_median_rate_hz,
        maximum_p95_gap_ms=args.maximum_p95_gap_ms,
        maximum_gap_ms=args.maximum_gap_ms,
        write=args.write,
    )
    print(json.dumps(report, indent=2, sort_keys=False))
    if report_path is not None:
        print(f"Wrote {report_path}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
