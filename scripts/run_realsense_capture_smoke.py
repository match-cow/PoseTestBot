#!/usr/bin/env python3
"""Run a RealSense-only sequential RGB-D capture smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.sensors.realsense_smoke import (
    write_realsense_capture_smoke_with_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a RealSense-only run_config.json, then capture a short "
            "sequential RGB-D smoke sample from each configured RealSense."
        )
    )
    parser.add_argument("run_root", help="Run folder containing run_config.json.")
    parser.add_argument(
        "--expected-count",
        type=int,
        default=3,
        help="Expected number of configured and visible RealSense devices.",
    )
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show OpenCV previews during the sequential smoke capture.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the full smoke report JSON after writing it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path, report = write_realsense_capture_smoke_with_manifest(
        Path(args.run_root),
        expected_count=args.expected_count,
        fps=args.fps,
        max_frames=args.max_frames,
        warmup_frames=args.warmup_frames,
        preview=args.preview,
    )
    print(f"Wrote {path}")
    print(f"RealSense smoke: {report['status']} - {report['message']}")
    if args.print_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
