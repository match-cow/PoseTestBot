#!/usr/bin/env python3
"""Build calibration observation datasets from synchronized target-pose outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.calibration.observations import (
    build_calibration_observations,
    write_calibration_observations_with_manifest,
)
from posetestbot.calibration.targets import load_calibration_target_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract solver-ready calibration observations from "
            "processed/synchronized/* target-pose estimation JSON files."
        )
    )
    parser.add_argument(
        "run_root",
        help="Run folder containing synchronized target-pose outputs.",
    )
    parser.add_argument(
        "--min-marker-count",
        type=int,
        default=4,
        help=(
            "Minimum detected marker/corner/feature count for a frame to "
            "become an observation."
        ),
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=6,
        help="Recommended minimum usable observation count per sensor.",
    )
    parser.add_argument(
        "--aruco-path",
        action="append",
        default=None,
        help=(
            "Explicit target-pose JSON path. Repeat to process a sensor subset; "
            "omit to preserve run-wide discovery."
        ),
    )
    parser.add_argument(
        "--output-root",
        help=(
            "Alternate directory for calibration_observations.json. Omit to "
            "preserve the canonical run-root output."
        ),
    )
    parser.add_argument(
        "--target-spec",
        default="calibration_target.json",
        help=(
            "Current calibration_target.v2 JSON. Relative paths are "
            "resolved from the run root when not found from the current directory."
        ),
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print a report without writing calibration_observations.json.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON report.",
    )
    return parser.parse_args()


def _target_spec_from_args(args: argparse.Namespace) -> dict:
    target_path = Path(args.target_spec)
    if not target_path.is_absolute() and not target_path.exists():
        target_path = Path(args.run_root) / target_path
    return load_calibration_target_spec(target_path)


def main() -> None:
    args = parse_args()
    target = _target_spec_from_args(args)
    report_args = {
        "min_marker_count": args.min_marker_count,
        "min_observations": args.min_observations,
        "aruco_paths": args.aruco_path,
        "target": target,
    }
    if args.no_write:
        report = build_calibration_observations(Path(args.run_root), **report_args)
        path = None
    else:
        path, report = write_calibration_observations_with_manifest(
            Path(args.run_root),
            output_root=args.output_root,
            **report_args,
        )

    if path is not None:
        print(f"Wrote {path}")
    print(
        "Calibration observations: "
        f"{report['overall_status']} "
        f"({report['observation_count']} usable / {report['frame_count']} frames, "
        f"{report['sensor_count']} sensors)"
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))

    if report["overall_status"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
