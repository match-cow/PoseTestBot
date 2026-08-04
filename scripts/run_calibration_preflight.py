#!/usr/bin/env python3
"""Validate calibration profile coverage for a configured run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.calibration.preflight import (
    build_calibration_preflight,
    write_calibration_preflight_with_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check that run_config.json calibration profiles cover each enabled "
            "sensor and carry usable validation metrics."
        )
    )
    parser.add_argument("run_root", help="Run folder containing run_config.json.")
    parser.add_argument(
        "--require-valid",
        action="store_true",
        help="Treat non-valid profile status as an error instead of a warning.",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=6,
        help="Recommended minimum calibration observation count.",
    )
    parser.add_argument(
        "--max-mean-reprojection-error-px",
        type=float,
        default=2.0,
        help="Warn when mean reprojection error is absent or above this value.",
    )
    parser.add_argument(
        "--no-reprojection-threshold",
        action="store_true",
        help="Do not check mean reprojection error.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print a report without writing calibration_preflight_report.json.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    threshold = (
        None
        if args.no_reprojection_threshold
        else args.max_mean_reprojection_error_px
    )
    report_args = {
        "require_valid": args.require_valid,
        "min_observations": args.min_observations,
        "max_mean_reprojection_error_px": threshold,
    }

    if args.no_write:
        report = build_calibration_preflight(Path(args.run_root), **report_args)
        path = None
    else:
        path, report = write_calibration_preflight_with_manifest(
            Path(args.run_root),
            **report_args,
        )

    if path is not None:
        print(f"Wrote {path}")
    print(
        "Calibration preflight: "
        f"{report['overall_status']} "
        f"({report['matched_sensor_count']}/{report['sensor_count']} sensors matched)"
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))

    if report["overall_status"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
