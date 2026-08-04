#!/usr/bin/env python3
"""Write a run-scoped read-only hardware status report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.pipeline.hardware_status import (
    build_hardware_status_report,
    write_hardware_status_report_with_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect robot profile, sensor visibility, and external runtime "
            "readiness into hardware_status_report.json without starting capture."
        )
    )
    parser.add_argument("run_root", help="Run folder for the hardware status report.")
    parser.add_argument(
        "--no-sensors",
        action="store_true",
        help="Skip camera SDK/device discovery checks.",
    )
    parser.add_argument(
        "--no-runtimes",
        action="store_true",
        help="Skip external runtime readiness checks.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the report without writing hardware_status_report.json.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full report JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_sensor_status = not args.no_sensors
    include_runtime_status = not args.no_runtimes
    run_root = Path(args.run_root)
    if args.plan_only:
        report = build_hardware_status_report(
            run_root,
            include_sensor_status=include_sensor_status,
            include_runtime_status=include_runtime_status,
        )
        print("Hardware status report: " f"{report['overall_status']} (plan-only)")
    else:
        path, report = write_hardware_status_report_with_manifest(
            run_root,
            include_sensor_status=include_sensor_status,
            include_runtime_status=include_runtime_status,
        )
        print(f"Wrote {path}")
        print(f"Hardware status report: {report['overall_status']}")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
