#!/usr/bin/env python3
"""Validate capture_plan.json before starting capture processes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.pipeline.capture_plan_preflight import (
    build_capture_plan_preflight,
    write_capture_plan_preflight_with_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate capture_plan.json command shape, real robot safety, "
            "scripts, and optional sensor readiness without launching capture."
        )
    )
    parser.add_argument("run_root", help="Run folder containing run_config.json.")
    parser.add_argument(
        "--allow-real-robot",
        action="store_true",
        help="Allow a real-robot capture plan to pass the safety gate.",
    )
    parser.add_argument(
        "--no-sensors",
        action="store_true",
        help="Skip SDK/device discovery checks.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print a report without writing capture_plan_preflight_report.json.",
    )
    parser.add_argument(
        "--no-write-plan-if-missing",
        action="store_true",
        help="Do not create capture_plan.json when it is missing.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    report_args = {
        "include_sensor_status": not args.no_sensors,
        "allow_real_robot": args.allow_real_robot,
        "write_plan_if_missing": not args.no_write_plan_if_missing,
    }

    if args.no_write:
        report = build_capture_plan_preflight(run_root, **report_args)
        path = None
    else:
        path, report = write_capture_plan_preflight_with_manifest(
            run_root,
            **report_args,
        )

    if path is not None:
        print(f"Wrote {path}")
    print(f"Capture plan preflight: {report['overall_status']}")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))

    if report["overall_status"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
