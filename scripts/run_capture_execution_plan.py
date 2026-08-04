#!/usr/bin/env python3
"""Write a manifest-tracked capture execution command-selection plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.pipeline.capture_execution import (
    build_capture_execution_plan,
    write_capture_execution_plan_with_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select full capture-plan commands without "
            "launching robot or camera processes."
        )
    )
    parser.add_argument("run_root", help="Run folder containing run_config.json.")
    parser.add_argument(
        "--allow-cameras",
        action="store_true",
        help="Allow camera capture commands to be selected.",
    )
    parser.add_argument(
        "--allow-real-robot",
        action="store_true",
        help="Allow a real-robot plan to pass the robot safety gate.",
    )
    parser.add_argument(
        "--include-sensors",
        action="store_true",
        help=(
            "Force SDK/device discovery checks."
        ),
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print a plan without writing capture_execution_plan.json.",
    )
    parser.add_argument(
        "--no-write-plan-if-missing",
        action="store_true",
        help="Do not create capture_plan.json when it is missing.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON plan.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    include_sensor_status = args.include_sensors
    plan_args = {
        "allow_cameras": args.allow_cameras,
        "allow_real_robot": args.allow_real_robot,
        "include_sensor_status": include_sensor_status,
        "write_plan_if_missing": not args.no_write_plan_if_missing,
    }

    if args.no_write:
        plan = build_capture_execution_plan(run_root, **plan_args)
        path = None
    else:
        path, plan = write_capture_execution_plan_with_manifest(
            run_root,
            **plan_args,
        )

    if path is not None:
        print(f"Wrote {path}")
    print(f"Capture execution plan: {plan['status']} (full)")
    print(
        "Selected "
        f"{len(plan['selected_commands'])} command(s), skipped "
        f"{len(plan['skipped_commands'])}."
    )
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))

    if plan["status"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
