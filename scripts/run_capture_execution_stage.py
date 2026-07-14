#!/usr/bin/env python3
"""Execute selected capture-plan commands through the capture supervisor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.pipeline.capture_execution import run_capture_execution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute selected capture-plan commands with process-group "
            "supervision for the real robot and configured cameras."
        )
    )
    parser.add_argument("run_root", help="Run folder containing run_config.json.")
    parser.add_argument(
        "--allow-cameras",
        action="store_true",
        help="Allow camera capture commands to execute.",
    )
    parser.add_argument(
        "--allow-real-robot",
        action="store_true",
        help="Allow a real-robot plan to pass the robot safety gate.",
    )
    parser.add_argument(
        "--include-sensors",
        action="store_true",
        help="Force SDK/device discovery checks before execution.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=30.0,
        help="Seconds to wait for the pose receiver command.",
    )
    parser.add_argument(
        "--startup-wait",
        type=float,
        default=0.2,
        help="Seconds to wait after background startup before pose receiver.",
    )
    parser.add_argument(
        "--terminate-timeout-s",
        type=float,
        default=2.0,
        help="Seconds to wait for background process termination.",
    )
    parser.add_argument(
        "--no-write-plan-if-missing",
        action="store_true",
        help="Do not create capture_plan.json when it is missing.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the full execution report JSON after writing it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_sensor_status = args.include_sensors
    path, report = run_capture_execution(
        Path(args.run_root),
        allow_cameras=args.allow_cameras,
        allow_real_robot=args.allow_real_robot,
        include_sensor_status=include_sensor_status,
        timeout_s=args.timeout_s,
        startup_wait_s=args.startup_wait,
        terminate_timeout_s=args.terminate_timeout_s,
        write_plan_if_missing=not args.no_write_plan_if_missing,
    )

    print(f"Wrote {path}")
    print(
        "Capture execution: "
        f"{report['status']} ({report['mode']}), "
        f"raw poses: {report['raw_pose_count']}"
    )
    if args.print_json:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
