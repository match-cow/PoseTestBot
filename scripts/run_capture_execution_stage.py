#!/usr/bin/env python3
"""Execute selected capture-plan commands through the capture supervisor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.pipeline.capture_execution import (
    DEFAULT_CAMERA_READINESS_TIMEOUT_S,
    DEFAULT_CAMERA_STARTUP_ATTEMPTS,
    DEFAULT_CAMERA_STARTUP_RETRY_DELAY_S,
    DEFAULT_CAPTURE_EXECUTION_TIMEOUT_S,
    MAX_EXPLICIT_CAMERA_METADATA_IDLE_TIMEOUT_S,
    run_capture_execution,
)
from posetestbot.robot.pose_receiver import (
    DEFAULT_RECEIVE_IDLE_TIMEOUT_S,
    DEFAULT_RECEIVE_START_TIMEOUT_S,
)


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
        default=DEFAULT_CAPTURE_EXECUTION_TIMEOUT_S,
        help="Seconds to wait for the pose receiver command.",
    )
    parser.add_argument(
        "--startup-wait",
        type=float,
        default=DEFAULT_CAMERA_READINESS_TIMEOUT_S,
        help=(
            "Maximum seconds per startup attempt to wait for the current camera "
            "to publish the required valid committed frame_metadata.jsonl records."
        ),
    )
    parser.add_argument(
        "--camera-startup-attempts",
        type=int,
        default=DEFAULT_CAMERA_STARTUP_ATTEMPTS,
        help=(
            "Maximum startup attempts per camera. A retry is allowed only when "
            "the failed attempt left no sensor output evidence."
        ),
    )
    parser.add_argument(
        "--camera-startup-retry-delay-s",
        type=float,
        default=DEFAULT_CAMERA_STARTUP_RETRY_DELAY_S,
        help="Seconds to wait between safe camera startup attempts.",
    )
    parser.add_argument(
        "--terminate-timeout-s",
        type=float,
        default=2.0,
        help="Seconds to wait for background process termination.",
    )
    parser.add_argument(
        "--receive-start-timeout-s",
        type=float,
        default=DEFAULT_RECEIVE_START_TIMEOUT_S,
        help="Seconds the pose receiver waits for its first robot packet.",
    )
    parser.add_argument(
        "--receive-idle-timeout-s",
        type=float,
        default=DEFAULT_RECEIVE_IDLE_TIMEOUT_S,
        help="Seconds the pose receiver waits between robot packets.",
    )
    parser.add_argument(
        "--camera-metadata-idle-timeout-s",
        type=float,
        default=None,
        help=(
            "Maximum seconds a live camera may stop appending valid frame "
            "metadata. By default this is derived from capture FPS and bounded "
            "to a few seconds; explicit values must not exceed "
            f"{MAX_EXPLICIT_CAMERA_METADATA_IDLE_TIMEOUT_S:g} seconds."
        ),
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
        camera_startup_attempts=args.camera_startup_attempts,
        camera_startup_retry_delay_s=args.camera_startup_retry_delay_s,
        terminate_timeout_s=args.terminate_timeout_s,
        receive_start_timeout_s=args.receive_start_timeout_s,
        receive_idle_timeout_s=args.receive_idle_timeout_s,
        camera_metadata_idle_timeout_s=(
            args.camera_metadata_idle_timeout_s
        ),
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
