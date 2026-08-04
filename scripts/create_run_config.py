#!/usr/bin/env python3
"""Create a versioned PoseTestBot run configuration artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.config import DEFAULT_CAPTURE_VELOCITY_M_S
from posetestbot.pipeline.run_config import (
    CAPTURE_SYNCHRONIZATION_SCHEMA_VERSION,
    DEFAULT_MAX_DEPTH_TIMESTAMP_SKEW_MS,
    HARDWARE_TRIGGER_IMPLEMENTATION,
    HARDWARE_TRIGGER_SCOPE,
    capture_synchronization_from_mapping,
    create_run_config,
    default_lab_sensors,
    fixed_transform_from_mapping,
    sensor_config_from_token,
    sequence_plan_from_run_config,
    write_run_config_with_manifest,
)
from posetestbot.pipeline.sequences import PIPELINE_SEQUENCES
from posetestbot.robot.reference_frames import POSE_TEMPLATE_BASE_SUNRISE_PATH
from posetestbot.sensors.contracts import MountingMode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write run_config.json for a PoseTestBot run and record it in "
            "dataset_manifest.json."
        )
    )
    parser.add_argument("run_root", help="Run folder that will own run_config.json.")
    parser.add_argument("--run-name", default=None, help="Human-readable run name.")
    parser.add_argument(
        "--fixed-transform-json",
        action="append",
        default=[],
        help=(
            "Typed fixed frame edge as JSON with from, to, "
            "rotation_quaternion_wxyz, and translation_mm. May be repeated."
        ),
    )
    parser.add_argument(
        "--robot-pose-sunrise-reference-frame-path",
        default=POSE_TEMPLATE_BASE_SUNRISE_PATH,
        help=(
            "Exact absolute Sunrise Application Data frame path expected in "
            "robot_pose.v1 packets. New CLI runs default to the canonical "
            f"dataset world frame {POSE_TEMPLATE_BASE_SUNRISE_PATH}."
        ),
    )
    parser.add_argument("--resolution", default="720p")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument(
        "--velocity",
        type=float,
        default=DEFAULT_CAPTURE_VELOCITY_M_S,
        help=(
            "Requested capture-motion speed in m/s "
            f"(default {DEFAULT_CAPTURE_VELOCITY_M_S:g}; execution is capped "
            "independently)."
        ),
    )
    synchronization_source = parser.add_mutually_exclusive_group()
    synchronization_source.add_argument(
        "--synchronization-json",
        default=None,
        help=(
            "Capture synchronization JSON object. Use "
            '{"schema_version":"capture_synchronization.v1",'
            '"mode":"timestamp_aligned"} or the validated hardware_trigger form.'
        ),
    )
    synchronization_source.add_argument(
        "--synchronization-file",
        default=None,
        help="Path to a JSON file containing one capture synchronization object.",
    )
    parser.add_argument(
        "--hardware-trigger",
        action="store_true",
        help=(
            "Configure RealSense inter-camera depth-exposure triggering. Requires "
            "--hardware-sync-group-id and --hardware-sync-master-sensor."
        ),
    )
    parser.add_argument(
        "--hardware-sync-group-id",
        default=None,
        help="Safe identifier for the hardware-triggered RealSense camera group.",
    )
    parser.add_argument(
        "--hardware-sync-master-sensor",
        default=None,
        metavar="SENSOR_KEY",
        help=(
            "Exact enabled RealSense master key, for example "
            "realsense_d435:825412070181."
        ),
    )
    parser.add_argument(
        "--max-depth-timestamp-skew-ms",
        type=float,
        default=None,
        help=(
            "Maximum accepted earliest-to-latest depth timestamp span across "
            "every camera in one hardware-triggered group "
            f"(default {DEFAULT_MAX_DEPTH_TIMESTAMP_SKEW_MS} ms)."
        ),
    )
    parser.add_argument(
        "--sensor",
        action="append",
        default=None,
        help=(
            "Sensor entry sensor_type:device_id[:mounting_mode[:display_name[:orientation]]]. "
            "Use orientation inverted/normal for RealSense mounts. "
            "May be repeated. Defaults to the current lab profile."
        ),
    )
    parser.add_argument(
        "--mounting-mode",
        choices=tuple(mode.value for mode in MountingMode),
        default=MountingMode.EYE_IN_HAND.value,
        help="Default mounting mode for --sensor entries and lab defaults.",
    )
    parser.add_argument(
        "--dataset-mode",
        choices=("objectless", "pose_template"),
        default="objectless",
        help="Create an objectless run or one awaiting pose-template selection.",
    )
    parser.add_argument(
        "--calibration-profiles",
        default=None,
        help="Optional calibration profile collection path.",
    )
    parser.add_argument(
        "--sequence",
        choices=tuple(sorted(PIPELINE_SEQUENCES)),
        default="real_full_capture_validation",
        help="Default pipeline sequence ID for this run config.",
    )
    parser.add_argument(
        "--sequence-options-json",
        default=None,
        help="JSON object passed to the configured pipeline sequence.",
    )
    parser.add_argument(
        "--sequence-options-file",
        default=None,
        help="JSON options file merged after --sequence-options-json.",
    )
    parser.add_argument(
        "--execute-sequence",
        action="store_true",
        help="Store plan_only=false for the configured sequence.",
    )
    parser.add_argument(
        "--print-sequence-plan",
        action="store_true",
        help="Print the derived sequence plan JSON after writing the config.",
    )
    return parser.parse_args()


def load_sequence_options(
    *,
    options_json: str | None,
    options_file: str | None,
) -> dict:
    options: dict = {}
    if options_json:
        value = json.loads(options_json)
        if not isinstance(value, dict):
            raise ValueError("--sequence-options-json must decode to a JSON object")
        options.update(value)
    if options_file:
        with open(options_file, "r") as f:
            value = json.load(f)
        if not isinstance(value, dict):
            raise ValueError("--sequence-options-file must contain a JSON object")
        options.update(value)
    return options


def load_capture_synchronization(args: argparse.Namespace):
    """Resolve CLI synchronization input through the production validator."""

    hardware_flag_values = (
        args.hardware_trigger,
        args.hardware_sync_group_id is not None,
        args.hardware_sync_master_sensor is not None,
        args.max_depth_timestamp_skew_ms is not None,
    )
    if (
        args.synchronization_json is not None
        or args.synchronization_file is not None
    ) and any(hardware_flag_values):
        raise ValueError(
            "--synchronization-json/--synchronization-file cannot be combined "
            "with --hardware-trigger flags"
        )
    if args.synchronization_json is not None:
        value = json.loads(args.synchronization_json)
        if not isinstance(value, dict):
            raise ValueError("--synchronization-json must decode to a JSON object")
        return capture_synchronization_from_mapping(value)
    if args.synchronization_file is not None:
        with open(args.synchronization_file, "r") as f:
            value = json.load(f)
        if not isinstance(value, dict):
            raise ValueError("--synchronization-file must contain a JSON object")
        return capture_synchronization_from_mapping(value)
    if not any(hardware_flag_values):
        return capture_synchronization_from_mapping(None)
    if not args.hardware_trigger:
        raise ValueError(
            "--hardware-sync-group-id, --hardware-sync-master-sensor, and "
            "--max-depth-timestamp-skew-ms require --hardware-trigger"
        )
    if not args.hardware_sync_group_id or not args.hardware_sync_master_sensor:
        raise ValueError(
            "--hardware-trigger requires --hardware-sync-group-id and "
            "--hardware-sync-master-sensor"
        )
    return capture_synchronization_from_mapping(
        {
            "schema_version": CAPTURE_SYNCHRONIZATION_SCHEMA_VERSION,
            "mode": "hardware_trigger",
            "implementation": HARDWARE_TRIGGER_IMPLEMENTATION,
            "scope": HARDWARE_TRIGGER_SCOPE,
            "group_id": args.hardware_sync_group_id,
            "master_sensor_key": args.hardware_sync_master_sensor,
            "max_depth_timestamp_skew_ms": (
                args.max_depth_timestamp_skew_ms
                if args.max_depth_timestamp_skew_ms is not None
                else DEFAULT_MAX_DEPTH_TIMESTAMP_SKEW_MS
            ),
        }
    )


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    sequence_options = load_sequence_options(
        options_json=args.sequence_options_json,
        options_file=args.sequence_options_file,
    )
    sensors = (
        tuple(
            sensor_config_from_token(
                token,
                default_mounting_mode=args.mounting_mode,
            )
            for token in args.sensor
        )
        if args.sensor
        else default_lab_sensors(mounting_mode=args.mounting_mode)
    )
    synchronization = load_capture_synchronization(args)
    config = create_run_config(
        run_root=run_root,
        run_name=args.run_name,
        resolution=args.resolution,
        fps=args.fps,
        velocity_m_s=args.velocity,
        sensors=sensors,
        dataset_mode=args.dataset_mode,
        calibration_profiles=args.calibration_profiles,
        sequence_id=args.sequence,
        sequence_options=sequence_options,
        plan_only=not args.execute_sequence,
        fixed_transforms=tuple(
            fixed_transform_from_mapping(json.loads(value))
            for value in args.fixed_transform_json
        ),
        robot_pose_sunrise_reference_frame_path=(
            args.robot_pose_sunrise_reference_frame_path
        ),
        synchronization=synchronization,
    )
    path = write_run_config_with_manifest(run_root, config)

    print(f"Wrote {path}")
    if args.print_sequence_plan:
        plan = sequence_plan_from_run_config(config.to_dict())
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
