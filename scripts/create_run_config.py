#!/usr/bin/env python3
"""Create a versioned PoseTestBot run configuration artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.pipeline.run_config import (
    create_run_config,
    default_lab_sensors,
    fixed_transform_from_mapping,
    sensor_config_from_token,
    sequence_plan_from_run_config,
    write_run_config_with_manifest,
)
from posetestbot.pipeline.sequences import PIPELINE_SEQUENCES
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
    parser.add_argument("--resolution", default="720p")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--velocity", type=float, default=0.2)
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
        "--object-folder",
        default="object_models",
        help="Object registry folder for preparation and BOP export stages.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--object-name",
        action="append",
        default=None,
        help="Registry object to include. May be repeated; defaults to all valid objects.",
    )
    parser.add_argument(
        "--dataset-mode",
        choices=("objectless", "pose_template", "legacy_registry"),
        default=None,
        help=(
            "Explicit run dataset mode. Existing --object-name behavior maps to "
            "legacy_registry; use pose_template before selecting Ground Truth."
        ),
    )
    selection.add_argument(
        "--objectless",
        action="store_true",
        help="Snapshot an explicit objectless RGB-D run.",
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
    config = create_run_config(
        run_root=run_root,
        run_name=args.run_name,
        resolution=args.resolution,
        fps=args.fps,
        velocity_m_s=args.velocity,
        sensors=sensors,
        object_folder=args.object_folder,
        selected_objects=[] if args.objectless else args.object_name,
        dataset_mode=args.dataset_mode,
        calibration_profiles=args.calibration_profiles,
        sequence_id=args.sequence,
        sequence_options=sequence_options,
        plan_only=not args.execute_sequence,
        fixed_transforms=tuple(
            fixed_transform_from_mapping(json.loads(value))
            for value in args.fixed_transform_json
        ),
    )
    path = write_run_config_with_manifest(run_root, config)

    print(f"Wrote {path}")
    if args.print_sequence_plan:
        plan = sequence_plan_from_run_config(config.to_dict())
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
