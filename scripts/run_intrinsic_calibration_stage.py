#!/usr/bin/env python3
"""Wrap factory color intrinsics or solve them from stored ArUco detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.calibration.intrinsics import (
    DEFAULT_MAX_RMS_PX,
    DEFAULT_MAX_VIEW_ERROR_PX,
    DEFAULT_MIN_ACCEPTED_VIEWS,
    DEFAULT_MIN_COVERAGE_CELLS,
    calibrate_intrinsic_profile,
    factory_intrinsic_profile,
    IntrinsicCalibrationError,
    write_intrinsic_profile_collection,
)
from posetestbot.calibration.targets import load_calibration_target_spec
from posetestbot.io.artifacts import ARUCO_DETECTIONS, CALIBRATION_TARGET, INTRINSIC_CALIBRATION_PROFILES, PROCESSED_DIR, SYNCHRONIZED_DIR
from posetestbot.io.manifest import load_or_create_run_manifest, upsert_stage, write_run_manifest
from posetestbot.io.atomic import atomic_write_json
from posetestbot.calibration.intrinsics import SCHEMA_VERSION as INTRINSIC_SCHEMA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("--mode", choices=("factory", "calibrate"), default="factory")
    parser.add_argument("--calibration-target")
    parser.add_argument("--input-root")
    parser.add_argument("--min-accepted-views", type=int, default=DEFAULT_MIN_ACCEPTED_VIEWS)
    parser.add_argument("--min-coverage-cells", type=int, default=DEFAULT_MIN_COVERAGE_CELLS)
    parser.add_argument("--max-view-error-px", type=float, default=DEFAULT_MAX_VIEW_ERROR_PX)
    parser.add_argument("--max-rms-px", type=float, default=DEFAULT_MAX_RMS_PX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.run_root)
    input_root = Path(args.input_root) if args.input_root else root / PROCESSED_DIR / SYNCHRONIZED_DIR
    sensors = [path for path in sorted(input_root.iterdir()) if path.is_dir() and (path / ARUCO_DETECTIONS).is_file()]
    if not sensors:
        raise FileNotFoundError(f"No ArUco detection artifacts: {input_root}")
    target = None
    if args.mode == "calibrate":
        target_path = Path(args.calibration_target) if args.calibration_target else root / CALIBRATION_TARGET
        target = load_calibration_target_spec(target_path)
    manifest = load_or_create_run_manifest(root)
    upsert_stage(manifest, name="intrinsic_calibration", status="running")
    write_run_manifest(manifest, root)
    try:
        profiles = []
        failures = []
        for sensor in sensors:
            if args.mode == "factory":
                profiles.append(factory_intrinsic_profile(sensor))
            else:
                detections = json.loads((sensor / ARUCO_DETECTIONS).read_text())
                try:
                    profile = calibrate_intrinsic_profile(
                        sensor,
                        detections,
                        target,
                        min_accepted_views=args.min_accepted_views,
                        min_coverage_cells=args.min_coverage_cells,
                        max_view_error_px=args.max_view_error_px,
                        max_rms_px=args.max_rms_px,
                    )
                except IntrinsicCalibrationError as exc:
                    failures.append({"sensor_name": sensor.name, **exc.report})
                    continue
                profiles.append(profile)
        if failures:
            output = atomic_write_json(
                root / INTRINSIC_CALIBRATION_PROFILES,
                {
                    "schema_version": INTRINSIC_SCHEMA,
                    "profiles": profiles,
                    "failures": failures,
                },
            )
            raise ValueError(
                f"Intrinsic calibration quality gates failed for {len(failures)} sensor(s); see {output}"
            )
        output = write_intrinsic_profile_collection(profiles, root / INTRINSIC_CALIBRATION_PROFILES)
        upsert_stage(
            manifest,
            name="intrinsic_calibration",
            status="succeeded",
            artifacts={INTRINSIC_CALIBRATION_PROFILES: output},
            run_root=root,
        )
        write_run_manifest(manifest, root)
    except Exception as exc:
        upsert_stage(manifest, name="intrinsic_calibration", status="failed", message=str(exc))
        write_run_manifest(manifest, root)
        raise
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
