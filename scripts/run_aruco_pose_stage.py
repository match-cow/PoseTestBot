#!/usr/bin/env python3
"""Solve grid-to-camera poses from stored detections and selected native intrinsics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.aruco.grid import estimate_sensor_poses
from posetestbot.calibration.intrinsics import load_intrinsic_profile_collection, select_intrinsic_profile, sensor_intrinsic_identity
from posetestbot.calibration.targets import load_calibration_target_spec
from posetestbot.io.artifacts import ARUCO_DETECTIONS, ARUCO_POSE_ESTIMATION, CALIBRATION_TARGET, INTRINSIC_CALIBRATION_PROFILES, PROCESSED_DIR, SYNCHRONIZED_DIR
from posetestbot.io.manifest import load_or_create_run_manifest, upsert_stage, write_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("--calibration-target")
    parser.add_argument("--intrinsic-profiles")
    parser.add_argument("--input-root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.run_root)
    input_root = Path(args.input_root) if args.input_root else root / PROCESSED_DIR / SYNCHRONIZED_DIR
    target_path = Path(args.calibration_target) if args.calibration_target else root / CALIBRATION_TARGET
    profiles_path = Path(args.intrinsic_profiles) if args.intrinsic_profiles else root / INTRINSIC_CALIBRATION_PROFILES
    target = load_calibration_target_spec(target_path)
    profiles = load_intrinsic_profile_collection(profiles_path)
    sensors = [path for path in sorted(input_root.iterdir()) if path.is_dir() and (path / ARUCO_DETECTIONS).is_file()]
    if not sensors:
        raise FileNotFoundError(f"No ArUco detection artifacts: {input_root}")
    manifest = load_or_create_run_manifest(root)
    upsert_stage(manifest, name="aruco_pose", status="running")
    write_run_manifest(manifest, root)
    try:
        artifacts = {}
        for sensor in sensors:
            sensor_id, orientation, resolution = sensor_intrinsic_identity(sensor)
            profile = select_intrinsic_profile(profiles, sensor_id=sensor_id, resolution=resolution, orientation=orientation)
            detections = json.loads((sensor / ARUCO_DETECTIONS).read_text())
            estimate_sensor_poses(sensor, detections, target, profile)
            artifacts[f"{sensor.name}:{ARUCO_POSE_ESTIMATION}"] = sensor / ARUCO_POSE_ESTIMATION
        upsert_stage(manifest, name="aruco_pose", status="succeeded", artifacts=artifacts, run_root=root)
        write_run_manifest(manifest, root)
    except Exception as exc:
        upsert_stage(manifest, name="aruco_pose", status="failed", message=str(exc))
        write_run_manifest(manifest, root)
        raise
    print(f"Solved ArUco grid poses for {len(sensors)} sensor folder(s).")


if __name__ == "__main__":
    main()
