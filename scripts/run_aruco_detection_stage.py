#!/usr/bin/env python3
"""Detect an imported ArUco grid once in synchronized native RGB frames."""

from __future__ import annotations

import argparse
from pathlib import Path

from posetestbot.aruco.grid import detect_sensor_folder, draw_detection_images
from posetestbot.calibration.targets import load_calibration_target_spec
from posetestbot.io.artifacts import ARUCO_DETECTIONS, CALIBRATION_TARGET, PROCESSED_DIR, RGB_DIR, SYNCHRONIZED_DIR
from posetestbot.io.manifest import load_or_create_run_manifest, upsert_stage, write_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("--calibration-target")
    parser.add_argument("--input-root")
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.run_root)
    target_path = Path(args.calibration_target) if args.calibration_target else root / CALIBRATION_TARGET
    target = load_calibration_target_spec(target_path)
    input_root = Path(args.input_root) if args.input_root else root / PROCESSED_DIR / SYNCHRONIZED_DIR
    sensors = [path for path in sorted(input_root.iterdir()) if path.is_dir() and (path / RGB_DIR).is_dir()]
    if not sensors:
        raise FileNotFoundError(f"No synchronized RGB sensor folders: {input_root}")
    manifest = load_or_create_run_manifest(root)
    upsert_stage(manifest, name="aruco_detection", status="running")
    write_run_manifest(manifest, root)
    try:
        artifacts = {}
        for sensor in sensors:
            detections = detect_sensor_folder(sensor, target)
            artifacts[f"{sensor.name}:{ARUCO_DETECTIONS}"] = sensor / ARUCO_DETECTIONS
            if args.save_images:
                artifacts[f"{sensor.name}:aruco_images"] = draw_detection_images(sensor, detections)
        upsert_stage(manifest, name="aruco_detection", status="succeeded", artifacts=artifacts, run_root=root)
        write_run_manifest(manifest, root)
    except Exception as exc:
        upsert_stage(manifest, name="aruco_detection", status="failed", message=str(exc))
        write_run_manifest(manifest, root)
        raise
    print(f"Detected ArUco grid in {len(sensors)} sensor folder(s).")


if __name__ == "__main__":
    main()
