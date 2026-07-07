#!/usr/bin/env python3
"""Run ArUco estimation as a manifest-tracked pipeline stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from aruco_pose_estimation import process_sensor_folder

from posetestbot.io.artifacts import (
    ARUCO_POSE_ESTIMATION,
    MATCH_ROBOT_EE_POSES,
    PROCESSED_DIR,
    RGB_DIR,
    SYNCHRONIZED_DIR,
)
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    upsert_stage,
    write_run_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ArUco pose estimation on one synchronized sensor folder or all "
            "synchronized sensors in a run."
        )
    )
    parser.add_argument(
        "target",
        help=(
            "Synchronized sensor folder, or run root containing "
            "processed/synchronized/<sensor> folders."
        ),
    )
    parser.add_argument(
        "--run-root",
        default=None,
        help="Run root for manifest updates when target is a sensor folder.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save ArUco visualization images beside the synchronized frames.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display OpenCV windows while processing.",
    )
    parser.add_argument(
        "--wait-time",
        type=int,
        default=1,
        help="OpenCV wait time in ms when displaying frames.",
    )
    return parser.parse_args()


def is_sensor_folder(path: Path) -> bool:
    return (path / RGB_DIR).is_dir() and (path / MATCH_ROBOT_EE_POSES).is_file()


def infer_run_root(sensor_folder: Path) -> Path:
    if (
        sensor_folder.parent.name == SYNCHRONIZED_DIR
        and sensor_folder.parent.parent.name == PROCESSED_DIR
    ):
        return sensor_folder.parent.parent.parent
    return sensor_folder.parent


def target_sensor_folders(target: Path) -> list[Path]:
    if is_sensor_folder(target):
        return [target]

    synchronized_root = target / PROCESSED_DIR / SYNCHRONIZED_DIR
    if not synchronized_root.is_dir():
        raise FileNotFoundError(
            f"Could not find synchronized sensor root: {synchronized_root}"
        )

    folders = [folder for folder in sorted(synchronized_root.iterdir()) if is_sensor_folder(folder)]
    if not folders:
        raise FileNotFoundError(f"No synchronized sensor folders found in {synchronized_root}")
    return folders


def run_aruco_stage(
    sensor_folder: Path,
    *,
    run_root: Path,
    save_images: bool,
    quiet: bool,
    wait_time: int,
) -> Path:
    manifest = load_or_create_run_manifest(run_root)
    stage_name = f"aruco:{sensor_folder.name}"
    upsert_stage(manifest, name=stage_name, status="running")
    write_run_manifest(manifest, run_root)

    try:
        import cv2 as cv

        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_50)
        board = cv.aruco.GridBoard((4, 3), 50, 65, aruco_dict)
        process_sensor_folder(
            str(sensor_folder),
            aruco_dict,
            board,
            save_images,
            quiet,
            wait_time,
        )
    except Exception as exc:
        upsert_stage(manifest, name=stage_name, status="failed", message=str(exc))
        write_run_manifest(manifest, run_root)
        raise

    output_path = sensor_folder / ARUCO_POSE_ESTIMATION
    artifacts = {ARUCO_POSE_ESTIMATION: output_path}
    if save_images:
        artifacts["aruco_images"] = sensor_folder / "aruco"

    upsert_stage(
        manifest,
        name=stage_name,
        status="succeeded",
        artifacts=artifacts,
        run_root=run_root,
    )
    write_run_manifest(manifest, run_root)
    return output_path


def main() -> None:
    args = parse_args()
    target = Path(args.target)
    sensor_folders = target_sensor_folders(target)
    run_root = Path(args.run_root) if args.run_root else infer_run_root(sensor_folders[0])
    quiet = not args.show

    outputs = [
        run_aruco_stage(
            sensor_folder,
            run_root=run_root,
            save_images=args.save_images,
            quiet=quiet,
            wait_time=args.wait_time,
        )
        for sensor_folder in sensor_folders
    ]

    print(f"ArUco finished for {len(outputs)} sensor folder(s).")


if __name__ == "__main__":
    main()

