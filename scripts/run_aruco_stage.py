#!/usr/bin/env python3
"""Run ArUco estimation as a manifest-tracked pipeline stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from posetestbot.aruco.grid import (
    detect_sensor_folder,
    draw_detection_images,
    estimate_sensor_poses,
)
from posetestbot.calibration.intrinsics import (
    DEFAULT_MAX_RMS_PX,
    DEFAULT_MAX_VIEW_ERROR_PX,
    DEFAULT_MIN_ACCEPTED_VIEWS,
    DEFAULT_MIN_COVERAGE_CELLS,
    calibrate_intrinsic_profile,
    factory_intrinsic_profile,
    load_intrinsic_profile_collection,
    write_intrinsic_profile_collection,
)
from posetestbot.calibration.targets import (
    load_calibration_target_spec,
    normalize_calibration_target_spec,
)
from posetestbot.io.artifacts import (
    ARUCO_DETECTIONS,
    ARUCO_POSE_ESTIMATION,
    CALIBRATION_TARGET,
    INTRINSIC_CALIBRATION_PROFILES,
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
from posetestbot.pipeline.sensor_selection import filter_enabled_sensor_folders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ArUco pose estimation on one synchronized sensor folder or all "
            "synchronized sensors in a run."
        )
    )
    parser.add_argument(
        "--calibration-target",
        help=(
            "Current calibration_target.v2 JSON. Defaults to the run-owned "
            "calibration_target.json."
        ),
    )
    parser.add_argument(
        "--intrinsics-mode",
        choices=("factory", "calibrate"),
        default="factory",
        help="Wrap SDK color intrinsics or calibrate them from GridBoard views.",
    )
    parser.add_argument(
        "--min-accepted-views", type=int, default=DEFAULT_MIN_ACCEPTED_VIEWS
    )
    parser.add_argument(
        "--min-coverage-cells", type=int, default=DEFAULT_MIN_COVERAGE_CELLS
    )
    parser.add_argument(
        "--max-view-error-px", type=float, default=DEFAULT_MAX_VIEW_ERROR_PX
    )
    parser.add_argument("--max-rms-px", type=float, default=DEFAULT_MAX_RMS_PX)
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


def target_sensor_folders(
    target: Path,
    *,
    run_root: Path | None = None,
) -> list[Path]:
    if is_sensor_folder(target):
        return [target]

    synchronized_root = target / PROCESSED_DIR / SYNCHRONIZED_DIR
    if not synchronized_root.is_dir():
        raise FileNotFoundError(
            f"Could not find synchronized sensor root: {synchronized_root}"
        )

    folders = filter_enabled_sensor_folders(
        run_root or target,
        (
            folder
            for folder in sorted(synchronized_root.iterdir())
            if is_sensor_folder(folder)
        ),
    )
    if not folders:
        raise FileNotFoundError(
            f"No synchronized sensor folders found in {synchronized_root}"
        )
    return folders


def run_aruco_stage(
    sensor_folder: Path,
    *,
    run_root: Path,
    save_images: bool,
    quiet: bool,
    wait_time: int,
    target: dict | None = None,
    intrinsics_mode: str = "factory",
    min_accepted_views: int = DEFAULT_MIN_ACCEPTED_VIEWS,
    min_coverage_cells: int = DEFAULT_MIN_COVERAGE_CELLS,
    max_view_error_px: float = DEFAULT_MAX_VIEW_ERROR_PX,
    max_rms_px: float = DEFAULT_MAX_RMS_PX,
) -> Path:
    manifest = load_or_create_run_manifest(run_root)
    stage_name = f"aruco:{sensor_folder.name}"
    upsert_stage(manifest, name=stage_name, status="running")
    write_run_manifest(manifest, run_root)

    try:
        if target is None:
            raise ValueError("A current calibration_target.v2 target is required")
        target = normalize_calibration_target_spec(target)
        detections = detect_sensor_folder(sensor_folder, target)
        if intrinsics_mode == "calibrate":
            if not target.get("generator_source"):
                raise ValueError(
                    "--intrinsics-mode calibrate requires a selected PoseGridGen calibration target"
                )
            intrinsic_profile = calibrate_intrinsic_profile(
                sensor_folder,
                detections,
                target,
                min_accepted_views=min_accepted_views,
                min_coverage_cells=min_coverage_cells,
                max_view_error_px=max_view_error_px,
                max_rms_px=max_rms_px,
            )
        else:
            intrinsic_profile = factory_intrinsic_profile(sensor_folder)
        intrinsic_path = run_root / INTRINSIC_CALIBRATION_PROFILES
        existing = (
            load_intrinsic_profile_collection(intrinsic_path)
            if intrinsic_path.is_file()
            else []
        )
        identity = (
            intrinsic_profile["sensor_id"],
            tuple(intrinsic_profile["resolution"]),
            intrinsic_profile["orientation"],
        )
        retained = [
            profile
            for profile in existing
            if (
                profile["sensor_id"],
                tuple(profile["resolution"]),
                profile["orientation"],
            )
            != identity
        ]
        write_intrinsic_profile_collection(
            [*retained, intrinsic_profile], intrinsic_path
        )
        estimate_sensor_poses(sensor_folder, detections, target, intrinsic_profile)
        if save_images or not quiet:
            draw_detection_images(
                sensor_folder,
                detections,
                show=not quiet,
                wait_time=wait_time,
            )
    except Exception as exc:
        upsert_stage(manifest, name=stage_name, status="failed", message=str(exc))
        write_run_manifest(manifest, run_root)
        raise

    output_path = sensor_folder / ARUCO_POSE_ESTIMATION
    artifacts = {
        ARUCO_DETECTIONS: sensor_folder / ARUCO_DETECTIONS,
        ARUCO_POSE_ESTIMATION: output_path,
        INTRINSIC_CALIBRATION_PROFILES: run_root / INTRINSIC_CALIBRATION_PROFILES,
    }
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
    requested_run_root = Path(args.run_root) if args.run_root else None
    sensor_folders = target_sensor_folders(target, run_root=requested_run_root)
    run_root = (
        Path(args.run_root) if args.run_root else infer_run_root(sensor_folders[0])
    )
    quiet = not args.show
    target_path = (
        Path(args.calibration_target)
        if args.calibration_target
        else run_root / CALIBRATION_TARGET
    )
    target = load_calibration_target_spec(target_path)

    outputs = [
        run_aruco_stage(
            sensor_folder,
            run_root=run_root,
            save_images=args.save_images,
            quiet=quiet,
            wait_time=args.wait_time,
            target=target,
            intrinsics_mode=args.intrinsics_mode,
            min_accepted_views=args.min_accepted_views,
            min_coverage_cells=args.min_coverage_cells,
            max_view_error_px=args.max_view_error_px,
            max_rms_px=args.max_rms_px,
        )
        for sensor_folder in sensor_folders
    ]

    print(f"ArUco finished for {len(outputs)} sensor folder(s).")


if __name__ == "__main__":
    main()
