#!/usr/bin/env python3
"""Run BlenderProc preparation as a manifest-tracked pipeline stage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from posetestbot.calibration.profiles import (
    blenderproc_camera_transform_map_from_profiles,
    load_profile_collection,
    write_legacy_camera_ee_transform_map,
)
from posetestbot.io.artifacts import (
    CALIBRATION_DIR,
    CALIBRATION_PROFILES,
    DERIVED_CAMERA_EE_TRANSFORM,
    PROCESSED_DIR,
    SYNCHRONIZED_DIR,
)
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    upsert_stage,
    write_run_manifest,
)

import blenderproc_prepare_multi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare BlenderProc inputs for derived synchronized sensor folders "
            "and record the stage in dataset_manifest.json."
        )
    )
    parser.add_argument(
        "run_root",
        help="Run root containing processed/synchronized sensor folders.",
    )
    parser.add_argument(
        "--input-folder",
        default=None,
        help="Folder containing synchronized sensor folders. Defaults to <run_root>/processed/synchronized.",
    )
    parser.add_argument(
        "--object-folder",
        default="object_models",
        help="Folder containing objects.json and object model files.",
    )
    parser.add_argument(
        "--camera-transformations",
        default="scripts/default_data/camera_ee_transform.json",
        help="Camera transform JSON used by BlenderProc preparation.",
    )
    parser.add_argument(
        "--calibration-profiles",
        default=None,
        help=(
            "calibration.v1 profile collection. When supplied, matching "
            "eye-in-hand or static profiles are resolved for synchronized sensor "
            "folders and converted to the camera transform map consumed by the "
            "current prep script."
        ),
    )
    parser.add_argument(
        "--subdir",
        default="blenderproc",
        help="Subdirectory created inside each synchronized sensor folder.",
    )
    return parser.parse_args()


def synchronized_input_folder(run_root: Path, explicit_input_folder: str | None) -> Path:
    if explicit_input_folder:
        return Path(explicit_input_folder)
    return run_root / PROCESSED_DIR / SYNCHRONIZED_DIR


def blenderproc_artifacts(input_folder: Path, subdir: str) -> dict[str, Path]:
    artifacts = {}
    for sensor_folder in sorted(input_folder.iterdir()):
        if not sensor_folder.is_dir():
            continue
        blenderproc_folder = sensor_folder / subdir
        if blenderproc_folder.exists():
            artifacts[f"{sensor_folder.name}:{subdir}"] = blenderproc_folder
    return artifacts


def synchronized_sensor_names(input_folder: Path) -> list[str]:
    if not input_folder.is_dir():
        raise FileNotFoundError(f"Synchronized input folder not found: {input_folder}")
    return [child.name for child in sorted(input_folder.iterdir()) if child.is_dir()]


def derived_camera_transform_path(run_root: Path) -> Path:
    return run_root / PROCESSED_DIR / CALIBRATION_DIR / DERIVED_CAMERA_EE_TRANSFORM


def camera_transformations_from_calibration_profiles(
    *,
    run_root: Path,
    input_folder: Path,
    calibration_profiles_path: Path,
) -> Path:
    profiles = load_profile_collection(calibration_profiles_path)
    transform_map = blenderproc_camera_transform_map_from_profiles(
        profiles, synchronized_sensor_names(input_folder)
    )
    return write_legacy_camera_ee_transform_map(
        transform_map, derived_camera_transform_path(run_root)
    )


def run_prepare(
    *,
    input_folder: Path,
    object_folder: Path,
    camera_transformations: Path,
    subdir: str,
) -> None:
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "blenderproc_prepare_multi.py",
            str(input_folder),
            str(object_folder),
            str(camera_transformations),
            subdir,
        ]
        blenderproc_prepare_multi.main()
    finally:
        sys.argv = original_argv


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    input_folder = synchronized_input_folder(run_root, args.input_folder)
    object_folder = Path(args.object_folder)
    camera_transformations = Path(args.camera_transformations)
    calibration_profiles = Path(args.calibration_profiles) if args.calibration_profiles else None

    manifest = load_or_create_run_manifest(run_root)
    upsert_stage(manifest, name="blenderproc_prepare", status="running")
    write_run_manifest(manifest, run_root)

    try:
        stage_artifacts: dict[str, Path] = {}
        if calibration_profiles is not None:
            camera_transformations = camera_transformations_from_calibration_profiles(
                run_root=run_root,
                input_folder=input_folder,
                calibration_profiles_path=calibration_profiles,
            )
            stage_artifacts[CALIBRATION_PROFILES] = calibration_profiles
            stage_artifacts[DERIVED_CAMERA_EE_TRANSFORM] = camera_transformations

        run_prepare(
            input_folder=input_folder,
            object_folder=object_folder,
            camera_transformations=camera_transformations,
            subdir=args.subdir,
        )
    except Exception as exc:
        upsert_stage(
            manifest,
            name="blenderproc_prepare",
            status="failed",
            message=str(exc),
        )
        write_run_manifest(manifest, run_root)
        raise

    artifacts = {**stage_artifacts, **blenderproc_artifacts(input_folder, args.subdir)}
    upsert_stage(
        manifest,
        name="blenderproc_prepare",
        status="succeeded",
        artifacts=artifacts,
        run_root=run_root,
        message=f"Prepared BlenderProc inputs for {len(artifacts)} sensor folder(s).",
    )
    write_run_manifest(manifest, run_root)
    print(f"Prepared BlenderProc inputs for {len(artifacts)} sensor folder(s).")


if __name__ == "__main__":
    main()
