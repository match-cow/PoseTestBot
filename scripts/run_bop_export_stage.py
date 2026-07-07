#!/usr/bin/env python3
"""Export synchronized sensor folders into a minimal BOP scene layout."""

from __future__ import annotations

import argparse
from pathlib import Path

from posetestbot.bop.writer import (
    copy_bop_models,
    export_sensor_scene_to_bop,
    object_registry_from_folder,
    targets_filename,
    write_bop_coco_annotations,
    write_bop_export_manifest,
    write_bop_multiview_targets,
    write_bop_targets,
)
from posetestbot.calibration.profiles import load_profile_collection, select_profile_for_sensor
from posetestbot.io.artifacts import (
    BOP_DIR,
    BOP_COCO_ANNOTATIONS,
    BOP_EXPORT_MANIFEST,
    BOP_TARGETS_BOP19,
    CALIBRATION_PROFILES,
    DEPTH_DIR,
    MODELS_DIR,
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
            "Copy synchronized RGB/depth frames into BOP scene folders and "
            "record the export in dataset_manifest.json."
        )
    )
    parser.add_argument("run_root", help="Run root containing processed/synchronized.")
    parser.add_argument(
        "--input-folder",
        default=None,
        help="Synchronized sensor folder root. Defaults to <run_root>/processed/synchronized.",
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        help="BOP export root. Defaults to <run_root>/bop.",
    )
    parser.add_argument("--split", default="test", help="BOP split folder name.")
    parser.add_argument(
        "--object-folder",
        default="object_models",
        help=(
            "Object registry folder containing objects.json and .ply models. "
            "Defaults to object_models; pass --no-model-export to skip."
        ),
    )
    parser.add_argument(
        "--no-model-export",
        action="store_true",
        help="Skip copying object models and generating target files.",
    )
    parser.add_argument(
        "--write-multiview-targets",
        action="store_true",
        help=(
            "Also write posetestbot_multiview_targets.json, a per-object "
            "summary of scenes/sensors/images containing each target."
        ),
    )
    parser.add_argument(
        "--write-coco-annotations",
        action="store_true",
        help=(
            "Also write posetestbot_coco_annotations.json, a COCO-style "
            "annotation file derived from exported BOP scene GT, GT info, RGB "
            "files, and masks."
        ),
    )
    parser.add_argument(
        "--scene-start",
        type=int,
        default=1,
        help="Scene ID assigned to the first exported sensor folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing exported scene folders.",
    )
    parser.add_argument(
        "--calibration-profiles",
        default=None,
        help=(
            "Optional calibration.v1 profile collection. Matching profiles are "
            "recorded in scene_camera.json and bop_export_manifest.json."
        ),
    )
    return parser.parse_args()


def default_input_folder(run_root: Path, explicit_input_folder: str | None) -> Path:
    if explicit_input_folder:
        return Path(explicit_input_folder)
    return run_root / PROCESSED_DIR / SYNCHRONIZED_DIR


def default_output_folder(run_root: Path, explicit_output_folder: str | None) -> Path:
    if explicit_output_folder:
        return Path(explicit_output_folder)
    return run_root / BOP_DIR


def discover_exportable_sensor_folders(input_folder: Path) -> list[Path]:
    if not input_folder.is_dir():
        raise FileNotFoundError(f"Synchronized input folder not found: {input_folder}")
    sensors = [
        child
        for child in sorted(input_folder.iterdir())
        if child.is_dir() and (child / RGB_DIR).is_dir() and (child / DEPTH_DIR).is_dir()
    ]
    if not sensors:
        raise FileNotFoundError(f"No synchronized RGB-D sensor folders in {input_folder}")
    return sensors


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    input_folder = default_input_folder(run_root, args.input_folder)
    output_folder = default_output_folder(run_root, args.output_folder)
    object_folder = Path(args.object_folder)
    calibration_profiles_path = (
        Path(args.calibration_profiles) if args.calibration_profiles else None
    )

    manifest = load_or_create_run_manifest(run_root)
    upsert_stage(manifest, name="bop_export", status="running")
    write_run_manifest(manifest, run_root)

    try:
        sensor_folders = discover_exportable_sensor_folders(input_folder)
        calibration_profiles = (
            load_profile_collection(calibration_profiles_path)
            if calibration_profiles_path is not None
            else []
        )
        object_name_to_id = None
        object_models = []
        if not args.no_model_export:
            object_name_to_id = object_registry_from_folder(object_folder)
            object_models = copy_bop_models(output_folder, object_folder)
        exports = []
        artifacts: dict[str, Path] = {BOP_DIR: output_folder}
        if object_models:
            artifacts[MODELS_DIR] = output_folder / MODELS_DIR
        if calibration_profiles_path is not None:
            artifacts[CALIBRATION_PROFILES] = calibration_profiles_path
        for offset, sensor_folder in enumerate(sensor_folders):
            calibration_profile = (
                select_profile_for_sensor(calibration_profiles, sensor_folder.name)
                if calibration_profiles
                else None
            )
            export = export_sensor_scene_to_bop(
                sensor_folder,
                output_folder,
                split=args.split,
                scene_id=args.scene_start + offset,
                overwrite=args.overwrite,
                calibration_profile=calibration_profile,
                object_name_to_id=object_name_to_id,
            )
            exports.append(export)
            artifacts[f"{sensor_folder.name}:bop_scene"] = Path(export.scene_folder)

        targets_path = None
        multiview_targets_path = None
        coco_annotations_path = None
        if not args.no_model_export:
            targets_path = write_bop_targets(output_folder, exports, split=args.split)
            artifacts[targets_filename(args.split)] = targets_path
            if args.split == "test":
                artifacts[BOP_TARGETS_BOP19] = targets_path
            if args.write_multiview_targets:
                multiview_targets_path = write_bop_multiview_targets(
                    output_folder,
                    exports,
                    split=args.split,
                )
                artifacts[multiview_targets_path.name] = multiview_targets_path
            if args.write_coco_annotations:
                coco_annotations_path = write_bop_coco_annotations(
                    output_folder,
                    exports,
                    split=args.split,
                    object_models=object_models,
                )
                artifacts[BOP_COCO_ANNOTATIONS] = coco_annotations_path

        export_manifest = write_bop_export_manifest(
            output_folder,
            exports,
            calibration_profiles_path=calibration_profiles_path,
            calibration_profiles=calibration_profiles,
            object_models=object_models,
            targets_path=targets_path,
            multiview_targets_path=multiview_targets_path,
            coco_annotations_path=coco_annotations_path,
        )
        artifacts[BOP_EXPORT_MANIFEST] = export_manifest
        message = f"Exported {len(exports)} synchronized sensor folder(s) to BOP."
        upsert_stage(
            manifest,
            name="bop_export",
            status="succeeded",
            artifacts=artifacts,
            run_root=run_root,
            message=message,
        )
        write_run_manifest(manifest, run_root)
    except Exception as exc:
        upsert_stage(manifest, name="bop_export", status="failed", message=str(exc))
        write_run_manifest(manifest, run_root)
        raise

    print(message)


if __name__ == "__main__":
    main()
