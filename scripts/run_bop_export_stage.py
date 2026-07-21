#!/usr/bin/env python3
"""Export synchronized sensor folders into a minimal BOP scene layout."""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path

from posetestbot.bop.writer import (
    copy_bop_instance_models,
    export_sensor_scene_to_bop,
    targets_filename,
    validate_bop_dataset,
    write_bop_coco_annotations,
    write_bop_dataset_info,
    write_bop_export_manifest,
    write_bop_frame_map,
    write_bop_instance_map,
    write_bop_multiview_targets,
    write_bop_pose_template,
    write_bop_targets,
)
from posetestbot.calibration.profiles import (
    load_profile_collection,
    select_valid_profile_for_sensor,
)
from posetestbot.io.atomic import replace_directory
from posetestbot.io.artifacts import (
    BOP_DIR,
    BOP_COCO_ANNOTATIONS,
    BOP_EXPORT_MANIFEST,
    BOP_FRAME_MAP_JSON,
    BOP_INSTANCE_MAP,
    BOP_MULTIVIEW_TARGETS,
    BOP_POSE_TEMPLATE,
    BOP_TARGETS_BOP19,
    CALIBRATION_PROFILES,
    DEPTH_DIR,
    MODELS_DIR,
    OBJECT_INSTANCES,
    PROCESSED_DIR,
    RGB_DIR,
    SYNCHRONIZED_DIR,
)
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    upsert_stage,
    utc_now_iso,
    write_run_manifest,
)
from posetestbot.pipeline.run_config import load_run_config_for_run_root
from posetestbot.pose_templates.selection import (
    load_pose_template_selection,
    prepare_object_instances,
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
        "--objectless", action="store_true",
        help="Export RGB-D and camera metadata with explicitly empty object data.",
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
            "Optional calibration.v2 profile collection. Matching profiles are "
            "recorded in scene_camera.json and bop_export_manifest.json."
        ),
    )
    return parser.parse_args()


def default_input_folder(run_root: Path, explicit_input_folder: str | None) -> Path:
    if explicit_input_folder:
        return Path(explicit_input_folder)
    rectified = run_root / PROCESSED_DIR / "rectified"
    if rectified.is_dir():
        return rectified
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
    calibration_profiles_path = (
        Path(args.calibration_profiles) if args.calibration_profiles else None
    )

    manifest = load_or_create_run_manifest(run_root)
    upsert_stage(manifest, name="bop_export", status="running")
    write_run_manifest(manifest, run_root)

    staging_folder = output_folder.with_name(
        f".{output_folder.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        if output_folder.exists() and not args.overwrite:
            raise FileExistsError(
                f"BOP dataset already exists: {output_folder}; pass --overwrite"
            )
        sensor_folders = discover_exportable_sensor_folders(input_folder)
        calibration_profiles = (
            load_profile_collection(calibration_profiles_path)
            if calibration_profiles_path is not None
            else []
        )
        if calibration_profiles_path is not None and not calibration_profiles:
            raise ValueError("Calibration profile collection must not be empty")

        staging_folder.parent.mkdir(parents=True, exist_ok=True)
        staging_folder.mkdir(parents=False, exist_ok=False)
        try:
            run_config = load_run_config_for_run_root(run_root)
        except FileNotFoundError:
            run_config = None
        dataset_mode = (
            str(run_config.get("dataset_mode"))
            if run_config is not None
            else "objectless"
        )
        if args.objectless:
            dataset_mode = "objectless"
        template_mode = dataset_mode == "pose_template" and not args.objectless
        objectless_mode = dataset_mode == "objectless"
        selection = load_pose_template_selection(run_root) if template_mode else None
        object_instances = prepare_object_instances(run_root) if template_mode else None
        object_name_to_id = None
        object_models = []
        if objectless_mode:
            object_name_to_id = {}
        if not args.no_model_export:
            object_name_to_id = (
                {
                    str(item["instance_uuid"]): int(item["obj_id"])
                    for item in object_instances["instances"]
                }
                if object_instances is not None
                else {}
            )
            geometry_cache = None
            previous_models_info = output_folder / MODELS_DIR / "models_info.json"
            if previous_models_info.is_file():
                try:
                    loaded_cache = json.loads(previous_models_info.read_text())
                except json.JSONDecodeError:
                    loaded_cache = None
                if isinstance(loaded_cache, dict):
                    geometry_cache = loaded_cache
            if object_instances is not None:
                object_models = copy_bop_instance_models(
                    staging_folder,
                    run_root,
                    object_instances,
                    geometry_cache=geometry_cache,
                )
        exports = []
        for offset, sensor_folder in enumerate(sensor_folders):
            calibration_profile = (
                select_valid_profile_for_sensor(
                    calibration_profiles, sensor_folder.name
                )
                if calibration_profiles_path is not None
                else None
            )
            exports.append(
                export_sensor_scene_to_bop(
                    sensor_folder,
                    staging_folder,
                    split=args.split,
                    scene_id=args.scene_start + offset,
                    overwrite=False,
                    calibration_profile=calibration_profile,
                    object_name_to_id=object_name_to_id,
                    template_instances=(
                        object_instances["instances"]
                        if object_instances is not None
                        else None
                    ),
                )
            )

        targets_path = None
        multiview_targets_path = None
        coco_annotations_path = None
        if (not args.no_model_export or objectless_mode) and args.split == "test":
            targets_path = write_bop_targets(staging_folder, exports, split=args.split)
        if args.write_multiview_targets:
            multiview_targets_path = write_bop_multiview_targets(
                staging_folder,
                exports,
                split=args.split,
            )
        if args.write_coco_annotations:
            coco_annotations_path = write_bop_coco_annotations(
                staging_folder,
                exports,
                split=args.split,
                object_models=object_models,
            )

        frame_map_path = write_bop_frame_map(staging_folder, exports)
        instance_map_path = (
            write_bop_instance_map(staging_folder, exports) if template_mode else None
        )
        pose_template_path = (
            write_bop_pose_template(staging_folder, selection)
            if selection is not None
            else None
        )
        dataset_info_path = write_bop_dataset_info(
            staging_folder,
            exports,
            dataset_name=run_root.name,
            generated_at=utc_now_iso(),
        )
        validation = validate_bop_dataset(
            staging_folder,
            exports,
            object_models=object_models,
            targets_path=targets_path,
        )
        write_bop_export_manifest(
            staging_folder,
            exports,
            calibration_profiles_path=calibration_profiles_path,
            calibration_profiles=calibration_profiles,
            object_models=object_models,
            targets_path=targets_path,
            multiview_targets_path=multiview_targets_path,
            coco_annotations_path=coco_annotations_path,
            frame_map_path=frame_map_path,
            dataset_info_path=dataset_info_path,
            validation=validation,
            stable_id_mapping=(
                {
                    str(item["catalog_uuid"]): int(item["obj_id"])
                    for item in object_instances["instances"]
                }
                if object_instances is not None
                else {}
            ),
            dataset_mode=dataset_mode,
            pose_template_provenance=(
                {
                    "template_uuid": selection["template_uuid"],
                    "bundle_sha256": selection["bundle_sha256"],
                    "configuration_sha256": selection["configuration_sha256"],
                    "instance_count": len(selection["instances"]),
                }
                if selection is not None
                else None
            ),
            instance_map_path=instance_map_path,
            pose_template_path=pose_template_path,
        )
        replace_directory(staging_folder, output_folder)

        artifacts: dict[str, Path] = {
            BOP_DIR: output_folder,
            BOP_EXPORT_MANIFEST: output_folder / BOP_EXPORT_MANIFEST,
            BOP_FRAME_MAP_JSON: output_folder / BOP_FRAME_MAP_JSON,
            "dataset_info.json": output_folder / "dataset_info.json",
        }
        if object_models:
            artifacts[MODELS_DIR] = output_folder / MODELS_DIR
        if object_instances is not None:
            artifacts[OBJECT_INSTANCES] = run_root / OBJECT_INSTANCES
            artifacts[BOP_INSTANCE_MAP] = output_folder / BOP_INSTANCE_MAP
            artifacts[BOP_POSE_TEMPLATE] = output_folder / BOP_POSE_TEMPLATE
        if calibration_profiles_path is not None:
            artifacts[CALIBRATION_PROFILES] = calibration_profiles_path
        for export in exports:
            artifacts[f"{export.sensor_name}:bop_scene"] = (
                output_folder / export.scene_folder
            )
        if targets_path is not None:
            artifacts[targets_filename(args.split)] = (
                output_folder / targets_filename(args.split)
            )
            artifacts[BOP_TARGETS_BOP19] = output_folder / BOP_TARGETS_BOP19
        if multiview_targets_path is not None:
            artifacts[BOP_MULTIVIEW_TARGETS] = output_folder / BOP_MULTIVIEW_TARGETS
        if coco_annotations_path is not None:
            artifacts[BOP_COCO_ANNOTATIONS] = output_folder / BOP_COCO_ANNOTATIONS
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
        if staging_folder.exists():
            shutil.rmtree(staging_folder)
        upsert_stage(manifest, name="bop_export", status="failed", message=str(exc))
        write_run_manifest(manifest, run_root)
        raise

    print(message)


if __name__ == "__main__":
    main()
