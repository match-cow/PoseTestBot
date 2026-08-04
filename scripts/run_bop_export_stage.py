#!/usr/bin/env python3
"""Export synchronized sensor folders into a minimal BOP scene layout."""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2

from posetestbot.bop.mask_driver import (
    GENERATION_REPORT,
    run_official_bop_mask_generation,
)
from posetestbot.bop.writer import (
    ANNOTATION_MODES,
    ANNOTATION_SOURCES,
    copy_bop_instance_models,
    export_sensor_scene_to_bop,
    finalize_official_scene_annotations,
    resolve_annotation_mode,
    targets_filename,
    validate_bop_dataset,
    write_bop_coco_annotations,
    write_bop_dataset_info,
    write_bop_export_manifest,
    write_bop_frame_map,
    write_bop_frame_sets,
    write_bop_instance_map,
    write_bop_multiview_targets,
    write_bop_pose_template,
    write_bop_targets,
)
from posetestbot.calibration.profiles import (
    CalibrationProfile,
    load_profile_collection,
    select_valid_profile_for_sensor,
)
from posetestbot.calibration.rectification import (
    RECTIFIED_DIR,
    rgbd_camera_artifact_fingerprint,
    validate_rectification_provenance,
)
from posetestbot.calibration.static_reuse import (
    verify_static_profile_destination_reference,
)
from posetestbot.io.atomic import replace_directory
from posetestbot.io.artifacts import (
    BOP_DIR,
    BOP_COCO_ANNOTATIONS,
    BOP_EXPORT_MANIFEST,
    BOP_FRAME_MAP_JSON,
    BOP_FRAME_SETS,
    BOP_INSTANCE_MAP,
    BOP_MULTIVIEW_TARGETS,
    BOP_POSE_TEMPLATE,
    BOP_TARGETS_BOP19,
    CALIBRATION_PROFILES,
    CALIBRATION_PROFILE_SELECTION,
    DEPTH_DIR,
    MATCH_ROBOT_EE_POSES,
    MODELS_DIR,
    MODELS_EVAL_DIR,
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
from posetestbot.pipeline.run_config import (
    capture_synchronization_from_mapping,
    load_run_config_for_run_root,
)
from posetestbot.pipeline.sensor_selection import (
    enabled_sensor_mounting_modes_by_folder,
    filter_enabled_sensor_folders,
)
from posetestbot.pose_templates.selection import (
    load_pose_template_selection,
    prepare_object_instances,
)
from posetestbot.sync.hardware import load_hardware_sync_frame_groups
from posetestbot.sensors.contracts import MountingMode
from posetestbot.sensors.registry import sensor_folder_name
from posetestbot.sensors.hardware_sync_qualification import (
    validate_hardware_sync_qualification,
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
        "--objectless",
        action="store_true",
        help="Export only RGB-D and camera metadata, without object artifacts.",
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
        "--annotation-source",
        choices=sorted(ANNOTATION_SOURCES),
        default="none",
        help=(
            "Source for BOP scene GT. The acquisition-first default 'none' "
            "omits GT-derived files rather than writing placeholders; "
            "pose-template object targets remain available for inference. Use "
            "'blenderproc' only after optional GT pose generation has completed."
        ),
    )
    parser.add_argument(
        "--annotation-mode",
        choices=sorted(ANNOTATION_MODES),
        default=None,
        help=(
            "GT capability to publish: 'none', BlenderProc-derived 'pose', or "
            "'pose_and_masks'. When omitted, the legacy source contract is "
            "preserved: annotation-source blenderproc means pose_and_masks."
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
    rectified = run_root / PROCESSED_DIR / RECTIFIED_DIR
    if rectified.is_dir():
        return rectified
    return run_root / PROCESSED_DIR / SYNCHRONIZED_DIR


def default_output_folder(run_root: Path, explicit_output_folder: str | None) -> Path:
    if explicit_output_folder:
        return Path(explicit_output_folder)
    return run_root / BOP_DIR


def _run_input_path(run_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else run_root / path


def _selected_calibration_configured(run_root: Path, run_config: dict | None) -> bool:
    return (run_config or {}).get("calibration_profile_selection") is not None or (
        run_root / CALIBRATION_PROFILE_SELECTION
    ).exists()


def calibration_profile_for_sensor(
    profiles: list[CalibrationProfile],
    sensor_name: str,
    *,
    profile_ids_by_sensor_name: Mapping[str, str] | None = None,
    mounting_modes_by_sensor_name: Mapping[str, MountingMode] | None = None,
) -> CalibrationProfile:
    """Resolve a BOP camera profile, honoring managed selection when present."""

    profile_id = None
    if profile_ids_by_sensor_name is not None:
        try:
            profile_id = profile_ids_by_sensor_name[sensor_name]
        except KeyError as exc:
            raise KeyError(
                f"Calibration selection has no profile for {sensor_name!r}"
            ) from exc
    mounting_mode = None
    if mounting_modes_by_sensor_name is not None:
        try:
            mounting_mode = mounting_modes_by_sensor_name[sensor_name]
        except KeyError as exc:
            raise KeyError(
                f"Run configuration has no mounting mode for {sensor_name!r}"
            ) from exc
    return select_valid_profile_for_sensor(
        profiles,
        sensor_name,
        mounting_mode=mounting_mode,
        profile_id=profile_id,
    )


def _load_required_hardware_frame_groups(
    run_root: Path,
    run_config: dict | None,
) -> dict | None:
    capture = (run_config or {}).get("capture")
    policy = capture_synchronization_from_mapping(
        capture.get("synchronization") if isinstance(capture, dict) else None
    ).to_dict()
    if policy["mode"] != "hardware_trigger":
        return None

    groups = load_hardware_sync_frame_groups(run_root)
    qualification = validate_hardware_sync_qualification(
        run_root,
        run_config=run_config,
    )
    expected = {
        "group_id": policy["group_id"],
        "implementation": policy["implementation"],
        "scope": policy["scope"],
        "master_sensor_key": policy["master_sensor_key"],
    }
    mismatches = [
        f"{key}={groups.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if groups.get(key) != value
    ]
    expected_skew_ns = int(
        round(float(policy["max_depth_timestamp_skew_ms"]) * 1_000_000)
    )
    if groups.get("max_depth_timestamp_skew_ns") != expected_skew_ns:
        mismatches.append(
            "max_depth_timestamp_skew_ns="
            f"{groups.get('max_depth_timestamp_skew_ns')!r}, expected "
            f"{expected_skew_ns!r}"
        )
    if groups.get("hardware_sync_qualification") != qualification:
        mismatches.append(
            "hardware_sync_qualification does not exactly match the current "
            "validated physical qualification"
        )
    expected_inventory = _hardware_sensor_inventory(run_config, policy)
    if groups.get("sensor_order") != [
        item["sensor_key"] for item in expected_inventory
    ]:
        mismatches.append(
            "sensor_order does not exactly match the enabled run-config "
            "hardware sensor order"
        )
    raw_inventory = groups.get("sensors")
    if not isinstance(raw_inventory, list) or len(raw_inventory) != len(
        expected_inventory
    ):
        mismatches.append(
            "sensor inventory does not contain exactly the enabled run-config "
            "hardware sensors"
        )
    else:
        inventory_fields = (
            "sensor_key",
            "sensor_type",
            "device_id",
            "sensor_folder",
            "mounting_mode",
            "hardware_sync_role",
        )
        for index, (actual, expected_sensor) in enumerate(
            zip(raw_inventory, expected_inventory, strict=True)
        ):
            if not isinstance(actual, Mapping):
                mismatches.append(f"sensors[{index}] is not an object")
                continue
            for field in inventory_fields:
                if actual.get(field) != expected_sensor[field]:
                    mismatches.append(
                        f"sensors[{index}].{field}={actual.get(field)!r}, "
                        f"expected {expected_sensor[field]!r}"
                    )
    if mismatches:
        raise ValueError(
            "Hardware-sync frame-group provenance does not match run_config.json: "
            + "; ".join(mismatches)
        )
    return groups


def _hardware_sensor_inventory(
    run_config: Mapping[str, Any] | None,
    policy: Mapping[str, Any],
) -> list[dict[str, str]]:
    capture = (run_config or {}).get("capture")
    if not isinstance(capture, Mapping):
        raise ValueError("Hardware-sync BOP export requires run_config.capture")
    raw_sensors = capture.get("sensors")
    if not isinstance(raw_sensors, list):
        raise ValueError("Hardware-sync BOP export requires run_config.capture.sensors")
    master_sensor_key = str(policy["master_sensor_key"])
    inventory: list[dict[str, str]] = []
    for index, raw_sensor in enumerate(raw_sensors):
        if not isinstance(raw_sensor, Mapping):
            raise ValueError(f"run_config.capture.sensors[{index}] is not an object")
        if raw_sensor.get("enabled", True) is not True:
            continue
        sensor_type = str(raw_sensor.get("sensor_type") or "")
        device_id = str(raw_sensor.get("device_id") or "")
        mounting_mode = str(raw_sensor.get("mounting_mode") or "")
        sensor_key = f"{sensor_type}:{device_id}"
        inventory.append(
            {
                "sensor_key": sensor_key,
                "sensor_type": sensor_type,
                "device_id": device_id,
                "sensor_folder": (
                    f"{PROCESSED_DIR}/{SYNCHRONIZED_DIR}/"
                    f"{sensor_folder_name(sensor_type, device_id)}"
                ),
                "mounting_mode": mounting_mode,
                "hardware_sync_role": (
                    "master" if sensor_key == master_sensor_key else "subordinate"
                ),
            }
        )
    masters = [item for item in inventory if item["sensor_key"] == master_sensor_key]
    if len(masters) != 1:
        raise ValueError(
            "Hardware-sync BOP export master_sensor_key must identify exactly "
            "one enabled run-config sensor"
        )
    return masters + [
        item for item in inventory if item["sensor_key"] != master_sensor_key
    ]


def _run_relative(path: Path, run_root: Path) -> str:
    try:
        return path.resolve().relative_to(run_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Hardware-sync BOP input escapes the run root: {path}"
        ) from exc


def _portable_run_path(path: Path, run_root: Path) -> str | None:
    """Return a portable run-relative path, never an absolute host path."""

    try:
        return path.resolve().relative_to(run_root.resolve()).as_posix()
    except ValueError:
        return None


def _validated_hardware_input_evidence(
    *,
    run_root: Path,
    input_folder: Path,
    run_config: Mapping[str, Any],
    hardware_frame_groups: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Validate the one canonical native/rectified source tree for BOP."""

    synchronized_root = run_root / PROCESSED_DIR / SYNCHRONIZED_DIR
    rectified_root = run_root / PROCESSED_DIR / RECTIFIED_DIR
    if input_folder.is_symlink():
        raise ValueError("Hardware-sync BOP input root must not be a symbolic link")
    resolved_input = input_folder.resolve()
    if resolved_input == synchronized_root.resolve():
        projection = "native"
    elif resolved_input == rectified_root.resolve():
        projection = "rectified"
    else:
        raise ValueError(
            "Hardware-sync BOP input must be the canonical "
            f"{synchronized_root} or {rectified_root}; arbitrary input roots "
            "cannot preserve authoritative frame-group provenance"
        )
    if not input_folder.is_dir():
        raise FileNotFoundError(
            f"Hardware-sync BOP input folder does not exist: {input_folder}"
        )

    policy = capture_synchronization_from_mapping(
        run_config["capture"]["synchronization"]
    ).to_dict()
    inventory = _hardware_sensor_inventory(run_config, policy)
    group_sensors = hardware_frame_groups.get("sensors")
    if not isinstance(group_sensors, list):
        raise ValueError("Hardware-sync frame groups have no sensor inventory")
    group_by_key = {
        str(item.get("sensor_key")): item
        for item in group_sensors
        if isinstance(item, Mapping)
    }
    evidence: dict[str, dict[str, str]] = {}
    for expected in inventory:
        sensor_name = Path(expected["sensor_folder"]).name
        source_sensor = synchronized_root / sensor_name
        input_sensor = input_folder / sensor_name
        if (
            source_sensor.is_symlink()
            or input_sensor.is_symlink()
            or not source_sensor.is_dir()
            or not input_sensor.is_dir()
        ):
            raise ValueError(
                "Hardware-sync BOP sensor folders must be existing regular "
                f"directories: source={source_sensor}, input={input_sensor}"
            )
        group_sensor = group_by_key.get(expected["sensor_key"])
        if (
            not isinstance(group_sensor, Mapping)
            or group_sensor.get("sensor_folder") != expected["sensor_folder"]
        ):
            raise ValueError(
                "Hardware-sync frame-group sensor folder does not match the "
                f"current run config for {expected['sensor_key']}"
            )
        if projection == "rectified":
            provenance = validate_rectification_provenance(
                source_sensor,
                input_sensor,
            )
            input_fingerprint = provenance["output_fingerprint"]
            source_fingerprint = provenance["source_fingerprint"]
        else:
            source_fingerprint = rgbd_camera_artifact_fingerprint(source_sensor)
            input_fingerprint = source_fingerprint
        evidence[sensor_name] = {
            "projection": projection,
            "input_sensor_folder": _run_relative(input_sensor, run_root),
            "authoritative_source_sensor_folder": _run_relative(
                source_sensor,
                run_root,
            ),
            "input_fingerprint_sha256": str(input_fingerprint["digest"]),
            "authoritative_source_fingerprint_sha256": str(
                source_fingerprint["digest"]
            ),
        }
    return evidence


def _revalidate_hardware_publication_inputs(
    *,
    run_root: Path,
    input_folder: Path,
    initial_run_config: Mapping[str, Any],
    initial_hardware_frame_groups: Mapping[str, Any],
    initial_hardware_input_evidence: Mapping[str, Mapping[str, str]],
) -> None:
    """Close the hardware provenance TOCTOU immediately before publication."""

    current_run_config = load_run_config_for_run_root(run_root)
    if current_run_config != initial_run_config:
        raise RuntimeError(
            "Run configuration changed while the hardware-sync BOP export was running"
        )
    current_hardware_frame_groups = _load_required_hardware_frame_groups(
        run_root,
        current_run_config,
    )
    if current_hardware_frame_groups != initial_hardware_frame_groups:
        raise RuntimeError(
            "Authoritative hardware-sync frame groups changed while the BOP "
            "export was running"
        )
    current_hardware_input_evidence = _validated_hardware_input_evidence(
        run_root=run_root,
        input_folder=input_folder,
        run_config=current_run_config,
        hardware_frame_groups=current_hardware_frame_groups,
    )
    if current_hardware_input_evidence != initial_hardware_input_evidence:
        raise RuntimeError(
            "Hardware-sync BOP input changed while the export was running"
        )


def discover_exportable_sensor_folders(
    input_folder: Path,
    *,
    run_root: Path | None = None,
) -> list[Path]:
    if not input_folder.is_dir():
        raise FileNotFoundError(f"Synchronized input folder not found: {input_folder}")
    sensors = [
        child
        for child in sorted(input_folder.iterdir())
        if child.is_dir()
        and (child / RGB_DIR).is_dir()
        and (child / DEPTH_DIR).is_dir()
    ]
    if run_root is not None:
        sensors = filter_enabled_sensor_folders(run_root, sensors)
    if not sensors:
        raise FileNotFoundError(
            f"No synchronized RGB-D sensor folders in {input_folder}"
        )
    return sensors


def _uniform_export_image_size(
    output_root: Path,
    exports: list[Any],
) -> tuple[int, int]:
    sizes: set[tuple[int, int]] = set()
    for export in exports:
        rgb_folder = output_root / export.scene_folder / RGB_DIR
        first_rgb = next(iter(sorted(rgb_folder.glob("*.png"))), None)
        image = (
            cv2.imread(first_rgb.as_posix(), cv2.IMREAD_UNCHANGED)
            if first_rgb is not None
            else None
        )
        if image is None:
            raise ValueError(
                f"Cannot determine BOP scene image size: {export.scene_folder}"
            )
        height, width = image.shape[:2]
        sizes.add((int(width), int(height)))
    if len(sizes) != 1:
        raise ValueError(
            "Official BOP mask generation requires one resolution across all scenes"
        )
    return next(iter(sizes))


def complete_official_mask_annotations(
    output_root: Path,
    exports: list[Any],
    object_models: list[Any],
    *,
    split: str,
    mask_runner: Callable[..., Mapping[str, object]] = (
        run_official_bop_mask_generation
    ),
) -> tuple[list[Any], dict[str, object]]:
    """Complete staged pose GT through the injectable official-toolkit boundary."""

    report = dict(
        mask_runner(
            output_root,
            split=split,
            scene_ids=[export.scene_id for export in exports],
            object_ids=[model.obj_id for model in object_models],
            image_size=_uniform_export_image_size(output_root, exports),
            app_root=Path(__file__).resolve().parents[1],
        )
    )
    return finalize_official_scene_annotations(output_root, exports), report


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    input_folder = default_input_folder(run_root, args.input_folder)
    output_folder = default_output_folder(run_root, args.output_folder)
    calibration_profiles_path = (
        _run_input_path(run_root, args.calibration_profiles)
        if args.calibration_profiles
        else None
    )

    manifest = load_or_create_run_manifest(run_root)
    upsert_stage(manifest, name="bop_export", status="running")
    write_run_manifest(manifest, run_root)

    staging_folder = output_folder.with_name(
        f".{output_folder.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        annotation_mode = resolve_annotation_mode(
            args.annotation_source,
            args.annotation_mode,
        )
        if annotation_mode == "none" and (
            args.write_multiview_targets or args.write_coco_annotations
        ):
            raise ValueError(
                "Multiview targets and COCO annotations require "
                "--annotation-source blenderproc"
            )
        if args.write_coco_annotations and annotation_mode != "pose_and_masks":
            raise ValueError(
                "COCO annotations require --annotation-mode pose_and_masks"
            )
        try:
            run_config = load_run_config_for_run_root(run_root)
        except FileNotFoundError:
            run_config = None
        mounting_modes_by_sensor_name = enabled_sensor_mounting_modes_by_folder(
            run_config
        )
        hardware_frame_groups = _load_required_hardware_frame_groups(
            run_root,
            run_config,
        )
        hardware_input_evidence = (
            _validated_hardware_input_evidence(
                run_root=run_root,
                input_folder=input_folder,
                run_config=run_config,
                hardware_frame_groups=hardware_frame_groups,
            )
            if hardware_frame_groups is not None and run_config is not None
            else {}
        )
        calibration_profile_ids_by_sensor_name = None
        if _selected_calibration_configured(run_root, run_config):
            if calibration_profiles_path is None:
                raise ValueError(
                    "A run with selected calibration provenance must pass its "
                    "calibration_profiles snapshot to BOP export"
                )
            from posetestbot.calibration.profile_library import (
                selected_calibration_profile_ids_by_sensor_folder,
                verify_calibration_profile_selection,
            )

            calibration_selection = verify_calibration_profile_selection(
                run_root,
                expected_calibration_profiles=calibration_profiles_path,
            )
            calibration_profile_ids_by_sensor_name = (
                selected_calibration_profile_ids_by_sensor_folder(
                    run_root,
                    selection=calibration_selection,
                )
            )
            from posetestbot.sync.calibration_policy import (
                resolve_calibration_profile_sync_policy,
            )
            from posetestbot.sync.quality import (
                verify_profile_bound_sync_evidence,
            )

            calibration_sync_policy = resolve_calibration_profile_sync_policy(run_root)
            if calibration_sync_policy is None:
                raise ValueError(
                    "Selected calibration is not bound to a synchronization policy"
                )
            verify_profile_bound_sync_evidence(
                run_root,
                calibration_sync_policy,
            )
        if output_folder.exists() and not args.overwrite:
            raise FileExistsError(
                f"BOP dataset already exists: {output_folder}; pass --overwrite"
            )
        sensor_folders = discover_exportable_sensor_folders(
            input_folder,
            run_root=(
                run_root
                if args.input_folder is None or hardware_frame_groups is not None
                else None
            ),
        )
        calibration_profiles = (
            load_profile_collection(calibration_profiles_path)
            if calibration_profiles_path is not None
            else []
        )
        if calibration_profiles_path is not None and not calibration_profiles:
            raise ValueError("Calibration profile collection must not be empty")
        calibration_profiles_by_sensor_name = (
            {
                sensor_folder.name: calibration_profile_for_sensor(
                    calibration_profiles,
                    sensor_folder.name,
                    profile_ids_by_sensor_name=(
                        calibration_profile_ids_by_sensor_name
                    ),
                    mounting_modes_by_sensor_name=mounting_modes_by_sensor_name,
                )
                for sensor_folder in sensor_folders
            }
            if calibration_profiles_path is not None
            else {}
        )
        verify_static_profile_destination_reference(
            run_root,
            run_config,
            calibration_profiles_by_sensor_name.values(),
            matched_robot_pose_paths_by_sensor_name={
                sensor_folder.name: sensor_folder / MATCH_ROBOT_EE_POSES
                for sensor_folder in sensor_folders
            },
        )

        staging_folder.parent.mkdir(parents=True, exist_ok=True)
        staging_folder.mkdir(parents=False, exist_ok=False)
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
            input_evidence = hardware_input_evidence.get(sensor_folder.name)
            if hardware_frame_groups is not None and input_evidence is None:
                raise ValueError(
                    "Hardware-sync BOP input does not exactly cover the "
                    f"authoritative sensor set: {sensor_folder.name}"
                )
            calibration_profile = calibration_profiles_by_sensor_name.get(
                sensor_folder.name
            )
            portable_sensor_folder = (
                _portable_run_path(sensor_folder, run_root) or sensor_folder.name
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
                    source_projection=(
                        input_evidence["projection"]
                        if input_evidence is not None
                        else None
                    ),
                    input_sensor_folder=(
                        input_evidence["input_sensor_folder"]
                        if input_evidence is not None
                        else portable_sensor_folder
                    ),
                    authoritative_source_sensor_folder=(
                        input_evidence["authoritative_source_sensor_folder"]
                        if input_evidence is not None
                        else portable_sensor_folder
                    ),
                    input_fingerprint_sha256=(
                        input_evidence["input_fingerprint_sha256"]
                        if input_evidence is not None
                        else None
                    ),
                    authoritative_source_fingerprint_sha256=(
                        input_evidence["authoritative_source_fingerprint_sha256"]
                        if input_evidence is not None
                        else None
                    ),
                    annotation_source=args.annotation_source,
                    annotation_mode=annotation_mode,
                )
            )
        if hardware_frame_groups is not None and run_config is not None:
            current_input_evidence = _validated_hardware_input_evidence(
                run_root=run_root,
                input_folder=input_folder,
                run_config=run_config,
                hardware_frame_groups=hardware_frame_groups,
            )
            if current_input_evidence != hardware_input_evidence:
                raise RuntimeError(
                    "Hardware-sync BOP input changed while the export was running"
                )

        pose_generation_provenance = {
            "source": "blenderproc_analytic_gt",
            "scenes": [
                {
                    "scene_id": export.scene_id,
                    "sensor_name": export.sensor_name,
                    **export.annotation_provenance,
                }
                for export in exports
                if export.annotation_source == "blenderproc"
            ],
        }
        annotation_provenance: dict[str, object] = {}
        if annotation_mode == "pose":
            annotation_provenance = {
                "schema_version": "posetestbot_bop_gt_generation.v1",
                "annotation_mode": "pose",
                "pose_generation": pose_generation_provenance,
                "mask_generation": {"state": "absent"},
            }
        elif annotation_mode == "pose_and_masks":
            if not object_models:
                raise ValueError(
                    "Official BOP mask generation requires exported object models"
                )
            exports, mask_generation_provenance = complete_official_mask_annotations(
                staging_folder,
                exports,
                object_models,
                split=args.split,
            )
            annotation_provenance = {
                "schema_version": "posetestbot_bop_gt_generation.v1",
                "annotation_mode": "pose_and_masks",
                "pose_generation": pose_generation_provenance,
                "mask_generation": mask_generation_provenance,
            }

        targets_path = None
        multiview_targets_path = None
        coco_annotations_path = None
        if (
            args.split == "test"
            and not args.no_model_export
            and any(export.targets for export in exports)
        ):
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
        frame_sets_path = (
            write_bop_frame_sets(
                staging_folder,
                exports,
                hardware_frame_groups,
            )
            if hardware_frame_groups is not None
            else None
        )
        instance_map_path = (
            write_bop_instance_map(staging_folder, exports)
            if template_mode and args.annotation_source == "blenderproc"
            else None
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
        if frame_sets_path is not None:
            frame_sets = json.loads(frame_sets_path.read_text())
            validation["frame_set_count"] = int(frame_sets["frame_set_count"])
            validation["hardware_sync_scope"] = frame_sets["scope"]
        exported_profile_ids = {
            export.calibration_profile_id
            for export in exports
            if export.calibration_profile_id is not None
        }
        exported_calibration_profiles = [
            profile
            for profile in calibration_profiles
            if profile.profile_id in exported_profile_ids
        ]
        write_bop_export_manifest(
            staging_folder,
            exports,
            calibration_profiles_path=(
                _portable_run_path(calibration_profiles_path, run_root)
                if calibration_profiles_path is not None
                else None
            ),
            calibration_profiles=exported_calibration_profiles,
            object_models=object_models,
            targets_path=targets_path,
            multiview_targets_path=multiview_targets_path,
            coco_annotations_path=coco_annotations_path,
            frame_map_path=frame_map_path,
            frame_sets_path=frame_sets_path,
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
            annotation_source=args.annotation_source,
            annotation_mode=annotation_mode,
            annotation_provenance=annotation_provenance,
        )
        if hardware_frame_groups is not None and run_config is not None:
            _revalidate_hardware_publication_inputs(
                run_root=run_root,
                input_folder=input_folder,
                initial_run_config=run_config,
                initial_hardware_frame_groups=hardware_frame_groups,
                initial_hardware_input_evidence=hardware_input_evidence,
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
            artifacts[MODELS_EVAL_DIR] = output_folder / MODELS_EVAL_DIR
        if object_instances is not None:
            artifacts[OBJECT_INSTANCES] = run_root / OBJECT_INSTANCES
            artifacts[BOP_POSE_TEMPLATE] = output_folder / BOP_POSE_TEMPLATE
        if instance_map_path is not None:
            artifacts[BOP_INSTANCE_MAP] = output_folder / BOP_INSTANCE_MAP
        if annotation_mode == "pose_and_masks":
            artifacts[GENERATION_REPORT] = output_folder / GENERATION_REPORT
        if calibration_profiles_path is not None:
            artifacts[CALIBRATION_PROFILES] = calibration_profiles_path
        for export in exports:
            artifacts[f"{export.sensor_name}:bop_scene"] = (
                output_folder / export.scene_folder
            )
        if targets_path is not None:
            artifacts[targets_filename(args.split)] = output_folder / targets_filename(
                args.split
            )
            artifacts[BOP_TARGETS_BOP19] = output_folder / BOP_TARGETS_BOP19
        if multiview_targets_path is not None:
            artifacts[BOP_MULTIVIEW_TARGETS] = output_folder / BOP_MULTIVIEW_TARGETS
        if coco_annotations_path is not None:
            artifacts[BOP_COCO_ANNOTATIONS] = output_folder / BOP_COCO_ANNOTATIONS
        if frame_sets_path is not None:
            artifacts[BOP_FRAME_SETS] = output_folder / BOP_FRAME_SETS
        message = f"Exported {len(exports)} synchronized sensor folder(s) to BOP."
        if validation["capabilities"]["bop19_evaluation"]:
            message += (
                " GT poses, official full/visible masks, visibility info, and "
                "BOP19 evaluation targets are complete."
            )
        elif validation["capabilities"]["gt_poses"]:
            message += (
                " GT poses are complete; masks and BOP19 visibility evidence "
                "were intentionally omitted."
            )
        elif validation["capabilities"]["pose_estimation_input"]:
            message += (
                " RGB-D scenes, models, and populated targets are "
                "pose-estimation inputs; rendered GT and masks are not present."
            )
        if frame_sets_path is not None:
            message += (
                f" Published {validation['frame_set_count']} authoritative "
                "hardware-synchronized frame set(s)."
            )
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
