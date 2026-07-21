"""Compose browser-ready cell scenes with pytransform3d as frame authority."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

import numpy as np
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

from posetestbot.calibration.profiles import (
    CalibrationProfile,
    CalibrationStatus,
    load_profile_collection,
    select_valid_profile_for_sensor,
)
from posetestbot.io.artifacts import (
    BOP_DIR,
    BOP_EXPORT_MANIFEST,
    CALIBRATION_TARGET,
    MATCH_ROBOT_EE_POSES,
    PROCESSED_DIR,
    RAW_ROBOT_EE_POSES,
    SYNCHRONIZED_DIR,
)
from posetestbot.objects.registry import load_object_registry
from posetestbot.pipeline.run_config import load_run_config_for_run_root
from posetestbot.pose_templates.selection import load_pose_template_selection

SCENE_SCHEMA_VERSION = "cell_scene.v1"
TIMELINE_SCHEMA_VERSION = "cell_timeline.v1"
MAX_TIMELINE_PAGE = 2_000
MAX_PREVIEW_POSES = 200


def _matrix(quaternion: Any, translation: Any) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float)
    t = np.asarray(translation, dtype=float)
    if q.shape != (4,) or t.shape != (3,) or not np.all(np.isfinite([*q, *t])):
        raise ValueError("Transform requires finite quaternion[4] and translation[3]")
    norm = float(np.linalg.norm(q))
    if not math.isclose(norm, 1.0, abs_tol=1e-3):
        raise ValueError("Transform quaternion must be normalized")
    return pt.transform_from(pr.matrix_from_quaternion(q), t)


def _transform_dict(matrix: np.ndarray, parent: str) -> dict[str, Any]:
    return {
        "semantics": "entity_to_parent",
        "parent_frame": parent,
        "translation_mm": matrix[:3, 3].tolist(),
        "rotation_quaternion_wxyz": pr.quaternion_from_matrix(matrix[:3, :3]).tolist(),
    }


def _identity(parent: str | None = None) -> dict[str, Any]:
    return {
        "semantics": "entity_to_parent",
        "parent_frame": parent,
        "translation_mm": [0.0, 0.0, 0.0],
        "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    }


def _entity(
    entity_id: str,
    entity_type: str,
    label: str,
    *,
    transform: dict[str, Any] | None,
    status: str,
    provenance: Mapping[str, Any],
    reason: str | None = None,
    geometry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": entity_id,
        "type": entity_type,
        "label": label,
        "status": status,
        "transform": transform,
        "unresolved_reason": reason,
        "geometry": dict(geometry or {}),
        "provenance": dict(provenance),
    }


def _kuka_pose(value: Mapping[str, Any]) -> np.ndarray:
    try:
        translation = [float(value[key]) for key in ("X", "Y", "Z")]
        euler = [float(value[key]) for key in ("C", "B", "A")]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("KUKA pose requires finite X/Y/Z/A/B/C values") from exc
    if not np.all(np.isfinite([*translation, *euler])):
        raise ValueError("KUKA pose contains non-finite values")
    return pt.transform_from(pr.matrix_from_euler(euler, 0, 1, 2, True), translation)


def _read_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _matched_timeline(sensor_folder: Path) -> list[dict[str, Any]]:
    values = _read_mapping(sensor_folder / MATCH_ROBOT_EE_POSES)
    poses: list[dict[str, Any]] = []
    for filename, record in values.items():
        if not isinstance(record, Mapping) or not isinstance(record.get("robot_ee_pose"), Mapping):
            raise ValueError(f"Invalid matched robot pose record {filename!r}")
        try:
            frame_index = int(Path(str(filename)).stem)
        except ValueError as exc:
            raise ValueError(f"Matched pose frame must have a numeric stem: {filename!r}") from exc
        poses.append(
            {
                "frame_index": frame_index,
                "frame_id": str(filename),
                "timestamp_ns": record.get("frame_timestamp_ns") or record.get("timestamp_ns"),
                "motion": record.get("motion"),
                "matrix": _kuka_pose(record["robot_ee_pose"]),
            }
        )
    return sorted(poses, key=lambda item: (item["frame_index"], item["frame_id"]))


def _raw_timeline(path: Path) -> list[dict[str, Any]]:
    values = _read_mapping(path)
    poses: list[dict[str, Any]] = []
    for key, record in values.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"Invalid raw robot pose record {key!r}")
        pose = record.get("pose") or record.get("robot_ee_pose")
        if not isinstance(pose, Mapping):
            raise ValueError(f"Raw robot pose {key!r} lacks pose coordinates")
        try:
            frame_index = int(key)
        except ValueError:
            frame_index = len(poses)
        poses.append(
            {
                "frame_index": frame_index,
                "frame_id": str(record.get("framename") or key),
                "timestamp_ns": record.get("host_received_timestamp_ns") or record.get("host_wall_timestamp_ns"),
                "motion": record.get("motion"),
                "matrix": _kuka_pose(pose),
            }
        )
    return sorted(poses, key=lambda item: item["frame_index"])


def _timeline_sources(run_root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = [run_root / PROCESSED_DIR / "rectified", run_root / PROCESSED_DIR / SYNCHRONIZED_DIR]
    input_root = next((path for path in candidates if path.is_dir()), None)
    folders = [] if input_root is None else [path for path in input_root.iterdir() if path.is_dir()]
    order: list[str] = []
    for sensor in config.get("capture", {}).get("sensors", []):
        if not isinstance(sensor, Mapping):
            continue
        device_id = str(sensor.get("device_id", ""))
        sensor_type = str(sensor.get("sensor_type", ""))
        for folder in folders:
            if folder.name not in order and (device_id in folder.name or sensor_type.split("_")[0] in folder.name):
                order.append(folder.name)
    order.extend(folder.name for folder in sorted(folders) if folder.name not in order)
    sources: list[dict[str, Any]] = []
    for name in order:
        folder = next(folder for folder in folders if folder.name == name)
        path = folder / MATCH_ROBOT_EE_POSES
        if path.is_file():
            sources.append({"id": f"sensor:{name}", "label": name, "source": path, "kind": "synchronized", "poses": _matched_timeline(folder)})
    if sources:
        return sources
    raw_candidates = [run_root / RAW_ROBOT_EE_POSES]
    raw_candidates.extend(sorted(run_root.glob(f"*/{RAW_ROBOT_EE_POSES}")))
    raw = next((path for path in raw_candidates if path.is_file()), None)
    if raw is not None:
        return [{"id": "raw:robot", "label": "Raw robot poses", "source": raw, "kind": "raw", "poses": _raw_timeline(raw)}]
    return []


def _timeline_metadata(source: Mapping[str, Any], *, default: bool) -> dict[str, Any]:
    poses = source["poses"]
    return {
        "id": source["id"],
        "label": source["label"],
        "kind": source["kind"],
        "frame_count": len(poses),
        "default": default,
        "exact": True,
        "interpolation": "none",
        "page_limit": MAX_TIMELINE_PAGE,
        "source": Path(source["source"]).as_posix(),
    }


def _pose_payload(item: Mapping[str, Any], index: int) -> dict[str, Any]:
    return {
        "index": index,
        "frame_index": item["frame_index"],
        "frame_id": item["frame_id"],
        "timestamp_ns": item["timestamp_ns"],
        "motion": item["motion"],
        "transform": _transform_dict(item["matrix"], "template_base"),
    }


def _preview(poses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(poses) <= MAX_PREVIEW_POSES:
        indices = list(range(len(poses)))
    else:
        indices = sorted({round(i * (len(poses) - 1) / (MAX_PREVIEW_POSES - 1)) for i in range(MAX_PREVIEW_POSES)})
    return [_pose_payload(poses[index], index) for index in indices]


def _profiles(run_root: Path, config: Mapping[str, Any], warnings: list[dict[str, str]]) -> list[CalibrationProfile]:
    value = config.get("calibration_profiles")
    candidates: list[Path] = []
    if isinstance(value, str) and value:
        raw = Path(value)
        candidates.extend([raw, run_root / raw] if not raw.is_absolute() else [raw])
    candidates.append(run_root / "calibration_profiles.json")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        warnings.append({"code": "missing_calibration_profiles", "message": "No calibration profile collection is available; cameras remain unresolved."})
        return []
    try:
        return load_profile_collection(path)
    except (OSError, ValueError) as exc:
        warnings.append({"code": "invalid_calibration_profiles", "message": str(exc)})
        return []


def _bop_export_provenance(
    run_root: Path,
    selected_objects: set[str],
    registry_provenance: Mapping[str, Any],
    warnings: list[dict[str, str]],
    pose_template_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = run_root / BOP_DIR / BOP_EXPORT_MANIFEST
    provenance: dict[str, Any] = {
        "status": "not_exported",
        "manifest_path": path.as_posix(),
        "exported_selected_objects": None,
        "selection_matches": None,
        "registry_matches": None,
    }
    if not path.is_file():
        return provenance
    try:
        manifest = _read_mapping(path)
        if pose_template_selection is not None:
            exported_template = manifest.get("pose_template")
            matches = (
                manifest.get("dataset_mode") == "pose_template"
                and isinstance(exported_template, Mapping)
                and exported_template.get("template_uuid")
                == pose_template_selection.get("template_uuid")
                and exported_template.get("bundle_sha256")
                == pose_template_selection.get("bundle_sha256")
            )
            provenance.update(
                {
                    "status": "current" if matches else "stale",
                    "manifest_schema_version": manifest.get("schema_version"),
                    "pose_template_matches": matches,
                    "template_uuid": pose_template_selection.get("template_uuid"),
                }
            )
            if not matches:
                warnings.append(
                    {
                        "code": "stale_bop_pose_template_provenance",
                        "message": "The BOP export does not match the selected immutable pose template.",
                    }
                )
            return provenance
        exported = manifest.get("selected_objects")
        if not isinstance(exported, list) or any(
            not isinstance(name, str) for name in exported
        ):
            raise ValueError("BOP export manifest has no valid selected_objects snapshot")
        exported_registry = manifest.get("registry_provenance")
        exported_sha = (
            exported_registry.get("source_sha256")
            if isinstance(exported_registry, Mapping)
            else None
        )
        current_sha = registry_provenance.get("source_sha256")
        selection_matches = set(exported) == selected_objects
        registry_matches = bool(exported_sha) and exported_sha == current_sha
        provenance.update(
            {
                "status": (
                    "current" if selection_matches and registry_matches else "stale"
                ),
                "manifest_schema_version": manifest.get("schema_version"),
                "exported_selected_objects": exported,
                "selection_matches": selection_matches,
                "registry_matches": registry_matches,
                "exported_registry_sha256": exported_sha,
                "current_registry_sha256": current_sha,
            }
        )
        if provenance["status"] == "stale":
            warnings.append(
                {
                    "code": "stale_bop_export_provenance",
                    "message": (
                        "The BOP export object snapshot does not match the current "
                        "run selection or object registry; re-export before dataset use."
                    ),
                }
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        provenance.update({"status": "invalid", "error": str(exc)})
        warnings.append(
            {
                "code": "invalid_bop_export_provenance",
                "message": f"Cannot validate BOP export provenance: {exc}",
            }
        )
    return provenance


def _sensor_key(sensor: Mapping[str, Any]) -> str:
    family = str(sensor.get("sensor_type", "sensor"))
    if family == "realsense_d435":
        family = "realsense"
    elif family == "oak_d_pro":
        family = "luxonis"
    elif family == "zed_2i":
        family = "zed_2i"
    return f"{family}_{sensor.get('device_id', 'unknown')}"


def build_cell_scene(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root)
    config = load_run_config_for_run_root(root)
    warnings: list[dict[str, str]] = []
    manager = TransformManager()
    fixed_sources: dict[str, Mapping[str, Any]] = {}
    for edge in config.get("frames", {}).get("fixed_transforms", []):
        try:
            matrix = _matrix(edge["rotation_quaternion_wxyz"], edge["translation_mm"])
            manager.add_transform(str(edge["from"]), str(edge["to"]), matrix)
            fixed_sources[str(edge["from"])] = edge
        except (KeyError, ValueError) as exc:
            warnings.append({"code": "invalid_fixed_transform", "message": str(exc)})

    timelines = _timeline_sources(root, config)
    first_pose = timelines[0]["poses"][0]["matrix"] if timelines and timelines[0]["poses"] else None
    entities = [
        _entity("template_base", "reference_frame", "Template base", transform=_identity(None), status="planned", provenance={"source": "run_config.frames.dataset_reference_frame"}, geometry={"kind": "axes", "size_mm": 100}),
        _entity("hri_template", "template", "HRI cell template", transform=_identity("template_base"), status="planned", provenance={"source": "packaged_hri_template"}, geometry={"kind": "svg_plane", "width_mm": 420, "height_mm": 297, "asset_url": "/assets/cell/template_HRI_LBR_all_center_v2.svg", "mapping": "center=template_base;right=+X;down=+Y"}),
    ]

    for frame, label, kind in (("physical_robot_base", "Physical robot base", "robot_base"), ("tcp", "Robot TCP", "tcp")):
        parent = "robot_flange" if frame == "tcp" else "template_base"
        try:
            transform = manager.get_transform(frame, parent)
        except KeyError:
            entities.append(_entity(frame, kind, label, transform=None, status="unresolved", reason=f"No fixed transform resolves {frame} to {parent}", provenance={"source": "run_config.frames.fixed_transforms"}, geometry={"kind": kind}))
        else:
            entities.append(_entity(frame, kind, label, transform=_transform_dict(transform, parent), status="planned", provenance={"source": fixed_sources.get(frame, {}).get("source", "run_config.frames.fixed_transforms")}, geometry={"kind": kind}))

    if first_pose is not None:
        manager.add_transform("robot_flange", "template_base", first_pose)
        first_pose = manager.get_transform("robot_flange", "template_base")
    entities.append(_entity("robot_flange", "robot_flange", "Robot flange", transform=_transform_dict(first_pose, "template_base") if first_pose is not None else None, status="recorded" if first_pose is not None else "unresolved", reason=None if first_pose is not None else "No synchronized or raw flange pose timeline is available", provenance={"source": Path(timelines[0]["source"]).as_posix() if timelines else None}, geometry={"kind": "flange_proxy"}))

    profiles = _profiles(root, config, warnings)
    for sensor in config.get("capture", {}).get("sensors", []):
        if not isinstance(sensor, Mapping) or not sensor.get("enabled", True):
            continue
        key = _sensor_key(sensor)
        label = str(sensor.get("display_name") or key)
        try:
            profile = select_valid_profile_for_sensor(profiles, key)
            if profile.status != CalibrationStatus.VALID:
                raise ValueError("profile is not valid")
            parent = "robot_flange" if profile.mounting_mode.value == "eye_in_hand" else "template_base"
            transform = _matrix(profile.extrinsics.rotation_quaternion_wxyz, profile.extrinsics.translation_mm)
            manager.add_transform(f"camera:{key}", parent, transform)
            resolved = manager.get_transform(f"camera:{key}", parent)
            intrinsics = profile.rectified_intrinsics or profile.intrinsics
            geometry = {"kind": "camera_frustum", "width": intrinsics.width, "height": intrinsics.height, "fx": intrinsics.cam_k[0], "fy": intrinsics.cam_k[4], "cx": intrinsics.cam_k[2], "cy": intrinsics.cam_k[5], "depth_mm": 180}
            entities.append(_entity(f"camera:{key}", "camera", label, transform=_transform_dict(resolved, parent), status="planned", provenance={"profile_id": profile.profile_id, "schema_version": profile.schema_version}, geometry=geometry))
        except (KeyError, ValueError) as exc:
            entities.append(_entity(f"camera:{key}", "camera", label, transform=None, status="unresolved", reason=f"No valid calibration profile: {exc}", provenance={"source": "calibration_profiles"}, geometry={"kind": "camera_frustum"}))

    encoded_root = quote(root.as_posix(), safe="")
    pose_selection = None
    if config.get("dataset_mode") == "pose_template":
        try:
            pose_selection = load_pose_template_selection(root)
            selected = {str(item["name"]) for item in pose_selection["instances"]}
            registry_provenance = {
                "schema_version": "pose_template_selection.v1",
                "template_uuid": pose_selection["template_uuid"],
                "bundle_sha256": pose_selection["bundle_sha256"],
                "instance_count": len(pose_selection["instances"]),
            }
            for item in pose_selection["instances"]:
                instance_uuid = item["instance_uuid"]
                transform = np.asarray(
                    item["template_base_from_object"]["matrix"], dtype=float
                )
                geometry = {
                    "kind": "mesh",
                    "obj_id": item["obj_id"],
                    "mesh_url": f"/ui/cell-pose-template-assets/{instance_uuid}/mesh?run_root={encoded_root}",
                    "texture_url": (
                        f"/ui/cell-pose-template-assets/{instance_uuid}/texture?run_root={encoded_root}"
                        if "texture" in item["assets"]
                        else None
                    ),
                }
                entities.append(
                    _entity(
                        f"object:{instance_uuid}",
                        "object",
                        item["name"],
                        transform=_transform_dict(transform, "template_base"),
                        status="planned",
                        provenance={
                            "instance_uuid": instance_uuid,
                            "catalog_uuid": item["catalog_uuid"],
                            "obj_id": item["obj_id"],
                            **registry_provenance,
                        },
                        geometry=geometry,
                    )
                )
        except (OSError, ValueError) as exc:
            selected = set()
            registry_provenance = {"schema_version": "pose_template_selection.v1"}
            warnings.append(
                {"code": "invalid_pose_template_selection", "message": str(exc)}
            )
    else:
        registry = load_object_registry(config["object_folder"])
        registry_provenance = registry.provenance()
        selected = set(registry.validate_selection(config.get("selected_objects", [])))
        for entry in registry.entries:
            if entry.name not in selected:
                continue
            if not entry.valid or entry.object_to_template is None:
                entities.append(_entity(f"object:{entry.name}", "object", entry.name, transform=None, status="unresolved", reason="; ".join(entry.errors), provenance={"obj_id": entry.obj_id}, geometry={"kind": "mesh"}))
                continue
            manager.add_transform(f"object:{entry.name}", "template_base", entry.object_to_template)
            object_to_template = manager.get_transform(f"object:{entry.name}", "template_base")
            geometry = {"kind": "mesh", "obj_id": entry.obj_id, "mesh_url": f"/ui/cell-assets/{entry.name}/mesh?run_root={encoded_root}", "texture_url": f"/ui/cell-assets/{entry.name}/texture?run_root={encoded_root}" if entry.texture_path else None}
            entities.append(_entity(f"object:{entry.name}", "object", entry.name, transform=_transform_dict(object_to_template, "template_base"), status="planned", provenance={"obj_id": entry.obj_id, **registry_provenance}, geometry=geometry))

    target_path = root / CALIBRATION_TARGET
    if target_path.is_file():
        try:
            target = _read_mapping(target_path)
            placement = target.get("placement")
            if not isinstance(placement, Mapping):
                raise ValueError("Calibration target has no known placement")
            matrix = _matrix(placement["rotation_quaternion_wxyz"], placement["translation_mm"])
            parent = str(placement.get("to", "template_base"))
            manager.add_transform("calibration_target", parent, matrix)
            matrix = manager.get_transform("calibration_target", parent)
            geometry = {
                "kind": "calibration_target",
                "target_type": target.get("target_type"),
                "target_id": target.get("target_id"),
                "geometry_sha256": target.get("geometry_sha256"),
                "target_bounds": target.get("target_bounds"),
                "grid_size": target.get("grid_size"),
                "marker_length_mm": target.get("marker_length"),
                "marker_separation_mm": target.get("marker_separation"),
                "square_length_mm": target.get("square_length"),
            }
            entities.append(_entity("calibration_target", "calibration_target", "Calibration target", transform=_transform_dict(matrix, parent), status="planned", provenance={"source": target_path.as_posix()}, geometry=geometry))
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            entities.append(_entity("calibration_target", "calibration_target", "Calibration target", transform=None, status="unresolved", reason=str(exc), provenance={"source": target_path.as_posix()}, geometry={"kind": "calibration_target"}))

    timeline_meta = [_timeline_metadata(item, default=index == 0) for index, item in enumerate(timelines)]
    bop_export_provenance = _bop_export_provenance(
        root, selected, registry_provenance, warnings, pose_selection
    )
    return {
        "schema_version": SCENE_SCHEMA_VERSION,
        "coordinate_system": {"units": "millimetres", "handedness": "right", "up_axis": "+Z", "reference_frame": "template_base", "transform_semantics": "entity_to_parent"},
        "run_root": root.as_posix(),
        "entities": entities,
        "warnings": warnings,
        "timelines": timeline_meta,
        "default_timeline_id": timeline_meta[0]["id"] if timeline_meta else None,
        "trajectory_preview": _preview(timelines[0]["poses"]) if timelines else [],
        "object_selection": {
            "selected_objects": sorted(selected),
            "objectless": config.get("dataset_mode") == "objectless",
            "dataset_mode": config.get("dataset_mode", "legacy_registry"),
            "pose_template": registry_provenance if pose_selection is not None else None,
            "registry": registry_provenance,
            "bop_export": bop_export_provenance,
        },
    }


def cell_timeline_page(
    run_root: str | Path,
    timeline_id: str,
    *,
    offset: int = 0,
    limit: int = MAX_TIMELINE_PAGE,
) -> dict[str, Any]:
    if offset < 0:
        raise ValueError("offset must be greater than or equal to 0")
    if limit < 1:
        raise ValueError("limit must be positive")
    limit = min(limit, MAX_TIMELINE_PAGE)
    root = Path(run_root)
    config = load_run_config_for_run_root(root)
    sources = {source["id"]: source for source in _timeline_sources(root, config)}
    if timeline_id not in sources:
        raise KeyError(f"Unknown timeline_id: {timeline_id}")
    source = sources[timeline_id]
    poses = source["poses"]
    page = poses[offset : offset + limit]
    return {
        "schema_version": TIMELINE_SCHEMA_VERSION,
        "timeline": _timeline_metadata(source, default=False),
        "offset": offset,
        "limit": limit,
        "total": len(poses),
        "next_offset": offset + len(page) if offset + len(page) < len(poses) else None,
        "previous_offset": max(0, offset - limit) if offset > 0 else None,
        "poses": [_pose_payload(item, offset + index) for index, item in enumerate(page)],
    }
