"""Minimal BOP scene writer for synchronized PoseTestBot sensor folders."""

from __future__ import annotations

import json
import hashlib
import math
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
import trimesh

from posetestbot.calibration.profiles import (
    CalibrationProfile,
    CalibrationStatus,
    profile_to_dict,
)
from posetestbot.io.atomic import atomic_write_json
from posetestbot.io.artifacts import (
    BOP_COCO_ANNOTATIONS,
    BOP_DIR,
    BOP_DATASET_INFO,
    BOP_EXPORT_MANIFEST,
    BOP_FRAME_MAP_JSON,
    BOP_MULTIVIEW_TARGETS,
    BOP_TARGETS_BOP19,
    CAM_K,
    DEPTH_DIR,
    DEPTH_SCALE,
    MASKS_DIR,
    MODELS_DIR,
    RGB_DIR,
)
from posetestbot.objects.registry import load_object_registry

SCHEMA_VERSION = "bop_export_manifest.v2"
FRAME_MAP_SCHEMA_VERSION = "posetestbot_bop_frame_map.v2"
DATASET_INFO_SCHEMA_VERSION = "posetestbot_bop_dataset_info.v1"
SAFE_OBJECT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class BopSceneExport:
    sensor_name: str
    scene_id: int
    split: str
    scene_folder: str
    rgb_count: int
    depth_count: int
    artifacts: dict[str, str]
    calibration_profile_id: str | None = None
    targets: list[dict] | None = None
    frame_map: dict[str, dict[str, str | int]] = field(default_factory=dict)


@dataclass(frozen=True)
class BopObjectModel:
    object_name: str
    obj_id: int
    source_path: str
    bop_path: str


def read_camera_matrix(path: Path) -> list[float]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing camera matrix file: {path}")
    values = [float(value) for value in path.read_text().split()]
    if len(values) < 9:
        raise ValueError(f"Camera matrix file {path} has fewer than 9 values")
    return values[:9]


def read_depth_scale(path: Path) -> float:
    if not path.is_file():
        return 1.0
    values = path.read_text().split()
    if not values:
        return 1.0
    return float(values[0])


def _frame_pairs(sensor_folder: Path) -> list[tuple[Path, Path]]:
    rgb_folder = sensor_folder / RGB_DIR
    depth_folder = sensor_folder / DEPTH_DIR
    if not rgb_folder.is_dir():
        raise FileNotFoundError(f"Missing RGB folder: {rgb_folder}")
    if not depth_folder.is_dir():
        raise FileNotFoundError(f"Missing depth folder: {depth_folder}")

    rgb_by_name = {path.name: path for path in rgb_folder.glob("*.png")}
    depth_by_name = {path.name: path for path in depth_folder.glob("*.png")}
    if not rgb_by_name or not depth_by_name:
        raise FileNotFoundError(
            f"No matching RGB/depth PNG frame pairs in {sensor_folder}"
        )
    if set(rgb_by_name) != set(depth_by_name):
        missing_depth = sorted(set(rgb_by_name) - set(depth_by_name))
        missing_rgb = sorted(set(depth_by_name) - set(rgb_by_name))
        raise ValueError(
            "RGB/depth frame names do not match; "
            f"missing_depth={missing_depth}, missing_rgb={missing_rgb}"
        )
    return [
        (rgb_by_name[name], depth_by_name[name]) for name in sorted(rgb_by_name)
    ]


def _write_json(path: Path, value: object) -> Path:
    return atomic_write_json(path, value)


def object_registry_from_folder(object_folder: str | Path) -> dict[str, int]:
    registry = load_object_registry(object_folder)
    invalid = [entry.name for entry in registry.entries if not entry.valid]
    if invalid:
        raise ValueError("Invalid object registry entries: " + ", ".join(invalid))
    return registry.id_mapping


def mesh_vertices(path: Path) -> np.ndarray:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        vertices = [
            np.asarray(geometry.vertices, dtype=float)
            for geometry in mesh.geometry.values()
            if hasattr(geometry, "vertices") and len(geometry.vertices)
        ]
        if not vertices:
            return np.empty((0, 3), dtype=float)
        return np.vstack(vertices)
    if not hasattr(mesh, "vertices"):
        return np.empty((0, 3), dtype=float)
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        return np.empty((0, 3), dtype=float)
    return vertices


def exact_vertex_diameter(vertices: np.ndarray, *, chunk_size: int = 512) -> float:
    max_distance_sq = 0.0
    for start in range(0, len(vertices), chunk_size):
        chunk = vertices[start : start + chunk_size]
        distances_sq = np.sum((chunk[:, None, :] - vertices[None, :, :]) ** 2, axis=2)
        max_distance_sq = max(max_distance_sq, float(np.max(distances_sq)))
    return float(np.sqrt(max_distance_sq))


def model_geometry_info(
    path: Path, cached: Mapping[str, object] | None = None
) -> dict[str, object]:
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if cached is not None:
        geometry = cached.get("posetestbot_geometry")
        if (
            isinstance(geometry, Mapping)
            and geometry.get("source_sha256") == source_sha256
            and geometry.get("diameter_method")
            == "exact_convex_hull_vertex_pairwise"
        ):
            required = {
                "diameter",
                "min_x",
                "min_y",
                "min_z",
                "size_x",
                "size_y",
                "size_z",
            }
            if required <= set(cached):
                return {
                    key: cached[key]
                    for key in (*sorted(required), "posetestbot_geometry")
                }
    vertices = mesh_vertices(path)
    vertex_count = int(len(vertices))
    if vertex_count == 0:
        raise ValueError(f"Object model contains no vertices: {path}")

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    size = maxs - mins
    if not np.all(np.isfinite(vertices)):
        raise ValueError(f"Object model contains non-finite vertices: {path}")
    hull_vertices = vertices
    if vertex_count > 4:
        try:
            hull_vertices = np.asarray(
                trimesh.Trimesh(vertices=vertices, process=False).convex_hull.vertices,
                dtype=float,
            )
        except Exception as exc:
            raise ValueError(f"Unable to compute convex hull for {path}: {exc}") from exc
    diameter = exact_vertex_diameter(hull_vertices)
    if not math.isfinite(diameter) or diameter <= 0:
        raise ValueError(f"Object model diameter must be finite and positive: {path}")
    return {
        "diameter": diameter,
        "min_x": float(mins[0]),
        "min_y": float(mins[1]),
        "min_z": float(mins[2]),
        "size_x": float(size[0]),
        "size_y": float(size[1]),
        "size_z": float(size[2]),
        "posetestbot_geometry": {
            "diameter_method": "exact_convex_hull_vertex_pairwise",
            "vertex_count": vertex_count,
            "convex_hull_vertex_count": int(len(hull_vertices)),
            "source_sha256": source_sha256,
        },
    }


def copy_bop_models(
    output_root: str | Path,
    object_folder: str | Path,
    *,
    geometry_cache: Mapping[str, object] | None = None,
    selected_objects: list[str] | tuple[str, ...] | None = None,
) -> list[BopObjectModel]:
    output_root = Path(output_root)
    object_folder = Path(object_folder)
    registry = load_object_registry(object_folder)
    selected = registry.selected_entries(
        registry.valid_names if selected_objects is None else selected_objects
    )
    object_name_to_id = {entry.name: entry.obj_id for entry in selected}
    models_folder = output_root / MODELS_DIR
    models_folder.mkdir(parents=True, exist_ok=True)

    models_info: dict[str, dict[str, object]] = {}
    models: list[BopObjectModel] = []
    for object_name, obj_id in object_name_to_id.items():
        source_path = object_folder / f"{object_name}.ply"
        try:
            source_path.resolve().relative_to(object_folder.resolve())
        except ValueError as exc:
            raise ValueError(f"Object model escapes registry folder: {source_path}") from exc
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing object model: {source_path}")
        destination = models_folder / f"obj_{obj_id:06d}.ply"
        shutil.copy2(source_path, destination)
        cached = geometry_cache.get(str(obj_id)) if geometry_cache else None
        cached_geometry = (
            cached
            if isinstance(cached, Mapping)
            and cached.get("source_name") == object_name
            else None
        )
        models_info[str(obj_id)] = {
            "source_name": object_name,
            "source_path": source_path.as_posix(),
            **model_geometry_info(source_path, cached_geometry),
        }
        models.append(
            BopObjectModel(
                object_name=object_name,
                obj_id=obj_id,
                source_path=source_path.as_posix(),
                bop_path=destination.relative_to(output_root).as_posix(),
            )
        )

    _write_json(models_folder / "models_info.json", models_info)
    return models


def _load_json_if_present(path: Path) -> object | None:
    if not path.is_file():
        return None
    with open(path, "r") as f:
        return json.load(f)


def blenderproc_output_folder(sensor_folder: Path) -> Path:
    return sensor_folder / "blenderproc" / "output"


def load_blenderproc_scene_json(
    sensor_folder: Path, filename: str
) -> dict[str, object] | None:
    value = _load_json_if_present(blenderproc_output_folder(sensor_folder) / filename)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(
            f"Expected {filename} in {blenderproc_output_folder(sensor_folder)} "
            "to contain a JSON object"
        )
    return value


def normalize_scene_gt_object_ids(
    scene_gt: Mapping[str, object],
    object_name_to_id: Mapping[str, int] | None = None,
) -> dict[str, object]:
    if object_name_to_id is None:
        return dict(scene_gt)

    normalized: dict[str, object] = {}
    for image_id, image_annotations in scene_gt.items():
        if not isinstance(image_annotations, list):
            normalized[image_id] = image_annotations
            continue
        normalized_annotations = []
        for annotation in image_annotations:
            if not isinstance(annotation, dict):
                normalized_annotations.append(annotation)
                continue
            annotation_copy = dict(annotation)
            obj_id = annotation_copy.get("obj_id")
            if isinstance(obj_id, str) and obj_id in object_name_to_id:
                annotation_copy["obj_id"] = object_name_to_id[obj_id]
                annotation_copy["posetestbot_object_name"] = obj_id
            normalized_annotations.append(annotation_copy)
        normalized[image_id] = normalized_annotations
    return normalized


def targets_from_scene_gt(
    scene_gt: Mapping[str, object], *, scene_id: int
) -> list[dict[str, int]]:
    targets: list[dict[str, int]] = []
    for image_id, image_annotations in sorted(
        scene_gt.items(), key=lambda item: int(item[0])
    ):
        if not isinstance(image_annotations, list):
            continue
        counts: dict[int, int] = {}
        for annotation in image_annotations:
            if not isinstance(annotation, dict):
                continue
            obj_id = annotation.get("obj_id")
            try:
                obj_id_int = int(obj_id)
            except (TypeError, ValueError):
                continue
            counts[obj_id_int] = counts.get(obj_id_int, 0) + 1
        for obj_id, inst_count in sorted(counts.items()):
            targets.append(
                {
                    "scene_id": scene_id,
                    "im_id": int(image_id),
                    "obj_id": obj_id,
                    "inst_count": inst_count,
                }
            )
    return targets


def mask_filename(image_id: int, annotation_index: int) -> str:
    return f"{image_id:06d}_{annotation_index:06d}.png"


def mask_pixels(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    image = cv2.imread(path.as_posix(), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = image[:, :, 0]
    return np.asarray(image) > 0


def _read_rgbd_pair(rgb_path: Path, depth_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rgb = cv2.imread(rgb_path.as_posix(), cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(depth_path.as_posix(), cv2.IMREAD_UNCHANGED)
    if rgb is None:
        raise ValueError(f"RGB PNG is unreadable: {rgb_path}")
    if depth is None:
        raise ValueError(f"Depth PNG is unreadable: {depth_path}")
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] not in {3, 4}:
        raise ValueError(f"RGB image must be uint8 with 3 or 4 channels: {rgb_path}")
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(f"Depth image must be single-channel uint16: {depth_path}")
    if rgb.shape[:2] != depth.shape:
        raise ValueError(
            f"RGB/depth dimensions do not match: {rgb_path}, {depth_path}"
        )
    return rgb, depth


def bbox_from_mask(mask: np.ndarray | None) -> list[int]:
    if mask is None or not np.any(mask):
        return [0, 0, 0, 0]
    ys, xs = np.where(mask)
    x_min = int(xs.min())
    y_min = int(ys.min())
    width = int(xs.max() - x_min + 1)
    height = int(ys.max() - y_min + 1)
    return [x_min, y_min, width, height]


def scene_gt_info_from_masks(
    scene_gt: Mapping[str, object],
    sensor_folder: Path,
    frame_pairs: list[tuple[Path, Path]],
) -> dict[str, object]:
    mask_folder = sensor_folder / MASKS_DIR
    mask_visib_folder = blenderproc_output_folder(sensor_folder) / "mask_visib"
    scene_gt_info: dict[str, object] = {}

    for image_id, image_annotations in sorted(
        scene_gt.items(), key=lambda item: int(item[0])
    ):
        if not isinstance(image_annotations, list):
            scene_gt_info[image_id] = []
            continue

        image_infos = []
        image_index = int(image_id)
        if image_index < 0 or image_index >= len(frame_pairs):
            raise ValueError(f"scene_gt image ID is outside exported frames: {image_id}")
        _rgb, depth_image = _read_rgbd_pair(*frame_pairs[image_index])
        for annotation_index, _annotation in enumerate(image_annotations):
            filename = mask_filename(int(image_id), annotation_index)
            object_mask = mask_pixels(mask_folder / filename)
            visible_mask = mask_pixels(mask_visib_folder / filename)
            if object_mask is None and visible_mask is not None:
                object_mask = visible_mask
            if visible_mask is None:
                visible_mask = object_mask

            if object_mask is None:
                raise FileNotFoundError(
                    f"Missing object mask for GT annotation: {mask_folder / filename}"
                )
            if object_mask.shape != depth_image.shape:
                raise ValueError(
                    f"Object mask dimensions do not match depth image: {filename}"
                )
            if visible_mask is not None and visible_mask.shape != depth_image.shape:
                raise ValueError(
                    f"Visible mask dimensions do not match depth image: {filename}"
                )

            px_count_all = int(np.count_nonzero(object_mask))
            px_count_visib = int(np.count_nonzero(visible_mask))
            px_count_valid = int(np.count_nonzero(object_mask & (depth_image > 0)))
            image_infos.append(
                {
                    "bbox_obj": bbox_from_mask(object_mask),
                    "bbox_visib": bbox_from_mask(visible_mask),
                    "px_count_all": px_count_all,
                    "px_count_valid": px_count_valid,
                    "px_count_visib": px_count_visib,
                    "visib_fract": (
                        float(px_count_visib / px_count_all)
                        if px_count_all
                        else 0.0
                    ),
                }
            )
        scene_gt_info[image_id] = image_infos

    return scene_gt_info


def validate_scene_gt(
    scene_gt: Mapping[str, object],
    *,
    frame_count: int,
    object_name_to_id: Mapping[str, int] | None,
) -> None:
    expected_keys = {str(index) for index in range(frame_count)}
    actual_keys = {str(key) for key in scene_gt}
    if actual_keys != expected_keys:
        raise ValueError(
            "scene_gt image IDs must exactly match exported frames; "
            f"expected={sorted(expected_keys)}, actual={sorted(actual_keys)}"
        )
    known_ids = set(object_name_to_id.values()) if object_name_to_id is not None else None
    for image_id, image_annotations in scene_gt.items():
        if not isinstance(image_annotations, list):
            raise ValueError(f"scene_gt[{image_id!r}] must be a list")
        for annotation_index, annotation in enumerate(image_annotations):
            if not isinstance(annotation, Mapping):
                raise ValueError(
                    f"scene_gt[{image_id!r}][{annotation_index}] must be an object"
                )
            try:
                obj_id = int(annotation["obj_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid obj_id in scene_gt[{image_id!r}][{annotation_index}]"
                ) from exc
            if obj_id <= 0 or (known_ids is not None and obj_id not in known_ids):
                raise ValueError(f"Unknown BOP obj_id {obj_id} in scene_gt")
            for key, length in (("cam_R_m2c", 9), ("cam_t_m2c", 3)):
                values = annotation.get(key)
                if not isinstance(values, list) or len(values) != length:
                    raise ValueError(
                        f"scene_gt annotation {key} must contain {length} values"
                    )
                if not all(math.isfinite(float(value)) for value in values):
                    raise ValueError(f"scene_gt annotation {key} must be finite")


def copy_optional_tree(source: Path, destination: Path) -> Path | None:
    if not source.is_dir():
        return None
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    return destination


def copy_scene_masks(
    source: Path,
    destination: Path,
    scene_gt: Mapping[str, object],
) -> Path | None:
    """Copy only masks referenced by validated scene GT annotations."""

    expected = {
        mask_filename(int(image_id), annotation_index)
        for image_id, annotations in scene_gt.items()
        if isinstance(annotations, list)
        for annotation_index in range(len(annotations))
    }
    if not expected:
        return None
    if not source.is_dir():
        return None
    missing = sorted(name for name in expected if not (source / name).is_file())
    if missing:
        raise FileNotFoundError(
            f"Missing referenced masks in {source}: " + ", ".join(missing)
        )
    destination.mkdir(parents=True, exist_ok=False)
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    return destination


def camera_matrix_from_profile(
    profile: CalibrationProfile | None, *, projection: str = "native"
) -> list[float] | None:
    if profile is None:
        return None
    if projection == "rectified":
        if profile.rectified_intrinsics is None:
            raise ValueError(
                f"Calibration profile {profile.profile_id} has no rectified intrinsics"
            )
        return list(profile.rectified_intrinsics.cam_k)
    return list(profile.intrinsics.cam_k)


def depth_scale_from_profile(profile: CalibrationProfile | None) -> float | None:
    if profile is None:
        return None
    return float(profile.intrinsics.depth_scale_to_mm)


def scene_camera_calibration_metadata(
    profile: CalibrationProfile | None, *, projection: str = "native"
) -> dict:
    if profile is None:
        return {}
    return {
        "calibration_profile_id": profile.profile_id,
        "sensor_id": profile.sensor_id,
        "sensor_type": profile.sensor_type.value,
        "mounting_mode": profile.mounting_mode.value,
        "rig_position": profile.rig_position,
        "status": profile.status.value,
        "schema_version": profile.schema_version,
        "projection": projection,
        "projection_provenance": {
            "native_distortion_model": "brown_conrady",
            "rectification": "alpha0" if projection == "rectified" else None,
            "output_resolution_unchanged": projection == "rectified",
        },
        "sync_delta_ms": profile.sync_delta_ms,
        "extrinsics": {
            "from": profile.extrinsics.from_frame.value,
            "to": profile.extrinsics.to_frame.value,
            "rotation_quaternion_wxyz": list(
                profile.extrinsics.rotation_quaternion_wxyz
            ),
            "translation_mm": list(profile.extrinsics.translation_mm),
        },
    }


def export_sensor_scene_to_bop(
    sensor_folder: str | Path,
    output_root: str | Path,
    *,
    split: str = "test",
    scene_id: int = 1,
    overwrite: bool = False,
    calibration_profile: CalibrationProfile | None = None,
    object_name_to_id: Mapping[str, int] | None = None,
) -> BopSceneExport:
    sensor_folder = Path(sensor_folder)
    output_root = Path(output_root)
    sensor_name = sensor_folder.name
    if not re.fullmatch(r"(?:train|val|test)(?:_[A-Za-z0-9.-]+)?", split):
        raise ValueError(f"Invalid BOP split name: {split!r}")
    if scene_id < 0:
        raise ValueError("BOP scene_id must be greater than or equal to 0")
    if calibration_profile is not None:
        calibration_profile.validate()
        if calibration_profile.status != CalibrationStatus.VALID:
            raise ValueError(
                f"Calibration profile {calibration_profile.profile_id} is not valid"
            )
    scene_folder = output_root / split / f"{scene_id:06d}"
    if scene_folder.exists():
        if not overwrite:
            raise FileExistsError(
                f"BOP scene folder already exists: {scene_folder}; pass overwrite=True"
            )
        shutil.rmtree(scene_folder)

    rgb_dest = scene_folder / RGB_DIR
    depth_dest = scene_folder / DEPTH_DIR
    rgb_dest.mkdir(parents=True)
    depth_dest.mkdir(parents=True)

    projection = (
        "rectified"
        if (sensor_folder / "rectification_provenance.json").is_file()
        else "native"
    )
    cam_k = camera_matrix_from_profile(
        calibration_profile, projection=projection
    ) or read_camera_matrix(
        sensor_folder / CAM_K
    )
    depth_scale = depth_scale_from_profile(calibration_profile)
    if depth_scale is None:
        depth_scale = read_depth_scale(sensor_folder / DEPTH_SCALE)
    if len(cam_k) != 9 or not all(math.isfinite(float(value)) for value in cam_k):
        raise ValueError("BOP camera matrix must contain 9 finite values")
    if float(cam_k[0]) <= 0 or float(cam_k[4]) <= 0:
        raise ValueError("BOP camera focal lengths must be positive")
    if not math.isfinite(float(depth_scale)) or float(depth_scale) <= 0:
        raise ValueError("BOP depth scale must be finite and positive")
    calibration_metadata = scene_camera_calibration_metadata(
        calibration_profile, projection=projection
    )
    frame_map: dict[str, dict[str, str | int]] = {}
    scene_camera: dict[str, dict[str, object]] = {}
    scene_gt: dict[str, object] = {}
    scene_gt_info: dict[str, object] = {}

    frame_pairs = _frame_pairs(sensor_folder)
    for image_id, (rgb_source, depth_source) in enumerate(frame_pairs):
        _read_rgbd_pair(rgb_source, depth_source)
        image_name = f"{image_id:06d}.png"
        shutil.copy2(rgb_source, rgb_dest / image_name)
        shutil.copy2(depth_source, depth_dest / image_name)
        image_id_key = str(image_id)
        scene_camera[image_id_key] = {
            "cam_K": cam_k,
            "depth_scale": depth_scale,
        }
        if calibration_metadata:
            scene_camera[image_id_key]["posetestbot_calibration"] = calibration_metadata
        frame_map[image_id_key] = {
            "sensor_name": sensor_name,
            "scene_id": scene_id,
            "source_rgb": rgb_source.relative_to(sensor_folder).as_posix(),
            "source_depth": depth_source.relative_to(sensor_folder).as_posix(),
            "bop_rgb": f"{RGB_DIR}/{image_name}",
            "bop_depth": f"{DEPTH_DIR}/{image_name}",
        }

    scene_gt = load_blenderproc_scene_json(sensor_folder, "scene_gt.json") or {
        str(image_id): [] for image_id in range(len(frame_pairs))
    }
    scene_gt = normalize_scene_gt_object_ids(scene_gt, object_name_to_id)
    validate_scene_gt(
        scene_gt,
        frame_count=len(frame_pairs),
        object_name_to_id=object_name_to_id,
    )
    scene_gt_info = scene_gt_info_from_masks(
        scene_gt,
        sensor_folder,
        frame_pairs,
    )
    targets = targets_from_scene_gt(scene_gt, scene_id=scene_id)

    artifacts = {
        "scene_camera": _write_json(scene_folder / "scene_camera.json", scene_camera),
        "scene_gt": _write_json(scene_folder / "scene_gt.json", scene_gt),
        "scene_gt_info": _write_json(scene_folder / "scene_gt_info.json", scene_gt_info),
    }
    objectless = object_name_to_id is not None and not object_name_to_id
    mask_folder = None if objectless else copy_scene_masks(
        sensor_folder / MASKS_DIR, scene_folder / "mask", scene_gt
    )
    if mask_folder is not None:
        artifacts["mask"] = mask_folder

    mask_visib_folder = None if objectless else copy_scene_masks(
        blenderproc_output_folder(sensor_folder) / "mask_visib",
        scene_folder / "mask_visib",
        scene_gt,
    )
    if mask_visib_folder is not None:
        artifacts["mask_visib"] = mask_visib_folder

    return BopSceneExport(
        sensor_name=sensor_name,
        scene_id=scene_id,
        split=split,
        scene_folder=scene_folder.relative_to(output_root).as_posix(),
        rgb_count=len(frame_pairs),
        depth_count=len(frame_pairs),
        artifacts={
            key: path.relative_to(output_root).as_posix()
            for key, path in artifacts.items()
        },
        calibration_profile_id=(
            calibration_profile.profile_id if calibration_profile is not None else None
        ),
        targets=targets,
        frame_map=frame_map,
    )


def targets_filename(split: str) -> str:
    if split == "test":
        return BOP_TARGETS_BOP19
    return f"{split}_targets_bop19.json"


def write_bop_targets(
    output_root: str | Path, exports: list[BopSceneExport], *, split: str
) -> Path:
    output_root = Path(output_root)
    targets = [
        target
        for export in exports
        for target in export.targets or []
        if export.split == split
    ]
    return _write_json(output_root / targets_filename(split), targets)


def multiview_targets_from_exports(
    exports: list[BopSceneExport],
    *,
    split: str,
) -> dict[str, object]:
    grouped: dict[int, dict[str, object]] = {}
    for export in exports:
        if export.split != split:
            continue
        for target in export.targets or []:
            try:
                obj_id = int(target["obj_id"])
                scene_id = int(target["scene_id"])
                image_id = int(target["im_id"])
                inst_count = int(target["inst_count"])
            except (KeyError, TypeError, ValueError):
                continue
            group = grouped.setdefault(
                obj_id,
                {
                    "obj_id": obj_id,
                    "sensor_names": set(),
                    "scene_ids": set(),
                    "view_count": 0,
                    "instance_count": 0,
                    "views": [],
                },
            )
            group["sensor_names"].add(export.sensor_name)
            group["scene_ids"].add(scene_id)
            group["view_count"] = int(group["view_count"]) + 1
            group["instance_count"] = int(group["instance_count"]) + inst_count
            group["views"].append(
                {
                    "scene_id": scene_id,
                    "sensor_name": export.sensor_name,
                    "im_id": image_id,
                    "inst_count": inst_count,
                }
            )

    targets: list[dict[str, object]] = []
    for obj_id, group in sorted(grouped.items()):
        views = sorted(
            group["views"],
            key=lambda item: (
                int(item["scene_id"]),
                str(item["sensor_name"]),
                int(item["im_id"]),
            ),
        )
        targets.append(
            {
                "obj_id": obj_id,
                "sensor_names": sorted(group["sensor_names"]),
                "scene_ids": sorted(group["scene_ids"]),
                "view_count": group["view_count"],
                "instance_count": group["instance_count"],
                "views": views,
            }
        )

    return {
        "schema_version": "posetestbot_bop_multiview_targets.v1",
        "split": split,
        "scene_count": len(
            {
                export.scene_id
                for export in exports
                if export.split == split and export.targets
            }
        ),
        "object_count": len(targets),
        "targets": targets,
    }


def write_bop_multiview_targets(
    output_root: str | Path,
    exports: list[BopSceneExport],
    *,
    split: str,
) -> Path:
    output_root = Path(output_root)
    return _write_json(
        output_root / BOP_MULTIVIEW_TARGETS,
        multiview_targets_from_exports(exports, split=split),
    )


def write_bop_frame_map(
    output_root: str | Path, exports: list[BopSceneExport]
) -> Path:
    output_root = Path(output_root)
    scenes = {
        str(export.scene_id): {
            "sensor_name": export.sensor_name,
            "split": export.split,
            "scene_folder": export.scene_folder,
            "frames": export.frame_map,
        }
        for export in sorted(exports, key=lambda item: item.scene_id)
    }
    return _write_json(
        output_root / BOP_FRAME_MAP_JSON,
        {"schema_version": FRAME_MAP_SCHEMA_VERSION, "scenes": scenes},
    )


def write_bop_dataset_info(
    output_root: str | Path,
    exports: list[BopSceneExport],
    *,
    dataset_name: str,
    generated_at: str,
) -> Path:
    output_root = Path(output_root)
    splits = sorted({export.split for export in exports})
    return _write_json(
        output_root / BOP_DATASET_INFO,
        {
            "schema_version": DATASET_INFO_SCHEMA_VERSION,
            "name": dataset_name,
            "description": "BOP-scenewise dataset exported by PoseTestBot",
            "bop_format": "scenewise",
            "splits": splits,
            "scene_count": len(exports),
            "sensors": sorted({export.sensor_name for export in exports}),
            "generated_at": generated_at,
        },
    )


def validate_bop_dataset(
    output_root: str | Path,
    exports: list[BopSceneExport],
    *,
    object_models: list[BopObjectModel] | None = None,
    targets_path: str | Path | None = None,
) -> dict[str, object]:
    output_root = Path(output_root)
    scene_ids: set[int] = set()
    scene_image_ids: dict[int, set[int]] = {}
    scene_object_ids: dict[tuple[int, int], set[int]] = {}
    for export in exports:
        if export.scene_id in scene_ids:
            raise ValueError(f"Duplicate BOP scene ID: {export.scene_id}")
        scene_ids.add(export.scene_id)
        scene_folder = output_root / export.scene_folder
        rgb_names = {path.name for path in (scene_folder / RGB_DIR).glob("*.png")}
        depth_names = {path.name for path in (scene_folder / DEPTH_DIR).glob("*.png")}
        if rgb_names != depth_names or len(rgb_names) != export.rgb_count:
            raise ValueError(f"BOP scene frame sets are inconsistent: {scene_folder}")
        image_ids = {int(Path(name).stem) for name in rgb_names}
        scene_image_ids[export.scene_id] = image_ids
        scene_camera = _load_json_if_present(scene_folder / "scene_camera.json")
        scene_gt = _load_json_if_present(scene_folder / "scene_gt.json")
        scene_gt_info = _load_json_if_present(scene_folder / "scene_gt_info.json")
        expected_keys = {str(image_id) for image_id in image_ids}
        for name, value in (
            ("scene_camera", scene_camera),
            ("scene_gt", scene_gt),
            ("scene_gt_info", scene_gt_info),
        ):
            if not isinstance(value, Mapping) or set(value) != expected_keys:
                raise ValueError(f"{name} keys do not match scene images: {scene_folder}")
        assert isinstance(scene_gt, Mapping)
        for image_id, image_annotations in scene_gt.items():
            object_ids = {
                int(annotation["obj_id"])
                for annotation in image_annotations
                if isinstance(annotation, Mapping) and "obj_id" in annotation
            }
            scene_object_ids[(export.scene_id, int(image_id))] = object_ids

    model_ids = {model.obj_id for model in object_models or []}
    if object_models:
        models_info = _load_json_if_present(output_root / MODELS_DIR / "models_info.json")
        if not isinstance(models_info, Mapping) or {
            int(key) for key in models_info
        } != model_ids:
            raise ValueError("models_info.json does not match exported object models")
        for model in object_models:
            if not (output_root / model.bop_path).is_file():
                raise FileNotFoundError(f"Missing BOP object model: {model.bop_path}")

    target_count = 0
    if targets_path is not None:
        targets = _load_json_if_present(Path(targets_path))
        if not isinstance(targets, list):
            raise ValueError("BOP targets must be a JSON list")
        target_count = len(targets)
        for target in targets:
            if not isinstance(target, Mapping):
                raise ValueError("Each BOP target must be a JSON object")
            scene_id = int(target["scene_id"])
            image_id = int(target["im_id"])
            obj_id = int(target["obj_id"])
            if image_id not in scene_image_ids.get(scene_id, set()):
                raise ValueError(f"BOP target references missing scene/image: {target}")
            if obj_id not in scene_object_ids.get((scene_id, image_id), set()):
                raise ValueError(f"BOP target references missing object instance: {target}")
            if model_ids and obj_id not in model_ids:
                raise ValueError(f"BOP target references missing model: {target}")

    return {
        "status": "ok",
        "scene_count": len(exports),
        "frame_count": sum(export.rgb_count for export in exports),
        "model_count": len(object_models or []),
        "target_count": target_count,
    }


def _image_size(path: Path) -> tuple[int, int] | None:
    image = cv2.imread(path.as_posix(), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    height, width = image.shape[:2]
    return int(width), int(height)


def _annotation_bbox(
    annotation: Mapping[str, object],
    annotation_info: object,
) -> list[float]:
    bbox = None
    if isinstance(annotation_info, Mapping):
        bbox = annotation_info.get("bbox_visib") or annotation_info.get("bbox_obj")
    if bbox is None:
        bbox = annotation.get("bbox_visib") or annotation.get("bbox_obj")
    if not isinstance(bbox, list | tuple) or len(bbox) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    try:
        return [float(value) for value in bbox[:4]]
    except (TypeError, ValueError):
        return [0.0, 0.0, 0.0, 0.0]


def _annotation_area(
    bbox: list[float],
    annotation_info: object,
) -> float:
    if isinstance(annotation_info, Mapping):
        for key in ("px_count_visib", "px_count_valid", "px_count_all"):
            value = annotation_info.get(key)
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return float(max(0.0, bbox[2]) * max(0.0, bbox[3]))


def _segmentation_from_mask(mask_path: Path) -> tuple[list[list[float]], int]:
    mask = mask_pixels(mask_path)
    if mask is None:
        return [], 0
    contours, _hierarchy = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    segmentations: list[list[float]] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        points = contour.reshape(-1, 2)
        segmentations.append(points.astype(float).reshape(-1).tolist())
    return segmentations, int(np.count_nonzero(mask))


def coco_annotations_from_exports(
    output_root: str | Path,
    exports: list[BopSceneExport],
    *,
    split: str,
    object_models: list[BopObjectModel] | None = None,
) -> dict[str, object]:
    output_root = Path(output_root)
    categories_by_id: dict[int, dict[str, object]] = {}
    for model in object_models or []:
        categories_by_id[model.obj_id] = {
            "id": model.obj_id,
            "name": model.object_name,
            "supercategory": "object",
        }

    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    annotation_id = 1
    image_id = 1

    for export in sorted(
        (export for export in exports if export.split == split),
        key=lambda export: (export.scene_id, export.sensor_name),
    ):
        scene_folder = output_root / export.scene_folder
        scene_gt = _load_json_if_present(scene_folder / "scene_gt.json")
        scene_gt_info = _load_json_if_present(scene_folder / "scene_gt_info.json")
        if not isinstance(scene_gt, Mapping):
            scene_gt = {}
        if not isinstance(scene_gt_info, Mapping):
            scene_gt_info = {}

        for rgb_path in sorted((scene_folder / RGB_DIR).glob("*.png")):
            try:
                bop_image_id = int(rgb_path.stem)
            except ValueError:
                continue
            size = _image_size(rgb_path)
            width, height = size if size is not None else (0, 0)
            image_record = {
                "id": image_id,
                "file_name": rgb_path.relative_to(output_root).as_posix(),
                "width": width,
                "height": height,
                "posetestbot": {
                    "scene_id": export.scene_id,
                    "im_id": bop_image_id,
                    "sensor_name": export.sensor_name,
                    "split": export.split,
                    "scene_folder": scene_folder.as_posix(),
                },
            }
            images.append(image_record)

            image_key = str(bop_image_id)
            image_annotations = scene_gt.get(image_key, [])
            image_annotation_infos = scene_gt_info.get(image_key, [])
            if not isinstance(image_annotations, list):
                image_id += 1
                continue
            if not isinstance(image_annotation_infos, list):
                image_annotation_infos = []

            for annotation_index, annotation in enumerate(image_annotations):
                if not isinstance(annotation, Mapping):
                    continue
                try:
                    category_id = int(annotation["obj_id"])
                except (KeyError, TypeError, ValueError):
                    continue
                categories_by_id.setdefault(
                    category_id,
                    {
                        "id": category_id,
                        "name": f"obj_{category_id:06d}",
                        "supercategory": "object",
                    },
                )
                annotation_info = (
                    image_annotation_infos[annotation_index]
                    if annotation_index < len(image_annotation_infos)
                    else {}
                )
                bbox = _annotation_bbox(annotation, annotation_info)
                mask_path = scene_folder / "mask" / mask_filename(
                    bop_image_id,
                    annotation_index,
                )
                segmentation, mask_area = _segmentation_from_mask(mask_path)
                area = float(mask_area) if mask_area else _annotation_area(
                    bbox,
                    annotation_info,
                )
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": segmentation,
                        "posetestbot": {
                            "scene_id": export.scene_id,
                            "im_id": bop_image_id,
                            "annotation_index": annotation_index,
                            "sensor_name": export.sensor_name,
                            "mask_path": (
                                mask_path.relative_to(output_root).as_posix()
                                if mask_path.is_file()
                                else None
                            ),
                            "bop_annotation": dict(annotation),
                            "bop_gt_info": (
                                dict(annotation_info)
                                if isinstance(annotation_info, Mapping)
                                else None
                            ),
                        },
                    }
                )
                annotation_id += 1
            image_id += 1

    return {
        "schema_version": "posetestbot_coco_annotations.v1",
        "info": {
            "description": "PoseTestBot COCO-style annotations derived from BOP export.",
            "source_layout": BOP_DIR,
            "split": split,
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            categories_by_id[obj_id] for obj_id in sorted(categories_by_id)
        ],
        "posetestbot": {
            "split": split,
            "scene_count": len(
                {
                    export.scene_id
                    for export in exports
                    if export.split == split
                }
            ),
            "image_count": len(images),
            "annotation_count": len(annotations),
        },
    }


def write_bop_coco_annotations(
    output_root: str | Path,
    exports: list[BopSceneExport],
    *,
    split: str,
    object_models: list[BopObjectModel] | None = None,
) -> Path:
    output_root = Path(output_root)
    return _write_json(
        output_root / BOP_COCO_ANNOTATIONS,
        coco_annotations_from_exports(
            output_root,
            exports,
            split=split,
            object_models=object_models,
        ),
    )


def write_bop_export_manifest(
    output_root: str | Path,
    exports: list[BopSceneExport],
    *,
    calibration_profiles_path: str | Path | None = None,
    calibration_profiles: list[CalibrationProfile] | None = None,
    object_models: list[BopObjectModel] | None = None,
    targets_path: str | Path | None = None,
    multiview_targets_path: str | Path | None = None,
    coco_annotations_path: str | Path | None = None,
    frame_map_path: str | Path | None = None,
    dataset_info_path: str | Path | None = None,
    validation: Mapping[str, object] | None = None,
    selected_objects: list[str] | tuple[str, ...] | None = None,
    stable_id_mapping: Mapping[str, int] | None = None,
    registry_provenance: Mapping[str, object] | None = None,
) -> Path:
    output_root = Path(output_root)
    manifest_path = output_root / BOP_EXPORT_MANIFEST

    def artifact_path(value: str | Path | None) -> str | None:
        if value is None:
            return None
        path = Path(value)
        try:
            return path.relative_to(output_root).as_posix()
        except ValueError:
            return path.as_posix()

    export_entries = []
    for export in exports:
        data = asdict(export)
        data.pop("frame_map", None)
        export_entries.append(data)
    _write_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "format": "bop-scenewise",
            "layout": "<split>/<scene_id>",
            "dataset_root": ".",
            "exports": export_entries,
            "calibration_profiles_path": (
                Path(calibration_profiles_path).as_posix()
                if calibration_profiles_path is not None
                else None
            ),
            "calibration_profiles": [
                profile_to_dict(profile) for profile in calibration_profiles or []
            ],
            "object_models": [asdict(model) for model in object_models or []],
            "selected_objects": list(selected_objects or []),
            "objectless": selected_objects is not None and len(selected_objects) == 0,
            "stable_id_mapping": dict(stable_id_mapping or {}),
            "registry_provenance": dict(registry_provenance or {}),
            "registry_validation": {
                "valid_count": (registry_provenance or {}).get("valid_count", 0),
                "invalid_count": (registry_provenance or {}).get("invalid_count", 0),
            },
            "targets_path": artifact_path(targets_path),
            "multiview_targets_path": artifact_path(multiview_targets_path),
            "coco_annotations_path": artifact_path(coco_annotations_path),
            "frame_map_path": artifact_path(frame_map_path),
            "dataset_info_path": artifact_path(dataset_info_path),
            "validation": dict(validation or {}),
        },
    )
    return manifest_path
