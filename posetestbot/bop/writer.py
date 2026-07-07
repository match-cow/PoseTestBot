"""Minimal BOP scene writer for synchronized PoseTestBot sensor folders."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
import trimesh

from posetestbot.calibration.profiles import CalibrationProfile, profile_to_dict
from posetestbot.io.artifacts import (
    BOP_COCO_ANNOTATIONS,
    BOP_DIR,
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

SCHEMA_VERSION = "bop_export_manifest.v1"


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

    depth_by_name = {path.name: path for path in depth_folder.glob("*.png")}
    pairs = [
        (rgb_path, depth_by_name[rgb_path.name])
        for rgb_path in sorted(rgb_folder.glob("*.png"))
        if rgb_path.name in depth_by_name
    ]
    if not pairs:
        raise FileNotFoundError(
            f"No matching RGB/depth PNG frame pairs in {sensor_folder}"
        )
    return pairs


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def object_registry_from_folder(object_folder: str | Path) -> dict[str, int]:
    object_folder = Path(object_folder)
    objects_json = object_folder / "objects.json"
    if not objects_json.is_file():
        raise FileNotFoundError(f"Missing object registry: {objects_json}")
    value = _load_json_if_present(objects_json)
    if not isinstance(value, dict):
        raise ValueError(f"Object registry must be a JSON object: {objects_json}")
    return {
        object_name: index
        for index, object_name in enumerate(sorted(value), start=1)
    }


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


def exact_vertex_diameter(vertices: np.ndarray, *, chunk_size: int = 2048) -> float:
    max_distance_sq = 0.0
    for start in range(0, len(vertices), chunk_size):
        chunk = vertices[start : start + chunk_size]
        distances_sq = np.sum((chunk[:, None, :] - vertices[None, :, :]) ** 2, axis=2)
        max_distance_sq = max(max_distance_sq, float(np.max(distances_sq)))
    return float(np.sqrt(max_distance_sq))


def model_geometry_info(path: Path) -> dict[str, object]:
    try:
        vertices = mesh_vertices(path)
    except Exception as exc:
        return {
            "posetestbot_geometry": {
                "diameter_method": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
            }
        }

    vertex_count = int(len(vertices))
    if vertex_count == 0:
        return {
            "posetestbot_geometry": {
                "diameter_method": "unavailable",
                "vertex_count": 0,
            }
        }

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    size = maxs - mins
    if vertex_count <= 5000:
        diameter = exact_vertex_diameter(vertices)
        diameter_method = "exact_vertex_pairwise"
    else:
        diameter = float(np.linalg.norm(size))
        diameter_method = "aabb_diagonal"

    return {
        "diameter": diameter,
        "min_x": float(mins[0]),
        "min_y": float(mins[1]),
        "min_z": float(mins[2]),
        "size_x": float(size[0]),
        "size_y": float(size[1]),
        "size_z": float(size[2]),
        "posetestbot_geometry": {
            "diameter_method": diameter_method,
            "vertex_count": vertex_count,
        },
    }


def copy_bop_models(
    output_root: str | Path, object_folder: str | Path
) -> list[BopObjectModel]:
    output_root = Path(output_root)
    object_folder = Path(object_folder)
    object_name_to_id = object_registry_from_folder(object_folder)
    models_folder = output_root / MODELS_DIR
    models_folder.mkdir(parents=True, exist_ok=True)

    models_info: dict[str, dict[str, object]] = {}
    models: list[BopObjectModel] = []
    for object_name, obj_id in object_name_to_id.items():
        source_path = object_folder / f"{object_name}.ply"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing object model: {source_path}")
        destination = models_folder / f"obj_{obj_id:06d}.ply"
        shutil.copy2(source_path, destination)
        models_info[str(obj_id)] = {
            "source_name": object_name,
            "source_path": source_path.as_posix(),
            **model_geometry_info(source_path),
        }
        models.append(
            BopObjectModel(
                object_name=object_name,
                obj_id=obj_id,
                source_path=source_path.as_posix(),
                bop_path=destination.as_posix(),
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
    if not object_name_to_id:
        return dict(scene_gt)

    normalized: dict[str, object] = {}
    for image_id, annotations in scene_gt.items():
        if not isinstance(annotations, list):
            normalized[image_id] = annotations
            continue
        normalized_annotations = []
        for annotation in annotations:
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
    for image_id, annotations in sorted(scene_gt.items(), key=lambda item: int(item[0])):
        if not isinstance(annotations, list):
            continue
        counts: dict[int, int] = {}
        for annotation in annotations:
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
    scene_gt: Mapping[str, object], sensor_folder: Path
) -> dict[str, object]:
    mask_folder = sensor_folder / MASKS_DIR
    mask_visib_folder = blenderproc_output_folder(sensor_folder) / "mask_visib"
    scene_gt_info: dict[str, object] = {}

    for image_id, annotations in sorted(scene_gt.items(), key=lambda item: int(item[0])):
        if not isinstance(annotations, list):
            scene_gt_info[image_id] = []
            continue

        image_infos = []
        for annotation_index, _annotation in enumerate(annotations):
            filename = mask_filename(int(image_id), annotation_index)
            object_mask = mask_pixels(mask_folder / filename)
            visible_mask = mask_pixels(mask_visib_folder / filename)
            if object_mask is None and visible_mask is not None:
                object_mask = visible_mask
            if visible_mask is None:
                visible_mask = object_mask

            if object_mask is None:
                image_infos.append({})
                continue

            px_count_all = int(np.count_nonzero(object_mask))
            px_count_visib = int(np.count_nonzero(visible_mask))
            image_infos.append(
                {
                    "bbox_obj": bbox_from_mask(object_mask),
                    "bbox_visib": bbox_from_mask(visible_mask),
                    "px_count_all": px_count_all,
                    "px_count_valid": px_count_visib,
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


def copy_optional_tree(source: Path, destination: Path) -> Path | None:
    if not source.is_dir():
        return None
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    return destination


def camera_matrix_from_profile(profile: CalibrationProfile | None) -> list[float] | None:
    if profile is None:
        return None
    return list(profile.intrinsics.cam_k)


def depth_scale_from_profile(profile: CalibrationProfile | None) -> float | None:
    if profile is None:
        return None
    return float(profile.intrinsics.depth_scale_to_mm)


def scene_camera_calibration_metadata(profile: CalibrationProfile | None) -> dict:
    if profile is None:
        return {}
    return {
        "calibration_profile_id": profile.profile_id,
        "sensor_id": profile.sensor_id,
        "sensor_type": profile.sensor_type.value,
        "mounting_mode": profile.mounting_mode.value,
        "rig_position": profile.rig_position,
        "status": profile.status.value,
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
    scene_folder = output_root / sensor_name / split / f"{scene_id:06d}"
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

    cam_k = camera_matrix_from_profile(calibration_profile) or read_camera_matrix(
        sensor_folder / CAM_K
    )
    depth_scale = depth_scale_from_profile(calibration_profile)
    if depth_scale is None:
        depth_scale = read_depth_scale(sensor_folder / DEPTH_SCALE)
    calibration_metadata = scene_camera_calibration_metadata(calibration_profile)
    frame_map: dict[str, dict[str, str | int]] = {}
    scene_camera: dict[str, dict[str, object]] = {}
    scene_gt: dict[str, object] = {}
    scene_gt_info: dict[str, object] = {}

    frame_pairs = _frame_pairs(sensor_folder)
    for image_id, (rgb_source, depth_source) in enumerate(frame_pairs):
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
            "source_rgb": rgb_source.relative_to(sensor_folder).as_posix(),
            "source_depth": depth_source.relative_to(sensor_folder).as_posix(),
            "bop_rgb": f"{RGB_DIR}/{image_name}",
            "bop_depth": f"{DEPTH_DIR}/{image_name}",
        }

    scene_gt = load_blenderproc_scene_json(sensor_folder, "scene_gt.json") or {
        str(image_id): [] for image_id in range(len(frame_pairs))
    }
    scene_gt = normalize_scene_gt_object_ids(scene_gt, object_name_to_id)
    imported_scene_gt_info = load_blenderproc_scene_json(
        sensor_folder, "scene_gt_info.json"
    )
    scene_gt_info = (
        imported_scene_gt_info
        if imported_scene_gt_info is not None
        else scene_gt_info_from_masks(scene_gt, sensor_folder)
    )
    targets = targets_from_scene_gt(scene_gt, scene_id=scene_id)

    artifacts = {
        "scene_camera": _write_json(scene_folder / "scene_camera.json", scene_camera),
        "scene_gt": _write_json(scene_folder / "scene_gt.json", scene_gt),
        "scene_gt_info": _write_json(scene_folder / "scene_gt_info.json", scene_gt_info),
        "frame_map": _write_json(scene_folder / BOP_FRAME_MAP_JSON, frame_map),
    }
    mask_folder = copy_optional_tree(sensor_folder / MASKS_DIR, scene_folder / "mask")
    if mask_folder is not None:
        artifacts["mask"] = mask_folder

    mask_visib_folder = copy_optional_tree(
        blenderproc_output_folder(sensor_folder) / "mask_visib",
        scene_folder / "mask_visib",
    )
    if mask_visib_folder is not None:
        artifacts["mask_visib"] = mask_visib_folder

    return BopSceneExport(
        sensor_name=sensor_name,
        scene_id=scene_id,
        split=split,
        scene_folder=scene_folder.as_posix(),
        rgb_count=len(frame_pairs),
        depth_count=len(frame_pairs),
        artifacts={key: path.as_posix() for key, path in artifacts.items()},
        calibration_profile_id=(
            calibration_profile.profile_id if calibration_profile is not None else None
        ),
        targets=targets,
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
        scene_folder = Path(export.scene_folder)
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
) -> Path:
    output_root = Path(output_root)
    manifest_path = output_root / BOP_EXPORT_MANIFEST
    _write_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "layout": BOP_DIR,
            "exports": [asdict(export) for export in exports],
            "calibration_profiles_path": (
                Path(calibration_profiles_path).as_posix()
                if calibration_profiles_path is not None
                else None
            ),
            "calibration_profiles": [
                profile_to_dict(profile) for profile in calibration_profiles or []
            ],
            "object_models": [asdict(model) for model in object_models or []],
            "targets_path": (
                Path(targets_path).as_posix() if targets_path is not None else None
            ),
            "multiview_targets_path": (
                Path(multiview_targets_path).as_posix()
                if multiview_targets_path is not None
                else None
            ),
            "coco_annotations_path": (
                Path(coco_annotations_path).as_posix()
                if coco_annotations_path is not None
                else None
            ),
        },
    )
    return manifest_path
