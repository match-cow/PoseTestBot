"""Transactional, importable BlenderProc input preparation."""

from __future__ import annotations

import json
import math
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

from posetestbot.io.atomic import atomic_write_json, replace_directories
from posetestbot.io.artifacts import CAM_K, MATCH_ROBOT_EE_POSES
from posetestbot.objects.registry import load_object_registry

SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class PreparedSensor:
    sensor_name: str
    output_folder: Path
    frame_count: int
    object_count: int


def validate_subdir(subdir: str) -> str:
    """Require one safe relative directory component."""

    if not SAFE_COMPONENT.fullmatch(subdir) or Path(subdir).name != subdir:
        raise ValueError(f"BlenderProc subdir must be one safe path component: {subdir!r}")
    return subdir


def _read_json_mapping(path: Path, description: str) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {description} {path}: {exc}") from exc
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{description} must be a non-empty JSON object: {path}")
    return value


def load_camera_transformations(path: str | Path) -> Mapping[str, object]:
    return _read_json_mapping(Path(path), "camera transformations")


def _sensor_type_key(sensor_name: str) -> str:
    name = sensor_name.lower()
    if name.startswith("realsense"):
        return "realsense"
    if name.startswith(("luxonis", "oak")):
        return "luxonis"
    if name.startswith(("zed_2i", "zed")):
        return "zed_2i"
    return name.split("_")[0]


def camera_transform_for_sensor(
    camera_transforms: Mapping[str, object], sensor_name: str
) -> Mapping[str, object]:
    value = camera_transforms.get(sensor_name)
    fallback_key = _sensor_type_key(sensor_name)
    if value is None:
        value = camera_transforms.get(fallback_key)
    if not isinstance(value, Mapping):
        available = ", ".join(sorted(str(key) for key in camera_transforms))
        raise KeyError(
            f"No camera transform for sensor {sensor_name!r} or fallback "
            f"{fallback_key!r}. Available keys: {available}"
        )
    return value


def _finite_vector(value: object, *, length: int, field: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain {length} finite numbers") from exc
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must contain {length} finite numbers")
    return array


def _camera_transform(value: Mapping[str, object]) -> tuple[np.ndarray, bool]:
    quaternion = _finite_vector(value.get("quaternion"), length=4, field="quaternion")
    norm = float(np.linalg.norm(quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-3):
        raise ValueError(f"Camera transform quaternion must be normalized; norm={norm}")
    position = _finite_vector(value.get("position"), length=3, field="position")
    mounting_mode = value.get("mounting_mode", "eye_in_hand")
    if mounting_mode not in {"eye_in_hand", "static"}:
        raise ValueError(f"Unsupported camera mounting_mode: {mounting_mode!r}")
    if mounting_mode == "static" and value.get("to") not in {
        "template_base",
        "robot_base",
        "cell_world",
        None,
    }:
        raise ValueError("Static camera transforms must target template_base")
    return pt.transform_from(pr.matrix_from_quaternion(quaternion), position), (
        mounting_mode == "static"
    )


def read_camera_parameters(sensor_folder: Path) -> tuple[np.ndarray, np.ndarray]:
    path = sensor_folder / CAM_K
    if not path.is_file():
        raise FileNotFoundError(f"Missing camera intrinsics: {path}")
    lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < 3 or any(len(line) != 3 for line in lines[:3]):
        raise ValueError(f"Camera intrinsics must start with a 3x3 matrix: {path}")
    try:
        camera_matrix = np.asarray(lines[:3], dtype=float)
        distortion = (
            np.asarray(lines[3], dtype=float)
            if len(lines) > 3
            else np.zeros(5, dtype=float)
        )
    except ValueError as exc:
        raise ValueError(f"Camera intrinsics contain non-numeric values: {path}") from exc
    if not np.all(np.isfinite(camera_matrix)) or camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0:
        raise ValueError(f"Camera intrinsics must be finite with positive focal lengths: {path}")
    if distortion.shape != (5,) or not np.all(np.isfinite(distortion)):
        raise ValueError(f"Distortion coefficients must contain five finite values: {path}")
    return camera_matrix, distortion.reshape(5, 1)


def _ordered_robot_poses(sensor_folder: Path) -> list[Mapping[str, object]]:
    values = _read_json_mapping(sensor_folder / MATCH_ROBOT_EE_POSES, "matched robot poses")
    ordered: list[tuple[int, Mapping[str, object]]] = []
    for filename, record in values.items():
        if not isinstance(filename, str) or Path(filename).suffix != ".png":
            raise ValueError(f"Matched pose key must be a PNG filename: {filename!r}")
        try:
            frame_id = int(Path(filename).stem)
        except ValueError as exc:
            raise ValueError(f"Matched pose key must have a numeric stem: {filename!r}") from exc
        if not isinstance(record, Mapping):
            raise ValueError(f"Matched pose record must be an object: {filename}")
        robot_pose = record.get("robot_ee_pose")
        if not isinstance(robot_pose, Mapping):
            raise ValueError(f"Matched pose record lacks robot_ee_pose: {filename}")
        ordered.append((frame_id, robot_pose))
    ordered.sort(key=lambda item: item[0])
    if len({frame_id for frame_id, _pose in ordered}) != len(ordered):
        raise ValueError(f"Duplicate numeric frame IDs in {sensor_folder / MATCH_ROBOT_EE_POSES}")
    return [pose for _frame_id, pose in ordered]


def _camera_poses(
    robot_poses: list[Mapping[str, object]], camera_transform: Mapping[str, object]
) -> np.ndarray:
    transform, is_static = _camera_transform(camera_transform)
    poses = []
    for index, robot_pose in enumerate(robot_poses):
        if is_static:
            camera_to_template = transform
        else:
            try:
                translation = [float(robot_pose[key]) for key in ("X", "Y", "Z")]
                euler = [float(robot_pose[key]) for key in ("C", "B", "A")]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid robot pose at frame {index}: {robot_pose}") from exc
            if not np.all(np.isfinite([*translation, *euler])):
                raise ValueError(f"Non-finite robot pose at frame {index}")
            flange_to_template_base = pt.transform_from(
                pr.matrix_from_euler(euler, 0, 1, 2, True), translation
            )
            manager = TransformManager()
            manager.add_transform(
                "robot_flange", "template_base", flange_to_template_base
            )
            manager.add_transform("camera", "robot_flange", transform)
            camera_to_template = manager.get_transform("camera", "template_base")
        pose_metres = camera_to_template.copy()
        pose_metres[:3, 3] /= 1000.0
        poses.append(pose_metres)
    return np.asarray(poses, dtype=float)


def _object_transforms(
    object_folder: Path, selected_objects: list[str] | tuple[str, ...] | None
) -> dict[str, np.ndarray]:
    registry = load_object_registry(object_folder)
    names = registry.valid_names if selected_objects is None else selected_objects
    transforms: dict[str, np.ndarray] = {}
    for entry in registry.selected_entries(names):
        assert entry.object_to_template is not None
        transform_metres = entry.object_to_template.copy()
        transform_metres[:3, 3] *= 0.001
        transforms[entry.name] = transform_metres
    return transforms


def _prepare_sensor(
    *,
    sensor_folder: Path,
    staging: Path,
    object_folder: Path,
    object_transforms: Mapping[str, np.ndarray],
    camera_transform: Mapping[str, object],
) -> PreparedSensor:
    camera_matrix, distortion = read_camera_parameters(sensor_folder)
    robot_poses = _ordered_robot_poses(sensor_folder)
    camera_poses = _camera_poses(robot_poses, camera_transform)
    objects_output = staging / "objects"
    objects_output.mkdir(parents=True)
    raw_objects = json.loads((object_folder / "objects.json").read_text())
    atomic_write_json(
        staging / "objects.json",
        {name: raw_objects[name] for name in object_transforms},
    )
    np.save(staging / "camera_matrix.npy", camera_matrix)
    np.save(staging / "dist_coefficients.npy", distortion)
    np.save(staging / "camera_poses.npy", camera_poses)
    for name, transform in object_transforms.items():
        shutil.copy2(object_folder / f"{name}.ply", objects_output / f"{name}.ply")
        texture = object_folder / f"{name}.png"
        if texture.is_file():
            shutil.copy2(texture, objects_output / texture.name)
        np.save(objects_output / f"{name}.npy", transform)
    if np.load(staging / "camera_poses.npy").shape != (len(robot_poses), 4, 4):
        raise ValueError(f"Prepared camera pose count is invalid for {sensor_folder.name}")
    return PreparedSensor(
        sensor_name=sensor_folder.name,
        output_folder=sensor_folder,
        frame_count=len(robot_poses),
        object_count=len(object_transforms),
    )


def prepare_sensor_folders(
    *,
    input_folder: str | Path,
    object_folder: str | Path,
    camera_transformations: Mapping[str, object],
    subdir: str = "blenderproc",
    selected_objects: list[str] | tuple[str, ...] | None = None,
) -> list[PreparedSensor]:
    """Prepare every sensor in staging and promote only after all validate."""

    input_path = Path(input_folder)
    object_path = Path(object_folder)
    validate_subdir(subdir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Synchronized input folder not found: {input_path}")
    sensors = [path for path in sorted(input_path.iterdir()) if path.is_dir()]
    if not sensors:
        raise FileNotFoundError(f"No synchronized sensor folders in {input_path}")
    object_transforms = _object_transforms(object_path, selected_objects)
    staged: list[tuple[Path, Path]] = []
    prepared: list[PreparedSensor] = []
    try:
        for sensor_folder in sensors:
            destination = sensor_folder / subdir
            staging = sensor_folder / f".{subdir}.{uuid.uuid4().hex}.staging"
            staged.append((staging, destination))
            prepared.append(
                _prepare_sensor(
                    sensor_folder=sensor_folder,
                    staging=staging,
                    object_folder=object_path,
                    object_transforms=object_transforms,
                    camera_transform=camera_transform_for_sensor(
                        camera_transformations, sensor_folder.name
                    ),
                )
            )
        replace_directories(staged)
    except Exception:
        for staging, _destination in staged:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return [
        PreparedSensor(
            sensor_name=item.sensor_name,
            output_folder=item.output_folder / subdir,
            frame_count=item.frame_count,
            object_count=item.object_count,
        )
        for item in prepared
    ]


def write_camera_transformations(
    path: str | Path, camera_transformations: Mapping[str, object]
) -> Path:
    return atomic_write_json(path, camera_transformations)
