"""Rigid-transform contract for pose-template instances and run placement."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


def validate_rigid_matrix(value: Any, *, label: str = "transform") -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite rigid 4x4 matrix") from exc
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite rigid 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"{label} bottom row must be [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6
    ):
        raise ValueError(f"{label} rotation must be right-handed and orthonormal")
    return matrix


def matrix_from_xyz_rpy(
    *, x_mm: float, y_mm: float, z_mm: float, roll_deg: float, pitch_deg: float, yaw_deg: float
) -> np.ndarray:
    values = np.asarray([x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Pose values must be finite")
    matrix = np.eye(4)
    # scipy lowercase xyz composes the requested Rz(yaw) * Ry(pitch) * Rx(roll).
    matrix[:3, :3] = Rotation.from_euler("xyz", values[3:], degrees=True).as_matrix()
    matrix[:3, 3] = values[:3]
    return matrix


def transform_record(matrix: Sequence[Sequence[float]], *, parent: str, child: str) -> dict[str, Any]:
    rigid = validate_rigid_matrix(matrix)
    xyzw = Rotation.from_matrix(rigid[:3, :3]).as_quat()
    return {
        "semantics": "entity_to_parent",
        "parent_frame": parent,
        "child_frame": child,
        "matrix": rigid.tolist(),
        "translation_mm": rigid[:3, 3].tolist(),
        "rotation_quaternion_wxyz": [
            float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])
        ],
    }


def matrix_from_record(value: Mapping[str, Any], *, label: str = "transform") -> np.ndarray:
    if "matrix" in value:
        return validate_rigid_matrix(value["matrix"], label=label)
    translation = value.get("translation_mm")
    quaternion = value.get("rotation_quaternion_wxyz")
    if not isinstance(translation, (list, tuple)) or len(translation) != 3:
        raise ValueError(f"{label}.translation_mm must contain 3 values")
    if not isinstance(quaternion, (list, tuple)) or len(quaternion) != 4:
        raise ValueError(f"{label}.rotation_quaternion_wxyz must contain 4 values")
    numbers = np.asarray([*translation, *quaternion], dtype=float)
    if not np.all(np.isfinite(numbers)):
        raise ValueError(f"{label} values must be finite")
    norm = float(np.linalg.norm(numbers[3:]))
    if not math.isclose(norm, 1.0, abs_tol=1e-6):
        raise ValueError(f"{label} quaternion must be normalized")
    w, x, y, z = numbers[3:]
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    matrix[:3, 3] = numbers[:3]
    return matrix
