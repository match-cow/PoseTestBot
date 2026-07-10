"""Explicit calibration/run frame graph composition helpers."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

from posetestbot.calibration.profiles import CalibrationProfile


def transform_matrix(value: Mapping[str, Any]) -> np.ndarray:
    quaternion = np.asarray(value.get("rotation_quaternion_wxyz"), dtype=float)
    translation = np.asarray(value.get("translation_mm"), dtype=float)
    if quaternion.shape != (4,) or translation.shape != (3,):
        raise ValueError("Frame transform requires quaternion[4] and translation_mm[3]")
    if not np.all(np.isfinite([*quaternion, *translation])) or not math.isclose(
        float(np.linalg.norm(quaternion)), 1.0, abs_tol=1e-3
    ):
        raise ValueError("Frame transform must be finite with a normalized quaternion")
    return pt.transform_from(pr.matrix_from_quaternion(quaternion), translation)


def profile_transform_matrix(profile: CalibrationProfile) -> np.ndarray:
    profile.validate()
    return transform_matrix(
        {
            "rotation_quaternion_wxyz": profile.extrinsics.rotation_quaternion_wxyz,
            "translation_mm": profile.extrinsics.translation_mm,
        }
    )


def resolve_profile_transform(
    profile: CalibrationProfile,
    to_frame: str,
    *,
    fixed_transforms: Sequence[Mapping[str, Any]] = (),
) -> np.ndarray:
    """Resolve camera to a requested frame through typed fixed edges."""

    manager = TransformManager()
    manager.add_transform(
        profile.extrinsics.from_frame.value,
        profile.extrinsics.to_frame.value,
        profile_transform_matrix(profile),
    )
    for edge in fixed_transforms:
        from_frame = str(edge.get("from", ""))
        edge_to = str(edge.get("to", ""))
        if not from_frame or not edge_to or from_frame == edge_to:
            raise ValueError("Fixed frame edges require distinct from/to endpoints")
        manager.add_transform(from_frame, edge_to, transform_matrix(edge))
    try:
        return manager.get_transform("camera", to_frame)
    except KeyError as exc:
        raise ValueError(f"No frame path from camera to {to_frame!r}") from exc


def robot_flange_to_template_base(robot_pose: Mapping[str, Any]) -> np.ndarray:
    """Decode the iiwa KUKA A/B/C-radian stream with explicit endpoints."""

    try:
        translation = np.asarray([float(robot_pose[key]) for key in ("X", "Y", "Z")])
        euler_cba = np.asarray([float(robot_pose[key]) for key in ("C", "B", "A")])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Robot pose must contain finite X,Y,Z,A,B,C values") from exc
    if not np.all(np.isfinite([*translation, *euler_cba])):
        raise ValueError("Robot pose must contain finite X,Y,Z,A,B,C values")
    return pt.transform_from(pr.matrix_from_euler(euler_cba, 0, 1, 2, True), translation)
