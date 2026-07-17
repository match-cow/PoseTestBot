"""Deterministic multi-method solving for intent-level calibration attempts.

This module is deliberately independent from the legacy stage writers.  It
operates on explicit observations and returns JSON-ready evidence so an attempt
can be kept immutable until an operator promotes a selected result.
"""

from __future__ import annotations

import math
from itertools import combinations
from statistics import mean, median
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from pytransform3d import transformations as pt

from posetestbot.calibration.candidates import (
    _average_transform,
    _robot_ee_to_reference,
)


PNP_METHODS: dict[str, int] = {
    "IPPE": cv2.SOLVEPNP_IPPE,
    "ITERATIVE": cv2.SOLVEPNP_ITERATIVE,
    "SQPNP": cv2.SOLVEPNP_SQPNP,
}
HAND_EYE_METHODS: dict[str, int] = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}
ROBOT_WORLD_HAND_EYE_METHODS: dict[str, int] = {
    "shah": cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
    "li": cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI,
}
EXTRINSIC_METHOD_ORDER = (
    "tsai",
    "park",
    "horaud",
    "andreff",
    "daniilidis",
    "shah",
    "li",
)
PNP_METHOD_ORDER = ("IPPE", "ITERATIVE", "SQPNP")

DEFAULT_MIN_INLIERS = 6
DEFAULT_MAX_MEAN_TRANSLATION_MM = 10.0
DEFAULT_MAX_MEAN_ROTATION_DEG = 5.0
DEFAULT_MAX_OUTLIER_RATIO = 0.25


def _finite_transform(value: np.ndarray) -> bool:
    return value.shape == (4, 4) and bool(np.all(np.isfinite(value)))


def _transform(rotation: Any, translation: Any) -> np.ndarray:
    result = pt.transform_from(
        np.asarray(rotation, dtype=float).reshape(3, 3),
        np.asarray(translation, dtype=float).reshape(3),
    )
    if not _finite_transform(result):
        raise ValueError("solver produced a non-finite transform")
    return result


def invert_transform(value: np.ndarray) -> np.ndarray:
    if not _finite_transform(value):
        raise ValueError("cannot invert a non-finite transform")
    return pt.invert_transform(value)


def transform_record(
    value: np.ndarray,
    *,
    from_frame: str,
    to_frame: str,
) -> dict[str, Any]:
    if not _finite_transform(value):
        raise ValueError("transform must be finite")
    x, y, z, qw, qx, qy, qz = pt.pq_from_transform(value)
    return {
        "from": from_frame,
        "to": to_frame,
        "matrix": np.asarray(value, dtype=float).tolist(),
        "rotation_quaternion_wxyz": [
            float(qw),
            float(qx),
            float(qy),
            float(qz),
        ],
        "translation_mm": [float(x), float(y), float(z)],
    }


def transform_from_record(value: Mapping[str, Any]) -> np.ndarray:
    matrix = value.get("matrix")
    if matrix is not None:
        result = np.asarray(matrix, dtype=float)
        if not _finite_transform(result):
            raise ValueError("recorded transform matrix is invalid")
        return result
    quaternion = np.asarray(value.get("rotation_quaternion_wxyz"), dtype=float)
    translation = np.asarray(value.get("translation_mm"), dtype=float)
    if quaternion.shape != (4,) or translation.shape != (3,):
        raise ValueError("recorded transform requires quaternion and translation")
    result = pt.transform_from_pq(
        np.asarray([*translation.tolist(), *quaternion.tolist()], dtype=float)
    )
    if not _finite_transform(result):
        raise ValueError("recorded transform is non-finite")
    return result


def transform_residual(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    translation_mm = float(np.linalg.norm(left[:3, 3] - right[:3, 3]))
    delta = left[:3, :3].T @ right[:3, :3]
    cosine = max(-1.0, min(1.0, (float(np.trace(delta)) - 1.0) / 2.0))
    return {
        "translation_mm": translation_mm,
        "rotation_deg": math.degrees(math.acos(cosine)),
    }


def residual_summary(records: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not records:
        return {
            "mean_translation_mm": 0.0,
            "median_translation_mm": 0.0,
            "max_translation_mm": 0.0,
            "mean_rotation_deg": 0.0,
            "median_rotation_deg": 0.0,
            "max_rotation_deg": 0.0,
        }
    translations = [float(item["translation_mm"]) for item in records]
    rotations = [float(item["rotation_deg"]) for item in records]
    return {
        "mean_translation_mm": float(mean(translations)),
        "median_translation_mm": float(median(translations)),
        "max_translation_mm": float(max(translations)),
        "mean_rotation_deg": float(mean(rotations)),
        "median_rotation_deg": float(median(rotations)),
        "max_rotation_deg": float(max(rotations)),
    }


def _common_pnp_inliers(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    success, _rvec, _tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        iterationsCount=200,
        reprojectionError=4.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success or inliers is None:
        raise ValueError("robust PnP could not find a common inlier set")
    indices = np.unique(np.asarray(inliers, dtype=int).reshape(-1))
    if len(indices) < 4:
        raise ValueError("robust PnP found fewer than four common inliers")
    return indices


def solve_planar_pnp_candidates(
    object_points: Any,
    image_points: Any,
    camera_matrix: Any,
    distortion: Any,
    *,
    methods: Sequence[str] = PNP_METHOD_ORDER,
) -> dict[str, Any]:
    """Compare supported planar PnP algorithms with one robust point mask."""

    object_array = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_array = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    distortion_array = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if len(object_array) != len(image_array) or len(object_array) < 4:
        raise ValueError("PnP requires at least four paired object/image points")
    if not all(method in PNP_METHODS for method in methods):
        raise ValueError("Unsupported PnP method subset")
    common_indices = _common_pnp_inliers(
        object_array, image_array, matrix, distortion_array
    )
    inlier_objects = object_array[common_indices]
    inlier_images = image_array[common_indices]
    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for method in methods:
        try:
            result = cv2.solvePnPGeneric(
                inlier_objects,
                inlier_images,
                matrix,
                distortion_array,
                flags=PNP_METHODS[method],
            )
            success = bool(result[0])
            rvecs = result[1] if len(result) > 1 else ()
            tvecs = result[2] if len(result) > 2 else ()
            if not success or not rvecs or not tvecs:
                raise ValueError("no pose hypotheses")
            method_candidates = []
            for hypothesis_index, (raw_rvec, raw_tvec) in enumerate(
                zip(rvecs, tvecs, strict=True)
            ):
                rvec = np.asarray(raw_rvec, dtype=np.float64).reshape(3, 1)
                tvec = np.asarray(raw_tvec, dtype=np.float64).reshape(3, 1)
                rvec, tvec = cv2.solvePnPRefineLM(
                    inlier_objects,
                    inlier_images,
                    matrix,
                    distortion_array,
                    rvec,
                    tvec,
                )
                rotation, _ = cv2.Rodrigues(rvec)
                pose = _transform(rotation, tvec)
                depths = (rotation @ object_array.T + tvec).T[:, 2]
                if not np.all(np.isfinite(depths)):
                    raise ValueError("non-finite camera depths")
                if float(np.min(depths)) <= 0.0:
                    failures.append(
                        {
                            "method": method,
                            "hypothesis": str(hypothesis_index),
                            "reason": "non_cheiral_pose",
                        }
                    )
                    continue
                projected, _ = cv2.projectPoints(
                    object_array,
                    rvec,
                    tvec,
                    matrix,
                    distortion_array,
                )
                errors = np.linalg.norm(
                    projected.reshape(-1, 2) - image_array,
                    axis=1,
                )
                if not np.all(np.isfinite(errors)):
                    raise ValueError("non-finite reprojection errors")
                item = {
                    "method": method,
                    "hypothesis": hypothesis_index,
                    "selected_for_method": False,
                    "refinement": "solvePnPRefineLM",
                    "common_inlier_indices": common_indices.astype(int).tolist(),
                    "common_inlier_count": int(len(common_indices)),
                    "mean_reprojection_error_px": float(
                        np.mean(errors[common_indices])
                    ),
                    "max_reprojection_error_px": float(
                        np.max(errors[common_indices])
                    ),
                    "all_point_mean_reprojection_error_px": float(np.mean(errors)),
                    "transform": transform_record(
                        pose,
                        from_frame="aruco_grid",
                        to_frame="camera",
                    ),
                }
                method_candidates.append(item)
            if not method_candidates:
                raise ValueError("all pose hypotheses were non-cheiral")
            method_candidates.sort(
                key=lambda item: (
                    item["mean_reprojection_error_px"],
                    item["hypothesis"],
                )
            )
            method_candidates[0]["selected_for_method"] = True
            candidates.extend(method_candidates)
        except (cv2.error, ValueError, TypeError) as exc:
            failures.append({"method": method, "reason": str(exc)})

    selected = {
        item["method"]: item
        for item in candidates
        if item["selected_for_method"]
    }
    return {
        "common_inlier_indices": common_indices.astype(int).tolist(),
        "common_inlier_count": int(len(common_indices)),
        "candidates": candidates,
        "selected": selected,
        "failures": failures,
    }


def _observation_transforms(
    observations: Sequence[Mapping[str, Any]],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    robot: list[np.ndarray] = []
    target_camera: list[np.ndarray] = []
    for observation in observations:
        robot_pose = observation.get("robot_ee_pose")
        target_pose = observation.get("target_to_camera")
        if not isinstance(robot_pose, Mapping) or not isinstance(target_pose, Mapping):
            raise ValueError("observation requires robot and target-camera transforms")
        robot.append(_robot_ee_to_reference(robot_pose))
        target_camera.append(transform_from_record(target_pose))
    return robot, target_camera


def _calibrate_hand_eye(
    robot: Sequence[np.ndarray],
    target_camera: Sequence[np.ndarray],
    *,
    mode: str,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "eye_in_hand":
        gripper_to_base = list(robot)
    elif mode == "eye_to_hand":
        gripper_to_base = [invert_transform(item) for item in robot]
    else:
        raise ValueError("mode must be eye_in_hand or eye_to_hand")
    rotation, translation = cv2.calibrateHandEye(
        [item[:3, :3] for item in gripper_to_base],
        [item[:3, 3] for item in gripper_to_base],
        [item[:3, :3] for item in target_camera],
        [item[:3, 3] for item in target_camera],
        method=HAND_EYE_METHODS[method],
    )
    primary = _transform(rotation, translation)
    if mode == "eye_in_hand":
        companions = [
            flange_to_base @ primary @ target_to_camera
            for flange_to_base, target_to_camera in zip(
                robot, target_camera, strict=True
            )
        ]
    else:
        companions = [
            invert_transform(flange_to_base) @ primary @ target_to_camera
            for flange_to_base, target_to_camera in zip(
                robot, target_camera, strict=True
            )
        ]
    return primary, _average_transform(companions)


def _calibrate_robot_world_hand_eye(
    robot: Sequence[np.ndarray],
    target_camera: Sequence[np.ndarray],
    *,
    mode: str,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    base_to_flange = [invert_transform(item) for item in robot]
    if mode == "eye_in_hand":
        world_to_camera = list(target_camera)
    elif mode == "eye_to_hand":
        # Relabel the fixed camera as OpenCV's world and the moving target as
        # OpenCV's camera.  Inverting both returned constants then gives the
        # requested camera->base and target->flange transforms.
        world_to_camera = [invert_transform(item) for item in target_camera]
    else:
        raise ValueError("mode must be eye_in_hand or eye_to_hand")
    base_to_world_r, base_to_world_t, gripper_to_camera_r, gripper_to_camera_t = (
        cv2.calibrateRobotWorldHandEye(
            [item[:3, :3] for item in world_to_camera],
            [item[:3, 3] for item in world_to_camera],
            [item[:3, :3] for item in base_to_flange],
            [item[:3, 3] for item in base_to_flange],
            method=ROBOT_WORLD_HAND_EYE_METHODS[method],
        )
    )
    base_to_world = _transform(base_to_world_r, base_to_world_t)
    gripper_to_camera = _transform(gripper_to_camera_r, gripper_to_camera_t)
    if mode == "eye_in_hand":
        return invert_transform(gripper_to_camera), invert_transform(base_to_world)
    return invert_transform(base_to_world), invert_transform(gripper_to_camera)


def solve_extrinsic(
    observations: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    if len(observations) < 3:
        raise ValueError("extrinsic calibration requires at least three observations")
    robot, target_camera = _observation_transforms(observations)
    if method in HAND_EYE_METHODS:
        return _calibrate_hand_eye(
            robot,
            target_camera,
            mode=mode,
            method=method,
        )
    if method in ROBOT_WORLD_HAND_EYE_METHODS:
        return _calibrate_robot_world_hand_eye(
            robot,
            target_camera,
            mode=mode,
            method=method,
        )
    raise ValueError(f"Unsupported extrinsic method: {method}")


def _companion_estimate(
    observation: Mapping[str, Any],
    primary: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    robot, target_camera = _observation_transforms([observation])
    if mode == "eye_in_hand":
        return robot[0] @ primary @ target_camera[0]
    return invert_transform(robot[0]) @ primary @ target_camera[0]


def _companion_estimates(
    observations: Sequence[Mapping[str, Any]],
    primary: np.ndarray,
    *,
    mode: str,
) -> list[np.ndarray]:
    return [
        _companion_estimate(observation, primary, mode=mode)
        for observation in observations
    ]


def _consensus_companion(
    estimates: Sequence[np.ndarray],
    *,
    max_translation_mm: float,
    max_rotation_deg: float,
) -> np.ndarray:
    """Return a deterministic medoid-refined companion transform.

    A single bad target pose can substantially move the arithmetic transform
    average.  Selecting a transform medoid first and averaging only its closure
    inliers keeps the subsequent leave-one-pose-out evaluation robust while
    preserving the existing mean-transform convention for clean evidence.
    """

    if not estimates:
        raise ValueError("companion-transform consensus requires observations")

    def medoid_key(index: int) -> tuple[float, float, int]:
        residuals = [
            transform_residual(estimates[index], candidate)
            for candidate in estimates
        ]
        normalized = [
            item["translation_mm"] / max_translation_mm
            + item["rotation_deg"] / max_rotation_deg
            for item in residuals
        ]
        return float(median(normalized)), float(mean(normalized)), index

    medoid_index = min(range(len(estimates)), key=medoid_key)
    medoid_transform = estimates[medoid_index]
    retained = [
        candidate
        for candidate in estimates
        if (
            (residual := transform_residual(medoid_transform, candidate))[
                "translation_mm"
            ]
            <= max_translation_mm
            and residual["rotation_deg"] <= max_rotation_deg
        )
    ]
    return _average_transform(retained or [medoid_transform])


def _closure_residuals(
    observations: Sequence[Mapping[str, Any]],
    primary: np.ndarray,
    companion: np.ndarray,
    *,
    mode: str,
) -> list[dict[str, float]]:
    return [
        transform_residual(
            _companion_estimate(observation, primary, mode=mode),
            companion,
        )
        for observation in observations
    ]


def _pose_training_sets(
    observations: Sequence[Mapping[str, Any]],
) -> list[tuple[str, ...]]:
    """Build deterministic robust seed sets without combinatorial growth."""

    pose_keys = sorted({_pose_key(item) for item in observations})
    sets = {tuple(pose_keys)}
    sets.update(
        tuple(candidate for candidate in pose_keys if candidate != held_out)
        for held_out in pose_keys
    )
    sample_size = min(6, len(pose_keys) - 1)
    if len(pose_keys) >= 8 and sample_size >= 4:
        combination_count = math.comb(len(pose_keys), sample_size)
        if combination_count <= 64:
            sets.update(combinations(pose_keys, sample_size))
        else:
            generator = np.random.default_rng(0xCA11B)
            attempts = 0
            while len([item for item in sets if len(item) == sample_size]) < 64:
                indices = tuple(
                    sorted(
                        int(item)
                        for item in generator.choice(
                            len(pose_keys), size=sample_size, replace=False
                        )
                    )
                )
                sets.add(tuple(pose_keys[index] for index in indices))
                attempts += 1
                if attempts >= 1024:
                    break
    return sorted(sets, key=lambda item: (-len(item), item))


def _robust_extrinsic_seed(
    observations: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    method: str,
    max_translation_mm: float,
    max_rotation_deg: float,
) -> tuple[np.ndarray, np.ndarray, list[bool]]:
    seeds = []
    for pose_subset in _pose_training_sets(observations):
        training = [
            item for item in observations if _pose_key(item) in pose_subset
        ]
        try:
            primary, _unused_companion = solve_extrinsic(
                training,
                mode=mode,
                method=method,
            )
            companion = _consensus_companion(
                _companion_estimates(observations, primary, mode=mode),
                max_translation_mm=max_translation_mm,
                max_rotation_deg=max_rotation_deg,
            )
            residuals = _closure_residuals(
                observations,
                primary,
                companion,
                mode=mode,
            )
        except (cv2.error, ValueError, TypeError, np.linalg.LinAlgError):
            continue
        mask = [
            item["translation_mm"] <= max_translation_mm
            and item["rotation_deg"] <= max_rotation_deg
            for item in residuals
        ]
        inlier_residuals = [
            item for item, keep in zip(residuals, mask, strict=True) if keep
        ]
        summary = residual_summary(inlier_residuals)
        seeds.append(
            (
                (
                    -sum(mask),
                    summary["median_translation_mm"] / max_translation_mm
                    + summary["median_rotation_deg"] / max_rotation_deg,
                    summary["mean_translation_mm"],
                    summary["mean_rotation_deg"],
                    pose_subset,
                ),
                primary,
                companion,
                mask,
            )
        )
    if not seeds:
        raise ValueError("extrinsic solver could not form a finite closure model")
    _key, primary, companion, mask = min(seeds, key=lambda item: item[0])
    return primary, companion, mask


def _pose_key(observation: Mapping[str, Any]) -> str:
    value = observation.get("motion")
    return str(value) if value not in {None, ""} else str(observation.get("frame_id"))


def _observability_check(observations: Sequence[Mapping[str, Any]]) -> None:
    pose_keys = {_pose_key(item) for item in observations}
    if len(pose_keys) < 4:
        raise ValueError("leave-one-pose-out validation requires at least four poses")
    robot, _target = _observation_transforms(observations)
    relative_rotation = []
    relative_translation = []
    reference = robot[0]
    for item in robot[1:]:
        residual = transform_residual(reference, item)
        relative_rotation.append(residual["rotation_deg"])
        relative_translation.append(residual["translation_mm"])
    if max(relative_rotation, default=0.0) < 1e-3:
        raise ValueError("degenerate robot motion: no rotational excitation")
    if max(relative_translation, default=0.0) < 1e-3:
        raise ValueError("degenerate robot motion: no translational excitation")


def evaluate_extrinsic_candidate(
    observations: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    pnp_method: str,
    extrinsic_method: str,
    sensor_key: str,
    min_inliers: int = DEFAULT_MIN_INLIERS,
    max_mean_translation_mm: float = DEFAULT_MAX_MEAN_TRANSLATION_MM,
    max_mean_rotation_deg: float = DEFAULT_MAX_MEAN_ROTATION_DEG,
    max_outlier_ratio: float = DEFAULT_MAX_OUTLIER_RATIO,
) -> dict[str, Any]:
    """Evaluate one PnP/extrinsic pair with deterministic leave-one-pose-out."""

    candidate_id = f"{sensor_key}|{pnp_method}|{extrinsic_method}"
    try:
        if max_mean_translation_mm <= 0 or max_mean_rotation_deg <= 0:
            raise ValueError("residual thresholds must be greater than zero")
        if not 0 <= max_outlier_ratio <= 1:
            raise ValueError("max_outlier_ratio must be between zero and one")
        _observability_check(observations)
        primary, companion, inlier_mask = _robust_extrinsic_seed(
            observations,
            mode=mode,
            method=extrinsic_method,
            max_translation_mm=max_mean_translation_mm,
            max_rotation_deg=max_mean_rotation_deg,
        )
        for _iteration in range(8):
            fit_observations = [
                item
                for item, keep in zip(observations, inlier_mask, strict=True)
                if keep
            ]
            if len(fit_observations) < 3:
                fit_observations = list(observations)
            primary, _unused_companion = solve_extrinsic(
                fit_observations,
                mode=mode,
                method=extrinsic_method,
            )
            companion = _consensus_companion(
                _companion_estimates(fit_observations, primary, mode=mode),
                max_translation_mm=max_mean_translation_mm,
                max_rotation_deg=max_mean_rotation_deg,
            )
            full_residuals = _closure_residuals(
                observations,
                primary,
                companion,
                mode=mode,
            )
            next_mask = [
                item["translation_mm"] <= max_mean_translation_mm
                and item["rotation_deg"] <= max_mean_rotation_deg
                for item in full_residuals
            ]
            if next_mask == inlier_mask:
                break
            inlier_mask = next_mask

        inlier_observations = [
            item
            for item, keep in zip(observations, inlier_mask, strict=True)
            if keep
        ]
        inlier_pose_keys = sorted({_pose_key(item) for item in inlier_observations})
        held_out_records: list[dict[str, Any]] = []
        for pose_key in inlier_pose_keys:
            train = [
                item
                for item in inlier_observations
                if _pose_key(item) != pose_key
            ]
            holdout = [
                item
                for item in inlier_observations
                if _pose_key(item) == pose_key
            ]
            fold_primary, fold_companion = solve_extrinsic(
                train,
                mode=mode,
                method=extrinsic_method,
            )
            for observation in holdout:
                estimate = _companion_estimate(
                    observation, fold_primary, mode=mode
                )
                residual = transform_residual(estimate, fold_companion)
                held_out_records.append(
                    {
                        "pose": pose_key,
                        "frame_id": observation.get("frame_id"),
                        "validation_split": "leave_one_pose_out_inlier",
                        **residual,
                    }
                )
        for observation, keep, residual in zip(
            observations, inlier_mask, full_residuals, strict=True
        ):
            if keep:
                continue
            held_out_records.append(
                {
                    "pose": _pose_key(observation),
                    "frame_id": observation.get("frame_id"),
                    "validation_split": "rejected_closure_outlier",
                    **residual,
                }
            )
        held_out_summary = residual_summary(held_out_records)
        inlier_count = sum(inlier_mask)
        outlier_count = len(inlier_mask) - inlier_count
        outlier_ratio = outlier_count / len(inlier_mask) if inlier_mask else 1.0
        full_summary = residual_summary(full_residuals)
        reprojection_values = [
            float(item.get("mean_reprojection_error_px", 0.0))
            for item in observations
        ]
        passing = (
            inlier_count >= min_inliers
            and held_out_summary["mean_translation_mm"] <= max_mean_translation_mm
            and held_out_summary["mean_rotation_deg"] <= max_mean_rotation_deg
            and outlier_ratio <= max_outlier_ratio
        )
        score = (
            held_out_summary["median_translation_mm"] / max_mean_translation_mm
            + held_out_summary["median_rotation_deg"] / max_mean_rotation_deg
        )
        primary_frames = (
            ("camera", "robot_flange")
            if mode == "eye_in_hand"
            else ("camera", "template_base")
        )
        companion_frames = (
            ("aruco_grid", "template_base")
            if mode == "eye_in_hand"
            else ("aruco_grid", "robot_flange")
        )
        checks = [
            {
                "name": "minimum_inliers",
                "status": "ok" if inlier_count >= min_inliers else "error",
                "actual": inlier_count,
                "threshold": min_inliers,
            },
            {
                "name": "mean_translation_residual",
                "status": (
                    "ok"
                    if held_out_summary["mean_translation_mm"]
                    <= max_mean_translation_mm
                    else "error"
                ),
                "actual": held_out_summary["mean_translation_mm"],
                "threshold": max_mean_translation_mm,
                "unit": "mm",
            },
            {
                "name": "mean_rotation_residual",
                "status": (
                    "ok"
                    if held_out_summary["mean_rotation_deg"] <= max_mean_rotation_deg
                    else "error"
                ),
                "actual": held_out_summary["mean_rotation_deg"],
                "threshold": max_mean_rotation_deg,
                "unit": "deg",
            },
            {
                "name": "outlier_ratio",
                "status": "ok" if outlier_ratio <= max_outlier_ratio else "error",
                "actual": outlier_ratio,
                "threshold": max_outlier_ratio,
            },
        ]
        return {
            "candidate_id": candidate_id,
            "sensor_key": sensor_key,
            "pnp_method": pnp_method,
            "extrinsic_method": extrinsic_method,
            "algorithms": [pnp_method, extrinsic_method],
            "status": "passing" if passing else "failed",
            "validation_state": "passed" if passing else "failed",
            "score": float(score),
            "observation_count": len(observations),
            "inlier_count": inlier_count,
            "outlier_count": outlier_count,
            "outlier_ratio": float(outlier_ratio),
            "mean_reprojection_error_px": (
                float(mean(reprojection_values)) if reprojection_values else None
            ),
            "primary_transform": transform_record(
                primary,
                from_frame=primary_frames[0],
                to_frame=primary_frames[1],
            ),
            "companion_transform": transform_record(
                companion,
                from_frame=companion_frames[0],
                to_frame=companion_frames[1],
            ),
            "held_out_residuals": held_out_summary,
            "fit_residuals": full_summary,
            "leave_one_pose_out": held_out_records,
            "checks": checks,
        }
    except (cv2.error, ValueError, TypeError, np.linalg.LinAlgError) as exc:
        return {
            "candidate_id": candidate_id,
            "sensor_key": sensor_key,
            "pnp_method": pnp_method,
            "extrinsic_method": extrinsic_method,
            "algorithms": [pnp_method, extrinsic_method],
            "status": "error",
            "validation_state": "failed",
            "score": None,
            "observation_count": len(observations),
            "inlier_count": 0,
            "outlier_count": len(observations),
            "outlier_ratio": 1.0,
            "error": str(exc),
            "checks": [
                {
                    "name": "solver",
                    "status": "error",
                    "message": str(exc),
                }
            ],
        }


def rank_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rank passing candidates, followed by deterministic failed evidence."""

    pnp_order = {name: index for index, name in enumerate(PNP_METHOD_ORDER)}
    extrinsic_order = {
        name: index for index, name in enumerate(EXTRINSIC_METHOD_ORDER)
    }

    def key(item: Mapping[str, Any]) -> tuple[Any, ...]:
        passing = item.get("status") == "passing"
        score = float(item.get("score")) if item.get("score") is not None else math.inf
        reprojection = (
            float(item.get("mean_reprojection_error_px"))
            if item.get("mean_reprojection_error_px") is not None
            else math.inf
        )
        return (
            0 if passing else 1,
            score,
            reprojection,
            -int(item.get("inlier_count", 0)),
            pnp_order.get(str(item.get("pnp_method")), len(pnp_order)),
            extrinsic_order.get(
                str(item.get("extrinsic_method")), len(extrinsic_order)
            ),
            str(item.get("candidate_id")),
        )

    ranked = [dict(item) for item in sorted(candidates, key=key)]
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
        item["recommended"] = index == 1 and item.get("status") == "passing"
    return ranked
