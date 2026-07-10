"""Single-pass ArUco GridBoard detection and explicit grid-to-camera pose solving."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from posetestbot.calibration.targets import normalize_calibration_target_spec, opencv_grid_board
from posetestbot.io.atomic import atomic_write_json
from posetestbot.io.artifacts import ARUCO_DETECTIONS, ARUCO_POSE_ESTIMATION, MATCH_ROBOT_EE_POSES, RGB_DIR


DETECTION_SCHEMA_VERSION = "aruco_detections.v1"
POSE_SCHEMA_VERSION = "aruco_pose_estimation.v2"


def _image_paths(sensor_folder: Path) -> list[Path]:
    rgb = sensor_folder / RGB_DIR
    if not rgb.is_dir():
        raise FileNotFoundError(f"Missing synchronized RGB folder: {rgb}")
    paths = sorted(path for path in rgb.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not paths:
        raise FileNotFoundError(f"No RGB frames found: {rgb}")
    return paths


def _target_provenance(target: Mapping[str, Any]) -> dict[str, Any]:
    source = target.get("generator_source")
    return {
        "schema_version": target.get("schema_version"),
        "dictionary": target.get("dictionary"),
        "grid_size": target.get("grid_size"),
        "marker_length_mm": target.get("marker_length"),
        "marker_separation_mm": target.get("marker_separation"),
        "generator_sha256": source.get("sha256") if isinstance(source, Mapping) else None,
    }


def detect_sensor_folder(
    sensor_folder: str | Path,
    target: Mapping[str, Any],
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Detect markers once from native synchronized color frames."""

    folder = Path(sensor_folder)
    normalized = normalize_calibration_target_spec(target)
    dictionary, _board = opencv_grid_board(normalized)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    allowed_ids = set(int(item) for item in normalized["marker_ids"])
    frames: dict[str, Any] = {}
    image_size: list[int] | None = None
    for image_path in _image_paths(folder):
        image = cv2.imread(image_path.as_posix(), cv2.IMREAD_COLOR)
        if image is None:
            frames[image_path.name] = {"ids": [], "corners": [], "rejected_reason": "unreadable_image"}
            continue
        height, width = image.shape[:2]
        current_size = [width, height]
        if image_size is None:
            image_size = current_size
        elif image_size != current_size:
            raise ValueError(f"Mixed RGB resolutions in {folder}: {image_size} and {current_size}")
        corners, ids, rejected = detector.detectMarkers(image)
        matched: list[tuple[int, np.ndarray]] = []
        if ids is not None:
            matched = [
                (int(marker_id), np.asarray(corner, dtype=float).reshape(4, 2))
                for marker_id, corner in zip(ids.reshape(-1), corners, strict=True)
                if int(marker_id) in allowed_ids
            ]
        matched.sort(key=lambda item: item[0])
        all_points = np.concatenate([item[1] for item in matched], axis=0) if matched else None
        frames[image_path.name] = {
            "ids": [item[0] for item in matched],
            "corners": [item[1].tolist() for item in matched],
            "marker_count": len(matched),
            "image_centroid_px": all_points.mean(axis=0).tolist() if all_points is not None else None,
            "rejected_candidate_count": len(rejected),
        }
    report = {
        "schema_version": DETECTION_SCHEMA_VERSION,
        "sensor_name": folder.name,
        "source_projection": "synchronized_native_rgb",
        "image_size": image_size,
        "target": _target_provenance(normalized),
        "frame_count": len(frames),
        "detected_frame_count": sum(1 for frame in frames.values() if frame["marker_count"] > 0),
        "frames": frames,
    }
    atomic_write_json(Path(output_path) if output_path else folder / ARUCO_DETECTIONS, report)
    return report


def _projection(profile: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    native = profile.get("native")
    if not isinstance(native, Mapping):
        raise ValueError("Intrinsic profile requires native projection")
    matrix = np.asarray(native.get("cam_K"), dtype=float).reshape(3, 3)
    distortion = np.asarray(native.get("distortion"), dtype=float).reshape(-1)
    if distortion.shape != (5,):
        raise ValueError("Native Brown-Conrady distortion must contain five coefficients")
    return matrix, distortion


def _matched_points(frame: Mapping[str, Any], board: Any) -> tuple[np.ndarray, np.ndarray] | None:
    ids = frame.get("ids")
    corners = frame.get("corners")
    if not isinstance(ids, list) or not isinstance(corners, list) or not ids or len(ids) != len(corners):
        return None
    object_points, image_points = board.matchImagePoints(
        [np.asarray(corner, dtype=np.float32).reshape(1, 4, 2) for corner in corners],
        np.asarray(ids, dtype=np.int32).reshape(-1, 1),
    )
    if object_points is None or len(object_points) < 4:
        return None
    return np.asarray(object_points).reshape(-1, 3), np.asarray(image_points).reshape(-1, 2)


def _pose_record(
    frame: Mapping[str, Any],
    board: Any,
    matrix: np.ndarray,
    distortion: np.ndarray,
    *,
    target: Mapping[str, Any],
    intrinsic_profile: Mapping[str, Any],
) -> dict[str, Any]:
    ids = frame.get("ids") if isinstance(frame.get("ids"), list) else []
    empty = {
        "schema_version": POSE_SCHEMA_VERSION,
        "rvec": [],
        "tvec": [],
        "len_ids": len(ids),
        "pnp_inlier_indices": [],
        "pnp_inlier_count": 0,
        "mean_reprojection_error_px": None,
        "max_reprojection_error_px": None,
        "transform": {"from": "aruco_grid", "to": "camera", "convention": "opencv_right_down_forward", "unit": "mm"},
        "target": _target_provenance(target),
        "intrinsic_profile_id": intrinsic_profile.get("profile_id"),
        "intrinsic_projection": "native",
    }
    points = _matched_points(frame, board)
    if points is None:
        return empty
    object_points, image_points = points
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success or rvec is None or tvec is None or inliers is None or len(inliers) < 4:
        return empty
    inlier_indices = np.asarray(inliers, dtype=int).reshape(-1)
    rvec, tvec = cv2.solvePnPRefineLM(
        object_points[inlier_indices],
        image_points[inlier_indices],
        matrix,
        distortion,
        rvec,
        tvec,
    )
    projected, _jacobian = cv2.projectPoints(object_points, rvec, tvec, matrix, distortion)
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    inlier_errors = errors[inlier_indices]
    if not np.all(np.isfinite(inlier_errors)):
        raise ValueError("PnP produced non-finite reprojection errors")
    return {
        **empty,
        "rvec": np.asarray(rvec).reshape(3).astype(float).tolist(),
        "tvec": np.asarray(tvec).reshape(3).astype(float).tolist(),
        "pnp_inlier_indices": inlier_indices.astype(int).tolist(),
        "pnp_inlier_count": len(inlier_indices),
        "mean_reprojection_error_px": float(np.mean(inlier_errors)),
        "max_reprojection_error_px": float(np.max(inlier_errors)),
        "all_point_mean_reprojection_error_px": float(np.mean(errors)),
    }


def estimate_sensor_poses(
    sensor_folder: str | Path,
    detections: Mapping[str, Any],
    target: Mapping[str, Any],
    intrinsic_profile: Mapping[str, Any],
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Estimate grid -> camera from stored detections without detecting again."""

    folder = Path(sensor_folder)
    normalized = normalize_calibration_target_spec(target)
    _dictionary, board = opencv_grid_board(normalized)
    matrix, distortion = _projection(intrinsic_profile)
    frames = detections.get("frames")
    if not isinstance(frames, Mapping):
        raise ValueError("ArUco detection report requires frames")
    matched_path = folder / MATCH_ROBOT_EE_POSES
    matched = json.loads(matched_path.read_text()) if matched_path.is_file() else {}
    if not isinstance(matched, Mapping):
        raise ValueError("Matched robot poses must be an object")
    output: dict[str, Any] = {}
    for frame_name, raw_detection in sorted(frames.items()):
        if not isinstance(raw_detection, Mapping):
            continue
        record = dict(matched.get(frame_name, {})) if isinstance(matched.get(frame_name, {}), Mapping) else {}
        record["aruco_detection"] = {
            "ids": list(raw_detection.get("ids", [])),
            "corners": list(raw_detection.get("corners", [])),
        }
        record["aruco_pose_estimation"] = _pose_record(
            raw_detection,
            board,
            matrix,
            distortion,
            target=normalized,
            intrinsic_profile=intrinsic_profile,
        )
        output[str(frame_name)] = record
    atomic_write_json(Path(output_path) if output_path else folder / ARUCO_POSE_ESTIMATION, output)
    return output


def draw_detection_images(
    sensor_folder: str | Path,
    detections: Mapping[str, Any],
    *,
    show: bool = False,
    wait_time: int = 1,
) -> Path:
    folder = Path(sensor_folder)
    output = folder / "aruco"
    output.mkdir(parents=True, exist_ok=True)
    frames = detections.get("frames", {})
    for path in _image_paths(folder):
        image = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
        frame = frames.get(path.name, {}) if isinstance(frames, Mapping) else {}
        ids = frame.get("ids", []) if isinstance(frame, Mapping) else []
        corners = frame.get("corners", []) if isinstance(frame, Mapping) else []
        if image is None:
            continue
        if ids:
            cv2.aruco.drawDetectedMarkers(
                image,
                [np.asarray(corner, dtype=np.float32).reshape(1, 4, 2) for corner in corners],
                np.asarray(ids, dtype=np.int32).reshape(-1, 1),
            )
        if not cv2.imwrite((output / path.name).as_posix(), image):
            raise OSError(f"Failed to write ArUco preview: {output / path.name}")
        if show:
            cv2.imshow("aruco", image)
            if cv2.waitKey(wait_time) == 27:
                break
    if show:
        cv2.destroyAllWindows()
    return output
