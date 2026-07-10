"""Calibration target import and normalized geometry contracts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from posetestbot.io.atomic import atomic_write_json


SCHEMA_VERSION = "calibration_target.v1"
SUPPORTED_GENERATOR_VERSION = "1.0"
SUPPORTED_TARGET_TYPES = ("aruco_grid", "charuco", "checkerboard")
SUPPORTED_ARUCO_DICTIONARIES = frozenset(
    [
        *(f"DICT_{bits}X{bits}_{capacity}" for bits in range(4, 8) for capacity in (50, 100, 250, 1000)),
        "DICT_ARUCO_ORIGINAL",
        "DICT_APRILTAG_16h5",
        "DICT_APRILTAG_25h9",
        "DICT_APRILTAG_36h10",
        "DICT_APRILTAG_36h11",
        "DICT_ARUCO_MIP_36h12",
    ]
)
DEFAULT_TARGET_SPEC = {
    "schema_version": SCHEMA_VERSION,
    "target_type": "aruco_grid",
    "dictionary": "DICT_5X5_50",
    "grid_size": [4, 3],
    "marker_length": 50.0,
    "marker_separation": 65.0,
    "marker_ids": list(range(12)),
    "unit": "mm",
    "frame": {
        "name": "aruco_grid",
        "origin": "marker_0_outer_top_left",
        "axes": {"x": "right", "y": "down", "z": "into_board"},
    },
}


def _size(value: str | list[Any] | tuple[Any, ...], *, label: str) -> list[int]:
    if isinstance(value, str):
        separator = "x" if "x" in value.lower() else ","
        parts = value.lower().split(separator)
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        raise ValueError(f"{label} must be a COLSxROWS string or two-item list")
    if len(parts) != 2:
        raise ValueError(f"{label} must contain exactly two values")
    try:
        parsed = [int(item) for item in parts]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} values must be integers") from exc
    if parsed[0] < 1 or parsed[1] < 1:
        raise ValueError(f"{label} values must be positive")
    return parsed


def _positive_float(value: Any, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{label} must be finite and greater than 0")
    return parsed


def _validate_dictionary(value: Any) -> str:
    dictionary = str(value or "")
    if dictionary not in SUPPORTED_ARUCO_DICTIONARIES:
        raise ValueError(f"Unsupported ArUco dictionary: {dictionary!r}")
    return dictionary


def _normalized_grid_frame() -> dict[str, Any]:
    return {
        "name": "aruco_grid",
        "origin": "marker_0_outer_top_left",
        "axes": {"x": "right", "y": "down", "z": "into_board"},
    }


def normalize_calibration_target_spec(
    value: Mapping[str, Any] | None = None,
    *,
    target_type: str | None = None,
    dictionary: str | None = None,
    grid_size: str | list[Any] | tuple[Any, ...] | None = None,
    marker_length: float | None = None,
    marker_separation: float | None = None,
    square_length: float | None = None,
    checkerboard_size: str | list[Any] | tuple[Any, ...] | None = None,
    unit: str | None = None,
) -> dict[str, Any]:
    """Return normalized target metadata, including deterministic grid IDs/frame."""

    data = dict(DEFAULT_TARGET_SPEC if value is None else value)
    overrides = {
        "target_type": target_type,
        "dictionary": dictionary,
        "grid_size": grid_size,
        "marker_length": marker_length,
        "marker_separation": marker_separation,
        "square_length": square_length,
        "checkerboard_size": checkerboard_size,
        "unit": unit,
    }
    data.update({key: item for key, item in overrides.items() if item is not None})
    data["schema_version"] = SCHEMA_VERSION
    data["target_type"] = str(data.get("target_type", "aruco_grid"))
    if data["target_type"] not in SUPPORTED_TARGET_TYPES:
        raise ValueError("target_type must be one of: " + ", ".join(SUPPORTED_TARGET_TYPES))
    data["unit"] = str(data.get("unit", "mm"))
    if data["unit"] != "mm":
        raise ValueError("Calibration target geometry must use millimetres")

    if "grid_size" in data:
        data["grid_size"] = _size(data["grid_size"], label="grid_size")
    if "checkerboard_size" in data:
        data["checkerboard_size"] = _size(data["checkerboard_size"], label="checkerboard_size")
    for key in ("marker_length", "marker_separation", "square_length"):
        if key in data and data[key] is not None:
            data[key] = _positive_float(data[key], label=key)

    if data["target_type"] in {"aruco_grid", "charuco"}:
        data["dictionary"] = _validate_dictionary(data.get("dictionary"))
        if "grid_size" not in data or "marker_length" not in data:
            raise ValueError(f"{data['target_type']} target requires grid_size and marker_length")
    if data["target_type"] == "aruco_grid":
        if "marker_separation" not in data:
            raise ValueError("aruco_grid target requires marker_separation")
        expected_ids = list(range(data["grid_size"][0] * data["grid_size"][1]))
        marker_ids = [int(item) for item in data.get("marker_ids", expected_ids)]
        if marker_ids != expected_ids:
            raise ValueError("ArUco grid marker IDs must be contiguous row-major IDs starting at 0")
        data["marker_ids"] = marker_ids
        data["frame"] = _normalized_grid_frame()
    if data["target_type"] == "charuco" and "square_length" not in data:
        raise ValueError("charuco target requires square_length")
    if data["target_type"] == "checkerboard":
        if "checkerboard_size" not in data or "square_length" not in data:
            raise ValueError("checkerboard target requires checkerboard_size and square_length")
    return data


def import_aruco_gridgen_export(
    source_path: str | Path,
    *,
    aligned_to_template_base: bool = False,
) -> dict[str, Any]:
    """Import an exact ArUcoGridGen 1.0 JSON export into calibration_target.v1."""

    path = Path(source_path)
    source_bytes = path.read_bytes()
    try:
        source = json.loads(source_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid ArUcoGridGen JSON: {path}: {exc}") from exc
    if not isinstance(source, Mapping):
        raise ValueError("ArUcoGridGen export must be a JSON object")
    if str(source.get("version")) != SUPPORTED_GENERATOR_VERSION:
        raise ValueError(
            f"ArUcoGridGen version must be {SUPPORTED_GENERATOR_VERSION!r}"
        )
    settings = source.get("settings")
    grid_info = source.get("grid_info")
    if not isinstance(settings, Mapping) or not isinstance(grid_info, Mapping):
        raise ValueError("ArUcoGridGen export requires settings and grid_info objects")
    if settings.get("board_type") != "aruco_grid":
        raise ValueError("ArUcoGridGen board_type must be 'aruco_grid'")
    for scale_name in ("horizontal_scale", "vertical_scale"):
        try:
            scale = float(settings.get(scale_name, 100.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ArUcoGridGen {scale_name} must be numeric") from exc
        if not math.isclose(scale, 100.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"ArUcoGridGen {scale_name} must be exactly 100%")

    rows = int(settings.get("rows", 0))
    cols = int(settings.get("cols", 0))
    expected_ids = list(range(rows * cols))
    ids = grid_info.get("marker_ids")
    if not isinstance(ids, list) or [int(item) for item in ids] != expected_ids:
        raise ValueError("ArUcoGridGen marker_ids must be contiguous row-major IDs starting at 0")
    if int(grid_info.get("total_markers", -1)) != len(expected_ids):
        raise ValueError("ArUcoGridGen total_markers does not match rows × columns")
    positions = grid_info.get("marker_positions_mm")
    if not isinstance(positions, list) or len(positions) != len(expected_ids):
        raise ValueError("ArUcoGridGen marker_positions_mm does not match marker IDs")
    for expected_id, position in zip(expected_ids, positions, strict=True):
        if not isinstance(position, Mapping) or int(position.get("id", -1)) != expected_id:
            raise ValueError("ArUcoGridGen marker positions must follow row-major marker IDs")
        if int(position.get("row", -1)) != expected_id // cols or int(position.get("col", -1)) != expected_id % cols:
            raise ValueError("ArUcoGridGen marker positions contain inconsistent row/column values")

    target = normalize_calibration_target_spec(
        {
            "target_type": "aruco_grid",
            "dictionary": settings.get("dictionary"),
            "grid_size": [cols, rows],
            "marker_length": settings.get("marker_size_mm"),
            "marker_separation": settings.get("separation_mm"),
            "marker_ids": expected_ids,
            "unit": "mm",
        }
    )
    target["generator_source"] = {
        "format": "ArUcoGridGen",
        "version": SUPPORTED_GENERATOR_VERSION,
        "path": path.as_posix(),
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "export": dict(source),
        "ignored_fields": ["transformation", "grid_info.marker_positions_mm"],
    }
    if aligned_to_template_base:
        target["placement"] = {
            "from": "aruco_grid",
            "to": "template_base",
            "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "translation_mm": [0.0, 0.0, 0.0],
            "source": "operator_declared_aligned_identity",
        }
    return target


def load_calibration_target_spec(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as file:
        value = json.load(file)
    if not isinstance(value, Mapping):
        raise ValueError(f"Calibration target spec must be a JSON object: {path}")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Calibration target schema must be {SCHEMA_VERSION!r}")
    normalized = normalize_calibration_target_spec(value)
    for provenance_key in ("generator_source", "placement"):
        if provenance_key in value:
            normalized[provenance_key] = value[provenance_key]
    return normalized


def write_calibration_target(target: Mapping[str, Any], path: str | Path) -> Path:
    normalized = normalize_calibration_target_spec(target)
    for provenance_key in ("generator_source", "placement"):
        if provenance_key in target:
            normalized[provenance_key] = target[provenance_key]
    return atomic_write_json(Path(path), normalized)


def opencv_grid_board(target: Mapping[str, Any]):
    """Construct the OpenCV dictionary/GridBoard represented by a normalized target."""

    import cv2

    normalized = normalize_calibration_target_spec(target)
    if normalized["target_type"] != "aruco_grid":
        raise ValueError("OpenCV GridBoard construction requires an aruco_grid target")
    dictionary_name = normalized["dictionary"]
    try:
        dictionary_id = getattr(cv2.aruco, dictionary_name)
    except AttributeError as exc:
        raise ValueError(f"Installed OpenCV does not support {dictionary_name}") from exc
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    cols, rows = normalized["grid_size"]
    board = cv2.aruco.GridBoard(
        (cols, rows),
        float(normalized["marker_length"]),
        float(normalized["marker_separation"]),
        dictionary,
        np.array(normalized["marker_ids"], dtype=np.int32),
    )
    return dictionary, board
