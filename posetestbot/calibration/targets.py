"""Calibration target metadata helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


SUPPORTED_TARGET_TYPES = ("aruco_grid", "charuco", "checkerboard")
DEFAULT_TARGET_SPEC = {
    "target_type": "aruco_grid",
    "dictionary": "DICT_5X5_50",
    "grid_size": [4, 3],
    "marker_length": 50.0,
    "marker_separation": 65.0,
    "unit": "mm",
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
    if parsed <= 0:
        raise ValueError(f"{label} must be greater than 0")
    return parsed


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
    """Return a normalized calibration target metadata record."""

    data = dict(DEFAULT_TARGET_SPEC if value is None else value)
    if target_type is not None:
        data["target_type"] = target_type
    if dictionary is not None:
        data["dictionary"] = dictionary
    if grid_size is not None:
        data["grid_size"] = _size(grid_size, label="grid_size")
    if marker_length is not None:
        data["marker_length"] = marker_length
    if marker_separation is not None:
        data["marker_separation"] = marker_separation
    if square_length is not None:
        data["square_length"] = square_length
    if checkerboard_size is not None:
        data["checkerboard_size"] = _size(
            checkerboard_size,
            label="checkerboard_size",
        )
    if unit is not None:
        data["unit"] = unit

    data["target_type"] = str(data.get("target_type", "aruco_grid"))
    if data["target_type"] not in SUPPORTED_TARGET_TYPES:
        raise ValueError(
            "target_type must be one of: " + ", ".join(SUPPORTED_TARGET_TYPES)
        )
    data["unit"] = str(data.get("unit", "mm"))

    if "grid_size" in data:
        data["grid_size"] = _size(data["grid_size"], label="grid_size")
    if "checkerboard_size" in data:
        data["checkerboard_size"] = _size(
            data["checkerboard_size"],
            label="checkerboard_size",
        )
    for key in ("marker_length", "marker_separation", "square_length"):
        if key in data and data[key] is not None:
            data[key] = _positive_float(data[key], label=key)

    if data["target_type"] in {"aruco_grid", "charuco"}:
        if not data.get("dictionary"):
            raise ValueError(f"{data['target_type']} target requires dictionary")
        if "grid_size" not in data:
            raise ValueError(f"{data['target_type']} target requires grid_size")
        if "marker_length" not in data:
            raise ValueError(f"{data['target_type']} target requires marker_length")
    if data["target_type"] == "aruco_grid" and "marker_separation" not in data:
        raise ValueError("aruco_grid target requires marker_separation")
    if data["target_type"] == "charuco" and "square_length" not in data:
        raise ValueError("charuco target requires square_length")
    if data["target_type"] == "checkerboard":
        if "checkerboard_size" not in data:
            raise ValueError("checkerboard target requires checkerboard_size")
        if "square_length" not in data:
            raise ValueError("checkerboard target requires square_length")

    return data


def load_calibration_target_spec(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        value = json.load(f)
    if not isinstance(value, Mapping):
        raise ValueError(f"Calibration target spec must be a JSON object: {path}")
    return normalize_calibration_target_spec(value)
