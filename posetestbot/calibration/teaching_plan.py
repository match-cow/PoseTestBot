"""Validation and pose conversion for the iiwa Workbench teaching plan."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from posetestbot.calibration.frame_graph import robot_flange_to_template_base


SCHEMA_VERSION = "iiwa_calibration_teaching_plan.v2"
DEFAULT_TEACHING_PLAN_PATH = (
    Path(__file__).resolve().parents[2] / "iiwa" / "calibration_teaching_plan.v2.json"
)
POSE_KEYS = ("X", "Y", "Z", "A", "B", "C")
ORIENTATION_KEYS = ("A", "B", "C")
EXPECTED_PHASE_ORDER = ("coverage_raster", "orientation_dither")


def load_teaching_plan(path: str | Path = DEFAULT_TEACHING_PLAN_PATH) -> dict[str, Any]:
    """Load and validate a versioned, repository-owned teaching plan."""

    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Teaching plan must be a JSON object: {source}")
    validate_teaching_plan(data)
    return data


def _finite_values(value: Mapping[str, Any], keys: Sequence[str], label: str) -> list[float]:
    if set(value) != set(keys):
        raise ValueError(f"{label} must contain exactly {','.join(keys)}")
    try:
        result = [float(value[key]) for key in keys]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must be finite")
    return result


def validate_teaching_plan(plan: Mapping[str, Any]) -> None:
    """Reject plans that can drift from the nine-frame Workbench contract."""

    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Teaching plan schema_version must be {SCHEMA_VERSION!r}")

    base_path = plan.get("base_path")
    if base_path != "/PoseTestBot/TemplateBase":
        raise ValueError("Teaching plan base_path must be /PoseTestBot/TemplateBase")
    if plan.get("motion_point") != "robot_flange":
        raise ValueError("Teaching plan motion_point must be robot_flange")
    if plan.get("phase_anchor") != "CalibrationCenter":
        raise ValueError("Teaching plan phase_anchor must be CalibrationCenter")

    units = plan.get("units")
    if not isinstance(units, Mapping) or units.get("translation") != "millimetres":
        raise ValueError("Teaching plan translation units must be millimetres")
    if units.get("orientation") != "degrees":
        raise ValueError("Teaching plan orientation units must be degrees")
    if units.get("orientation_convention") != "KUKA A/B/C":
        raise ValueError("Teaching plan must declare the KUKA A/B/C convention")

    template = plan.get("template")
    if not isinstance(template, Mapping):
        raise ValueError("Teaching plan must describe the HRI template")
    if template.get("width_mm") != 420 or template.get("height_mm") != 297:
        raise ValueError("Teaching plan HRI template must be exactly 420 x 297 mm")
    if template.get("centered_at") != "TemplateBase":
        raise ValueError("Teaching plan template must be centered at TemplateBase")

    frames = plan.get("frames")
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise ValueError("Teaching plan frames must be a list")
    if len(frames) != 9:
        raise ValueError("Teaching plan must contain exactly nine taught grid frames")

    frame_map: dict[str, Mapping[str, Any]] = {}
    paths: set[str] = set()
    for frame in frames:
        if not isinstance(frame, Mapping):
            raise ValueError("Every teaching-plan frame must be an object")
        name = frame.get("name")
        path = frame.get("path")
        if not isinstance(name, str) or not name:
            raise ValueError("Every teaching-plan frame requires a name")
        if name in frame_map:
            raise ValueError(f"Duplicate teaching-plan frame name: {name}")
        expected_path = f"{base_path}/{name}"
        if path != expected_path:
            raise ValueError(f"Frame {name} must be a direct child at {expected_path}")
        if path in paths:
            raise ValueError(f"Duplicate teaching-plan frame path: {path}")
        seed = frame.get("seed")
        if not isinstance(seed, Mapping):
            raise ValueError(f"Frame {name} requires one XYZABC seed")
        _finite_values(seed, POSE_KEYS, f"Frame {name} seed")
        frame_map[name] = frame
        paths.add(str(path))

    if "CalibrationCenter" not in frame_map:
        raise ValueError("Teaching plan requires one taught CalibrationCenter anchor")

    phase_order = plan.get("phase_order")
    if phase_order != list(EXPECTED_PHASE_ORDER):
        raise ValueError(f"Teaching plan phase_order must be {list(EXPECTED_PHASE_ORDER)!r}")
    phases = plan.get("phases")
    if not isinstance(phases, Sequence) or isinstance(phases, (str, bytes)):
        raise ValueError("Teaching plan phases must be a list")
    if [phase.get("id") for phase in phases if isinstance(phase, Mapping)] != list(
        EXPECTED_PHASE_ORDER
    ):
        raise ValueError("Teaching-plan phases must match phase_order")

    coverage = phases[0]
    if coverage.get("anchor_frame") != "CalibrationCenter":
        raise ValueError("Coverage raster must be anchored at CalibrationCenter")
    coverage_motions = coverage.get("motions")
    if not isinstance(coverage_motions, Sequence) or len(coverage_motions) != 10:
        raise ValueError("Coverage raster must contain two PTP transits and eight LIN legs")
    previous_to: str | None = None
    capture_labels: set[str] = set()
    for index, motion in enumerate(coverage_motions):
        if not isinstance(motion, Mapping):
            raise ValueError(f"Coverage motion {index} must be an object")
        from_name = motion.get("from")
        to_name = motion.get("to")
        if from_name not in frame_map or to_name not in frame_map:
            raise ValueError(f"Coverage motion {index} references an unknown frame")
        if previous_to is not None and previous_to != from_name:
            raise ValueError("Coverage motions are not a continuous route")
        previous_to = str(to_name)
        expected_type = "PTP" if index in {0, 9} else "LIN"
        if motion.get("motion_type") != expected_type:
            raise ValueError(f"Coverage motion {index} must be {expected_type}")
        label = motion.get("capture_label")
        if expected_type == "PTP" and label is not None:
            raise ValueError("Coverage PTP anchor transits must not be capture motions")
        if expected_type == "LIN":
            if not isinstance(label, str) or not label:
                raise ValueError("Every coverage LIN leg requires a capture label")
            if label in capture_labels:
                raise ValueError(f"Duplicate capture label: {label}")
            capture_labels.add(label)
    if coverage_motions[0].get("from") != "CalibrationCenter":
        raise ValueError("Coverage raster must start at CalibrationCenter")
    if coverage_motions[-1].get("to") != "CalibrationCenter":
        raise ValueError("Coverage raster must return to CalibrationCenter")

    orientation = phases[1]
    if orientation.get("anchor_frame") != "CalibrationCenter":
        raise ValueError("Orientation dither must be anchored at CalibrationCenter")
    if orientation.get("reference_frame") != "CalibrationCenter":
        raise ValueError("Relative orientation must use CalibrationCenter as reference")
    if orientation.get("motion_type") != "LIN_REL":
        raise ValueError("Orientation dither must use LIN_REL motions")
    orientation_motions = orientation.get("motions")
    if not isinstance(orientation_motions, Sequence) or len(orientation_motions) != 9:
        raise ValueError("Orientation dither must contain exactly nine relative legs")

    cumulative = {key: 0.0 for key in ORIENTATION_KEYS}
    for index, motion in enumerate(orientation_motions):
        if not isinstance(motion, Mapping):
            raise ValueError(f"Orientation motion {index} must be an object")
        delta = motion.get("delta")
        result_offset = motion.get("result_offset")
        if not isinstance(delta, Mapping) or not isinstance(result_offset, Mapping):
            raise ValueError(f"Orientation motion {index} requires delta and result_offset")
        delta_values = _finite_values(delta, POSE_KEYS, f"Orientation motion {index} delta")
        if any(not math.isclose(value, 0.0, abs_tol=1e-12) for value in delta_values[:3]):
            raise ValueError("Relative orientation legs must keep XYZ fixed")
        result_values = _finite_values(
            result_offset,
            ORIENTATION_KEYS,
            f"Orientation motion {index} result_offset",
        )
        for key in ORIENTATION_KEYS:
            cumulative[key] += float(delta[key])
        if any(
            not math.isclose(cumulative[key], result_values[key_index], abs_tol=1e-12)
            for key_index, key in enumerate(ORIENTATION_KEYS)
        ):
            raise ValueError(f"Orientation motion {index} result_offset is inconsistent")
        label = motion.get("capture_label")
        if not isinstance(label, str) or not label:
            raise ValueError("Every relative orientation leg requires a capture label")
        if label in capture_labels:
            raise ValueError(f"Duplicate capture label: {label}")
        capture_labels.add(label)
    if any(not math.isclose(value, 0.0, abs_tol=1e-12) for value in cumulative.values()):
        raise ValueError("Relative orientation sequence must finish at CalibrationCenter")


def frames_by_name(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Return validated frame records keyed by their Workbench child name."""

    validate_teaching_plan(plan)
    return {str(frame["name"]): frame for frame in plan["frames"]}


def seed_pose_radians(frame: Mapping[str, Any]) -> dict[str, float]:
    """Convert a manifest seed from KUKA degrees to the streamed radian schema."""

    seed = frame.get("seed")
    if not isinstance(seed, Mapping):
        raise ValueError(f"Frame {frame.get('name', '<unknown>')} has no numeric seed")
    return {
        "X": float(seed["X"]),
        "Y": float(seed["Y"]),
        "Z": float(seed["Z"]),
        "A": math.radians(float(seed["A"])),
        "B": math.radians(float(seed["B"])),
        "C": math.radians(float(seed["C"])),
    }


def seed_transform_matrix(frame: Mapping[str, Any]) -> np.ndarray:
    """Decode one degree-authored seed through the existing KUKA pose decoder."""

    return robot_flange_to_template_base(seed_pose_radians(frame))


def relative_result_transform_matrix(
    center_frame: Mapping[str, Any], result_offset: Mapping[str, Any]
) -> np.ndarray:
    """Decode one documented relative result using the taught center pose."""

    center_seed = center_frame.get("seed")
    if not isinstance(center_seed, Mapping):
        raise ValueError("CalibrationCenter requires a numeric seed")
    _finite_values(result_offset, ORIENTATION_KEYS, "Relative result offset")
    pose = {
        "X": float(center_seed["X"]),
        "Y": float(center_seed["Y"]),
        "Z": float(center_seed["Z"]),
        "A": math.radians(float(center_seed["A"]) + float(result_offset["A"])),
        "B": math.radians(float(center_seed["B"]) + float(result_offset["B"])),
        "C": math.radians(float(center_seed["C"]) + float(result_offset["C"])),
    }
    return robot_flange_to_template_base(pose)
