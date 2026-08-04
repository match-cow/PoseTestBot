"""Destination-frame checks for reusable static calibration profiles."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from posetestbot.calibration.profiles import (
    CalibrationProfile,
    require_static_profile_pose_template_base,
)
from posetestbot.robot.reference_frames import (
    POSE_TEMPLATE_BASE_SUNRISE_PATH,
    configured_sunrise_reference_frame_path,
    robot_pose_reference_evidence,
    verified_sunrise_reference_frame_path,
)
from posetestbot.sensors.contracts import MountingMode
from posetestbot.sensors.registry import sensor_folder_name


def _robot_pose_reference_from_file(path: Path) -> dict[str, Any]:
    if os.path.lexists(path) and (path.is_symlink() or not path.is_file()):
        raise ValueError(f"Robot-pose evidence must be a regular file: {path}")
    try:
        value = json.loads(path.read_bytes())
    except FileNotFoundError as exc:
        raise ValueError(f"Robot-pose evidence is missing: {path}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Robot-pose evidence is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"Robot-pose evidence must contain a JSON object: {path}")
    return robot_pose_reference_evidence(value)


def verify_static_profile_destination_reference(
    run_root: str | Path,
    run_config: Mapping[str, Any] | None,
    selected_profiles: Iterable[CalibrationProfile],
    *,
    matched_robot_pose_paths_by_sensor_name: Mapping[str, Path] | None = None,
) -> dict[str, Any] | None:
    """Prove that static and moving profiles share PoseTemplateBase.

    A static profile is already expressed directly in the dataset world.  When
    the same run also uses an eye-in-hand profile, the synchronized flange poses
    are the bridge into that world and must carry verified ``robot_pose.v1``
    provenance for the exact PoseTemplateBase Sunrise frame.
    """

    profiles = list(selected_profiles)
    static_profiles = [
        profile
        for profile in profiles
        if profile.mounting_mode == MountingMode.STATIC
    ]
    if not static_profiles:
        return None
    for profile in static_profiles:
        require_static_profile_pose_template_base(profile)

    if run_config is None:
        raise ValueError(
            "Static calibration reuse requires run_config.json with the exact "
            "PoseTemplateBase robot-pose reference"
        )
    configured_path = configured_sunrise_reference_frame_path(run_config)
    if configured_path != POSE_TEMPLATE_BASE_SUNRISE_PATH:
        actual = configured_path if configured_path is not None else "unconfigured"
        raise ValueError(
            "Static calibration reuse requires "
            "frames.robot_pose.sunrise_reference_frame_path="
            f"{POSE_TEMPLATE_BASE_SUNRISE_PATH!r}; found {actual!r}"
        )

    eye_profiles = [
        profile
        for profile in profiles
        if profile.mounting_mode == MountingMode.EYE_IN_HAND
    ]
    if not eye_profiles:
        return {
            "sunrise_reference_frame_path": configured_path,
            "static_profile_ids": sorted(
                profile.profile_id for profile in static_profiles
            ),
            "eye_in_hand_profile_ids": [],
            "matched_robot_pose_artifacts": [],
        }
    if matched_robot_pose_paths_by_sensor_name is None:
        raise ValueError(
            "Mixed static and eye-in-hand calibration reuse requires synchronized "
            "robot-pose reference evidence for every moving camera"
        )

    verified_artifacts: list[str] = []
    for profile in eye_profiles:
        folder = sensor_folder_name(profile.sensor_type, profile.sensor_id)
        try:
            path = Path(matched_robot_pose_paths_by_sensor_name[folder])
        except KeyError as exc:
            raise ValueError(
                "Mixed calibration reuse has no synchronized robot-pose artifact "
                f"for eye-in-hand camera {folder!r}"
            ) from exc
        evidence = _robot_pose_reference_from_file(path)
        try:
            observed_path = verified_sunrise_reference_frame_path(evidence)
        except ValueError as exc:
            raise ValueError(
                f"Synchronized robot poses for {folder!r} have invalid reference "
                f"evidence: {exc}"
            ) from exc
        if observed_path != POSE_TEMPLATE_BASE_SUNRISE_PATH:
            actual = observed_path if observed_path is not None else "unverified"
            raise ValueError(
                f"Synchronized robot poses for eye-in-hand camera {folder!r} must "
                f"be expressed in {POSE_TEMPLATE_BASE_SUNRISE_PATH!r}; found "
                f"{actual!r}. Re-synchronize from a current robot_pose.v1 capture."
            )
        verified_artifacts.append(path.as_posix())

    return {
        "sunrise_reference_frame_path": configured_path,
        "static_profile_ids": sorted(profile.profile_id for profile in static_profiles),
        "eye_in_hand_profile_ids": sorted(
            profile.profile_id for profile in eye_profiles
        ),
        "matched_robot_pose_artifacts": sorted(verified_artifacts),
    }
