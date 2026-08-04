"""Fail-closed guards for pre-attempt calibration compatibility stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from posetestbot.calibration.target_library import validate_run_target_selection


def observation_mounting_modes(report: Mapping[str, Any]) -> set[str]:
    modes = {
        str(sensor.get("mounting_mode") or "")
        for sensor in report.get("sensors", [])
        if isinstance(sensor, Mapping)
    }
    if "" in modes:
        modes.remove("")
    for observation in report.get("observations", []):
        if not isinstance(observation, Mapping):
            continue
        mode = str(observation.get("mounting_mode") or "")
        if mode:
            modes.add(mode)
    return modes


def require_legacy_static_known_target(
    run_root: str | Path,
    observation_report: Mapping[str, Any],
    *,
    target_to_reference: Mapping[str, Any] | None,
    stage_label: str,
) -> None:
    """Prevent fixed-target legacy math from consuming a robot-carried grid.

    Eye-in-hand compatibility remains unchanged. Static compatibility is kept
    only for an expert-provided known target transform and never falls back to
    the historical implicit transform. A current run-owned target selection is
    authoritative: robot-flange/unknown means the guided joint solver is the
    only valid path.
    """

    if "static" not in observation_mounting_modes(observation_report):
        return

    try:
        selection = validate_run_target_selection(
            run_root,
            require_mounting_frame=True,
        )
    except FileNotFoundError:
        selection = None

    if selection is not None and (
        selection.get("placement_mode") == "unknown"
        or selection.get("effective_mounting_frame") != "template_base"
    ):
        raise ValueError(
            f"{stage_label} assumes a fixed known grid-to-PoseTemplateBase "
            "transform, but this run records a robot-carried grid with unknown "
            "grid-to-flange attachment. Use Workflow step 5 for the static-camera "
            "joint solve."
        )
    if target_to_reference is None:
        raise ValueError(
            f"{stage_label} cannot use its historical implicit target transform "
            "for a static camera. Supply an explicitly measured fixed "
            "grid-to-PoseTemplateBase transform, or use Workflow step 5 for a "
            "robot-carried grid."
        )
