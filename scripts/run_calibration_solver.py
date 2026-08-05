#!/usr/bin/env python3
"""Solve calibration profiles from calibration observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from posetestbot.calibration.candidates import (
    DEFAULT_MAX_ROTATION_RESIDUAL_DEG,
    DEFAULT_MAX_TRANSLATION_RESIDUAL_MM,
)
from posetestbot.calibration.solver import (
    DEFAULT_COMPARE_HAND_EYE_METHODS,
    DEFAULT_HAND_EYE_METHOD,
    DEFAULT_HOLDOUT_FRACTION,
    HAND_EYE_METHODS,
    build_calibration_solver,
    write_calibration_solver_with_manifest,
)
from posetestbot.calibration.extrinsics import (
    DEFAULT_MAX_CROSS_ROTATION_DEG,
    DEFAULT_MAX_CROSS_TRANSLATION_MM,
    DEFAULT_MAX_MEAN_ROTATION_DEG,
    DEFAULT_MAX_MEAN_TRANSLATION_MM,
    DEFAULT_MAX_OUTLIER_RATIO,
    MODES,
    build_grid_extrinsic_solver,
    write_grid_extrinsic_solver_with_manifest,
)
from posetestbot.calibration.targets import load_calibration_target_spec
from posetestbot.io.artifacts import CALIBRATION_TARGET, RUN_CONFIG


def _residual_threshold(value: str) -> float | None:
    if value.strip().lower() in {"none", "off"}:
        return None
    try:
        threshold = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "threshold must be a number, none, or off"
        ) from exc
    if threshold < 0:
        raise argparse.ArgumentTypeError("threshold must be greater than or equal to 0")
    return threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve needs-validation calibration profiles from ArUco calibration "
            "observations. Eye-in-hand sensors use OpenCV calibrateHandEye; "
            "static sensors use the configured calibration-target-to-reference "
            "transform with residual-threshold filtering."
        )
    )
    parser.add_argument(
        "run_root", help="Run folder containing calibration observations."
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        help=(
            "Use explicit grid calibration mode. Omit only for the legacy solver "
            "compatibility path."
        ),
    )
    parser.add_argument(
        "--calibration-target",
        help="calibration_target.v2 path; defaults to <run_root>/calibration_target.json.",
    )
    parser.add_argument(
        "--max-outlier-ratio",
        type=_residual_threshold,
        default=DEFAULT_MAX_OUTLIER_RATIO,
    )
    parser.add_argument(
        "--max-cross-translation-mm",
        type=_residual_threshold,
        default=DEFAULT_MAX_CROSS_TRANSLATION_MM,
    )
    parser.add_argument(
        "--max-cross-rotation-deg",
        type=_residual_threshold,
        default=DEFAULT_MAX_CROSS_ROTATION_DEG,
    )
    parser.add_argument(
        "--observations",
        help="Path to calibration_observations.json. Relative paths are run-root relative.",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=6,
        help="Recommended minimum inlier observation count per sensor.",
    )
    parser.add_argument(
        "--target-to-reference",
        help=(
            "Legacy JSON transform from calibration_target to robot_base. "
            "Used for static sensors and recorded in the report."
        ),
    )
    parser.add_argument(
        "--hand-eye-method",
        default=DEFAULT_HAND_EYE_METHOD,
        choices=tuple(sorted(HAND_EYE_METHODS)),
        help="OpenCV hand-eye solver method for eye-in-hand sensors.",
    )
    parser.add_argument(
        "--max-translation-residual-mm",
        type=_residual_threshold,
        default=None,
        help=(
            "Reject solved observations farther than this translation residual "
            "from the refined consistency estimate. Use 'none' or 'off' to disable."
        ),
    )
    parser.add_argument(
        "--max-rotation-residual-deg",
        type=_residual_threshold,
        default=None,
        help=(
            "Reject solved observations farther than this rotation residual "
            "from the refined consistency estimate. Use 'none' or 'off' to disable."
        ),
    )
    parser.add_argument(
        "--no-residual-thresholds",
        action="store_true",
        help="Disable residual-threshold outlier filtering.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=DEFAULT_HOLDOUT_FRACTION,
        help=(
            "Fraction of observations to reserve for held-out residual "
            "validation. Defaults to 0.0."
        ),
    )
    parser.add_argument(
        "--compare-hand-eye-methods",
        action="store_true",
        default=DEFAULT_COMPARE_HAND_EYE_METHODS,
        help=(
            "Evaluate all OpenCV hand-eye methods for eye-in-hand sensors and "
            "record comparison residuals in the solver report."
        ),
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print a report without writing calibration solver artifacts.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON report.",
    )
    return parser.parse_args()


def _load_target_to_reference(
    path: str | None, *, run_root: Path
) -> dict[str, Any] | None:
    if path is None:
        return None
    target_path = Path(path)
    if not target_path.is_absolute() and not target_path.exists():
        target_path = run_root / target_path
    with open(target_path, "r") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Target-to-reference JSON must be an object: {target_path}")
    return value


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    target_to_reference = _load_target_to_reference(
        args.target_to_reference,
        run_root=run_root,
    )
    default_translation = (
        DEFAULT_MAX_MEAN_TRANSLATION_MM
        if args.mode
        else DEFAULT_MAX_TRANSLATION_RESIDUAL_MM
    )
    default_rotation = (
        DEFAULT_MAX_MEAN_ROTATION_DEG
        if args.mode
        else DEFAULT_MAX_ROTATION_RESIDUAL_DEG
    )
    max_translation_residual_mm = (
        None
        if args.no_residual_thresholds
        else (
            args.max_translation_residual_mm
            if args.max_translation_residual_mm is not None
            else default_translation
        )
    )
    max_rotation_residual_deg = (
        None
        if args.no_residual_thresholds
        else (
            args.max_rotation_residual_deg
            if args.max_rotation_residual_deg is not None
            else default_rotation
        )
    )
    report_args = {
        "observations_path": args.observations,
        "min_observations": args.min_observations,
        "target_to_reference": target_to_reference,
        "hand_eye_method": args.hand_eye_method,
        "max_translation_residual_mm": max_translation_residual_mm,
        "max_rotation_residual_deg": max_rotation_residual_deg,
        "holdout_fraction": args.holdout_fraction,
        "compare_hand_eye_methods": args.compare_hand_eye_methods,
    }
    if args.mode:
        target_path = (
            Path(args.calibration_target)
            if args.calibration_target
            else run_root / CALIBRATION_TARGET
        )
        if not target_path.is_absolute() and not target_path.exists():
            target_path = run_root / target_path
        target = load_calibration_target_spec(target_path)
        fixed_transforms = []
        run_config_path = run_root / RUN_CONFIG
        if run_config_path.is_file():
            run_config = json.loads(run_config_path.read_text())
            frames = (
                run_config.get("frames", {}) if isinstance(run_config, dict) else {}
            )
            if isinstance(frames, dict) and isinstance(
                frames.get("fixed_transforms"), list
            ):
                fixed_transforms = frames["fixed_transforms"]
        explicit_args = {
            "target": target,
            "mode": args.mode,
            "observations_path": args.observations,
            "hand_eye_method": args.hand_eye_method,
            "min_inliers": args.min_observations,
            "max_mean_translation_mm": max_translation_residual_mm,
            "max_mean_rotation_deg": max_rotation_residual_deg,
            "max_outlier_ratio": args.max_outlier_ratio,
            "max_cross_translation_mm": args.max_cross_translation_mm,
            "max_cross_rotation_deg": args.max_cross_rotation_deg,
            "fixed_transforms": fixed_transforms,
        }
        if args.no_write:
            report = build_grid_extrinsic_solver(run_root, **explicit_args)
            report_path = None
            profiles_path = None
        else:
            report_path, profiles_path, report = (
                write_grid_extrinsic_solver_with_manifest(run_root, **explicit_args)
            )
    elif args.no_write:
        report = build_calibration_solver(run_root, **report_args)
        report_path = None
        profiles_path = None
    else:
        report_path, profiles_path, report = write_calibration_solver_with_manifest(
            run_root,
            **report_args,
        )

    if report_path is not None and profiles_path is not None:
        print(f"Wrote {report_path}")
        print(f"Wrote {profiles_path}")
    print(
        "Calibration solver: "
        f"{report['overall_status']} "
        f"({report['profile_count']} profiles, "
        f"{report['inlier_count']} inliers / "
        f"{report['observation_count']} observations, "
        f"{report['sensor_count']} sensors)"
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))

    if report["overall_status"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
