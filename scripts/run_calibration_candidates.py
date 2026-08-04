#!/usr/bin/env python3
"""Generate validation-gated calibration profile candidates from observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from posetestbot.calibration.candidates import (
    DEFAULT_MAX_ROTATION_RESIDUAL_DEG,
    DEFAULT_MAX_TRANSLATION_RESIDUAL_MM,
    build_calibration_candidates,
    write_calibration_candidates_with_manifest,
)


def _residual_threshold(value: str) -> float | None:
    if value.strip().lower() in {"none", "off"}:
        return None
    try:
        threshold = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("threshold must be a number, none, or off") from exc
    if threshold < 0:
        raise argparse.ArgumentTypeError("threshold must be greater than or equal to 0")
    return threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Average calibration observation transforms into needs-validation "
            "calibration profile candidates."
        )
    )
    parser.add_argument("run_root", help="Run folder containing calibration observations.")
    parser.add_argument(
        "--observations",
        help="Path to calibration_observations.json. Relative paths are run-root relative.",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=6,
        help="Recommended minimum observation count per sensor.",
    )
    parser.add_argument(
        "--target-to-reference",
        help=(
            "Optional JSON transform from calibration_target to robot_base. "
            "Defaults to the legacy ArUco grid/template transform."
        ),
    )
    parser.add_argument(
        "--max-translation-residual-mm",
        type=_residual_threshold,
        default=DEFAULT_MAX_TRANSLATION_RESIDUAL_MM,
        help=(
            "Reject candidate transforms farther than this translation residual "
            "from the refined average. Use 'none' or 'off' to disable."
        ),
    )
    parser.add_argument(
        "--max-rotation-residual-deg",
        type=_residual_threshold,
        default=DEFAULT_MAX_ROTATION_RESIDUAL_DEG,
        help=(
            "Reject candidate transforms farther than this rotation residual "
            "from the refined average. Use 'none' or 'off' to disable."
        ),
    )
    parser.add_argument(
        "--no-residual-thresholds",
        action="store_true",
        help="Disable residual-threshold outlier filtering.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print a report without writing calibration candidate artifacts.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON report.",
    )
    return parser.parse_args()


def _load_target_to_reference(path: str | None, *, run_root: Path) -> dict[str, Any] | None:
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
    max_translation_residual_mm = (
        None if args.no_residual_thresholds else args.max_translation_residual_mm
    )
    max_rotation_residual_deg = (
        None if args.no_residual_thresholds else args.max_rotation_residual_deg
    )
    report_args = {
        "observations_path": args.observations,
        "min_observations": args.min_observations,
        "target_to_reference": target_to_reference,
        "max_translation_residual_mm": max_translation_residual_mm,
        "max_rotation_residual_deg": max_rotation_residual_deg,
    }
    if args.no_write:
        report = build_calibration_candidates(run_root, **report_args)
        report_path = None
        profiles_path = None
    else:
        report_path, profiles_path, report = write_calibration_candidates_with_manifest(
            run_root,
            **report_args,
        )

    if report_path is not None and profiles_path is not None:
        print(f"Wrote {report_path}")
        print(f"Wrote {profiles_path}")
    print(
        "Calibration candidates: "
        f"{report['overall_status']} "
        f"({report['profile_count']} profiles, "
        f"{report['candidate_count']} frame candidates, "
        f"{report['sensor_count']} sensors)"
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))

    if report["overall_status"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
