#!/usr/bin/env python3
"""Validate and explicitly promote calibration profile candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.calibration.validation import (
    DEFAULT_MAX_MEAN_ROTATION_RESIDUAL_DEG,
    DEFAULT_MAX_MEAN_TRANSLATION_RESIDUAL_MM,
    DEFAULT_MAX_OUTLIER_RATIO,
    DEFAULT_MIN_INLIERS,
    build_calibration_validation,
    write_calibration_validation_with_manifest,
)


def _optional_float(value: str) -> float | None:
    if value.strip().lower() in {"none", "off"}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a number, none, or off") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate calibration candidate profiles and optionally promote "
            "them to calibration_profiles.json."
        )
    )
    parser.add_argument("run_root", help="Run folder containing calibration candidates.")
    parser.add_argument(
        "--candidates",
        help="Path to calibration_candidates.json. Relative paths are run-root relative.",
    )
    parser.add_argument(
        "--profiles",
        help=(
            "Candidate profile collection path. Defaults to "
            "calibration_profiles_from_observations.json when present, otherwise "
            "uses profiles embedded in calibration_candidates.json."
        ),
    )
    parser.add_argument(
        "--min-inliers",
        type=int,
        default=DEFAULT_MIN_INLIERS,
        help="Minimum inlier count required for promotion.",
    )
    parser.add_argument(
        "--max-mean-translation-residual-mm",
        type=_optional_float,
        default=DEFAULT_MAX_MEAN_TRANSLATION_RESIDUAL_MM,
        help="Maximum mean inlier translation residual. Use 'none' or 'off' to disable.",
    )
    parser.add_argument(
        "--max-mean-rotation-residual-deg",
        type=_optional_float,
        default=DEFAULT_MAX_MEAN_ROTATION_RESIDUAL_DEG,
        help="Maximum mean inlier rotation residual. Use 'none' or 'off' to disable.",
    )
    parser.add_argument(
        "--max-outlier-ratio",
        type=_optional_float,
        default=DEFAULT_MAX_OUTLIER_RATIO,
        help="Maximum rejected-candidate ratio. Use 'none' or 'off' to disable.",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Write promoted valid profiles when all validation checks pass.",
    )
    parser.add_argument(
        "--output-profiles",
        help="Output profile collection path for --promote. Defaults to calibration_profiles.json.",
    )
    parser.add_argument(
        "--operator",
        help="Operator name recorded on promoted profiles.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print a report without writing calibration_validation_report.json.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    report_args = {
        "candidates_path": args.candidates,
        "profiles_path": args.profiles,
        "min_inliers": args.min_inliers,
        "max_mean_translation_residual_mm": args.max_mean_translation_residual_mm,
        "max_mean_rotation_residual_deg": args.max_mean_rotation_residual_deg,
        "max_outlier_ratio": args.max_outlier_ratio,
    }
    if args.no_write:
        report = build_calibration_validation(run_root, **report_args)
        report_path = None
        promoted_path = None
    else:
        report_path, promoted_path, report = write_calibration_validation_with_manifest(
            run_root,
            **report_args,
            promote=args.promote,
            output_profiles_path=args.output_profiles,
            operator=args.operator,
        )

    if report_path is not None:
        print(f"Wrote {report_path}")
    if promoted_path is not None:
        print(f"Wrote {promoted_path}")
    print(
        "Calibration validation: "
        f"{report['overall_status']} "
        f"({report['promotable_profile_count']}/{report['profile_count']} "
        "profiles promotable)"
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))

    if report["overall_status"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
