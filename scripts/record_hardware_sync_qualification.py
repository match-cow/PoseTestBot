#!/usr/bin/env python3
"""Record externally produced physical D435 sync evidence for one run.

The command copies existing evidence files and never opens cameras, contacts
the robot, or attempts to perform the qualification itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.sensors.hardware_sync_qualification import (
    SUPPORTED_METHODS,
    record_hardware_sync_qualification,
    validate_hardware_sync_qualification,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record an operator-confirmed physical RealSense inter-camera "
            "depth-sync qualification. This copies supplied evidence only and "
            "does not access hardware."
        )
    )
    parser.add_argument("run_root", help="Configured run root.")
    parser.add_argument(
        "--operator",
        required=True,
        help="Operator identity recorded in the qualification.",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=sorted(SUPPORTED_METHODS),
        help="Physical exposure-timing qualification method.",
    )
    parser.add_argument(
        "--observed-max-depth-timestamp-skew-ms",
        type=float,
        required=True,
        help=(
            "Maximum earliest-to-latest depth-exposure timestamp span across "
            "the complete camera group observed by the supplied physical "
            "qualification evidence."
        ),
    )
    parser.add_argument(
        "--evidence",
        action="append",
        required=True,
        help=(
            "Physical qualification evidence file to copy into the run. "
            "Repeat for multiple files."
        ),
    )
    parser.add_argument(
        "--confirm-passed",
        action="store_true",
        required=True,
        help=(
            "Explicitly confirm that the supplied physical evidence passed "
            "the configured depth-exposure synchronization threshold."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    record_hardware_sync_qualification(
        run_root,
        operator=args.operator,
        method=args.method,
        observed_max_depth_timestamp_skew_ms=(
            args.observed_max_depth_timestamp_skew_ms
        ),
        evidence_paths=args.evidence,
        confirm_passed=args.confirm_passed,
    )
    provenance = validate_hardware_sync_qualification(run_root)
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
