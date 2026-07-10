#!/usr/bin/env python3
"""Rectify synchronized RGB/aligned-depth data without modifying source frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.calibration.intrinsics import load_intrinsic_profile_collection
from posetestbot.calibration.rectification import rectify_run
from posetestbot.io.artifacts import CAMERA_RECTIFICATION_REPORT, INTRINSIC_CALIBRATION_PROFILES
from posetestbot.io.manifest import load_or_create_run_manifest, upsert_stage, write_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("--intrinsic-profiles")
    parser.add_argument("--input-root")
    parser.add_argument("--output-root")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    profiles_path = Path(args.intrinsic_profiles) if args.intrinsic_profiles else run_root / INTRINSIC_CALIBRATION_PROFILES
    profiles = load_intrinsic_profile_collection(profiles_path)
    manifest = load_or_create_run_manifest(run_root)
    upsert_stage(manifest, name="camera_rectification", status="running")
    write_run_manifest(manifest, run_root)
    try:
        report_path, report = rectify_run(
            run_root,
            profiles,
            input_root=args.input_root,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )
        upsert_stage(
            manifest,
            name="camera_rectification",
            status="succeeded",
            artifacts={
                CAMERA_RECTIFICATION_REPORT: report_path,
                "rectified": Path(report["output_root"]),
            },
            run_root=run_root,
        )
        write_run_manifest(manifest, run_root)
    except Exception as exc:
        upsert_stage(manifest, name="camera_rectification", status="failed", message=str(exc))
        write_run_manifest(manifest, run_root)
        raise
    print(f"Wrote {report_path}")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
