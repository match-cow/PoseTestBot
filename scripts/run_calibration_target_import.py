#!/usr/bin/env python3
"""Import an ArUcoGridGen export into calibration_target.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.calibration.targets import import_aruco_gridgen_export, write_calibration_target
from posetestbot.io.artifacts import CALIBRATION_TARGET
from posetestbot.io.manifest import load_or_create_run_manifest, upsert_stage, write_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument(
        "--source", required=True, help="ArUcoGridGen version 1.0 JSON export."
    )
    parser.add_argument(
        "--aligned-to-template-base",
        action="store_true",
        help="Declare the imported aruco_grid frame identical to template_base.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    manifest = load_or_create_run_manifest(run_root)
    upsert_stage(manifest, name="calibration_target_import", status="running")
    write_run_manifest(manifest, run_root)
    try:
        target = import_aruco_gridgen_export(
            args.source,
            aligned_to_template_base=args.aligned_to_template_base,
        )
        path = write_calibration_target(target, run_root / CALIBRATION_TARGET)
        upsert_stage(
            manifest,
            name="calibration_target_import",
            status="succeeded",
            artifacts={CALIBRATION_TARGET: path},
            run_root=run_root,
        )
        write_run_manifest(manifest, run_root)
    except Exception as exc:
        upsert_stage(manifest, name="calibration_target_import", status="failed", message=str(exc))
        write_run_manifest(manifest, run_root)
        raise
    print(f"Wrote {path}")
    if args.json:
        print(json.dumps(target, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
