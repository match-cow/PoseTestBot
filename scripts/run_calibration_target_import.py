#!/usr/bin/env python3
"""Validate the current run-owned calibration-target selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.calibration.target_library import validate_run_target_selection
from posetestbot.calibration.targets import (
    load_calibration_target_spec,
)
from posetestbot.io.artifacts import CALIBRATION_TARGET
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    upsert_stage,
    write_run_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    manifest = load_or_create_run_manifest(run_root)
    upsert_stage(manifest, name="calibration_target_import", status="running")
    write_run_manifest(manifest, run_root)
    try:
        validate_run_target_selection(run_root)
        path = run_root / CALIBRATION_TARGET
        target = load_calibration_target_spec(path)
        upsert_stage(
            manifest,
            name="calibration_target_import",
            status="succeeded",
            artifacts={CALIBRATION_TARGET: path},
            run_root=run_root,
        )
        write_run_manifest(manifest, run_root)
    except Exception as exc:
        upsert_stage(
            manifest,
            name="calibration_target_import",
            status="failed",
            message=str(exc),
        )
        write_run_manifest(manifest, run_root)
        raise
    print(f"Wrote {path}")
    if args.json:
        print(json.dumps(target, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
