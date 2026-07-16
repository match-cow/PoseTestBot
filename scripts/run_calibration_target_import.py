#!/usr/bin/env python3
"""Resolve a selected target or import ArUcoGridGen/PoseGridGen JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.calibration.target_library import validate_run_target_selection
from posetestbot.calibration.posegridgen import (
    POSEGRIDGEN_REVISION,
    load_posegridgen_backend,
)
from posetestbot.calibration.targets import (
    import_aruco_gridgen_export,
    import_posegridgen_export,
    load_calibration_target_spec,
    write_calibration_target,
)
from posetestbot.io.artifacts import CALIBRATION_TARGET
from posetestbot.io.manifest import load_or_create_run_manifest, upsert_stage, write_run_manifest
from posetestbot.pipeline.run_config import load_run_config_for_run_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument(
        "--source",
        help=(
            "Legacy ArUcoGridGen 1.0 or PoseGridGen 2.0 JSON fallback; "
            "defaults to <run>/aruco_grid_config.json."
        ),
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
        config = load_run_config_for_run_root(run_root)
        selection = config.get("calibration_target")
        if isinstance(selection, dict):
            validate_run_target_selection(run_root)
            path = run_root / CALIBRATION_TARGET
            target = load_calibration_target_spec(path)
        else:
            source_path = Path(args.source) if args.source else run_root / "aruco_grid_config.json"
            with open(source_path, "r") as handle:
                source = json.load(handle)
            if not isinstance(source, dict):
                raise ValueError("Calibration target source must be a JSON object")
            if str(source.get("schema_version")) == "2.0":
                # New imports are accepted only through the exact source checkout.
                # Already-written target specs remain readable without PoseGridGen.
                load_posegridgen_backend()
                target = import_posegridgen_export(source_path)
                target["posegridgen"] = {
                    **dict(target["posegridgen"]),
                    "revision": POSEGRIDGEN_REVISION,
                }
                if args.aligned_to_template_base:
                    target["placement"] = {
                        "from": "aruco_grid",
                        "to": "template_base",
                        "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                        "translation_mm": [0.0, 0.0, 0.0],
                        "source": "operator_declared_aligned_identity",
                    }
            else:
                target = import_aruco_gridgen_export(
                    source_path,
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
