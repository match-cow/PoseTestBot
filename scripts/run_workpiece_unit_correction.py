#!/usr/bin/env python3
"""Publish one queued, immutable workpiece unit-correction revision."""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path

from posetestbot.pose_templates.catalog import (
    correct_catalog_object_units,
    default_working_data_root,
)


def _managed_cleanup_folder(request_path: Path) -> Path:
    if request_path.is_symlink() or not request_path.is_file():
        raise ValueError("Unit correction request must be a regular file")
    resolved_request = request_path.resolve()
    managed_root = (
        default_working_data_root()
        / "jobs"
        / "workpiece_catalog_requests"
        / "unit_correction"
    ).resolve()
    folder = resolved_request.parent
    if resolved_request.name != "request.json" or folder.parent != managed_root:
        raise ValueError("Cleanup is only allowed for managed unit correction requests")
    try:
        if uuid.UUID(folder.name).hex != folder.name:
            raise ValueError
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            "Managed unit correction request directory must use a request UUID"
        ) from exc
    return folder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request_path = Path(args.request)
    value = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Unit correction request must contain a JSON object")
    cleanup_folder = None
    if value.get("cleanup_request_folder") is True:
        cleanup_folder = _managed_cleanup_folder(request_path)
    try:
        result = correct_catalog_object_units(
            value["catalog_uuid"],
            conversion=value["conversion"],
            confirm=value.get("confirm") is True,
            operator=value.get("operator"),
            expected_geometry_revision=value["expected_geometry_revision"],
            expected_canonical_sha256=value["expected_canonical_sha256"],
            catalog_root=value.get("catalog_root"),
        )
        print(
            json.dumps(
                {
                    "catalog_uuid": result["catalog_uuid"],
                    "obj_id": result["obj_id"],
                    "geometry_revision": result["geometry_revision"],
                    "canonical_ply_sha256": result["canonical_ply_sha256"],
                    "source_to_mm_scale": result["source_to_mm_scale"],
                    "orientation_analysis_cache": result.get(
                        "orientation_analysis_cache"
                    ),
                },
                sort_keys=True,
            )
        )
    finally:
        if cleanup_folder is not None:
            shutil.rmtree(cleanup_folder, ignore_errors=True)


if __name__ == "__main__":
    main()
