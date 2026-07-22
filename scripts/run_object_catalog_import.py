#!/usr/bin/env python3
"""Inspect and commit one staged managed-catalog upload."""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path

from posetestbot.pose_templates.catalog import (
    default_working_data_root,
    import_catalog_object,
)
from posetestbot.pose_templates.orientations import analyze_catalog_orientations


def _managed_cleanup_folder(request_path: Path) -> Path:
    """Resolve only the one-request directory created by the web upload API."""

    if request_path.is_symlink() or not request_path.is_file():
        raise ValueError("Workpiece import request must be a regular file")
    resolved_request = request_path.resolve()
    managed_root = (
        default_working_data_root()
        / "jobs"
        / "workpiece_catalog_requests"
        / "catalog_upload"
    ).resolve()
    folder = resolved_request.parent
    if resolved_request.name != "request.json" or folder.parent != managed_root:
        raise ValueError(
            "Cleanup is only allowed for managed workpiece upload requests"
        )
    try:
        if uuid.UUID(folder.name).hex != folder.name:
            raise ValueError
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            "Managed workpiece request directory must use a request UUID"
        ) from exc
    return folder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request_path = Path(args.request)
    value = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Workpiece import request must contain a JSON object")
    cleanup_folder = None
    if value.get("cleanup_request_folder") is True:
        cleanup_folder = _managed_cleanup_folder(request_path)
    try:
        result = import_catalog_object(
            name=value["name"],
            alias=value.get("alias"),
            description=value.get("description"),
            tags=value.get("tags"),
            groups=value.get("groups"),
            attributes=value.get("attributes"),
            cad_path=value["cad_path"],
            texture_path=value.get("texture_path"),
            catalog_root=value.get("catalog_root"),
        )
        try:
            analysis = analyze_catalog_orientations(
                result["catalog_uuid"], catalog_root=value.get("catalog_root")
            )
        except Exception as exc:
            # Catalogue import is still valid when a degenerate model has no
            # printable stable pose. Keep the workpiece and expose the derived
            # preview failure in the job log; the operator can correct the CAD
            # or retry analysis without uploading the source again.
            result["orientation_analysis"] = {
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
            }
        else:
            result["orientation_analysis"] = {
                "status": "ready",
                "orientation_count": len(analysis["orientations"]),
                "canonical_ply_sha256": analysis["source"][
                    "canonical_ply_sha256"
                ],
            }
        print(json.dumps(result, sort_keys=True))
    finally:
        if cleanup_folder is not None:
            shutil.rmtree(cleanup_folder, ignore_errors=True)


if __name__ == "__main__":
    main()
