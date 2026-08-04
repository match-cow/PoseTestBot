#!/usr/bin/env python3
"""Analyze one managed workpiece's stable template orientations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.pose_templates.orientations import analyze_catalog_orientations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    value = json.loads(Path(args.request).read_text(encoding="utf-8"))
    result = analyze_catalog_orientations(
        value["catalog_uuid"], catalog_root=value.get("catalog_root")
    )
    print(
        json.dumps(
            {
                "catalog_uuid": result["catalog_uuid"],
                "schema_version": result["schema_version"],
                "orientation_count": len(result["orientations"]),
                "canonical_ply_sha256": result["source"]["canonical_ply_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
