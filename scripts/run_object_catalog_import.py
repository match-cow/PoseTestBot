#!/usr/bin/env python3
"""Inspect and commit one staged managed-catalog upload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.pose_templates.catalog import import_catalog_object


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request_path = Path(args.request)
    value = json.loads(request_path.read_text())
    result = import_catalog_object(
        name=value["name"],
        description=value.get("description"),
        cad_path=value["cad_path"],
        texture_path=value.get("texture_path"),
        catalog_root=value.get("catalog_root"),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
