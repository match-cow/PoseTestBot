#!/usr/bin/env python3
"""Commit an immutable pose-template bundle in a local job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.pose_templates.library import generate_template_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    value = json.loads(Path(args.request).read_text())
    result = generate_template_bundle(
        value["configuration"], cloned_from=value.get("cloned_from")
    )
    print(
        json.dumps(
            {
                "template_uuid": result["template_uuid"],
                "display_name": result["display_name"],
                "instance_count": len(result["instances"]),
                "bundle_sha256": result["bundle_sha256"],
                "bundle_path": result["bundle_path"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
