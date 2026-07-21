#!/usr/bin/env python3
"""Copy and resolve one immutable pose-template selection for a run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.pose_templates.selection import select_pose_template


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    value = json.loads(Path(args.request).read_text())
    result = select_pose_template(
        value["run_root"],
        value["template_uuid"],
        placement=value["placement"],
        confirmed=bool(value.get("confirmed", False)),
        operator=value["operator"],
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
