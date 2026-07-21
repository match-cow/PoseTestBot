#!/usr/bin/env python3
"""Generate an exact server-side pose-template preview job result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.io.atomic import atomic_write_json
from posetestbot.pose_templates.library import build_template_preview


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    value = json.loads(Path(args.request).read_text())
    result = build_template_preview(value["configuration"])
    output = atomic_write_json(args.output, result)
    print(json.dumps({"output": output.as_posix(), "valid": result["valid"]}))


if __name__ == "__main__":
    main()
