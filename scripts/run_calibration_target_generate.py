#!/usr/bin/env python3
"""Generate one immutable calibration-target bundle from a queued request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.calibration.target_library import generate_target_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, help="Generation request envelope JSON")
    parser.add_argument("--library-root")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request_path = Path(args.request)
    with open(request_path, "r") as handle:
        request = json.load(handle)
    if not isinstance(request, dict):
        raise ValueError("Generation request must be a JSON object")
    bundle = generate_target_bundle(
        display_name=request.get("display_name", ""),
        configuration=request.get("configuration", {}),
        target_id=request.get("target_id"),
        library_root=args.library_root,
    )
    print(json.dumps({"target_id": bundle["target_id"], "bundle_path": bundle["bundle_path"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
