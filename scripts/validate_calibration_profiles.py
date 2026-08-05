#!/usr/bin/env python3
"""Validate current PoseTestBot calibration profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.calibration.profiles import (
    load_profile_collection,
    profile_from_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate calibration.v2 profile files."
    )
    parser.add_argument(
        "path",
        help="Calibration profile or profile collection JSON to validate.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable validation summary.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> object:
    with open(path, "r") as f:
        return json.load(f)


def validate_path(path: str | Path) -> list[str]:
    value = load_json(path)
    if isinstance(value, dict) and "profiles" in value:
        profiles = load_profile_collection(path)
    elif isinstance(value, dict):
        profiles = [profile_from_dict(value)]
    else:
        raise ValueError(f"Unsupported calibration JSON root in {path}")
    return [profile.profile_id for profile in profiles]


def main() -> int:
    args = parse_args()
    profile_ids = validate_path(args.path)

    summary = {
        "status": "valid",
        "profile_count": len(profile_ids),
        "profiles": profile_ids,
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Valid calibration profiles: {len(profile_ids)}")
        for profile_id in profile_ids:
            print(f"- {profile_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
