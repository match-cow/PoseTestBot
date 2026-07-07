#!/usr/bin/env python3
"""Validate or migrate PoseTestBot calibration profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.calibration.profiles import (
    load_profile_collection,
    migrate_legacy_camera_ee_profiles,
    profile_from_dict,
    write_profile_collection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate calibration.v1 profile files, or migrate legacy "
            "camera_ee_transform.json data into calibration profiles."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Calibration profile or profile collection JSON to validate.",
    )
    parser.add_argument(
        "--legacy-camera-ee",
        help="Legacy camera_ee_transform.json to migrate.",
    )
    parser.add_argument(
        "--legacy-sync-data",
        help="Optional legacy sync_data.json containing sync deltas in milliseconds.",
    )
    parser.add_argument(
        "--output",
        help="Write migrated calibration profile collection JSON to this path.",
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
    if args.legacy_camera_ee:
        camera_ee_transform = load_json(args.legacy_camera_ee)
        sync_data = load_json(args.legacy_sync_data) if args.legacy_sync_data else {}
        if not isinstance(camera_ee_transform, dict):
            raise ValueError("legacy camera_ee_transform root must be a JSON object")
        if not isinstance(sync_data, dict):
            raise ValueError("legacy sync_data root must be a JSON object")
        profiles = migrate_legacy_camera_ee_profiles(
            camera_ee_transform,
            sync_deltas_ms={key: float(value) for key, value in sync_data.items()},
        )
        if args.output:
            write_profile_collection(profiles, args.output)
        profile_ids = [profile.profile_id for profile in profiles]
    else:
        if not args.path:
            raise SystemExit("path is required unless --legacy-camera-ee is supplied")
        profile_ids = validate_path(args.path)

    summary = {"status": "valid", "profile_count": len(profile_ids), "profiles": profile_ids}
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Valid calibration profiles: {len(profile_ids)}")
        for profile_id in profile_ids:
            print(f"- {profile_id}")
        if args.output:
            print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
