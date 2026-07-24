#!/usr/bin/env python3
"""Remove the retired assets for one globally deleted pose-template bundle."""

from __future__ import annotations

import argparse
import json

from posetestbot.pose_templates.library import delete_template_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-uuid", required=True)
    args = parser.parse_args()
    result = delete_template_bundle(args.template_uuid)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "deleted":
        detail = result["asset_cleanup"].get("last_error") or "unknown cleanup error"
        raise RuntimeError(f"Pose-template asset cleanup remains pending: {detail}")


if __name__ == "__main__":
    main()
