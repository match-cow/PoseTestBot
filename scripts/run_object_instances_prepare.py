#!/usr/bin/env python3
"""Write object_instances.v1 from a confirmed run selection."""

from __future__ import annotations

import argparse
import json

from posetestbot.pose_templates.selection import prepare_object_instances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    args = parser.parse_args()
    print(json.dumps(prepare_object_instances(args.run_root), sort_keys=True))


if __name__ == "__main__":
    main()
