#!/usr/bin/env python3
"""Print registered RGB-D sensor capture adapter capabilities."""

from __future__ import annotations

import argparse
import json

from posetestbot.sensors.registry import list_sensor_adapters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List PoseTestBot RGB-D sensor adapter capabilities without "
            "opening camera SDKs."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write the full adapter registry as JSON.",
    )
    return parser.parse_args()


def print_adapter_table(adapters: list[dict]) -> None:
    rows = []
    for adapter in adapters:
        rows.append(
            [
                adapter["sensor_type"],
                adapter["display_name"],
                adapter["sdk_module"],
                adapter["capture_script"],
                ", ".join(adapter["supported_resolutions"]),
                adapter["folder_prefix"],
            ]
        )
    headers = ["Type", "Name", "SDK", "Script", "Resolutions", "Folder"]
    widths = [
        max(len(str(row[index])) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    print("PoseTestBot sensor adapters")
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main() -> int:
    adapters = list_sensor_adapters()
    if parse_args().json:
        print(json.dumps({"adapters": adapters}, indent=2, sort_keys=True))
    else:
        print_adapter_table(adapters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
