#!/usr/bin/env python3
"""Print external runtime readiness for PoseTestBot."""

from __future__ import annotations

import argparse
import json

from posetestbot.runtime.status import collect_runtime_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check lightweight readiness for external PoseTestBot runtimes such "
            "as BlenderProc, FoundationPose, MegaPose, SAM6D, BOP Toolkit, "
            "and the ZED SDK."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write the full runtime status snapshot as JSON.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with status 2 when any runtime is unavailable.",
    )
    return parser.parse_args()


def status_label(runtime: dict) -> str:
    return "OK" if runtime["available"] else "MISSING"


def print_status_table(status: dict) -> None:
    rows = []
    for runtime in status["runtimes"]:
        failed_checks = [
            check["name"]
            for check in runtime["checks"]
            if not check["ok"]
        ]
        rows.append(
            [
                runtime["display_name"],
                runtime["category"],
                status_label(runtime),
                ", ".join(failed_checks) if failed_checks else "-",
            ]
        )

    headers = ["Runtime", "Category", "Status", "Missing checks"]
    widths = [
        max(len(str(row[index])) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    print("PoseTestBot runtime status")
    print(f"Generated: {status['generated_at']}")
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main() -> int:
    args = parse_args()
    status = collect_runtime_status()
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print_status_table(status)
    if args.check and not status["all_available"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
