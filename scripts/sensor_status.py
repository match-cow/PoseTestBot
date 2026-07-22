#!/usr/bin/env python3
"""Print connected RGB-D sensor status."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from collections.abc import Iterator

from posetestbot.sensors.status import collect_sensor_status, parse_expected_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover RealSense D435, OAK-D Pro, and ZED 2i devices and print a "
            "JSON-friendly status snapshot. Expected-count checks run only when "
            "--expected values are provided."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write the full status snapshot as JSON.",
    )
    parser.add_argument(
        "--expected",
        action="append",
        default=[],
        metavar="SENSOR_TYPE=COUNT",
        help=(
            "Request expected-count checks. Valid sensor types: realsense_d435, "
            "oak_d_pro, zed_2i. Use COUNT=none to leave a type unchecked."
        ),
    )
    parser.add_argument(
        "--check-expected",
        action="store_true",
        help="Exit with status 2 when any expected sensor count is not met.",
    )
    return parser.parse_args()


def family_status_label(family: dict) -> str:
    if family["error"]:
        return "ERROR"
    if family["meets_expected"] is True:
        return "OK"
    if family["meets_expected"] is False:
        return "MISSING"
    return "UNCHECKED"


def format_devices(family: dict) -> str:
    if not family["devices"]:
        return "-"
    return ", ".join(device["device_id"] for device in family["devices"])


def print_status_table(status: dict) -> None:
    rows = []
    for family in status["families"]:
        expected = family["expected_count"]
        sdk = "yes" if family["sdk_available"] else "no"
        rows.append(
            [
                family["display_name"],
                sdk,
                str(family["connected_count"]),
                str(family.get("capture_ready_count", family["connected_count"])),
                "-" if expected is None else str(expected),
                family_status_label(family),
                format_devices(family),
            ]
        )

    headers = [
        "Sensor",
        "SDK",
        "Connected",
        "Capture ready",
        "Expected",
        "Status",
        "Devices",
    ]
    widths = [
        max(len(str(row[index])) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    print("PoseTestBot sensor status")
    print(f"Generated: {status['generated_at']}")
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))

    errors = [
        f"{family['display_name']}: {family['error']}"
        for family in status["families"]
        if family["error"]
    ]
    if errors:
        print()
        print("Discovery errors:")
        for error in errors:
            print(f"- {error}")

    diagnostics = [
        (family["display_name"], diagnostic)
        for family in status["families"]
        for diagnostic in family.get("diagnostics", [])
    ]
    if diagnostics:
        print()
        print("Diagnostics:")
        for display_name, diagnostic in diagnostics:
            print(f"- {display_name}: {diagnostic['message']}")
            for hint in diagnostic.get("hints", []):
                print(f"  - {hint}")


@contextlib.contextmanager
def redirect_stdout_to_stderr() -> Iterator[None]:
    """Keep machine-readable stdout clean while vendor SDKs probe hardware."""

    try:
        saved_stdout_fd = os.dup(1)
    except OSError:
        with contextlib.redirect_stdout(sys.stderr):
            yield
        return

    sys.stdout.flush()
    sys.stderr.flush()
    try:
        os.dup2(2, 1)
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, 1)
        os.close(saved_stdout_fd)


def main() -> int:
    args = parse_args()
    try:
        expected_counts = parse_expected_counts(args.expected)
    except ValueError as exc:
        print(f"sensor_status.py: {exc}", file=sys.stderr)
        return 2

    if args.json:
        with redirect_stdout_to_stderr():
            status = collect_sensor_status(expected_counts=expected_counts)
    else:
        status = collect_sensor_status(expected_counts=expected_counts)
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print_status_table(status)

    if args.check_expected and not status["all_expected_connected"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
