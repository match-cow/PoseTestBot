#!/usr/bin/env python3
"""Audit rewrite milestone gates for a run folder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from posetestbot.pipeline.rewrite_gate import (
    BOP_EXPORT_READINESS_GATE_ID,
    CALIBRATION_VALIDATION_GATE_ID,
    FAKE_E2E_GATE_ID,
    FULL_CAPTURE_GATE_ID,
    build_gate_report,
    format_blocker_detail_lines,
    write_gate_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether a run folder proves a rewrite milestone gate."
    )
    parser.add_argument("run_root", help="Run root to audit.")
    parser.add_argument(
        "--gate",
        default=FAKE_E2E_GATE_ID,
        choices=(
            FAKE_E2E_GATE_ID,
            FULL_CAPTURE_GATE_ID,
            CALIBRATION_VALIDATION_GATE_ID,
            BOP_EXPORT_READINESS_GATE_ID,
        ),
        help="Gate ID to audit.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write rewrite_gate_report.json under the run root.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    if args.write:
        path, report = write_gate_report(run_root, gate_id=args.gate)
    else:
        path = None
        report = build_gate_report(run_root, gate_id=args.gate)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = report["overall_status"]
        summary = report["summary"]
        print(
            f"{report['gate_id']}: {status} "
            f"({summary['ready_count']}/{summary['check_count']} ready)"
        )
        for blocker in report["next_blockers"]:
            print(f"- {blocker['name']}: {blocker['message']}")
            for line in format_blocker_detail_lines(blocker, indent="  "):
                print(line)
        if path is not None:
            print(path)

    if report["overall_status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
