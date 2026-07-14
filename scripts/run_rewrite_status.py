#!/usr/bin/env python3
"""Summarize all rewrite milestone gates for a run folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.pipeline.rewrite_gate import (
    GATE_IDS,
    build_rewrite_status_report,
    format_blocker_detail_lines,
    write_rewrite_status_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize all rewrite milestone gate statuses for a run."
    )
    parser.add_argument("run_root", help="Run root to audit.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write rewrite_status_report.json under the run root.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON report.",
    )
    parser.add_argument(
        "--gate-run-root",
        action="append",
        default=[],
        metavar="GATE_ID=RUN_ROOT",
        help=(
            "Override the evidence run root for one gate. May be repeated, "
            f"for example {GATE_IDS[0]}=/data/real-run."
        ),
    )
    return parser.parse_args()


def parse_gate_run_roots(values: list[str]) -> dict[str, Path]:
    gate_run_roots: dict[str, Path] = {}
    valid_gates = set(GATE_IDS)
    for value in values:
        gate_id, sep, run_root = value.partition("=")
        if not sep or not gate_id or not run_root:
            raise ValueError(
                "--gate-run-root values must use GATE_ID=RUN_ROOT format"
            )
        if gate_id not in valid_gates:
            raise ValueError(
                f"Unknown gate ID for --gate-run-root: {gate_id}. "
                f"Expected one of: {', '.join(GATE_IDS)}"
            )
        gate_run_roots[gate_id] = Path(run_root)
    return gate_run_roots


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    gate_run_roots = parse_gate_run_roots(args.gate_run_root)
    if args.write:
        path, report = write_rewrite_status_report(
            run_root,
            gate_run_roots=gate_run_roots,
        )
    else:
        path = None
        report = build_rewrite_status_report(
            run_root,
            gate_run_roots=gate_run_roots,
        )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        summary = report["summary"]
        print(
            f"rewrite milestones: {report['overall_status']} "
            f"({summary['ready_gate_count']}/{summary['gate_count']} gates ready, "
            f"{summary['ready_check_count']}/{summary['check_count']} checks ready)"
        )
        for gate in report["gates"]:
            gate_summary = gate["summary"]
            print(
                f"- {gate['gate_id']}: {gate['overall_status']} "
                f"({gate_summary['ready_count']}/{gate_summary['check_count']} ready)"
            )
        next_actions = report.get("next_actions", [])
        next_blockers = report.get("next_blockers", [])
        if isinstance(next_blockers, list) and next_blockers:
            print("next blockers:")
            for blocker in next_blockers[:3]:
                if not isinstance(blocker, dict):
                    continue
                print(f"- {blocker.get('gate_id')}: {blocker.get('name')}")
                message = blocker.get("message")
                if message:
                    print(f"  {message}")
                for line in format_blocker_detail_lines(blocker, indent="  "):
                    print(line)
        if isinstance(next_actions, list) and next_actions:
            actions = [action for action in next_actions if isinstance(action, dict)]
            if len(actions) == 1:
                first_action = actions[0]
                command = first_action.get("command", [])
                print(f"next action: {first_action.get('label', 'Run next action')}")
                if isinstance(command, list):
                    print("  " + " ".join(str(part) for part in command))
            elif actions:
                print("next actions:")
                for index, action in enumerate(actions, start=1):
                    command = action.get("command", [])
                    print(f"{index}. {action.get('label', 'Run next action')}")
                    if isinstance(command, list):
                        print("   " + " ".join(str(part) for part in command))
        if path is not None:
            print(path)

    if report["overall_status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
