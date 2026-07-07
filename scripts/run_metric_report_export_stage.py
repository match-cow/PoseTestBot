#!/usr/bin/env python3
"""Export PoseTestBot metric summaries as JSON, CSV, and XLSX reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.evaluation.metric_reports import write_metric_reports_with_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write manifest-tracked metric report files from discovered "
            "accuracy_HRC-Hub/all_results artifacts and BOP Toolkit score "
            "summaries."
        )
    )
    parser.add_argument("run_root", help="Run root containing metric artifacts.")
    parser.add_argument(
        "--output-folder",
        default=None,
        help="Output folder. Defaults to <run_root>/results/metrics.",
    )
    parser.add_argument(
        "--group-limit",
        type=int,
        default=200,
        help="Maximum combined all_results groups to include in the dashboard.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the written report artifact paths as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    artifacts = write_metric_reports_with_manifest(
        run_root,
        output_folder=args.output_folder,
        group_limit=args.group_limit,
    )
    payload = {
        "run_root": run_root.as_posix(),
        "json_path": artifacts.json_path.as_posix(),
        "csv_path": artifacts.csv_path.as_posix(),
        "xlsx_path": artifacts.xlsx_path.as_posix(),
        "row_count": artifacts.row_count,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "Wrote metric reports: "
            f"{artifacts.json_path}, {artifacts.csv_path}, {artifacts.xlsx_path}"
        )


if __name__ == "__main__":
    main()
