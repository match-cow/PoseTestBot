#!/usr/bin/env python3
"""Create synthetic BOP19 result CSVs from a BOP export manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.evaluation.synthetic_bop_results import (
    DEFAULT_METHOD,
    DEFAULT_SCORE,
    DEFAULT_TIME,
    write_synthetic_bop_results_with_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write deterministic synthetic BOP19 result rows for exported BOP "
            "frames. This is fixture evidence, not estimator execution."
        )
    )
    parser.add_argument("run_root", help="Run root containing bop/bop_export_manifest.json.")
    parser.add_argument("--bop-root", default=None)
    parser.add_argument("--output-folder", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--object-name", default=None)
    parser.add_argument("--score", type=float, default=DEFAULT_SCORE)
    parser.add_argument("--time", type=float, default=DEFAULT_TIME)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path, manifest = write_synthetic_bop_results_with_manifest(
        run_root=Path(args.run_root),
        bop_root=args.bop_root,
        output_folder=args.output_folder,
        dataset_name=args.dataset_name,
        method=args.method,
        object_name=args.object_name,
        score=args.score,
        time_s=args.time,
    )
    if args.json:
        print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    else:
        row_count = sum(result.row_count for result in manifest.results)
        print(f"Wrote {row_count} synthetic BOP result row(s).")
        print(path)


if __name__ == "__main__":
    main()
