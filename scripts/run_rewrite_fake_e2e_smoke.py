#!/usr/bin/env python3
"""Run the hardware-free rewrite golden-path smoke test."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_SEQUENCE_OPTIONS = {
    "capture_execution": {
        "timeout_s": 5,
        "startup_wait_s": 0.1,
    },
    "synthetic_rgbd_fixture": {
        "overwrite": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a fake run, execute supervised fake capture, synthesize "
            "RGB-D frames, export BOP, write synthetic BOP results, dry-run BOP "
            "evaluation, export metrics, then write rewrite_gate_report.json."
        )
    )
    parser.add_argument("run_root", help="Run root to create/use for the smoke.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing run root before running the smoke.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=5.0,
        help="Pose receiver timeout for supervised fake capture.",
    )
    parser.add_argument(
        "--startup-wait-s",
        type=float,
        default=0.1,
        help="Wait after fake controller startup before the pose receiver.",
    )
    return parser.parse_args()


def run(command: list[str], *, cwd: Path) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run_root = Path(args.run_root)

    if run_root.exists() and any(run_root.iterdir()):
        if not args.overwrite:
            raise SystemExit(
                f"Run root is not empty: {run_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(run_root)

    sequence_options = dict(DEFAULT_SEQUENCE_OPTIONS)
    sequence_options["capture_execution"] = {
        "timeout_s": args.timeout_s,
        "startup_wait_s": args.startup_wait_s,
    }
    options_json = json.dumps(sequence_options, sort_keys=True)

    base = ["uv", "run", "python"]
    commands = [
        [
            *base,
            "scripts/create_run_config.py",
            run_root.as_posix(),
            "--sensor",
            "realsense:synthetic:static:Synthetic",
            "--sequence",
            "fake_capture_to_bop_eval_dry_run",
            "--sequence-options-json",
            options_json,
        ],
        [
            *base,
            "scripts/run_preflight.py",
            run_root.as_posix(),
            "--write",
            "--no-sensors",
            "--no-runtimes",
        ],
        [
            *base,
            "scripts/run_capture_execution_stage.py",
            run_root.as_posix(),
            "--mode",
            "pose_only_fake",
            "--timeout-s",
            str(args.timeout_s),
            "--startup-wait",
            str(args.startup_wait_s),
        ],
        [
            *base,
            "scripts/create_synthetic_rgbd_fixture.py",
            run_root.as_posix(),
            "--overwrite",
        ],
        [*base, "scripts/sync_run_non_destructive.py", run_root.as_posix()],
        [*base, "scripts/run_sync_quality.py", run_root.as_posix()],
        [
            *base,
            "scripts/run_bop_export_stage.py",
            run_root.as_posix(),
            "--overwrite",
        ],
        [*base, "scripts/create_synthetic_bop_results.py", run_root.as_posix()],
        [
            *base,
            "scripts/run_bop_evaluation_stage.py",
            run_root.as_posix(),
            "--result-file",
            (run_root / "results" / "bop" / "synthetic_bop-test.csv").as_posix(),
            "--dry-run",
        ],
        [*base, "scripts/run_metric_report_export_stage.py", run_root.as_posix()],
        [
            *base,
            "scripts/run_rewrite_gate.py",
            run_root.as_posix(),
            "--write",
        ],
    ]

    for command in commands:
        run(command, cwd=repo_root)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
