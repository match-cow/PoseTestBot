#!/usr/bin/env python3
"""Plan or execute a dependency-aware PoseTestBot pipeline sequence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from posetestbot.io.artifacts import PIPELINE_SEQUENCE_PLAN
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    upsert_stage,
    write_run_manifest,
)
from posetestbot.pipeline.sequences import (
    PIPELINE_SEQUENCES,
    SEQUENCE_EXECUTION_ACK_ENV,
    build_sequence_plan,
    execute_sequence_plan,
    validate_sequence_execution_options,
    write_sequence_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a dependency-aware pipeline sequence plan and optionally "
            "execute its stage commands in order."
        )
    )
    parser.add_argument("run_root", help="Run root for the pipeline sequence.")
    parser.add_argument(
        "--sequence",
        required=True,
        choices=tuple(sorted(PIPELINE_SEQUENCES)),
        help="Pipeline sequence ID to plan or execute.",
    )
    parser.add_argument(
        "--options-json",
        default=None,
        help=(
            "JSON object keyed by sequence step ID or stage ID with per-stage "
            "option objects."
        ),
    )
    parser.add_argument(
        "--options-file",
        default=None,
        help="Path to a JSON options object. Merged after --options-json.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write the sequence plan and manifest stage without executing stages.",
    )
    return parser.parse_args()


def load_options(*, options_json: str | None, options_file: str | None) -> dict:
    options: dict = {}
    if options_json:
        loaded = json.loads(options_json)
        if not isinstance(loaded, dict):
            raise ValueError("--options-json must decode to a JSON object")
        options.update(loaded)

    if options_file:
        with open(options_file, "r") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError("--options-file must contain a JSON object")
        options.update(loaded)

    return options


def merge_ephemeral_acknowledgements(options: dict) -> dict:
    raw_value = os.environ.pop(SEQUENCE_EXECUTION_ACK_ENV, None)
    if raw_value is None:
        return options
    loaded = json.loads(raw_value)
    if not isinstance(loaded, dict):
        raise ValueError(
            f"{SEQUENCE_EXECUTION_ACK_ENV} must contain a JSON object"
        )
    merged: dict = {}
    for key, value in options.items():
        if not isinstance(value, dict):
            raise ValueError(f"Pipeline sequence options for {key!r} must be an object")
        merged[key] = dict(value)
    for group, values in loaded.items():
        if not isinstance(values, dict) or any(
            key not in {"allow_cameras", "allow_real_robot"}
            or value is not True
            for key, value in values.items()
        ):
            raise ValueError(
                f"{SEQUENCE_EXECUTION_ACK_ENV} contains invalid acknowledgements"
            )
        group_options = dict(merged.get(group, {}))
        group_options.update(values)
        merged[group] = group_options
    return merged


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    options = merge_ephemeral_acknowledgements(
        load_options(
            options_json=args.options_json,
            options_file=args.options_file,
        )
    )

    if not args.plan_only:
        validate_sequence_execution_options(
            sequence_id=args.sequence,
            options=options,
        )

    plan = build_sequence_plan(
        sequence_id=args.sequence,
        run_root=run_root,
        options=options,
        plan_only=args.plan_only,
    )
    plan_path = write_sequence_plan(run_root, plan)
    stage_name = f"pipeline_sequence:{plan.sequence_id}"
    artifacts = {PIPELINE_SEQUENCE_PLAN: plan_path}

    manifest = load_or_create_run_manifest(run_root)
    upsert_stage(
        manifest,
        name=stage_name,
        status="running",
        artifacts=artifacts,
        run_root=run_root,
        message=f"Prepared sequence plan for {len(plan.steps)} step(s).",
    )
    write_run_manifest(manifest, run_root)

    try:
        if args.plan_only:
            message = (
                f"Pipeline sequence {plan.sequence_id} planned "
                f"with {len(plan.steps)} step(s)."
            )
        else:
            execute_sequence_plan(
                plan,
                cwd=Path(__file__).resolve().parents[1],
            )
            # Sequence steps update the shared manifest in their own processes.
            # Reload their committed state before recording sequence completion
            # so this parent does not replace it with the pre-execution snapshot.
            manifest = load_or_create_run_manifest(run_root)
            message = (
                f"Pipeline sequence {plan.sequence_id} completed "
                f"{len(plan.steps)} step(s)."
            )
        upsert_stage(
            manifest,
            name=stage_name,
            status="succeeded",
            artifacts=artifacts,
            run_root=run_root,
            message=message,
        )
        write_run_manifest(manifest, run_root)
    except Exception as exc:
        # A failing child may also have committed partial evidence and its own
        # terminal stage state. Preserve that state before marking the parent.
        manifest = load_or_create_run_manifest(run_root)
        upsert_stage(
            manifest,
            name=stage_name,
            status="failed",
            artifacts=artifacts,
            run_root=run_root,
            message=str(exc),
        )
        write_run_manifest(manifest, run_root)
        raise

    print(message)
    print(plan_path)


if __name__ == "__main__":
    main()
