#!/usr/bin/env python3
"""Plan or run MegaPose through a manifest-tracked legacy wrapper adapter."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from posetestbot.estimation.legacy_estimators import (
    LegacyEstimatorPlan,
    discover_estimator_jobs,
    existing_output_artifacts,
    synchronized_input_folder,
    wrapper_exists,
    write_legacy_estimator_plan,
)
from posetestbot.io.artifacts import MEGAPOSE_PLAN
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    upsert_stage,
    write_run_manifest,
)


SCHEMA_VERSION = "megapose_plan.v1"
ESTIMATOR_ID = "megapose"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or run MegaPose on synchronized sensor folders and record "
            "the stage in dataset_manifest.json."
        )
    )
    parser.add_argument(
        "run_root",
        help="Run root containing processed/synchronized sensor folders.",
    )
    parser.add_argument(
        "--input-folder",
        default=None,
        help="Folder containing synchronized sensor folders. Defaults to <run_root>/processed/synchronized.",
    )
    parser.add_argument(
        "--wrapper-script",
        default="scripts/megapose_wrapper.py",
        help="MegaPose wrapper script to execute when --dry-run is not set.",
    )
    parser.add_argument("--model", default="megapose-1.0-RGB")
    parser.add_argument("--roi-scale", type=float, default=1.0)
    parser.add_argument("--object-id", type=int, default=0)
    parser.add_argument(
        "--result-id",
        default=None,
        help=(
            "Optional result ID embedded in expected output folders, e.g. "
            "megapose_rgb_obj0_output."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and write a plan without starting MegaPose.",
    )
    return parser.parse_args()


def build_command(
    *,
    input_folder: Path,
    wrapper_script: str,
    model: str,
    roi_scale: float,
    object_id: int,
) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        wrapper_script,
        input_folder.as_posix(),
        f"--model={model}",
        f"--ROI_scale={roi_scale}",
        f"--object_id={object_id}",
    ]


def build_plan(
    *,
    input_folder: Path,
    repo_root: Path,
    wrapper_script: str,
    model: str,
    roi_scale: float,
    object_id: int,
    result_id: str | None,
    dry_run: bool,
) -> LegacyEstimatorPlan:
    jobs = discover_estimator_jobs(
        input_folder=input_folder,
        estimator_id=ESTIMATOR_ID,
        object_id=object_id,
        result_id=result_id,
    )
    command = build_command(
        input_folder=input_folder,
        wrapper_script=wrapper_script,
        model=model,
        roi_scale=roi_scale,
        object_id=object_id,
    )
    return LegacyEstimatorPlan(
        schema_version=SCHEMA_VERSION,
        dry_run=dry_run,
        estimator_id=ESTIMATOR_ID,
        input_folder=input_folder.as_posix(),
        wrapper_script=wrapper_script,
        wrapper_exists=wrapper_exists(wrapper_script, repo_root=repo_root),
        object_id=object_id,
        result_id=result_id,
        command=command,
        options={"model": model, "roi_scale": roi_scale},
        jobs=jobs,
    )


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    input_folder = synchronized_input_folder(run_root, args.input_folder)
    repo_root = Path(__file__).resolve().parents[1]

    manifest = load_or_create_run_manifest(run_root)
    upsert_stage(manifest, name=ESTIMATOR_ID, status="running")
    write_run_manifest(manifest, run_root)

    try:
        plan = build_plan(
            input_folder=input_folder,
            repo_root=repo_root,
            wrapper_script=args.wrapper_script,
            model=args.model,
            roi_scale=args.roi_scale,
            object_id=args.object_id,
            result_id=args.result_id,
            dry_run=args.dry_run,
        )
        plan_path = write_legacy_estimator_plan(run_root, MEGAPOSE_PLAN, plan)
        artifacts: dict[str, Path] = {MEGAPOSE_PLAN: plan_path}

        if args.dry_run:
            message = f"Dry-run MegaPose plan created for {len(plan.jobs)} sensor folder(s)."
        else:
            if not plan.wrapper_exists:
                raise FileNotFoundError(f"MegaPose wrapper script not found: {args.wrapper_script}")
            subprocess.run(plan.command, check=True, cwd=repo_root)
            artifacts.update(
                existing_output_artifacts(plan, artifact_suffix="megapose_output")
            )
            message = f"MegaPose completed for {len(plan.jobs)} sensor folder(s)."

        upsert_stage(
            manifest,
            name=ESTIMATOR_ID,
            status="succeeded",
            artifacts=artifacts,
            run_root=run_root,
            message=message,
        )
        write_run_manifest(manifest, run_root)
    except Exception as exc:
        upsert_stage(
            manifest,
            name=ESTIMATOR_ID,
            status="failed",
            message=str(exc),
        )
        write_run_manifest(manifest, run_root)
        raise

    print(message)


if __name__ == "__main__":
    main()
