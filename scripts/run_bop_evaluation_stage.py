#!/usr/bin/env python3
"""Run or plan BOP Toolkit evaluation as a manifest-tracked stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.evaluation.bop_toolkit import (
    build_bop_evaluation_report,
    build_bop_evaluation_plan,
    run_bop_toolkit_evaluation,
    write_bop_evaluation_report,
)
from posetestbot.io.artifacts import (
    BOP_DIR,
    BOP_EVALUATION_PLAN,
    BOP_EVALUATION_REPORT,
    BOP_TARGETS_BOP19,
    EVALUATION_DIR,
)
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    upsert_stage,
    write_run_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a BOP19 result CSV, build a BOP Toolkit eval_bop19_pose.py "
            "invocation, write plan/report artifacts, and record the evaluation "
            "in dataset_manifest.json."
        )
    )
    parser.add_argument("run_root", help="Run root containing the BOP export.")
    parser.add_argument(
        "--result-file",
        required=True,
        help=(
            "BOP19 result CSV. The filename dataset must match the BOP dataset "
            "folder under BOP_PATH, e.g. foundationpose_bop-test.csv for "
            "<run_root>/bop."
        ),
    )
    parser.add_argument(
        "--bop-root",
        default=None,
        help="BOP dataset folder. Defaults to <run_root>/bop.",
    )
    parser.add_argument(
        "--bop-path",
        default=None,
        help="Folder containing BOP datasets. Defaults to the parent of --bop-root.",
    )
    parser.add_argument(
        "--eval-path",
        default=None,
        help="BOP Toolkit output folder. Defaults to <run_root>/evaluation/bop_toolkit/<result_name>.",
    )
    parser.add_argument(
        "--targets-filename",
        default=BOP_TARGETS_BOP19,
        help="Targets file name inside the BOP dataset folder.",
    )
    parser.add_argument(
        "--bop-toolkit-root",
        default=None,
        help="BOP Toolkit checkout root containing scripts/eval_bop19_pose.py.",
    )
    parser.add_argument(
        "--eval-script",
        default=None,
        help="Explicit BOP Toolkit eval_bop19_pose.py path.",
    )
    parser.add_argument(
        "--python-executable",
        default=None,
        help="Python executable for BOP Toolkit. Defaults to the current uv Python.",
    )
    parser.add_argument(
        "--renderer-type",
        default="vispy",
        help="BOP Toolkit renderer type for VSD calculation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of BOP Toolkit worker processes.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Pass --use_gpu to BOP Toolkit.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="GPU device forwarded to BOP Toolkit when --use-gpu is set.",
    )
    parser.add_argument(
        "--cleanup-eval",
        action="store_true",
        help="Ask BOP Toolkit to delete intermediate error folders after scoring.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the evaluation plan without executing BOP Toolkit.",
    )
    return parser.parse_args()


def write_evaluation_plan(run_root: Path, plan: object) -> Path:
    path = run_root / BOP_EVALUATION_PLAN
    with open(path, "w") as f:
        json.dump(plan.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    bop_root = Path(args.bop_root) if args.bop_root else run_root / BOP_DIR

    manifest = load_or_create_run_manifest(run_root)
    upsert_stage(manifest, name="bop_evaluation", status="running")
    write_run_manifest(manifest, run_root)

    plan = None
    plan_path = None
    try:
        plan = build_bop_evaluation_plan(
            run_root=run_root,
            result_file=args.result_file,
            bop_root=bop_root,
            bop_path=args.bop_path,
            eval_path=args.eval_path,
            targets_filename=args.targets_filename,
            eval_script=args.eval_script,
            bop_toolkit_root=args.bop_toolkit_root,
            python_executable=args.python_executable,
            renderer_type=args.renderer_type,
            num_workers=args.num_workers,
            use_gpu=args.use_gpu,
            device=args.device,
            cleanup_eval=args.cleanup_eval,
            dry_run=args.dry_run,
        )
        plan_path = write_evaluation_plan(run_root, plan)

        if args.dry_run:
            message = (
                "Dry-run BOP Toolkit evaluation plan created for "
                f"{plan.result.filename}."
            )
            report_status = "planned"
        else:
            run_bop_toolkit_evaluation(plan)
            message = f"BOP Toolkit evaluation completed for {plan.result.filename}."
            report_status = "succeeded"

        report = build_bop_evaluation_report(
            run_root=run_root,
            plan=plan,
            plan_path=plan_path,
            status=report_status,
            message=message,
        )
        report_path = write_bop_evaluation_report(run_root, report)
        artifacts = {
            BOP_EVALUATION_PLAN: plan_path,
            BOP_EVALUATION_REPORT: report_path,
            "bop_result_file": Path(args.result_file),
            "bop_eval_path": Path(plan.eval_path),
        }

        upsert_stage(
            manifest,
            name="bop_evaluation",
            status="succeeded",
            artifacts=artifacts,
            run_root=run_root,
            message=message,
        )
        write_run_manifest(manifest, run_root)
    except Exception as exc:
        if plan is not None and plan_path is not None:
            try:
                report = build_bop_evaluation_report(
                    run_root=run_root,
                    plan=plan,
                    plan_path=plan_path,
                    status="failed",
                    message="BOP Toolkit evaluation failed.",
                    error=str(exc),
                )
                write_bop_evaluation_report(run_root, report)
            except Exception:
                pass
        upsert_stage(
            manifest,
            name="bop_evaluation",
            status="failed",
            message=str(exc),
        )
        write_run_manifest(manifest, run_root)
        raise

    print(message)
    if args.dry_run:
        print(" ".join(plan.command))
    else:
        print(f"Evaluation output: {plan.eval_path}")


if __name__ == "__main__":
    main()
