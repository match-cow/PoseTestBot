#!/usr/bin/env python3
"""Export estimator outputs into BOP19 result CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

from posetestbot.evaluation.bop_results import (
    export_aruco_bop_results,
    export_foundationpose_bop_results,
    export_megapose_bop_results,
    export_sam6d_bop_results,
    write_bop_result_export_manifest,
)
from posetestbot.io.artifacts import (
    BOP_DIR,
    BOP_RESULT_EXPORT_MANIFEST,
    PROCESSED_DIR,
    RESULTS_DIR,
    SYNCHRONIZED_DIR,
)
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    upsert_stage,
    write_run_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert estimator outputs into BOP19 result CSVs "
            "and record the result-export stage in dataset_manifest.json."
        )
    )
    parser.add_argument("run_root", help="Run root containing BOP export and estimator outputs.")
    parser.add_argument(
        "--source",
        choices=("foundationpose", "aruco", "megapose", "sam6d"),
        default="foundationpose",
        help="Estimator output source to convert. Defaults to FoundationPose.",
    )
    parser.add_argument(
        "--input-folder",
        default=None,
        help="Folder containing synchronized sensor folders. Defaults to <run_root>/processed/synchronized.",
    )
    parser.add_argument(
        "--foundationpose-output",
        action="append",
        default=None,
        help=(
            "Specific FoundationPose output folder to convert. May be repeated. "
            "When omitted, all foundationpose*_output folders under --input-folder "
            "are discovered."
        ),
    )
    parser.add_argument(
        "--aruco-pose-file",
        action="append",
        default=None,
        help=(
            "Specific aruco_pose_estimation.json file to convert. May be repeated. "
            "When omitted with --source aruco, all synchronized sensor ArUco files "
            "under --input-folder are discovered."
        ),
    )
    parser.add_argument(
        "--megapose-output",
        action="append",
        default=None,
        help=(
            "Specific MegaPose output folder to convert. May be repeated. "
            "When omitted with --source megapose, all megapose*_output folders "
            "under --input-folder are discovered."
        ),
    )
    parser.add_argument(
        "--sam6d-output",
        action="append",
        default=None,
        help=(
            "Specific SAM6D output folder to convert. May be repeated. "
            "When omitted with --source sam6d, all sam6d*_output folders "
            "under --input-folder are discovered."
        ),
    )
    parser.add_argument(
        "--aruco-object-name",
        default="aruco",
        help="BOP object/model name used for ArUco result rows.",
    )
    parser.add_argument(
        "--min-marker-count",
        type=int,
        default=1,
        help="Minimum detected ArUco marker count required to export a frame.",
    )
    parser.add_argument(
        "--bop-root",
        default=None,
        help="BOP dataset folder with bop_export_manifest.json. Defaults to <run_root>/bop.",
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        help="Output folder for BOP result CSVs. Defaults to <run_root>/results/bop.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset name embedded in BOP result filenames. Defaults to the BOP root folder name.",
    )
    parser.add_argument(
        "--default-score",
        type=float,
        default=1.0,
        help="Score written for exported predictions.",
    )
    parser.add_argument(
        "--default-time",
        type=float,
        default=-1.0,
        help="Runtime value written when per-frame time is unavailable.",
    )
    parser.add_argument(
        "--translation-scale-to-mm",
        type=float,
        default=None,
        help=(
            "Scale applied to translation vectors before writing BOP t values. "
            "Defaults to 1000 for FoundationPose/MegaPose and 1 for ArUco/SAM6D."
        ),
    )
    return parser.parse_args()


def default_translation_scale_to_mm(source: str) -> float:
    if source in {"foundationpose", "megapose"}:
        return 1000.0
    return 1.0


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    input_folder = (
        Path(args.input_folder)
        if args.input_folder
        else run_root / PROCESSED_DIR / SYNCHRONIZED_DIR
    )
    bop_root = Path(args.bop_root) if args.bop_root else run_root / BOP_DIR
    output_folder = (
        Path(args.output_folder)
        if args.output_folder
        else run_root / RESULTS_DIR / BOP_DIR
    )

    manifest = load_or_create_run_manifest(run_root)
    upsert_stage(manifest, name="bop_result_export", status="running")
    write_run_manifest(manifest, run_root)

    try:
        translation_scale_to_mm = (
            args.translation_scale_to_mm
            if args.translation_scale_to_mm is not None
            else default_translation_scale_to_mm(args.source)
        )
        if args.source == "aruco":
            export_manifest = export_aruco_bop_results(
                run_root=run_root,
                input_folder=input_folder,
                aruco_pose_files=args.aruco_pose_file,
                bop_root=bop_root,
                output_folder=output_folder,
                dataset_name=args.dataset_name,
                object_name=args.aruco_object_name,
                default_score=args.default_score,
                default_time=args.default_time,
                translation_scale_to_mm=translation_scale_to_mm,
                min_marker_count=args.min_marker_count,
            )
        elif args.source == "foundationpose":
            export_manifest = export_foundationpose_bop_results(
                run_root=run_root,
                input_folder=input_folder,
                foundationpose_outputs=args.foundationpose_output,
                bop_root=bop_root,
                output_folder=output_folder,
                dataset_name=args.dataset_name,
                default_score=args.default_score,
                default_time=args.default_time,
                translation_scale_to_mm=translation_scale_to_mm,
            )
        elif args.source == "megapose":
            export_manifest = export_megapose_bop_results(
                run_root=run_root,
                input_folder=input_folder,
                megapose_outputs=args.megapose_output,
                bop_root=bop_root,
                output_folder=output_folder,
                dataset_name=args.dataset_name,
                default_score=args.default_score,
                default_time=args.default_time,
                translation_scale_to_mm=translation_scale_to_mm,
            )
        else:
            export_manifest = export_sam6d_bop_results(
                run_root=run_root,
                input_folder=input_folder,
                sam6d_outputs=args.sam6d_output,
                bop_root=bop_root,
                output_folder=output_folder,
                dataset_name=args.dataset_name,
                default_score=args.default_score,
                default_time=args.default_time,
                translation_scale_to_mm=translation_scale_to_mm,
            )
        export_manifest_path = write_bop_result_export_manifest(
            run_root, export_manifest
        )
        artifacts = {
            BOP_RESULT_EXPORT_MANIFEST: export_manifest_path,
            RESULTS_DIR: output_folder,
        }
        for result in export_manifest.results:
            artifacts[f"bop_result:{result.filename}"] = Path(result.path)

        message = (
            "Exported "
            f"{len(export_manifest.results)} BOP result file(s) from "
            f"{export_manifest.source_type} outputs."
        )
        upsert_stage(
            manifest,
            name="bop_result_export",
            status="succeeded",
            artifacts=artifacts,
            run_root=run_root,
            message=message,
        )
        write_run_manifest(manifest, run_root)
    except Exception as exc:
        upsert_stage(
            manifest,
            name="bop_result_export",
            status="failed",
            message=str(exc),
        )
        write_run_manifest(manifest, run_root)
        raise

    print(message)
    for result in export_manifest.results:
        print(result.path)


if __name__ == "__main__":
    main()
