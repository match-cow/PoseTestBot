#!/usr/bin/env python3
"""Run BlenderProc rendering as a manifest-tracked pipeline stage."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from posetestbot.io.artifacts import (
    BLENDERPROC_RENDER_PLAN,
    MASKS_DIR,
    PROCESSED_DIR,
    SYNCHRONIZED_DIR,
)
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    upsert_stage,
    write_run_manifest,
)


@dataclass(frozen=True)
class RenderJob:
    sensor_name: str
    sensor_folder: str
    blenderproc_folder: str
    camera_poses: str
    camera_matrix: str
    command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render prepared BlenderProc scenes for synchronized sensor folders "
            "and record the stage in dataset_manifest.json."
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
        "--render-script",
        default="scripts/blenderproc_render_720p_multi.py",
        help="BlenderProc render script.",
    )
    parser.add_argument(
        "--subdir",
        default="blenderproc",
        help="Prepared BlenderProc subdirectory inside each sensor folder.",
    )
    parser.add_argument(
        "--blenderproc",
        default="blenderproc",
        help="BlenderProc executable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate prepared folders and write a render plan without executing BlenderProc.",
    )
    return parser.parse_args()


def synchronized_input_folder(run_root: Path, explicit_input_folder: str | None) -> Path:
    if explicit_input_folder:
        return Path(explicit_input_folder)
    return run_root / PROCESSED_DIR / SYNCHRONIZED_DIR


def validate_prepared_folder(sensor_folder: Path, subdir: str) -> Path:
    blenderproc_folder = sensor_folder / subdir
    required = [
        blenderproc_folder / "objects.json",
        blenderproc_folder / "camera_matrix.npy",
        blenderproc_folder / "camera_poses.npy",
        blenderproc_folder / "objects",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_list = ", ".join(path.as_posix() for path in missing)
        raise FileNotFoundError(
            f"Prepared BlenderProc folder for {sensor_folder.name} is missing: {missing_list}"
        )
    return blenderproc_folder


def discover_render_jobs(
    *,
    input_folder: Path,
    render_script: Path,
    subdir: str,
    blenderproc_executable: str,
) -> list[RenderJob]:
    if not input_folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    jobs = []
    for sensor_folder in sorted(input_folder.iterdir()):
        if not sensor_folder.is_dir():
            continue
        blenderproc_folder = validate_prepared_folder(sensor_folder, subdir)
        camera_poses = blenderproc_folder / "camera_poses.npy"
        camera_matrix = blenderproc_folder / "camera_matrix.npy"
        command = [
            blenderproc_executable,
            "run",
            render_script.as_posix(),
            camera_poses.as_posix(),
            camera_matrix.as_posix(),
            blenderproc_folder.as_posix(),
        ]
        jobs.append(
            RenderJob(
                sensor_name=sensor_folder.name,
                sensor_folder=sensor_folder.as_posix(),
                blenderproc_folder=blenderproc_folder.as_posix(),
                camera_poses=camera_poses.as_posix(),
                camera_matrix=camera_matrix.as_posix(),
                command=command,
            )
        )

    if not jobs:
        raise FileNotFoundError(f"No prepared BlenderProc sensor folders in {input_folder}")
    return jobs


def cleanup_blenderproc_output(sensor_folder: Path, blenderproc_folder: Path) -> dict[str, Path]:
    bproc_output = blenderproc_folder / "train_pbr" / "000000"
    mask_source_folder = bproc_output / "mask"
    mask_dest_folder = sensor_folder / MASKS_DIR
    bproc_output_new = blenderproc_folder / "output"

    artifacts: dict[str, Path] = {}
    if mask_source_folder.exists():
        if mask_dest_folder.exists():
            shutil.rmtree(mask_dest_folder)
        shutil.move(mask_source_folder.as_posix(), mask_dest_folder.as_posix())
        artifacts[f"{sensor_folder.name}:masks"] = mask_dest_folder

    if bproc_output.exists():
        if bproc_output_new.exists():
            shutil.rmtree(bproc_output_new)
        shutil.move(bproc_output.as_posix(), bproc_output_new.as_posix())
        artifacts[f"{sensor_folder.name}:blenderproc_output"] = bproc_output_new

    train_pbr = blenderproc_folder / "train_pbr"
    if train_pbr.exists() and not any(train_pbr.iterdir()):
        train_pbr.rmdir()

    return artifacts


def write_render_plan(run_root: Path, jobs: list[RenderJob], *, dry_run: bool) -> Path:
    plan_path = run_root / BLENDERPROC_RENDER_PLAN
    with open(plan_path, "w") as f:
        json.dump(
            {
                "schema_version": "blenderproc_render_plan.v1",
                "dry_run": dry_run,
                "jobs": [asdict(job) for job in jobs],
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")
    return plan_path


def run_render_jobs(jobs: list[RenderJob]) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for job in jobs:
        subprocess.run(job.command, check=True)
        artifacts.update(
            cleanup_blenderproc_output(
                Path(job.sensor_folder), Path(job.blenderproc_folder)
            )
        )
    return artifacts


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    input_folder = synchronized_input_folder(run_root, args.input_folder)
    render_script = Path(args.render_script)

    manifest = load_or_create_run_manifest(run_root)
    upsert_stage(manifest, name="blenderproc_render", status="running")
    write_run_manifest(manifest, run_root)

    try:
        jobs = discover_render_jobs(
            input_folder=input_folder,
            render_script=render_script,
            subdir=args.subdir,
            blenderproc_executable=args.blenderproc,
        )
        plan_path = write_render_plan(run_root, jobs, dry_run=args.dry_run)
        artifacts: dict[str, Path] = {BLENDERPROC_RENDER_PLAN: plan_path}

        if args.dry_run:
            message = f"Dry-run render plan created for {len(jobs)} sensor folder(s)."
        else:
            artifacts.update(run_render_jobs(jobs))
            message = f"Rendered BlenderProc outputs for {len(jobs)} sensor folder(s)."

        upsert_stage(
            manifest,
            name="blenderproc_render",
            status="succeeded",
            artifacts=artifacts,
            run_root=run_root,
            message=message,
        )
        write_run_manifest(manifest, run_root)
    except Exception as exc:
        upsert_stage(
            manifest,
            name="blenderproc_render",
            status="failed",
            message=str(exc),
        )
        write_run_manifest(manifest, run_root)
        raise

    print(message)


if __name__ == "__main__":
    main()

