#!/usr/bin/env python3
"""Plan or run FoundationPose as a manifest-tracked pipeline stage."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from posetestbot.io.artifacts import (
    FOUNDATIONPOSE_PLAN,
    PROCESSED_DIR,
    SYNCHRONIZED_DIR,
)
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    upsert_stage,
    write_run_manifest,
)


SCHEMA_VERSION = "foundationpose_plan.v1"


@dataclass(frozen=True)
class FoundationPoseJob:
    sensor_name: str
    sensor_folder: str
    object_name: str
    object_id: int
    expected_output_folder: str


@dataclass(frozen=True)
class FoundationPosePlan:
    schema_version: str
    dry_run: bool
    input_folder: str
    foundationpose_folder: str
    no_tracking: bool
    est_refine_iter: int
    track_refine_iter: int
    object_id: int
    command: list[str]
    jobs: list[FoundationPoseJob]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["jobs"] = [asdict(job) for job in self.jobs]
        return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or run FoundationPose on synchronized sensor folders and "
            "record the stage in dataset_manifest.json."
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
        "--foundationpose-folder",
        default=str(Path.home() / "FoundationPose"),
        help="FoundationPose checkout mounted/used by the legacy Docker wrapper.",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Use FoundationPose no-tracking mode.",
    )
    parser.add_argument("--est-refine-iter", type=int, default=5)
    parser.add_argument("--track-refine-iter", type=int, default=2)
    parser.add_argument("--object-id", type=int, default=0)
    parser.add_argument(
        "--run-level",
        action="store_true",
        help="Pass --run_level to the legacy wrapper.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and write a plan without starting Docker.",
    )
    return parser.parse_args()


def synchronized_input_folder(run_root: Path, explicit_input_folder: str | None) -> Path:
    if explicit_input_folder:
        return Path(explicit_input_folder)
    return run_root / PROCESSED_DIR / SYNCHRONIZED_DIR


def _object_name_for_sensor(sensor_folder: Path, object_id: int) -> str:
    objects_json = sensor_folder / "blenderproc" / "objects.json"
    if not objects_json.is_file():
        raise FileNotFoundError(f"Missing BlenderProc objects file: {objects_json}")
    with open(objects_json, "r") as f:
        objects = json.load(f)
    if not isinstance(objects, dict) or not objects:
        raise ValueError(f"BlenderProc objects file must be a non-empty object: {objects_json}")
    object_names = list(objects.keys())
    try:
        return object_names[object_id]
    except IndexError as exc:
        raise ValueError(
            f"Object ID {object_id} is not present in {objects_json}; "
            f"available objects: {', '.join(object_names)}"
        ) from exc


def expected_output_name(
    *,
    no_tracking: bool,
    est_refine_iter: int,
    track_refine_iter: int,
    object_id: int,
) -> str:
    method = "foundationposeNoTracking" if no_tracking else "foundationpose"
    return f"{method}_est{est_refine_iter}_track{track_refine_iter}_obj{object_id}_output"


def discover_foundationpose_jobs(
    *,
    input_folder: Path,
    no_tracking: bool,
    est_refine_iter: int,
    track_refine_iter: int,
    object_id: int,
) -> list[FoundationPoseJob]:
    if not input_folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    output_name = expected_output_name(
        no_tracking=no_tracking,
        est_refine_iter=est_refine_iter,
        track_refine_iter=track_refine_iter,
        object_id=object_id,
    )
    jobs = []
    for sensor_folder in sorted(input_folder.iterdir()):
        if not sensor_folder.is_dir():
            continue
        object_name = _object_name_for_sensor(sensor_folder, object_id)
        jobs.append(
            FoundationPoseJob(
                sensor_name=sensor_folder.name,
                sensor_folder=sensor_folder.as_posix(),
                object_name=object_name,
                object_id=object_id,
                expected_output_folder=(sensor_folder / output_name).as_posix(),
            )
        )
    if not jobs:
        raise FileNotFoundError(f"No synchronized sensor folders in {input_folder}")
    return jobs


def build_foundationpose_command(
    *,
    input_folder: Path,
    foundationpose_folder: Path,
    no_tracking: bool,
    est_refine_iter: int,
    track_refine_iter: int,
    object_id: int,
    run_level: bool,
) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "scripts/foundationpose_wrapper_multi.py",
        input_folder.as_posix(),
        f"--foundationpose_folder={foundationpose_folder.as_posix()}",
        f"--no_tracking={'y' if no_tracking else 'n'}",
        f"--est_refine_iter={est_refine_iter}",
        f"--track_refine_iter={track_refine_iter}",
        f"--object_id={object_id}",
    ]
    if run_level:
        command.append("--run_level")
    return command


def build_foundationpose_plan(
    *,
    input_folder: Path,
    foundationpose_folder: Path,
    no_tracking: bool,
    est_refine_iter: int,
    track_refine_iter: int,
    object_id: int,
    run_level: bool,
    dry_run: bool,
) -> FoundationPosePlan:
    jobs = discover_foundationpose_jobs(
        input_folder=input_folder,
        no_tracking=no_tracking,
        est_refine_iter=est_refine_iter,
        track_refine_iter=track_refine_iter,
        object_id=object_id,
    )
    command = build_foundationpose_command(
        input_folder=input_folder,
        foundationpose_folder=foundationpose_folder,
        no_tracking=no_tracking,
        est_refine_iter=est_refine_iter,
        track_refine_iter=track_refine_iter,
        object_id=object_id,
        run_level=run_level,
    )
    return FoundationPosePlan(
        schema_version=SCHEMA_VERSION,
        dry_run=dry_run,
        input_folder=input_folder.as_posix(),
        foundationpose_folder=foundationpose_folder.as_posix(),
        no_tracking=no_tracking,
        est_refine_iter=est_refine_iter,
        track_refine_iter=track_refine_iter,
        object_id=object_id,
        command=command,
        jobs=jobs,
    )


def write_foundationpose_plan(run_root: Path, plan: FoundationPosePlan) -> Path:
    path = run_root / FOUNDATIONPOSE_PLAN
    with open(path, "w") as f:
        json.dump(plan.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def existing_output_artifacts(plan: FoundationPosePlan) -> dict[str, Path]:
    artifacts = {}
    for job in plan.jobs:
        output_folder = Path(job.expected_output_folder)
        if output_folder.exists():
            artifacts[f"{job.sensor_name}:foundationpose_output"] = output_folder
    return artifacts


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    input_folder = synchronized_input_folder(run_root, args.input_folder)
    foundationpose_folder = Path(args.foundationpose_folder)

    manifest = load_or_create_run_manifest(run_root)
    upsert_stage(manifest, name="foundationpose", status="running")
    write_run_manifest(manifest, run_root)

    try:
        plan = build_foundationpose_plan(
            input_folder=input_folder,
            foundationpose_folder=foundationpose_folder,
            no_tracking=args.no_tracking,
            est_refine_iter=args.est_refine_iter,
            track_refine_iter=args.track_refine_iter,
            object_id=args.object_id,
            run_level=args.run_level,
            dry_run=args.dry_run,
        )
        plan_path = write_foundationpose_plan(run_root, plan)
        artifacts: dict[str, Path] = {FOUNDATIONPOSE_PLAN: plan_path}

        if args.dry_run:
            message = f"Dry-run FoundationPose plan created for {len(plan.jobs)} sensor folder(s)."
        else:
            subprocess.run(plan.command, check=True, cwd=Path(__file__).resolve().parents[1])
            artifacts.update(existing_output_artifacts(plan))
            message = f"FoundationPose completed for {len(plan.jobs)} sensor folder(s)."

        upsert_stage(
            manifest,
            name="foundationpose",
            status="succeeded",
            artifacts=artifacts,
            run_root=run_root,
            message=message,
        )
        write_run_manifest(manifest, run_root)
    except Exception as exc:
        upsert_stage(
            manifest,
            name="foundationpose",
            status="failed",
            message=str(exc),
        )
        write_run_manifest(manifest, run_root)
        raise

    print(message)


if __name__ == "__main__":
    main()
