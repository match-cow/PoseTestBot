"""Transactional BlenderProc render planning and output promotion."""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from posetestbot.blenderproc.preparation import validate_subdir
from posetestbot.io.atomic import atomic_write_json, replace_directories
from posetestbot.io.artifacts import (
    BLENDERPROC_RENDER_PLAN,
    DEPTH_DIR,
    MASKS_DIR,
    RGB_DIR,
)


@dataclass(frozen=True)
class RenderJob:
    sensor_name: str
    sensor_folder: str
    blenderproc_folder: str
    camera_poses: str
    camera_matrix: str
    expected_frame_count: int
    command: list[str]


def validate_prepared_folder(sensor_folder: Path, subdir: str) -> tuple[Path, int]:
    validate_subdir(subdir)
    blenderproc_folder = sensor_folder / subdir
    required_files = [
        blenderproc_folder / "objects.json",
        blenderproc_folder / "camera_matrix.npy",
        blenderproc_folder / "camera_poses.npy",
    ]
    missing = [path for path in required_files if not path.is_file()]
    if not (blenderproc_folder / "objects").is_dir():
        missing.append(blenderproc_folder / "objects")
    if missing:
        raise FileNotFoundError(
            f"Prepared BlenderProc folder for {sensor_folder.name} is missing: "
            + ", ".join(path.as_posix() for path in missing)
        )
    try:
        camera_matrix = np.load(blenderproc_folder / "camera_matrix.npy")
        camera_poses = np.load(blenderproc_folder / "camera_poses.npy")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid prepared arrays in {blenderproc_folder}: {exc}") from exc
    if camera_matrix.shape != (3, 3) or not np.all(np.isfinite(camera_matrix)):
        raise ValueError(f"camera_matrix.npy must be a finite 3x3 array: {blenderproc_folder}")
    if (
        camera_poses.ndim != 3
        or camera_poses.shape[0] < 1
        or camera_poses.shape[1:] != (4, 4)
        or not np.all(np.isfinite(camera_poses))
    ):
        raise ValueError(f"camera_poses.npy must be a non-empty finite Nx4x4 array: {blenderproc_folder}")
    return blenderproc_folder, int(camera_poses.shape[0])


def discover_render_jobs(
    *,
    input_folder: str | Path,
    render_script: str | Path,
    subdir: str,
    blenderproc_executable: str,
) -> list[RenderJob]:
    input_path = Path(input_folder)
    script_path = Path(render_script)
    validate_subdir(subdir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_path}")
    if not script_path.is_file():
        raise FileNotFoundError(f"BlenderProc render script not found: {script_path}")
    if not blenderproc_executable.strip():
        raise ValueError("BlenderProc executable cannot be empty")
    jobs = []
    for sensor_folder in sorted(input_path.iterdir()):
        if not sensor_folder.is_dir() or sensor_folder.name.startswith("."):
            continue
        prepared, frame_count = validate_prepared_folder(sensor_folder, subdir)
        camera_poses = prepared / "camera_poses.npy"
        camera_matrix = prepared / "camera_matrix.npy"
        jobs.append(
            RenderJob(
                sensor_name=sensor_folder.name,
                sensor_folder=sensor_folder.as_posix(),
                blenderproc_folder=prepared.as_posix(),
                camera_poses=camera_poses.as_posix(),
                camera_matrix=camera_matrix.as_posix(),
                expected_frame_count=frame_count,
                command=[
                    blenderproc_executable,
                    "run",
                    script_path.as_posix(),
                    camera_poses.as_posix(),
                    camera_matrix.as_posix(),
                    prepared.as_posix(),
                ],
            )
        )
    if not jobs:
        raise FileNotFoundError(f"No prepared BlenderProc sensor folders in {input_path}")
    return jobs


def write_render_plan(
    run_root: str | Path,
    jobs: list[RenderJob],
    *,
    dry_run: bool,
    skipped: bool = False,
    skip_reason: str | None = None,
) -> Path:
    return atomic_write_json(
        Path(run_root) / BLENDERPROC_RENDER_PLAN,
        {
            "schema_version": "blenderproc_render_plan.v1",
            "dry_run": dry_run,
            "skipped": skipped,
            "skip_reason": skip_reason,
            "jobs": [asdict(job) for job in jobs],
        },
    )


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing BlenderProc render artifact: {path}")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON render artifact {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"Render artifact must be a JSON object: {path}")
    return value


def validate_render_output(output: str | Path, *, expected_frame_count: int) -> None:
    scene = Path(output)
    expected_ids = {f"{index:06d}" for index in range(expected_frame_count)}
    for folder_name in (RGB_DIR, DEPTH_DIR):
        folder = scene / folder_name
        actual = {path.stem for path in folder.glob("*.png")} if folder.is_dir() else set()
        if actual != expected_ids:
            raise ValueError(
                f"BlenderProc {folder_name} frame IDs do not match camera poses: "
                f"expected {sorted(expected_ids)}, got {sorted(actual)}"
            )
    scene_camera = _read_json_mapping(scene / "scene_camera.json")
    scene_gt = _read_json_mapping(scene / "scene_gt.json")
    scene_gt_info = _read_json_mapping(scene / "scene_gt_info.json")
    expected_json_ids = {str(index) for index in range(expected_frame_count)}
    for name, value in (
        ("scene_camera.json", scene_camera),
        ("scene_gt.json", scene_gt),
        ("scene_gt_info.json", scene_gt_info),
    ):
        if set(value) != expected_json_ids:
            raise ValueError(f"{name} keys do not match camera pose count")
    expected_masks: set[str] = set()
    for image_id in range(expected_frame_count):
        annotations = scene_gt[str(image_id)]
        info = scene_gt_info[str(image_id)]
        if not isinstance(annotations, list) or not annotations:
            raise ValueError(f"scene_gt.json frame {image_id} has no object annotations")
        if not isinstance(info, list) or len(info) != len(annotations):
            raise ValueError(f"scene_gt_info.json frame {image_id} does not match scene_gt")
        expected_masks.update(
            f"{image_id:06d}_{index:06d}" for index in range(len(annotations))
        )
    for folder_name in ("mask", "mask_visib"):
        folder = scene / folder_name
        actual = {path.stem for path in folder.glob("*.png")} if folder.is_dir() else set()
        if actual != expected_masks:
            raise ValueError(
                f"BlenderProc {folder_name} files do not match GT annotations: "
                f"expected {sorted(expected_masks)}, got {sorted(actual)}"
            )
    instance_sidecar = scene / "posetestbot_render_instances.json"
    prepared_instances = scene.parent.parent / "objects.json"
    if prepared_instances.is_file():
        prepared = _read_json_mapping(prepared_instances)
        if prepared.get("schema_version") == "blenderproc_object_instances.v1":
            rendered = _read_json_mapping(instance_sidecar)
            if rendered.get("schema_version") != "posetestbot_render_instances.v1":
                raise ValueError("Rendered instance identity sidecar has the wrong schema")
            if rendered.get("instances") != prepared.get("instances"):
                raise ValueError("Rendered instance identity does not match prepared objects")
            if rendered.get("blenderproc_version") != "2.8.0":
                raise ValueError("Rendered pose-template GT was not produced by BlenderProc 2.8.0")
            if rendered.get("identity_contract") != "bop_gt_index_matches_loaded_instance_order.v1":
                raise ValueError("Rendered instance identity contract is missing or unsupported")
            frames = rendered.get("frames")
            if not isinstance(frames, Mapping) or set(frames) != expected_json_ids:
                raise ValueError("Rendered instance identity does not cover every output frame")


def _workspace_command(job: RenderJob, workspace: Path) -> list[str]:
    return [
        *job.command[:3],
        (workspace / "camera_poses.npy").as_posix(),
        (workspace / "camera_matrix.npy").as_posix(),
        workspace.as_posix(),
    ]


def run_render_jobs(
    jobs: Sequence[RenderJob],
    *,
    command_runner: Callable[..., object] = subprocess.run,
) -> dict[str, Path]:
    """Render all sensors in workspaces and atomically promote every result."""

    promotions: list[tuple[Path, Path]] = []
    workspaces: list[Path] = []
    artifacts: dict[str, Path] = {}
    try:
        for job in jobs:
            sensor_folder = Path(job.sensor_folder)
            prepared = Path(job.blenderproc_folder)
            workspace = sensor_folder / f".blenderproc-render.{uuid.uuid4().hex}.work"
            workspaces.append(workspace)
            shutil.copytree(prepared, workspace)
            command_runner(_workspace_command(job, workspace), check=True)
            scene = workspace / "train_pbr" / "000000"
            validate_render_output(scene, expected_frame_count=job.expected_frame_count)

            mask_staging = sensor_folder / f".{MASKS_DIR}.{uuid.uuid4().hex}.staging"
            output_staging = prepared / f".output.{uuid.uuid4().hex}.staging"
            shutil.move((scene / "mask").as_posix(), mask_staging.as_posix())
            shutil.move(scene.as_posix(), output_staging.as_posix())
            mask_destination = sensor_folder / MASKS_DIR
            output_destination = prepared / "output"
            promotions.extend(
                [
                    (mask_staging, mask_destination),
                    (output_staging, output_destination),
                ]
            )
            artifacts[f"{job.sensor_name}:masks"] = mask_destination
            artifacts[f"{job.sensor_name}:blenderproc_output"] = output_destination
        replace_directories(promotions)
    except Exception:
        for staging, _destination in promotions:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        for workspace in workspaces:
            shutil.rmtree(workspace, ignore_errors=True)
    return artifacts
