from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from posetestbot.blenderproc.rendering import discover_render_jobs, run_render_jobs
from posetestbot.io.artifacts import (
    BLENDERPROC_RENDER_PLAN,
    DATASET_MANIFEST,
    MASKS_DIR,
)

def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=2)


def create_prepared_render_fixture(tmp_path: Path) -> tuple[Path, Path]:
    run_root = tmp_path / "run-1"
    bproc_folder = (
        run_root
        / "processed"
        / "synchronized"
        / "realsense_123"
        / "blenderproc"
    )
    objects_folder = bproc_folder / "objects"
    objects_folder.mkdir(parents=True)
    write_json(bproc_folder / "objects.json", {"cube": np.eye(4).tolist()})
    np.save(bproc_folder / "camera_matrix.npy", np.eye(3))
    np.save(bproc_folder / "camera_poses.npy", np.eye(4)[None, :, :])
    (objects_folder / "cube.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n"
    )
    np.save(objects_folder / "cube.npy", np.eye(4))
    return run_root, bproc_folder


def test_blenderproc_render_stage_dry_run_writes_plan_and_manifest(
    tmp_path: Path,
) -> None:
    run_root, _ = create_prepared_render_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_blenderproc_render_stage.py"),
            str(run_root),
            "--dry-run",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Dry-run render plan created for 1 sensor folder" in result.stdout

    plan = json.loads((run_root / BLENDERPROC_RENDER_PLAN).read_text())
    assert plan["schema_version"] == "blenderproc_render_plan.v1"
    assert plan["dry_run"] is True
    assert plan["jobs"][0]["sensor_name"] == "realsense_123"
    assert plan["jobs"][0]["command"][:2] == ["blenderproc", "run"]

    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage for stage in manifest["stages"] if stage["name"] == "blenderproc_render"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][BLENDERPROC_RENDER_PLAN] == BLENDERPROC_RENDER_PLAN


def write_fake_render_output(workspace: Path) -> None:
    scene = workspace / "train_pbr" / "000000"
    for folder in ("rgb", "depth", "mask", "mask_visib"):
        (scene / folder).mkdir(parents=True, exist_ok=True)
    (scene / "rgb" / "000000.png").write_bytes(b"rgb")
    (scene / "depth" / "000000.png").write_bytes(b"depth")
    (scene / "mask" / "000000_000000.png").write_bytes(b"mask")
    (scene / "mask_visib" / "000000_000000.png").write_bytes(b"visible")
    write_json(scene / "scene_camera.json", {"0": {}})
    write_json(scene / "scene_gt.json", {"0": [{"obj_id": 1}]})
    write_json(scene / "scene_gt_info.json", {"0": [{}]})


def test_render_jobs_promote_validated_masks_and_output(tmp_path: Path) -> None:
    run_root, bproc_folder = create_prepared_render_fixture(tmp_path)
    sensor_folder = bproc_folder.parent
    (sensor_folder / MASKS_DIR).mkdir()
    (sensor_folder / MASKS_DIR / "old.txt").write_text("old")
    (bproc_folder / "output").mkdir()
    (bproc_folder / "output" / "old.txt").write_text("old")
    jobs = discover_render_jobs(
        input_folder=run_root / "processed" / "synchronized",
        render_script=Path(__file__).resolve().parents[1]
        / "scripts"
        / "blenderproc_render_720p_multi.py",
        subdir="blenderproc",
        blenderproc_executable="blenderproc",
    )

    def fake_runner(command: list[str], *, check: bool) -> None:
        assert check is True
        write_fake_render_output(Path(command[-1]))

    artifacts = run_render_jobs(jobs, command_runner=fake_runner)

    assert (sensor_folder / MASKS_DIR / "000000_000000.png").read_bytes() == b"mask"
    assert not (sensor_folder / MASKS_DIR / "old.txt").exists()
    assert (bproc_folder / "output" / "scene_gt.json").is_file()
    assert not (bproc_folder / "output" / "old.txt").exists()
    assert artifacts["realsense_123:masks"] == sensor_folder / MASKS_DIR
    assert artifacts["realsense_123:blenderproc_output"] == bproc_folder / "output"


def test_render_failure_preserves_every_previous_sensor_output(tmp_path: Path) -> None:
    run_root, first_prepared = create_prepared_render_fixture(tmp_path)
    synchronized = run_root / "processed" / "synchronized"
    second_prepared = synchronized / "zed_2i_456" / "blenderproc"
    shutil.copytree(first_prepared, second_prepared)
    for prepared in (first_prepared, second_prepared):
        sensor = prepared.parent
        (sensor / MASKS_DIR).mkdir()
        (sensor / MASKS_DIR / "previous.txt").write_text(sensor.name)
        (prepared / "output").mkdir()
        (prepared / "output" / "previous.txt").write_text(sensor.name)
    jobs = discover_render_jobs(
        input_folder=synchronized,
        render_script=Path(__file__).resolve().parents[1]
        / "scripts"
        / "blenderproc_render_720p_multi.py",
        subdir="blenderproc",
        blenderproc_executable="blenderproc",
    )
    calls = 0

    def failing_runner(command: list[str], *, check: bool) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise subprocess.CalledProcessError(1, command)
        write_fake_render_output(Path(command[-1]))

    with pytest.raises(subprocess.CalledProcessError):
        run_render_jobs(jobs, command_runner=failing_runner)

    for prepared in (first_prepared, second_prepared):
        sensor = prepared.parent
        assert (sensor / MASKS_DIR / "previous.txt").read_text() == sensor.name
        assert (prepared / "output" / "previous.txt").read_text() == sensor.name
    assert not list(synchronized.rglob("*.staging"))
    assert not list(synchronized.rglob("*.work"))
