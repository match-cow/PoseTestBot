from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from posetestbot.io.artifacts import (
    BLENDERPROC_RENDER_PLAN,
    DATASET_MANIFEST,
    MASKS_DIR,
)


def load_render_stage_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_blenderproc_render_stage.py"
    )
    spec = importlib.util.spec_from_file_location("run_blenderproc_render_stage", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_cleanup_blenderproc_output_moves_masks_and_output(tmp_path: Path) -> None:
    module = load_render_stage_module()
    sensor_folder = tmp_path / "realsense_123"
    bproc_folder = sensor_folder / "blenderproc"
    mask_folder = bproc_folder / "train_pbr" / "000000" / "mask"
    mask_folder.mkdir(parents=True)
    (mask_folder / "000000_000000.png").write_bytes(b"mask")
    (bproc_folder / "train_pbr" / "000000" / "scene_gt.json").write_text("{}")

    artifacts = module.cleanup_blenderproc_output(sensor_folder, bproc_folder)

    assert (sensor_folder / MASKS_DIR / "000000_000000.png").read_bytes() == b"mask"
    assert (bproc_folder / "output" / "scene_gt.json").read_text() == "{}"
    assert not (bproc_folder / "train_pbr").exists()
    assert artifacts["realsense_123:masks"] == sensor_folder / MASKS_DIR
    assert artifacts["realsense_123:blenderproc_output"] == bproc_folder / "output"
