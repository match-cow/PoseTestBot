from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from posetestbot.io.artifacts import (
    BOP_COCO_ANNOTATIONS,
    BOP_DIR,
    BOP_EXPORT_MANIFEST,
    BOP_FRAME_MAP_JSON,
    BOP_TARGETS_BOP19,
    CAM_K,
    DATASET_MANIFEST,
    DEPTH_DIR,
    DEPTH_SCALE,
    RGB_DIR,
)


def create_synchronized_sensor_fixture(tmp_path: Path) -> Path:
    run_root = tmp_path / "run-1"
    sensor = run_root / "processed" / "synchronized" / "realsense_123"
    rgb = sensor / RGB_DIR
    depth = sensor / DEPTH_DIR
    rgb.mkdir(parents=True)
    depth.mkdir()
    for frame_id, value in ((10, 1), (20, 2)):
        assert cv2.imwrite(
            (rgb / f"{frame_id:06d}.png").as_posix(),
            np.full((5, 6, 3), value, dtype=np.uint8),
        )
        assert cv2.imwrite(
            (depth / f"{frame_id:06d}.png").as_posix(),
            np.full((5, 6), value, dtype=np.uint16),
        )
    (sensor / CAM_K).write_text("1 0 2\n0 3 4\n0 0 1\n")
    (sensor / DEPTH_SCALE).write_text("0.001\n")
    return run_root


def export_command(run_root: Path) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    return [
        sys.executable,
        str(repo_root / "scripts" / "run_bop_export_stage.py"),
        str(run_root),
    ]


def test_bop_export_stage_writes_objectless_dataset_and_manifest(tmp_path: Path) -> None:
    run_root = create_synchronized_sensor_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [*export_command(run_root), "--write-coco-annotations"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Exported 1 synchronized sensor folder" in result.stdout
    bop = run_root / BOP_DIR
    scene = bop / "test" / "000001"
    manifest = json.loads((bop / BOP_EXPORT_MANIFEST).read_text())
    assert manifest["schema_version"] == "bop_export_manifest.v3"
    assert manifest["dataset_mode"] == "objectless"
    assert manifest["objectless"] is True
    assert manifest["object_models"] == []
    assert manifest["stable_id_mapping"] == {}
    assert json.loads((bop / BOP_TARGETS_BOP19).read_text()) == []
    assert (bop / BOP_FRAME_MAP_JSON).is_file()
    assert len(list((scene / RGB_DIR).glob("*.png"))) == 2
    assert len(list((scene / DEPTH_DIR).glob("*.png"))) == 2
    assert all(
        rows == []
        for rows in json.loads((scene / "scene_gt.json").read_text()).values()
    )
    coco = json.loads((bop / BOP_COCO_ANNOTATIONS).read_text())
    assert coco["images"]
    assert coco["categories"] == []
    assert coco["annotations"] == []
    run_manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(item for item in run_manifest["stages"] if item["name"] == "bop_export")
    assert stage["status"] == "succeeded"


def test_bop_export_objectless_rejects_stale_object_gt(tmp_path: Path) -> None:
    run_root = create_synchronized_sensor_fixture(tmp_path)
    output = (
        run_root
        / "processed"
        / "synchronized"
        / "realsense_123"
        / "blenderproc"
        / "output"
    )
    output.mkdir(parents=True)
    (output / "scene_gt.json").write_text(
        json.dumps(
            {
                "0": [
                    {
                        "obj_id": 1,
                        "cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        "cam_t_m2c": [0, 0, 1],
                    }
                ],
                "1": [],
            }
        )
    )
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        export_command(run_root),
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "Unknown BOP obj_id" in result.stderr
    assert not (run_root / BOP_DIR).exists()


def test_bop_overwrite_failure_preserves_previous_dataset(tmp_path: Path) -> None:
    run_root = create_synchronized_sensor_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    command = export_command(run_root)
    subprocess.run(command, cwd=repo_root, check=True, capture_output=True, text=True)
    manifest_path = run_root / BOP_DIR / BOP_EXPORT_MANIFEST
    previous_manifest = manifest_path.read_bytes()

    sensor = run_root / "processed" / "synchronized" / "realsense_123"
    (sensor / DEPTH_DIR / "000020.png").unlink()
    failed = subprocess.run(
        [*command, "--overwrite"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert failed.returncode != 0
    assert manifest_path.read_bytes() == previous_manifest
    assert (run_root / BOP_DIR / "test" / "000001" / RGB_DIR / "000001.png").is_file()
    assert not list(run_root.glob(".bop.*.tmp"))
