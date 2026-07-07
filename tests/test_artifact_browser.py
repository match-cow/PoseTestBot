from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from posetestbot.io.artifact_browser import (
    ArtifactPathError,
    bop_frame_detail,
    bop_scene_detail,
    collect_run_artifacts,
    preview_artifact,
    render_bop_frame_overlay_png,
    resolve_artifact_path,
)
from posetestbot.io.artifacts import (
    BOP_DIR,
    BOP_EXPORT_MANIFEST,
    BOP_TARGETS_BOP19,
    DATASET_MANIFEST,
    DEPTH_DIR,
    RGB_DIR,
    RUN_CONFIG,
)
from posetestbot.pipeline.run_config import create_run_config, write_run_config


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_png(path: Path, value: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((8, 10, 3), value, dtype=np.uint8)
    assert cv2.imwrite(path.as_posix(), image)


def make_bop_scene(run_root: Path) -> Path:
    scene = run_root / BOP_DIR / "realsense_123" / "test" / "000001"
    write_png(scene / RGB_DIR / "000000.png")
    write_png(scene / DEPTH_DIR / "000000.png", value=10)
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[2:6, 3:8] = 255
    (scene / "mask_visib").mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite((scene / "mask_visib" / "000000_000000.png").as_posix(), mask)
    write_json(
        scene / "scene_camera.json",
        {"0": {"cam_K": [100, 0, 5, 0, 100, 4, 0, 0, 1], "depth_scale": 1.0}},
    )
    write_json(scene / "scene_gt.json", {"0": [{"obj_id": 1, "cam_R_m2c": [], "cam_t_m2c": []}]})
    write_json(scene / "scene_gt_info.json", {"0": [{"bbox_obj": [3, 2, 5, 4], "px_count_visib": 20}]})
    write_json(scene / "posetestbot_bop_frame_map.json", {"0": {"sensor_name": "realsense_123"}})
    write_json(
        run_root / BOP_DIR / BOP_EXPORT_MANIFEST,
        {
            "schema_version": "bop_export_manifest.v1",
            "targets_path": "bop/test_targets_bop19.json",
            "exports": [
                {
                    "sensor_name": "realsense_123",
                    "scene_id": 1,
                    "split": "test",
                    "scene_folder": scene.relative_to(run_root).as_posix(),
                    "artifacts": {
                        "scene_camera": (scene / "scene_camera.json").relative_to(run_root).as_posix(),
                    },
                }
            ],
            "object_models": [
                {"object_name": "cube", "obj_id": 1, "bop_path": "bop/models/obj_000001.ply"}
            ],
        },
    )
    write_json(run_root / BOP_DIR / BOP_TARGETS_BOP19, [{"scene_id": 1, "im_id": 0, "obj_id": 1, "inst_count": 1}])
    write_json(run_root / BOP_DIR / "models" / "models_info.json", {"1": {"diameter": 1}})
    return scene


def test_collect_run_artifacts_lists_acquisition_artifacts(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_run_config(run_root, create_run_config(run_root=run_root))
    make_bop_scene(run_root)
    write_json(run_root / DATASET_MANIFEST, {"schema_version": "run_manifest.v1", "stages": []})

    records = collect_run_artifacts(run_root)
    by_key = {(record.key, record.source): record for record in records}

    assert (RUN_CONFIG, "known") in by_key
    assert (BOP_EXPORT_MANIFEST, "known") in by_key
    assert (BOP_TARGETS_BOP19, "known") in by_key
    assert by_key[(BOP_EXPORT_MANIFEST, "known")].summary["type"] == "bop_export_manifest"
    assert by_key[(BOP_EXPORT_MANIFEST, "known")].summary["export_count"] == 1


def test_preview_artifact_and_path_safety(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_run_config(run_root, create_run_config(run_root=run_root))
    (run_root / "notes.txt").write_text("hello\n")

    preview = preview_artifact(run_root, "notes.txt")
    assert preview["preview_type"] == "text"
    assert preview["text"] == "hello\n"
    assert resolve_artifact_path(run_root, RUN_CONFIG) == (run_root / RUN_CONFIG).resolve()

    with pytest.raises(ArtifactPathError):
        resolve_artifact_path(run_root, "../outside.txt")


def test_bop_scene_and_frame_detail_report_dataset_artifacts(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    scene = make_bop_scene(run_root)

    scene_detail = bop_scene_detail(run_root, scene.relative_to(run_root))
    frame_detail = bop_frame_detail(run_root, scene.relative_to(run_root), image_id=0)

    assert scene_detail["summary"]["image_count"] == 1
    assert scene_detail["frames"][0]["mask_visib_files"] == ["000000_000000.png"]
    assert frame_detail["type"] == "bop_frame_detail"
    assert frame_detail["scene"]["scene_id"] == 1
    assert frame_detail["gt_count"] == 1
    assert frame_detail["result"] is None


def test_render_bop_frame_overlay_png_draws_masks_and_gt(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    scene = make_bop_scene(run_root)

    png = render_bop_frame_overlay_png(run_root, scene.relative_to(run_root), image_id=0)

    assert png.startswith(b"\x89PNG")
    assert len(png) > 50
