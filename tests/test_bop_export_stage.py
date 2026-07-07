from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from posetestbot.calibration.profiles import (
    SCHEMA_VERSION,
    CalibrationProfile,
    CalibrationStatus,
    RigidTransform,
    TransformFrame,
    write_profile_collection,
)
from posetestbot.io.artifacts import (
    BOP_COCO_ANNOTATIONS,
    BOP_DIR,
    BOP_EXPORT_MANIFEST,
    BOP_FRAME_MAP_JSON,
    BOP_MULTIVIEW_TARGETS,
    BOP_TARGETS_BOP19,
    CALIBRATION_PROFILES,
    CAM_K,
    DATASET_MANIFEST,
    DEPTH_DIR,
    DEPTH_SCALE,
    MASKS_DIR,
    MODELS_DIR,
    RGB_DIR,
)
from posetestbot.sensors.contracts import CameraIntrinsics, MountingMode, SensorType


def create_synchronized_sensor_fixture(tmp_path: Path) -> Path:
    run_root = tmp_path / "run-1"
    sensor_folder = run_root / "processed" / "synchronized" / "realsense_123"
    rgb_folder = sensor_folder / RGB_DIR
    depth_folder = sensor_folder / DEPTH_DIR
    rgb_folder.mkdir(parents=True)
    depth_folder.mkdir()
    (rgb_folder / "000010.png").write_bytes(b"rgb-10")
    (depth_folder / "000010.png").write_bytes(b"depth-10")
    (rgb_folder / "000020.png").write_bytes(b"rgb-20")
    (depth_folder / "000020.png").write_bytes(b"depth-20")
    (sensor_folder / CAM_K).write_text("1 0 2\n0 3 4\n0 0 1\n")
    (sensor_folder / DEPTH_SCALE).write_text("0.001\n")
    return run_root


def test_bop_export_stage_writes_scene_and_manifest(tmp_path: Path) -> None:
    run_root = create_synchronized_sensor_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_bop_export_stage.py"),
            str(run_root),
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Exported 1 synchronized sensor folder" in result.stdout

    scene_folder = run_root / BOP_DIR / "realsense_123" / "test" / "000001"
    assert (scene_folder / RGB_DIR / "000000.png").read_bytes() == b"rgb-10"
    assert (scene_folder / DEPTH_DIR / "000001.png").read_bytes() == b"depth-20"

    scene_camera = json.loads((scene_folder / "scene_camera.json").read_text())
    assert scene_camera["0"]["cam_K"] == [1, 0, 2, 0, 3, 4, 0, 0, 1]
    assert scene_camera["0"]["depth_scale"] == 0.001

    scene_gt = json.loads((scene_folder / "scene_gt.json").read_text())
    scene_gt_info = json.loads((scene_folder / "scene_gt_info.json").read_text())
    assert scene_gt == {"0": [], "1": []}
    assert scene_gt_info == {"0": [], "1": []}

    frame_map = json.loads((scene_folder / BOP_FRAME_MAP_JSON).read_text())
    assert frame_map["0"]["source_rgb"] == "rgb/000010.png"
    assert frame_map["1"]["bop_depth"] == "depth/000001.png"

    bop_manifest = json.loads(
        (run_root / BOP_DIR / BOP_EXPORT_MANIFEST).read_text()
    )
    assert bop_manifest["schema_version"] == "bop_export_manifest.v1"
    assert bop_manifest["exports"][0]["rgb_count"] == 2

    run_manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(stage for stage in run_manifest["stages"] if stage["name"] == "bop_export")
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][BOP_EXPORT_MANIFEST] == "bop/bop_export_manifest.json"
    assert stage["artifacts"]["realsense_123:bop_scene"] == (
        "bop/realsense_123/test/000001"
    )


def test_bop_export_stage_records_calibration_profile_metadata(
    tmp_path: Path,
) -> None:
    run_root = create_synchronized_sensor_fixture(tmp_path)
    calibration_profiles = run_root / CALIBRATION_PROFILES
    write_profile_collection(
        [
            CalibrationProfile(
                schema_version=SCHEMA_VERSION,
                profile_id="realsense_d435_123_static_front_left_v2026_01",
                sensor_id="123",
                sensor_type=SensorType.REALSENSE_D435,
                mounting_mode=MountingMode.STATIC,
                rig_position="front_left",
                intrinsics=CameraIntrinsics(
                    cam_k=(9.0, 0.0, 8.0, 0.0, 7.0, 6.0, 0.0, 0.0, 1.0),
                    width=80,
                    height=60,
                    depth_scale_to_mm=2.5,
                ),
                extrinsics=RigidTransform(
                    from_frame=TransformFrame.CAMERA,
                    to_frame=TransformFrame.ROBOT_BASE,
                    rotation_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                    translation_mm=(100.0, 200.0, 300.0),
                ),
                status=CalibrationStatus.VALID,
                sync_delta_ms=12.5,
            )
        ],
        calibration_profiles,
    )
    repo_root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_bop_export_stage.py"),
            str(run_root),
            "--calibration-profiles",
            str(calibration_profiles),
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    scene_folder = run_root / BOP_DIR / "realsense_123" / "test" / "000001"
    scene_camera = json.loads((scene_folder / "scene_camera.json").read_text())
    assert scene_camera["0"]["cam_K"] == [9.0, 0.0, 8.0, 0.0, 7.0, 6.0, 0.0, 0.0, 1.0]
    assert scene_camera["0"]["depth_scale"] == 2.5
    calibration = scene_camera["0"]["posetestbot_calibration"]
    assert calibration["calibration_profile_id"] == (
        "realsense_d435_123_static_front_left_v2026_01"
    )
    assert calibration["mounting_mode"] == "static"
    assert calibration["extrinsics"]["to"] == "robot_base"
    assert calibration["sync_delta_ms"] == 12.5

    bop_manifest = json.loads(
        (run_root / BOP_DIR / BOP_EXPORT_MANIFEST).read_text()
    )
    assert bop_manifest["exports"][0]["calibration_profile_id"] == (
        "realsense_d435_123_static_front_left_v2026_01"
    )
    assert bop_manifest["calibration_profiles_path"] == calibration_profiles.as_posix()
    assert bop_manifest["calibration_profiles"][0]["profile_id"] == (
        "realsense_d435_123_static_front_left_v2026_01"
    )

    run_manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(stage for stage in run_manifest["stages"] if stage["name"] == "bop_export")
    assert stage["artifacts"][CALIBRATION_PROFILES] == CALIBRATION_PROFILES


def test_bop_export_stage_imports_blenderproc_gt_and_masks(tmp_path: Path) -> None:
    run_root = create_synchronized_sensor_fixture(tmp_path)
    sensor_folder = run_root / "processed" / "synchronized" / "realsense_123"
    blenderproc_output = sensor_folder / "blenderproc" / "output"
    blenderproc_output.mkdir(parents=True)
    (blenderproc_output / "scene_gt.json").write_text(
        json.dumps(
            {
                "0": [
                    {
                        "obj_id": 1,
                        "cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        "cam_t_m2c": [10, 20, 30],
                    }
                ],
                "1": [],
            }
        )
    )
    (blenderproc_output / "scene_gt_info.json").write_text(
        json.dumps(
            {
                "0": [
                    {
                        "bbox_obj": [1, 2, 3, 4],
                        "px_count_all": 12,
                        "visib_fract": 1.0,
                    }
                ],
                "1": [],
            }
        )
    )
    masks_folder = sensor_folder / MASKS_DIR
    masks_folder.mkdir()
    (masks_folder / "000000_000000.png").write_bytes(b"mask-0")
    repo_root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_bop_export_stage.py"),
            str(run_root),
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    scene_folder = run_root / BOP_DIR / "realsense_123" / "test" / "000001"
    scene_gt = json.loads((scene_folder / "scene_gt.json").read_text())
    scene_gt_info = json.loads((scene_folder / "scene_gt_info.json").read_text())
    assert scene_gt["0"][0]["obj_id"] == 1
    assert scene_gt["0"][0]["cam_t_m2c"] == [10, 20, 30]
    assert scene_gt_info["0"][0]["bbox_obj"] == [1, 2, 3, 4]
    assert (scene_folder / "mask" / "000000_000000.png").read_bytes() == b"mask-0"

    bop_manifest = json.loads(
        (run_root / BOP_DIR / BOP_EXPORT_MANIFEST).read_text()
    )
    artifacts = bop_manifest["exports"][0]["artifacts"]
    assert artifacts["scene_gt"].endswith("scene_gt.json")
    assert artifacts["scene_gt_info"].endswith("scene_gt_info.json")
    assert artifacts["mask"].endswith("/mask")


def test_bop_export_stage_derives_scene_gt_info_from_masks(tmp_path: Path) -> None:
    run_root = create_synchronized_sensor_fixture(tmp_path)
    sensor_folder = run_root / "processed" / "synchronized" / "realsense_123"
    blenderproc_output = sensor_folder / "blenderproc" / "output"
    blenderproc_output.mkdir(parents=True)
    (blenderproc_output / "scene_gt.json").write_text(
        json.dumps(
            {
                "0": [
                    {
                        "obj_id": 1,
                        "cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        "cam_t_m2c": [10, 20, 30],
                    }
                ],
                "1": [],
            }
        )
    )
    masks_folder = sensor_folder / MASKS_DIR
    masks_folder.mkdir()
    mask = np.zeros((5, 6), dtype=np.uint8)
    mask[1:4, 2:5] = 255
    assert cv2.imwrite((masks_folder / "000000_000000.png").as_posix(), mask)
    repo_root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_bop_export_stage.py"),
            str(run_root),
            "--no-model-export",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    scene_folder = run_root / BOP_DIR / "realsense_123" / "test" / "000001"
    scene_gt_info = json.loads((scene_folder / "scene_gt_info.json").read_text())
    assert scene_gt_info["0"][0]["bbox_obj"] == [2, 1, 3, 3]
    assert scene_gt_info["0"][0]["bbox_visib"] == [2, 1, 3, 3]
    assert scene_gt_info["0"][0]["px_count_all"] == 9
    assert scene_gt_info["0"][0]["px_count_valid"] == 9
    assert scene_gt_info["0"][0]["px_count_visib"] == 9
    assert scene_gt_info["0"][0]["visib_fract"] == 1.0
    assert scene_gt_info["1"] == []
    assert (scene_folder / "mask" / "000000_000000.png").is_file()


def test_bop_export_stage_writes_models_and_targets(tmp_path: Path) -> None:
    run_root = create_synchronized_sensor_fixture(tmp_path)
    sensor_folder = run_root / "processed" / "synchronized" / "realsense_123"
    blenderproc_output = sensor_folder / "blenderproc" / "output"
    blenderproc_output.mkdir(parents=True)
    (blenderproc_output / "scene_gt.json").write_text(
        json.dumps(
            {
                "0": [
                    {
                        "obj_id": "cube",
                        "cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        "cam_t_m2c": [10, 20, 30],
                    }
                ],
                "1": [],
            }
        )
    )
    object_folder = tmp_path / "objects"
    object_folder.mkdir()
    (object_folder / "objects.json").write_text(
        json.dumps({"cube": [], "sphere": []})
    )
    (object_folder / "cube.ply").write_text("ply\nformat ascii 1.0\nend_header\n")
    (object_folder / "sphere.ply").write_text("ply\nformat ascii 1.0\nend_header\n")
    repo_root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_bop_export_stage.py"),
            str(run_root),
            "--object-folder",
            str(object_folder),
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    scene_folder = run_root / BOP_DIR / "realsense_123" / "test" / "000001"
    scene_gt = json.loads((scene_folder / "scene_gt.json").read_text())
    assert scene_gt["0"][0]["obj_id"] == 1
    assert scene_gt["0"][0]["posetestbot_object_name"] == "cube"

    models_folder = run_root / BOP_DIR / MODELS_DIR
    assert (models_folder / "obj_000001.ply").read_text().startswith("ply")
    assert (models_folder / "obj_000002.ply").read_text().startswith("ply")
    models_info = json.loads((models_folder / "models_info.json").read_text())
    assert models_info["1"]["source_name"] == "cube"
    assert models_info["2"]["source_name"] == "sphere"

    targets = json.loads((run_root / BOP_DIR / BOP_TARGETS_BOP19).read_text())
    assert targets == [
        {"scene_id": 1, "im_id": 0, "obj_id": 1, "inst_count": 1}
    ]

    bop_manifest = json.loads(
        (run_root / BOP_DIR / BOP_EXPORT_MANIFEST).read_text()
    )
    assert bop_manifest["targets_path"].endswith(BOP_TARGETS_BOP19)
    assert bop_manifest["object_models"][0]["object_name"] == "cube"
    assert bop_manifest["object_models"][0]["obj_id"] == 1

    run_manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(stage for stage in run_manifest["stages"] if stage["name"] == "bop_export")
    assert stage["artifacts"][MODELS_DIR] == "bop/models"
    assert stage["artifacts"][BOP_TARGETS_BOP19] == "bop/test_targets_bop19.json"


def test_bop_export_stage_writes_multiview_targets(tmp_path: Path) -> None:
    run_root = create_synchronized_sensor_fixture(tmp_path)
    first_sensor = run_root / "processed" / "synchronized" / "realsense_123"
    second_sensor = run_root / "processed" / "synchronized" / "zed_2i_42"
    (second_sensor / RGB_DIR).mkdir(parents=True)
    (second_sensor / DEPTH_DIR).mkdir()
    (second_sensor / RGB_DIR / "000010.png").write_bytes(b"zed-rgb-10")
    (second_sensor / DEPTH_DIR / "000010.png").write_bytes(b"zed-depth-10")
    (second_sensor / CAM_K).write_text("1 0 2\n0 3 4\n0 0 1\n")
    (second_sensor / DEPTH_SCALE).write_text("0.001\n")
    for sensor_folder in (first_sensor, second_sensor):
        blenderproc_output = sensor_folder / "blenderproc" / "output"
        blenderproc_output.mkdir(parents=True)
        blenderproc_output.joinpath("scene_gt.json").write_text(
            json.dumps(
                {
                    "0": [
                        {
                            "obj_id": "cube",
                            "cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                            "cam_t_m2c": [10, 20, 30],
                        }
                    ]
                }
            )
        )
    object_folder = tmp_path / "objects"
    object_folder.mkdir()
    (object_folder / "objects.json").write_text(json.dumps({"cube": []}))
    (object_folder / "cube.ply").write_text("ply\nformat ascii 1.0\nend_header\n")
    repo_root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_bop_export_stage.py"),
            str(run_root),
            "--object-folder",
            str(object_folder),
            "--write-multiview-targets",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    multiview = json.loads((run_root / BOP_DIR / BOP_MULTIVIEW_TARGETS).read_text())
    assert multiview["schema_version"] == "posetestbot_bop_multiview_targets.v1"
    assert multiview["scene_count"] == 2
    assert multiview["object_count"] == 1
    target = multiview["targets"][0]
    assert target["obj_id"] == 1
    assert target["sensor_names"] == ["realsense_123", "zed_2i_42"]
    assert target["scene_ids"] == [1, 2]
    assert target["view_count"] == 2
    assert target["instance_count"] == 2
    assert target["views"] == [
        {"scene_id": 1, "sensor_name": "realsense_123", "im_id": 0, "inst_count": 1},
        {"scene_id": 2, "sensor_name": "zed_2i_42", "im_id": 0, "inst_count": 1},
    ]

    bop_manifest = json.loads(
        (run_root / BOP_DIR / BOP_EXPORT_MANIFEST).read_text()
    )
    assert bop_manifest["multiview_targets_path"].endswith(BOP_MULTIVIEW_TARGETS)
    run_manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(stage for stage in run_manifest["stages"] if stage["name"] == "bop_export")
    assert stage["artifacts"][BOP_MULTIVIEW_TARGETS] == (
        "bop/posetestbot_multiview_targets.json"
    )


def test_bop_export_stage_writes_coco_annotations(tmp_path: Path) -> None:
    run_root = create_synchronized_sensor_fixture(tmp_path)
    sensor_folder = run_root / "processed" / "synchronized" / "realsense_123"
    cv2.imwrite(
        (sensor_folder / RGB_DIR / "000010.png").as_posix(),
        np.zeros((3, 4, 3), dtype=np.uint8),
    )
    masks_folder = sensor_folder / MASKS_DIR
    masks_folder.mkdir()
    mask = np.zeros((3, 4), dtype=np.uint8)
    mask[1:3, 1:4] = 255
    cv2.imwrite((masks_folder / "000000_000000.png").as_posix(), mask)
    blenderproc_output = sensor_folder / "blenderproc" / "output"
    blenderproc_output.mkdir(parents=True)
    blenderproc_output.joinpath("scene_gt.json").write_text(
        json.dumps(
            {
                "0": [
                    {
                        "obj_id": "cube",
                        "cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        "cam_t_m2c": [10, 20, 30],
                    }
                ],
                "1": [],
            }
        )
    )
    object_folder = tmp_path / "objects"
    object_folder.mkdir()
    (object_folder / "objects.json").write_text(json.dumps({"cube": []}))
    (object_folder / "cube.ply").write_text("ply\nformat ascii 1.0\nend_header\n")
    repo_root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_bop_export_stage.py"),
            str(run_root),
            "--object-folder",
            str(object_folder),
            "--write-coco-annotations",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    coco = json.loads((run_root / BOP_DIR / BOP_COCO_ANNOTATIONS).read_text())
    assert coco["schema_version"] == "posetestbot_coco_annotations.v1"
    assert coco["posetestbot"] == {
        "split": "test",
        "scene_count": 1,
        "image_count": 2,
        "annotation_count": 1,
    }
    assert coco["images"][0]["file_name"] == (
        "realsense_123/test/000001/rgb/000000.png"
    )
    assert coco["images"][0]["width"] == 4
    assert coco["images"][0]["height"] == 3
    assert coco["categories"] == [
        {"id": 1, "name": "cube", "supercategory": "object"}
    ]
    annotation = coco["annotations"][0]
    assert annotation["image_id"] == coco["images"][0]["id"]
    assert annotation["category_id"] == 1
    assert annotation["bbox"] == [1.0, 1.0, 3.0, 2.0]
    assert annotation["area"] == 6.0
    assert annotation["segmentation"]
    assert annotation["posetestbot"]["scene_id"] == 1
    assert annotation["posetestbot"]["im_id"] == 0
    assert annotation["posetestbot"]["mask_path"] == (
        "realsense_123/test/000001/mask/000000_000000.png"
    )

    bop_manifest = json.loads(
        (run_root / BOP_DIR / BOP_EXPORT_MANIFEST).read_text()
    )
    assert bop_manifest["coco_annotations_path"].endswith(BOP_COCO_ANNOTATIONS)
    run_manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(stage for stage in run_manifest["stages"] if stage["name"] == "bop_export")
    assert stage["artifacts"][BOP_COCO_ANNOTATIONS] == (
        "bop/posetestbot_coco_annotations.json"
    )


def test_bop_export_stage_writes_model_geometry_metadata(tmp_path: Path) -> None:
    run_root = create_synchronized_sensor_fixture(tmp_path)
    object_folder = tmp_path / "objects"
    object_folder.mkdir()
    (object_folder / "objects.json").write_text(json.dumps({"cuboid": []}))
    (object_folder / "cuboid.ply").write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "element face 0",
                "property list uchar int vertex_indices",
                "end_header",
                "0 0 0",
                "3 4 12",
                "",
            ]
        )
    )
    repo_root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_bop_export_stage.py"),
            str(run_root),
            "--object-folder",
            str(object_folder),
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    models_info = json.loads(
        (run_root / BOP_DIR / MODELS_DIR / "models_info.json").read_text()
    )
    assert models_info["1"]["source_name"] == "cuboid"
    assert models_info["1"]["diameter"] == 13.0
    assert models_info["1"]["min_x"] == 0.0
    assert models_info["1"]["min_y"] == 0.0
    assert models_info["1"]["min_z"] == 0.0
    assert models_info["1"]["size_x"] == 3.0
    assert models_info["1"]["size_y"] == 4.0
    assert models_info["1"]["size_z"] == 12.0
    assert models_info["1"]["posetestbot_geometry"]["vertex_count"] == 2
    assert (
        models_info["1"]["posetestbot_geometry"]["diameter_method"]
        == "exact_vertex_pairwise"
    )
