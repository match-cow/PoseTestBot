from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from posetestbot.calibration.profiles import (
    SCHEMA_VERSION,
    CalibrationProfile,
    CalibrationQuality,
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


IDENTITY_OBJECT_TRANSFORM = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def create_synchronized_sensor_fixture(tmp_path: Path) -> Path:
    run_root = tmp_path / "run-1"
    sensor_folder = run_root / "processed" / "synchronized" / "realsense_123"
    rgb_folder = sensor_folder / RGB_DIR
    depth_folder = sensor_folder / DEPTH_DIR
    rgb_folder.mkdir(parents=True)
    depth_folder.mkdir()
    rgb_first = np.zeros((5, 6, 3), dtype=np.uint8)
    rgb_first[:, :, 1] = 10
    rgb_second = np.zeros((5, 6, 3), dtype=np.uint8)
    rgb_second[:, :, 1] = 20
    depth_first = np.ones((5, 6), dtype=np.uint16)
    depth_second = np.ones((5, 6), dtype=np.uint16) * 2
    assert cv2.imwrite((rgb_folder / "000010.png").as_posix(), rgb_first)
    assert cv2.imwrite((depth_folder / "000010.png").as_posix(), depth_first)
    assert cv2.imwrite((rgb_folder / "000020.png").as_posix(), rgb_second)
    assert cv2.imwrite((depth_folder / "000020.png").as_posix(), depth_second)
    (sensor_folder / CAM_K).write_text("1 0 2\n0 3 4\n0 0 1\n")
    (sensor_folder / DEPTH_SCALE).write_text("0.001\n")
    return run_root


def write_simple_ply(path: Path) -> None:
    path.write_text(
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
                "1 0 0",
                "",
            ]
        )
    )


def write_annotation_mask(sensor_folder: Path, *, image_id: int = 0) -> None:
    masks_folder = sensor_folder / MASKS_DIR
    masks_folder.mkdir(exist_ok=True)
    mask = np.ones((5, 6), dtype=np.uint8) * 255
    assert cv2.imwrite(
        (masks_folder / f"{image_id:06d}_000000.png").as_posix(), mask
    )


def test_bop_export_stage_writes_scene_and_manifest(tmp_path: Path) -> None:
    run_root = create_synchronized_sensor_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
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

    assert "Exported 1 synchronized sensor folder" in result.stdout

    scene_folder = run_root / BOP_DIR / "test" / "000001"
    assert cv2.imread(
        (scene_folder / RGB_DIR / "000000.png").as_posix(), cv2.IMREAD_UNCHANGED
    )[0, 0, 1] == 10
    assert cv2.imread(
        (scene_folder / DEPTH_DIR / "000001.png").as_posix(),
        cv2.IMREAD_UNCHANGED,
    )[0, 0] == 2

    scene_camera = json.loads((scene_folder / "scene_camera.json").read_text())
    assert scene_camera["0"]["cam_K"] == [1, 0, 2, 0, 3, 4, 0, 0, 1]
    assert scene_camera["0"]["depth_scale"] == 0.001

    scene_gt = json.loads((scene_folder / "scene_gt.json").read_text())
    scene_gt_info = json.loads((scene_folder / "scene_gt_info.json").read_text())
    assert scene_gt == {"0": [], "1": []}
    assert scene_gt_info == {"0": [], "1": []}

    frame_map = json.loads((run_root / BOP_DIR / BOP_FRAME_MAP_JSON).read_text())
    frames = frame_map["scenes"]["1"]["frames"]
    assert frames["0"]["source_rgb"] == "rgb/000010.png"
    assert frames["1"]["bop_depth"] == "depth/000001.png"

    bop_manifest = json.loads(
        (run_root / BOP_DIR / BOP_EXPORT_MANIFEST).read_text()
    )
    assert bop_manifest["schema_version"] == "bop_export_manifest.v2"
    assert bop_manifest["format"] == "bop-scenewise"
    assert bop_manifest["exports"][0]["rgb_count"] == 2
    assert bop_manifest["validation"]["status"] == "ok"

    run_manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(stage for stage in run_manifest["stages"] if stage["name"] == "bop_export")
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][BOP_EXPORT_MANIFEST] == "bop/bop_export_manifest.json"
    assert stage["artifacts"]["realsense_123:bop_scene"] == (
        "bop/test/000001"
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
                quality=CalibrationQuality(num_observations=1, num_inliers=1),
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
            "--no-model-export",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    scene_folder = run_root / BOP_DIR / "test" / "000001"
    scene_camera = json.loads((scene_folder / "scene_camera.json").read_text())
    assert scene_camera["0"]["cam_K"] == [9.0, 0.0, 8.0, 0.0, 7.0, 6.0, 0.0, 0.0, 1.0]
    assert scene_camera["0"]["depth_scale"] == 2.5
    calibration = scene_camera["0"]["posetestbot_calibration"]
    assert calibration["calibration_profile_id"] == (
        "realsense_d435_123_static_front_left_v2026_01"
    )
    assert calibration["mounting_mode"] == "static"
    assert calibration["extrinsics"]["to"] == "template_base"
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


def test_bop_export_prefers_rectified_tree_and_records_projection_provenance(
    tmp_path: Path,
) -> None:
    run_root = create_synchronized_sensor_fixture(tmp_path)
    synchronized = run_root / "processed" / "synchronized" / "realsense_123"
    rectified = run_root / "processed" / "rectified" / "realsense_123"
    shutil.copytree(synchronized, rectified)
    (rectified / "rectification_provenance.json").write_text(
        json.dumps({"projection": "rectified_alpha0"})
    )
    profiles_path = run_root / CALIBRATION_PROFILES
    write_profile_collection(
        [
            CalibrationProfile(
                schema_version=SCHEMA_VERSION,
                profile_id="rectified_profile",
                sensor_id="123",
                sensor_type=SensorType.REALSENSE_D435,
                mounting_mode=MountingMode.STATIC,
                rig_position="static",
                intrinsics=CameraIntrinsics(
                    cam_k=(1.0, 0.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 1.0),
                    width=6,
                    height=5,
                    distortion=(0.1, 0.0, 0.0, 0.0, 0.0),
                    depth_scale_to_mm=0.001,
                ),
                rectified_intrinsics=CameraIntrinsics(
                    cam_k=(2.0, 0.0, 2.5, 0.0, 4.0, 2.0, 0.0, 0.0, 1.0),
                    width=6,
                    height=5,
                    distortion=(0.0,) * 5,
                    depth_scale_to_mm=0.001,
                ),
                extrinsics=RigidTransform(
                    from_frame=TransformFrame.CAMERA,
                    to_frame=TransformFrame.TEMPLATE_BASE,
                    rotation_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                    translation_mm=(0.0, 0.0, 0.0),
                ),
                status=CalibrationStatus.VALID,
                quality=CalibrationQuality(num_observations=6, num_inliers=6),
            )
        ],
        profiles_path,
    )
    repo_root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_bop_export_stage.py"),
            str(run_root),
            "--calibration-profiles",
            str(profiles_path),
            "--no-model-export",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    scene_camera = json.loads(
        (run_root / BOP_DIR / "test" / "000001" / "scene_camera.json").read_text()
    )
    assert scene_camera["0"]["cam_K"] == [2.0, 0.0, 2.5, 0.0, 4.0, 2.0, 0.0, 0.0, 1.0]
    metadata = scene_camera["0"]["posetestbot_calibration"]
    assert metadata["projection"] == "rectified"
    assert metadata["projection_provenance"]["rectification"] == "alpha0"


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

    scene_folder = run_root / BOP_DIR / "test" / "000001"
    scene_gt = json.loads((scene_folder / "scene_gt.json").read_text())
    scene_gt_info = json.loads((scene_folder / "scene_gt_info.json").read_text())
    assert scene_gt["0"][0]["obj_id"] == 1
    assert scene_gt["0"][0]["cam_t_m2c"] == [10, 20, 30]
    assert scene_gt_info["0"][0]["bbox_obj"] == [2, 1, 3, 3]
    assert (scene_folder / "mask" / "000000_000000.png").is_file()

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
    depth = cv2.imread(
        (sensor_folder / DEPTH_DIR / "000010.png").as_posix(),
        cv2.IMREAD_UNCHANGED,
    )
    depth[1:4, 2:5] = 0
    depth[1, 2] = 1
    assert cv2.imwrite(
        (sensor_folder / DEPTH_DIR / "000010.png").as_posix(), depth
    )
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

    scene_folder = run_root / BOP_DIR / "test" / "000001"
    scene_gt_info = json.loads((scene_folder / "scene_gt_info.json").read_text())
    assert scene_gt_info["0"][0]["bbox_obj"] == [2, 1, 3, 3]
    assert scene_gt_info["0"][0]["bbox_visib"] == [2, 1, 3, 3]
    assert scene_gt_info["0"][0]["px_count_all"] == 9
    assert scene_gt_info["0"][0]["px_count_valid"] == 1
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
    write_annotation_mask(sensor_folder)
    object_folder = tmp_path / "objects"
    object_folder.mkdir()
    (object_folder / "objects.json").write_text(
        json.dumps({"cube": IDENTITY_OBJECT_TRANSFORM, "sphere": IDENTITY_OBJECT_TRANSFORM})
    )
    write_simple_ply(object_folder / "cube.ply")
    write_simple_ply(object_folder / "sphere.ply")
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

    scene_folder = run_root / BOP_DIR / "test" / "000001"
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
    assert cv2.imwrite(
        (second_sensor / RGB_DIR / "000010.png").as_posix(),
        np.zeros((5, 6, 3), dtype=np.uint8),
    )
    assert cv2.imwrite(
        (second_sensor / DEPTH_DIR / "000010.png").as_posix(),
        np.ones((5, 6), dtype=np.uint16),
    )
    (second_sensor / CAM_K).write_text("1 0 2\n0 3 4\n0 0 1\n")
    (second_sensor / DEPTH_SCALE).write_text("0.001\n")
    for sensor_folder in (first_sensor, second_sensor):
        blenderproc_output = sensor_folder / "blenderproc" / "output"
        blenderproc_output.mkdir(parents=True)
        scene_gt = {
            "0": [
                {
                    "obj_id": "cube",
                    "cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    "cam_t_m2c": [10, 20, 30],
                }
            ]
        }
        if sensor_folder == first_sensor:
            scene_gt["1"] = []
        blenderproc_output.joinpath("scene_gt.json").write_text(
            json.dumps(scene_gt)
        )
        write_annotation_mask(sensor_folder)
    object_folder = tmp_path / "objects"
    object_folder.mkdir()
    (object_folder / "objects.json").write_text(json.dumps({"cube": IDENTITY_OBJECT_TRANSFORM}))
    write_simple_ply(object_folder / "cube.ply")
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
    cv2.imwrite(
        (sensor_folder / DEPTH_DIR / "000010.png").as_posix(),
        np.ones((3, 4), dtype=np.uint16),
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
    (object_folder / "objects.json").write_text(json.dumps({"cube": IDENTITY_OBJECT_TRANSFORM}))
    write_simple_ply(object_folder / "cube.ply")
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
        "test/000001/rgb/000000.png"
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
        "test/000001/mask/000000_000000.png"
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
    (object_folder / "objects.json").write_text(json.dumps({"cuboid": IDENTITY_OBJECT_TRANSFORM}))
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
        == "exact_convex_hull_vertex_pairwise"
    )


def test_bop_overwrite_failure_preserves_previous_dataset(tmp_path: Path) -> None:
    run_root = create_synchronized_sensor_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(repo_root / "scripts" / "run_bop_export_stage.py"),
        str(run_root),
        "--no-model-export",
    ]
    subprocess.run(command, cwd=repo_root, check=True, capture_output=True, text=True)
    manifest_path = run_root / BOP_DIR / BOP_EXPORT_MANIFEST
    previous_manifest = manifest_path.read_bytes()

    sensor_folder = run_root / "processed" / "synchronized" / "realsense_123"
    (sensor_folder / DEPTH_DIR / "000020.png").unlink()
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


def test_bop_export_rejects_unvalidated_calibration_profile(
    tmp_path: Path,
) -> None:
    run_root = create_synchronized_sensor_fixture(tmp_path)
    profiles_path = run_root / CALIBRATION_PROFILES
    profile = CalibrationProfile(
        schema_version=SCHEMA_VERSION,
        profile_id="realsense_candidate",
        sensor_id="123",
        sensor_type=SensorType.REALSENSE_D435,
        mounting_mode=MountingMode.STATIC,
        rig_position="front",
        intrinsics=CameraIntrinsics(
            cam_k=(9.0, 0.0, 8.0, 0.0, 7.0, 6.0, 0.0, 0.0, 1.0),
            width=80,
            height=60,
            depth_scale_to_mm=1.0,
        ),
        extrinsics=RigidTransform(
            from_frame=TransformFrame.CAMERA,
            to_frame=TransformFrame.ROBOT_BASE,
            rotation_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            translation_mm=(0.0, 0.0, 0.0),
        ),
        status=CalibrationStatus.NEEDS_VALIDATION,
        quality=CalibrationQuality(num_observations=4, num_inliers=4),
    )
    write_profile_collection([profile], profiles_path)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_bop_export_stage.py"),
            str(run_root),
            "--no-model-export",
            "--calibration-profiles",
            str(profiles_path),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert not (run_root / BOP_DIR).exists()


def test_bop_export_objectless_writes_empty_gt_targets_and_coco(tmp_path: Path) -> None:
    run_root = create_synchronized_sensor_fixture(tmp_path)
    object_folder = tmp_path / "objects"
    object_folder.mkdir()
    (object_folder / "objects.json").write_text(json.dumps({"cube": IDENTITY_OBJECT_TRANSFORM}))
    write_simple_ply(object_folder / "cube.ply")
    repo_root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "run_bop_export_stage.py"), str(run_root), "--object-folder", str(object_folder), "--objectless", "--write-coco-annotations"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    bop = run_root / BOP_DIR
    manifest = json.loads((bop / BOP_EXPORT_MANIFEST).read_text())
    scene = bop / "test" / "000001"
    assert manifest["objectless"] is True
    assert manifest["selected_objects"] == []
    assert manifest["stable_id_mapping"] == {"cube": 1}
    assert not (bop / MODELS_DIR).exists()
    assert json.loads((bop / BOP_TARGETS_BOP19).read_text()) == []
    assert all(value == [] for value in json.loads((scene / "scene_gt.json").read_text()).values())
    assert all(value == [] for value in json.loads((scene / "scene_gt_info.json").read_text()).values())
    assert not (scene / "mask").exists()
    coco = json.loads((bop / BOP_COCO_ANNOTATIONS).read_text())
    assert coco["images"]
    assert coco["categories"] == []
    assert coco["annotations"] == []


def test_bop_export_objectless_rejects_stale_object_gt(tmp_path: Path) -> None:
    run_root = create_synchronized_sensor_fixture(tmp_path)
    sensor = run_root / "processed" / "synchronized" / "realsense_123"
    output = sensor / "blenderproc" / "output"
    output.mkdir(parents=True)
    (output / "scene_gt.json").write_text(json.dumps({"0": [{"obj_id": "cube", "cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1], "cam_t_m2c": [0, 0, 1]}], "1": []}))
    object_folder = tmp_path / "objects"
    object_folder.mkdir()
    (object_folder / "objects.json").write_text(json.dumps({"cube": IDENTITY_OBJECT_TRANSFORM}))
    write_simple_ply(object_folder / "cube.ply")
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "run_bop_export_stage.py"), str(run_root), "--object-folder", str(object_folder), "--objectless"],
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "Invalid obj_id" in result.stderr or "Unknown BOP obj_id" in result.stderr
    assert not (run_root / BOP_DIR).exists()
