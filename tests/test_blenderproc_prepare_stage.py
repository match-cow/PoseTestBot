from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from posetestbot.calibration.profiles import (
    SCHEMA_VERSION,
    CalibrationProfile,
    RigidTransform,
    TransformFrame,
    write_profile_collection,
)
from posetestbot.io.artifacts import (
    CALIBRATION_PROFILES,
    CAM_K,
    DATASET_MANIFEST,
    DERIVED_CAMERA_EE_TRANSFORM,
    MATCH_ROBOT_EE_POSES,
)
from posetestbot.sensors.contracts import CameraIntrinsics, MountingMode, SensorType


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=2)


def create_blenderproc_prepare_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    run_root = tmp_path / "run-1"
    sensor_folder = run_root / "processed" / "synchronized" / "realsense_123"
    sensor_folder.mkdir(parents=True)
    (sensor_folder / CAM_K).write_text("50 0 40\n0 50 40\n0 0 1\n")
    write_json(
        sensor_folder / MATCH_ROBOT_EE_POSES,
        {
            "000000.png": {
                "motion": "circ_far",
                "robot_ee_pose": {
                    "X": 0.0,
                    "Y": 0.0,
                    "Z": 0.0,
                    "A": 0.0,
                    "B": 0.0,
                    "C": 0.0,
                },
            }
        },
    )

    object_folder = tmp_path / "objects"
    object_folder.mkdir()
    write_json(
        object_folder / "objects.json",
        {
            "cube": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        },
    )
    (object_folder / "cube.ply").write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 0\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )

    camera_transforms = tmp_path / "camera_ee_transform.json"
    write_json(
        camera_transforms,
        {
            "realsense": {
                "quaternion": [1.0, 0.0, 0.0, 0.0],
                "position": [0.0, 0.0, 0.0],
            }
        },
    )
    return run_root, object_folder, camera_transforms


def test_blenderproc_prepare_stage_writes_artifacts_and_manifest(
    tmp_path: Path,
) -> None:
    run_root, object_folder, camera_transforms = create_blenderproc_prepare_fixture(
        tmp_path
    )
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_blenderproc_prepare_stage.py"),
            str(run_root),
            "--object-folder",
            str(object_folder),
            "--camera-transformations",
            str(camera_transforms),
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Prepared BlenderProc inputs for 1 sensor folder" in result.stdout

    blenderproc_folder = (
        run_root / "processed" / "synchronized" / "realsense_123" / "blenderproc"
    )
    assert (blenderproc_folder / "objects" / "cube.ply").exists()
    assert (blenderproc_folder / "objects" / "cube.npy").exists()
    assert (blenderproc_folder / "objects.json").exists()
    np.testing.assert_allclose(
        np.load(blenderproc_folder / "camera_matrix.npy"),
        np.array([[50.0, 0.0, 40.0], [0.0, 50.0, 40.0], [0.0, 0.0, 1.0]]),
    )
    np.testing.assert_allclose(
        np.load(blenderproc_folder / "dist_coefficients.npy"),
        np.zeros((5, 1)),
    )
    camera_poses = np.load(blenderproc_folder / "camera_poses.npy")
    assert camera_poses.shape == (1, 4, 4)

    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage for stage in manifest["stages"] if stage["name"] == "blenderproc_prepare"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"]["realsense_123:blenderproc"].endswith(
        "processed/synchronized/realsense_123/blenderproc"
    )


def test_blenderproc_prepare_stage_accepts_calibration_profiles(
    tmp_path: Path,
) -> None:
    run_root, object_folder, _ = create_blenderproc_prepare_fixture(tmp_path)
    calibration_profiles = tmp_path / "calibration_profiles.json"
    write_profile_collection(
        [
            CalibrationProfile(
                schema_version=SCHEMA_VERSION,
                profile_id="realsense_d435_123_eye_in_hand_wrist_test",
                sensor_id="123",
                sensor_type=SensorType.REALSENSE_D435,
                mounting_mode=MountingMode.EYE_IN_HAND,
                rig_position="wrist",
                intrinsics=CameraIntrinsics(
                    cam_k=(50.0, 0.0, 40.0, 0.0, 50.0, 40.0, 0.0, 0.0, 1.0),
                    width=80,
                    height=80,
                ),
                extrinsics=RigidTransform(
                    from_frame=TransformFrame.CAMERA,
                    to_frame=TransformFrame.END_EFFECTOR,
                    rotation_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                    translation_mm=(10.0, 20.0, 30.0),
                ),
            )
        ],
        calibration_profiles,
    )
    repo_root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_blenderproc_prepare_stage.py"),
            str(run_root),
            "--object-folder",
            str(object_folder),
            "--calibration-profiles",
            str(calibration_profiles),
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    derived_transform = (
        run_root / "processed" / "calibration" / DERIVED_CAMERA_EE_TRANSFORM
    )
    transform_map = json.loads(derived_transform.read_text())
    assert transform_map["realsense_123"]["position"] == [10.0, 20.0, 30.0]

    blenderproc_folder = (
        run_root / "processed" / "synchronized" / "realsense_123" / "blenderproc"
    )
    camera_poses = np.load(blenderproc_folder / "camera_poses.npy")
    np.testing.assert_allclose(camera_poses[0, :3, 3], [0.01, 0.02, 0.03])

    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage for stage in manifest["stages"] if stage["name"] == "blenderproc_prepare"
    )
    assert stage["artifacts"][CALIBRATION_PROFILES].endswith(
        "calibration_profiles.json"
    )
    assert stage["artifacts"][DERIVED_CAMERA_EE_TRANSFORM] == (
        "processed/calibration/camera_ee_transform_from_calibration_profiles.json"
    )


def test_blenderproc_prepare_stage_accepts_static_calibration_profiles(
    tmp_path: Path,
) -> None:
    run_root, object_folder, _ = create_blenderproc_prepare_fixture(tmp_path)
    sensor_folder = run_root / "processed" / "synchronized" / "realsense_123"
    write_json(
        sensor_folder / MATCH_ROBOT_EE_POSES,
        {
            "000000.png": {
                "motion": "circ_far",
                "robot_ee_pose": {
                    "X": 0.0,
                    "Y": 0.0,
                    "Z": 0.0,
                    "A": 0.0,
                    "B": 0.0,
                    "C": 0.0,
                },
            },
            "000001.png": {
                "motion": "circ_close",
                "robot_ee_pose": {
                    "X": 500.0,
                    "Y": 600.0,
                    "Z": 700.0,
                    "A": 0.1,
                    "B": 0.2,
                    "C": 0.3,
                },
            },
        },
    )
    calibration_profiles = tmp_path / "calibration_profiles.json"
    write_profile_collection(
        [
            CalibrationProfile(
                schema_version=SCHEMA_VERSION,
                profile_id="realsense_d435_123_static_cell_front_test",
                sensor_id="123",
                sensor_type=SensorType.REALSENSE_D435,
                mounting_mode=MountingMode.STATIC,
                rig_position="cell_front",
                intrinsics=CameraIntrinsics(
                    cam_k=(50.0, 0.0, 40.0, 0.0, 50.0, 40.0, 0.0, 0.0, 1.0),
                    width=80,
                    height=80,
                ),
                extrinsics=RigidTransform(
                    from_frame=TransformFrame.CAMERA,
                    to_frame=TransformFrame.ROBOT_BASE,
                    rotation_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                    translation_mm=(100.0, 200.0, 300.0),
                ),
            )
        ],
        calibration_profiles,
    )
    repo_root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_blenderproc_prepare_stage.py"),
            str(run_root),
            "--object-folder",
            str(object_folder),
            "--calibration-profiles",
            str(calibration_profiles),
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    derived_transform = (
        run_root / "processed" / "calibration" / DERIVED_CAMERA_EE_TRANSFORM
    )
    transform_map = json.loads(derived_transform.read_text())
    assert transform_map["realsense_123"]["mounting_mode"] == "static"
    assert transform_map["realsense_123"]["to"] == "robot_base"

    blenderproc_folder = sensor_folder / "blenderproc"
    camera_poses = np.load(blenderproc_folder / "camera_poses.npy")
    assert camera_poses.shape == (2, 4, 4)
    np.testing.assert_allclose(camera_poses[0, :3, 3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(camera_poses[1, :3, 3], [0.1, 0.2, 0.3])
