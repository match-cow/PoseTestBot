from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from posetestbot.blenderproc.preparation import (
    load_camera_transformations,
    prepare_sensor_folders,
)
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


def create_blenderproc_prepare_fixture(tmp_path: Path) -> tuple[Path, Path]:
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
    return run_root, camera_transforms


def test_blenderproc_prepare_stage_writes_artifacts_and_manifest(
    tmp_path: Path,
) -> None:
    run_root, camera_transforms = create_blenderproc_prepare_fixture(
        tmp_path
    )
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_blenderproc_prepare_stage.py"),
            str(run_root),
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
    assert json.loads((blenderproc_folder / "objects.json").read_text())["instances"] == []
    assert list((blenderproc_folder / "objects").iterdir()) == []
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


def test_blenderproc_prepare_prefers_rectified_sensor_tree(tmp_path: Path) -> None:
    run_root, camera_transforms = create_blenderproc_prepare_fixture(
        tmp_path
    )
    synchronized = run_root / "processed" / "synchronized" / "realsense_123"
    rectified = run_root / "processed" / "rectified" / "realsense_123"
    shutil.copytree(synchronized, rectified)
    repo_root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_blenderproc_prepare_stage.py"),
            str(run_root),
            "--camera-transformations",
            str(camera_transforms),
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert (rectified / "blenderproc" / "camera_matrix.npy").is_file()
    assert not (synchronized / "blenderproc").exists()


def test_blenderproc_prepare_stage_accepts_calibration_profiles(
    tmp_path: Path,
) -> None:
    run_root, _ = create_blenderproc_prepare_fixture(tmp_path)
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
                status=CalibrationStatus.VALID,
                quality=CalibrationQuality(num_observations=8, num_inliers=8),
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
    run_root, _ = create_blenderproc_prepare_fixture(tmp_path)
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
                status=CalibrationStatus.VALID,
                quality=CalibrationQuality(num_observations=8, num_inliers=8),
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
    assert transform_map["realsense_123"]["to"] == "template_base"

    blenderproc_folder = sensor_folder / "blenderproc"
    camera_poses = np.load(blenderproc_folder / "camera_poses.npy")
    assert camera_poses.shape == (2, 4, 4)
    np.testing.assert_allclose(camera_poses[0, :3, 3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(camera_poses[1, :3, 3], [0.1, 0.2, 0.3])


def test_blenderproc_prepare_failure_preserves_all_existing_outputs(
    tmp_path: Path,
) -> None:
    run_root, camera_transforms = create_blenderproc_prepare_fixture(
        tmp_path
    )
    synchronized = run_root / "processed" / "synchronized"
    first_output = synchronized / "realsense_123" / "blenderproc"
    first_output.mkdir()
    (first_output / "previous.txt").write_text("keep")
    invalid_sensor = synchronized / "zed_2i_456"
    invalid_sensor.mkdir()
    (invalid_sensor / CAM_K).write_text("50 0 40\n0 50 40\n0 0 1\n")

    transforms = dict(load_camera_transformations(camera_transforms))
    transforms["zed_2i"] = transforms["realsense"]
    with pytest.raises(FileNotFoundError, match="matched robot poses"):
        prepare_sensor_folders(
            input_folder=synchronized,
            camera_transformations=transforms,
        )

    assert (first_output / "previous.txt").read_text() == "keep"
    assert not (invalid_sensor / "blenderproc").exists()
    assert not list(synchronized.rglob("*.staging"))


def test_blenderproc_prepare_objectless_clears_stale_models(tmp_path: Path) -> None:
    run_root, camera_transforms = create_blenderproc_prepare_fixture(tmp_path)
    sensor = run_root / "processed" / "synchronized" / "realsense_123"
    stale = sensor / "blenderproc" / "objects"
    stale.mkdir(parents=True)
    (stale / "stale.ply").write_text("stale")

    prepared = prepare_sensor_folders(
        input_folder=run_root / "processed" / "synchronized",
        camera_transformations=load_camera_transformations(camera_transforms),
    )

    output = prepared[0].output_folder
    assert prepared[0].object_count == 0
    assert json.loads((output / "objects.json").read_text())["instances"] == []
    assert list((output / "objects").iterdir()) == []
    assert (output / "camera_poses.npy").is_file()
