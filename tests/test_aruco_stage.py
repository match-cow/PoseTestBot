from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

from posetestbot.io.artifacts import (
    ARUCO_POSE_ESTIMATION,
    CAM_K,
    CALIBRATION_TARGET,
    DATASET_MANIFEST,
    MATCH_ROBOT_EE_POSES,
    RGB_DIR,
)
from posetestbot.calibration.targets import DEFAULT_TARGET_SPEC
from posetestbot.pipeline.run_config import (
    SensorRunConfig,
    create_run_config,
    write_run_config,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=2)


def test_run_aruco_stage_updates_manifest_for_synchronized_sensor(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-1"
    sensor_folder = run_root / "processed" / "synchronized" / "realsense_123"
    rgb_folder = sensor_folder / RGB_DIR
    rgb_folder.mkdir(parents=True)

    image = np.zeros((80, 80, 3), dtype=np.uint8)
    assert cv.imwrite(str(rgb_folder / "000000.png"), image)
    (sensor_folder / CAM_K).write_text("50 0 40\n0 50 40\n0 0 1\n")
    write_json(
        sensor_folder / MATCH_ROBOT_EE_POSES,
        {
            "000000.png": {
                "motion": "circ_far",
                "image_frame": 1000,
                "robot_ee_pose": {"X": 1, "Y": 2, "Z": 3, "A": 4, "B": 5, "C": 6},
            }
        },
    )
    write_json(run_root / CALIBRATION_TARGET, DEFAULT_TARGET_SPEC)

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_aruco_stage.py"),
            str(run_root),
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "ArUco finished for 1 sensor folder" in result.stdout

    aruco_output = json.loads((sensor_folder / ARUCO_POSE_ESTIMATION).read_text())
    assert aruco_output["000000.png"]["aruco_pose_estimation"]["len_ids"] == 0

    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage for stage in manifest["stages"] if stage["name"] == "aruco:realsense_123"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][ARUCO_POSE_ESTIMATION].endswith(
        "processed/synchronized/realsense_123/aruco_pose_estimation.json"
    )


def test_run_wide_aruco_stage_ignores_disabled_configured_sensor(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-disabled"
    enabled = run_root / "processed" / "synchronized" / "realsense_123"
    rgb_folder = enabled / RGB_DIR
    rgb_folder.mkdir(parents=True)
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    assert cv.imwrite(str(rgb_folder / "000000.png"), image)
    (enabled / CAM_K).write_text("50 0 40\n0 50 40\n0 0 1\n")
    write_json(
        enabled / MATCH_ROBOT_EE_POSES,
        {
            "000000.png": {
                "motion": "circ_far",
                "image_frame": 1000,
                "robot_ee_pose": {
                    "X": 1,
                    "Y": 2,
                    "Z": 3,
                    "A": 4,
                    "B": 5,
                    "C": 6,
                },
            }
        },
    )
    write_json(run_root / CALIBRATION_TARGET, DEFAULT_TARGET_SPEC)
    disabled = enabled.parent / "realsense_999"
    shutil.copytree(enabled, disabled)
    write_run_config(
        run_root,
        create_run_config(
            run_root=run_root,
            sensors=(
                SensorRunConfig("realsense_d435", "123", "Enabled"),
                SensorRunConfig("realsense_d435", "999", "Disabled", enabled=False),
            ),
        ),
    )
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_aruco_stage.py"),
            str(run_root),
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "ArUco finished for 1 sensor folder" in result.stdout
    assert (enabled / ARUCO_POSE_ESTIMATION).is_file()
    assert not (disabled / ARUCO_POSE_ESTIMATION).exists()
