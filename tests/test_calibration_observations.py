from __future__ import annotations

import json
import subprocess
from pathlib import Path

from posetestbot.calibration.observations import (
    build_calibration_observations,
    discover_calibration_pose_outputs,
    discover_aruco_outputs,
    write_calibration_observations_with_manifest,
)
from posetestbot.calibration.targets import normalize_calibration_target_spec
from posetestbot.io.artifacts import (
    ARUCO_POSE_ESTIMATION,
    CALIBRATION_OBSERVATIONS,
    CHARUCO_POSE_ESTIMATION,
    CHECKERBOARD_POSE_ESTIMATION,
    DATASET_MANIFEST,
)
from posetestbot.pipeline.run_config import (
    create_run_config,
    sensor_config_from_token,
    write_run_config,
)


def write_aruco_fixture(run_root: Path) -> Path:
    config = create_run_config(
        run_root=run_root,
        sensors=(
            sensor_config_from_token("realsense:123:static:Static RealSense"),
        ),
        sequence_id="sync_aruco_calibration_observations",
    )
    write_run_config(run_root, config)
    aruco_path = (
        run_root
        / "processed"
        / "synchronized"
        / "realsense_123"
        / ARUCO_POSE_ESTIMATION
    )
    aruco_path.parent.mkdir(parents=True, exist_ok=True)
    aruco_path.write_text(
        json.dumps(
            {
                "000000.png": {
                    "motion": "circ_far",
                    "image_frame": 1000,
                    "source_rgb": "rgb/1000.png",
                    "synchronized_rgb": "processed/synchronized/realsense_123/rgb/000000.png",
                    "nearest_robot_delta_ns": 10_000_000,
                    "robot_ee_pose": {
                        "X": 1,
                        "Y": 2,
                        "Z": 3,
                        "A": 4,
                        "B": 5,
                        "C": 6,
                    },
                    "aruco_pose_estimation": {
                        "rvec": [0.1, 0.2, 0.3],
                        "tvec": [10, 20, 30],
                        "len_ids": 4,
                    },
                },
                "000001.png": {
                    "motion": "circ_far",
                    "robot_ee_pose": {"X": 1},
                    "aruco_pose_estimation": {
                        "rvec": [],
                        "tvec": [],
                        "len_ids": 1,
                    },
                },
            }
        )
        + "\n"
    )
    return aruco_path


def test_build_calibration_observations_extracts_valid_frames(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    aruco_path = write_aruco_fixture(run_root)

    report = build_calibration_observations(
        run_root,
        min_marker_count=4,
        min_observations=1,
    )

    assert discover_aruco_outputs(run_root) == [aruco_path]
    assert report["schema_version"] == "calibration_observations.v1"
    assert report["overall_status"] == "ok"
    assert report["target"] == report["board"]
    assert report["target"]["target_type"] == "aruco_grid"
    assert report["target"]["grid_size"] == [4, 3]
    assert report["sensor_count"] == 1
    assert report["frame_count"] == 2
    assert report["observation_count"] == 1
    assert report["rejected_count"] == 1
    assert report["sensors"][0]["mounting_mode"] == "static"
    observation = report["observations"][0]
    assert observation["observation_id"] == "realsense_123:000000.png"
    assert observation["sensor_type"] == "realsense_d435"
    assert observation["device_id"] == "123"
    assert observation["target_type"] == "aruco_grid"
    assert observation["target_to_camera"]["rotation_vector_rodrigues"] == [
        0.1,
        0.2,
        0.3,
    ]
    assert observation["target_to_camera"]["translation"] == [10.0, 20.0, 30.0]
    assert report["rejected"][0]["reason"] == "insufficient_markers"


def test_build_calibration_observations_records_target_metadata(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    write_aruco_fixture(run_root)
    target = normalize_calibration_target_spec(
        target_type="charuco",
        grid_size="5x7",
        dictionary="DICT_4X4_50",
        marker_length=32.0,
        square_length=40.0,
    )

    report = build_calibration_observations(
        run_root,
        min_observations=1,
        target=target,
    )

    assert report["target"]["target_type"] == "charuco"
    assert report["target"]["grid_size"] == [5, 7]
    assert report["target"]["square_length"] == 40.0
    assert report["observations"][0]["target_type"] == "charuco"
    assert report["observations"][0]["target_to_camera"]["unit"] == "mm"


def test_build_calibration_observations_reads_charuco_pose_outputs(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    charuco_path = (
        run_root
        / "processed"
        / "synchronized"
        / "realsense_123"
        / CHARUCO_POSE_ESTIMATION
    )
    charuco_path.parent.mkdir(parents=True, exist_ok=True)
    charuco_path.write_text(
        json.dumps(
            {
                "000000.png": {
                    "motion": "calibration_sweep",
                    "robot_ee_pose": {"X": 1, "Y": 2, "Z": 3},
                    "charuco_pose_estimation": {
                        "rvec": [0.1, 0.2, 0.3],
                        "tvec": [11, 22, 33],
                        "corner_count": 12,
                    },
                }
            }
        )
        + "\n"
    )
    target = normalize_calibration_target_spec(
        target_type="charuco",
        dictionary="DICT_4X4_50",
        grid_size="5x7",
        marker_length=32.0,
        square_length=40.0,
    )

    report = build_calibration_observations(
        run_root,
        min_marker_count=8,
        min_observations=1,
        target=target,
    )

    assert discover_calibration_pose_outputs(run_root, target_type="charuco") == [
        charuco_path
    ]
    assert report["overall_status"] == "ok"
    assert report["checks"][0]["name"] == "calibration_pose_outputs_present"
    assert report["sensors"][0]["calibration_pose_file"] == (
        "processed/synchronized/realsense_123/charuco_pose_estimation.json"
    )
    observation = report["observations"][0]
    assert observation["target_type"] == "charuco"
    assert observation["target_pose_source"] == "charuco_pose_estimation"
    assert observation["feature_count"] == 12
    assert observation["target_to_camera"]["translation"] == [11.0, 22.0, 33.0]


def test_build_calibration_observations_reads_checkerboard_generic_pose_outputs(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    checkerboard_path = (
        run_root
        / "processed"
        / "synchronized"
        / "zed_2i_auto"
        / CHECKERBOARD_POSE_ESTIMATION
    )
    checkerboard_path.parent.mkdir(parents=True, exist_ok=True)
    checkerboard_path.write_text(
        json.dumps(
            {
                "000000.png": {
                    "robot_ee_pose": {"X": 1, "Y": 2, "Z": 3},
                    "target_pose_estimation": {
                        "rvec": [0.0, 0.1, 0.2],
                        "tvec": [100, 200, 300],
                        "feature_count": 20,
                    },
                },
                "000001.png": {
                    "robot_ee_pose": {"X": 1, "Y": 2, "Z": 3},
                    "target_pose_estimation": {
                        "rvec": [0.0, 0.1, 0.2],
                        "tvec": [100, 200, 300],
                        "feature_count": 3,
                    },
                },
            }
        )
        + "\n"
    )
    target = normalize_calibration_target_spec(
        target_type="checkerboard",
        checkerboard_size="6x9",
        square_length=25.0,
    )

    report = build_calibration_observations(
        run_root,
        min_marker_count=10,
        min_observations=1,
        target=target,
    )

    assert discover_calibration_pose_outputs(run_root, target_type="checkerboard") == [
        checkerboard_path
    ]
    assert report["overall_status"] == "ok"
    assert report["target"]["target_type"] == "checkerboard"
    assert report["observation_count"] == 1
    assert report["observations"][0]["target_pose_source"] == "target_pose_estimation"
    assert report["observations"][0]["feature_count"] == 20
    assert report["rejected"][0]["reason"] == "insufficient_target_features"
    assert report["rejected"][0]["feature_count"] == 3


def test_build_calibration_observations_errors_without_usable_frames(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    aruco_path = (
        run_root
        / "processed"
        / "synchronized"
        / "realsense_123"
        / ARUCO_POSE_ESTIMATION
    )
    aruco_path.parent.mkdir(parents=True, exist_ok=True)
    aruco_path.write_text(
        json.dumps(
            {
                "000000.png": {
                    "aruco_pose_estimation": {"rvec": [], "tvec": [], "len_ids": 0}
                }
            }
        )
    )

    report = build_calibration_observations(run_root)

    assert report["overall_status"] == "error"
    assert report["observation_count"] == 0
    assert any(
        check["name"] == "calibration_observations_present"
        for check in report["checks"]
    )


def test_write_calibration_observations_updates_manifest(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_aruco_fixture(run_root)

    path, report = write_calibration_observations_with_manifest(
        run_root,
        min_observations=1,
    )

    assert path == run_root / CALIBRATION_OBSERVATIONS
    assert report["overall_status"] == "ok"
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage for stage in manifest["stages"] if stage["name"] == "calibration_observations"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][CALIBRATION_OBSERVATIONS] == CALIBRATION_OBSERVATIONS


def test_observation_stage_accepts_explicit_sensor_paths_and_output_root(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    selected_path = write_aruco_fixture(run_root)
    unselected_path = (
        run_root
        / "processed"
        / "synchronized"
        / "luxonis_other"
        / ARUCO_POSE_ESTIMATION
    )
    unselected_path.parent.mkdir(parents=True)
    unselected_path.write_text(selected_path.read_text())
    output_root = run_root / "processed" / "calibration" / "attempt-1"

    path, report = write_calibration_observations_with_manifest(
        run_root,
        min_observations=1,
        aruco_paths=[selected_path],
        output_root=output_root,
    )

    assert path == output_root / CALIBRATION_OBSERVATIONS
    assert report["sensor_count"] == 1
    assert report["sensors"][0]["sensor_name"] == "realsense_123"
    assert not (run_root / CALIBRATION_OBSERVATIONS).exists()


def test_calibration_observations_cli_writes_report(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_aruco_fixture(run_root)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(repo_root / "scripts" / "run_calibration_observations.py"),
            str(run_root),
            "--min-observations",
            "1",
            "--target-type",
            "charuco",
            "--grid-size",
            "5x7",
            "--dictionary",
            "DICT_4X4_50",
            "--marker-length-mm",
            "32",
            "--square-length-mm",
            "40",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert (
        "Calibration observations: ok (1 usable / 2 frames, 1 sensors)"
        in result.stdout
    )
    assert (run_root / CALIBRATION_OBSERVATIONS).is_file()
    data = json.loads((run_root / CALIBRATION_OBSERVATIONS).read_text())
    assert data["target"]["target_type"] == "charuco"
    assert data["target"]["grid_size"] == [5, 7]
