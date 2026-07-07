from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

import posetestbot.calibration.solver as solver_module
from posetestbot.calibration.profiles import load_profile_collection
from posetestbot.calibration.solver import (
    HAND_EYE_METHODS,
    build_calibration_solver,
    write_calibration_solver_with_manifest,
)
from posetestbot.io.artifacts import (
    CALIBRATION_OBSERVATIONS,
    CALIBRATION_PROFILES_SOLVED,
    CALIBRATION_SOLVER_REPORT,
    DATASET_MANIFEST,
)


IDENTITY_TARGET_TO_REFERENCE = {
    "from": "calibration_target",
    "to": "robot_base",
    "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    "translation_mm": [0.0, 0.0, 0.0],
    "unit": "mm",
    "source": "test_identity_target",
}


def write_observations_fixture(
    run_root: Path,
    *,
    observation_count: int = 3,
    mounting_mode: str = "static",
) -> Path:
    observations = []
    for index in range(observation_count):
        observations.append(
            {
                "observation_id": f"realsense_123:{index:06d}.png",
                "sensor_name": "realsense_123",
                "sensor_type": "realsense_d435",
                "device_id": "123",
                "mounting_mode": mounting_mode,
                "motion": "calibration",
                "frame_id": f"{index:06d}.png",
                "target_to_camera": {
                    "rotation_vector_rodrigues": [0.0, 0.0, 0.0],
                    "translation": [0.0, 0.0, 0.0],
                    "unit": "mm",
                },
                "robot_ee_pose": {
                    "X": 0.0,
                    "Y": 0.0,
                    "Z": 0.0,
                    "A": 0.0,
                    "B": 0.0,
                    "C": 0.0,
                },
            }
        )
    report = {
        "schema_version": "calibration_observations.v1",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "run_root": run_root.as_posix(),
        "overall_status": "ok",
        "sensor_count": 1,
        "frame_count": observation_count,
        "observation_count": observation_count,
        "rejected_count": 0,
        "motion_count": 1,
        "checks": [],
        "sensors": [
            {
                "sensor_name": "realsense_123",
                "sensor_type": "realsense_d435",
                "device_id": "123",
                "mounting_mode": mounting_mode,
                "frame_count": observation_count,
                "observation_count": observation_count,
                "rejected_count": 0,
                "motions": ["calibration"],
            }
        ],
        "observations": observations,
        "rejected": [],
    }
    path = run_root / CALIBRATION_OBSERVATIONS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report) + "\n")
    return path


def test_build_calibration_solver_solves_static_identity(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_observations_fixture(run_root, observation_count=2, mounting_mode="static")

    report = build_calibration_solver(
        run_root,
        min_observations=2,
        target_to_reference=IDENTITY_TARGET_TO_REFERENCE,
        max_translation_residual_mm=None,
        max_rotation_residual_deg=None,
    )

    assert report["schema_version"] == "calibration_solver.v1"
    assert report["overall_status"] == "ok"
    assert report["profile_count"] == 1
    assert report["observation_count"] == 2
    assert report["inlier_count"] == 2
    profile = report["profiles"][0]
    assert profile["profile_id"] == "realsense_123_static_aruco_solved"
    assert profile["status"] == "needs_validation"
    assert profile["method"] == "static_target_reference_transform_average"
    assert profile["extrinsics"]["from"] == "camera"
    assert profile["extrinsics"]["to"] == "robot_base"
    assert profile["extrinsics"]["translation_mm"] == [0.0, 0.0, 0.0]
    assert profile["quality"]["num_inliers"] == 2
    assert report["solutions"][0]["residuals"]["mean_translation_mm"] == 0.0


def test_build_calibration_solver_reports_held_out_residuals(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_observations_fixture(run_root, observation_count=4, mounting_mode="static")

    report = build_calibration_solver(
        run_root,
        min_observations=2,
        target_to_reference=IDENTITY_TARGET_TO_REFERENCE,
        max_translation_residual_mm=None,
        max_rotation_residual_deg=None,
        holdout_fraction=0.25,
    )

    assert report["overall_status"] == "ok"
    assert report["holdout_fraction"] == 0.25
    solution = report["solutions"][0]
    assert solution["train_observation_count"] == 3
    assert solution["holdout_observation_count"] == 1
    assert solution["holdout_status"] == "ok"
    assert solution["holdout_residuals"]["mean_translation_mm"] == 0.0
    residual_splits = {record["split"] for record in report["residuals"]}
    assert residual_splits == {"train", "holdout"}
    profile = report["profiles"][0]
    assert profile["metadata"]["holdout_count"] == 1
    assert profile["metadata"]["holdout_mean_residual_translation_mm"] == 0.0


def test_build_calibration_solver_uses_opencv_hand_eye(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "run"
    write_observations_fixture(
        run_root,
        observation_count=3,
        mounting_mode="eye_in_hand",
    )
    calls = []

    def fake_calibrate_hand_eye(
        rotations_gripper_to_base,
        translations_gripper_to_base,
        rotations_target_to_camera,
        translations_target_to_camera,
        *,
        method,
    ):
        calls.append(
            {
                "gripper_count": len(rotations_gripper_to_base),
                "target_count": len(rotations_target_to_camera),
                "method": method,
            }
        )
        return np.eye(3), np.array([[1.0], [2.0], [3.0]])

    monkeypatch.setattr(
        solver_module.cv2,
        "calibrateHandEye",
        fake_calibrate_hand_eye,
    )

    report = build_calibration_solver(
        run_root,
        min_observations=3,
        hand_eye_method="tsai",
        max_translation_residual_mm=None,
        max_rotation_residual_deg=None,
    )

    assert calls
    assert calls[0]["gripper_count"] == 3
    assert report["overall_status"] == "ok"
    assert report["profile_count"] == 1
    profile = report["profiles"][0]
    assert profile["method"] == "opencv_calibrateHandEye_tsai"
    assert profile["extrinsics"]["to"] == "end_effector"
    assert profile["extrinsics"]["translation_mm"] == [1.0, 2.0, 3.0]


def test_build_calibration_solver_compares_hand_eye_methods(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "run"
    write_observations_fixture(
        run_root,
        observation_count=4,
        mounting_mode="eye_in_hand",
    )
    calls = []

    def fake_calibrate_hand_eye(
        rotations_gripper_to_base,
        translations_gripper_to_base,
        rotations_target_to_camera,
        translations_target_to_camera,
        *,
        method,
    ):
        calls.append(method)
        return np.eye(3), np.array([[0.0], [0.0], [0.0]])

    monkeypatch.setattr(
        solver_module.cv2,
        "calibrateHandEye",
        fake_calibrate_hand_eye,
    )

    report = build_calibration_solver(
        run_root,
        min_observations=3,
        hand_eye_method="park",
        max_translation_residual_mm=None,
        max_rotation_residual_deg=None,
        holdout_fraction=0.25,
        compare_hand_eye_methods=True,
    )

    assert report["overall_status"] == "ok"
    assert report["compare_hand_eye_methods"] is True
    assert len(report["method_comparisons"]) == len(HAND_EYE_METHODS)
    assert {record["status"] for record in report["method_comparisons"]} == {"ok"}
    selected = [
        record
        for record in report["method_comparisons"]
        if record["selected"]
    ]
    assert [record["method"] for record in selected] == ["opencv_calibrateHandEye_park"]
    assert selected[0]["holdout_status"] == "ok"
    assert len(report["solutions"][0]["method_comparisons"]) == len(HAND_EYE_METHODS)
    assert set(calls) == set(HAND_EYE_METHODS.values())


def test_write_calibration_solver_updates_manifest_and_profiles(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    write_observations_fixture(run_root, observation_count=2, mounting_mode="static")

    report_path, profiles_path, report = write_calibration_solver_with_manifest(
        run_root,
        min_observations=2,
        target_to_reference=IDENTITY_TARGET_TO_REFERENCE,
        max_translation_residual_mm=None,
        max_rotation_residual_deg=None,
    )

    assert report_path == run_root / CALIBRATION_SOLVER_REPORT
    assert profiles_path == run_root / CALIBRATION_PROFILES_SOLVED
    assert report["overall_status"] == "ok"
    profiles = load_profile_collection(profiles_path)
    assert len(profiles) == 1
    assert profiles[0].method == "static_target_reference_transform_average"
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage for stage in manifest["stages"] if stage["name"] == "calibration_solver"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][CALIBRATION_SOLVER_REPORT] == CALIBRATION_SOLVER_REPORT
    assert stage["artifacts"][CALIBRATION_PROFILES_SOLVED] == CALIBRATION_PROFILES_SOLVED


def test_calibration_solver_cli_writes_report(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_observations_fixture(run_root, observation_count=2, mounting_mode="static")
    target_path = run_root / "target_to_reference.json"
    target_path.write_text(json.dumps(IDENTITY_TARGET_TO_REFERENCE) + "\n")
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(repo_root / "scripts" / "run_calibration_solver.py"),
            str(run_root),
            "--min-observations",
            "2",
            "--target-to-reference",
            str(target_path),
            "--no-residual-thresholds",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert (
        "Calibration solver: ok (1 profiles, 2 inliers / 2 observations, 1 sensors)"
        in result.stdout
    )
    assert (run_root / CALIBRATION_SOLVER_REPORT).is_file()
    assert (run_root / CALIBRATION_PROFILES_SOLVED).is_file()
