from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

import posetestbot.calibration.extrinsics as extrinsics_module
from posetestbot.calibration.candidates import _robot_ee_to_reference
from posetestbot.calibration.extrinsics import (
    build_grid_extrinsic_solver,
    write_grid_extrinsic_solver_with_manifest,
)
from posetestbot.calibration.profiles import load_profile_collection
from posetestbot.calibration.targets import (
    DEFAULT_TARGET_SPEC,
    normalize_calibration_target_spec,
)
from posetestbot.calibration.validation import (
    build_calibration_validation,
    write_calibration_validation_with_manifest,
)
from posetestbot.io.artifacts import CALIBRATION_OBSERVATIONS, CALIBRATION_PROFILES
from posetestbot.sensors.contracts import CameraIntrinsics
from posetestbot.sensors.frame_writer import write_legacy_camera_sidecars


def target() -> dict:
    return normalize_calibration_target_spec(
        {
            **DEFAULT_TARGET_SPEC,
            "placement": {
                "from": "aruco_grid",
                "to": "template_base",
                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "translation_mm": [0.0, 0.0, 0.0],
                "source": "test_aligned_identity",
            },
        }
    )


def write_eye_in_hand_observations(run_root: Path) -> np.ndarray:
    sensor = run_root / "processed" / "synchronized" / "realsense_123"
    write_legacy_camera_sidecars(
        sensor,
        CameraIntrinsics(
            cam_k=(600.0, 0.0, 320.0, 0.0, 605.0, 240.0, 0.0, 0.0, 1.0),
            width=640,
            height=480,
            distortion=(0.01, -0.01, 0.0, 0.0, 0.0),
            depth_scale_to_mm=1.0,
        ),
        include_distortion_in_cam_k=True,
    )
    camera_to_flange = pt.transform_from(
        pr.matrix_from_compact_axis_angle(np.array([0.08, -0.04, 0.03])),
        np.array([35.0, -20.0, 80.0]),
    )
    robot_poses = [
        {
            "X": 50.0 + 15 * i,
            "Y": -40.0 + 8 * (i % 3),
            "Z": 500.0 + 10 * (i % 4),
            "A": -0.18 + 0.05 * (i % 4),
            "B": 0.12 - 0.04 * (i % 5),
            "C": -0.25 + 0.07 * i,
        }
        for i in range(10)
    ]
    observations = []
    for index, robot_pose in enumerate(robot_poses):
        flange_to_template = _robot_ee_to_reference(robot_pose)
        manager = TransformManager()
        manager.add_transform("aruco_grid", "template_base", np.eye(4))
        manager.add_transform("camera", "robot_flange", camera_to_flange)
        manager.add_transform("robot_flange", "template_base", flange_to_template)
        target_to_camera = manager.get_transform("aruco_grid", "camera")
        rvec = cv2.Rodrigues(target_to_camera[:3, :3])[0].reshape(3)
        observations.append(
            {
                "observation_id": f"realsense_123:{index:06d}.png",
                "sensor_name": "realsense_123",
                "sensor_type": "realsense_d435",
                "device_id": "123",
                "mounting_mode": "eye_in_hand",
                "frame_id": f"{index:06d}.png",
                "target_to_camera": {
                    "rotation_vector_rodrigues": rvec.tolist(),
                    "translation": target_to_camera[:3, 3].tolist(),
                    "unit": "mm",
                },
                "robot_ee_pose": robot_pose,
            }
        )
    report = {
        "schema_version": "calibration_observations.v1",
        "overall_status": "ok",
        "sensors": [
            {
                "sensor_name": "realsense_123",
                "sensor_type": "realsense_d435",
                "device_id": "123",
                "mounting_mode": "eye_in_hand",
            }
        ],
        "observations": observations,
    }
    (run_root / CALIBRATION_OBSERVATIONS).write_text(json.dumps(report))
    return camera_to_flange


def test_compare_recovers_known_camera_to_flange_and_derives_tcp(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    expected = write_eye_in_hand_observations(run_root)
    fixed = [
        {
            "from": "robot_flange",
            "to": "tcp",
            "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "translation_mm": [0.0, 0.0, 50.0],
            "source": "test_tool",
        }
    ]

    report = build_grid_extrinsic_solver(
        run_root,
        target=target(),
        mode="compare",
        min_inliers=6,
        max_mean_translation_mm=1.0,
        max_mean_rotation_deg=1.0,
        max_cross_translation_mm=1.0,
        max_cross_rotation_deg=1.0,
        fixed_transforms=fixed,
    )

    assert report["schema_version"] == "calibration_solver.v2"
    assert report["overall_status"] == "ok"
    assert report["profile_count"] == 2
    assert report["comparisons"][0]["translation_disagreement_mm"] < 1e-3
    assert report["comparisons"][0]["rotation_disagreement_deg"] < 1e-3
    for solution in report["solutions"]:
        actual = np.asarray(solution["transform"]["translation_mm"])
        assert np.allclose(actual, expected[:3, 3], atol=1e-3)
    assert all(
        "derived_camera_to_tcp" in profile["metadata"] for profile in report["profiles"]
    )


def test_unknown_target_static_is_explicitly_unobservable(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    write_eye_in_hand_observations(run_root)
    value = json.loads((run_root / CALIBRATION_OBSERVATIONS).read_text())
    value["sensors"][0]["mounting_mode"] = "static"
    for observation in value["observations"]:
        observation["mounting_mode"] = "static"
    (run_root / CALIBRATION_OBSERVATIONS).write_text(json.dumps(value))

    with pytest.raises(ValueError, match="Workflow step 5"):
        build_grid_extrinsic_solver(
            run_root,
            target=target(),
            mode="hand_eye_unknown_target",
        )


def test_known_target_solves_static_camera_to_template_base(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    write_eye_in_hand_observations(run_root)
    value = json.loads((run_root / CALIBRATION_OBSERVATIONS).read_text())
    value["sensors"][0]["mounting_mode"] = "static"
    for observation in value["observations"]:
        observation["mounting_mode"] = "static"
        observation["target_to_camera"] = {
            "rotation_vector_rodrigues": [0.0, 0.0, 0.0],
            "translation": [100.0, 200.0, 300.0],
            "unit": "mm",
        }
    (run_root / CALIBRATION_OBSERVATIONS).write_text(json.dumps(value))

    report = build_grid_extrinsic_solver(
        run_root,
        target=target(),
        mode="known_target",
    )

    assert report["overall_status"] == "ok"
    profile = report["profiles"][0]
    assert profile["mounting_mode"] == "static"
    assert profile["extrinsics"]["to"] == "template_base"
    assert np.allclose(profile["extrinsics"]["translation_mm"], [-100, -200, -300])


def test_comparison_requires_explicit_selection_before_promotion(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    write_eye_in_hand_observations(run_root)
    _report_path, _profiles_path, solver_report = (
        write_grid_extrinsic_solver_with_manifest(
            run_root,
            target=target(),
            mode="compare",
            max_mean_translation_mm=1.0,
            max_mean_rotation_deg=1.0,
            max_cross_translation_mm=1.0,
            max_cross_rotation_deg=1.0,
        )
    )

    missing = build_calibration_validation(run_root)
    assert missing["overall_status"] == "error"
    assert missing["selection"]["explicit_selection_required"] is True

    selected_id = next(
        profile["profile_id"]
        for profile in solver_report["profiles"]
        if profile["method"] == "known_target"
    )
    report_path, promoted_path, selected = write_calibration_validation_with_manifest(
        run_root,
        select_profiles={"realsense_123": selected_id},
        promote=True,
    )

    assert report_path.is_file()
    assert selected["overall_status"] == "ok"
    assert selected["selection"]["selected_profile_ids"] == [selected_id]
    profiles = load_profile_collection(promoted_path)
    assert len(profiles) == 1
    assert profiles[0].profile_id == selected_id
    assert promoted_path == run_root / CALIBRATION_PROFILES


def test_cross_method_gate_blocks_and_records_disabled_override(
    tmp_path: Path, monkeypatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    write_eye_in_hand_observations(run_root)
    original = extrinsics_module._unknown_solution

    def shifted(*args, **kwargs):
        solution, residuals, inliers, estimate = original(*args, **kwargs)
        solution = solution.copy()
        solution[0, 3] += 20.0
        return solution, residuals, inliers, estimate

    monkeypatch.setattr(extrinsics_module, "_unknown_solution", shifted)
    blocked = build_grid_extrinsic_solver(
        run_root,
        target=target(),
        mode="compare",
        max_mean_translation_mm=30.0,
        max_mean_rotation_deg=5.0,
    )
    overridden = build_grid_extrinsic_solver(
        run_root,
        target=target(),
        mode="compare",
        max_mean_translation_mm=30.0,
        max_mean_rotation_deg=5.0,
        max_cross_translation_mm=None,
        max_cross_rotation_deg=None,
    )

    assert blocked["overall_status"] == "error"
    assert blocked["comparisons"][0]["status"] == "error"
    assert overridden["overall_status"] == "ok"
    assert overridden["comparisons"][0]["gate_override"]["disabled"] is True
