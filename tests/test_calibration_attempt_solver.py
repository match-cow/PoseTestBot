from __future__ import annotations

import cv2
import json
import numpy as np
import pytest
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt

from posetestbot.calibration.attempt_solver import (
    EXTRINSIC_METHOD_ORDER,
    evaluate_extrinsic_candidate,
    invert_transform,
    rank_candidates,
    solve_extrinsic,
    solve_planar_pnp_candidates,
    transform_from_record,
    transform_record,
    transform_residual,
)
from posetestbot.calibration import attempts as attempt_module
from posetestbot.calibration.candidates import _robot_ee_to_reference


def _fixture_observations(mode: str) -> tuple[list[dict], np.ndarray, np.ndarray]:
    camera_to_flange = pt.transform_from(
        pr.matrix_from_compact_axis_angle(np.array([0.08, -0.04, 0.03])),
        np.array([35.0, -20.0, 80.0]),
    )
    target_to_base = pt.transform_from(
        pr.matrix_from_compact_axis_angle(np.array([0.03, 0.02, -0.01])),
        np.array([100.0, 20.0, 400.0]),
    )
    camera_to_base = pt.transform_from(
        pr.matrix_from_compact_axis_angle(np.array([-0.1, 0.03, 0.06])),
        np.array([400.0, -100.0, 800.0]),
    )
    target_to_flange = pt.transform_from(
        pr.matrix_from_compact_axis_angle(np.array([0.02, -0.07, 0.04])),
        np.array([20.0, 10.0, 120.0]),
    )
    poses = [
        {
            "X": 50.0 + 15 * index,
            "Y": -40.0 + 8 * (index % 3),
            "Z": 500.0 + 10 * (index % 4),
            "A": -0.18 + 0.05 * (index % 4),
            "B": 0.12 - 0.04 * (index % 5),
            "C": -0.25 + 0.07 * index,
        }
        for index in range(10)
    ]
    observations = []
    for index, robot_pose in enumerate(poses):
        flange_to_base = _robot_ee_to_reference(robot_pose)
        if mode == "eye_in_hand":
            target_to_camera = (
                pt.invert_transform(camera_to_flange)
                @ pt.invert_transform(flange_to_base)
                @ target_to_base
            )
            expected_primary, expected_companion = camera_to_flange, target_to_base
        else:
            target_to_camera = (
                pt.invert_transform(camera_to_base)
                @ flange_to_base
                @ target_to_flange
            )
            expected_primary, expected_companion = camera_to_base, target_to_flange
        observations.append(
            {
                "frame_id": f"{index:06d}.png",
                "motion": f"pose_{index:02d}",
                "robot_ee_pose": robot_pose,
                "target_to_camera": transform_record(
                    target_to_camera,
                    from_frame="aruco_grid",
                    to_frame="camera",
                ),
                "mean_reprojection_error_px": 0.1,
            }
        )
    return observations, expected_primary, expected_companion


@pytest.mark.parametrize("mode", ["eye_in_hand", "eye_to_hand"])
@pytest.mark.parametrize("method", EXTRINSIC_METHOD_ORDER)
def test_all_hand_eye_and_robot_world_methods_recover_both_geometries(
    mode: str,
    method: str,
) -> None:
    observations, expected_primary, expected_companion = _fixture_observations(mode)

    primary, companion = solve_extrinsic(observations, mode=mode, method=method)

    assert transform_residual(primary, expected_primary)["translation_mm"] < 1e-5
    assert transform_residual(primary, expected_primary)["rotation_deg"] < 1e-4
    assert transform_residual(companion, expected_companion)["translation_mm"] < 1e-5
    assert transform_residual(companion, expected_companion)["rotation_deg"] < 1e-4


@pytest.mark.parametrize("mode", ["eye_in_hand", "eye_to_hand"])
def test_leave_one_pose_out_ranking_recovers_known_transform(mode: str) -> None:
    observations, expected, _companion = _fixture_observations(mode)

    candidate = evaluate_extrinsic_candidate(
        observations,
        mode=mode,
        pnp_method="ITERATIVE",
        extrinsic_method="park",
        sensor_key="realsense_d435:1",
    )

    assert candidate["status"] == "passing"
    assert candidate["inlier_count"] == len(observations)
    actual = np.asarray(candidate["primary_transform"]["matrix"])
    assert transform_residual(actual, expected)["translation_mm"] < 1e-5
    assert candidate["held_out_residuals"]["median_translation_mm"] < 1e-5


@pytest.mark.parametrize("mode", ["eye_in_hand", "eye_to_hand"])
def test_robust_closure_rejects_one_outlier_and_recovers_transform(mode: str) -> None:
    observations, expected, _companion = _fixture_observations(mode)
    corrupted = transform_from_record(observations[-1]["target_to_camera"])
    corrupted[:3, 3] += np.asarray([20.0, -10.0, 5.0])
    observations[-1]["target_to_camera"] = transform_record(
        corrupted,
        from_frame="aruco_grid",
        to_frame="camera",
    )

    candidate = evaluate_extrinsic_candidate(
        observations,
        mode=mode,
        pnp_method="ITERATIVE",
        extrinsic_method="park",
        sensor_key="realsense_d435:1",
    )

    assert candidate["status"] == "passing"
    assert candidate["inlier_count"] == 9
    assert candidate["outlier_count"] == 1
    assert candidate["outlier_ratio"] == pytest.approx(0.1)
    actual = np.asarray(candidate["primary_transform"]["matrix"])
    assert transform_residual(actual, expected)["translation_mm"] < 1e-5
    assert candidate["leave_one_pose_out"][-1]["validation_split"] == (
        "rejected_closure_outlier"
    )


@pytest.mark.parametrize("mode", ["eye_in_hand", "eye_to_hand"])
def test_target_camera_frame_inversion_is_not_silently_accepted(mode: str) -> None:
    observations, expected, _companion = _fixture_observations(mode)
    inverted = []
    for observation in observations:
        wrong_direction = invert_transform(
            transform_from_record(observation["target_to_camera"])
        )
        inverted.append(
            {
                **observation,
                "target_to_camera": transform_record(
                    wrong_direction,
                    from_frame="aruco_grid",
                    to_frame="camera",
                ),
            }
        )

    try:
        wrong_primary, _wrong_companion = solve_extrinsic(
            inverted,
            mode=mode,
            method="park",
        )
    except ValueError:
        return
    residual = transform_residual(wrong_primary, expected)
    assert residual["translation_mm"] > 1.0 or residual["rotation_deg"] > 1.0


def test_degenerate_motion_is_reported_as_candidate_failure() -> None:
    observations, _expected, _companion = _fixture_observations("eye_in_hand")
    same_pose = dict(observations[0]["robot_ee_pose"])
    for observation in observations:
        observation["robot_ee_pose"] = same_pose

    candidate = evaluate_extrinsic_candidate(
        observations,
        mode="eye_in_hand",
        pnp_method="IPPE",
        extrinsic_method="tsai",
        sensor_key="realsense_d435:1",
    )

    assert candidate["status"] == "error"
    assert "degenerate robot motion" in candidate["error"]


def test_planar_pnp_uses_shared_inliers_refines_and_retains_ippe_ambiguity() -> None:
    object_points = np.asarray(
        [[x, y, 0.0] for y in (0.0, 40.0, 80.0) for x in (0.0, 40.0, 80.0, 120.0)],
        dtype=float,
    )
    camera = np.asarray(
        [[700.0, 0.0, 320.0], [0.0, 705.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    image_points = cv2.projectPoints(
        object_points,
        np.asarray([0.1, -0.05, 0.03]),
        np.asarray([10.0, -20.0, 600.0]),
        camera,
        np.zeros(5),
    )[0].reshape(-1, 2)

    result = solve_planar_pnp_candidates(
        object_points,
        image_points,
        camera,
        np.zeros(5),
    )

    assert set(result["selected"]) == {"IPPE", "ITERATIVE", "SQPNP"}
    assert result["common_inlier_count"] == len(object_points)
    assert len([item for item in result["candidates"] if item["method"] == "IPPE"]) == 2
    assert all(
        item["common_inlier_indices"] == result["common_inlier_indices"]
        and item["refinement"] == "solvePnPRefineLM"
        for item in result["candidates"]
    )


def test_planar_pnp_rejects_non_cheiral_hypotheses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    object_points = np.asarray(
        [[0.0, 0.0, 0.0], [40.0, 0.0, 0.0], [40.0, 40.0, 0.0], [0.0, 40.0, 0.0]]
    )
    image_points = np.asarray(
        [[100.0, 100.0], [140.0, 100.0], [140.0, 140.0], [100.0, 140.0]]
    )
    camera = np.asarray(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
    )
    monkeypatch.setattr(
        "posetestbot.calibration.attempt_solver._common_pnp_inliers",
        lambda *_args: np.arange(4),
    )
    monkeypatch.setattr(
        cv2,
        "solvePnPGeneric",
        lambda *_args, **_kwargs: (
            True,
            (np.zeros((3, 1)),),
            (np.asarray([[0.0], [0.0], [-500.0]]),),
        ),
    )
    monkeypatch.setattr(
        cv2,
        "solvePnPRefineLM",
        lambda _objects, _images, _camera, _distortion, rvec, tvec: (
            rvec,
            tvec,
        ),
    )

    result = solve_planar_pnp_candidates(
        object_points,
        image_points,
        camera,
        np.zeros(5),
        methods=["IPPE"],
    )

    assert result["selected"] == {}
    assert any(
        item.get("reason") == "non_cheiral_pose" for item in result["failures"]
    )


def test_candidate_ranking_has_stable_method_tie_breaks() -> None:
    common = {
        "status": "passing",
        "score": 0.5,
        "mean_reprojection_error_px": 0.2,
        "inlier_count": 8,
        "sensor_key": "realsense_d435:1",
    }
    values = [
        {**common, "candidate_id": "sq", "pnp_method": "SQPNP", "extrinsic_method": "tsai"},
        {**common, "candidate_id": "it", "pnp_method": "ITERATIVE", "extrinsic_method": "tsai"},
        {**common, "candidate_id": "ip", "pnp_method": "IPPE", "extrinsic_method": "park"},
        {**common, "candidate_id": "ip-tsai", "pnp_method": "IPPE", "extrinsic_method": "tsai"},
    ]

    ranked = rank_candidates(values)

    assert [item["candidate_id"] for item in ranked] == ["ip-tsai", "ip", "it", "sq"]
    assert ranked[0]["recommended"] is True


def test_parent_attempt_runs_four_phases_writes_evidence_and_cannot_be_replayed(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations, _expected, _companion = _fixture_observations("eye_in_hand")
    run_root = tmp_path / "run"
    attempt_id = "a" * 32
    attempt_root = run_root / "processed" / "calibration" / attempt_id
    attempt_root.mkdir(parents=True)
    sensor_key = "realsense_d435:1"
    request_value = {
        "schema_version": "calibration_attempt_request.v1",
        "attempt_id": attempt_id,
        "run_root": run_root.as_posix(),
        "created_at": "2026-07-17T00:00:00+00:00",
        "mode": "eye_in_hand",
        "sensor_keys": [sensor_key],
        "sensors": [
            {
                "sensor_key": sensor_key,
                "sensor_name": "realsense_1",
                "sensor_type": "realsense_d435",
                "device_id": "1",
                "display_name": "D435",
            }
        ],
        "target_id": "target-1",
        "target": {"target_type": "aruco_grid", "unit": "mm"},
        "target_mounting": {
            "from": "aruco_grid",
            "to": "template_base",
            "state": "estimated",
        },
        "solver_policy": "auto_compare",
        "pnp_methods": ["ITERATIVE"],
        "extrinsic_methods": ["park"],
        "intrinsics_policy": "reuse_compatible_or_factory",
    }
    (attempt_root / "request.json").write_text(json.dumps(request_value))
    (attempt_root / "progress.json").write_text(
        json.dumps(attempt_module._initial_progress(attempt_id))
    )
    intrinsic = {
        "profile_id": "factory-1",
        "native": {
            "cam_K": [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0],
            "width": 640,
            "height": 480,
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "depth": {"scale_to_mm": 1.0},
    }
    monkeypatch.setattr(
        attempt_module,
        "_prepare_attempt_data",
        lambda *_args: ({sensor_key: attempt_root / "sensor"}, {sensor_key: intrinsic}),
    )
    monkeypatch.setattr(
        attempt_module,
        "_estimate_target_poses",
        lambda *_args: (
            {"sensors": []},
            {sensor_key: {"ITERATIVE": observations}},
        ),
    )

    ranking = attempt_module.run_calibration_attempt(run_root, attempt_id)

    assert ranking["status"] == "complete"
    assert ranking["recommended_camera_count"] == 1
    assert ranking["results"][0]["recommendation"]["extrinsic_method"] == "park"
    progress = json.loads((attempt_root / "progress.json").read_text())
    assert progress["status"] == "complete"
    assert [item["status"] for item in progress["phases"]] == [
        "complete",
        "complete",
        "complete",
        "complete",
    ]
    for filename in (
        "calibration_observations.json",
        "extrinsic_candidates.json",
        "ranking.json",
        "checks.json",
        "candidate_profiles.json",
    ):
        assert (attempt_root / filename).is_file()
    with pytest.raises(ValueError, match="immutable"):
        attempt_module.run_calibration_attempt(run_root, attempt_id)


def test_multi_camera_ranking_keeps_passing_camera_when_peer_fails(tmp_path) -> None:
    observations, _expected, _companion = _fixture_observations("eye_in_hand")
    passing = evaluate_extrinsic_candidate(
        observations,
        mode="eye_in_hand",
        pnp_method="ITERATIVE",
        extrinsic_method="park",
        sensor_key="realsense_d435:1",
    )
    failing_observations = [dict(item) for item in observations]
    same_pose = dict(failing_observations[0]["robot_ee_pose"])
    for observation in failing_observations:
        observation["robot_ee_pose"] = same_pose
    failing = evaluate_extrinsic_candidate(
        failing_observations,
        mode="eye_in_hand",
        pnp_method="ITERATIVE",
        extrinsic_method="park",
        sensor_key="oak_d_pro:2",
    )
    request_value = {
        "attempt_id": "a" * 32,
        "mode": "eye_in_hand",
        "sensor_keys": ["realsense_d435:1", "oak_d_pro:2"],
        "sensors": [
            {
                "sensor_key": "realsense_d435:1",
                "sensor_name": "realsense_1",
                "sensor_type": "realsense_d435",
                "device_id": "1",
                "display_name": "D435",
            },
            {
                "sensor_key": "oak_d_pro:2",
                "sensor_name": "luxonis_2",
                "sensor_type": "oak_d_pro",
                "device_id": "2",
                "display_name": "OAK",
            },
        ],
        "target_id": "target-1",
        "target_mounting": {
            "from": "aruco_grid",
            "to": "template_base",
            "state": "estimated",
        },
        "solver_policy": "auto_compare",
        "intrinsics_policy": "reuse_compatible_or_factory",
    }
    intrinsic = {
        "profile_id": "factory",
        "sensor_id": "1",
        "native": {
            "cam_K": [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0],
            "width": 640,
            "height": 480,
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "depth": {"scale_to_mm": 1.0},
    }

    ranking = attempt_module._validate_and_rank(
        tmp_path,
        request_value,
        [passing, failing],
        {
            "realsense_d435:1": intrinsic,
            "oak_d_pro:2": {**intrinsic, "sensor_id": "2"},
        },
    )

    assert ranking["status"] == "partial"
    assert ranking["recommended_camera_count"] == 1
    assert ranking["failed_camera_count"] == 1
    assert ranking["results"][0]["recommended_candidate_id"] == (
        "realsense_d435:1|ITERATIVE|park"
    )
    assert ranking["results"][1]["recommended_candidate_id"] is None
