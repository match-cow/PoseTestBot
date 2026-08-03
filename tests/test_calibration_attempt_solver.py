from __future__ import annotations

import cv2
import json
import numpy as np
import pytest
from pathlib import Path
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from types import SimpleNamespace

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
from posetestbot.calibration.intrinsics import (
    IntrinsicCalibrationError,
    factory_intrinsic_profile,
    write_intrinsic_profile_collection,
)
from posetestbot.calibration.targets import DEFAULT_TARGET_SPEC, opencv_grid_board
from posetestbot.io.artifacts import (
    INTRINSIC_CALIBRATION_PROFILES,
    INTRINSIC_COMPARISON,
)
from posetestbot.sensors.contracts import CameraIntrinsics
from posetestbot.sensors.frame_writer import write_legacy_camera_sidecars


@pytest.mark.parametrize(
    ("actual_sync_delta_ms", "should_fail"),
    [(0.0, False), (100.0, True), ("nan", True)],
)
def test_prepare_attempt_normalizes_paths_and_requires_zero_sync_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    actual_sync_delta_ms: float | str,
    should_fail: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    run_root = Path("run")
    attempt_root = run_root / "processed" / "calibration" / ("a" * 32)
    attempt_root.mkdir(parents=True)
    sensor_folder = run_root / "realsense_1"
    sensor_folder.mkdir(parents=True)
    (sensor_folder / "frame_metadata.jsonl").write_text(
        json.dumps(
            {
                "frame_id": "1000.png",
                "sensor_timestamp_ns": 10_000_000_000,
                "color_timestamp_domain": "global_time",
            }
        )
        + "\n"
    )
    (run_root / "raw_robot_ee_poses.json").write_text(
        json.dumps(
            {
                "0": {
                    "host_wall_timestamp_ns": 10_000_000_000,
                    "motion": "pose_0",
                    "pose": {
                        "X": 0.0,
                        "Y": 0.0,
                        "Z": 500.0,
                        "A": 0.0,
                        "B": 0.0,
                        "C": 0.0,
                    },
                }
            }
        )
    )
    output_folder = (
        attempt_root / "processed" / "preparation_synchronized" / "realsense_1"
    )
    report_path = output_folder / "sync_report.json"

    def fake_synchronize_run(*_args, **kwargs):
        assert kwargs["output_root"] == (
            attempt_root / "processed" / "preparation_synchronized"
        )
        assert kwargs["sync_delta"] == 0.0
        assert kwargs["timestamp_source"] == "sensor"
        assert kwargs["robot_timestamp_source"] == "host_wall"
        assert kwargs["max_nearest_pose_delta_ms"] == 150.0
        return [
            SimpleNamespace(
                sensor_folder=sensor_folder,
                output_folder=output_folder,
                report_path=report_path,
            )
        ]

    monkeypatch.setattr(
        attempt_module,
        "synchronize_run",
        fake_synchronize_run,
    )

    def fake_quality(
        root: Path,
        *,
        report_paths: list[Path],
        max_nearest_pose_delta_ms: float,
        require_timestamp_source: dict[str, str],
        require_robot_timestamp_source: dict[str, str],
    ) -> dict:
        assert root == run_root
        assert report_paths == [report_path.resolve()]
        assert max_nearest_pose_delta_ms == 150.0
        assert require_timestamp_source == {"realsense_1": "sensor"}
        assert require_robot_timestamp_source == {"realsense_1": "host_wall"}
        return {
            "overall_status": "ok",
            "sensors": [
                {
                    "sensor_name": "realsense_1",
                    "sync_delta_ms": actual_sync_delta_ms,
                }
            ],
            "checks": [
                {
                    "name": "sync_timestamp_source:realsense_1",
                    "status": "ok",
                },
                {
                    "name": "sync_robot_timestamp_source:realsense_1",
                    "status": "ok",
                },
                {
                    "name": "sync_nearest_pose_delta:realsense_1",
                    "status": "ok",
                },
            ],
        }

    monkeypatch.setattr(attempt_module, "build_sync_quality_report", fake_quality)
    monkeypatch.setattr(
        attempt_module,
        "_intrinsics_for_sensors",
        lambda *_args: ([], {}),
    )

    arguments = (
        run_root,
        attempt_root,
        {
            "sensors": [
                {
                    "sensor_key": "realsense_d435:1",
                    "folder": "realsense_1",
                    "sensor_type": "realsense_d435",
                    "robot_pose_path": "raw_robot_ee_poses.json",
                }
            ]
        },
    )
    if should_fail:
        with pytest.raises(ValueError, match="strict eye-in-hand policy"):
            attempt_module._prepare_attempt_data(*arguments)
        return

    synchronized, intrinsics = attempt_module._prepare_attempt_data(*arguments)

    assert synchronized == {"realsense_d435:1": output_folder.resolve()}
    assert intrinsics == {}


def test_authoritative_sync_uses_selected_offset_without_replacing_preparation_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    attempt_id = "a" * 32
    attempt_root = run_root / "processed" / "calibration" / attempt_id
    sensor_folder = run_root / "realsense_1"
    preparation_folder = (
        attempt_root / "processed" / "preparation_synchronized" / "realsense_1"
    )
    preparation_folder.mkdir(parents=True)
    sensor_folder.mkdir(parents=True)
    preparation_detection = preparation_folder / "aruco_detections.json"
    preparation_detection.write_text('{"retained": true}')

    request_value = {
        "attempt_id": attempt_id,
        "sensor_keys": ["realsense_d435:1"],
        "sensors": [
            {
                "sensor_key": "realsense_d435:1",
                "sensor_name": "realsense_1",
                "sensor_type": "realsense_d435",
                "device_id": "1",
                "folder": "realsense_1",
            }
        ],
    }
    time_offset_search = {
        "policy": "auto_offset",
        "sign_convention": attempt_module.time_offset_sign_convention(),
        "search": {"max_nearest_pose_delta_ms": 20.0},
        "sensors": [
            {
                "sensor_key": "realsense_d435:1",
                "status": "applied",
                "selected_robot_pose_time_offset_ms": 75.0,
                "selected_sync_delta_ms": -75.0,
            }
        ],
    }
    observations = {
        "realsense_d435:1": {
            "IPPE": [
                {
                    "observation_id": "old",
                    "frame_id": "000004.png",
                    "source_frame_id": "source-004.png",
                    "image_timestamp_ns": 1_000_000_000,
                    "motion": "old_motion",
                    "robot_ee_pose": {"X": 0.0},
                }
            ]
        }
    }
    timestamp_policy = {
        "schema_version": "calibration_timestamp_policy.v1",
        "per_sensor": {
            "realsense_d435:1": {
                "frame_timestamp_source": "sensor",
                "robot_timestamp_source": "host_wall",
            }
        },
    }
    monkeypatch.setattr(
        attempt_module,
        "_calibration_timestamp_preflight",
        lambda *_args: timestamp_policy,
    )

    final_folder = attempt_root / "processed" / "synchronized" / "realsense_1"
    report_path = final_folder / "sync_report.json"

    def fake_synchronize_run(*_args, **kwargs):
        assert kwargs["output_root"] == attempt_root / "processed" / "synchronized"
        assert kwargs["sync_delta"] == -75.0
        assert kwargs["copy_files"] is False
        assert kwargs["timestamp_source"] == "sensor"
        assert kwargs["robot_timestamp_source"] == "host_wall"
        assert kwargs["max_nearest_pose_delta_ms"] == 20.0
        final_folder.mkdir(parents=True)
        (final_folder / "match_robot_ee_poses.json").write_text(
            json.dumps(
                {
                    "000000.png": {
                        "source_frame_id": "source-004.png",
                        "image_timestamp_ns": 1_000_000_000,
                        "delayed_timestamp_ns": 1_075_000_000,
                        "motion": "motion_4",
                        "robot_ee_pose": {
                            "X": 1.0,
                            "Y": 2.0,
                            "Z": 3.0,
                            "A": 0.1,
                            "B": 0.2,
                            "C": 0.3,
                        },
                        "matched_robot_pose_index": 44,
                        "robot_timestamp_ns": 1_074_000_000,
                        "nearest_robot_delta_ns": -1_000_000,
                    }
                }
            )
        )
        return [
            SimpleNamespace(
                sensor_folder=sensor_folder,
                output_folder=final_folder,
                report_path=report_path,
            )
        ]

    monkeypatch.setattr(attempt_module, "synchronize_run", fake_synchronize_run)
    monkeypatch.setattr(
        attempt_module,
        "build_sync_quality_report",
        lambda *_args, **_kwargs: {
            "overall_status": "ok",
            "sensors": [{"sensor_name": "realsense_1", "sync_delta_ms": -75.0}],
            "checks": [
                {
                    "name": "sync_nearest_pose_delta:realsense_1",
                    "status": "ok",
                }
            ],
        },
    )

    synchronized, remapped = attempt_module._materialize_authoritative_synchronization(
        run_root,
        attempt_root,
        request_value,
        time_offset_search,
        observations,
    )

    assert synchronized == {"realsense_d435:1": final_folder.resolve()}
    assert json.loads(preparation_detection.read_text()) == {"retained": True}
    item = remapped["realsense_d435:1"]["IPPE"][0]
    assert item["frame_id"] == "000000.png"
    assert item["source_frame_id"] == "source-004.png"
    assert item["robot_pose_time_offset_ms"] == 75.0
    assert item["sync_delta_ms"] == -75.0
    assert item["timestamp_alignment"]["source"] == (
        f"processed/calibration/{attempt_id}/time_offset_search.json"
    )
    quality = json.loads((attempt_root / "sync_quality_report.json").read_text())
    policy = quality["calibration_attempt_policy"]
    assert policy["per_sensor"] == timestamp_policy["per_sensor"]
    assert policy["per_sensor_offsets"]["realsense_d435:1"] == {
        "robot_pose_time_offset_ms": 75.0,
        "sync_delta_ms": -75.0,
        "status": "applied",
    }


def test_auto_sync_execution_problem_warns_and_keeps_recorded_timing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    attempt_root = run_root / "processed" / "calibration" / ("a" * 32)
    attempt_root.mkdir(parents=True)
    sensor_key = "realsense_d435:1"
    request_value = {
        "attempt_id": "a" * 32,
        "mode": "eye_in_hand",
        "sensor_keys": [sensor_key],
        "sensors": [
            {
                "sensor_key": sensor_key,
                "sensor_name": "realsense_1",
                "sensor_type": "realsense_d435",
                "device_id": "1",
                "folder": "realsense_1",
            }
        ],
        "synchronization_policy": "auto_offset",
        "synchronization_search": attempt_module.time_offset_search_configuration(),
        "synchronization_implementation_revision": (
            attempt_module.TIME_OFFSET_IMPLEMENTATION_REVISION
        ),
    }

    monkeypatch.setattr(attempt_module, "load_robot_poses", lambda *_args: {})
    monkeypatch.setattr(
        attempt_module,
        "indexed_robot_poses",
        lambda *_args, **_kwargs: [
            {
                "pose_index": 0,
                "timestamp_ns": 1_000_000_000,
                "motion": "motion_0",
                "pose": {"X": 0.0, "Y": 0.0, "Z": 0.0, "A": 0.0, "B": 0.0, "C": 0.0},
            }
        ],
    )

    report, adjusted = attempt_module._estimate_and_apply_time_offsets(
        run_root,
        attempt_root,
        request_value,
        {sensor_key: {"ITERATIVE": []}},
    )

    assert report == json.loads((attempt_root / "time_offset_search.json").read_text())
    assert report["status"] == "complete"
    assert report["failed_sensor_keys"] == []
    assert report["warning_sensor_keys"] == [sensor_key]
    assert report["sensors"][0]["status"] == "kept_zero"
    assert report["sensors"][0]["warning_fallback_used"] is True
    assert report["sensors"][0]["checks"][0]["name"] == ("time_offset_search_execution")
    assert report["sensors"][0]["checks"][0]["status"] == "warning"
    assert adjusted == {sensor_key: {"ITERATIVE": []}}


def test_legacy_auto_sync_failure_requires_fresh_attempt_after_backend_restart(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    attempt_root = run_root / "processed" / "calibration" / ("a" * 32)
    attempt_root.mkdir(parents=True)
    sensor_key = "realsense_d435:1"
    search = attempt_module.time_offset_search_configuration()
    search["minimum_motion_count_per_cross_validation_fold"] = 3
    search.pop("maximum_leave_one_motion_out_search_adjusted_sign_p_value")
    request_value = {
        "attempt_id": "a" * 32,
        "mode": "eye_in_hand",
        "sensor_keys": [sensor_key],
        "sensors": [
            {
                "sensor_key": sensor_key,
                "sensor_name": "realsense_1",
                "sensor_type": "realsense_d435",
                "device_id": "1",
                "folder": "realsense_1",
            }
        ],
        "synchronization_policy": "auto_offset",
        "synchronization_search": search,
        "synchronization_implementation_revision": (
            attempt_module.TIME_OFFSET_LEGACY_IMPLEMENTATION_REVISION
        ),
    }

    with pytest.raises(
        ValueError,
        match=(
            "immutable attempt records legacy timing revision.*"
            "Restart the PoseTestBot backend and create a new attempt"
        ),
    ):
        attempt_module._estimate_and_apply_time_offsets(
            run_root,
            attempt_root,
            request_value,
            {sensor_key: {"ITERATIVE": []}},
        )


def test_realsense_calibration_timestamp_preflight_requires_global_time(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    sensor_folder = run_root / "realsense_1"
    sensor_folder.mkdir(parents=True)
    (sensor_folder / "frame_metadata.jsonl").write_text(
        json.dumps(
            {
                "frame_id": "1000.png",
                "sensor_timestamp_ns": 10_000_000_000,
                "color_timestamp_domain": "system_time",
            }
        )
        + "\n"
    )
    (run_root / "raw_robot_ee_poses.json").write_text(
        json.dumps(
            {
                "0": {
                    "host_wall_timestamp_ns": 10_000_000_000,
                    "motion": "pose_0",
                    "pose": {},
                }
            }
        )
    )
    sensors = [
        {
            "sensor_key": "realsense_d435:1",
            "sensor_type": "realsense_d435",
            "folder": "realsense_1",
            "robot_pose_path": "raw_robot_ee_poses.json",
        }
    ]

    with pytest.raises(ValueError, match="must all use global_time"):
        attempt_module._calibration_timestamp_preflight(run_root, sensors)


def test_prepare_attempt_rejects_mutated_per_sensor_timestamp_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    attempt_root = run_root / "processed" / "calibration" / ("a" * 32)
    sensor_folder = run_root / "realsense_1"
    attempt_root.mkdir(parents=True)
    sensor_folder.mkdir(parents=True)
    (sensor_folder / "frame_metadata.jsonl").write_text(
        json.dumps(
            {
                "frame_id": "1000.png",
                "sensor_timestamp_ns": 10_000_000_000,
                "color_timestamp_domain": "global_time",
            }
        )
        + "\n"
    )
    (run_root / "raw_robot_ee_poses.json").write_text(
        json.dumps(
            {
                "0": {
                    "host_wall_timestamp_ns": 10_000_000_000,
                    "motion": "pose_0",
                    "pose": {},
                }
            }
        )
    )
    sensors = [
        {
            "sensor_key": "realsense_d435:1",
            "sensor_type": "realsense_d435",
            "folder": "realsense_1",
            "robot_pose_path": "raw_robot_ee_poses.json",
        }
    ]
    recorded_policy = attempt_module._calibration_timestamp_preflight(run_root, sensors)
    recorded_policy["per_sensor"]["realsense_d435:1"]["frame_timestamp_source"] = (
        "host_received"
    )
    monkeypatch.setattr(
        attempt_module,
        "synchronize_run",
        lambda *_args, **_kwargs: pytest.fail(
            "synchronization must not run after timestamp-policy mutation"
        ),
    )

    with pytest.raises(
        ValueError,
        match="realsense_d435:1: frame_timestamp_source",
    ):
        attempt_module._prepare_attempt_data(
            run_root,
            attempt_root,
            {"sensors": sensors, "timestamp_policy": recorded_policy},
        )


def _intrinsic_sensor_fixture(folder: Path) -> dict:
    folder.mkdir(parents=True)
    write_legacy_camera_sidecars(
        folder,
        CameraIntrinsics(
            cam_k=(600.0, 0.0, 320.0, 0.0, 601.0, 240.0, 0.0, 0.0, 1.0),
            width=640,
            height=480,
            distortion=(0.01, -0.02, 0.001, -0.002, 0.003),
            depth_scale_to_mm=1.0,
            distortion_model="brown_conrady",
            projection_source="test_factory_color",
        ),
        include_distortion_in_cam_k=True,
    )
    return factory_intrinsic_profile(folder)


def _unsupported_intrinsic_profile(profile: dict, *, profile_id: str) -> dict:
    return {
        **profile,
        "profile_id": profile_id,
        "native": {
            **profile["native"],
            "distortion_model": "inverse_brown_conrady",
        },
        "rectified": None,
        "source": {
            **profile["source"],
            "opencv_projection_compatible": False,
            "rectification_available": False,
            "rectification_unavailable_reason": (
                "sdk_distortion_model_is_not_forward_opencv_compatible"
            ),
        },
    }


def _intrinsic_split_detections(
    count: int,
    *,
    reverse_mapping: bool = False,
) -> dict:
    _dictionary, board = opencv_grid_board(DEFAULT_TARGET_SPEC)
    ids = board.getIds().reshape(-1).astype(int).tolist()
    objects = [
        np.asarray(item, dtype=np.float32).reshape(4, 3)
        for item in board.getObjPoints()
    ]
    camera = np.asarray(
        [[600.0, 0.0, 320.0], [0.0, 605.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    cell_x = (-300.0, -125.0, 50.0)
    cell_y = (-250.0, -90.0, 160.0)
    frames = {}
    for index in range(count):
        unique_index = index // 3
        cell = unique_index % 9
        cycle = unique_index // 9
        row, column = divmod(cell, 3)
        rvec = np.asarray(
            [
                -0.28 + 0.035 * (cycle % 9),
                -0.22 + 0.045 * ((cycle * 2 + cell) % 9),
                -0.20 + 0.05 * ((cycle + cell * 2) % 9),
            ]
        )
        tvec = np.asarray(
            [
                cell_x[column] + 8.0 * (cycle % 5),
                cell_y[row] + 6.0 * ((cycle + cell) % 5),
                650.0 + 28.0 * (cycle % 7),
            ]
        )
        corners = [
            cv2.projectPoints(points, rvec, tvec, camera, np.zeros(5))[0]
            .reshape(4, 2)
            .tolist()
            for points in objects
        ]
        all_points = np.concatenate([np.asarray(item, dtype=float) for item in corners])
        name = f"{index:06d}.png"
        frames[name] = {
            "ids": ids,
            "corners": corners,
            "marker_count": len(ids),
            "image_centroid_px": all_points.mean(axis=0).tolist(),
        }
    items = list(frames.items())
    if reverse_mapping:
        items.reverse()
    return {
        "schema_version": "aruco_detections.v1",
        "image_size": [640, 480],
        "frames": dict(items),
    }


def test_intrinsic_split_caps_views_preserves_coverage_and_blocks_leakage() -> None:
    detections = _intrinsic_split_detections(240)

    training, holdout, split = attempt_module._intrinsic_detection_split(
        detections,
        DEFAULT_TARGET_SPEC,
    )
    shuffled_training, shuffled_holdout, shuffled_split = (
        attempt_module._intrinsic_detection_split(
            _intrinsic_split_detections(240, reverse_mapping=True),
            DEFAULT_TARGET_SPEC,
        )
    )

    assert list(training["frames"]) == split["training_views"]
    assert list(holdout["frames"]) == split["heldout_views"]
    assert len(training["frames"]) == 45
    assert len(holdout["frames"]) == 15
    assert set(training["frames"]).isdisjoint(holdout["frames"])
    assert len(split["training_coverage_cells"]) >= 6
    assert split["holdout_guard"] == {
        "requested_temporal_radius_views": 5,
        "effective_temporal_radius_views": 5,
        "requested_descriptor_distance": 1.0,
        "effective_descriptor_distance": 1.0,
        "relaxed_for_minimum_split_feasibility": False,
    }
    assert shuffled_split["training_views"] == split["training_views"]
    assert shuffled_split["heldout_views"] == split["heldout_views"]
    assert list(shuffled_training["frames"]) == split["training_views"]
    assert list(shuffled_holdout["frames"]) == split["heldout_views"]

    evidence = {item["frame"]: item for item in split["selected_view_evidence"]}
    scale = split["descriptor"]["normalized_corner_coordinate_scale"]
    for training_name in split["training_views"]:
        training_view = evidence[training_name]
        training_descriptor = (
            np.asarray(training_view["projected_board_corners_normalized"]) / scale
        )
        for holdout_name in split["heldout_views"]:
            holdout_view = evidence[holdout_name]
            holdout_descriptor = (
                np.asarray(holdout_view["projected_board_corners_normalized"]) / scale
            )
            assert (
                abs(
                    training_view["chronological_index"]
                    - holdout_view["chronological_index"]
                )
                > 5
            )
            assert np.linalg.norm(training_descriptor - holdout_descriptor) >= 1.0


def test_intrinsic_split_relaxes_guards_only_to_keep_small_dataset_feasible() -> None:
    detections = _intrinsic_split_detections(20)

    training, holdout, split = attempt_module._intrinsic_detection_split(
        detections,
        DEFAULT_TARGET_SPEC,
    )

    assert len(training["frames"]) == 15
    assert len(holdout["frames"]) == 5
    assert set(training["frames"]).isdisjoint(holdout["frames"])
    assert len(split["training_coverage_cells"]) >= 6
    assert split["holdout_guard"]["relaxed_for_minimum_split_feasibility"] is True
    assert split["selected_usable_view_count"] == 20
    assert split["omitted_usable_view_count"] == 0


def test_intrinsic_split_audits_duplicate_heavy_omissions() -> None:
    detections = _intrinsic_split_detections(180)

    _training, _holdout, split = attempt_module._intrinsic_detection_split(
        detections,
        DEFAULT_TARGET_SPEC,
    )

    assert split["strategy"] == ("deterministic_projective_maximin_guarded_views_v2")
    assert split["max_training_views"] == 45
    assert split["max_holdout_views"] == 15
    assert split["usable_view_count"] == 180
    assert split["omitted_usable_view_count"] == len(split["omitted_views"])
    assert split["omitted_correlated_view_count"] > 0
    assert {
        reason for item in split["omitted_views"] for reason in item["reasons"]
    } >= {"holdout_temporal_guard", "training_diversity_cap"}
    assert len(split["selected_view_evidence"]) == (
        split["training_usable_view_count"] + split["heldout_usable_view_count"]
    )


def test_reuse_intrinsics_rejects_incompatible_existing_and_uses_factory(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    attempt_root = run_root / "processed" / "calibration" / ("d" * 32)
    attempt_root.mkdir(parents=True)
    folder = attempt_root / "processed" / "synchronized" / "realsense_1"
    factory = _intrinsic_sensor_fixture(folder)
    existing = _unsupported_intrinsic_profile(
        factory,
        profile_id="stored-inverse-projection",
    )
    write_intrinsic_profile_collection(
        [existing],
        run_root / INTRINSIC_CALIBRATION_PROFILES,
    )

    profiles, by_sensor = attempt_module._intrinsics_for_sensors(
        run_root,
        attempt_root,
        {"realsense_d435:1": folder},
        {
            "attempt_id": "d" * 32,
            "intrinsics_policy": "reuse_compatible_or_factory",
            "target": {},
        },
    )

    assert profiles[0]["profile_id"] == factory["profile_id"]
    selected = by_sensor["realsense_d435:1"]
    assert selected["attempt_intrinsics_source"] == (
        "factory_capture_sidecars_existing_projection_unusable"
    )
    comparison = json.loads((attempt_root / INTRINSIC_COMPARISON).read_text())
    sensor = comparison["sensors"][0]
    assert sensor["status"] == "factory_selected"
    assert sensor["existing_projection"] == {
        "profile_id": "stored-inverse-projection",
        "opencv_projection_compatible": False,
        "distortion_model": "inverse_brown_conrady",
        "reason": "distortion_model_is_not_forward_opencv_compatible",
    }
    assert sensor["factory_projection"]["opencv_projection_compatible"] is True
    assert sensor["unusable_projection"] is None
    assert {item["profile_id"] for item in sensor["candidates"]} == {
        factory["profile_id"],
        "stored-inverse-projection",
    }


def test_reuse_intrinsics_preserves_evidence_and_fails_when_all_unusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    attempt_root = run_root / "processed" / "calibration" / ("e" * 32)
    attempt_root.mkdir(parents=True)
    folder = attempt_root / "processed" / "synchronized" / "realsense_1"
    factory = _intrinsic_sensor_fixture(folder)
    existing = _unsupported_intrinsic_profile(
        factory,
        profile_id="stored-inverse-projection",
    )
    captured = _unsupported_intrinsic_profile(
        factory,
        profile_id="captured-inverse-projection",
    )
    write_intrinsic_profile_collection(
        [existing],
        run_root / INTRINSIC_CALIBRATION_PROFILES,
    )
    monkeypatch.setattr(
        attempt_module,
        "factory_intrinsic_profile",
        lambda *_args: captured,
    )

    with pytest.raises(
        ValueError,
        match="No OpenCV-compatible intrinsic projection.*realsense_d435:1",
    ):
        attempt_module._intrinsics_for_sensors(
            run_root,
            attempt_root,
            {"realsense_d435:1": folder},
            {
                "attempt_id": "e" * 32,
                "intrinsics_policy": "reuse_compatible_or_factory",
                "target": {},
            },
        )

    comparison = json.loads((attempt_root / INTRINSIC_COMPARISON).read_text())
    sensor = comparison["sensors"][0]
    assert sensor["status"] == "unusable"
    assert sensor["selected_profile_id"] is None
    assert sensor["unusable_projection"] == {
        "reason": "no_opencv_compatible_intrinsic_projection",
        "factory": {
            "profile_id": "captured-inverse-projection",
            "opencv_projection_compatible": False,
            "distortion_model": "inverse_brown_conrady",
            "reason": "distortion_model_is_not_forward_opencv_compatible",
        },
        "existing": {
            "profile_id": "stored-inverse-projection",
            "opencv_projection_compatible": False,
            "distortion_model": "inverse_brown_conrady",
            "reason": "distortion_model_is_not_forward_opencv_compatible",
        },
        "selected": None,
    }


@pytest.mark.parametrize("factory_compatible", [False, True])
def test_attempt_intrinsic_comparison_keeps_compatible_factory_as_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_compatible: bool,
) -> None:
    run_root = tmp_path / "run"
    attempt_root = run_root / "processed" / "calibration" / ("b" * 32)
    attempt_root.mkdir(parents=True)
    folder = attempt_root / "processed" / "synchronized" / "realsense_1"
    factory = _intrinsic_sensor_fixture(folder)
    manual = {
        **factory,
        "profile_id": "1_640x480_normal_aruco",
        "native": {
            **factory["native"],
            "cam_K": [
                602.0,
                0.0,
                319.0,
                0.0,
                604.0,
                241.5,
                0.0,
                0.0,
                1.0,
            ],
            "distortion": [0.02, -0.01, 0.002, -0.001, 0.004],
        },
        "source": {"mode": "calibrate", "algorithm": "cv2.calibrateCameraExtended"},
        "quality": {
            "status": "accepted",
            "accepted_view_count": 18,
            "coverage_cells": [0, 1, 2, 3, 4, 5],
            "rms_reprojection_error_px": 0.8,
            "rejected_views": [],
        },
    }
    unusable_factory = {
        **factory,
        "native": {
            **factory["native"],
            "distortion_model": "inverse_brown_conrady",
        },
        "rectified": None,
        "source": {
            **factory["source"],
            "opencv_projection_compatible": False,
            "rectification_available": False,
            "rectification_unavailable_reason": (
                "sdk_distortion_model_is_not_forward_opencv_compatible"
            ),
        },
    }
    monkeypatch.setattr(
        attempt_module,
        "factory_intrinsic_profile",
        lambda *_args: factory if factory_compatible else unusable_factory,
    )
    monkeypatch.setattr(
        attempt_module,
        "detect_sensor_folder",
        lambda *_args, **_kwargs: {"frames": {}},
    )
    monkeypatch.setattr(
        attempt_module,
        "calibrate_intrinsic_profile",
        lambda *_args, **_kwargs: manual,
    )
    monkeypatch.setattr(
        attempt_module,
        "_intrinsic_detection_split",
        lambda *_args: (
            {"frames": {}},
            {"frames": {}},
            {
                "training_usable_view_count": 15,
                "heldout_usable_view_count": 5,
            },
        ),
    )
    monkeypatch.setattr(
        attempt_module,
        "_intrinsic_holdout_evaluation",
        lambda profile, *_args: {
            "status": "accepted",
            "comparable": True,
            "rms_reprojection_error_px": (
                0.8 if profile["profile_id"] == manual["profile_id"] else 1.0
            ),
        },
    )
    monkeypatch.setattr(
        attempt_module,
        "_manual_intrinsic_plausibility",
        lambda *_args: {"status": "accepted"},
    )

    profiles, by_sensor = attempt_module._intrinsics_for_sensors(
        run_root,
        attempt_root,
        {"realsense_d435:1": folder},
        {
            "attempt_id": "b" * 32,
            "intrinsics_policy": "compare_factory_opencv",
            "target": {},
        },
    )

    expected = factory if factory_compatible else manual
    assert profiles[0]["profile_id"] == expected["profile_id"]
    assert by_sensor["realsense_d435:1"]["attempt_intrinsics_source"] == (
        "factory_compatible_default_comparison_only"
        if factory_compatible
        else "opencv_manual_factory_projection_unavailable"
    )
    comparison = json.loads((attempt_root / INTRINSIC_COMPARISON).read_text())
    sensor = comparison["sensors"][0]
    assert sensor["status"] == (
        "factory_selected" if factory_compatible else "manual_selected"
    )
    assert sensor["factory_profile_id"] == factory["profile_id"]
    assert sensor["manual_profile_id"] == manual["profile_id"]
    assert len(sensor["candidates"]) == 2
    assert sensor["deltas"]["focal_length_delta_px"] == [2.0, 3.0]


def test_attempt_intrinsic_comparison_preserves_manual_failure_and_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    attempt_root = run_root / "processed" / "calibration" / ("c" * 32)
    attempt_root.mkdir(parents=True)
    folder = attempt_root / "processed" / "synchronized" / "realsense_1"
    factory = _intrinsic_sensor_fixture(folder)
    monkeypatch.setattr(
        attempt_module,
        "detect_sensor_folder",
        lambda *_args, **_kwargs: {"frames": {}},
    )

    def reject_manual(*_args, **_kwargs):
        report = {
            "status": "rejected",
            "reason": "coverage 2/9 is below 6/9",
            "accepted_views": ["000001.png"],
        }
        raise IntrinsicCalibrationError(report["reason"], report)

    monkeypatch.setattr(
        attempt_module,
        "calibrate_intrinsic_profile",
        reject_manual,
    )
    monkeypatch.setattr(
        attempt_module,
        "_intrinsic_detection_split",
        lambda *_args: (
            {"frames": {}},
            {"frames": {}},
            {
                "training_usable_view_count": 15,
                "heldout_usable_view_count": 5,
            },
        ),
    )

    profiles, _by_sensor = attempt_module._intrinsics_for_sensors(
        run_root,
        attempt_root,
        {"realsense_d435:1": folder},
        {
            "attempt_id": "c" * 32,
            "intrinsics_policy": "compare_factory_opencv",
            "target": {},
        },
    )

    assert profiles[0]["profile_id"] == factory["profile_id"]
    comparison = json.loads((attempt_root / INTRINSIC_COMPARISON).read_text())
    sensor = comparison["sensors"][0]
    assert sensor["status"] == "factory_selected"
    assert sensor["manual_profile_id"] is None
    assert sensor["manual_failure"]["quality"]["reason"] == (
        "coverage 2/9 is below 6/9"
    )
    assert sensor["candidates"][0]["profile_id"] == factory["profile_id"]


def test_manual_intrinsic_plausibility_rejects_absurd_parameters(
    tmp_path: Path,
) -> None:
    factory = _intrinsic_sensor_fixture(tmp_path / "realsense_1")
    manual = {
        **factory,
        "native": {
            **factory["native"],
            "cam_K": [
                50.0,
                0.0,
                -100.0,
                0.0,
                3000.0,
                900.0,
                0.0,
                0.0,
                1.0,
            ],
            "distortion": [2.0, -4.0, 0.2, -0.2, 8.0],
        },
    }

    result = attempt_module._manual_intrinsic_plausibility(factory, manual)

    assert result["status"] == "rejected"
    assert result["checks"]["principal_point_inside_image"] is False
    assert result["checks"]["distortion_magnitude"] is False


def _fixture_observations(
    mode: str,
    *,
    count: int = 10,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
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
        for index in range(count)
    ]
    observations = []
    for index, robot_pose in enumerate(poses):
        coverage_row, coverage_column = divmod(index % 9, 3)
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
                pt.invert_transform(camera_to_base) @ flange_to_base @ target_to_flange
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
                "image_coverage_cell": index % 9,
                "image_centroid_px": [
                    (coverage_column + 0.5) * 640.0 / 3.0,
                    (coverage_row + 0.5) * 480.0 / 3.0,
                ],
                "image_size": [640, 480],
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


def test_motion_balanced_fit_still_validates_every_accepted_frame() -> None:
    observations, _expected, _companion = _fixture_observations("eye_in_hand", count=40)
    for index, observation in enumerate(observations):
        observation["motion"] = f"motion_{index // 10}"
    # Five evenly spaced solver samples per ten-frame motion are indices
    # 0, 2, 4, 7 and 9. Corrupt only discarded frames in one motion.
    for index in (1, 3, 5, 6, 8):
        corrupted = transform_from_record(observations[index]["target_to_camera"])
        corrupted[:3, 3] += np.asarray([30.0, 0.0, 0.0])
        observations[index]["target_to_camera"] = transform_record(
            corrupted,
            from_frame="aruco_grid",
            to_frame="camera",
        )

    candidate = evaluate_extrinsic_candidate(
        observations,
        mode="eye_in_hand",
        pnp_method="ITERATIVE",
        extrinsic_method="park",
        sensor_key="realsense_d435:1",
    )

    assert candidate["status"] == "failed"
    assert candidate["observation_count"] == 40
    assert candidate["solver_observation_count"] == 20
    assert candidate["full_input_validation"][
        "max_repeated_motion_outlier_ratio"
    ] == pytest.approx(0.5)


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


def test_single_axis_robot_rotation_is_not_observable() -> None:
    observations, _expected, _companion = _fixture_observations("eye_in_hand")
    for index, observation in enumerate(observations):
        observation["robot_ee_pose"] = {
            "X": float(index * 25),
            "Y": float((index % 3) * 20),
            "Z": 500.0,
            "A": 0.0,
            "B": 0.0,
            "C": float(index) * 0.08,
        }

    candidate = evaluate_extrinsic_candidate(
        observations,
        mode="eye_in_hand",
        pnp_method="ITERATIVE",
        extrinsic_method="park",
        sensor_key="realsense_d435:1",
    )

    assert candidate["status"] == "error"
    assert "rotation-axis second/first singular ratio" in candidate["error"]


def test_attempt_quality_gates_require_fifteen_views_and_six_coverage_cells() -> None:
    observations, _expected, _companion = _fixture_observations("eye_in_hand")

    too_few = evaluate_extrinsic_candidate(
        observations,
        mode="eye_in_hand",
        pnp_method="ITERATIVE",
        extrinsic_method="park",
        sensor_key="realsense_d435:1",
        min_accepted_views=15,
        min_coverage_cells=6,
    )

    assert too_few["status"] == "error"
    assert "accepted view count 10 is below required 15" in too_few["error"]

    many, _expected, _companion = _fixture_observations("eye_in_hand", count=18)
    for observation in many:
        observation["image_coverage_cell"] = 4
    poor_coverage = evaluate_extrinsic_candidate(
        many,
        mode="eye_in_hand",
        pnp_method="ITERATIVE",
        extrinsic_method="park",
        sensor_key="realsense_d435:1",
        min_accepted_views=15,
        min_coverage_cells=6,
    )

    assert poor_coverage["status"] == "error"
    assert "image-centroid coverage 1/9 is below required 6/9" in poor_coverage["error"]


def test_continuous_image_coverage_replaces_partition_dependent_cell_veto() -> None:
    observations, _expected, _companion = _fixture_observations(
        "eye_in_hand",
        count=18,
    )
    corners = (
        (200.0, 200.0),
        (750.0, 200.0),
        (200.0, 700.0),
        (750.0, 700.0),
    )
    for index, observation in enumerate(observations):
        observation.update(
            {
                "image_coverage_cell": 4,
                "image_centroid_px": list(corners[index % len(corners)]),
                "image_size": [1000, 1000],
            }
        )

    candidate = evaluate_extrinsic_candidate(
        observations,
        mode="eye_in_hand",
        pnp_method="ITERATIVE",
        extrinsic_method="park",
        sensor_key="realsense_d435:1",
        min_accepted_views=15,
        min_coverage_cells=6,
        image_coverage_tail_support_views=5,
        min_image_centroid_x_span_ratio=0.45,
        min_image_centroid_y_span_ratio=0.35,
        min_image_centroid_hull_area_ratio=0.10,
    )

    assert candidate["status"] == "passing"
    checks = {item["name"]: item for item in candidate["checks"]}
    assert checks["image_centroid_coverage"]["status"] == "warning"
    assert checks["continuous_image_centroid_coverage"]["status"] == "ok"
    evidence = candidate["observation_quality"]["continuous_image_coverage"]
    assert evidence["tail_support_views"] == 5
    assert evidence["supported_span_ratio_xy"] == pytest.approx([0.55, 0.5])
    assert evidence["supported_convex_hull_area_ratio"] == pytest.approx(0.275)


def test_continuous_image_coverage_rejects_wide_but_collinear_views() -> None:
    observations, _expected, _companion = _fixture_observations(
        "eye_in_hand",
        count=18,
    )
    for index, observation in enumerate(observations):
        coordinate = 200.0 if index % 2 == 0 else 800.0
        observation.update(
            {
                "image_coverage_cell": 4,
                "image_centroid_px": [coordinate, coordinate],
                "image_size": [1000, 1000],
            }
        )

    candidate = evaluate_extrinsic_candidate(
        observations,
        mode="eye_in_hand",
        pnp_method="ITERATIVE",
        extrinsic_method="park",
        sensor_key="realsense_d435:1",
        min_accepted_views=15,
        min_coverage_cells=6,
        image_coverage_tail_support_views=5,
        min_image_centroid_x_span_ratio=0.45,
        min_image_centroid_y_span_ratio=0.35,
        min_image_centroid_hull_area_ratio=0.10,
    )

    assert candidate["status"] == "error"
    assert "hull area 0.000/0.100" in candidate["error"]


def test_attempt_quality_gate_requires_distinct_motion_labels() -> None:
    observations, _expected, _companion = _fixture_observations("eye_in_hand", count=18)
    for observation in observations:
        observation["motion"] = "circ_1"

    candidate = evaluate_extrinsic_candidate(
        observations,
        mode="eye_in_hand",
        pnp_method="ITERATIVE",
        extrinsic_method="park",
        sensor_key="realsense_d435:1",
        min_accepted_views=15,
        min_coverage_cells=6,
        min_motion_poses=4,
        min_translation_span_mm=20.0,
        min_rotation_span_deg=5.0,
    )

    assert candidate["status"] == "error"
    assert "requires at least 4 distinct motion poses; found 1" in candidate["error"]


def _coplanar_pnp_ransac_regression_fixture() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    object_points = np.asarray(
        [
            [column * 55.275 + x, row * 55.0 + y, 0.0]
            for row in range(5)
            for column in range(7)
            for x, y in (
                (0.0, 0.0),
                (45.225, 0.0),
                (45.225, 45.0),
                (0.0, 45.0),
            )
        ],
        dtype=np.float64,
    )
    camera = np.asarray(
        [
            [903.128737, 0.0, 632.458263],
            [0.0, 914.254590, 385.141689],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.asarray(
        [0.124566, -0.378566, 0.002004, -0.000095, 0.288739],
        dtype=np.float64,
    )
    projected = cv2.projectPoints(
        object_points,
        np.asarray([-0.812730, 0.342254, 0.149498]),
        np.asarray([122.142598, -10.936042, 797.931882]),
        camera,
        distortion,
    )[0].reshape(-1, 2)
    image_points = projected + np.random.default_rng(0).normal(
        0.0,
        0.65,
        projected.shape,
    )
    return object_points, image_points, camera, distortion


def test_planar_pnp_avoids_sparse_minimal_pnp_consensus() -> None:
    object_points, image_points, camera, distortion = (
        _coplanar_pnp_ransac_regression_fixture()
    )
    direct_success, direct_rvec, direct_tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    direct_projection = cv2.projectPoints(
        object_points,
        direct_rvec,
        direct_tvec,
        camera,
        distortion,
    )[0].reshape(-1, 2)
    legacy_success, _rvec, _tvec, legacy_inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        camera,
        distortion,
        iterationsCount=200,
        reprojectionError=4.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    assert direct_success is True
    assert np.mean(np.linalg.norm(direct_projection - image_points, axis=1)) < 1.0
    assert legacy_success is True
    assert legacy_inliers is not None
    assert len(legacy_inliers) < len(object_points) / 2

    result = solve_planar_pnp_candidates(
        object_points,
        image_points,
        camera,
        distortion,
        methods=["ITERATIVE"],
    )

    assert result["common_inlier_count"] == len(object_points)
    assert set(result["selected"]) == {"ITERATIVE"}
    assert result["selected"]["ITERATIVE"]["all_point_mean_reprojection_error_px"] < 1.0


def test_planar_pnp_homography_consensus_rejects_gross_outliers() -> None:
    object_points, image_points, camera, distortion = (
        _coplanar_pnp_ransac_regression_fixture()
    )
    outlier_indices = np.arange(0, len(object_points), 7)
    image_points[outlier_indices] += np.asarray([80.0, -60.0])

    result = solve_planar_pnp_candidates(
        object_points,
        image_points,
        camera,
        distortion,
        methods=["ITERATIVE"],
    )

    expected_inliers = sorted(set(range(len(object_points))) - set(outlier_indices))
    assert result["common_inlier_indices"] == expected_inliers
    assert result["common_inlier_count"] == len(expected_inliers)
    assert result["selected"] == {}
    assert any(
        item.get("reason") == "whole_board_reprojection_error"
        for item in result["failures"]
    )


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


def test_planar_pnp_requires_marker_support_across_both_grid_axes() -> None:
    object_points = np.asarray(
        [
            [column * 60.0 + x, row * 60.0 + y, 0.0]
            for row, column in ((0, 0), (0, 1), (1, 0), (1, 1))
            for x, y in ((0.0, 0.0), (40.0, 0.0), (40.0, 40.0), (0.0, 40.0))
        ],
        dtype=float,
    )
    marker_ids = np.repeat(np.arange(4), 4)
    grid_indices = np.repeat(
        np.asarray(((0, 0), (0, 1), (1, 0), (1, 1))),
        4,
        axis=0,
    )
    camera = np.asarray(
        [[700.0, 0.0, 320.0], [0.0, 705.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    image_points = cv2.projectPoints(
        object_points,
        np.asarray([0.1, -0.05, 0.03]),
        np.asarray([10.0, -20.0, 700.0]),
        camera,
        np.zeros(5),
    )[0].reshape(-1, 2)

    accepted = solve_planar_pnp_candidates(
        object_points,
        image_points,
        camera,
        np.zeros(5),
        methods=["ITERATIVE"],
        point_marker_ids=marker_ids,
        point_grid_indices=grid_indices,
    )
    one_row = solve_planar_pnp_candidates(
        object_points,
        image_points,
        camera,
        np.zeros(5),
        methods=["ITERATIVE"],
        point_marker_ids=marker_ids,
        point_grid_indices=np.column_stack(
            (np.zeros(len(grid_indices), dtype=int), grid_indices[:, 1])
        ),
    )

    assert set(accepted["selected"]) == {"ITERATIVE"}
    assert accepted["supported_marker_count"] == 4
    assert accepted["supported_grid_rows"] == [0, 1]
    assert one_row["selected"] == {}
    assert one_row["failures"][0]["reason"] == ("insufficient_spatial_pnp_support")


def test_planar_pnp_rejects_non_cheiral_hypotheses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    object_points = np.asarray(
        [[0.0, 0.0, 0.0], [40.0, 0.0, 0.0], [40.0, 40.0, 0.0], [0.0, 40.0, 0.0]]
    )
    image_points = np.asarray(
        [[100.0, 100.0], [140.0, 100.0], [140.0, 140.0], [100.0, 140.0]]
    )
    camera = np.asarray([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
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
        min_common_inliers=4,
    )

    assert result["selected"] == {}
    assert any(item.get("reason") == "non_cheiral_pose" for item in result["failures"])


def test_planar_pnp_rejects_tiny_support_seen_with_wrong_target_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    object_points = np.asarray(
        [[float(index % 14), float(index // 14), 0.0] for index in range(140)]
    )
    image_points = object_points[:, :2] * 10.0 + 100.0
    monkeypatch.setattr(
        "posetestbot.calibration.attempt_solver._common_pnp_inliers",
        lambda *_args: np.arange(8),
    )

    result = solve_planar_pnp_candidates(
        object_points,
        image_points,
        np.asarray([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]),
        np.zeros(5),
        methods=["ITERATIVE"],
    )

    assert result["selected"] == {}
    assert result["correspondence_count"] == 140
    assert result["common_inlier_count"] == 8
    assert result["common_inlier_ratio"] == pytest.approx(8 / 140)
    assert result["failures"][0]["reason"] == "insufficient_common_pnp_support"


def test_planar_pnp_rejects_high_whole_board_error_even_with_good_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    object_points = np.asarray(
        [
            [x, y, 0.0]
            for y in (0.0, 40.0, 80.0, 120.0)
            for x in (0.0, 40.0, 80.0, 120.0)
        ],
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
    image_points[-4:] += np.asarray([80.0, -60.0])
    monkeypatch.setattr(
        "posetestbot.calibration.attempt_solver._common_pnp_inliers",
        lambda *_args: np.arange(12),
    )

    result = solve_planar_pnp_candidates(
        object_points,
        image_points,
        camera,
        np.zeros(5),
        methods=["ITERATIVE"],
    )

    assert result["common_inlier_ratio"] == pytest.approx(0.75)
    assert result["selected"] == {}
    assert result["candidates"][0]["quality_status"] == "rejected"
    assert result["candidates"][0]["all_point_mean_reprojection_error_px"] > 3.0
    assert any(
        item.get("reason") == "whole_board_reprojection_error"
        for item in result["failures"]
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
        {
            **common,
            "candidate_id": "sq",
            "pnp_method": "SQPNP",
            "extrinsic_method": "tsai",
        },
        {
            **common,
            "candidate_id": "it",
            "pnp_method": "ITERATIVE",
            "extrinsic_method": "tsai",
        },
        {
            **common,
            "candidate_id": "ip",
            "pnp_method": "IPPE",
            "extrinsic_method": "park",
        },
        {
            **common,
            "candidate_id": "ip-tsai",
            "pnp_method": "IPPE",
            "extrinsic_method": "tsai",
        },
    ]

    ranked = rank_candidates(values)

    assert [item["candidate_id"] for item in ranked] == ["ip-tsai", "ip", "it", "sq"]
    assert ranked[0]["recommended"] is True


def test_parent_attempt_runs_five_phases_writes_evidence_and_cannot_be_replayed(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations, _expected, _companion = _fixture_observations("eye_in_hand", count=18)
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
        "synchronization_policy": "fixed_zero",
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
            "distortion_model": "inverse_brown_conrady",
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
    time_offset_search = {
        "schema_version": "calibration_time_offset_search.v1",
        "policy": "fixed_zero",
        "status": "complete",
        "sign_convention": attempt_module.time_offset_sign_convention(),
        "sensors": [
            attempt_module.fixed_zero_sensor_result(
                sensor_key=sensor_key,
                observation_count=len(observations),
            )
        ],
    }

    def fake_time_offsets(*_args):
        attempt_module.atomic_write_json(
            attempt_root / "time_offset_search.json",
            time_offset_search,
        )
        return time_offset_search, {sensor_key: {"ITERATIVE": observations}}

    monkeypatch.setattr(
        attempt_module,
        "_estimate_and_apply_time_offsets",
        fake_time_offsets,
    )
    monkeypatch.setattr(
        attempt_module,
        "_materialize_authoritative_synchronization",
        lambda *_args: (
            {sensor_key: attempt_root / "sensor"},
            {sensor_key: {"ITERATIVE": observations}},
        ),
    )

    monkeypatch.chdir(tmp_path)
    ranking = attempt_module.run_calibration_attempt(Path("run"), attempt_id)

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
        "complete",
    ]
    for filename in (
        "calibration_observations.json",
        "extrinsic_candidates.json",
        "ranking.json",
        "checks.json",
        "candidate_profiles.json",
        "time_offset_search.json",
    ):
        assert (attempt_root / filename).is_file()
    candidate_profiles = json.loads(
        (attempt_root / "candidate_profiles.json").read_text()
    )
    candidate_profile = candidate_profiles["profiles"][0]
    assert candidate_profile["sync_delta_ms"] == 0.0
    assert candidate_profile["intrinsics"]["native"]["distortion_model"] == (
        "inverse_brown_conrady"
    )
    assert candidate_profile["intrinsics"]["rectified"] is not None
    assert candidate_profile["intrinsics"]["rectified"]["distortion"] == [0.0] * 5
    assert candidate_profile["metadata"]["synchronization"] == {
        "policy": "fixed_zero",
        "status": "fixed_zero",
        "robot_pose_time_offset_ms": 0.0,
        "sync_delta_ms": 0.0,
        "source": (
            "processed/calibration/"
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/time_offset_search.json"
        ),
        "timestamp_source": "sensor",
        "frame_timestamp_source": "sensor",
        "robot_timestamp_source": "host_wall",
        "required_frame_timestamp_domain": "global_time",
        "timestamp_fallback_allowed": False,
        "max_nearest_pose_delta_ms": 150.0,
        "warning_nearest_pose_delta_ms": 20.0,
        "warning_fallback_used": False,
        "historical_per_sensor_offsets_allowed": False,
        "auto_estimated_per_sensor_offset": False,
        "sensor_key": sensor_key,
        "quality_report": (
            f"processed/calibration/{attempt_id}/sync_quality_report.json"
        ),
    }
    with pytest.raises(ValueError, match="immutable"):
        attempt_module.run_calibration_attempt(Path("run"), attempt_id)


def _multi_camera_candidate_variant(
    base: dict,
    *,
    sensor_key: str,
    pnp_method: str,
    extrinsic_method: str,
    score: float,
    companion_translation_offset_mm: float,
) -> dict:
    candidate = json.loads(json.dumps(base))
    candidate.update(
        {
            "candidate_id": f"{sensor_key}|{pnp_method}|{extrinsic_method}",
            "sensor_key": sensor_key,
            "pnp_method": pnp_method,
            "extrinsic_method": extrinsic_method,
            "algorithms": [pnp_method, extrinsic_method],
            "score": score,
        }
    )
    companion = transform_from_record(candidate["companion_transform"])
    companion[0, 3] += companion_translation_offset_mm
    candidate["companion_transform"] = transform_record(
        companion,
        from_frame="aruco_grid",
        to_frame="template_base",
    )
    return candidate


def _multi_camera_request() -> dict:
    return {
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
        "pnp_methods": ["IPPE", "ITERATIVE"],
        "extrinsic_methods": ["park"],
        "intrinsics_policy": "reuse_compatible_or_factory",
    }


def _multi_camera_intrinsics() -> dict[str, dict]:
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
    return {
        "realsense_d435:1": intrinsic,
        "oak_d_pro:2": {**intrinsic, "sensor_id": "2"},
    }


def test_multi_camera_ranking_selects_best_common_bundle_and_records_evidence(
    tmp_path: Path,
) -> None:
    observations, _expected, _companion = _fixture_observations("eye_in_hand")
    base = evaluate_extrinsic_candidate(
        observations,
        mode="eye_in_hand",
        pnp_method="ITERATIVE",
        extrinsic_method="park",
        sensor_key="realsense_d435:1",
    )
    candidates = [
        _multi_camera_candidate_variant(
            base,
            sensor_key="realsense_d435:1",
            pnp_method="IPPE",
            extrinsic_method="park",
            score=0.05,
            companion_translation_offset_mm=0.0,
        ),
        _multi_camera_candidate_variant(
            base,
            sensor_key="realsense_d435:1",
            pnp_method="ITERATIVE",
            extrinsic_method="park",
            score=0.20,
            companion_translation_offset_mm=0.0,
        ),
        _multi_camera_candidate_variant(
            base,
            sensor_key="oak_d_pro:2",
            pnp_method="IPPE",
            extrinsic_method="park",
            score=0.60,
            companion_translation_offset_mm=2.0,
        ),
        _multi_camera_candidate_variant(
            base,
            sensor_key="oak_d_pro:2",
            pnp_method="ITERATIVE",
            extrinsic_method="park",
            score=0.21,
            companion_translation_offset_mm=6.0,
        ),
    ]
    request_value = _multi_camera_request()

    ranking = attempt_module._validate_and_rank(
        tmp_path,
        request_value,
        candidates,
        _multi_camera_intrinsics(),
    )

    assert ranking["status"] == "complete"
    assert ranking["recommended_camera_count"] == 2
    assert {
        result["sensor_key"]: result["recommendation"]["pnp_method"]
        for result in ranking["results"]
    } == {
        "realsense_d435:1": "ITERATIVE",
        "oak_d_pro:2": "ITERATIVE",
    }
    consistency = ranking["multi_camera_consistency"]
    assert consistency["status"] == "passing"
    assert consistency["recommended_bundle_id"] == "ITERATIVE|park"
    assert consistency["passing_bundle_count"] == 2
    assert [bundle["bundle_id"] for bundle in consistency["bundles"]] == [
        "ITERATIVE|park",
        "IPPE|park",
    ]
    recommended = consistency["recommendation"]
    assert recommended["mean_score"] == pytest.approx(0.205)
    assert recommended["aggregate_score"] == pytest.approx(0.41)
    assert recommended["max_pairwise_companion_translation_mm"] == pytest.approx(6.0)
    assert recommended["max_pairwise_companion_rotation_deg"] == pytest.approx(0.0)
    assert recommended["individual_score_equivalent_to_best"] is True
    assert consistency["bundles"][1]["individual_score_equivalent_to_best"] is False
    assert recommended["pairwise_companion_residuals"] == [
        {
            "left_sensor_key": "realsense_d435:1",
            "right_sensor_key": "oak_d_pro:2",
            "left_candidate_id": "realsense_d435:1|ITERATIVE|park",
            "right_candidate_id": "oak_d_pro:2|ITERATIVE|park",
            "translation_mm": pytest.approx(6.0),
            "rotation_deg": pytest.approx(0.0),
            "status": "ok",
        }
    ]
    checks = json.loads((tmp_path / "checks.json").read_text())["checks"]
    joint_checks = [
        check for check in checks if check.get("scope") == "multi_camera_bundle"
    ]
    assert len(joint_checks) == 12
    assert {check["bundle_id"] for check in joint_checks} == {
        "IPPE|park",
        "ITERATIVE|park",
    }

    attempt = {"request": request_value, "results": ranking}
    expected = {
        "realsense_d435:1": "realsense_d435:1|ITERATIVE|park",
        "oak_d_pro:2": "oak_d_pro:2|ITERATIVE|park",
    }
    assert attempt_module._promotion_selections(attempt, None) == expected
    assert attempt_module._promotion_selections(
        attempt,
        {
            "realsense_d435:1": "realsense_d435:1|IPPE|park",
            "oak_d_pro:2": "oak_d_pro:2|IPPE|park",
        },
    ) == {
        "realsense_d435:1": "realsense_d435:1|IPPE|park",
        "oak_d_pro:2": "oak_d_pro:2|IPPE|park",
    }
    with pytest.raises(ValueError, match="common algorithm bundle"):
        attempt_module._promotion_selections(
            attempt,
            {
                "realsense_d435:1": "realsense_d435:1|IPPE|park",
                "oak_d_pro:2": "oak_d_pro:2|ITERATIVE|park",
            },
        )
    with pytest.raises(ValueError, match="every jointly ranked sensor"):
        attempt_module._promotion_selections(
            attempt,
            {"realsense_d435:1": "realsense_d435:1|ITERATIVE|park"},
        )


def test_multi_camera_ranking_prefers_closure_within_individual_quality_band(
    tmp_path: Path,
) -> None:
    observations, _expected, _companion = _fixture_observations("eye_in_hand")
    base = evaluate_extrinsic_candidate(
        observations,
        mode="eye_in_hand",
        pnp_method="IPPE",
        extrinsic_method="park",
        sensor_key="realsense_d435:1",
    )
    bundle_specs = (
        ("shah", 0.666645, 5.689),
        ("horaud", 0.672523, 3.612),
        ("park", 0.672527, 3.617),
        ("li", 0.80, 1.0),
    )
    candidates = [
        _multi_camera_candidate_variant(
            base,
            sensor_key=sensor_key,
            pnp_method="IPPE",
            extrinsic_method=extrinsic_method,
            score=score,
            companion_translation_offset_mm=(
                0.0 if sensor_key == "realsense_d435:1" else closure_mm
            ),
        )
        for extrinsic_method, score, closure_mm in bundle_specs
        for sensor_key in ("realsense_d435:1", "oak_d_pro:2")
    ]
    request_value = _multi_camera_request()
    request_value["pnp_methods"] = ["IPPE"]
    request_value["extrinsic_methods"] = [
        "shah",
        "horaud",
        "park",
        "li",
    ]

    ranking = attempt_module._validate_and_rank(
        tmp_path,
        request_value,
        candidates,
        _multi_camera_intrinsics(),
    )

    consistency = ranking["multi_camera_consistency"]
    assert consistency["recommended_bundle_id"] == "IPPE|horaud"
    assert [bundle["bundle_id"] for bundle in consistency["bundles"]] == [
        "IPPE|horaud",
        "IPPE|park",
        "IPPE|shah",
        "IPPE|li",
    ]
    assert consistency["ranking_policy"] == {
        "individual_quality_metric": "mean_score",
        "best_individual_score": pytest.approx(0.666645),
        "individual_score_equivalence_tolerance": 0.01,
        "equivalent_quality_ordering_metric": ("normalized_companion_closure_score"),
        "normalized_companion_closure_score_definition": (
            "max_pairwise_translation_mm/max_translation_mm + "
            "max_pairwise_rotation_deg/max_rotation_deg"
        ),
        "numeric_round_decimals": 6,
        "closure_score_equivalence_tolerance": 1e-6,
        "closure_equivalent_ordering": "canonical_algorithm_order",
        "outside_equivalence_band_ordering_metric": "mean_score",
    }
    bundles = {bundle["bundle_id"]: bundle for bundle in consistency["bundles"]}
    assert bundles["IPPE|shah"]["individual_score_delta_from_best"] == pytest.approx(
        0.0
    )
    assert bundles["IPPE|horaud"]["individual_score_equivalent_to_best"] is True
    assert bundles["IPPE|horaud"][
        "normalized_companion_closure_score"
    ] == pytest.approx(0.3612)
    assert bundles["IPPE|li"]["individual_score_equivalent_to_best"] is False
    assert bundles["IPPE|li"]["normalized_companion_closure_score"] == pytest.approx(
        0.1
    )
    assert {
        result["sensor_key"]: result["recommended_candidate_id"]
        for result in ranking["results"]
    } == {
        "realsense_d435:1": "realsense_d435:1|IPPE|horaud",
        "oak_d_pro:2": "oak_d_pro:2|IPPE|horaud",
    }


def test_multi_camera_ranking_ignores_sub_micrometre_pnp_closure_dust(
    tmp_path: Path,
) -> None:
    observations, _expected, _companion = _fixture_observations("eye_in_hand")
    base = evaluate_extrinsic_candidate(
        observations,
        mode="eye_in_hand",
        pnp_method="IPPE",
        extrinsic_method="horaud",
        sensor_key="realsense_d435:1",
    )
    candidates = [
        _multi_camera_candidate_variant(
            base,
            sensor_key=sensor_key,
            pnp_method=pnp_method,
            extrinsic_method="horaud",
            score=0.6725234,
            companion_translation_offset_mm=(
                0.0 if sensor_key == "realsense_d435:1" else closure_mm
            ),
        )
        for pnp_method, closure_mm in (
            ("IPPE", 3.6120012),
            ("ITERATIVE", 3.6120000),
        )
        for sensor_key in ("realsense_d435:1", "oak_d_pro:2")
    ]
    request_value = _multi_camera_request()
    request_value["extrinsic_methods"] = ["horaud"]

    ranking = attempt_module._validate_and_rank(
        tmp_path,
        request_value,
        candidates,
        _multi_camera_intrinsics(),
    )

    consistency = ranking["multi_camera_consistency"]
    bundles = {bundle["bundle_id"]: bundle for bundle in consistency["bundles"]}
    assert (
        bundles["ITERATIVE|horaud"]["max_pairwise_companion_translation_mm"]
        < bundles["IPPE|horaud"]["max_pairwise_companion_translation_mm"]
    )
    assert (
        bundles["IPPE|horaud"]["max_pairwise_companion_translation_mm"]
        - bundles["ITERATIVE|horaud"]["max_pairwise_companion_translation_mm"]
    ) == pytest.approx(1.2e-6)
    assert round(
        bundles["IPPE|horaud"]["normalized_companion_closure_score"], 6
    ) == round(bundles["ITERATIVE|horaud"]["normalized_companion_closure_score"], 6)
    assert bundles["IPPE|horaud"]["closure_score_equivalent_to_best"] is True
    assert bundles["ITERATIVE|horaud"]["closure_score_equivalent_to_best"] is True
    assert consistency["ranking_policy"]["numeric_round_decimals"] == 6
    assert consistency["ranking_policy"][
        "closure_score_equivalence_tolerance"
    ] == pytest.approx(1e-6)
    assert consistency["recommended_bundle_id"] == "IPPE|horaud"


def test_multi_camera_promotion_rejects_tampered_pairwise_summary(
    tmp_path: Path,
) -> None:
    observations, _expected, _companion = _fixture_observations("eye_in_hand")
    base = evaluate_extrinsic_candidate(
        observations,
        mode="eye_in_hand",
        pnp_method="ITERATIVE",
        extrinsic_method="park",
        sensor_key="realsense_d435:1",
    )
    candidates = [
        _multi_camera_candidate_variant(
            base,
            sensor_key=sensor_key,
            pnp_method="ITERATIVE",
            extrinsic_method="park",
            score=0.2,
            companion_translation_offset_mm=offset,
        )
        for sensor_key, offset in (
            ("realsense_d435:1", 0.0),
            ("oak_d_pro:2", 2.0),
        )
    ]
    request_value = _multi_camera_request()
    request_value["pnp_methods"] = ["ITERATIVE"]
    ranking = attempt_module._validate_and_rank(
        tmp_path,
        request_value,
        candidates,
        _multi_camera_intrinsics(),
    )
    ranking["multi_camera_consistency"]["bundles"][0][
        "max_pairwise_companion_translation_mm"
    ] = 0.0
    attempt = {"request": request_value, "results": ranking}

    with pytest.raises(ValueError, match="bundle summary is inconsistent"):
        attempt_module._promotion_selections(attempt, None)


def test_multi_camera_ranking_rejects_inconsistent_passing_companions(
    tmp_path: Path,
) -> None:
    observations, _expected, _companion = _fixture_observations("eye_in_hand")
    base = evaluate_extrinsic_candidate(
        observations,
        mode="eye_in_hand",
        pnp_method="ITERATIVE",
        extrinsic_method="park",
        sensor_key="realsense_d435:1",
    )
    candidates = [
        _multi_camera_candidate_variant(
            base,
            sensor_key=sensor_key,
            pnp_method="ITERATIVE",
            extrinsic_method="park",
            score=0.2,
            companion_translation_offset_mm=offset,
        )
        for sensor_key, offset in (
            ("realsense_d435:1", 0.0),
            ("oak_d_pro:2", 10.01),
        )
    ]
    request_value = _multi_camera_request()
    request_value["pnp_methods"] = ["ITERATIVE"]

    ranking = attempt_module._validate_and_rank(
        tmp_path,
        request_value,
        candidates,
        _multi_camera_intrinsics(),
    )

    assert all(candidate["status"] == "passing" for candidate in candidates)
    assert ranking["status"] == "failed"
    assert ranking["recommended_camera_count"] == 0
    bundle = ranking["multi_camera_consistency"]["bundles"][0]
    assert bundle["status"] == "failed"
    assert bundle["max_pairwise_companion_translation_mm"] == pytest.approx(10.01)
    assert next(
        check
        for check in bundle["checks"]
        if check["name"] == "joint_companion_translation_consistency"
    ) == {
        "name": "joint_companion_translation_consistency",
        "status": "error",
        "actual": pytest.approx(10.01),
        "threshold": 10.0,
        "unit": "mm",
    }


def test_multi_camera_ranking_fails_closed_when_peer_fails(tmp_path) -> None:
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

    assert ranking["status"] == "failed"
    assert ranking["recommended_camera_count"] == 0
    assert ranking["failed_camera_count"] == 2
    assert ranking["results"][0]["recommended_candidate_id"] is None
    assert ranking["results"][1]["recommended_candidate_id"] is None
    consistency = ranking["multi_camera_consistency"]
    assert consistency["status"] == "failed"
    assert consistency["recommended_bundle_id"] is None
    assert consistency["bundles"][0]["candidate_ids"] == {
        "realsense_d435:1": "realsense_d435:1|ITERATIVE|park",
        "oak_d_pro:2": "oak_d_pro:2|ITERATIVE|park",
    }
    assert (
        next(
            check
            for check in consistency["bundles"][0]["checks"]
            if check["name"] == "joint_individual_candidate_validation"
        )["status"]
        == "error"
    )
