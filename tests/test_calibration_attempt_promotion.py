from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from posetestbot.calibration import attempts as attempt_module
from posetestbot.calibration.attempts import (
    create_calibration_attempt,
    create_promotion_request,
    promote_calibration_attempt,
)
from posetestbot.calibration.intrinsics import (
    factory_intrinsic_profile,
    load_intrinsic_profile_collection,
    write_intrinsic_profile_collection,
)
from posetestbot.calibration.posegridgen import posegridgen_capabilities
from posetestbot.calibration.profiles import (
    SCHEMA_VERSION,
    CalibrationProfile,
    CalibrationQuality,
    CalibrationStatus,
    CalibrationTargetType,
    RigidTransform,
    TransformFrame,
    load_profile_collection,
    write_profile_collection,
)
from posetestbot.calibration.target_library import (
    generate_target_bundle,
    select_target_bundle,
)
from posetestbot.io.artifacts import (
    CALIBRATION_CANDIDATES,
    CALIBRATION_OBSERVATIONS,
    CALIBRATION_PROFILES,
    CALIBRATION_PROFILES_SOLVED,
    CALIBRATION_SOLVER_REPORT,
    CALIBRATION_TARGET,
    CALIBRATION_VALIDATION_REPORT,
    DEPTH_DIR,
    FRAME_METADATA_JSONL,
    RGB_DIR,
    SYNC_QUALITY_REPORT,
    TIME_OFFSET_SEARCH,
)
from posetestbot.pipeline.run_config import (
    create_run_config,
    load_run_config_for_run_root,
    sensor_configs_from_values,
    write_run_config_with_manifest,
)
from posetestbot.pipeline.rewrite_gate import (
    build_calibration_validation_gate_report,
)
from posetestbot.sensors.contracts import CameraIntrinsics, MountingMode, SensorType
from posetestbot.sensors.frame_writer import write_legacy_camera_sidecars


def _configuration() -> dict:
    value = copy.deepcopy(posegridgen_capabilities()["defaults"])
    value["page"]["orientation"] = "landscape"
    value["board"].update({"rows": 2, "columns": 3, "marker_size_mm": 25.0})
    value["annotations"] = {
        "show_ruler": False,
        "show_parameters": False,
        "show_frame_legend": False,
    }
    return value


def _write_capture_folder(path: Path) -> None:
    image = np.zeros((8, 8), dtype=np.uint16)
    for directory in (RGB_DIR, DEPTH_DIR):
        target = path / directory / "1000.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(target.as_posix(), image)
    (path / FRAME_METADATA_JSONL).write_text(
        json.dumps(
            {
                "frame_id": "1000.png",
                "host_received_timestamp_ns": 1_000_000_000,
                "sensor_timestamp_ns": 10_000_000_000,
                "color_timestamp_domain": "global_time",
            }
        )
        + "\n"
    )
    write_legacy_camera_sidecars(
        path,
        CameraIntrinsics(
            cam_k=(600.0, 0.0, 4.0, 0.0, 600.0, 4.0, 0.0, 0.0, 1.0),
            width=8,
            height=8,
            distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
            depth_scale_to_mm=1.0,
        ),
        include_distortion_in_cam_k=True,
    )


def _write_fixed_zero_time_offset_evidence(
    run_root: Path,
    request_value: dict,
) -> tuple[dict, str]:
    """Write the report-backed timing evidence required by promotion."""

    attempt_id = request_value["attempt_id"]
    attempt_root = attempt_module.calibration_attempt_root(run_root, attempt_id)
    source_reference = attempt_module._attempt_artifact_reference(
        attempt_id,
        TIME_OFFSET_SEARCH,
    )
    sensor_results = []
    quality_sensors = []
    observations = []
    per_sensor_offsets = {}
    for sensor in request_value["sensors"]:
        sensor_key = sensor["sensor_key"]
        result = attempt_module.fixed_zero_sensor_result(
            sensor_key=sensor_key,
            observation_count=1,
        )
        result.update(
            {
                "display_name": sensor.get("display_name"),
                "sensor_name": sensor["sensor_name"],
            }
        )
        sensor_results.append(result)
        quality_sensors.append(
            {
                "sensor_name": sensor["sensor_name"],
                "sensor_type": sensor["sensor_type"],
                "sync_delta_ms": 0.0,
            }
        )
        per_sensor_offsets[sensor_key] = {
            "robot_pose_time_offset_ms": 0.0,
            "sync_delta_ms": 0.0,
            "status": "fixed_zero",
        }
        observations.append(
            {
                "observation_id": f"{sensor_key}:IPPE:1000.png",
                "sensor_type": sensor["sensor_type"],
                "device_id": sensor["device_id"],
                "robot_pose_time_offset_ms": 0.0,
                "sync_delta_ms": 0.0,
                "timestamp_alignment": {
                    "source": source_reference,
                    "frame_timestamp_ns": 10_000_000_000,
                    "robot_pose_query_timestamp_ns": 10_000_000_000,
                    "robot_pose_time_offset_ms": 0.0,
                    "sync_delta_ms": 0.0,
                    "matched_robot_pose_index": 0,
                    "robot_timestamp_ns": 10_000_000_000,
                    "nearest_robot_delta_ns": 0,
                },
            }
        )

    report = {
        "schema_version": attempt_module.TIME_OFFSET_SEARCH_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "policy": "fixed_zero",
        "implementation_revision": request_value[
            "synchronization_implementation_revision"
        ],
        "offset_kind": "effective_capture_and_pose_pipeline_latency",
        "sign_convention": attempt_module.time_offset_sign_convention(),
        "search": request_value["synchronization_search"],
        "status": "complete",
        "sensor_count": len(sensor_results),
        "failed_sensor_keys": [],
        "sensors": sensor_results,
    }
    (attempt_root / TIME_OFFSET_SEARCH).write_text(json.dumps(report))
    (attempt_root / SYNC_QUALITY_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "sync_quality_report.v2",
                "run_root": run_root.as_posix(),
                "overall_status": "ok",
                "sensor_count": len(quality_sensors),
                "checks": [],
                "sensors": quality_sensors,
                "calibration_attempt_policy": {
                    "purpose": "authoritative_calibration_solver_pairing",
                    "synchronization_policy": "fixed_zero",
                    "time_offset_search": source_reference,
                    "sign_convention": attempt_module.time_offset_sign_convention(),
                    "per_sensor_offsets": per_sensor_offsets,
                    "historical_per_sensor_offsets_allowed": False,
                    "auto_estimated_per_sensor_offsets": False,
                },
            }
        )
    )
    (attempt_root / CALIBRATION_OBSERVATIONS).write_text(
        json.dumps(
            {
                "schema_version": "calibration_observations.v1",
                "run_root": run_root.as_posix(),
                "attempt_id": attempt_id,
                "overall_status": "ok",
                "sensor_count": len(sensor_results),
                "observation_count": len(observations),
                "time_offset_search": source_reference,
                "synchronization_policy": "fixed_zero",
                "observations": observations,
            }
        )
    )
    return report, source_reference


def _profile(
    *,
    profile_id: str,
    sensor_type: SensorType,
    sensor_id: str,
    mounting_mode: MountingMode,
    candidate_id: str,
    translation: tuple[float, float, float],
    status: CalibrationStatus = CalibrationStatus.NEEDS_VALIDATION,
    observation_count: int = 8,
    inlier_count: int = 8,
    outlier_ratio: float = 0.0,
    sync_delta_ms: float | None = None,
) -> CalibrationProfile:
    intrinsics = CameraIntrinsics(
        cam_k=(600.0, 0.0, 4.0, 0.0, 600.0, 4.0, 0.0, 0.0, 1.0),
        width=8,
        height=8,
        distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
        depth_scale_to_mm=1.0,
    )
    return CalibrationProfile(
        schema_version=SCHEMA_VERSION,
        profile_id=profile_id,
        sensor_id=sensor_id,
        sensor_type=sensor_type,
        mounting_mode=mounting_mode,
        rig_position="wrist" if mounting_mode == MountingMode.EYE_IN_HAND else "static",
        intrinsics=intrinsics,
        extrinsics=RigidTransform(
            from_frame=TransformFrame.CAMERA,
            to_frame=(
                TransformFrame.ROBOT_FLANGE
                if mounting_mode == MountingMode.EYE_IN_HAND
                else TransformFrame.TEMPLATE_BASE
            ),
            rotation_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            translation_mm=translation,
        ),
        target_type=CalibrationTargetType.ARUCO_GRID,
        method="attempt-test",
        status=status,
        quality=CalibrationQuality(
            num_observations=observation_count,
            num_inliers=inlier_count,
            mean_reprojection_error_px=0.2,
            residual_translation_mm=0.5,
            residual_rotation_deg=0.25,
        ),
        sync_delta_ms=sync_delta_ms,
        metadata={
            "candidate_id": candidate_id,
            "sensor_key": f"{sensor_type.value}:{sensor_id}",
            "companion_transform": {
                "from": "aruco_grid",
                "to": (
                    "template_base"
                    if mounting_mode == MountingMode.EYE_IN_HAND
                    else "robot_flange"
                ),
                "matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "translation_mm": [0.0, 0.0, 0.0],
            },
            "outlier_count": observation_count - inlier_count,
            "outlier_ratio": outlier_ratio,
        },
    )


def _motion_balanced_candidate(
    candidate_id: str,
    *,
    translation: tuple[float, float, float] = (10.0, 20.0, 30.0),
) -> dict:
    per_motion = {
        f"clean_{index}": {
            "observation_count": 1,
            "inlier_count": 1,
            "outlier_count": 0,
            "outlier_ratio": 0.0,
            "residuals": {},
        }
        for index in range(12)
    }
    per_motion.update(
        {
            f"sparse_bad_{index}": {
                "observation_count": 3,
                "inlier_count": 0,
                "outlier_count": 3,
                "outlier_ratio": 1.0,
                "residuals": {},
            }
            for index in range(3)
        }
    )
    balanced_ratio = 3 / 15
    raw_ratio = 9 / 21
    return {
        "candidate_id": candidate_id,
        "pnp_method": "IPPE",
        "extrinsic_method": "park",
        "status": "passing",
        "primary_transform": {
            "from": "camera",
            "to": "robot_flange",
            "matrix": [
                [1.0, 0.0, 0.0, translation[0]],
                [0.0, 1.0, 0.0, translation[1]],
                [0.0, 0.0, 1.0, translation[2]],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "translation_mm": list(translation),
        },
        "companion_transform": {
            "from": "aruco_grid",
            "to": "template_base",
            "matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "translation_mm": [0.0, 0.0, 0.0],
        },
        "observation_count": 21,
        "inlier_count": 12,
        "outlier_count": 9,
        "outlier_ratio": balanced_ratio,
        "raw_outlier_ratio": raw_ratio,
        "full_input_validation": {
            "per_motion": per_motion,
            "motion_balanced_outlier_ratio": balanced_ratio,
            "max_repeated_motion_outlier_ratio": 0.0,
        },
        "checks": [
            {
                "name": "outlier_ratio",
                "status": "ok",
                "actual": balanced_ratio,
                "threshold": 0.25,
            },
            {
                "name": "full_input_repeated_motion_outlier_ratio",
                "status": "ok",
                "actual": 0.0,
                "threshold": 0.25,
            },
        ],
    }


def test_promotion_outlier_evidence_rejects_tampered_aggregate() -> None:
    candidate_id = "realsense_d435:1|IPPE|park"
    candidate = _motion_balanced_candidate(candidate_id)
    candidate["full_input_validation"]["motion_balanced_outlier_ratio"] = 0.1
    profile = _profile(
        profile_id="tamper-check",
        sensor_type=SensorType.REALSENSE_D435,
        sensor_id="1",
        mounting_mode=MountingMode.EYE_IN_HAND,
        candidate_id=candidate_id,
        translation=(10.0, 20.0, 30.0),
        observation_count=21,
        inlier_count=12,
        outlier_ratio=3 / 15,
    )

    with pytest.raises(ValueError, match="inconsistent aggregate outlier evidence"):
        attempt_module._promotion_outlier_evidence(
            candidate,
            profile,
            candidate_id=candidate_id,
        )


def _report_backed_fixed_zero_attempt(tmp_path: Path) -> tuple[dict, Path]:
    run_root = tmp_path / "run"
    attempt_id = "a" * 32
    attempt_root = attempt_module.calibration_attempt_root(run_root, attempt_id)
    attempt_root.mkdir(parents=True)
    request_value = {
        "attempt_id": attempt_id,
        "run_root": run_root.as_posix(),
        "sensor_keys": ["realsense_d435:1"],
        "sensors": [
            {
                "sensor_key": "realsense_d435:1",
                "sensor_name": "realsense_1",
                "sensor_type": "realsense_d435",
                "device_id": "1",
                "display_name": "D435",
            }
        ],
        "synchronization_policy": "fixed_zero",
        "synchronization_search": attempt_module.time_offset_search_configuration(),
        "synchronization_implementation_revision": (
            attempt_module.TIME_OFFSET_IMPLEMENTATION_REVISION
        ),
    }
    report, _ = _write_fixed_zero_time_offset_evidence(run_root, request_value)
    return {
        "attempt_id": attempt_id,
        "run_root": run_root.as_posix(),
        "request": request_value,
        "time_offset_search": report,
    }, attempt_root


def test_report_backed_fixed_zero_time_offset_evidence_is_promotable(
    tmp_path: Path,
) -> None:
    attempt, _ = _report_backed_fixed_zero_attempt(tmp_path)

    evidence = attempt_module._promotion_time_offset_evidence(attempt)

    assert set(evidence) == {"realsense_d435:1"}
    assert evidence["realsense_d435:1"]["status"] == "fixed_zero"
    assert evidence["realsense_d435:1"]["selected_sync_delta_ms"] == 0.0


def test_report_backed_applied_auto_offset_evidence_is_promotable(
    tmp_path: Path,
) -> None:
    attempt, attempt_root = _report_backed_fixed_zero_attempt(tmp_path)
    sensor_key = "realsense_d435:1"
    attempt["request"]["synchronization_policy"] = "auto_offset"
    report = attempt["time_offset_search"]
    report["policy"] = "auto_offset"
    sensor = report["sensors"][0]
    sensor.update(
        {
            "status": "applied",
            "decision": "auto_offset_applied",
            "selected_robot_pose_time_offset_ms": 75.0,
            "selected_sync_delta_ms": -75.0,
            "candidate_robot_pose_time_offset_ms": 75.0,
            "candidate_sync_delta_ms": -75.0,
            "boundary_hit": False,
            "checks": [
                {"name": name, "status": "ok"}
                for name in (
                    "fixed_full_range_observation_set",
                    "cross_validation_offset_stability",
                    "reference_method_sensitivity",
                    "search_optimum_not_at_boundary",
                    "cross_validated_translation_improvement",
                    "cross_validated_rotation_guard",
                    "zero_offset_identifiability",
                )
            ],
        }
    )

    quality_path = attempt_root / SYNC_QUALITY_REPORT
    quality = json.loads(quality_path.read_text())
    quality["sensors"][0]["sync_delta_ms"] = -75.0
    policy = quality["calibration_attempt_policy"]
    policy["synchronization_policy"] = "auto_offset"
    policy["auto_estimated_per_sensor_offsets"] = True
    policy["per_sensor_offsets"][sensor_key] = {
        "robot_pose_time_offset_ms": 75.0,
        "sync_delta_ms": -75.0,
        "status": "applied",
    }
    quality_path.write_text(json.dumps(quality))

    observations_path = attempt_root / CALIBRATION_OBSERVATIONS
    observations = json.loads(observations_path.read_text())
    observations["synchronization_policy"] = "auto_offset"
    observation = observations["observations"][0]
    observation["robot_pose_time_offset_ms"] = 75.0
    observation["sync_delta_ms"] = -75.0
    observation["timestamp_alignment"]["robot_pose_time_offset_ms"] = 75.0
    observation["timestamp_alignment"]["sync_delta_ms"] = -75.0
    observations_path.write_text(json.dumps(observations))

    evidence = attempt_module._promotion_time_offset_evidence(attempt)

    assert evidence[sensor_key]["status"] == "applied"
    assert evidence[sensor_key]["selected_robot_pose_time_offset_ms"] == 75.0
    assert evidence[sensor_key]["selected_sync_delta_ms"] == -75.0


@pytest.mark.parametrize(
    ("tamper_mode", "error"),
    [
        ("report_sign", "sign evidence is inconsistent"),
        ("quality_offset", "Authoritative synchronization offset is inconsistent"),
        ("observation_source", "observation timing is inconsistent"),
        ("unsupported_revision", "time-offset promotion evidence is invalid"),
        ("non_string_revision", "time-offset promotion evidence is invalid"),
    ],
)
def test_report_backed_fixed_zero_time_offset_tampering_blocks_promotion(
    tmp_path: Path,
    tamper_mode: str,
    error: str,
) -> None:
    attempt, attempt_root = _report_backed_fixed_zero_attempt(tmp_path)
    sensor_key = "realsense_d435:1"
    if tamper_mode == "report_sign":
        attempt["time_offset_search"]["sensors"][0]["selected_sync_delta_ms"] = 1.0
    elif tamper_mode == "quality_offset":
        path = attempt_root / SYNC_QUALITY_REPORT
        value = json.loads(path.read_text())
        value["calibration_attempt_policy"]["per_sensor_offsets"][sensor_key][
            "sync_delta_ms"
        ] = 1.0
        path.write_text(json.dumps(value))
    elif tamper_mode == "observation_source":
        path = attempt_root / CALIBRATION_OBSERVATIONS
        value = json.loads(path.read_text())
        value["observations"][0]["timestamp_alignment"]["source"] = (
            "processed/calibration/other/time_offset_search.json"
        )
        path.write_text(json.dumps(value))
    elif tamper_mode == "unsupported_revision":
        attempt["request"]["synchronization_implementation_revision"] = (
            "unsupported_revision.v0"
        )
        attempt["time_offset_search"]["implementation_revision"] = (
            "unsupported_revision.v0"
        )
    else:
        attempt["request"]["synchronization_implementation_revision"] = [
            attempt_module.TIME_OFFSET_IMPLEMENTATION_REVISION
        ]
        attempt["time_offset_search"]["implementation_revision"] = [
            attempt_module.TIME_OFFSET_IMPLEMENTATION_REVISION
        ]

    with pytest.raises(ValueError, match=error):
        attempt_module._promotion_time_offset_evidence(attempt)


@pytest.mark.parametrize(
    "tamper_mode",
    [None, "candidate_profile_transform", "promotion_status_selection"],
)
def test_promotion_transaction_preserves_unrelated_profiles_and_updates_selected_camera(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_mode: str | None,
) -> None:
    run_root = tmp_path / "run"
    configured = sensor_configs_from_values(
        [
            {
                "sensor_type": "realsense_d435",
                "device_id": "1",
                "display_name": "D435",
                "mounting_mode": "eye_in_hand",
            },
            {
                "sensor_type": "oak_d_pro",
                "device_id": "2",
                "display_name": "OAK",
                "mounting_mode": "static",
            },
        ]
    )
    write_run_config_with_manifest(
        run_root,
        create_run_config(run_root=run_root, sensors=configured),
    )
    _write_capture_folder(run_root / "realsense_1")
    _write_capture_folder(run_root / "luxonis_2")
    (run_root / "raw_robot_ee_poses.json").write_text(
        json.dumps(
            {
                "0": {
                    "motion": "pose_0",
                    "framename": 1000,
                    "host_wall_timestamp_ns": 10_000_000_000,
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
    library = tmp_path / "library"
    bundle = generate_target_bundle(
        display_name="Promotion target",
        configuration=_configuration(),
        library_root=library,
    )
    monkeypatch.setattr(attempt_module, "default_target_library_root", lambda: library)
    select_target_bundle(
        run_root=run_root,
        target_id=bundle["target_id"],
        placement_mode="unknown",
        library_root=library,
    )
    request_value = create_calibration_attempt(
        run_root,
        {
            "mode": "eye_in_hand",
            "sensor_keys": ["realsense_d435:1"],
            "target_id": bundle["target_id"],
            "synchronization_policy": "fixed_zero",
        },
    )
    attempt_id = request_value["attempt_id"]
    attempt_root = run_root / "processed" / "calibration" / attempt_id
    _, time_offset_source = _write_fixed_zero_time_offset_evidence(
        run_root,
        request_value,
    )
    candidate_id = "realsense_d435:1|IPPE|park"
    candidate = _profile(
        profile_id="new_d435_profile",
        sensor_type=SensorType.REALSENSE_D435,
        sensor_id="1",
        mounting_mode=MountingMode.EYE_IN_HAND,
        candidate_id=candidate_id,
        translation=(10.0, 20.0, 30.0),
        observation_count=21,
        inlier_count=12,
        outlier_ratio=3 / 15,
        sync_delta_ms=0.0,
    )
    synchronization = {
        "source": time_offset_source,
        "policy": "fixed_zero",
        "status": "fixed_zero",
        "robot_pose_time_offset_ms": 0.0,
        "sync_delta_ms": 0.0,
    }
    candidate = replace(
        candidate,
        intrinsics=replace(
            candidate.intrinsics,
            distortion_model="inverse_brown_conrady",
            projection_source="realsense_sdk_color_stream",
        ),
        metadata={
            **candidate.metadata,
            "synchronization": synchronization,
        },
    )
    candidate_evidence = _motion_balanced_candidate(candidate_id)
    candidate_evidence["synchronization"] = synchronization
    write_profile_collection([candidate], attempt_root / "candidate_profiles.json")
    intrinsic = factory_intrinsic_profile(run_root / "realsense_1")
    write_intrinsic_profile_collection(
        [intrinsic], attempt_root / "intrinsic_calibration_profiles.json"
    )
    unrelated_intrinsic = factory_intrinsic_profile(run_root / "luxonis_2")
    write_intrinsic_profile_collection(
        [unrelated_intrinsic],
        run_root / "intrinsic_calibration_profiles.json",
    )
    (attempt_root / "extrinsic_candidates.json").write_text(
        json.dumps({"candidates": [candidate_evidence]})
    )
    (attempt_root / "checks.json").write_text(json.dumps({"checks": []}))
    (attempt_root / "ranking.json").write_text(
        json.dumps(
            {
                "status": "partial",
                "recommended_camera_count": 1,
                "failed_camera_count": 1,
                "results": [
                    {
                        "sensor_key": "realsense_d435:1",
                        "recommended_candidate_id": candidate_id,
                        "candidates": [candidate_evidence],
                    },
                    {
                        "sensor_key": "oak_d_pro:2",
                        "recommended_candidate_id": None,
                        "candidates": [],
                    },
                ],
            }
        )
    )
    progress = json.loads((attempt_root / "progress.json").read_text())
    progress["status"] = "complete"
    (attempt_root / "progress.json").write_text(json.dumps(progress))
    unrelated = _profile(
        profile_id="keep_oak_profile",
        sensor_type=SensorType.OAK_D_PRO,
        sensor_id="2",
        mounting_mode=MountingMode.STATIC,
        candidate_id="old-oak",
        translation=(1.0, 2.0, 3.0),
        status=CalibrationStatus.VALID,
    )
    write_profile_collection([unrelated], run_root / CALIBRATION_PROFILES)
    create_promotion_request(
        run_root,
        attempt_id,
        selections={"realsense_d435:1": candidate_id},
        operator="test-operator",
    )

    if tamper_mode == "candidate_profile_transform":
        write_profile_collection(
            [
                replace(
                    candidate,
                    extrinsics=replace(
                        candidate.extrinsics,
                        translation_mm=(999.0, 20.0, 30.0),
                    ),
                )
            ],
            attempt_root / "candidate_profiles.json",
        )
        with pytest.raises(ValueError, match="ranked primary transform evidence"):
            promote_calibration_attempt(run_root, attempt_id)
        assert {
            profile.profile_id
            for profile in load_profile_collection(run_root / CALIBRATION_PROFILES)
        } == {"keep_oak_profile"}
        return
    if tamper_mode == "promotion_status_selection":
        promotion_status_path = attempt_root / "promotion.json"
        promotion_status = json.loads(promotion_status_path.read_text())
        promotion_status["selections"] = {
            "realsense_d435:1": "realsense_d435:1|ITERATIVE|park"
        }
        promotion_status_path.write_text(json.dumps(promotion_status))
        with pytest.raises(ValueError, match="selections are inconsistent"):
            promote_calibration_attempt(run_root, attempt_id)
        assert {
            profile.profile_id
            for profile in load_profile_collection(run_root / CALIBRATION_PROFILES)
        } == {"keep_oak_profile"}
        return

    result = promote_calibration_attempt(run_root, attempt_id)

    assert result["status"] == "promoted"
    assert candidate_evidence["raw_outlier_ratio"] > 0.25
    assert candidate_evidence["outlier_ratio"] <= 0.25
    profiles = load_profile_collection(run_root / CALIBRATION_PROFILES)
    assert {profile.profile_id for profile in profiles} == {
        "keep_oak_profile",
        "new_d435_profile",
    }
    promoted = next(
        profile for profile in profiles if profile.profile_id == "new_d435_profile"
    )
    assert promoted.status == CalibrationStatus.VALID
    assert promoted.operator == "test-operator"
    assert promoted.intrinsics.distortion_model == "inverse_brown_conrady"
    assert promoted.rectified_intrinsics is not None
    assert promoted.sync_delta_ms == 0.0
    canonical = json.loads((run_root / CALIBRATION_PROFILES).read_text())
    promoted_value = next(
        item
        for item in canonical["profiles"]
        if item["profile_id"] == "new_d435_profile"
    )
    assert promoted_value["intrinsics"]["rectified"] is not None
    assert promoted.metadata["promotion_attempt_id"] == attempt_id
    assert promoted.metadata["promotion_solver_provenance"] == {
        "solver_policy": "auto_compare",
        "pnp_method": "IPPE",
        "extrinsic_method": "park",
    }
    assert promoted.metadata["promotion_synchronization_provenance"] == {
        "source": time_offset_source,
        "status": "fixed_zero",
        "robot_pose_time_offset_ms": 0.0,
        "sync_delta_ms": 0.0,
    }
    config = load_run_config_for_run_root(run_root)
    sensors = {
        (item["sensor_type"], item["device_id"]): item
        for item in config["capture"]["sensors"]
    }
    assert sensors[("realsense_d435", "1")]["mounting_mode"] == "eye_in_hand"
    assert sensors[("realsense_d435", "1")]["calibration_profile_id"] == (
        "new_d435_profile"
    )
    assert sensors[("oak_d_pro", "2")]["calibration_profile_id"] is None
    assert config["calibration_target"]["target_id"] == bundle["target_id"]
    assert config["calibration_target"]["placement"] == {"mode": "unknown"}
    intrinsic_profiles = load_intrinsic_profile_collection(
        run_root / "intrinsic_calibration_profiles.json"
    )
    assert {item["sensor_id"] for item in intrinsic_profiles} == {"1", "2"}
    assert (run_root / CALIBRATION_TARGET).is_file()
    for filename in (
        CALIBRATION_CANDIDATES,
        CALIBRATION_SOLVER_REPORT,
        CALIBRATION_PROFILES_SOLVED,
        CALIBRATION_VALIDATION_REPORT,
    ):
        assert (run_root / filename).is_file()
    validation = json.loads((run_root / CALIBRATION_VALIDATION_REPORT).read_text())
    assert validation["promotion"]["profile_count"] == 2
    gate = build_calibration_validation_gate_report(run_root)
    assert gate["overall_status"] == "ready"
