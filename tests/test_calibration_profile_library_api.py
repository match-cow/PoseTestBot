from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from posetestbot.calibration.intrinsics import write_intrinsic_profile_collection
from posetestbot.calibration.profile_library import (
    selected_calibration_profile_ids_by_sensor_folder,
    verify_calibration_profile_selection,
)
from posetestbot.calibration.profiles import (
    CalibrationProfile,
    CalibrationQuality,
    CalibrationStatus,
    CalibrationTargetType,
    RigidTransform,
    TransformFrame,
    blenderproc_camera_transform_map_from_profiles,
    load_profile_collection,
    profile_to_dict,
    rectified_projection_from_native,
    write_profile_collection,
)
from posetestbot.io.atomic import atomic_write_json
from posetestbot.io.artifacts import (
    CALIBRATION_PROFILES,
    CALIBRATION_PROFILE_SELECTION,
    INTRINSIC_CALIBRATION_PROFILES,
    RAW_ROBOT_EE_POSES,
)
from posetestbot.pipeline.run_config import (
    SensorRunConfig,
    create_run_config,
    load_run_config_for_run_root,
    write_run_config,
)
from posetestbot.robot.reference_frames import POSE_TEMPLATE_BASE_SUNRISE_PATH
from posetestbot.sensors.contracts import CameraIntrinsics, MountingMode, SensorType
from posetestbot.sensors.registry import sensor_folder_name
from posetestbot.web.app import create_app
from scripts.run_bop_export_stage import calibration_profile_for_sensor


SENSOR_ID = "D435-123"
PROFILE_ID = "d435_123_eye_in_hand_valid"
INTRINSIC_ID = "D435-123_1280x720_normal_opencv"
STATIC_REFERENCE_PATH = POSE_TEMPLATE_BASE_SUNRISE_PATH


def _camera_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        cam_k=(900.0, 0.0, 640.0, 0.0, 900.0, 360.0, 0.0, 0.0, 1.0),
        width=1280,
        height=720,
        distortion=(0.01, -0.01, 0.0, 0.0, 0.0),
        depth_scale_to_mm=1.0,
        distortion_model="brown_conrady",
        projection_source="opencv_grid_calibration",
    )


def _rectified_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        cam_k=(895.0, 0.0, 640.0, 0.0, 895.0, 360.0, 0.0, 0.0, 1.0),
        width=1280,
        height=720,
        distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
        depth_scale_to_mm=1.0,
        distortion_model="brown_conrady",
        projection_source="opencv_alpha0",
    )


def _profile(
    *,
    mounting_mode: MountingMode = MountingMode.EYE_IN_HAND,
    sensor_id: str = SENSOR_ID,
    profile_id: str = PROFILE_ID,
    intrinsic_id: str = INTRINSIC_ID,
    static_reference_path: str | None = STATIC_REFERENCE_PATH,
) -> CalibrationProfile:
    to_frame = (
        TransformFrame.ROBOT_FLANGE
        if mounting_mode == MountingMode.EYE_IN_HAND
        else TransformFrame.TEMPLATE_BASE
    )
    metadata = {"intrinsic_profile_id": intrinsic_id}
    if mounting_mode == MountingMode.STATIC and static_reference_path is not None:
        metadata["robot_pose_reference"] = {
            "schema_version": "robot_pose_reference.v1",
            "status": "verified",
            "packet_schema_version": "robot_pose.v1",
            "from": "robot_flange",
            "to": "template_base",
            "sunrise_reference_frame_path": static_reference_path,
            "pose_count": 20,
        }
    return CalibrationProfile(
        schema_version="calibration.v2",
        profile_id=profile_id,
        sensor_id=sensor_id,
        sensor_type=SensorType.REALSENSE_D435,
        mounting_mode=mounting_mode,
        rig_position="wrist" if mounting_mode == MountingMode.EYE_IN_HAND else "cell",
        intrinsics=_camera_intrinsics(),
        rectified_intrinsics=_rectified_intrinsics(),
        rectified_valid_roi=(0, 0, 1280, 720),
        extrinsics=RigidTransform(
            from_frame=TransformFrame.CAMERA,
            to_frame=to_frame,
            rotation_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            translation_mm=(10.0, 20.0, 30.0),
        ),
        target_type=CalibrationTargetType.ARUCO_GRID,
        method="opencv_grid_and_hand_eye",
        status=CalibrationStatus.VALID,
        quality=CalibrationQuality(
            num_observations=20,
            num_inliers=18,
            mean_reprojection_error_px=0.4,
        ),
        operator="calibration-operator",
        calibrated_at="2026-07-20T10:00:00+00:00",
        metadata=metadata,
    )


def _intrinsic_profile(
    *, sensor_id: str = SENSOR_ID, intrinsic_id: str = INTRINSIC_ID
) -> dict:
    native = {
        "cam_K": list(_camera_intrinsics().cam_k),
        "width": 1280,
        "height": 720,
        "distortion_model": "brown_conrady",
        "distortion": [0.01, -0.01, 0.0, 0.0, 0.0],
    }
    rectified = {
        "cam_K": list(_rectified_intrinsics().cam_k),
        "width": 1280,
        "height": 720,
        "distortion_model": "brown_conrady",
        "distortion": [0.0] * 5,
        "alpha": 0.0,
        "valid_roi": [0, 0, 1280, 720],
    }
    return {
        "schema_version": "intrinsic_calibration.v1",
        "profile_id": intrinsic_id,
        "sensor_id": sensor_id,
        "sensor_name": f"realsense_{sensor_id}",
        "resolution": [1280, 720],
        "orientation": "normal",
        "native": native,
        "rectified": rectified,
        "depth": {
            "scale_to_mm": 1.0,
            "alignment": {"target": "rgb", "recalibrated": False},
        },
        "source": {"mode": "calibrate", "algorithm": "cv2.calibrateCameraExtended"},
        "quality": {"status": "accepted"},
    }


def _sensor(
    *, mounting_mode: str = "eye_in_hand", sensor_id: str = SENSOR_ID
) -> SensorRunConfig:
    return SensorRunConfig(
        sensor_type="realsense_d435",
        device_id=sensor_id,
        display_name="Front D435",
        mounting_mode=mounting_mode,
    )


def _write_source(
    source: Path,
    *,
    legacy: bool = False,
    bundle_marker: str | None = None,
    mounting_mode: MountingMode = MountingMode.EYE_IN_HAND,
    sensor_id: str = SENSOR_ID,
    profile_id: str = PROFILE_ID,
    intrinsic_id: str = INTRINSIC_ID,
    static_reference_path: str | None = STATIC_REFERENCE_PATH,
) -> None:
    source.mkdir(parents=True)
    profile = _profile(
        mounting_mode=mounting_mode,
        sensor_id=sensor_id,
        profile_id=profile_id,
        intrinsic_id=intrinsic_id,
        static_reference_path=static_reference_path,
    )
    intrinsic = _intrinsic_profile(sensor_id=sensor_id, intrinsic_id=intrinsic_id)
    if legacy:
        value = profile_to_dict(profile)
        value["schema_version"] = "calibration.v1"
        value["intrinsics"] = value["intrinsics"]["native"]
        value["extrinsics"]["to"] = "end_effector"
        atomic_write_json(
            source / CALIBRATION_PROFILES,
            {"schema_version": "calibration.v1", "profiles": [value]},
        )
        rectified, roi = rectified_projection_from_native(profile.intrinsics)
        assert rectified is not None and roi is not None
        intrinsic["rectified"]["cam_K"] = list(rectified.cam_k)
        intrinsic["rectified"]["valid_roi"] = list(roi)
    else:
        write_profile_collection([profile], source / CALIBRATION_PROFILES)
    if bundle_marker is not None:
        collection = json.loads((source / CALIBRATION_PROFILES).read_text())
        collection["promotion_marker"] = bundle_marker
        atomic_write_json(source / CALIBRATION_PROFILES, collection)
    write_intrinsic_profile_collection(
        [intrinsic], source / INTRINSIC_CALIBRATION_PROFILES
    )
    write_run_config(
        source,
        create_run_config(
            run_root=source,
            run_name="Reusable camera calibration",
            sensors=(
                _sensor(
                    mounting_mode=mounting_mode.value,
                    sensor_id=sensor_id,
                ),
            ),
        ),
    )


def _write_destination(
    destination: Path, *, mounting_mode: str = "eye_in_hand"
) -> None:
    destination.mkdir(parents=True)
    write_run_config(
        destination,
        create_run_config(
            run_root=destination,
            sensors=(_sensor(mounting_mode=mounting_mode),),
            sequence_id="calibrated_capture_to_bop_dataset_dry_run",
            robot_pose_sunrise_reference_frame_path=(
                STATIC_REFERENCE_PATH if mounting_mode == "static" else None
            ),
        ),
    )


def _write_raw_pose_reference(run_root: Path, reference_path: str) -> None:
    atomic_write_json(
        run_root / RAW_ROBOT_EE_POSES,
        {
            "0": {
                "motion": "capture",
                "pose": {"X": 0, "Y": 0, "Z": 0, "A": 0, "B": 0, "C": 0},
                "source_packet": {
                    "schema_version": "robot_pose.v1",
                    "from_frame": "robot_flange",
                    "to_frame": "template_base",
                    "sunrise_reference_frame_path": reference_path,
                },
            }
        },
    )


def _candidate(payload: dict, source: Path) -> dict:
    return next(
        item
        for item in payload["calibrations"]
        if item["source_run_root"] == source.as_posix()
    )


def _selection_request(destination: Path, source: Path, bundle_sha256: str) -> dict:
    return {
        "run_root": destination.as_posix(),
        "source_run_root": source.as_posix(),
        "expected_bundle_sha256": bundle_sha256,
    }


def _composite_selection_request(
    destination: Path, source: Path, bundle_sha256: str
) -> dict:
    return {
        "run_root": destination.as_posix(),
        "source_selections": [
            {
                "source_run_root": source.as_posix(),
                "expected_bundle_sha256": bundle_sha256,
                "sensor_keys": [f"realsense_d435:{SENSOR_ID}"],
            }
        ],
    }


def test_library_selection_snapshots_both_files_and_drives_sequence_options(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    source = runs / "calibration_run"
    destination = runs / "object_run"
    _write_source(source)
    _write_destination(destination)
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", runs.as_posix())
    client = create_app().test_client()

    listed = client.get("/ui/calibrations", query_string={"run_root": destination})

    assert listed.status_code == 200
    payload = listed.get_json()
    assert payload["schema_version"] == "calibration_library.v1"
    candidate = payload["calibrations"][0]
    assert candidate["source_run_name"] == "Reusable camera calibration"
    assert candidate["valid"] is True
    assert candidate["compatible"] is True
    assert candidate["calibration_profiles"]["valid_profile_count"] == 1
    assert candidate["intrinsic_calibration_profiles"]["profile_count"] == 1
    assert candidate["sensor_profiles"] == {f"realsense_d435:{SENSOR_ID}": PROFILE_ID}

    selected = client.post(
        "/ui/calibrations/select",
        json={
            "run_root": destination.as_posix(),
            "source_run_root": source.as_posix(),
            "expected_bundle_sha256": candidate["bundle_sha256"],
            "operator": "dataset-operator",
        },
    )

    assert selected.status_code == 201
    result = selected.get_json()
    full_relative = result["calibration_profiles"]
    intrinsic_relative = result["intrinsic_calibration_profiles"]
    assert full_relative.startswith("processed/calibration_inputs/")
    assert intrinsic_relative.startswith("processed/calibration_inputs/")
    assert (destination / full_relative).read_bytes() == (
        source / CALIBRATION_PROFILES
    ).read_bytes()
    assert (destination / intrinsic_relative).read_bytes() == (
        source / INTRINSIC_CALIBRATION_PROFILES
    ).read_bytes()
    assert (destination / CALIBRATION_PROFILE_SELECTION).is_file()
    assert result["stage_options"]["camera_rectification"] == {
        "intrinsic_profiles": intrinsic_relative
    }

    original_snapshot = (destination / full_relative).read_bytes()
    atomic_write_json(source / CALIBRATION_PROFILES, {"changed": True})
    assert (destination / full_relative).read_bytes() == original_snapshot

    saved = client.post(
        "/run-config",
        json={
            "run_root": destination.as_posix(),
            "run_name": "object_run",
            "resolution": "720p",
            "fps": 6,
            "velocity_m_s": 0.2,
            "sensors": [_sensor().to_dict()],
            "dataset_mode": "objectless",
            "calibration_profiles": full_relative,
            "sequence_id": "calibrated_capture_to_bop_dataset_dry_run",
            "plan_only": True,
        },
    )
    assert saved.status_code == 201, saved.get_json()
    config = load_run_config_for_run_root(destination)
    assert config["calibration_profiles"] == full_relative
    assert config["intrinsic_calibration_profiles"] == intrinsic_relative
    assert config["capture"]["sensors"][0]["calibration_profile_id"] == PROFILE_ID
    assert config["calibration_profile_selection"]["selection_artifact"] == (
        CALIBRATION_PROFILE_SELECTION
    )
    steps = {item["id"]: item for item in saved.get_json()["sequence_plan"]["steps"]}
    assert steps["camera_rectification"]["options"]["intrinsic_profiles"] == (
        intrinsic_relative
    )
    assert steps["bop_export"]["options"]["calibration_profiles"] == full_relative
    assert steps["bop_export"]["options"]["annotation_source"] == "none"
    assert "blenderproc_prepare" not in steps
    assert "blenderproc_render" not in steps


def test_composite_selection_combines_static_and_robot_mounted_source_runs(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    mobile_source = runs / "mobile_calibration"
    static_source = runs / "static_calibration"
    destination = runs / "object_run"
    static_sensor_id = "D435-STATIC-456"
    static_profile_id = "d435_static_456_valid"
    static_intrinsic_id = "D435-STATIC-456_1280x720_normal_opencv"
    _write_source(mobile_source)
    _write_source(
        static_source,
        mounting_mode=MountingMode.STATIC,
        sensor_id=static_sensor_id,
        profile_id=static_profile_id,
        intrinsic_id=static_intrinsic_id,
    )
    destination_sensors = (
        _sensor(),
        _sensor(mounting_mode="static", sensor_id=static_sensor_id),
    )
    intended_setup = {
        "resolution": "720p",
        "sensors": [sensor.to_dict() for sensor in destination_sensors],
    }
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", runs.as_posix())
    client = create_app().test_client()

    library = client.get(
        "/ui/calibrations", query_string={"run_root": destination}
    ).get_json()
    mobile = _candidate(library, mobile_source)
    static = _candidate(library, static_source)
    mobile_key = f"realsense_d435:{SENSOR_ID}"
    static_key = f"realsense_d435:{static_sensor_id}"
    assert mobile["valid"] is True and mobile["compatible"] is False
    assert static["valid"] is True and static["compatible"] is False
    assert mobile["issues"][0]["code"] == "destination_setup_required"
    assert static["issues"][0]["code"] == "destination_setup_required"
    assert mobile["sensor_profiles"] == {}
    assert static["sensor_profiles"] == {}
    assert (
        mobile["calibration_profiles"]["profiles"][0]["intrinsic_profile_id"]
        == INTRINSIC_ID
    )
    assert (
        static["calibration_profiles"]["profiles"][0]["intrinsic_profile_id"]
        == static_intrinsic_id
    )

    incomplete = client.post(
        "/ui/calibrations/select",
        json={
            "run_root": destination.as_posix(),
            "source_selections": [
                {
                    "source_run_root": mobile_source.as_posix(),
                    "expected_bundle_sha256": mobile["bundle_sha256"],
                    "sensor_keys": [mobile_key],
                }
            ],
            "robot_pose_sunrise_reference_frame_path": STATIC_REFERENCE_PATH,
            **intended_setup,
        },
    )
    assert incomplete.status_code == 409
    assert incomplete.get_json()["issues"] == [
        {
            "code": "calibration_source_assignment_missing",
            "message": f"Select a calibration source for {static_key}.",
            "sensor_key": static_key,
        }
    ]

    selected = client.post(
        "/ui/calibrations/select",
        json={
            "run_root": destination.as_posix(),
            "operator": "mixed-rig-operator",
            "source_selections": [
                {
                    "source_run_root": mobile_source.as_posix(),
                    "expected_bundle_sha256": mobile["bundle_sha256"],
                    "sensor_keys": [mobile_key],
                },
                {
                    "source_run_root": static_source.as_posix(),
                    "expected_bundle_sha256": static["bundle_sha256"],
                    "sensor_keys": [static_key],
                },
            ],
            "robot_pose_sunrise_reference_frame_path": STATIC_REFERENCE_PATH,
            **intended_setup,
        },
    )

    assert selected.status_code == 201, selected.get_json()
    result = selected.get_json()
    selection = result["selection"]
    assert result["schema_version"] == "calibration_profile_selection.v2"
    assert selection["source"]["kind"] == "composite"
    assert selection["source"]["source_count"] == 2
    assert selection["intended_setup"][
        "robot_pose_sunrise_reference_frame_path"
    ] == STATIC_REFERENCE_PATH
    assert {
        item["run_root"]: item["selected_sensor_keys"] for item in selection["sources"]
    } == {
        mobile_source.as_posix(): [mobile_key],
        static_source.as_posix(): [static_key],
    }
    assert result["sensor_profiles"] == {
        mobile_key: PROFILE_ID,
        static_key: static_profile_id,
    }
    combined_profiles = json.loads(
        (destination / result["calibration_profiles"]).read_text()
    )["profiles"]
    assert {
        (item["sensor_id"], item["mounting_mode"], item["profile_id"])
        for item in combined_profiles
    } == {
        (SENSOR_ID, "eye_in_hand", PROFILE_ID),
        (static_sensor_id, "static", static_profile_id),
    }

    saved = client.post(
        "/run-config",
        json={
            "run_root": destination.as_posix(),
            "run_name": "mixed-camera-object-run",
            "resolution": "720p",
            "fps": 6,
            "velocity_m_s": 0.2,
            "sensors": [sensor.to_dict() for sensor in destination_sensors],
            "dataset_mode": "objectless",
            "calibration_profiles": result["calibration_profiles"],
            "expected_calibration_bundle_sha256": selection["source"]["bundle_sha256"],
            "robot_pose_sunrise_reference_frame_path": STATIC_REFERENCE_PATH,
            "sequence_id": "calibrated_capture_to_bop_dataset_dry_run",
            "plan_only": True,
        },
    )
    assert saved.status_code == 201, saved.get_json()
    config = load_run_config_for_run_root(destination)
    assert {
        sensor["device_id"]: sensor["calibration_profile_id"]
        for sensor in config["capture"]["sensors"]
    } == {
        SENSOR_ID: PROFILE_ID,
        static_sensor_id: static_profile_id,
    }
    mobile_folder = sensor_folder_name("realsense_d435", SENSOR_ID)
    static_folder = sensor_folder_name("realsense_d435", static_sensor_id)
    profile_ids_by_sensor_name = selected_calibration_profile_ids_by_sensor_folder(
        destination
    )
    assert profile_ids_by_sensor_name == {
        mobile_folder: PROFILE_ID,
        static_folder: static_profile_id,
    }
    snapshot_profiles = load_profile_collection(
        destination / result["calibration_profiles"]
    )
    assert {
        sensor_name: calibration_profile_for_sensor(
            snapshot_profiles,
            sensor_name,
            profile_ids_by_sensor_name=profile_ids_by_sensor_name,
        ).profile_id
        for sensor_name in (mobile_folder, static_folder)
    } == profile_ids_by_sensor_name
    assert {
        sensor_name: transform["profile_id"]
        for sensor_name, transform in blenderproc_camera_transform_map_from_profiles(
            snapshot_profiles,
            (mobile_folder, static_folder),
            profile_ids_by_sensor_name=profile_ids_by_sensor_name,
        ).items()
    } == profile_ids_by_sensor_name

    changed_reference = client.post(
        "/run-config",
        json={
            "run_root": destination.as_posix(),
            "robot_pose_sunrise_reference_frame_path": (
                "/PoseTestBot/TemplateBase"
            ),
        },
    )
    assert changed_reference.status_code == 409
    assert changed_reference.get_json()["issues"][0]["code"] == (
        "calibration_selection_setup_mismatch"
    )
    assert (
        load_run_config_for_run_root(destination)["frames"]["robot_pose"]
        ["sunrise_reference_frame_path"]
        == STATIC_REFERENCE_PATH
    )

    atomic_write_json(mobile_source / CALIBRATION_PROFILES, {"changed": True})
    verified = verify_calibration_profile_selection(destination)
    assert verified["schema_version"] == "calibration_profile_selection.v2"


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    [
        ("source_count", "source_count does not match sources"),
        ("source_bundle", "bundle_sha256 does not match its artifact hashes"),
        ("artifact_hash", "bundle_sha256 does not match its artifact hashes"),
        ("source_run_root", "run_root is not canonical"),
        ("source_run_name", "source 0 run_name is invalid"),
        ("aggregate_run_name", "selection source run_name is invalid"),
    ],
)
def test_composite_selection_verifier_binds_complete_source_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    error_match: str,
) -> None:
    runs = tmp_path / "runs"
    source = runs / "calibration_run"
    destination = runs / "object_run"
    _write_source(source)
    _write_destination(destination)
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", runs.as_posix())
    client = create_app().test_client()
    candidate = _candidate(
        client.get(
            "/ui/calibrations", query_string={"run_root": destination}
        ).get_json(),
        source,
    )
    response = client.post(
        "/ui/calibrations/select",
        json=_composite_selection_request(
            destination, source, candidate["bundle_sha256"]
        ),
    )
    assert response.status_code == 201, response.get_json()
    selection_path = destination / CALIBRATION_PROFILE_SELECTION
    selection = json.loads(selection_path.read_text())
    assert selection["schema_version"] == "calibration_profile_selection.v2"

    if mutation == "source_count":
        selection["source"]["source_count"] = 2
    elif mutation == "source_bundle":
        selection["sources"][0]["bundle_sha256"] = "0" * 64
    elif mutation == "artifact_hash":
        selection["sources"][0]["calibration_profiles"]["sha256"] = "1" * 64
    elif mutation == "source_run_root":
        selection["sources"][0]["run_root"] = (
            f"{source.parent.as_posix()}/./{source.name}"
        )
    elif mutation == "source_run_name":
        selection["sources"][0]["run_name"] = " "
    else:
        selection["source"]["run_name"] = "Combined calibration"
    atomic_write_json(selection_path, selection)

    with pytest.raises(ValueError, match=error_match):
        verify_calibration_profile_selection(destination, verify_run_config=False)


def test_library_selected_status_requires_current_run_config_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    source = runs / "calibration_run"
    destination = runs / "object_run"
    _write_source(source)
    _write_destination(destination)
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", runs.as_posix())
    client = create_app().test_client()
    candidate = _candidate(
        client.get(
            "/ui/calibrations", query_string={"run_root": destination}
        ).get_json(),
        source,
    )
    selected_response = client.post(
        "/ui/calibrations/select",
        json=_selection_request(destination, source, candidate["bundle_sha256"]),
    )
    assert selected_response.status_code == 201, selected_response.get_json()
    selected = selected_response.get_json()
    saved = client.post(
        "/run-config",
        json={
            "run_root": destination.as_posix(),
            "run_name": "object_run",
            "resolution": "720p",
            "fps": 6,
            "velocity_m_s": 0.2,
            "sensors": [_sensor().to_dict()],
            "dataset_mode": "objectless",
            "calibration_profiles": selected["calibration_profiles"],
            "expected_calibration_bundle_sha256": candidate["bundle_sha256"],
            "sequence_id": "calibrated_capture_to_bop_dataset_dry_run",
            "plan_only": True,
        },
    )
    assert saved.status_code == 201, saved.get_json()
    listed = client.get(
        "/ui/calibrations", query_string={"run_root": destination}
    ).get_json()
    assert listed["selected"]["valid"] is True

    config = load_run_config_for_run_root(destination)
    config["capture"]["sensors"][0]["mounting_mode"] = "static"
    atomic_write_json(destination / "run_config.json", config)

    listed = client.get(
        "/ui/calibrations", query_string={"run_root": destination}
    ).get_json()
    assert listed["selected"]["valid"] is False
    assert listed["selected"]["issues"] == [
        {
            "code": "invalid_current_selection",
            "message": (
                "Current calibration selection is invalid: Run config camera "
                "setup or Sunrise robot-pose reference no longer matches the "
                "calibration selection"
            ),
        }
    ]


@pytest.mark.parametrize(
    ("profile_reference", "destination_reference", "issue_code"),
    [
        (
            STATIC_REFERENCE_PATH,
            None,
            "destination_robot_pose_reference_unconfigured",
        ),
        (
            None,
            STATIC_REFERENCE_PATH,
            "static_calibration_reference_unverified",
        ),
        (
            "/PoseTestBot/TemplateBase",
            STATIC_REFERENCE_PATH,
            "static_calibration_not_in_pose_template_base",
        ),
        (
            STATIC_REFERENCE_PATH,
            "/PoseTestBot/TemplateBase",
            "destination_pose_template_base_reference_required",
        ),
        (
            "/PoseTestBot/TemplateBase",
            "/PoseTestBot/TemplateBase",
            "destination_pose_template_base_reference_required",
        ),
    ],
)
def test_static_calibration_library_fails_closed_without_matching_v1_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile_reference: str | None,
    destination_reference: str | None,
    issue_code: str,
) -> None:
    runs = tmp_path / "runs"
    source = runs / "static_calibration"
    destination = runs / "object_run"
    _write_source(
        source,
        mounting_mode=MountingMode.STATIC,
        static_reference_path=profile_reference,
    )
    destination.mkdir(parents=True)
    write_run_config(
        destination,
        create_run_config(
            run_root=destination,
            sensors=(_sensor(mounting_mode="static"),),
            robot_pose_sunrise_reference_frame_path=destination_reference,
        ),
    )
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", runs.as_posix())

    candidate = _candidate(
        create_app()
        .test_client()
        .get("/ui/calibrations", query_string={"run_root": destination})
        .get_json(),
        source,
    )

    assert candidate["valid"] is True
    assert candidate["compatible"] is False
    assert candidate["issues"][0]["code"] == issue_code


def test_static_selection_rechecks_destination_raw_pose_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    source = runs / "static_calibration"
    destination = runs / "object_run"
    _write_source(source, mounting_mode=MountingMode.STATIC)
    _write_destination(destination, mounting_mode="static")
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", runs.as_posix())
    client = create_app().test_client()
    candidate = _candidate(
        client.get(
            "/ui/calibrations", query_string={"run_root": destination}
        ).get_json(),
        source,
    )
    assert candidate["compatible"] is True
    selected = client.post(
        "/ui/calibrations/select",
        json=_selection_request(destination, source, candidate["bundle_sha256"]),
    )
    assert selected.status_code == 201, selected.get_json()

    _write_raw_pose_reference(destination, STATIC_REFERENCE_PATH)
    verify_calibration_profile_selection(destination, verify_run_config=False)

    _write_raw_pose_reference(destination, "/PoseTestBot/TemplateBase")
    with pytest.raises(ValueError, match="does not match the selected static"):
        verify_calibration_profile_selection(destination, verify_run_config=False)
    listed = client.get(
        "/ui/calibrations", query_string={"run_root": destination}
    ).get_json()
    assert listed["selected"]["valid"] is False
    assert "raw robot-pose Sunrise reference frame" in listed["selected"]["issues"][
        0
    ]["message"]


def test_composite_selection_same_bundle_requires_replacement_for_membership_change(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    source = runs / "mixed_calibration"
    destination = runs / "object_run"
    static_sensor_id = "D435-STATIC-456"
    static_profile_id = "d435_static_456_valid"
    static_intrinsic_id = "D435-STATIC-456_1280x720_normal_opencv"
    source_sensors = (
        _sensor(),
        _sensor(mounting_mode="static", sensor_id=static_sensor_id),
    )
    source.mkdir(parents=True)
    write_profile_collection(
        [
            _profile(),
            _profile(
                mounting_mode=MountingMode.STATIC,
                sensor_id=static_sensor_id,
                profile_id=static_profile_id,
                intrinsic_id=static_intrinsic_id,
            ),
        ],
        source / CALIBRATION_PROFILES,
    )
    write_intrinsic_profile_collection(
        [
            _intrinsic_profile(),
            _intrinsic_profile(
                sensor_id=static_sensor_id,
                intrinsic_id=static_intrinsic_id,
            ),
        ],
        source / INTRINSIC_CALIBRATION_PROFILES,
    )
    write_run_config(
        source,
        create_run_config(
            run_root=source,
            run_name="Mixed reusable calibration",
            sensors=source_sensors,
        ),
    )
    destination.mkdir(parents=True)
    write_run_config(
        destination,
        create_run_config(
            run_root=destination,
            sensors=source_sensors,
            robot_pose_sunrise_reference_frame_path=STATIC_REFERENCE_PATH,
        ),
    )
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", runs.as_posix())
    client = create_app().test_client()
    candidate = _candidate(
        client.get(
            "/ui/calibrations", query_string={"run_root": destination}
        ).get_json(),
        source,
    )
    mobile_key = f"realsense_d435:{SENSOR_ID}"
    static_key = f"realsense_d435:{static_sensor_id}"
    full_request = {
        "run_root": destination.as_posix(),
        "source_selections": [
            {
                "source_run_root": source.as_posix(),
                "expected_bundle_sha256": candidate["bundle_sha256"],
                "sensor_keys": [mobile_key, static_key],
            }
        ],
        "resolution": "720p",
        "sensors": [sensor.to_dict() for sensor in source_sensors],
    }
    selected = client.post("/ui/calibrations/select", json=full_request)
    assert selected.status_code == 201, selected.get_json()
    selected_payload = selected.get_json()

    stale_membership_save = client.post(
        "/run-config",
        json={
            "run_root": destination.as_posix(),
            "run_name": "object_run",
            "resolution": "720p",
            "fps": 6,
            "velocity_m_s": 0.2,
            "sensors": [_sensor().to_dict()],
            "dataset_mode": "objectless",
            "calibration_profiles": selected_payload["calibration_profiles"],
            "expected_calibration_bundle_sha256": candidate["bundle_sha256"],
            "sequence_id": "calibrated_capture_to_bop_dataset_dry_run",
            "plan_only": True,
        },
    )
    assert stale_membership_save.status_code == 409
    assert stale_membership_save.get_json()["issues"][0]["code"] == (
        "calibration_selection_setup_mismatch"
    )

    reduced_request = {
        "run_root": destination.as_posix(),
        "source_selections": [
            {
                "source_run_root": source.as_posix(),
                "expected_bundle_sha256": candidate["bundle_sha256"],
                "sensor_keys": [mobile_key],
            }
        ],
        "resolution": "720p",
        "sensors": [_sensor().to_dict()],
        "expected_current_bundle_sha256": candidate["bundle_sha256"],
    }
    unconfirmed = client.post("/ui/calibrations/select", json=reduced_request)
    assert unconfirmed.status_code == 409
    assert unconfirmed.get_json()["issues"][0]["code"] == (
        "calibration_replacement_confirmation_required"
    )

    replaced = client.post(
        "/ui/calibrations/select",
        json={**reduced_request, "confirm_replace": True},
    )
    assert replaced.status_code == 201, replaced.get_json()
    replacement = replaced.get_json()
    assert replacement["idempotent"] is False
    assert (
        replacement["selection"]["source"]["bundle_sha256"]
        == candidate["bundle_sha256"]
    )
    assert replacement["sensor_profiles"] == {mobile_key: PROFILE_ID}
    assert replacement["selection"]["sources"][0]["selected_sensor_keys"] == [
        mobile_key
    ]

    saved = client.post(
        "/run-config",
        json={
            "run_root": destination.as_posix(),
            "run_name": "object_run",
            "resolution": "720p",
            "fps": 6,
            "velocity_m_s": 0.2,
            "sensors": [
                {
                    **_sensor().to_dict(),
                    "operator_alias": "Dataset wrist camera",
                }
            ],
            "dataset_mode": "objectless",
            "calibration_profiles": replacement["calibration_profiles"],
            "expected_calibration_bundle_sha256": candidate["bundle_sha256"],
            "sequence_id": "calibrated_capture_to_bop_dataset_dry_run",
            "plan_only": True,
        },
    )
    assert saved.status_code == 201, saved.get_json()
    assert (
        load_run_config_for_run_root(destination)["capture"]["sensors"][0][
            "operator_alias"
        ]
        == "Dataset wrist camera"
    )


def test_selection_same_bundle_from_another_source_is_not_idempotent(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    source_a = runs / "calibration_a"
    source_b = runs / "calibration_b"
    destination = runs / "object_run"
    _write_source(source_a)
    _write_source(source_b)
    _write_destination(destination)
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", runs.as_posix())
    client = create_app().test_client()
    library = client.get(
        "/ui/calibrations", query_string={"run_root": destination}
    ).get_json()
    candidate_a = _candidate(library, source_a)
    candidate_b = _candidate(library, source_b)
    assert candidate_a["bundle_sha256"] == candidate_b["bundle_sha256"]

    selected_a = client.post(
        "/ui/calibrations/select",
        json=_selection_request(destination, source_a, candidate_a["bundle_sha256"]),
    )
    assert selected_a.status_code == 201

    replacement_request = {
        **_selection_request(
            destination,
            source_b,
            candidate_b["bundle_sha256"],
        ),
        "expected_current_bundle_sha256": candidate_a["bundle_sha256"],
    }
    unconfirmed = client.post(
        "/ui/calibrations/select",
        json=replacement_request,
    )
    assert unconfirmed.status_code == 409
    assert unconfirmed.get_json()["issues"][0]["code"] == (
        "calibration_replacement_confirmation_required"
    )

    replaced = client.post(
        "/ui/calibrations/select",
        json={
            **replacement_request,
            "confirm_replace": True,
        },
    )
    assert replaced.status_code == 201, replaced.get_json()
    assert replaced.get_json()["idempotent"] is False
    assert replaced.get_json()["selection"]["source"]["run_root"] == (
        source_b.as_posix()
    )


def test_selection_accepts_intended_setup_before_destination_config(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    source = runs / "legacy_calibration"
    destination = runs / "new_object_run"
    _write_source(source, legacy=True)
    runs.mkdir(exist_ok=True)
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", runs.as_posix())
    client = create_app().test_client()

    listed = client.get("/ui/calibrations", query_string={"run_root": destination})
    candidate = listed.get_json()["calibrations"][0]
    assert candidate["valid"] is True
    assert candidate["compatible"] is False
    assert candidate["issues"][0]["code"] == "destination_setup_required"

    selected = client.post(
        "/ui/calibrations/select",
        json={
            "run_root": destination.as_posix(),
            "source_run_root": source.as_posix(),
            "expected_bundle_sha256": candidate["bundle_sha256"],
            "resolution": "720p",
            "sensors": [_sensor().to_dict()],
        },
    )

    assert selected.status_code == 201, selected.get_json()
    assert (
        selected.get_json()["selection"]["source"]["calibration_profiles"][
            "schema_version"
        ]
        == "calibration.v1"
    )


def test_selection_rejects_incompatible_override_and_stale_hash(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    source = runs / "calibration_run"
    destination = runs / "object_run"
    _write_source(source)
    _write_destination(destination)
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", runs.as_posix())
    client = create_app().test_client()
    candidate = client.get(
        "/ui/calibrations", query_string={"run_root": destination}
    ).get_json()["calibrations"][0]

    incompatible_sensor = _sensor(mounting_mode="static").to_dict()
    incompatible = client.post(
        "/ui/calibrations/select",
        json={
            "run_root": destination.as_posix(),
            "source_run_root": source.as_posix(),
            "expected_bundle_sha256": candidate["bundle_sha256"],
            "resolution": "720p",
            "sensors": [incompatible_sensor],
        },
    )
    assert incompatible.status_code == 409
    assert incompatible.get_json()["issues"][0]["code"] == "mounting_mode_mismatch"
    assert not (destination / CALIBRATION_PROFILE_SELECTION).exists()

    stale = client.post(
        "/ui/calibrations/select",
        json={
            "run_root": destination.as_posix(),
            "source_run_root": source.as_posix(),
            "expected_bundle_sha256": "0" * 64,
        },
    )
    assert stale.status_code == 409
    assert stale.get_json()["issues"] == [
        {
            "code": "stale_calibration_bundle",
            "message": "The source calibration hashes changed after the library was loaded.",
        }
    ]


def test_library_does_not_follow_symlinked_profile_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    source = runs / "unsafe_calibration"
    destination = runs / "object_run"
    safe_source = runs / "safe_calibration"
    _write_source(safe_source)
    source.mkdir(parents=True)
    (source / CALIBRATION_PROFILES).symlink_to(safe_source / CALIBRATION_PROFILES)
    (source / INTRINSIC_CALIBRATION_PROFILES).symlink_to(
        safe_source / INTRINSIC_CALIBRATION_PROFILES
    )
    _write_destination(destination)
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", runs.as_posix())
    client = create_app().test_client()

    payload = client.get(
        "/ui/calibrations", query_string={"run_root": destination}
    ).get_json()
    unsafe = next(
        item
        for item in payload["calibrations"]
        if item["source_run_root"] == source.as_posix()
    )
    assert unsafe["valid"] is False
    assert unsafe["compatible"] is False
    assert unsafe["issues"][0]["code"] == "invalid_calibration_bundle"


@pytest.mark.parametrize(
    "mismatch",
    ("native_k", "native_distortion", "rectified_k", "rectified_roi", "depth_scale"),
)
def test_library_rejects_cross_paired_projection_evidence(
    tmp_path: Path, monkeypatch, mismatch: str
) -> None:
    runs = tmp_path / "runs"
    source = runs / "cross_paired_calibration"
    destination = runs / "object_run"
    _write_source(source)
    _write_destination(destination)
    intrinsic_path = source / INTRINSIC_CALIBRATION_PROFILES
    collection = json.loads(intrinsic_path.read_text())
    intrinsic = collection["profiles"][0]
    if mismatch == "native_k":
        intrinsic["native"]["cam_K"][0] += 1.0
    elif mismatch == "native_distortion":
        intrinsic["native"]["distortion"][0] += 0.001
    elif mismatch == "rectified_k":
        intrinsic["rectified"]["cam_K"][0] += 1.0
    elif mismatch == "rectified_roi":
        intrinsic["rectified"]["valid_roi"][2] -= 1
    else:
        intrinsic["depth"]["scale_to_mm"] = 2.0
    atomic_write_json(intrinsic_path, collection)
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", runs.as_posix())
    client = create_app().test_client()

    payload = client.get(
        "/ui/calibrations", query_string={"run_root": destination}
    ).get_json()
    candidate = _candidate(payload, source)

    assert candidate["valid"] is True
    assert candidate["compatible"] is False
    assert candidate["issues"][0]["code"] == (
        "calibration_intrinsic_projection_mismatch"
    )
    rejected = client.post(
        "/ui/calibrations/select",
        json=_selection_request(destination, source, candidate["bundle_sha256"]),
    )
    assert rejected.status_code == 409
    assert rejected.get_json()["issues"][0]["code"] == (
        "calibration_intrinsic_projection_mismatch"
    )


def test_snapshot_verifier_detects_tampering_and_publication_is_read_only(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    source = runs / "calibration_run"
    destination = runs / "object_run"
    _write_source(source)
    _write_destination(destination)
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", runs.as_posix())
    client = create_app().test_client()
    candidate = _candidate(
        client.get(
            "/ui/calibrations", query_string={"run_root": destination}
        ).get_json(),
        source,
    )
    response = client.post(
        "/ui/calibrations/select",
        json=_selection_request(destination, source, candidate["bundle_sha256"]),
    )
    assert response.status_code == 201
    selected = response.get_json()
    snapshot_path = destination / selected["calibration_profiles"]
    snapshot_directory = snapshot_path.parent
    assert stat.S_IMODE(snapshot_path.stat().st_mode) & 0o222 == 0
    assert stat.S_IMODE(snapshot_directory.stat().st_mode) & 0o222 == 0
    verify_calibration_profile_selection(destination, verify_run_config=False)

    os.chmod(snapshot_path, 0o644)
    snapshot_path.write_bytes(b"{}\n")

    with pytest.raises(ValueError, match="snapshot hash changed"):
        verify_calibration_profile_selection(destination, verify_run_config=False)
    listed = client.get(
        "/ui/calibrations", query_string={"run_root": destination}
    ).get_json()
    assert listed["selected"]["valid"] is False
    assert listed["selected"]["issues"][0]["code"] == "invalid_current_selection"


def test_snapshot_verifier_rejects_lexically_rebound_path_identity(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    source = runs / "calibration_run"
    destination = runs / "object_run"
    _write_source(source)
    _write_destination(destination)
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", runs.as_posix())
    client = create_app().test_client()
    candidate = _candidate(
        client.get(
            "/ui/calibrations", query_string={"run_root": destination}
        ).get_json(),
        source,
    )
    selected = client.post(
        "/ui/calibrations/select",
        json=_selection_request(destination, source, candidate["bundle_sha256"]),
    )
    assert selected.status_code == 201
    selection_path = destination / CALIBRATION_PROFILE_SELECTION
    selection = json.loads(selection_path.read_text())
    original = selection["snapshot"]["calibration_profiles"]["relative_path"]
    selection["snapshot"]["calibration_profiles"]["relative_path"] = f"./{original}"
    atomic_write_json(selection_path, selection)

    with pytest.raises(ValueError, match="path identity changed"):
        verify_calibration_profile_selection(destination, verify_run_config=False)


def test_selection_replacement_requires_cas_confirmation_and_blocks_after_capture(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    source_a = runs / "calibration_a"
    source_b = runs / "calibration_b"
    destination = runs / "object_run"
    _write_source(source_a, bundle_marker="A")
    _write_source(source_b, bundle_marker="B")
    _write_destination(destination)
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", runs.as_posix())
    client = create_app().test_client()
    library = client.get(
        "/ui/calibrations", query_string={"run_root": destination}
    ).get_json()
    candidate_a = _candidate(library, source_a)
    candidate_b = _candidate(library, source_b)

    selected_a = client.post(
        "/ui/calibrations/select",
        json=_selection_request(destination, source_a, candidate_a["bundle_sha256"]),
    )
    assert selected_a.status_code == 201
    assert selected_a.get_json()["idempotent"] is False
    retry_a = client.post(
        "/ui/calibrations/select",
        json=_selection_request(destination, source_a, candidate_a["bundle_sha256"]),
    )
    assert retry_a.status_code == 201
    assert retry_a.get_json()["idempotent"] is True

    no_cas = client.post(
        "/ui/calibrations/select",
        json=_selection_request(destination, source_b, candidate_b["bundle_sha256"]),
    )
    assert no_cas.status_code == 409
    assert no_cas.get_json()["issues"][0]["code"] == (
        "stale_current_calibration_bundle"
    )
    no_confirmation = client.post(
        "/ui/calibrations/select",
        json={
            **_selection_request(destination, source_b, candidate_b["bundle_sha256"]),
            "expected_current_bundle_sha256": candidate_a["bundle_sha256"],
        },
    )
    assert no_confirmation.status_code == 409
    assert no_confirmation.get_json()["issues"][0]["code"] == (
        "calibration_replacement_confirmation_required"
    )
    selected_b = client.post(
        "/ui/calibrations/select",
        json={
            **_selection_request(destination, source_b, candidate_b["bundle_sha256"]),
            "expected_current_bundle_sha256": candidate_a["bundle_sha256"],
            "confirm_replace": True,
        },
    )
    assert selected_b.status_code == 201, selected_b.get_json()

    atomic_write_json(destination / RAW_ROBOT_EE_POSES, {"poses": []})
    blocked = client.post(
        "/ui/calibrations/select",
        json={
            **_selection_request(destination, source_a, candidate_a["bundle_sha256"]),
            "expected_current_bundle_sha256": candidate_b["bundle_sha256"],
            "confirm_replace": True,
        },
    )
    assert blocked.status_code == 409
    issue = blocked.get_json()["issues"][0]
    assert issue["code"] == "calibration_replacement_blocked"
    assert issue["blockers"] == [RAW_ROBOT_EE_POSES]


def test_run_config_bundle_cas_rejects_selection_change_and_old_snapshot_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    source_a = runs / "calibration_a"
    source_b = runs / "calibration_b"
    destination = runs / "object_run"
    _write_source(source_a, bundle_marker="A")
    _write_source(source_b, bundle_marker="B")
    _write_destination(destination)
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", runs.as_posix())
    client = create_app().test_client()
    library = client.get(
        "/ui/calibrations", query_string={"run_root": destination}
    ).get_json()
    candidate_a = _candidate(library, source_a)
    candidate_b = _candidate(library, source_b)
    selected_a = client.post(
        "/ui/calibrations/select",
        json=_selection_request(destination, source_a, candidate_a["bundle_sha256"]),
    ).get_json()
    old_path = selected_a["calibration_profiles"]

    replaced = client.post(
        "/ui/calibrations/select",
        json={
            **_selection_request(destination, source_b, candidate_b["bundle_sha256"]),
            "expected_current_bundle_sha256": candidate_a["bundle_sha256"],
            "confirm_replace": True,
        },
    )
    assert replaced.status_code == 201
    save_payload = {
        "run_root": destination.as_posix(),
        "run_name": "object_run",
        "resolution": "720p",
        "fps": 6,
        "velocity_m_s": 0.2,
        "sensors": [_sensor().to_dict()],
        "dataset_mode": "objectless",
        "calibration_profiles": old_path,
        "expected_calibration_bundle_sha256": candidate_a["bundle_sha256"],
        "sequence_id": "calibrated_capture_to_bop_dataset_dry_run",
        "plan_only": True,
    }

    stale = client.post("/run-config", json=save_payload)

    assert stale.status_code == 409
    assert stale.get_json()["issues"][0]["code"] == "stale_calibration_selection"
    assert load_run_config_for_run_root(destination)["calibration_profiles"] is None

    without_explicit_cas = dict(save_payload)
    without_explicit_cas.pop("expected_calibration_bundle_sha256")
    rebound = client.post("/run-config", json=without_explicit_cas)
    assert rebound.status_code == 409
    assert rebound.get_json()["issues"][0]["code"] == (
        "calibration_selection_path_mismatch"
    )

    omitted_path = {
        **save_payload,
        "calibration_profiles": None,
        "expected_calibration_bundle_sha256": candidate_b["bundle_sha256"],
    }
    missing = client.post("/run-config", json=omitted_path)
    assert missing.status_code == 409
    assert missing.get_json()["issues"][0]["code"] == (
        "calibration_selection_path_missing"
    )


def test_auto_device_id_does_not_wildcard_match_a_physical_profile(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    source = runs / "calibration_run"
    destination = runs / "object_run"
    _write_source(source)
    destination.mkdir(parents=True)
    write_run_config(
        destination,
        create_run_config(
            run_root=destination,
            sensors=(
                SensorRunConfig(
                    sensor_type="realsense_d435",
                    device_id="auto",
                    display_name="Auto D435",
                ),
            ),
        ),
    )
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", runs.as_posix())
    client = create_app().test_client()

    payload = client.get(
        "/ui/calibrations", query_string={"run_root": destination}
    ).get_json()
    candidate = _candidate(payload, source)

    assert candidate["valid"] is True
    assert candidate["compatible"] is False
    assert candidate["issues"][0]["code"] == "sensor_identity_not_calibrated"
