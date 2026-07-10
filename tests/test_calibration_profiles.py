from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from posetestbot.calibration.profiles import (
    SCHEMA_VERSION,
    CalibrationProfile,
    CalibrationQuality,
    CalibrationStatus,
    CalibrationTargetType,
    RigidTransform,
    TransformFrame,
    blenderproc_camera_transform_map_from_profiles,
    load_profile,
    load_profile_collection,
    migrate_legacy_camera_ee_profiles,
    write_profile,
    write_profile_collection,
)
import pytest
from posetestbot.sensors.contracts import CameraIntrinsics, MountingMode, SensorType


def static_profile() -> CalibrationProfile:
    return CalibrationProfile(
        schema_version=SCHEMA_VERSION,
        profile_id="zed_2i_SN0001_static_cell_top_v2026_01",
        sensor_id="SN0001",
        sensor_type=SensorType.ZED_2I,
        mounting_mode=MountingMode.STATIC,
        rig_position="cell_top",
        intrinsics=CameraIntrinsics(
            cam_k=(1.0, 0.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 1.0),
            width=1280,
            height=720,
            distortion=(0.1, 0.2, 0.0, 0.0, 0.0),
            depth_scale_to_mm=1.0,
        ),
        rectified_intrinsics=CameraIntrinsics(
            cam_k=(1.1, 0.0, 2.0, 0.0, 3.1, 4.0, 0.0, 0.0, 1.0),
            width=1280,
            height=720,
            distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
            depth_scale_to_mm=1.0,
        ),
        extrinsics=RigidTransform(
            from_frame=TransformFrame.CAMERA,
            to_frame=TransformFrame.ROBOT_BASE,
            rotation_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            translation_mm=(100.0, 200.0, 300.0),
        ),
        target_type=CalibrationTargetType.CHARUCO,
        method="fixture_static_solve",
        status=CalibrationStatus.VALID,
        quality=CalibrationQuality(
            num_observations=12,
            num_inliers=11,
            mean_reprojection_error_px=0.4,
        ),
        sync_delta_ms=2.5,
    )


def test_calibration_profile_round_trips_with_baseline_json_keys(tmp_path: Path) -> None:
    profile = static_profile()
    path = tmp_path / "profile.json"

    write_profile(profile, path)
    value = json.loads(path.read_text())

    assert value["schema_version"] == "calibration.v2"
    assert value["intrinsics"]["native"]["cam_K"] == [1.0, 0.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 1.0]
    assert value["intrinsics"]["rectified"]["distortion"] == [0.0] * 5
    assert value["extrinsics"]["from"] == "camera"
    assert value["extrinsics"]["to"] == "template_base"

    loaded = load_profile(path)
    assert loaded.profile_id == profile.profile_id
    assert loaded.intrinsics == profile.intrinsics
    assert loaded.rectified_intrinsics == profile.rectified_intrinsics
    assert loaded.rectified_valid_roi == tuple(value["intrinsics"]["rectified"]["valid_roi"])


def test_eye_in_hand_profile_requires_camera_to_robot_flange() -> None:
    profile = static_profile()
    invalid = replace(profile, mounting_mode=MountingMode.EYE_IN_HAND)

    try:
        invalid.validate()
    except ValueError as exc:
        assert "eye_in_hand calibration must transform camera to robot_flange" in str(exc)
    else:
        raise AssertionError("invalid eye-in-hand transform direction was accepted")


def test_migrate_legacy_camera_ee_profiles() -> None:
    profiles = migrate_legacy_camera_ee_profiles(
        {
            "realsense": {
                "quaternion": [1, 0, 0, 0],
                "position": [1, 2, 3],
            },
            "luxonis": {
                "quaternion": [0, 1, 0, 0],
                "position": [4, 5, 6],
            },
        },
        sync_deltas_ms={"realsense": 10.5, "luxonis": 20.5},
    )

    by_sensor = {profile.sensor_id: profile for profile in profiles}
    assert by_sensor["realsense"].mounting_mode == MountingMode.EYE_IN_HAND
    assert by_sensor["realsense"].extrinsics.to_frame == TransformFrame.END_EFFECTOR
    assert by_sensor["realsense"].sync_delta_ms == 10.5
    assert by_sensor["realsense"].metadata["legacy_sensor_key"] == "realsense"
    assert by_sensor["luxonis"].sensor_type == SensorType.OAK_D_PRO


def test_blenderproc_transform_map_accepts_static_profiles() -> None:
    profile = static_profile()

    transform_map = blenderproc_camera_transform_map_from_profiles(
        [profile],
        ["zed_2i_SN0001"],
    )

    entry = transform_map["zed_2i_SN0001"]
    assert entry["mounting_mode"] == "static"
    assert entry["to"] == "template_base"
    assert entry["profile_id"] == "zed_2i_SN0001_static_cell_top_v2026_01"
    assert entry["position"] == [100.0, 200.0, 300.0]


def test_validate_calibration_profiles_cli_migrates_legacy_defaults(
    tmp_path: Path,
) -> None:
    camera_ee = tmp_path / "camera_ee_transform.json"
    sync_data = tmp_path / "sync_data.json"
    output = tmp_path / "calibration_profiles.json"
    camera_ee.write_text(
        json.dumps({"realsense": {"quaternion": [1, 0, 0, 0], "position": [1, 2, 3]}})
    )
    sync_data.write_text(json.dumps({"realsense": 112.5}))
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "validate_calibration_profiles.py"),
            "--legacy-camera-ee",
            str(camera_ee),
            "--legacy-sync-data",
            str(sync_data),
            "--output",
            str(output),
            "--json",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    summary = json.loads(result.stdout)
    assert summary["status"] == "valid"
    assert summary["profile_count"] == 1

    profiles = load_profile_collection(output)
    assert profiles[0].profile_id == "realsense_d435_realsense_eye_in_hand_wrist_legacy"
    assert profiles[0].sync_delta_ms == 112.5


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        (
            replace(
                static_profile(),
                extrinsics=replace(
                    static_profile().extrinsics,
                    rotation_quaternion_wxyz=(2.0, 0.0, 0.0, 0.0),
                ),
            ),
            "normalized",
        ),
        (
            replace(
                static_profile(),
                intrinsics=replace(
                    static_profile().intrinsics,
                    depth_scale_to_mm=float("nan"),
                ),
            ),
            "depth_scale_to_mm",
        ),
        (
            replace(
                static_profile(),
                quality=replace(
                    static_profile().quality,
                    residual_translation_mm=-1.0,
                ),
            ),
            "nonnegative",
        ),
    ],
)
def test_calibration_profile_rejects_invalid_numeric_contracts(
    profile: CalibrationProfile, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        profile.validate()


def test_profile_collection_rejects_duplicate_valid_sensor_slot(
    tmp_path: Path,
) -> None:
    first = static_profile()
    second = replace(first, profile_id="second_valid_profile")

    with pytest.raises(ValueError, match="same sensor/mount/rig slot"):
        write_profile_collection([first, second], tmp_path / "profiles.json")


def test_calibration_v1_collection_loads_and_normalizes_explicit_frames(
    tmp_path: Path,
) -> None:
    legacy = {
        "schema_version": "calibration.v1",
        "profiles": [
            {
                "schema_version": "calibration.v1",
                "profile_id": "legacy_wrist",
                "sensor_id": "123",
                "sensor_type": "realsense_d435",
                "mounting_mode": "eye_in_hand",
                "rig_position": "wrist",
                "intrinsics": {
                    "cam_K": [600, 0, 320, 0, 600, 240, 0, 0, 1],
                    "width": 640,
                    "height": 480,
                    "distortion": [0.1, 0, 0, 0, 0],
                    "depth_scale_to_mm": 1.0,
                },
                "extrinsics": {
                    "from": "camera",
                    "to": "end_effector",
                    "rotation_quaternion_wxyz": [1, 0, 0, 0],
                    "translation_mm": [1, 2, 3],
                },
                "status": "needs_validation",
                "quality": {"num_observations": 1, "num_inliers": 1},
            }
        ],
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy))

    profile = load_profile_collection(path)[0]

    assert profile.schema_version == "calibration.v2"
    assert profile.extrinsics.to_frame == TransformFrame.ROBOT_FLANGE
    assert profile.rectified_intrinsics is not None
    assert profile.rectified_intrinsics.distortion == (0.0,) * 5
