from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from posetestbot.calibration import preflight as preflight_module
from posetestbot.calibration.preflight import (
    build_calibration_preflight,
    write_calibration_preflight_with_manifest,
)
from posetestbot.calibration.profiles import (
    SCHEMA_VERSION,
    CalibrationProfile,
    CalibrationQuality,
    CalibrationStatus,
    CalibrationTargetType,
    RigidTransform,
    TransformFrame,
    write_profile_collection,
)
from posetestbot.io.artifacts import CALIBRATION_PREFLIGHT_REPORT, DATASET_MANIFEST
from posetestbot.pipeline.preflight import build_run_preflight
from posetestbot.pipeline.run_config import (
    create_run_config,
    sensor_config_from_token,
    write_run_config,
)
from posetestbot.sensors.contracts import CameraIntrinsics, MountingMode, SensorType


def profile(
    *,
    profile_id: str,
    sensor_id: str,
    sensor_type: SensorType = SensorType.REALSENSE_D435,
    mounting_mode: MountingMode = MountingMode.EYE_IN_HAND,
    status: CalibrationStatus = CalibrationStatus.VALID,
    observations: int = 12,
    mean_error: float | None = 0.5,
) -> CalibrationProfile:
    return CalibrationProfile(
        schema_version=SCHEMA_VERSION,
        profile_id=profile_id,
        sensor_id=sensor_id,
        sensor_type=sensor_type,
        mounting_mode=mounting_mode,
        rig_position="fixture",
        intrinsics=CameraIntrinsics(
            cam_k=(1.0, 0.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 1.0),
            width=1280,
            height=720,
        ),
        extrinsics=RigidTransform(
            from_frame=TransformFrame.CAMERA,
            to_frame=(
                TransformFrame.END_EFFECTOR
                if mounting_mode == MountingMode.EYE_IN_HAND
                else TransformFrame.ROBOT_BASE
            ),
            rotation_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            translation_mm=(1.0, 2.0, 3.0),
        ),
        target_type=CalibrationTargetType.CHARUCO,
        method="fixture",
        status=status,
        quality=CalibrationQuality(
            num_observations=observations,
            num_inliers=max(0, observations - 1),
            mean_reprojection_error_px=mean_error,
        ),
    )


def test_calibration_preflight_requires_explicit_target_mounting_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "legacy-target"
    digest = "0" * 64
    calibration_target = {
        "target_id": "legacy-target",
        "bundle_path": "calibration_targets/legacy-target",
        "source_sha256": digest,
        "spec_sha256": digest,
        "pdf_sha256": digest,
        "configuration_sha256": digest,
        "geometry_sha256": digest,
        "placement": {"mode": "unknown"},
    }
    write_run_config(
        run_root,
        create_run_config(
            run_root=run_root,
            calibration_target=calibration_target,
        ),
    )
    calls = []

    def reject_legacy_selection(_run_root, **kwargs):
        calls.append(kwargs)
        raise ValueError("mounting_frame is missing")

    monkeypatch.setattr(
        preflight_module,
        "validate_run_target_selection",
        reject_legacy_selection,
    )

    report = build_calibration_preflight(run_root)

    target_check = next(
        item
        for item in report["checks"]
        if item["name"] == "calibration_target_selection"
    )
    assert calls == [{"require_mounting_frame": True}]
    assert target_check["status"] == "error"
    assert "mounting_frame is missing" in target_check["message"]


def test_calibration_preflight_matches_profiles_for_run_sensors(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    profiles_path = run_root / "calibration_profiles.json"
    config = create_run_config(
        run_root=run_root,
        sensors=(
            sensor_config_from_token("realsense:123:eye_in_hand:Cell RealSense"),
            sensor_config_from_token("oak:auto:eye_in_hand:Cell OAK-D Pro"),
        ),
        calibration_profiles=profiles_path.as_posix(),
    )
    write_run_config(run_root, config)
    write_profile_collection(
        [
            profile(profile_id="rs_123_valid", sensor_id="123"),
            profile(
                profile_id="oak_auto_valid",
                sensor_id="auto",
                sensor_type=SensorType.OAK_D_PRO,
            ),
        ],
        profiles_path,
    )

    report = build_calibration_preflight(run_root)

    assert report["schema_version"] == "calibration_preflight.v1"
    assert report["overall_status"] == "ok"
    assert report["profile_count"] == 2
    assert report["matched_sensor_count"] == 2
    assert [match["profile_id"] for match in report["matched_sensors"]] == [
        "rs_123_valid",
        "oak_auto_valid",
    ]


def test_calibration_preflight_warns_for_missing_collection(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:eye_in_hand:Cell RealSense"),),
    )
    write_run_config(run_root, config)

    report = build_calibration_preflight(run_root)

    assert report["overall_status"] == "warning"
    assert report["profile_count"] == 0
    assert report["matched_sensor_count"] == 0
    assert report["checks"][0]["name"] == "calibration_profiles_configured"


def test_calibration_preflight_errors_for_missing_explicit_profile(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    profiles_path = run_root / "calibration_profiles.json"
    sensor = replace(
        sensor_config_from_token("realsense:123:eye_in_hand:Cell RealSense"),
        calibration_profile_id="missing_profile",
    )
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor,),
        calibration_profiles=profiles_path.as_posix(),
    )
    write_run_config(run_root, config)
    write_profile_collection(
        [profile(profile_id="rs_123_valid", sensor_id="123")], profiles_path
    )

    report = build_calibration_preflight(run_root)

    assert report["overall_status"] == "error"
    match_check = next(
        check
        for check in report["checks"]
        if check["name"] == "profile_match:realsense_123"
    )
    assert "missing_profile" in match_check["message"]


@pytest.mark.parametrize(
    ("profile_kwargs", "expected_message"),
    [
        ({"sensor_id": "999"}, "belongs to sensor '999'"),
        ({"sensor_type": SensorType.OAK_D_PRO}, "belongs to sensor type 'oak_d_pro'"),
        (
            {"mounting_mode": MountingMode.STATIC},
            "uses mounting mode 'static'",
        ),
    ],
)
def test_calibration_preflight_rejects_explicit_profile_for_different_camera_identity(
    tmp_path: Path,
    profile_kwargs: dict,
    expected_message: str,
) -> None:
    run_root = tmp_path / "run"
    profiles_path = run_root / "calibration_profiles.json"
    sensor = replace(
        sensor_config_from_token("realsense:123:eye_in_hand:Cell RealSense"),
        calibration_profile_id="explicit_profile",
    )
    write_run_config(
        run_root,
        create_run_config(
            run_root=run_root,
            sensors=(sensor,),
            calibration_profiles=profiles_path.as_posix(),
        ),
    )
    write_profile_collection(
        [
            profile(
                profile_id="explicit_profile",
                **({"sensor_id": "123"} | profile_kwargs),
            )
        ],
        profiles_path,
    )

    report = build_calibration_preflight(run_root, require_valid=True)

    match_check = next(
        check
        for check in report["checks"]
        if check["name"] == "profile_match:realsense_123"
    )
    assert match_check["status"] == "error"
    assert expected_message in match_check["message"]


def test_calibration_preflight_require_valid_turns_status_warning_into_error(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    profiles_path = run_root / "calibration_profiles.json"
    sensor = replace(
        sensor_config_from_token("realsense:123:eye_in_hand:Cell RealSense"),
        calibration_profile_id="rs_needs_validation",
    )
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor,),
        calibration_profiles=profiles_path.as_posix(),
    )
    write_run_config(run_root, config)
    write_profile_collection(
        [
            profile(
                profile_id="rs_needs_validation",
                sensor_id="123",
                status=CalibrationStatus.NEEDS_VALIDATION,
                observations=3,
                mean_error=None,
            )
        ],
        profiles_path,
    )

    loose_report = build_calibration_preflight(run_root)
    strict_report = build_calibration_preflight(run_root, require_valid=True)

    assert loose_report["overall_status"] == "warning"
    assert strict_report["overall_status"] == "error"


def test_guided_dataset_run_preflight_includes_strict_calibration_readiness(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "guided-dataset"
    profiles_path = run_root / "calibration_profiles.json"
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:eye_in_hand:Cell RealSense"),),
        calibration_profiles=profiles_path.as_posix(),
        sequence_id="calibrated_capture_to_bop_dataset_dry_run",
        plan_only=False,
    )
    write_run_config(run_root, config)
    write_profile_collection(
        [
            profile(
                profile_id="rs_needs_validation",
                sensor_id="123",
                status=CalibrationStatus.NEEDS_VALIDATION,
            )
        ],
        profiles_path,
    )

    report = build_run_preflight(
        run_root,
        include_sensor_status=False,
        include_runtime_status=False,
        collect_robot=lambda: {
            "schema_version": "robot_status.v2",
            "selected_profile": {"mode": "real"},
        },
    )

    readiness = next(
        check for check in report["checks"] if check["name"] == "calibration_readiness"
    )
    selection = next(
        check
        for check in report["checks"]
        if check["name"] == "calibration_profile_selection"
    )
    assert readiness["status"] == "error"
    assert readiness["details"]["require_valid"] is True
    assert readiness["details"]["matched_sensor_count"] == 1
    assert selection["status"] == "error"
    assert selection["details"]["required_by_guided_workflow"] is True
    assert report["overall_status"] == "error"


def test_guided_dataset_run_rejects_valid_raw_profile_path_without_selection(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "guided-raw-profile"
    profiles_path = run_root / "legacy_profiles.json"
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:eye_in_hand:Cell RealSense"),),
        calibration_profiles=profiles_path.as_posix(),
        sequence_id="calibrated_capture_to_bop_dataset_dry_run",
        plan_only=False,
    )
    write_run_config(run_root, config)
    write_profile_collection(
        [profile(profile_id="rs_123_valid", sensor_id="123")], profiles_path
    )

    report = build_run_preflight(
        run_root,
        include_sensor_status=False,
        include_runtime_status=False,
        collect_robot=lambda: {
            "schema_version": "robot_status.v2",
            "selected_profile": {"mode": "real"},
        },
    )

    readiness = next(
        check for check in report["checks"] if check["name"] == "calibration_readiness"
    )
    selection = next(
        check
        for check in report["checks"]
        if check["name"] == "calibration_profile_selection"
    )
    assert readiness["status"] == "ok"
    assert selection["status"] == "error"
    assert selection["details"]["required_by_guided_workflow"] is True
    assert report["overall_status"] == "error"


def test_calibration_preflight_cli_writes_manifest_artifact(tmp_path: Path) -> None:
    run_root = tmp_path / "run-cli"
    repo_root = Path(__file__).resolve().parents[1]
    profiles_path = run_root / "calibration_profiles.json"
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:eye_in_hand:Cell RealSense"),),
        calibration_profiles=profiles_path.as_posix(),
    )
    write_run_config(run_root, config)
    write_profile_collection(
        [profile(profile_id="rs_123_valid", sensor_id="123")], profiles_path
    )

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_calibration_preflight.py",
            run_root.as_posix(),
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert f"Wrote {run_root / CALIBRATION_PREFLIGHT_REPORT}" in result.stdout
    assert "Calibration preflight: ok (1/1 sensors matched)" in result.stdout
    report = json.loads((run_root / CALIBRATION_PREFLIGHT_REPORT).read_text())
    assert report["overall_status"] == "ok"
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage
        for stage in manifest["stages"]
        if stage["name"] == "calibration_preflight"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][CALIBRATION_PREFLIGHT_REPORT] == (
        CALIBRATION_PREFLIGHT_REPORT
    )


def test_write_calibration_preflight_with_manifest_records_warning_stage(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-warning"
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:eye_in_hand:Cell RealSense"),),
    )
    write_run_config(run_root, config)

    path, report = write_calibration_preflight_with_manifest(run_root)

    assert path == run_root / CALIBRATION_PREFLIGHT_REPORT
    assert report["overall_status"] == "warning"
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage
        for stage in manifest["stages"]
        if stage["name"] == "calibration_preflight"
    )
    assert stage["status"] == "succeeded"
