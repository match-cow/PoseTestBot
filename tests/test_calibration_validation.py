from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

from posetestbot.calibration.profiles import (
    SCHEMA_VERSION as PROFILE_SCHEMA,
    CalibrationProfile,
    CalibrationQuality,
    CalibrationStatus,
    CalibrationTargetType,
    RigidTransform,
    TransformFrame,
    load_profile_collection,
    profile_to_dict,
    write_profile_collection,
)
from posetestbot.calibration.validation import (
    build_calibration_validation,
    write_calibration_validation_with_manifest,
)
from posetestbot.io.artifacts import (
    CALIBRATION_CANDIDATES,
    CALIBRATION_PROFILES,
    CALIBRATION_PROFILES_FROM_OBSERVATIONS,
    CALIBRATION_PROFILES_SOLVED,
    CALIBRATION_SOLVER_REPORT,
    CALIBRATION_VALIDATION_REPORT,
    DATASET_MANIFEST,
)
from posetestbot.sensors.contracts import CameraIntrinsics, MountingMode, SensorType


def candidate_profile(
    *,
    profile_id: str = "realsense_123_static_aruco_candidate",
    num_observations: int = 8,
    num_inliers: int = 7,
    residual_translation_mm: float = 1.2,
    residual_rotation_deg: float = 0.4,
    outlier_count: int = 1,
) -> CalibrationProfile:
    return CalibrationProfile(
        schema_version=PROFILE_SCHEMA,
        profile_id=profile_id,
        sensor_id="123",
        sensor_type=SensorType.REALSENSE_D435,
        mounting_mode=MountingMode.STATIC,
        rig_position="static",
        intrinsics=CameraIntrinsics(
            cam_k=(1.0, 0.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 1.0),
            width=1280,
            height=720,
        ),
        extrinsics=RigidTransform(
            from_frame=TransformFrame.CAMERA,
            to_frame=TransformFrame.ROBOT_BASE,
            rotation_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            translation_mm=(1.0, 2.0, 3.0),
        ),
        target_type=CalibrationTargetType.ARUCO_GRID,
        method="aruco_observation_transform_average",
        status=CalibrationStatus.NEEDS_VALIDATION,
        quality=CalibrationQuality(
            num_observations=num_observations,
            num_inliers=num_inliers,
            residual_translation_mm=residual_translation_mm,
            residual_rotation_deg=residual_rotation_deg,
        ),
        metadata={
            "sensor_name": "realsense_123",
            "outlier_count": outlier_count,
        },
    )


def write_candidate_fixture(
    run_root: Path,
    *,
    profile: CalibrationProfile | None = None,
) -> tuple[Path, Path]:
    profile = profile or candidate_profile()
    profiles_path = run_root / CALIBRATION_PROFILES_FROM_OBSERVATIONS
    write_profile_collection([profile], profiles_path)
    report = {
        "schema_version": "calibration_candidates.v1",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "run_root": run_root.as_posix(),
        "overall_status": "warning",
        "sensor_count": 1,
        "profile_count": 1,
        "candidate_count": profile.quality.num_observations,
        "inlier_count": profile.quality.num_inliers,
        "outlier_count": profile.metadata["outlier_count"],
        "profiles": [profile_to_dict(profile)],
        "checks": [],
        "candidates": [],
        "residuals": [],
    }
    report_path = run_root / CALIBRATION_CANDIDATES
    report_path.write_text(json.dumps(report) + "\n")
    return report_path, profiles_path


def test_build_calibration_validation_accepts_candidate_profile(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    write_candidate_fixture(run_root)

    report = build_calibration_validation(
        run_root,
        min_inliers=6,
        max_mean_translation_residual_mm=2.0,
        max_mean_rotation_residual_deg=1.0,
        max_outlier_ratio=0.25,
    )

    assert report["schema_version"] == "calibration_validation.v1"
    assert report["overall_status"] == "ok"
    assert report["profile_count"] == 1
    assert report["promotable_profile_count"] == 1
    assert report["profiles"][0]["promotable"] is True
    assert report["profiles"][0]["outlier_ratio"] == 0.125


def test_calibration_validation_blocks_low_inlier_profile(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    write_candidate_fixture(
        run_root,
        profile=candidate_profile(num_observations=8, num_inliers=3, outlier_count=5),
    )

    report = build_calibration_validation(
        run_root,
        min_inliers=6,
        max_outlier_ratio=0.25,
    )

    assert report["overall_status"] == "error"
    assert report["promotable_profile_count"] == 0
    assert any(
        check["name"].startswith("profile_inliers:")
        and check["status"] == "error"
        for check in report["checks"]
    )


def test_build_calibration_validation_accepts_solver_profile_collection(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    profile = candidate_profile(
        profile_id="realsense_123_static_aruco_solved",
        num_observations=8,
        num_inliers=8,
        residual_translation_mm=0.5,
        residual_rotation_deg=0.2,
        outlier_count=0,
    )
    profiles_path = run_root / CALIBRATION_PROFILES_SOLVED
    write_profile_collection([profile], profiles_path)
    solver_report = {
        "schema_version": "calibration_solver.v1",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "run_root": run_root.as_posix(),
        "overall_status": "ok",
        "sensor_count": 1,
        "profile_count": 1,
        "candidate_count": profile.quality.num_observations,
        "observation_count": profile.quality.num_observations,
        "inlier_count": profile.quality.num_inliers,
        "outlier_count": 0,
        "profiles": [profile_to_dict(profile)],
        "checks": [],
        "solutions": [],
        "residuals": [],
    }
    (run_root / CALIBRATION_SOLVER_REPORT).write_text(json.dumps(solver_report) + "\n")

    report = build_calibration_validation(
        run_root,
        candidates_path=CALIBRATION_SOLVER_REPORT,
        min_inliers=6,
        max_mean_translation_residual_mm=2.0,
        max_mean_rotation_residual_deg=1.0,
        max_outlier_ratio=0.25,
    )

    assert report["overall_status"] == "ok"
    assert report["candidate_report_path"].endswith(CALIBRATION_SOLVER_REPORT)
    assert report["profile_source"].endswith(CALIBRATION_PROFILES_SOLVED)
    assert report["promotable_profile_count"] == 1


def test_write_calibration_validation_promotes_profiles_when_requested(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    write_candidate_fixture(run_root)

    report_path, promoted_path, report = write_calibration_validation_with_manifest(
        run_root,
        min_inliers=6,
        max_mean_translation_residual_mm=2.0,
        max_mean_rotation_residual_deg=1.0,
        max_outlier_ratio=0.25,
        promote=True,
        operator="test-operator",
    )

    assert report_path == run_root / CALIBRATION_VALIDATION_REPORT
    assert promoted_path == run_root / CALIBRATION_PROFILES
    assert report["overall_status"] == "ok"
    assert report["promotion"]["promoted"] is True
    profiles = load_profile_collection(promoted_path)
    assert len(profiles) == 1
    assert profiles[0].status == CalibrationStatus.VALID
    assert profiles[0].operator == "test-operator"
    assert profiles[0].metadata["validated_from_status"] == "needs_validation"
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage for stage in manifest["stages"] if stage["name"] == "calibration_validation"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][CALIBRATION_VALIDATION_REPORT] == (
        CALIBRATION_VALIDATION_REPORT
    )
    assert stage["artifacts"][CALIBRATION_PROFILES] == CALIBRATION_PROFILES


def test_calibration_promotion_preserves_unrelated_valid_profiles(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    write_candidate_fixture(run_root)
    unrelated = replace(
        candidate_profile(profile_id="oak_existing_valid"),
        sensor_id="oak-1",
        sensor_type=SensorType.OAK_D_PRO,
        rig_position="cell_left",
        status=CalibrationStatus.VALID,
    )
    write_profile_collection([unrelated], run_root / CALIBRATION_PROFILES)

    _report_path, promoted_path, report = write_calibration_validation_with_manifest(
        run_root,
        min_inliers=6,
        max_mean_translation_residual_mm=2.0,
        max_mean_rotation_residual_deg=1.0,
        max_outlier_ratio=0.25,
        promote=True,
    )

    profiles = load_profile_collection(promoted_path)
    assert {profile.profile_id for profile in profiles} == {
        "oak_existing_valid",
        "realsense_123_static_aruco_candidate",
    }
    assert report["promotion"]["preserved_profile_ids"] == ["oak_existing_valid"]


def test_calibration_validation_cli_writes_report(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_candidate_fixture(run_root)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(repo_root / "scripts" / "run_calibration_validation.py"),
            str(run_root),
            "--min-inliers",
            "6",
            "--max-mean-translation-residual-mm",
            "2",
            "--max-mean-rotation-residual-deg",
            "1",
            "--max-outlier-ratio",
            "0.25",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Calibration validation: ok (1/1 profiles promotable)" in result.stdout
    assert (run_root / CALIBRATION_VALIDATION_REPORT).is_file()
