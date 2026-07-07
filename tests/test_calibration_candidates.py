from __future__ import annotations

import json
import subprocess
from pathlib import Path

from posetestbot.calibration.candidates import (
    build_calibration_candidates,
    write_calibration_candidates_with_manifest,
)
from posetestbot.calibration.profiles import load_profile_collection
from posetestbot.io.artifacts import (
    CALIBRATION_CANDIDATES,
    CALIBRATION_OBSERVATIONS,
    CALIBRATION_PROFILES_FROM_OBSERVATIONS,
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
    observation_count: int = 2,
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
                "frame_key": f"{index:06d}.png",
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


def test_build_calibration_candidates_averages_static_identity(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    write_observations_fixture(run_root)

    report = build_calibration_candidates(
        run_root,
        min_observations=2,
        target_to_reference=IDENTITY_TARGET_TO_REFERENCE,
    )

    assert report["schema_version"] == "calibration_candidates.v1"
    assert report["overall_status"] == "ok"
    assert report["profile_count"] == 1
    assert report["candidate_count"] == 2
    assert report["inlier_count"] == 2
    assert report["outlier_count"] == 0
    profile = report["profiles"][0]
    assert profile["status"] == "needs_validation"
    assert profile["mounting_mode"] == "static"
    assert profile["extrinsics"]["from"] == "camera"
    assert profile["extrinsics"]["to"] == "robot_base"
    assert profile["extrinsics"]["rotation_quaternion_wxyz"] == [
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    assert profile["extrinsics"]["translation_mm"] == [0.0, 0.0, 0.0]
    assert profile["quality"]["num_observations"] == 2
    assert profile["quality"]["num_inliers"] == 2
    assert profile["quality"]["residual_translation_mm"] == 0.0
    assert profile["quality"]["residual_rotation_deg"] == 0.0


def test_build_calibration_candidates_rejects_residual_outlier(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    observations_path = write_observations_fixture(run_root, observation_count=3)
    report_data = json.loads(observations_path.read_text())
    report_data["observations"][2]["target_to_camera"]["translation"] = [
        1000.0,
        0.0,
        0.0,
    ]
    observations_path.write_text(json.dumps(report_data) + "\n")

    report = build_calibration_candidates(
        run_root,
        min_observations=2,
        target_to_reference=IDENTITY_TARGET_TO_REFERENCE,
        max_translation_residual_mm=50.0,
        max_rotation_residual_deg=15.0,
    )

    assert report["overall_status"] == "warning"
    assert report["profile_count"] == 1
    assert report["candidate_count"] == 3
    assert report["inlier_count"] == 2
    assert report["outlier_count"] == 1
    profile = report["profiles"][0]
    assert profile["quality"]["num_observations"] == 3
    assert profile["quality"]["num_inliers"] == 2
    assert profile["extrinsics"]["translation_mm"] == [0.0, 0.0, 0.0]
    candidates = {candidate["observation_id"]: candidate for candidate in report["candidates"]}
    assert candidates["realsense_123:000000.png"]["inlier"] is True
    assert candidates["realsense_123:000001.png"]["inlier"] is True
    assert candidates["realsense_123:000002.png"]["inlier"] is False
    assert candidates["realsense_123:000002.png"]["residual_translation_mm"] > 50.0
    assert report["residuals"][0]["inlier_count"] == 2
    assert report["residuals"][0]["outlier_count"] == 1
    assert any(
        check["name"] == "candidate_outliers:realsense_123"
        and check["status"] == "warning"
        for check in report["checks"]
    )


def test_build_calibration_candidates_warns_below_recommended_count(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    write_observations_fixture(run_root, observation_count=1)

    report = build_calibration_candidates(
        run_root,
        min_observations=2,
        target_to_reference=IDENTITY_TARGET_TO_REFERENCE,
    )

    assert report["overall_status"] == "warning"
    assert report["profile_count"] == 1
    assert any(
        check["name"] == "candidate_observations:realsense_123"
        and check["status"] == "warning"
        for check in report["checks"]
    )


def test_write_calibration_candidates_updates_manifest_and_profiles(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    write_observations_fixture(run_root)

    report_path, profiles_path, report = write_calibration_candidates_with_manifest(
        run_root,
        min_observations=2,
        target_to_reference=IDENTITY_TARGET_TO_REFERENCE,
    )

    assert report_path == run_root / CALIBRATION_CANDIDATES
    assert profiles_path == run_root / CALIBRATION_PROFILES_FROM_OBSERVATIONS
    assert report["overall_status"] == "ok"
    profiles = load_profile_collection(profiles_path)
    assert len(profiles) == 1
    assert profiles[0].status == "needs_validation"
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage for stage in manifest["stages"] if stage["name"] == "calibration_candidates"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][CALIBRATION_CANDIDATES] == CALIBRATION_CANDIDATES
    assert (
        stage["artifacts"][CALIBRATION_PROFILES_FROM_OBSERVATIONS]
        == CALIBRATION_PROFILES_FROM_OBSERVATIONS
    )


def test_calibration_candidates_cli_writes_report(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_observations_fixture(run_root)
    target_path = run_root / "target_to_reference.json"
    target_path.write_text(json.dumps(IDENTITY_TARGET_TO_REFERENCE) + "\n")
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(repo_root / "scripts" / "run_calibration_candidates.py"),
            str(run_root),
            "--min-observations",
            "2",
            "--target-to-reference",
            str(target_path),
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert (
        "Calibration candidates: ok (1 profiles, 2 frame candidates, 1 sensors)"
        in result.stdout
    )
    assert (run_root / CALIBRATION_CANDIDATES).is_file()
    assert (run_root / CALIBRATION_PROFILES_FROM_OBSERVATIONS).is_file()
