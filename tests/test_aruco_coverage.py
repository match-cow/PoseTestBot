from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from posetestbot.aruco.coverage import (
    build_aruco_coverage_report,
    write_aruco_coverage_report_with_manifest,
)
from posetestbot.io.artifact_browser import collect_run_artifacts
from posetestbot.io.artifacts import (
    ARUCO_COVERAGE_REPORT,
    ARUCO_POSE_ESTIMATION,
    DATASET_MANIFEST,
    PROCESSED_DIR,
    RGB_DIR,
    SYNCHRONIZED_DIR,
)
from posetestbot.io.manifest import load_run_manifest
from posetestbot.pipeline.recommendations import build_pipeline_recommendations
from posetestbot.pipeline.stages import build_pipeline_job


def create_aruco_output_fixture(tmp_path: Path) -> Path:
    run_root = tmp_path / "run"
    sensor_root = run_root / PROCESSED_DIR / SYNCHRONIZED_DIR / "realsense_123"
    (sensor_root / RGB_DIR).mkdir(parents=True)
    (sensor_root / ARUCO_POSE_ESTIMATION).write_text(
        json.dumps(
            {
                "000000.png": {
                    "motion": "motion_a",
                    "aruco_pose_estimation": {
                        "len_ids": 4,
                        "rvec": [0.1, 0.2, 0.3],
                        "tvec": [1.0, 2.0, 3.0],
                    },
                },
                "000001.png": {
                    "motion": "motion_a",
                    "aruco_pose_estimation": {
                        "len_ids": 2,
                        "rvec": [0.1, 0.2, 0.3],
                        "tvec": [1.0, 2.0, 3.0],
                    },
                },
                "000002.png": {
                    "motion": "motion_b",
                    "aruco_pose_estimation": {
                        "len_ids": 4,
                        "rvec": [],
                        "tvec": [1.0, 2.0, 3.0],
                    },
                },
                "000003.png": {"motion": "motion_b"},
            }
        )
    )
    return run_root


def test_build_aruco_coverage_report_counts_detection_and_pose_frames(
    tmp_path: Path,
) -> None:
    run_root = create_aruco_output_fixture(tmp_path)

    report = build_aruco_coverage_report(run_root, min_marker_count=4)

    assert report["schema_version"] == "aruco_coverage_report.v1"
    assert report["overall_status"] == "ok"
    assert report["sensor_count"] == 1
    assert report["frame_count"] == 4
    assert report["detected_frame_count"] == 3
    assert report["pose_frame_count"] == 2
    assert report["valid_pose_count"] == 1
    assert report["detection_ratio"] == 0.75
    assert report["valid_pose_ratio"] == 0.25
    sensor = report["sensors"][0]
    assert sensor["sensor_name"] == "realsense_123"
    assert sensor["insufficient_marker_count"] == 1
    assert sensor["invalid_pose_count"] == 1
    assert sensor["missing_count"] == 1
    assert sensor["motions"] == ["motion_a", "motion_b"]


def test_aruco_coverage_writes_manifest_and_artifact_summary(tmp_path: Path) -> None:
    run_root = create_aruco_output_fixture(tmp_path)

    report = write_aruco_coverage_report_with_manifest(run_root)

    assert report["valid_pose_count"] == 1
    assert (run_root / ARUCO_COVERAGE_REPORT).is_file()
    manifest = load_run_manifest(run_root)
    stage = next(stage for stage in manifest["stages"] if stage["name"] == "aruco_coverage")
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][ARUCO_COVERAGE_REPORT] == ARUCO_COVERAGE_REPORT
    assert (run_root / DATASET_MANIFEST).is_file()

    records = collect_run_artifacts(run_root)
    coverage = next(
        record
        for record in records
        if record.key == ARUCO_COVERAGE_REPORT and record.source == "known"
    )
    assert coverage.summary["type"] == "aruco_coverage_report"
    assert coverage.summary["aruco_coverage_ready_for_downstream"] is True
    assert coverage.summary["aruco_coverage_blocker"] is None
    assert coverage.summary["sensor_names"] == ["realsense_123"]
    assert coverage.summary["valid_pose_count"] == 1
    assert "aruco_coverage=ready" in coverage.to_dict()["display_label"]


def test_aruco_coverage_stage_cli_prints_json(tmp_path: Path) -> None:
    run_root = create_aruco_output_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_aruco_coverage_stage.py"),
            run_root.as_posix(),
            "--json",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["frame_count"] == 4
    assert payload["valid_pose_count"] == 1


def test_aruco_coverage_pipeline_stage_and_recommendation(tmp_path: Path) -> None:
    run_root = create_aruco_output_fixture(tmp_path)

    job = build_pipeline_job(stage_id="aruco_coverage", run_root=run_root)

    assert job.command == [
        "uv",
        "run",
        "python",
        "scripts/run_aruco_coverage_stage.py",
        run_root.as_posix(),
        "--min-marker-count",
        "4",
        "--min-valid-pose-ratio",
        "0.0",
    ]
    assert job.resources == ["disk_io"]

    recommendations = build_pipeline_recommendations(run_root)
    recommendation = next(
        item
        for item in recommendations["recommendations"]
        if item["id"] == "check_aruco_coverage"
    )
    assert recommendation["stage_id"] == "aruco_coverage"
    assert recommendation["expected_artifacts"] == [ARUCO_COVERAGE_REPORT]
