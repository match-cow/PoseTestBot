from __future__ import annotations

import json
import subprocess
from pathlib import Path

from posetestbot.io.artifacts import DATASET_MANIFEST, SYNC_QUALITY_REPORT, SYNC_REPORT
from posetestbot.sync.quality import (
    build_sync_quality_report,
    discover_sync_reports,
    write_sync_quality_report_with_manifest,
)


def write_sync_report(
    run_root: Path,
    *,
    sensor_name: str = "realsense_123",
    total_frames: int = 10,
    matched_frames: int = 8,
    dropped_frames: int = 2,
    timestamp_source: str = "host_received",
    robot_timestamp_source: str = "host_received",
    max_delta_ns: int = 10_000_000,
    schema_version: str = "sync_report.v2",
) -> Path:
    report_path = run_root / "processed" / "synchronized" / sensor_name / SYNC_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "schema_version": schema_version,
        "sensor_folder": str(run_root / sensor_name),
        "output_folder": str(report_path.parent),
        "timestamp_source": timestamp_source,
        "requested_timestamp_source": timestamp_source,
        "timestamp_source_counts": {timestamp_source: total_frames},
        "timestamp_fallback_count": 0,
        "timestamp_missing_count": 0,
        "sync_delta_ms": 0,
        "max_nearest_pose_delta_ms": 20.0,
        "nearest_pose_delta_rejection_count": 2,
        "total_frames": total_frames,
        "matched_frames": matched_frames,
        "dropped_frames": dropped_frames,
        "motion_intervals": [{"motion": "circ_far", "pose_count": matched_frames}],
        "mean_abs_nearest_pose_delta_ns": 5_000_000,
        "max_abs_nearest_pose_delta_ns": max_delta_ns,
    }
    if schema_version == "sync_report.v3":
        value.update(
            {
                "frame_timestamp_source": timestamp_source,
                "requested_frame_timestamp_source": timestamp_source,
                "robot_timestamp_source": robot_timestamp_source,
                "timestamp_pair": {
                    "frame_timestamp_source": timestamp_source,
                    "requested_frame_timestamp_source": timestamp_source,
                    "robot_timestamp_source": robot_timestamp_source,
                },
                "timestamp_pair_provenance_audited": True,
            }
        )
    report_path.write_text(json.dumps(value) + "\n")
    return report_path


def test_build_sync_quality_report_summarizes_sync_reports(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    report_path = write_sync_report(run_root)

    report = build_sync_quality_report(
        run_root,
        min_match_ratio=0.5,
        max_dropped_frames=3,
        max_nearest_pose_delta_ms=20.0,
        require_timestamp_source="host_received",
    )

    assert discover_sync_reports(run_root) == [report_path]
    assert report["schema_version"] == "sync_quality_report.v2"
    assert report["overall_status"] == "ok"
    assert report["sensor_count"] == 1
    assert report["matched_frames"] == 8
    assert report["total_frames"] == 10
    assert report["overall_match_ratio"] == 0.8
    assert report["sensors"][0]["sensor_name"] == "realsense_123"
    assert report["sensors"][0]["max_nearest_pose_delta_ms"] == 20.0
    assert report["sensors"][0]["nearest_pose_delta_rejection_count"] == 2
    assert {check["status"] for check in report["checks"]} == {"ok"}


def test_build_sync_quality_report_accepts_relative_run_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    run_root = Path("run")
    write_sync_report(run_root)

    report = build_sync_quality_report(run_root, min_match_ratio=0.5)

    assert report["overall_status"] == "ok"
    assert report["sensor_count"] == 1
    assert report["sensors"][0]["report_path"] == (
        "processed/synchronized/realsense_123/sync_report.json"
    )


def test_build_sync_quality_report_warns_on_quality_thresholds(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    write_sync_report(
        run_root,
        matched_frames=4,
        dropped_frames=6,
        timestamp_source="filename",
        max_delta_ns=90_000_000,
    )

    report = build_sync_quality_report(
        run_root,
        min_match_ratio=0.8,
        max_dropped_frames=2,
        max_nearest_pose_delta_ms=50.0,
        require_timestamp_source="host_received",
    )

    warnings = {
        check["name"] for check in report["checks"] if check["status"] == "warning"
    }
    assert report["overall_status"] == "error"
    assert warnings == {
        "sync_match_ratio:realsense_123",
        "sync_dropped_frames:realsense_123",
        "sync_nearest_pose_delta:realsense_123",
    }
    timestamp_check = next(
        check
        for check in report["checks"]
        if check["name"] == "sync_timestamp_source:realsense_123"
    )
    assert timestamp_check["status"] == "error"


def test_v1_sync_report_cannot_prove_required_timestamp_source(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    write_sync_report(run_root, schema_version="sync_report.v1")

    report = build_sync_quality_report(
        run_root,
        require_timestamp_source="host_received",
    )

    assert report["overall_status"] == "error"
    assert report["sensors"][0]["timestamp_provenance_audited"] is False


def test_v3_sync_report_audits_frame_and_robot_timestamp_pair(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    write_sync_report(
        run_root,
        schema_version="sync_report.v3",
        timestamp_source="sensor",
        robot_timestamp_source="host_wall",
    )

    report = build_sync_quality_report(
        run_root,
        require_timestamp_source="sensor",
        require_robot_timestamp_source="host_wall",
    )

    assert report["overall_status"] == "ok"
    assert report["sensors"][0]["timestamp_pair_provenance_audited"] is True
    assert report["sensors"][0]["timestamp_pair"] == {
        "frame_timestamp_source": "sensor",
        "requested_frame_timestamp_source": "sensor",
        "robot_timestamp_source": "host_wall",
    }
    check = next(
        item
        for item in report["checks"]
        if item["name"] == "sync_robot_timestamp_source:realsense_123"
    )
    assert check["status"] == "ok"


def test_v2_sync_report_cannot_prove_robot_timestamp_source(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    write_sync_report(run_root, schema_version="sync_report.v2")

    report = build_sync_quality_report(
        run_root,
        require_robot_timestamp_source="host_received",
    )

    assert report["overall_status"] == "error"
    assert report["sensors"][0]["timestamp_pair_provenance_audited"] is False


def test_build_sync_quality_report_errors_without_sync_reports(
    tmp_path: Path,
) -> None:
    report = build_sync_quality_report(tmp_path / "run")

    assert report["overall_status"] == "error"
    assert report["sensor_count"] == 0
    assert report["checks"][0]["name"] == "sync_reports_present"


def test_write_sync_quality_report_updates_manifest(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_sync_report(run_root)

    path, report = write_sync_quality_report_with_manifest(
        run_root,
        min_match_ratio=0.5,
    )

    assert path == run_root / SYNC_QUALITY_REPORT
    assert report["overall_status"] == "ok"
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage for stage in manifest["stages"] if stage["name"] == "sync_quality"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][SYNC_QUALITY_REPORT] == SYNC_QUALITY_REPORT


def test_sync_quality_cli_writes_manifest_report(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_sync_report(run_root)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(repo_root / "scripts" / "run_sync_quality.py"),
            str(run_root),
            "--min-match-ratio",
            "0.5",
            "--max-dropped-frames",
            "3",
            "--require-timestamp-source",
            "host_received",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Sync quality: ok (8/10 frames matched, 1 sensors)" in result.stdout
    assert (run_root / SYNC_QUALITY_REPORT).is_file()
