from __future__ import annotations

import json
import subprocess
from pathlib import Path

from posetestbot.io.artifacts import (
    CAM_K,
    DATASET_MANIFEST,
    DEPTH_DIR,
    FRAME_METADATA_JSONL,
    MATCH_ROBOT_EE_POSES,
    RAW_ROBOT_EE_POSES,
    RGB_DIR,
    SYNC_REPORT,
    SYNTHETIC_RGBD_REPORT,
)
from posetestbot.pipeline.synthetic_rgbd import write_synthetic_rgbd_fixture
from posetestbot.sync.non_destructive import synchronize_run


def write_raw_poses(run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / RAW_ROBOT_EE_POSES).write_text(
        json.dumps(
            {
                "0": {
                    "framename": 1000,
                    "host_received_timestamp_ns": 1_000_000_000,
                    "motion": "circ_far",
                    "pose": {"X": 1, "Y": 2, "Z": 3, "A": 0, "B": 0, "C": 0},
                },
                "1": {
                    "framename": 1100,
                    "host_received_timestamp_ns": 1_100_000_000,
                    "motion": "circ_far",
                    "pose": {"X": 2, "Y": 3, "Z": 4, "A": 0, "B": 0, "C": 0},
                },
                "2": {
                    "framename": 1200,
                    "host_received_timestamp_ns": 1_200_000_000,
                    "motion": "end",
                    "pose": {"X": 3, "Y": 4, "Z": 5, "A": 0, "B": 0, "C": 0},
                },
            }
        )
    )


def test_synthetic_rgbd_fixture_writes_syncable_sensor_folder(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_raw_poses(run_root)

    report_path, report = write_synthetic_rgbd_fixture(
        run_root,
        sensor_folder_name="realsense_synthetic",
        sensor_id="fixture",
    )

    assert report_path == run_root / SYNTHETIC_RGBD_REPORT
    assert report["frame_count"] == 2
    sensor_folder = run_root / "realsense_synthetic"
    assert (sensor_folder / RGB_DIR / "1100.png").is_file()
    assert (sensor_folder / DEPTH_DIR / "1200.png").is_file()
    assert (sensor_folder / CAM_K).read_text().splitlines()[0].startswith("64.0 ")
    metadata_rows = [
        json.loads(line)
        for line in (sensor_folder / FRAME_METADATA_JSONL).read_text().splitlines()
    ]
    assert metadata_rows[0]["host_received_timestamp_ns"] == 1_100_000_000
    assert metadata_rows[0]["expected_sync_delta_ms"] == 100.0

    results = synchronize_run(run_root)

    assert len(results) == 1
    assert results[0].matched_frames == 2
    synced = run_root / "processed" / "synchronized" / "realsense_synthetic"
    matched = json.loads((synced / MATCH_ROBOT_EE_POSES).read_text())
    assert list(matched) == ["000000.png", "000001.png"]
    assert matched["000000.png"]["nearest_robot_delta_ns"] == 0
    sync_report = json.loads((synced / SYNC_REPORT).read_text())
    assert sync_report["sync_delta_ms"] == 100.0
    assert sync_report["matched_frames"] == 2

    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage
        for stage in manifest["stages"]
        if stage["name"] == "synthetic_rgbd_fixture"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][SYNTHETIC_RGBD_REPORT] == SYNTHETIC_RGBD_REPORT
    assert manifest["sensors"][0]["status"] == "synthetic"


def test_synthetic_rgbd_fixture_cli(tmp_path: Path) -> None:
    run_root = tmp_path / "run-cli"
    write_raw_poses(run_root)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/create_synthetic_rgbd_fixture.py",
            run_root.as_posix(),
            "--frame-count",
            "1",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Wrote 1 synthetic RGB-D frame" in result.stdout
    report = json.loads((run_root / SYNTHETIC_RGBD_REPORT).read_text())
    assert report["frame_count"] == 1


def test_synthetic_rgbd_fixture_handles_many_pose_frames(tmp_path: Path) -> None:
    run_root = tmp_path / "run-many"
    run_root.mkdir(parents=True)
    poses = {}
    for index in range(20):
        poses[str(index)] = {
            "framename": 1000 + index * 10,
            "host_received_timestamp_ns": 1_000_000_000 + index * 10_000_000,
            "motion": "circ_far",
            "pose": {"X": index, "Y": 0, "Z": 0, "A": 0, "B": 0, "C": 0},
        }
    (run_root / RAW_ROBOT_EE_POSES).write_text(json.dumps(poses))

    _, report = write_synthetic_rgbd_fixture(run_root, frame_count=20)

    assert report["frame_count"] == 20
    assert (run_root / "realsense_synthetic" / RGB_DIR / "1290.png").is_file()
