from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from posetestbot.io.artifacts import (
    CAMERA_DATA_JSON,
    CAMERA_JSON,
    CAM_K,
    DATASET_MANIFEST,
    DEPTH_DIR,
    DEPTH_SCALE,
    FRAME_METADATA_JSONL,
    MATCH_ROBOT_EE_POSES,
    RAW_ROBOT_EE_POSES,
    RGB_DIR,
    SYNC_REPORT,
)
from posetestbot.pipeline.run_config import (
    SensorRunConfig,
    create_run_config,
    write_run_config,
)
from posetestbot.sync.non_destructive import (
    resolve_frame_timestamp,
    resolve_max_nearest_pose_delta_ms,
    resolve_sync_delta_ms,
    resolve_timestamp_pair,
    synchronize_run,
    synchronize_sensor_folder,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=2)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def create_sync_fixture(tmp_path: Path) -> tuple[Path, Path]:
    run_root = tmp_path / "run-1"
    sensor_folder = run_root / "realsense_123"
    rgb_folder = sensor_folder / RGB_DIR
    depth_folder = sensor_folder / DEPTH_DIR
    rgb_folder.mkdir(parents=True)
    depth_folder.mkdir()

    for frame_id in ["1000.png", "1050.png", "1500.png"]:
        (rgb_folder / frame_id).write_bytes(f"rgb:{frame_id}".encode())
        (depth_folder / frame_id).write_bytes(f"depth:{frame_id}".encode())

    (sensor_folder / CAM_K).write_text("1 0 2\n0 3 4\n0 0 1\n")
    (sensor_folder / DEPTH_SCALE).write_text("1.0\n")
    write_json(sensor_folder / CAMERA_JSON, {"cam_K": [1, 0, 2, 0, 3, 4, 0, 0, 1]})
    write_json(
        sensor_folder / CAMERA_DATA_JSON, {"K": [[1, 0, 2], [0, 3, 4], [0, 0, 1]]}
    )

    write_jsonl(
        sensor_folder / FRAME_METADATA_JSONL,
        [
            {
                "schema_version": "frame_metadata.v1",
                "sensor_type": "realsense_d435",
                "sensor_id": "123",
                "frame_index": 0,
                "frame_id": "1000.png",
                "rgb_path": "rgb/1000.png",
                "depth_path": "depth/1000.png",
                "sensor_timestamp_ns": 10,
                "host_received_timestamp_ns": 1_000_000_000,
                "host_wall_timestamp_ns": 10_000_000_000,
            },
            {
                "schema_version": "frame_metadata.v1",
                "sensor_type": "realsense_d435",
                "sensor_id": "123",
                "frame_index": 1,
                "frame_id": "1050.png",
                "rgb_path": "rgb/1050.png",
                "depth_path": "depth/1050.png",
                "sensor_timestamp_ns": 20,
                "host_received_timestamp_ns": 1_050_000_000,
                "host_wall_timestamp_ns": 10_050_000_000,
            },
            {
                "schema_version": "frame_metadata.v1",
                "sensor_type": "realsense_d435",
                "sensor_id": "123",
                "frame_index": 2,
                "frame_id": "1500.png",
                "rgb_path": "rgb/1500.png",
                "depth_path": "depth/1500.png",
                "sensor_timestamp_ns": 30,
                "host_received_timestamp_ns": 1_500_000_000,
                "host_wall_timestamp_ns": 10_500_000_000,
            },
        ],
    )

    write_json(
        run_root / RAW_ROBOT_EE_POSES,
        {
            "0": {
                "framename": 1000,
                "host_received_timestamp_ns": 1_000_000_000,
                "host_wall_timestamp_ns": 10_000_000_000,
                "motion": "circ_far",
                "pose": {"X": 1, "Y": 2, "Z": 3, "A": 4, "B": 5, "C": 6},
            },
            "1": {
                "framename": 1100,
                "host_received_timestamp_ns": 1_100_000_000,
                "host_wall_timestamp_ns": 10_100_000_000,
                "motion": "circ_far",
                "pose": {"X": 7, "Y": 8, "Z": 9, "A": 10, "B": 11, "C": 12},
            },
            "2": {
                "framename": 2000,
                "host_received_timestamp_ns": 2_000_000_000,
                "host_wall_timestamp_ns": 11_000_000_000,
                "motion": "zoom",
                "pose": {"X": 13, "Y": 14, "Z": 15, "A": 16, "B": 17, "C": 18},
            },
        },
    )
    return run_root, sensor_folder


def test_synchronize_sensor_folder_preserves_raw_frames(tmp_path: Path) -> None:
    run_root, sensor_folder = create_sync_fixture(tmp_path)

    result = synchronize_sensor_folder(
        sensor_folder,
        run_root=run_root,
        sync_delta=0,
        timestamp_source="host_received",
    )

    assert result.total_frames == 3
    assert result.matched_frames == 2
    assert result.dropped_frames == 1
    assert (sensor_folder / RGB_DIR / "1000.png").exists()
    assert (sensor_folder / DEPTH_DIR / "1050.png").exists()

    output_folder = Path(result.output_folder)
    assert (output_folder / RGB_DIR / "000000.png").read_bytes() == b"rgb:1000.png"
    assert (output_folder / DEPTH_DIR / "000001.png").read_bytes() == b"depth:1050.png"
    assert (output_folder / CAM_K).read_text() == "1 0 2\n0 3 4\n0 0 1\n"
    assert (output_folder / DEPTH_SCALE).read_text() == "1.0\n"
    assert (output_folder / CAMERA_JSON).exists()
    assert (output_folder / CAMERA_DATA_JSON).exists()
    assert (output_folder / FRAME_METADATA_JSONL).exists()
    derived_metadata = [
        json.loads(line)
        for line in (output_folder / FRAME_METADATA_JSONL).read_text().splitlines()
    ]
    assert [record["frame_id"] for record in derived_metadata] == [
        "000000.png",
        "000001.png",
    ]
    assert derived_metadata[0]["rgb_path"] == "rgb/000000.png"
    assert derived_metadata[0]["source_frame_id"] == "1000.png"
    assert derived_metadata[0]["sync_timestamp_source"] == "host_received"

    matched = json.loads((output_folder / MATCH_ROBOT_EE_POSES).read_text())
    assert list(matched) == ["000000.png", "000001.png"]
    assert matched["000000.png"]["motion"] == "circ_far"
    assert matched["000000.png"]["source_rgb"] == "rgb/1000.png"
    assert matched["000000.png"]["synchronized_rgb"].endswith(
        "processed/synchronized/realsense_123/rgb/000000.png"
    )
    assert abs(matched["000001.png"]["nearest_robot_delta_ns"]) == 50_000_000

    report = json.loads((output_folder / SYNC_REPORT).read_text())
    assert report["schema_version"] == "sync_report.v3"
    assert report["timestamp_pair"] == {
        "frame_timestamp_source": "host_received",
        "requested_frame_timestamp_source": "host_received",
        "robot_timestamp_source": "host_received",
    }
    assert report["timestamp_pair_provenance_audited"] is True
    assert report["matched_frames"] == 2
    assert report["dropped"][0]["frame_id"] == "1500.png"
    assert CAM_K in report["copied_metadata_artifacts"]
    assert FRAME_METADATA_JSONL not in report["copied_metadata_artifacts"]


def test_synchronize_sensor_folder_replaces_stale_derived_frames(
    tmp_path: Path,
) -> None:
    run_root, sensor_folder = create_sync_fixture(tmp_path)
    first = synchronize_sensor_folder(
        sensor_folder,
        run_root=run_root,
        sync_delta=0,
        timestamp_source="host_received",
    )
    output_folder = Path(first.output_folder)
    assert len(list((output_folder / RGB_DIR).glob("*.png"))) == 2

    metadata_records = [
        json.loads(line)
        for line in (sensor_folder / FRAME_METADATA_JSONL).read_text().splitlines()
    ]
    write_jsonl(sensor_folder / FRAME_METADATA_JSONL, [metadata_records[0]])

    second = synchronize_sensor_folder(
        sensor_folder,
        run_root=run_root,
        sync_delta=0,
        timestamp_source="host_received",
    )

    assert second.matched_frames == 1
    assert [path.name for path in (output_folder / RGB_DIR).glob("*.png")] == [
        "000000.png"
    ]
    assert [path.name for path in (output_folder / DEPTH_DIR).glob("*.png")] == [
        "000000.png"
    ]


def test_sync_strict_nearest_pose_delta_drops_outlier_before_derived_output(
    tmp_path: Path,
) -> None:
    run_root, sensor_folder = create_sync_fixture(tmp_path)

    result = synchronize_sensor_folder(
        sensor_folder,
        run_root=run_root,
        sync_delta=0,
        timestamp_source="host_received",
        max_nearest_pose_delta_ms=20.0,
    )

    assert result.total_frames == 3
    assert result.matched_frames == 1
    assert result.dropped_frames == 2
    assert (sensor_folder / RGB_DIR / "1050.png").is_file()
    output_folder = Path(result.output_folder)
    assert [path.name for path in (output_folder / RGB_DIR).glob("*.png")] == [
        "000000.png"
    ]
    matched = json.loads((output_folder / MATCH_ROBOT_EE_POSES).read_text())
    assert list(matched) == ["000000.png"]

    report = json.loads((output_folder / SYNC_REPORT).read_text())
    assert report["max_nearest_pose_delta_ms"] == 20.0
    assert report["nearest_pose_delta_rejection_count"] == 1
    assert report["mean_abs_nearest_pose_delta_ns"] == 0
    assert report["max_abs_nearest_pose_delta_ns"] == 0
    rejected = next(
        item
        for item in report["dropped"]
        if item["reason"] == "nearest robot pose delta exceeds threshold"
    )
    assert rejected == {
        "frame_id": "1050.png",
        "timestamp_ns": 1_050_000_000,
        "timestamp_source": "host_received",
        "robot_timestamp_source": "host_received",
        "delayed_timestamp_ns": 1_050_000_000,
        "motion": "circ_far",
        "matched_robot_pose_index": 0,
        "robot_timestamp_ns": 1_000_000_000,
        "nearest_robot_delta_ns": -50_000_000,
        "abs_nearest_robot_delta_ns": 50_000_000,
        "max_nearest_pose_delta_ms": 20.0,
        "max_nearest_pose_delta_ns": 20_000_000,
        "reason": "nearest robot pose delta exceeds threshold",
    }


def test_sync_reports_filename_timestamp_fallback(tmp_path: Path) -> None:
    run_root, sensor_folder = create_sync_fixture(tmp_path)
    records = [
        json.loads(line)
        for line in (sensor_folder / FRAME_METADATA_JSONL).read_text().splitlines()
    ]
    records[0].pop("host_received_timestamp_ns")
    write_jsonl(sensor_folder / FRAME_METADATA_JSONL, records)

    result = synchronize_sensor_folder(
        sensor_folder,
        run_root=run_root,
        sync_delta=0,
        timestamp_source="host_received",
    )
    report = json.loads(Path(result.report_path).read_text())

    assert report["timestamp_source"] == "mixed"
    assert report["timestamp_source_counts"] == {
        "filename": 1,
        "host_received": 2,
    }
    assert report["timestamp_fallback_count"] == 1


def test_sensor_exposure_timestamp_pairs_explicitly_with_robot_wall_clock(
    tmp_path: Path,
) -> None:
    run_root, sensor_folder = create_sync_fixture(tmp_path)
    records = [
        json.loads(line)
        for line in (sensor_folder / FRAME_METADATA_JSONL).read_text().splitlines()
    ]
    for record in records:
        record["sensor_timestamp_ns"] = record["host_wall_timestamp_ns"]
        record["color_timestamp_domain"] = "global_time"
    write_jsonl(sensor_folder / FRAME_METADATA_JSONL, records)

    result = synchronize_sensor_folder(
        sensor_folder,
        run_root=run_root,
        sync_delta=0,
        timestamp_source="sensor",
        robot_timestamp_source="host_wall",
    )

    assert result.matched_frames == 2
    output_folder = Path(result.output_folder)
    matched = json.loads((output_folder / MATCH_ROBOT_EE_POSES).read_text())
    assert matched["000000.png"]["image_timestamp_ns"] == 10_000_000_000
    assert matched["000000.png"]["robot_timestamp_ns"] == 10_000_000_000
    assert matched["000000.png"]["robot_timestamp_source"] == "host_wall"
    derived = [
        json.loads(line)
        for line in (output_folder / FRAME_METADATA_JSONL).read_text().splitlines()
    ]
    assert derived[0]["sync_timestamp_source"] == "sensor"
    assert derived[0]["sync_robot_timestamp_source"] == "host_wall"
    report = json.loads((output_folder / SYNC_REPORT).read_text())
    assert report["timestamp_pair"] == {
        "frame_timestamp_source": "sensor",
        "requested_frame_timestamp_source": "sensor",
        "robot_timestamp_source": "host_wall",
    }


def test_sensor_timestamp_requires_explicit_compatible_robot_clock() -> None:
    with pytest.raises(ValueError, match="requires an explicit"):
        resolve_timestamp_pair("sensor", None)
    with pytest.raises(ValueError, match="unsupported pair"):
        resolve_timestamp_pair("sensor", "host_received")


def test_sync_cli_updates_manifest(tmp_path: Path) -> None:
    run_root, sensor_folder = create_sync_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "sync_non_destructive.py"),
            str(sensor_folder),
            "--run-root",
            str(run_root),
            "--sync-delta",
            "0",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Matched 2/3 frames" in result.stdout

    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage for stage in manifest["stages"] if stage["name"] == "sync:realsense_123"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][MATCH_ROBOT_EE_POSES].endswith(
        "processed/synchronized/realsense_123/match_robot_ee_poses.json"
    )
    assert stage["artifacts"][SYNC_REPORT].endswith(
        "processed/synchronized/realsense_123/sync_report.json"
    )


def test_sync_run_cli_processes_all_discovered_sensors(tmp_path: Path) -> None:
    run_root, sensor_folder = create_sync_fixture(tmp_path)
    shutil.copytree(sensor_folder, run_root / "luxonis_abc")
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "sync_run_non_destructive.py"),
            str(run_root),
            "--sync-delta",
            "0",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Synchronized 2 sensor(s)" in result.stdout

    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stages = {stage["name"]: stage for stage in manifest["stages"]}

    assert stages["sync_run"]["status"] == "succeeded"
    assert stages["sync:realsense_123"]["status"] == "succeeded"
    assert stages["sync:luxonis_abc"]["status"] == "succeeded"
    assert (
        run_root / "processed" / "synchronized" / "luxonis_abc" / MATCH_ROBOT_EE_POSES
    ).exists()


def test_synchronize_run_defaults_to_enabled_run_config_sensors(
    tmp_path: Path,
) -> None:
    run_root, sensor_folder = create_sync_fixture(tmp_path)
    disabled_folder = run_root / "realsense_999"
    shutil.copytree(sensor_folder, disabled_folder)
    write_run_config(
        run_root,
        create_run_config(
            run_root=run_root,
            sensors=(
                SensorRunConfig("realsense_d435", "123", "Enabled"),
                SensorRunConfig("realsense_d435", "999", "Disabled", enabled=False),
            ),
        ),
    )

    default_results = synchronize_run(run_root, sync_delta=0)
    explicit_results = synchronize_run(
        run_root,
        sensor_folders=[disabled_folder],
        output_root=run_root / "processed" / "explicit-disabled",
        sync_delta=0,
    )

    assert [Path(item.sensor_folder).name for item in default_results] == [
        "realsense_123"
    ]
    assert [Path(item.sensor_folder).name for item in explicit_results] == [
        "realsense_999"
    ]


def test_synchronize_run_accepts_only_an_explicit_subset_and_output_root(
    tmp_path: Path,
) -> None:
    run_root, sensor_folder = create_sync_fixture(tmp_path)
    shutil.copytree(sensor_folder, run_root / "luxonis_abc")
    output_root = run_root / "processed" / "calibration" / "attempt" / "sync"

    results = synchronize_run(
        run_root,
        sensor_folders=[sensor_folder.relative_to(run_root)],
        output_root=output_root,
        sync_delta=0,
    )

    assert [Path(item.sensor_folder).name for item in results] == ["realsense_123"]
    assert (output_root / "realsense_123" / MATCH_ROBOT_EE_POSES).is_file()
    assert not (output_root / "luxonis_abc").exists()
    with pytest.raises(ValueError, match="remain below the run root"):
        synchronize_run(run_root, sensor_folders=[tmp_path / "outside"])


def test_invalid_filename_timestamp_is_reported_as_missing() -> None:
    assert resolve_frame_timestamp({"frame_id": "not-numeric.png"}, "filename") == (
        None,
        None,
        False,
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "invalid"])
def test_sync_delta_rejects_nonfinite_or_nonnumeric_values(value: object) -> None:
    with pytest.raises(ValueError, match="Synchronization delta"):
        resolve_sync_delta_ms("realsense_123", value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, True, "invalid"])
def test_max_nearest_pose_delta_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="Maximum nearest-pose delta"):
        resolve_max_nearest_pose_delta_ms(value)  # type: ignore[arg-type]
