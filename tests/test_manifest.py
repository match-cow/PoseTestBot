from __future__ import annotations

from pathlib import Path

from posetestbot.config import robot_profile
from posetestbot.io.artifacts import (
    ACCURACY_ARUCO_HRC_HUB,
    ACCURACY_HRC_HUB,
    ALL_RESULTS_JSON,
    ARUCO_COVERAGE_REPORT,
    BOP_EVALUATION_PLAN,
    BOP_RESULT_EXPORT_MANIFEST,
    CAPTURE_EXECUTION_LOGS_DIR,
    CAPTURE_EXECUTION_PLAN,
    CAPTURE_EXECUTION_REPORT,
    CALIBRATION_CANDIDATES,
    CALIBRATION_PREFLIGHT_REPORT,
    CALIBRATION_OBSERVATIONS,
    CALIBRATION_PROFILES_FROM_OBSERVATIONS,
    CALIBRATION_VALIDATION_REPORT,
    DATASET_MANIFEST,
    DEPTH_DIR,
    FRAME_METADATA_JSONL,
    METRIC_REPORT_CSV,
    METRIC_REPORT_JSON,
    METRIC_REPORT_XLSX,
    PIPELINE_SEQUENCE_PLAN,
    RAW_ROBOT_EE_POSES,
    RGB_DIR,
    RUN_CONFIG,
    SYNC_QUALITY_REPORT,
)
from posetestbot.io.manifest import (
    SCHEMA_VERSION,
    create_run_manifest,
    discover_sensor_records,
    load_run_manifest,
    make_sensor_record,
    record_raw_robot_pose_artifact,
    sensor_type_from_folder_name,
    set_manifest_sensors,
    upsert_stage,
    write_run_manifest,
)
from posetestbot.sensors.contracts import SensorType


def test_artifact_names_are_stable() -> None:
    assert DATASET_MANIFEST == "dataset_manifest.json"
    assert RUN_CONFIG == "run_config.json"
    assert RAW_ROBOT_EE_POSES == "raw_robot_ee_poses.json"
    assert ARUCO_COVERAGE_REPORT == "aruco_coverage_report.json"
    assert FRAME_METADATA_JSONL == "frame_metadata.jsonl"
    assert BOP_EVALUATION_PLAN == "bop_evaluation_plan.json"
    assert BOP_RESULT_EXPORT_MANIFEST == "bop_result_export_manifest.json"
    assert CAPTURE_EXECUTION_PLAN == "capture_execution_plan.json"
    assert CAPTURE_EXECUTION_REPORT == "capture_execution_report.json"
    assert CAPTURE_EXECUTION_LOGS_DIR == "capture_execution_logs"
    assert CALIBRATION_OBSERVATIONS == "calibration_observations.json"
    assert CALIBRATION_CANDIDATES == "calibration_candidates.json"
    assert (
        CALIBRATION_PROFILES_FROM_OBSERVATIONS
        == "calibration_profiles_from_observations.json"
    )
    assert CALIBRATION_VALIDATION_REPORT == "calibration_validation_report.json"
    assert CALIBRATION_PREFLIGHT_REPORT == "calibration_preflight_report.json"
    assert SYNC_QUALITY_REPORT == "sync_quality_report.json"
    assert PIPELINE_SEQUENCE_PLAN == "pipeline_sequence_plan.json"
    assert ACCURACY_HRC_HUB == "accuracy_HRC-Hub.json"
    assert ACCURACY_ARUCO_HRC_HUB == "accuracy_ArUco_HRC-Hub.json"
    assert ALL_RESULTS_JSON == "all_results.json"
    assert METRIC_REPORT_JSON == "metric_report.json"
    assert METRIC_REPORT_CSV == "metric_methods.csv"
    assert METRIC_REPORT_XLSX == "metric_report.xlsx"


def test_sensor_type_from_legacy_folder_names() -> None:
    assert sensor_type_from_folder_name("realsense_123") == "realsense_d435"
    assert sensor_type_from_folder_name("luxonis_abc") == "oak_d_pro"
    assert sensor_type_from_folder_name("zed_2i_456") == "zed_2i"
    assert sensor_type_from_folder_name("notes") is None


def test_manifest_write_load_and_stage_updates(tmp_path: Path) -> None:
    run_root = tmp_path / "run-1"
    sensor_root = run_root / "realsense_123"
    sensor_root.mkdir(parents=True)

    sensor = make_sensor_record(
        sensor_type=SensorType.REALSENSE_D435,
        device_id="123",
        folder=sensor_root,
        run_root=run_root,
        display_name="realsense_123",
        status="recording",
    )
    manifest = create_run_manifest(
        run_root,
        run_name="run-1",
        robot_profile=robot_profile("fake"),
        capture_config={"fps": 6, "resolution": "720p"},
        sensors=[sensor],
    )
    upsert_stage(manifest, name="capture", status="running")
    write_run_manifest(manifest, run_root)

    loaded = load_run_manifest(run_root)

    assert loaded["schema_version"] == SCHEMA_VERSION
    assert loaded["run_id"] == "run-1"
    assert loaded["capture_config"]["fps"] == 6
    assert loaded["sensors"][0]["folder"] == "realsense_123"
    assert loaded["stages"][0]["name"] == "capture"


def test_discover_sensor_records_from_run_folder(tmp_path: Path) -> None:
    run_root = tmp_path / "run-1"
    sensor_root = run_root / "luxonis_abc"
    (sensor_root / RGB_DIR).mkdir(parents=True)
    (sensor_root / DEPTH_DIR).mkdir()
    (sensor_root / FRAME_METADATA_JSONL).write_text("{}\n")
    (run_root / "not_a_sensor").mkdir()

    records = discover_sensor_records(run_root)

    assert len(records) == 1
    assert records[0]["sensor_type"] == "oak_d_pro"
    assert records[0]["device_id"] == "abc"
    assert records[0]["metadata"]["has_frame_metadata"] is True


def test_record_raw_robot_pose_artifact(tmp_path: Path) -> None:
    run_root = tmp_path / "run-1"
    run_root.mkdir()
    (run_root / RAW_ROBOT_EE_POSES).write_text("{}\n")

    manifest = create_run_manifest(run_root)
    set_manifest_sensors(manifest, [])
    record_raw_robot_pose_artifact(manifest, run_root)
    write_run_manifest(manifest, run_root)

    loaded = load_run_manifest(run_root)

    assert loaded["artifacts"][RAW_ROBOT_EE_POSES] == RAW_ROBOT_EE_POSES
    assert loaded["stages"][0]["name"] == "robot_pose_capture"
    assert loaded["stages"][0]["status"] == "succeeded"
