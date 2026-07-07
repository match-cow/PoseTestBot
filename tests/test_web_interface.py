from __future__ import annotations

import json
import sys
import importlib.util
from pathlib import Path

import cv2
import numpy as np

from posetestbot.jobs.runner import JobRecord, LocalJobRunner, SUCCEEDED
from posetestbot.calibration.profiles import (
    SCHEMA_VERSION as CALIBRATION_PROFILE_SCHEMA,
    CalibrationProfile,
    CalibrationQuality,
    CalibrationStatus,
    CalibrationTargetType,
    RigidTransform,
    TransformFrame,
    profile_to_dict,
)
from posetestbot.io.artifacts import (
    ACCURACY_HRC_HUB,
    BOP_DIR,
    BOP_EVALUATION_REPORT,
    BOP_EXPORT_MANIFEST,
    BOP_FRAME_MAP_JSON,
    BOP_RESULT_EXPORT_MANIFEST,
    CAPTURE_EXECUTION_PLAN,
    CAPTURE_EXECUTION_STATUS,
    CAPTURE_PLAN,
    CAPTURE_PLAN_PREFLIGHT_REPORT,
    DATASET_MANIFEST,
    HARDWARE_STATUS_REPORT,
    CALIBRATION_CANDIDATES,
    CALIBRATION_OBSERVATIONS,
    CALIBRATION_PROFILES_FROM_OBSERVATIONS,
    CALIBRATION_PROFILES,
    CALIBRATION_PROFILES_SOLVED,
    CALIBRATION_SOLVER_REPORT,
    CALIBRATION_VALIDATION_REPORT,
    PIPELINE_SEQUENCE_PLAN,
    DEPTH_DIR,
    REWRITE_STATUS_REPORT,
    MODELS_DIR,
    RESULTS_DIR,
    RGB_DIR,
    RUN_CONFIG,
    RUN_PREFLIGHT_REPORT,
    SYNC_QUALITY_REPORT,
    SYNC_REPORT,
)
from posetestbot.io.manifest import create_run_manifest, write_run_manifest
from posetestbot.pipeline.run_config import create_run_config, write_run_config
from posetestbot.pipeline.sequences import PipelineSequenceSpec, PipelineSequenceStepSpec
from posetestbot.pipeline.stages import PipelineParameter, PipelineStageSpec
from posetestbot.sensors.contracts import CameraIntrinsics, MountingMode, SensorType


def load_web_interface_module():
    module_path = Path(__file__).resolve().parents[1] / "web_interface.py"
    spec = importlib.util.spec_from_file_location("web_interface", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_command_submits_background_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    web_interface = load_web_interface_module()
    runner = LocalJobRunner(tmp_path / "jobs")
    monkeypatch.setattr(web_interface, "job_runner", runner)
    monkeypatch.setitem(
        web_interface.COMMANDS,
        "test_echo",
        {
            "label": "Test Echo",
            "command": [sys.executable, "-c", "print('web job ok')"],
            "resources": ["test_resource"],
        },
    )

    client = web_interface.app.test_client()
    response = client.post("/run-command", json={"command": "test_echo"})

    assert response.status_code == 202
    payload = response.get_json()
    assert payload["status"] in {"queued", "running", "succeeded"}
    job_id = payload["job_id"]

    finished = runner.wait(job_id, timeout=5)
    assert finished.status == SUCCEEDED

    status_response = client.get(f"/jobs/{job_id}")
    status_payload = status_response.get_json()
    assert status_payload["job"]["status"] == SUCCEEDED
    assert status_payload["job"]["resources"] == ["test_resource"]
    assert status_payload["job"]["parameters"]["label"] == "Test Echo"
    assert "web job ok" in status_payload["job"]["tail"]

    log_response = client.get(f"/jobs/{job_id}/log")
    assert "web job ok" in log_response.get_data(as_text=True)


def test_run_command_rejects_unknown_command() -> None:
    web_interface = load_web_interface_module()
    client = web_interface.app.test_client()

    response = client.post("/run-command", json={"command": "missing"})

    assert response.status_code == 404
    assert response.get_json()["output"] == "Unknown command"


def test_run_command_rejects_busy_resources(tmp_path: Path, monkeypatch) -> None:
    web_interface = load_web_interface_module()
    runner = LocalJobRunner(tmp_path / "jobs")
    monkeypatch.setattr(web_interface, "job_runner", runner)
    monkeypatch.setitem(
        web_interface.COMMANDS,
        "long_camera",
        {
            "label": "Long Camera",
            "command": [sys.executable, "-c", "import time; time.sleep(10)"],
            "resources": ["camera"],
        },
    )
    monkeypatch.setitem(
        web_interface.COMMANDS,
        "other_camera",
        {
            "label": "Other Camera",
            "command": [sys.executable, "-c", "print('blocked')"],
            "resources": ["camera"],
        },
    )
    client = web_interface.app.test_client()

    first = client.post("/run-command", json={"command": "long_camera"})
    second = client.post("/run-command", json={"command": "other_camera"})

    assert first.status_code == 202
    assert second.status_code == 409
    assert "camera held by job" in second.get_json()["output"]
    runner.cancel(first.get_json()["job_id"])


def test_capture_jobs_endpoint_filters_and_stops_capture_jobs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    web_interface = load_web_interface_module()
    runner = LocalJobRunner(tmp_path / "jobs")
    monkeypatch.setattr(web_interface, "job_runner", runner)
    run_root = tmp_path / "run"
    other_run_root = tmp_path / "other-run"
    run_root.mkdir()
    other_run_root.mkdir()
    client = web_interface.app.test_client()

    capture_job = runner.submit(
        name="pipeline:capture_execution",
        command=[sys.executable, "-c", "import time; time.sleep(10)"],
        resources=["robot_command", "camera", "disk_io"],
        parameters={
            "pipeline_stage": "capture_execution",
            "stage_label": "Capture Execution",
            "run_root": run_root.as_posix(),
            "options": {"mode": "pose_only_fake"},
        },
    )
    other_capture_job = runner.submit(
        name="pipeline:capture_execution",
        command=[sys.executable, "-c", "print('other run')"],
        resources=[],
        parameters={
            "pipeline_stage": "capture_execution",
            "run_root": other_run_root.as_posix(),
            "options": {"mode": "pose_only_fake"},
        },
    )

    response = client.get(f"/capture/jobs?run_root={run_root.as_posix()}")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["active_count"] == 1
    assert [job["id"] for job in payload["jobs"]] == [capture_job.id]
    assert payload["status_artifact"] is None
    assert payload["jobs"][0]["stage"] == "capture_execution"
    assert payload["jobs"][0]["mode"] == "pose_only_fake"
    assert payload["jobs"][0]["stop_endpoint"] == (
        f"/capture/jobs/{capture_job.id}/stop"
    )

    stop_response = client.post(f"/capture/jobs/{capture_job.id}/stop")

    assert stop_response.status_code == 200
    stop_payload = stop_response.get_json()
    assert stop_payload["capture_job"]["id"] == capture_job.id
    assert stop_payload["capture_job"]["active"] is False
    assert runner.get(capture_job.id).status == "canceled"

    runner.wait(other_capture_job.id, timeout=5)


def test_capture_status_endpoint_returns_latest_status_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    web_interface = load_web_interface_module()
    runner = LocalJobRunner(tmp_path / "jobs")
    monkeypatch.setattr(web_interface, "job_runner", runner)
    run_root = tmp_path / "run"
    run_root.mkdir()
    status = {
        "schema_version": "capture_execution_status.v1",
        "run_root": run_root.as_posix(),
        "status": "running",
        "mode": "pose_only_fake",
        "active_process_count": 1,
        "process_count": 2,
        "raw_pose_count": 3,
        "processes": [],
    }
    (run_root / CAPTURE_EXECUTION_STATUS).write_text(json.dumps(status))
    client = web_interface.app.test_client()

    response = client.get(f"/capture/status?run_root={run_root.as_posix()}")
    jobs_response = client.get(f"/capture/jobs?run_root={run_root.as_posix()}")

    assert response.status_code == 200
    assert response.get_json()["status"]["status"] == "running"
    assert response.get_json()["status"]["active_process_count"] == 1
    assert jobs_response.status_code == 200
    assert jobs_response.get_json()["status_artifact"]["raw_pose_count"] == 3


def test_capture_status_endpoint_reports_missing_artifact(tmp_path: Path) -> None:
    web_interface = load_web_interface_module()
    client = web_interface.app.test_client()

    response = client.get(f"/capture/status?run_root={(tmp_path / 'run').as_posix()}")

    assert response.status_code == 404
    assert CAPTURE_EXECUTION_STATUS in response.get_json()["output"]


def test_capture_job_stop_rejects_non_capture_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    web_interface = load_web_interface_module()
    runner = LocalJobRunner(tmp_path / "jobs")
    monkeypatch.setattr(web_interface, "job_runner", runner)
    client = web_interface.app.test_client()
    job = runner.submit(
        name="plain",
        command=[sys.executable, "-c", "print('plain job')"],
        resources=[],
        parameters={"run_root": (tmp_path / "run").as_posix()},
    )

    response = client.post(f"/capture/jobs/{job.id}/stop")

    assert response.status_code == 400
    assert response.get_json()["output"] == "Job is not a capture job"
    runner.wait(job.id, timeout=5)


def test_index_contains_run_config_controls() -> None:
    web_interface = load_web_interface_module()
    client = web_interface.app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'id="runRoot"' in html
    assert 'id="objectFolder"' in html
    assert 'id="robotStatusPanel"' in html
    assert 'id="sensorStatusPanel"' in html
    assert 'id="runtimeStatusPanel"' in html
    assert "saveRunConfig()" in html
    assert "queueRunConfig()" in html
    assert "runConfigQueuePayload()" in html
    assert "preflightRunConfig()" in html
    assert "renderRunConfigPreflight" in html
    assert html.count("renderRunConfigPreflight(result.data.preflight)") >= 3
    assert 'id="allowMissingPreflight"' in html
    assert 'id="allowFailedPreflight"' in html
    assert 'id="allowStalePreflight"' in html
    assert "allow_missing_preflight" in html
    assert "allow_failed_preflight" in html
    assert "allow_stale_preflight" in html
    assert "writeRunPreflight()" in html
    assert "preflightCalibration()" in html
    assert "buildCalibrationObservations()" in html
    assert "buildCalibrationCandidates()" in html
    assert "validateCalibrationCandidates()" in html
    assert "checkSyncQuality()" in html
    assert "writeCapturePlan()" in html
    assert "loadCapturePlan()" in html
    assert "preflightCapturePlan()" in html
    assert "writeCaptureExecutionPlan()" in html
    assert "loadCaptureExecutionPlan()" in html
    assert "queueCaptureExecution()" in html
    assert "captureExecutionOptions()" in html
    assert 'id="captureExecutionMode"' in html
    assert 'id="allowCaptureCameras"' in html
    assert 'id="includeCaptureSensors"' in html
    assert 'id="allowCaptureRealRobot"' in html
    assert "queueCaptureRehearsal()" in html
    assert "refreshCaptureJobs()" in html
    assert "stopCaptureJob(" in html
    assert "/capture/jobs" in html
    assert "/capture/status" in html
    assert "/capture-plan" in html
    assert "/capture-plan/execution" in html
    assert "/pipeline/preflight" in html
    assert "/calibration/preflight" in html
    assert "/calibration/observations" in html
    assert "/calibration/candidates" in html
    assert "/calibration/solver" in html
    assert "/calibration/validation" in html
    assert "/sync/quality" in html
    assert 'id="preflightPanel"' in html
    assert 'id="capturePlanPanel"' in html
    assert "/pipeline/run-config" in html
    assert "/robot/status" in html
    assert "/sensors/status" in html
    assert "/sensors/adapters" in html
    assert "/runtime/status" in html
    assert "/hardware/status" in html
    assert "writeHardwareStatus()" in html
    assert "loadHardwareStatus()" in html
    assert "loadRobotStatus()" in html
    assert "loadSensorStatus()" in html
    assert "loadSensorAdapters()" in html
    assert "loadRuntimeStatus()" in html
    assert "fake_capture_rehearsal" in html
    assert "fake_capture_execution" in html
    assert "sync_aruco_calibration_observations" in html
    assert "sync_aruco_calibration_candidates" in html
    assert "sync_aruco_calibration_solver" in html
    assert "sync_aruco_calibration_validation" in html
    assert "sync_to_bop_calibrated_dry_run" in html
    assert "capture_to_bop_foundationpose_dry_run" in html
    assert "foundationpose_runtime_to_bop_eval" in html
    assert "megapose_to_bop_eval_dry_run" in html
    assert "megapose_runtime_to_bop_eval" in html
    assert "sam6d_to_bop_eval_dry_run" in html
    assert "sam6d_runtime_to_bop_eval" in html
    assert 'id="jobsPanel"' in html
    assert 'id="captureJobsPanel"' in html
    assert 'id="artifactsPanel"' in html
    assert 'id="recommendationsPanel"' in html
    assert 'id="artifactPath"' in html
    assert 'id="metricsPanel"' in html
    assert 'id="bopInspectorPanel"' in html
    assert 'id="bopScenePath"' in html
    assert 'id="bopImageId"' in html
    assert 'id="bopResultPath"' in html
    assert "refreshJobs()" in html
    assert "loadRecommendations()" in html
    assert "recommendation.blocks_on" in html
    assert "Blocks: " in html
    assert "solveCalibration()" in html
    assert "renderCalibrationSolver(" in html
    assert "/pipeline/recommendations" in html
    assert "listArtifacts()" in html
    assert "previewArtifact()" in html
    assert "appendArtifactNextActions(" in html
    assert "next_action_blocks_on" in html
    assert "renderMetrics(" in html
    assert "loadBopFrame()" in html
    assert "loadBopResult()" in html
    assert "/artifacts/bop-frame-overlay" in html
    assert "bopFrameOverlayUrl()" in html
    assert "appendImageGallery(" in html
    assert "renderGtRows(" in html
    assert "Ground truth annotations" in html


def test_index_sequence_dropdown_uses_sequence_registry(monkeypatch) -> None:
    web_interface = load_web_interface_module()
    monkeypatch.setitem(
        web_interface.PIPELINE_SEQUENCES,
        "zzz_registry_sequence",
        PipelineSequenceSpec(
            id="zzz_registry_sequence",
            label="Registry Sequence",
            description="Sequence injected by the test registry.",
            steps=(PipelineSequenceStepSpec(id="sync", stage_id="sync_run"),),
        ),
    )
    client = web_interface.app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert (
        '<option value="sync_to_bop_dry_run" selected>sync_to_bop_dry_run</option>'
        in html
    )
    assert (
        '<option value="zzz_registry_sequence">zzz_registry_sequence</option>'
        in html
    )


def test_sensor_status_endpoint_reports_current_lab_profile(monkeypatch) -> None:
    web_interface = load_web_interface_module()

    def fake_collect_sensor_status(*, expected_counts=None):
        assert expected_counts is None
        return {
            "schema_version": "sensor_status.v1",
            "generated_at": "2026-06-16T00:00:00+00:00",
            "families": [
                {
                    "sensor_type": "realsense_d435",
                    "display_name": "Intel RealSense D435",
                    "sdk_module": "pyrealsense2",
                    "sdk_available": True,
                    "expected_count": 3,
                    "connected_count": 3,
                    "meets_expected": True,
                    "devices": [
                        {
                            "sensor_type": "realsense_d435",
                            "device_id": "825412070181",
                            "display_name": "RealSense 825412070181",
                            "connected": True,
                            "metadata": {},
                        }
                    ],
                    "error": None,
                }
            ],
            "total_connected": 3,
            "all_expected_connected": True,
        }

    monkeypatch.setattr(
        web_interface,
        "collect_sensor_status",
        fake_collect_sensor_status,
    )
    client = web_interface.app.test_client()

    response = client.get("/sensors/status")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["schema_version"] == "sensor_status.v1"
    assert payload["total_connected"] == 3
    assert payload["families"][0]["sensor_type"] == "realsense_d435"


def test_sensor_adapters_endpoint_lists_capture_capabilities() -> None:
    web_interface = load_web_interface_module()
    client = web_interface.app.test_client()

    response = client.get("/sensors/adapters")

    assert response.status_code == 200
    adapters = {
        adapter["sensor_type"]: adapter
        for adapter in response.get_json()["adapters"]
    }
    assert adapters["realsense_d435"]["capture_script"] == (
        "scripts/capture_realsense_720p.py"
    )
    assert adapters["oak_d_pro"]["sdk_module"] == "depthai"
    assert adapters["zed_2i"]["supported_resolutions"] == ["720p", "360p"]


def test_robot_status_endpoint_reports_selected_profile(monkeypatch) -> None:
    web_interface = load_web_interface_module()

    def fake_collect_robot_status():
        return {
            "schema_version": "robot_status.v1",
            "generated_at": "2026-06-16T00:00:00+00:00",
            "selected_profile": {
                "mode": "fake",
                "robot_ip": "127.0.0.1",
                "command_port": 30300,
                "receiver_ip": "127.0.0.1",
                "receiver_port": 8080,
                "cartesian_velocity_m_s": 0.2,
            },
            "profiles": {},
            "fake_first": True,
            "real_robot": {},
            "env_overrides": {},
            "command_protocols": ["legacy", "robot_command.v1"],
            "default_command_protocol": "legacy",
            "notes": [],
        }

    monkeypatch.setattr(
        web_interface,
        "collect_robot_status",
        fake_collect_robot_status,
    )
    client = web_interface.app.test_client()

    response = client.get("/robot/status")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["schema_version"] == "robot_status.v1"
    assert payload["selected_profile"]["mode"] == "fake"
    assert payload["fake_first"] is True


def test_sensor_status_endpoint_validates_expected_overrides() -> None:
    web_interface = load_web_interface_module()
    client = web_interface.app.test_client()

    response = client.get(
        "/sensors/status",
        query_string={"expected": "unknown=1"},
    )

    assert response.status_code == 400
    assert "Unknown sensor type" in response.get_json()["output"]


def test_runtime_status_endpoint_reports_external_runtime_snapshot(monkeypatch) -> None:
    web_interface = load_web_interface_module()

    def fake_collect_runtime_status():
        return {
            "schema_version": "runtime_status.v1",
            "generated_at": "2026-06-16T00:00:00+00:00",
            "available_count": 1,
            "runtime_count": 2,
            "all_available": False,
            "runtimes": [
                {
                    "runtime_id": "blenderproc",
                    "display_name": "BlenderProc",
                    "category": "renderer",
                    "required_for": "BlenderProc ground-truth rendering",
                    "available": True,
                    "checks": [
                        {
                            "name": "executable:blenderproc",
                            "ok": True,
                            "value": "/usr/bin/blenderproc",
                            "hint": None,
                        }
                    ],
                    "hint": None,
                }
            ],
        }

    monkeypatch.setattr(
        web_interface,
        "collect_runtime_status",
        fake_collect_runtime_status,
    )
    client = web_interface.app.test_client()

    response = client.get("/runtime/status")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["schema_version"] == "runtime_status.v1"
    assert payload["runtimes"][0]["runtime_id"] == "blenderproc"


def test_hardware_status_endpoint_writes_and_loads_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    web_interface = load_web_interface_module()
    run_root = tmp_path / "run-hardware-status"
    report = {
        "schema_version": "hardware_status_report.v1",
        "run_root": run_root.as_posix(),
        "overall_status": "warning",
        "checks": [{"name": "sensor_status", "status": "warning"}],
        "robot_status": {"selected_profile": {"mode": "fake"}},
        "sensor_status": {"total_connected": 3, "all_expected_connected": False},
        "runtime_status": {"available_count": 1, "runtime_count": 2},
        "include_sensor_status": True,
        "include_runtime_status": True,
    }

    def fake_write_hardware_status_report_with_manifest(
        run_root_arg,
        *,
        include_sensor_status=True,
        include_runtime_status=True,
    ):
        assert include_sensor_status is False
        assert include_runtime_status is True
        path = Path(run_root_arg) / HARDWARE_STATUS_REPORT
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report))
        return path, report

    monkeypatch.setattr(
        web_interface,
        "write_hardware_status_report_with_manifest",
        fake_write_hardware_status_report_with_manifest,
    )
    client = web_interface.app.test_client()

    post_response = client.post(
        "/hardware/status",
        json={"run_root": run_root.as_posix(), "no_sensors": "true"},
    )
    get_response = client.get(
        "/hardware/status",
        query_string={"run_root": run_root.as_posix()},
    )

    assert post_response.status_code == 201
    assert post_response.get_json()["report"]["overall_status"] == "warning"
    assert get_response.status_code == 200
    assert get_response.get_json()["report"]["sensor_status"]["total_connected"] == 3


class RecordingRunner:
    def __init__(self) -> None:
        self.submission = None

    def submit(self, **kwargs):
        self.submission = kwargs
        return JobRecord(
            id="pipeline123",
            name=kwargs["name"],
            command=kwargs["command"],
            cwd=Path(kwargs["cwd"]).as_posix() if kwargs.get("cwd") else None,
            status="queued",
            created_at="2026-06-16T00:00:00+00:00",
            log_path="/tmp/pipeline123/log.txt",
            resources=kwargs.get("resources", []),
            parameters=kwargs.get("parameters", {}),
        )


def test_pipeline_stages_endpoint_lists_stage_specs() -> None:
    web_interface = load_web_interface_module()
    client = web_interface.app.test_client()

    response = client.get("/pipeline/stages")

    assert response.status_code == 200
    stages = response.get_json()["stages"]
    assert any(stage["id"] == "sync_run" for stage in stages)
    assert any(stage["id"] == "capture_plan" for stage in stages)
    assert any(stage["id"] == "capture_plan_preflight" for stage in stages)
    assert any(stage["id"] == "capture_execution_plan" for stage in stages)
    assert any(stage["id"] == "capture_execution" for stage in stages)
    assert any(stage["id"] == "capture_rehearsal" for stage in stages)
    assert any(stage["id"] == "realsense_capture_smoke" for stage in stages)
    assert any(stage["id"] == "sync_quality" for stage in stages)
    assert any(stage["id"] == "calibration_observations" for stage in stages)
    assert any(stage["id"] == "calibration_candidates" for stage in stages)
    assert any(stage["id"] == "calibration_validation" for stage in stages)
    assert any(stage["id"] == "bop_evaluation" for stage in stages)


def test_pipeline_run_queues_typed_stage_job(tmp_path: Path, monkeypatch) -> None:
    web_interface = load_web_interface_module()
    runner = RecordingRunner()
    monkeypatch.setattr(web_interface, "job_runner", runner)
    monkeypatch.setitem(
        web_interface.PIPELINE_STAGES,
        "test_pipeline_stage",
        PipelineStageSpec(
            id="test_pipeline_stage",
            label="Test Pipeline Stage",
            script="scripts/test_pipeline_stage.py",
            description="Test-only pipeline stage.",
            resources=("camera",),
            parameters=(
                PipelineParameter(
                    name="dry_run",
                    flag="--dry-run",
                    kind="bool",
                    default=True,
                ),
            ),
        ),
    )
    run_root = tmp_path / "run"
    client = web_interface.app.test_client()

    response = client.post(
        "/pipeline/run",
        json={
            "stage": "test_pipeline_stage",
            "run_root": run_root.as_posix(),
            "options": {},
        },
    )

    assert response.status_code == 202
    payload = response.get_json()
    assert payload["job_id"] == "pipeline123"
    assert payload["pipeline"]["command"] == [
        "uv",
        "run",
        "python",
        "scripts/test_pipeline_stage.py",
        run_root.as_posix(),
        "--dry-run",
    ]
    assert runner.submission["name"] == "pipeline:test_pipeline_stage"
    assert runner.submission["resources"] == ["camera"]
    assert runner.submission["parameters"]["options"] == {"dry_run": True}


def test_pipeline_run_rejects_invalid_options(tmp_path: Path) -> None:
    web_interface = load_web_interface_module()
    client = web_interface.app.test_client()

    response = client.post(
        "/pipeline/run",
        json={
            "stage": "sync_run",
            "run_root": (tmp_path / "run").as_posix(),
            "options": {"timestamp_source": "not-a-clock"},
        },
    )

    assert response.status_code == 400
    assert "timestamp_source" in response.get_json()["output"]


def test_pipeline_sequences_endpoint_lists_sequence_specs() -> None:
    web_interface = load_web_interface_module()
    client = web_interface.app.test_client()

    response = client.get("/pipeline/sequences")

    assert response.status_code == 200
    sequences = response.get_json()["sequences"]
    assert any(sequence["id"] == "sync_aruco" for sequence in sequences)
    assert any(
        sequence["id"] == "sync_aruco_calibration_observations"
        for sequence in sequences
    )
    assert any(
        sequence["id"] == "sync_aruco_calibration_candidates"
        for sequence in sequences
    )
    assert any(
        sequence["id"] == "sync_aruco_calibration_validation"
        for sequence in sequences
    )
    assert any(sequence["id"] == "sync_to_bop_dry_run" for sequence in sequences)
    assert any(
        sequence["id"] == "sync_to_bop_calibrated_dry_run"
        for sequence in sequences
    )
    assert any(
        sequence["id"] == "foundationpose_to_bop_eval_dry_run"
        for sequence in sequences
    )
    assert any(
        sequence["id"] == "foundationpose_runtime_to_bop_eval"
        for sequence in sequences
    )
    assert any(sequence["id"] == "aruco_to_bop_eval_dry_run" for sequence in sequences)
    assert any(
        sequence["id"] == "megapose_to_bop_eval_dry_run"
        for sequence in sequences
    )
    assert any(
        sequence["id"] == "megapose_runtime_to_bop_eval"
        for sequence in sequences
    )
    assert any(sequence["id"] == "sam6d_to_bop_eval_dry_run" for sequence in sequences)
    assert any(sequence["id"] == "sam6d_runtime_to_bop_eval" for sequence in sequences)


def test_pipeline_run_sequence_queues_sequence_job(tmp_path: Path, monkeypatch) -> None:
    web_interface = load_web_interface_module()
    runner = RecordingRunner()
    monkeypatch.setattr(web_interface, "job_runner", runner)
    monkeypatch.setitem(
        web_interface.PIPELINE_STAGES,
        "test_sync",
        PipelineStageSpec(
            id="test_sync",
            label="Test Sync",
            script="scripts/test_sync.py",
            description="Test sync.",
            resources=("disk_io",),
        ),
    )
    monkeypatch.setitem(
        web_interface.PIPELINE_STAGES,
        "test_render",
        PipelineStageSpec(
            id="test_render",
            label="Test Render",
            script="scripts/test_render.py",
            description="Test render.",
            resources=("render",),
            parameters=(
                PipelineParameter(
                    name="dry_run",
                    flag="--dry-run",
                    kind="bool",
                    default=True,
                ),
            ),
        ),
    )
    monkeypatch.setitem(
        web_interface.PIPELINE_SEQUENCES,
        "test_sequence",
        PipelineSequenceSpec(
            id="test_sequence",
            label="Test Sequence",
            description="Test-only sequence.",
            steps=(
                PipelineSequenceStepSpec(id="sync", stage_id="test_sync"),
                PipelineSequenceStepSpec(
                    id="render",
                    stage_id="test_render",
                    depends_on=("sync",),
                ),
            ),
        ),
    )
    run_root = tmp_path / "run"
    client = web_interface.app.test_client()

    response = client.post(
        "/pipeline/run-sequence",
        json={
            "sequence": "test_sequence",
            "run_root": run_root.as_posix(),
            "options": {},
            "plan_only": True,
        },
    )

    assert response.status_code == 202
    payload = response.get_json()
    assert payload["job_id"] == "pipeline123"
    assert payload["sequence"]["command"][:4] == [
        "uv",
        "run",
        "python",
        "scripts/run_pipeline_sequence.py",
    ]
    assert payload["sequence"]["plan"]["steps"][1]["depends_on"] == ["sync"]
    assert runner.submission["name"] == "pipeline-sequence:test_sequence"
    assert runner.submission["resources"] == ["disk_io"]
    assert payload["sequence"]["plan"]["resources"] == ["disk_io", "render"]
    assert runner.submission["parameters"]["plan_only"] is True
    assert runner.submission["parameters"]["locked_resources"] == ["disk_io"]
    assert runner.submission["parameters"]["planned_resources"] == [
        "disk_io",
        "render",
    ]


def test_run_config_endpoint_loads_config_and_sequence_plan(tmp_path: Path) -> None:
    web_interface = load_web_interface_module()
    run_root = tmp_path / "run"
    config = create_run_config(
        run_root=run_root,
        sequence_id="sync_aruco",
        sequence_options={"aruco": {"save_images": True}},
    )
    write_run_config(run_root, config)
    client = web_interface.app.test_client()

    response = client.get("/run-config", query_string={"run_root": run_root.as_posix()})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["config"]["schema_version"] == "run_config.v1"
    assert payload["config"]["pipeline"]["sequence_id"] == "sync_aruco"
    assert payload["sequence_plan"]["sequence_id"] == "sync_aruco"
    assert payload["sequence_plan"]["steps"][2]["options"] == {
        "save_images": True,
        "show": False,
    }
    assert payload["preflight"] == {
        "path": (run_root / RUN_PREFLIGHT_REPORT).as_posix(),
        "exists": False,
        "overall_status": None,
        "matches_config": None,
        "ready_for_queue": False,
        "queue_blocker": "missing_preflight",
    }


def test_run_config_endpoint_reports_preflight_queue_readiness(
    tmp_path: Path,
) -> None:
    web_interface = load_web_interface_module()
    run_root = tmp_path / "run-ready-preflight"
    config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, config)
    (run_root / RUN_PREFLIGHT_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "run_preflight.v1",
                "overall_status": "warning",
                "config": config.to_dict(),
            }
        )
        + "\n"
    )
    client = web_interface.app.test_client()

    response = client.get("/run-config", query_string={"run_root": run_root.as_posix()})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["preflight"]["exists"] is True
    assert payload["preflight"]["overall_status"] == "warning"
    assert payload["preflight"]["matches_config"] is True
    assert payload["preflight"]["ready_for_queue"] is True
    assert payload["preflight"]["queue_blocker"] is None


def test_run_config_endpoint_creates_config_manifest_and_plan(
    tmp_path: Path,
) -> None:
    web_interface = load_web_interface_module()
    run_root = tmp_path / "run-created"
    client = web_interface.app.test_client()

    response = client.post(
        "/run-config",
        json={
            "run_root": run_root.as_posix(),
            "robot_mode": "fake",
            "sequence": "sync_aruco",
            "object_folder": "custom_object_models",
            "sensors": ["realsense:123:static:Cell RealSense"],
            "sequence_options": {"aruco": {"save_images": True}},
            "plan_only": True,
        },
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["config"]["capture"]["sensors"][0]["sensor_type"] == "realsense_d435"
    assert payload["config"]["capture"]["sensors"][0]["mounting_mode"] == "static"
    assert payload["config"]["object_folder"] == "custom_object_models"
    assert payload["sequence_plan"]["sequence_id"] == "sync_aruco"
    assert (run_root / RUN_CONFIG).is_file()
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(stage for stage in manifest["stages"] if stage["name"] == "run_config")
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][RUN_CONFIG] == RUN_CONFIG


def test_pipeline_recommendations_endpoint_reports_next_steps(
    tmp_path: Path,
) -> None:
    web_interface = load_web_interface_module()
    run_root = tmp_path / "run-recommendations"
    client = web_interface.app.test_client()

    response = client.get(
        "/pipeline/recommendations",
        query_string={"run_root": run_root.as_posix()},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["schema_version"] == "pipeline_recommendations.v1"
    assert payload["facts"]["has_run_config"] is False
    assert payload["recommendations"][0]["id"] == "create_run_config"
    assert payload["recommendations"][0]["command"][:3] == ["uv", "run", "python"]
    assert payload["recommendations"][0]["blocks_on"] == []


def test_capture_plan_endpoint_writes_and_loads_manifest_artifact(
    tmp_path: Path,
) -> None:
    web_interface = load_web_interface_module()
    run_root = tmp_path / "run-capture-plan"
    config = create_run_config(
        run_root=run_root,
        sensors=(
            web_interface.sensor_configs_from_values(["realsense:123:static:Cell RealSense"])[0],
        ),
        sequence_id="sync_aruco",
    )
    write_run_config(run_root, config)
    client = web_interface.app.test_client()

    response = client.post(
        "/capture-plan",
        json={"run_root": run_root.as_posix(), "max_frames": 1},
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["capture_plan"]["schema_version"] == "capture_plan.v1"
    assert payload["capture_plan"]["capture"]["max_frames"] == 1
    assert payload["capture_plan"]["commands"][1]["name"] == "realsense_123"
    assert (run_root / CAPTURE_PLAN).is_file()

    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(stage for stage in manifest["stages"] if stage["name"] == "capture_plan")
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][CAPTURE_PLAN] == CAPTURE_PLAN
    assert manifest["sensors"][0]["status"] == "planned"

    load_response = client.get(
        "/capture-plan",
        query_string={"run_root": run_root.as_posix()},
    )

    assert load_response.status_code == 200
    loaded = load_response.get_json()
    assert loaded["capture_plan"]["commands"][0]["name"] == "fake_iiwa_controller"


def test_capture_plan_endpoint_rejects_missing_plan(tmp_path: Path) -> None:
    web_interface = load_web_interface_module()
    client = web_interface.app.test_client()

    response = client.get(
        "/capture-plan",
        query_string={"run_root": (tmp_path / "missing").as_posix()},
    )

    assert response.status_code == 404
    assert CAPTURE_PLAN in response.get_json()["output"]


def test_capture_plan_preflight_endpoint_writes_manifest_report(
    tmp_path: Path,
) -> None:
    web_interface = load_web_interface_module()
    run_root = tmp_path / "run-capture-preflight"
    config = create_run_config(
        run_root=run_root,
        sensors=(
            web_interface.sensor_configs_from_values(["realsense:123:static:Cell RealSense"])[0],
        ),
        sequence_id="sync_aruco",
    )
    write_run_config(run_root, config)
    client = web_interface.app.test_client()

    response = client.post(
        "/capture-plan/preflight",
        json={"run_root": run_root.as_posix(), "no_sensors": True},
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["report"]["schema_version"] == "capture_plan_preflight.v1"
    assert payload["report"]["overall_status"] == "warning"
    assert (run_root / CAPTURE_PLAN_PREFLIGHT_REPORT).is_file()
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage for stage in manifest["stages"] if stage["name"] == "capture_plan_preflight"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][CAPTURE_PLAN_PREFLIGHT_REPORT] == (
        CAPTURE_PLAN_PREFLIGHT_REPORT
    )


def test_capture_execution_endpoint_writes_and_loads_manifest_artifact(
    tmp_path: Path,
) -> None:
    web_interface = load_web_interface_module()
    run_root = tmp_path / "run-capture-execution"
    config = create_run_config(
        run_root=run_root,
        sensors=(
            web_interface.sensor_configs_from_values(["realsense:123:static:Cell RealSense"])[0],
        ),
        sequence_id="sync_aruco",
    )
    write_run_config(run_root, config)
    client = web_interface.app.test_client()

    response = client.post(
        "/capture-plan/execution",
        json={"run_root": run_root.as_posix(), "mode": "pose_only_fake"},
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["plan"]["schema_version"] == "capture_execution_plan.v1"
    assert payload["plan"]["status"] == "ok"
    assert payload["plan"]["selected_roles"] == [
        "robot_controller",
        "robot_pose_receiver",
    ]
    assert (run_root / CAPTURE_EXECUTION_PLAN).is_file()
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage for stage in manifest["stages"] if stage["name"] == "capture_execution_plan"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][CAPTURE_EXECUTION_PLAN] == CAPTURE_EXECUTION_PLAN

    load_response = client.get(
        "/capture-plan/execution",
        query_string={"run_root": run_root.as_posix()},
    )

    assert load_response.status_code == 200
    loaded = load_response.get_json()
    assert loaded["plan"]["mode"] == "pose_only_fake"


def test_capture_execution_endpoint_keeps_string_false_safety_gates(
    tmp_path: Path,
) -> None:
    web_interface = load_web_interface_module()
    run_root = tmp_path / "run-capture-execution-full-blocked"
    config = create_run_config(
        run_root=run_root,
        sensors=(
            web_interface.sensor_configs_from_values(["realsense:123:static:Cell RealSense"])[0],
        ),
        sequence_id="sync_aruco",
    )
    write_run_config(run_root, config)
    client = web_interface.app.test_client()

    response = client.post(
        "/capture-plan/execution",
        json={
            "run_root": run_root.as_posix(),
            "mode": "full",
            "allow_cameras": "false",
            "include_sensors": "false",
        },
    )

    assert response.status_code == 409
    payload = response.get_json()
    assert payload["plan"]["mode"] == "full"
    assert payload["plan"]["allow_cameras"] is False
    assert payload["plan"]["include_sensor_status"] is False
    assert payload["plan"]["selected_commands"] == []
    gates = {gate["name"]: gate for gate in payload["plan"]["gates"]}
    assert gates["camera_permission"]["status"] == "error"


def test_calibration_preflight_endpoint_writes_manifest_report(
    tmp_path: Path,
) -> None:
    web_interface = load_web_interface_module()
    run_root = tmp_path / "run-calibration-preflight"
    config = create_run_config(
        run_root=run_root,
        sensors=(
            web_interface.sensor_configs_from_values(["realsense:123:static:Cell RealSense"])[0],
        ),
        sequence_id="sync_aruco",
    )
    write_run_config(run_root, config)
    client = web_interface.app.test_client()

    response = client.post(
        "/calibration/preflight",
        json={"run_root": run_root.as_posix()},
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["report"]["schema_version"] == "calibration_preflight.v1"
    assert payload["report"]["overall_status"] == "warning"
    assert (run_root / "calibration_preflight_report.json").is_file()


def test_calibration_observations_endpoint_writes_manifest_report(
    tmp_path: Path,
) -> None:
    web_interface = load_web_interface_module()
    run_root = tmp_path / "run-calibration-observations"
    aruco_path = (
        run_root
        / "processed"
        / "synchronized"
        / "realsense_123"
        / "aruco_pose_estimation.json"
    )
    aruco_path.parent.mkdir(parents=True)
    aruco_path.write_text(
        json.dumps(
            {
                "000000.png": {
                    "motion": "circ_far",
                    "robot_ee_pose": {"X": 1, "Y": 2, "Z": 3},
                    "aruco_pose_estimation": {
                        "rvec": [0.1, 0.2, 0.3],
                        "tvec": [10, 20, 30],
                        "len_ids": 4,
                    },
                }
            }
        )
    )
    client = web_interface.app.test_client()

    response = client.post(
        "/calibration/observations",
        json={
            "run_root": run_root.as_posix(),
            "min_marker_count": 4,
            "min_observations": 1,
            "target_type": "charuco",
            "grid_size": "5x7",
            "dictionary": "DICT_4X4_50",
            "marker_length_mm": 32,
            "square_length_mm": 40,
        },
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["report"]["schema_version"] == "calibration_observations.v1"
    assert payload["report"]["overall_status"] == "ok"
    assert payload["report"]["observation_count"] == 1
    assert payload["report"]["target"]["target_type"] == "charuco"
    assert payload["report"]["target"]["grid_size"] == [5, 7]
    assert (run_root / CALIBRATION_OBSERVATIONS).is_file()
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage for stage in manifest["stages"] if stage["name"] == "calibration_observations"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][CALIBRATION_OBSERVATIONS] == CALIBRATION_OBSERVATIONS


def test_calibration_candidates_endpoint_writes_manifest_report(
    tmp_path: Path,
) -> None:
    web_interface = load_web_interface_module()
    run_root = tmp_path / "run-calibration-candidates"
    (run_root / CALIBRATION_OBSERVATIONS).parent.mkdir(parents=True)
    (run_root / CALIBRATION_OBSERVATIONS).write_text(
        json.dumps(
            {
                "schema_version": "calibration_observations.v1",
                "overall_status": "ok",
                "sensor_count": 1,
                "frame_count": 1,
                "observation_count": 1,
                "rejected_count": 0,
                "motion_count": 1,
                "checks": [],
                "sensors": [
                    {
                        "sensor_name": "realsense_123",
                        "sensor_type": "realsense_d435",
                        "device_id": "123",
                        "mounting_mode": "static",
                    }
                ],
                "observations": [
                    {
                        "observation_id": "realsense_123:000000.png",
                        "sensor_name": "realsense_123",
                        "sensor_type": "realsense_d435",
                        "device_id": "123",
                        "mounting_mode": "static",
                        "target_to_camera": {
                            "rotation_vector_rodrigues": [0.0, 0.0, 0.0],
                            "translation": [0.0, 0.0, 0.0],
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
                ],
                "rejected": [],
            }
        )
    )
    client = web_interface.app.test_client()

    response = client.post(
        "/calibration/candidates",
        json={
            "run_root": run_root.as_posix(),
            "min_observations": 1,
            "max_translation_residual_mm": 50.0,
            "max_rotation_residual_deg": 15.0,
            "target_to_reference": {
                "from": "calibration_target",
                "to": "robot_base",
                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "translation_mm": [0.0, 0.0, 0.0],
            },
        },
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["report"]["schema_version"] == "calibration_candidates.v1"
    assert payload["report"]["overall_status"] == "ok"
    assert payload["report"]["profile_count"] == 1
    assert payload["report"]["inlier_count"] == 1
    assert payload["report"]["outlier_count"] == 0
    assert (run_root / CALIBRATION_CANDIDATES).is_file()
    assert (run_root / CALIBRATION_PROFILES_FROM_OBSERVATIONS).is_file()
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


def test_calibration_solver_endpoint_writes_manifest_report(
    tmp_path: Path,
) -> None:
    web_interface = load_web_interface_module()
    run_root = tmp_path / "run-calibration-solver"
    (run_root / CALIBRATION_OBSERVATIONS).parent.mkdir(parents=True)
    (run_root / CALIBRATION_OBSERVATIONS).write_text(
        json.dumps(
            {
                "schema_version": "calibration_observations.v1",
                "overall_status": "ok",
                "sensor_count": 1,
                "frame_count": 1,
                "observation_count": 1,
                "rejected_count": 0,
                "motion_count": 1,
                "checks": [],
                "sensors": [
                    {
                        "sensor_name": "realsense_123",
                        "sensor_type": "realsense_d435",
                        "device_id": "123",
                        "mounting_mode": "static",
                    }
                ],
                "observations": [
                    {
                        "observation_id": "realsense_123:000000.png",
                        "sensor_name": "realsense_123",
                        "sensor_type": "realsense_d435",
                        "device_id": "123",
                        "mounting_mode": "static",
                        "target_to_camera": {
                            "rotation_vector_rodrigues": [0.0, 0.0, 0.0],
                            "translation": [0.0, 0.0, 0.0],
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
                ],
                "rejected": [],
            }
        )
    )
    client = web_interface.app.test_client()

    response = client.post(
        "/calibration/solver",
        json={
            "run_root": run_root.as_posix(),
            "min_observations": 1,
            "no_residual_thresholds": True,
            "target_to_reference": {
                "from": "calibration_target",
                "to": "robot_base",
                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "translation_mm": [0.0, 0.0, 0.0],
            },
        },
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["report"]["schema_version"] == "calibration_solver.v1"
    assert payload["report"]["overall_status"] == "ok"
    assert payload["report"]["profile_count"] == 1
    assert payload["report"]["inlier_count"] == 1
    assert payload["report"]["outlier_count"] == 0
    assert (run_root / CALIBRATION_SOLVER_REPORT).is_file()
    assert (run_root / CALIBRATION_PROFILES_SOLVED).is_file()
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage for stage in manifest["stages"] if stage["name"] == "calibration_solver"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][CALIBRATION_SOLVER_REPORT] == CALIBRATION_SOLVER_REPORT
    assert stage["artifacts"][CALIBRATION_PROFILES_SOLVED] == CALIBRATION_PROFILES_SOLVED


def test_calibration_validation_endpoint_promotes_profiles(
    tmp_path: Path,
) -> None:
    web_interface = load_web_interface_module()
    run_root = tmp_path / "run-calibration-validation"
    profile = CalibrationProfile(
        schema_version=CALIBRATION_PROFILE_SCHEMA,
        profile_id="realsense_123_static_aruco_candidate",
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
            num_observations=8,
            num_inliers=7,
            residual_translation_mm=1.2,
            residual_rotation_deg=0.4,
        ),
        metadata={"sensor_name": "realsense_123", "outlier_count": 1},
    )
    run_root.mkdir(parents=True)
    (run_root / CALIBRATION_CANDIDATES).write_text(
        json.dumps(
            {
                "schema_version": "calibration_candidates.v1",
                "overall_status": "warning",
                "profile_count": 1,
                "candidate_count": 8,
                "inlier_count": 7,
                "outlier_count": 1,
                "profiles": [profile_to_dict(profile)],
                "checks": [],
                "candidates": [],
                "residuals": [],
            }
        )
        + "\n"
    )
    client = web_interface.app.test_client()

    response = client.post(
        "/calibration/validation",
        json={
            "run_root": run_root.as_posix(),
            "min_inliers": 6,
            "max_mean_translation_residual_mm": 2.0,
            "max_mean_rotation_residual_deg": 1.0,
            "max_outlier_ratio": 0.25,
            "promote": True,
            "operator": "web-test",
        },
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["report"]["schema_version"] == "calibration_validation.v1"
    assert payload["report"]["overall_status"] == "ok"
    assert payload["report"]["promotion"]["promoted"] is True
    assert (run_root / CALIBRATION_VALIDATION_REPORT).is_file()
    assert (run_root / CALIBRATION_PROFILES).is_file()
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage for stage in manifest["stages"] if stage["name"] == "calibration_validation"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][CALIBRATION_VALIDATION_REPORT] == (
        CALIBRATION_VALIDATION_REPORT
    )
    assert stage["artifacts"][CALIBRATION_PROFILES] == CALIBRATION_PROFILES


def test_sync_quality_endpoint_writes_manifest_report(tmp_path: Path) -> None:
    web_interface = load_web_interface_module()
    run_root = tmp_path / "run-sync-quality"
    sync_report = (
        run_root
        / "processed"
        / "synchronized"
        / "realsense_123"
        / SYNC_REPORT
    )
    sync_report.parent.mkdir(parents=True)
    sync_report.write_text(
        json.dumps(
            {
                "schema_version": "sync_report.v1",
                "timestamp_source": "host_received",
                "sync_delta_ms": 0,
                "total_frames": 3,
                "matched_frames": 2,
                "dropped_frames": 1,
                "motion_windows": {"circ_far": {"count": 2}},
                "max_abs_nearest_pose_delta_ns": 10_000_000,
            }
        )
    )
    client = web_interface.app.test_client()

    response = client.post(
        "/sync/quality",
        json={
            "run_root": run_root.as_posix(),
            "min_match_ratio": 0.5,
            "max_dropped_frames": 1,
            "require_timestamp_source": "host_received",
        },
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["report"]["schema_version"] == "sync_quality_report.v1"
    assert payload["report"]["overall_status"] == "ok"
    assert payload["report"]["matched_frames"] == 2
    assert (run_root / SYNC_QUALITY_REPORT).is_file()
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(stage for stage in manifest["stages"] if stage["name"] == "sync_quality")
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][SYNC_QUALITY_REPORT] == SYNC_QUALITY_REPORT


def test_pipeline_preflight_endpoint_reports_run_readiness(
    tmp_path: Path,
    monkeypatch,
) -> None:
    web_interface = load_web_interface_module()
    run_root = tmp_path / "run"

    def fake_build_run_preflight(
        requested_run_root,
        *,
        include_sensor_status=True,
        include_runtime_status=True,
    ):
        assert Path(requested_run_root) == run_root
        assert include_sensor_status is True
        assert include_runtime_status is False
        return {
            "schema_version": "run_preflight.v1",
            "run_root": run_root.as_posix(),
            "overall_status": "ok",
            "checks": [],
            "sequence_plan": {"sequence_id": "sync_aruco", "steps": []},
        }

    monkeypatch.setattr(
        web_interface,
        "build_run_preflight",
        fake_build_run_preflight,
    )
    client = web_interface.app.test_client()

    response = client.get(
        "/pipeline/preflight",
        query_string={
            "run_root": run_root.as_posix(),
            "include_runtimes": "false",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["schema_version"] == "run_preflight.v1"
    assert payload["overall_status"] == "ok"


def test_pipeline_preflight_endpoint_writes_manifest_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    web_interface = load_web_interface_module()
    run_root = tmp_path / "run-write-preflight"
    report_path = run_root / RUN_PREFLIGHT_REPORT

    def fake_write_run_preflight_with_manifest(
        requested_run_root,
        *,
        include_sensor_status=True,
        include_runtime_status=True,
    ):
        assert Path(requested_run_root) == run_root
        assert include_sensor_status is False
        assert include_runtime_status is True
        report_path.parent.mkdir(parents=True)
        report_path.write_text("{}\n")
        return report_path, {
            "schema_version": "run_preflight.v1",
            "run_root": run_root.as_posix(),
            "overall_status": "warning",
            "checks": [{"name": "sensor_status", "status": "warning"}],
            "sequence_plan": {"sequence_id": "sync_aruco", "steps": []},
        }

    monkeypatch.setattr(
        web_interface,
        "write_run_preflight_with_manifest",
        fake_write_run_preflight_with_manifest,
    )
    client = web_interface.app.test_client()

    response = client.post(
        "/pipeline/preflight",
        json={
            "run_root": run_root.as_posix(),
            "include_sensors": False,
            "include_runtimes": True,
        },
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["path"] == report_path.as_posix()
    assert payload["report"]["schema_version"] == "run_preflight.v1"
    assert payload["report"]["overall_status"] == "warning"


def test_pipeline_preflight_endpoint_rejects_missing_run_root() -> None:
    web_interface = load_web_interface_module()
    client = web_interface.app.test_client()

    response = client.get("/pipeline/preflight")

    assert response.status_code == 400
    assert response.get_json()["output"] == "Missing run_root"


def test_pipeline_run_config_queues_configured_sequence_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    web_interface = load_web_interface_module()
    runner = RecordingRunner()
    monkeypatch.setattr(web_interface, "job_runner", runner)
    run_root = tmp_path / "run"
    config = create_run_config(
        run_root=run_root,
        sequence_id="sync_aruco",
        sequence_options={"aruco": {"save_images": True}},
    )
    write_run_config(run_root, config)
    (run_root / RUN_PREFLIGHT_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "run_preflight.v1",
                "overall_status": "ok",
                "config": config.to_dict(),
            }
        )
        + "\n"
    )
    client = web_interface.app.test_client()

    response = client.post(
        "/pipeline/run-config",
        json={"run_root": run_root.as_posix()},
    )

    assert response.status_code == 202
    payload = response.get_json()
    assert payload["job_id"] == "pipeline123"
    assert payload["preflight"]["ready_for_queue"] is True
    assert payload["preflight"]["queue_blocker"] is None
    assert payload["run_config"]["pipeline"]["sequence_id"] == "sync_aruco"
    assert payload["sequence"]["sequence_id"] == "sync_aruco"
    assert payload["sequence"]["command"][:4] == [
        "uv",
        "run",
        "python",
        "scripts/run_pipeline_sequence.py",
    ]
    assert "--plan-only" in payload["sequence"]["command"]
    assert runner.submission["name"] == "pipeline-run-config:sync_aruco"
    assert runner.submission["resources"] == ["disk_io"]
    assert payload["sequence"]["plan"]["resources"] == ["cpu", "disk_io"]
    assert runner.submission["parameters"]["locked_resources"] == ["disk_io"]
    assert runner.submission["parameters"]["planned_resources"] == [
        "cpu",
        "disk_io",
    ]
    assert runner.submission["parameters"]["run_config"] == RUN_CONFIG
    assert runner.submission["parameters"]["options"] == {
        "aruco": {"save_images": True}
    }
    assert runner.submission["parameters"]["steps"][2]["options"] == {
        "save_images": True,
        "show": False,
    }


def test_pipeline_run_config_rejects_missing_run_preflight(
    tmp_path: Path,
    monkeypatch,
) -> None:
    web_interface = load_web_interface_module()
    runner = RecordingRunner()
    monkeypatch.setattr(web_interface, "job_runner", runner)
    run_root = tmp_path / "run-missing-preflight"
    config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, config)
    client = web_interface.app.test_client()

    response = client.post(
        "/pipeline/run-config",
        json={"run_root": run_root.as_posix()},
    )

    assert response.status_code == 409
    payload = response.get_json()
    assert "is missing" in payload["output"]
    assert payload["preflight"]["queue_blocker"] == "missing_preflight"
    assert payload["preflight"]["ready_for_queue"] is False
    assert payload["preflight_path"] == (run_root / RUN_PREFLIGHT_REPORT).as_posix()
    assert runner.submission is None


def test_pipeline_run_config_allows_missing_run_preflight_with_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    web_interface = load_web_interface_module()
    runner = RecordingRunner()
    monkeypatch.setattr(web_interface, "job_runner", runner)
    run_root = tmp_path / "run-missing-preflight-override"
    config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, config)
    client = web_interface.app.test_client()

    response = client.post(
        "/pipeline/run-config",
        json={
            "run_root": run_root.as_posix(),
            "allow_missing_preflight": True,
        },
    )

    assert response.status_code == 202
    assert response.get_json()["sequence"]["sequence_id"] == "sync_aruco"
    assert runner.submission["name"] == "pipeline-run-config:sync_aruco"


def test_pipeline_run_config_rejects_invalid_run_preflight(
    tmp_path: Path,
    monkeypatch,
) -> None:
    web_interface = load_web_interface_module()
    runner = RecordingRunner()
    monkeypatch.setattr(web_interface, "job_runner", runner)
    run_root = tmp_path / "run-invalid-preflight"
    config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, config)
    (run_root / RUN_PREFLIGHT_REPORT).write_text("[]\n")
    client = web_interface.app.test_client()

    response = client.post(
        "/pipeline/run-config",
        json={"run_root": run_root.as_posix()},
    )

    assert response.status_code == 409
    payload = response.get_json()
    assert "is invalid" in payload["output"]
    assert payload["preflight"]["queue_blocker"] == "invalid_preflight"
    assert payload["preflight"]["ready_for_queue"] is False
    assert RUN_PREFLIGHT_REPORT in payload["preflight"]["error"]
    assert runner.submission is None


def test_pipeline_run_config_rejects_failed_run_preflight(
    tmp_path: Path,
    monkeypatch,
) -> None:
    web_interface = load_web_interface_module()
    runner = RecordingRunner()
    monkeypatch.setattr(web_interface, "job_runner", runner)
    run_root = tmp_path / "run-failed-preflight"
    config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, config)
    (run_root / RUN_PREFLIGHT_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "run_preflight.v1",
                "overall_status": "error",
                "checks": [
                    {
                        "name": "runtime_requirements",
                        "status": "error",
                        "message": "BOP Toolkit missing.",
                    }
                ],
            }
        )
        + "\n"
    )
    client = web_interface.app.test_client()

    response = client.post(
        "/pipeline/run-config",
        json={"run_root": run_root.as_posix()},
    )

    assert response.status_code == 409
    payload = response.get_json()
    assert RUN_PREFLIGHT_REPORT in payload["output"]
    assert payload["preflight"]["queue_blocker"] == "failed_preflight"
    assert payload["preflight_report"]["overall_status"] == "error"
    assert payload["preflight_path"] == (run_root / RUN_PREFLIGHT_REPORT).as_posix()
    assert runner.submission is None


def test_pipeline_run_config_allows_failed_run_preflight_with_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    web_interface = load_web_interface_module()
    runner = RecordingRunner()
    monkeypatch.setattr(web_interface, "job_runner", runner)
    run_root = tmp_path / "run-failed-preflight-override"
    config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, config)
    (run_root / RUN_PREFLIGHT_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "run_preflight.v1",
                "overall_status": "error",
                "config": config.to_dict(),
            }
        )
        + "\n"
    )
    client = web_interface.app.test_client()

    response = client.post(
        "/pipeline/run-config",
        json={
            "run_root": run_root.as_posix(),
            "allow_failed_preflight": True,
        },
    )

    assert response.status_code == 202
    assert response.get_json()["sequence"]["sequence_id"] == "sync_aruco"
    assert runner.submission["name"] == "pipeline-run-config:sync_aruco"


def test_pipeline_run_config_rejects_stale_run_preflight(
    tmp_path: Path,
    monkeypatch,
) -> None:
    web_interface = load_web_interface_module()
    runner = RecordingRunner()
    monkeypatch.setattr(web_interface, "job_runner", runner)
    run_root = tmp_path / "run-stale-preflight"
    original_config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, original_config)
    (run_root / RUN_PREFLIGHT_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "run_preflight.v1",
                "overall_status": "ok",
                "config": original_config.to_dict(),
            }
        )
        + "\n"
    )
    updated_config = create_run_config(
        run_root=run_root,
        sequence_id="sync_to_bop_dry_run",
    )
    write_run_config(run_root, updated_config)
    client = web_interface.app.test_client()

    response = client.post(
        "/pipeline/run-config",
        json={"run_root": run_root.as_posix()},
    )

    assert response.status_code == 409
    payload = response.get_json()
    assert "does not match" in payload["output"]
    assert payload["preflight"]["queue_blocker"] == "stale_preflight"
    assert payload["preflight_report"]["config"]["pipeline"]["sequence_id"] == (
        "sync_aruco"
    )
    assert runner.submission is None


def test_pipeline_run_config_allows_stale_run_preflight_with_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    web_interface = load_web_interface_module()
    runner = RecordingRunner()
    monkeypatch.setattr(web_interface, "job_runner", runner)
    run_root = tmp_path / "run-stale-preflight-override"
    original_config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, original_config)
    (run_root / RUN_PREFLIGHT_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "run_preflight.v1",
                "overall_status": "ok",
                "config": original_config.to_dict(),
            }
        )
        + "\n"
    )
    updated_config = create_run_config(
        run_root=run_root,
        sequence_id="sync_to_bop_dry_run",
    )
    write_run_config(run_root, updated_config)
    client = web_interface.app.test_client()

    response = client.post(
        "/pipeline/run-config",
        json={
            "run_root": run_root.as_posix(),
            "allow_stale_preflight": True,
        },
    )

    assert response.status_code == 202
    assert response.get_json()["sequence"]["sequence_id"] == "sync_to_bop_dry_run"
    assert runner.submission["name"] == "pipeline-run-config:sync_to_bop_dry_run"


def test_pipeline_run_config_rejects_missing_config(tmp_path: Path) -> None:
    web_interface = load_web_interface_module()
    run_root = tmp_path / "run"
    client = web_interface.app.test_client()

    response = client.post(
        "/pipeline/run-config",
        json={"run_root": run_root.as_posix()},
    )

    assert response.status_code == 404
    assert RUN_CONFIG in response.get_json()["output"]


def test_artifacts_endpoint_lists_run_artifacts_and_job_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    web_interface = load_web_interface_module()
    runner = LocalJobRunner(tmp_path / "jobs")
    monkeypatch.setattr(web_interface, "job_runner", runner)
    run_root = tmp_path / "run"
    run_root.mkdir()
    write_run_manifest(create_run_manifest(run_root), run_root)
    (run_root / PIPELINE_SEQUENCE_PLAN).write_text('{"sequence_id": "sync_aruco"}\n')
    (run_root / REWRITE_STATUS_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "rewrite_status_report.v1",
                "overall_status": "blocked",
                "summary": {
                    "gate_count": 4,
                    "ready_gate_count": 1,
                    "blocked_gate_count": 3,
                    "check_count": 26,
                    "ready_check_count": 12,
                    "blocked_check_count": 14,
                },
                "gates": [],
                "next_gate": {"gate_id": "rewrite_full_capture.v1"},
                "next_actions": [
                    {
                        "gate_id": "rewrite_full_capture.v1",
                        "label": "Inspect sensor status",
                        "command": [
                            "uv",
                            "run",
                            "python",
                            "scripts/sensor_status.py",
                            "--json",
                        ],
                        "blocks_on": [
                            "sensor:realsense_d435",
                            "sensor:oak_d_pro",
                            "sensor:zed_2i",
                        ],
                    },
                    {
                        "gate_id": "rewrite_full_capture.v1",
                        "label": "Refresh hardware status after sensor fix",
                        "command": [
                            "uv",
                            "run",
                            "python",
                            "scripts/run_hardware_status_stage.py",
                            run_root.as_posix(),
                        ],
                        "blocks_on": [
                            "sensor:realsense_d435",
                            "sensor:oak_d_pro",
                            "sensor:zed_2i",
                        ],
                    },
                ],
                "next_blockers": [],
            }
        )
        + "\n"
    )
    sensor_root = run_root / "processed" / "synchronized" / "realsense_123"
    sensor_root.mkdir(parents=True)
    (sensor_root / ACCURACY_HRC_HUB).write_text(
        '{"foundationpose": {"all_motions": {"AP_p": 1.0, "x": [1.0]}}}\n',
    )
    job = runner.submit(
        name="pipeline-sequence:sync_aruco",
        command=[sys.executable, "-c", "print('linked log')"],
        parameters={"run_root": run_root.as_posix()},
    )
    runner.wait(job.id, timeout=5)
    client = web_interface.app.test_client()

    response = client.get("/artifacts", query_string={"run_root": run_root.as_posix()})

    assert response.status_code == 200
    artifacts = response.get_json()["artifacts"]
    sequence_plan = next(
        artifact
        for artifact in artifacts
        if artifact["key"] == PIPELINE_SEQUENCE_PLAN and artifact["source"] == "known"
    )
    assert sequence_plan["summary"]["type"] == "pipeline_sequence_plan"
    assert "sequence=sync_aruco" in sequence_plan["display_label"]
    assert "steps=0" in sequence_plan["display_label"]
    assert "exists" in sequence_plan["display_label"]
    rewrite_status = next(
        artifact
        for artifact in artifacts
        if artifact["key"] == REWRITE_STATUS_REPORT
    )
    assert rewrite_status["summary"]["next_action_count"] == 2
    assert rewrite_status["summary"]["next_action_labels"] == [
        "Inspect sensor status",
        "Refresh hardware status after sensor fix",
    ]
    assert rewrite_status["summary"]["next_action_blocks_on"] == [
        ["sensor:realsense_d435", "sensor:oak_d_pro", "sensor:zed_2i"],
        ["sensor:realsense_d435", "sensor:oak_d_pro", "sensor:zed_2i"],
    ]
    metric = next(
        artifact
        for artifact in artifacts
        if artifact["key"] == ACCURACY_HRC_HUB
    )
    assert metric["summary"]["type"] == "pose_accuracy_metrics"
    assert metric["summary"]["best_by_AP_p"] == {
        "method": "foundationpose",
        "AP_p": 1.0,
    }
    job_log = next(artifact for artifact in artifacts if artifact["kind"] == "job_log")
    assert job_log["job_id"] == job.id
    assert job_log["log_endpoint"] == f"/jobs/{job.id}/log"


def test_artifact_metrics_endpoint_reports_dashboard_summary(tmp_path: Path) -> None:
    web_interface = load_web_interface_module()
    run_root = tmp_path / "run"
    sensor_root = run_root / "processed" / "synchronized" / "realsense_123"
    sensor_root.mkdir(parents=True)
    (sensor_root / ACCURACY_HRC_HUB).write_text(
        '{"foundationpose": {"all_motions": {"AP_p": 1.0, "x": [1.0]}}}\n',
    )
    (run_root / BOP_EVALUATION_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "bop_evaluation_report.v1",
                "status": "succeeded",
                "dry_run": False,
                "eval_path": (run_root / "evaluation" / "bop_toolkit").as_posix(),
                "result": {
                    "filename": "foundationpose_bop-test.csv",
                    "path": (
                        run_root / RESULTS_DIR / BOP_DIR / "foundationpose_bop-test.csv"
                    ).as_posix(),
                },
                "checks": [],
                "output_artifacts": [],
                "score_summary": {
                    "score_file_count": 1,
                    "metrics": {
                        "bop19_average_recall": 0.8,
                    },
                    "files": [],
                },
            }
        )
    )
    client = web_interface.app.test_client()

    response = client.get(
        "/artifacts/metrics",
        query_string={"run_root": run_root.as_posix()},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["type"] == "metric_dashboard"
    assert payload["metric_artifact_count"] == 1
    assert payload["direct_method_count"] == 1
    assert payload["best_by_AP_p"] == {
        "method": "foundationpose",
        "AP_p": 1.0,
        "relative_path": f"processed/synchronized/realsense_123/{ACCURACY_HRC_HUB}",
    }
    assert payload["bop_score_count"] == 1
    assert payload["best_bop19_average_recall"] == {
        "result_filename": "foundationpose_bop-test.csv",
        "bop19_average_recall": 0.8,
        "relative_path": BOP_EVALUATION_REPORT,
    }
    assert payload["bop_scores"][0]["metrics"]["bop19_average_recall"] == 0.8


def test_bop_result_endpoint_reports_pose_rows(tmp_path: Path) -> None:
    web_interface = load_web_interface_module()
    run_root = tmp_path / "run"
    scene = run_root / BOP_DIR / "realsense_123" / "test" / "000001"
    scene.mkdir(parents=True)
    (run_root / BOP_DIR / BOP_EXPORT_MANIFEST).write_text(
        json.dumps(
            {
                "exports": [
                    {
                        "sensor_name": "realsense_123",
                        "scene_id": 1,
                        "split": "test",
                        "scene_folder": scene.as_posix(),
                    }
                ]
            }
        )
    )
    result_file = run_root / RESULTS_DIR / BOP_DIR / "foundationpose_bop-test.csv"
    result_file.parent.mkdir(parents=True)
    result_file.write_text(
        "scene_id,im_id,obj_id,score,R,t,time\n"
        "1,0,1,0.9,1 0 0 0 1 0 0 0 1,10 20 30,0.01\n"
    )
    (run_root / BOP_RESULT_EXPORT_MANIFEST).write_text(
        json.dumps(
            {
                "results": [
                    {
                        "filename": "foundationpose_bop-test.csv",
                        "path": result_file.as_posix(),
                    }
                ]
            }
        )
    )
    client = web_interface.app.test_client()

    response = client.get(
        "/artifacts/bop-result",
        query_string={
            "run_root": run_root.as_posix(),
            "path": f"{RESULTS_DIR}/{BOP_DIR}/foundationpose_bop-test.csv",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["type"] == "bop_result_detail"
    assert payload["metadata"]["method"] == "foundationpose"
    assert payload["row_count"] == 1
    assert payload["rows"][0]["score"] == 0.9
    assert payload["rows"][0]["t"] == [10.0, 20.0, 30.0]
    assert payload["rows"][0]["scene"]["relative_scene_folder"] == (
        f"{BOP_DIR}/realsense_123/test/000001"
    )


def test_artifact_preview_endpoint_reads_json_and_rejects_escape(tmp_path: Path) -> None:
    web_interface = load_web_interface_module()
    run_root = tmp_path / "run"
    run_root.mkdir()
    write_run_manifest(create_run_manifest(run_root), run_root)
    client = web_interface.app.test_client()

    preview = client.get(
        "/artifacts/preview",
        query_string={
            "run_root": run_root.as_posix(),
            "path": DATASET_MANIFEST,
        },
    )
    escaped = client.get(
        "/artifacts/preview",
        query_string={
            "run_root": run_root.as_posix(),
            "path": "../outside.txt",
        },
    )

    assert preview.status_code == 200
    assert preview.get_json()["preview"]["type"] == "json"
    assert preview.get_json()["artifact"]["summary"]["type"] == "dataset_manifest"
    assert escaped.status_code == 400


def test_artifact_file_endpoint_serves_run_scoped_files(tmp_path: Path) -> None:
    web_interface = load_web_interface_module()
    run_root = tmp_path / "run"
    run_root.mkdir()
    inside = run_root / "frames" / "sample.txt"
    inside.parent.mkdir()
    inside.write_text("artifact bytes\n")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n")
    client = web_interface.app.test_client()

    served = client.get(
        "/artifacts/file",
        query_string={
            "run_root": run_root.as_posix(),
            "path": "frames/sample.txt",
        },
    )
    escaped = client.get(
        "/artifacts/file",
        query_string={
            "run_root": run_root.as_posix(),
            "path": outside.as_posix(),
        },
    )

    assert served.status_code == 200
    assert served.get_data(as_text=True) == "artifact bytes\n"
    assert served.content_type.startswith("text/plain")
    assert escaped.status_code == 400


def test_bop_scene_endpoint_reports_scene_frames(tmp_path: Path) -> None:
    web_interface = load_web_interface_module()
    run_root = tmp_path / "run"
    scene = run_root / BOP_DIR / "realsense_123" / "test" / "000001"
    scene.mkdir(parents=True)
    (scene / "scene_camera.json").write_text(
        '{"0": {"cam_K": [1, 0, 2, 0, 1, 1, 0, 0, 1], "depth_scale": 1.0}}\n'
    )
    (scene / "scene_gt.json").write_text('{"0": [{"obj_id": 1}]}\n')
    (scene / "scene_gt_info.json").write_text('{"0": [{}]}\n')
    client = web_interface.app.test_client()

    response = client.get(
        "/artifacts/bop-scene",
        query_string={
            "run_root": run_root.as_posix(),
            "path": f"{BOP_DIR}/realsense_123/test/000001",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["type"] == "bop_scene_detail"
    assert payload["summary"]["image_count"] == 1
    assert payload["frames"][0]["gt_count"] == 1


def test_bop_frame_endpoint_reports_scene_and_result_bundle(tmp_path: Path) -> None:
    web_interface = load_web_interface_module()
    run_root = tmp_path / "run"
    scene = run_root / BOP_DIR / "realsense_123" / "test" / "000001"
    scene.mkdir(parents=True)
    (scene / "scene_camera.json").write_text(
        '{"0": {"cam_K": [1, 0, 2, 0, 1, 1, 0, 0, 1], "depth_scale": 1.0}}\n'
    )
    (scene / "scene_gt.json").write_text('{"0": [{"obj_id": 1}]}\n')
    (scene / "scene_gt_info.json").write_text(
        '{"0": [{"bbox_obj": [0, 0, 4, 3], "visib_fract": 0.75}]}\n'
    )
    (scene / RGB_DIR).mkdir()
    (scene / DEPTH_DIR).mkdir()
    (scene / "mask").mkdir()
    (scene / "mask_visib").mkdir()
    cv2.imwrite(
        (scene / RGB_DIR / "000000.png").as_posix(),
        np.zeros((3, 4, 3), dtype=np.uint8),
    )
    cv2.imwrite(
        (scene / DEPTH_DIR / "000000.png").as_posix(),
        np.zeros((3, 4), dtype=np.uint16),
    )
    cv2.imwrite(
        (scene / "mask" / "000000_000000.png").as_posix(),
        np.ones((3, 4), dtype=np.uint8) * 255,
    )
    cv2.imwrite(
        (scene / "mask_visib" / "000000_000000.png").as_posix(),
        np.ones((3, 4), dtype=np.uint8) * 255,
    )
    (scene / BOP_FRAME_MAP_JSON).write_text(
        json.dumps(
            {
                "0": {
                    "sensor_name": "realsense_123",
                    "source_rgb": "rgb/raw_000010.png",
                    "source_depth": "depth/raw_000010.png",
                    "source_frame_id": "raw_000010",
                }
            }
        )
    )
    models_folder = run_root / BOP_DIR / MODELS_DIR
    models_folder.mkdir()
    model_path = models_folder / "obj_000001.ply"
    model_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "element face 0",
                "property list uchar int vertex_indices",
                "end_header",
                "-10 -5 0",
                "10 -5 0",
                "10 5 0",
                "-10 5 0",
                "",
            ]
        )
    )
    (run_root / BOP_DIR / BOP_EXPORT_MANIFEST).write_text(
        json.dumps(
            {
                "exports": [
                    {
                        "sensor_name": "realsense_123",
                        "scene_id": 1,
                        "split": "test",
                        "scene_folder": scene.as_posix(),
                    }
                ],
                "object_models": [
                    {
                        "object_name": "cube",
                        "obj_id": 1,
                        "source_path": model_path.as_posix(),
                        "bop_path": model_path.as_posix(),
                    }
                ],
            }
        )
    )
    result_file = run_root / RESULTS_DIR / BOP_DIR / "foundationpose_bop-test.csv"
    result_file.parent.mkdir(parents=True)
    result_file.write_text(
        "scene_id,im_id,obj_id,score,R,t,time\n"
        "1,0,1,0.9,1 0 0 0 1 0 0 0 1,0 0 10,0.01\n"
    )
    client = web_interface.app.test_client()

    response = client.get(
        "/artifacts/bop-frame",
        query_string={
            "run_root": run_root.as_posix(),
            "path": f"{BOP_DIR}/realsense_123/test/000001",
            "image_id": "0",
            "result_path": f"{RESULTS_DIR}/{BOP_DIR}/foundationpose_bop-test.csv",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["type"] == "bop_frame_detail"
    assert payload["scene"]["scene_id"] == 1
    assert payload["rgb"]["relative_path"] == (
        f"{BOP_DIR}/realsense_123/test/000001/{RGB_DIR}/000000.png"
    )
    assert payload["depth"]["relative_path"] == (
        f"{BOP_DIR}/realsense_123/test/000001/{DEPTH_DIR}/000000.png"
    )
    assert payload["gt_count"] == 1
    assert payload["gt_info"] == [{"bbox_obj": [0, 0, 4, 3], "visib_fract": 0.75}]
    assert payload["mask_artifacts"][0]["relative_path"] == (
        f"{BOP_DIR}/realsense_123/test/000001/mask/000000_000000.png"
    )
    assert payload["mask_visib_artifacts"][0]["relative_path"] == (
        f"{BOP_DIR}/realsense_123/test/000001/mask_visib/000000_000000.png"
    )
    assert payload["frame_map"]["source_rgb"] == "rgb/raw_000010.png"
    assert payload["result"]["matching_row_count"] == 1
    assert payload["result"]["projected_origin_count"] == 1
    assert payload["result"]["projected_model_bbox_count"] == 1
    assert payload["result"]["rows"][0]["score"] == 0.9
    assert payload["result"]["rows"][0]["t"] == [0.0, 0.0, 10.0]
    assert payload["result"]["rows"][0]["projected_origin"] == {
        "u": 2.0,
        "v": 1.0,
        "depth": 10.0,
        "source": "bop19_t_object_origin",
    }
    assert payload["result"]["rows"][0]["projected_model_bbox"] == {
        "bbox": [1.0, 0.5, 2.0, 1.0],
        "vertex_count": 4,
        "projected_vertex_count": 4,
        "model_relative_path": f"{BOP_DIR}/{MODELS_DIR}/obj_000001.ply",
        "object_name": "cube",
        "source": "bop19_pose_model_vertices",
    }

    overlay = client.get(
        "/artifacts/bop-frame-overlay",
        query_string={
            "run_root": run_root.as_posix(),
            "path": f"{BOP_DIR}/realsense_123/test/000001",
            "image_id": "0",
            "result_path": f"{RESULTS_DIR}/{BOP_DIR}/foundationpose_bop-test.csv",
        },
    )
    overlay_image = cv2.imdecode(
        np.frombuffer(overlay.get_data(), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert overlay.status_code == 200
    assert overlay.content_type == "image/png"
    assert overlay_image is not None
    assert overlay_image.shape[:2] == (3, 4)
    assert int(overlay_image.max()) > 0
