from __future__ import annotations

import json
import importlib.util
import logging
import os
import threading
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("POSETESTBOT_WEB_RUN_ROOTS", "/tmp")
os.environ.setdefault("POSETESTBOT_WEB_INPUT_ROOTS", "/tmp")

from posetestbot.config import DEFAULT_ROBOT_PORT, LAB_ROBOT_IP
from posetestbot.pipeline.run_config import (
    FixedFrameTransform,
    SensorRunConfig,
    create_run_config,
    write_run_config,
)
from posetestbot.web import legacy as web_legacy
from posetestbot.web.app import _PreviewPollLogFilter
from posetestbot.web.routes import sensors as web_sensors


def load_web_interface_app():
    module_path = Path(__file__).resolve().parents[1] / "web_interface.py"
    spec = importlib.util.spec_from_file_location("web_interface", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.app


app = load_web_interface_app()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    assert cv2.imwrite(path.as_posix(), image)


def test_index_uses_theme_aware_cow_branding_assets() -> None:
    client = app.test_client()

    response = client.get("/")
    html = response.get_data(as_text=True)
    light_logo = client.get("/assets/cow_light.png")
    dark_logo = client.get("/assets/cow_dark.png")
    favicon = client.get("/assets/cow_favicon.png")

    assert response.status_code == 200
    assert 'rel="icon" type="image/png" href="/assets/cow_favicon.png"' in html
    assert 'rel="apple-touch-icon" href="/assets/cow_favicon.png"' in html
    assert "PoseTestBot Operator Console" in html
    for asset in (light_logo, dark_logo, favicon):
        assert asset.status_code == 200
        assert asset.mimetype == "image/png"


def test_ui_bootstrap_includes_robot_control_defaults() -> None:
    client = app.test_client()

    response = client.get("/ui/bootstrap")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["schema_version"] == "web_bootstrap.v1"
    assert payload["robot"] == {"ip": LAB_ROBOT_IP, "port": DEFAULT_ROBOT_PORT}
    assert payload["brand"] == {
        "name": "PoseTestBot",
        "logo_url": "/assets/cow_light.png",
        "logo_urls": {
            "light": "/assets/cow_light.png",
            "dark": "/assets/cow_dark.png",
        },
        "favicon_url": "/assets/cow_favicon.png",
    }
    assert "/tmp" in payload["allowed_run_roots"]


def test_capture_jobs_tolerates_runner_without_resource_lock_reporting(
    monkeypatch,
) -> None:
    class MinimalRunner:
        def list(self):
            return []

    monkeypatch.setattr(web_legacy, "job_runner", MinimalRunner())

    response = app.test_client().get("/capture/jobs")

    assert response.status_code == 200
    assert response.get_json() == {
        "active_count": 0,
        "jobs": [],
        "resources": {},
        "run_root": None,
        "status_artifact": None,
    }


def test_preview_poll_log_filter_only_hides_sensor_preview_successes() -> None:
    poll_filter = _PreviewPollLogFilter()

    def record(message: str) -> logging.LogRecord:
        return logging.LogRecord("werkzeug", logging.INFO, "", 0, message, (), None)

    assert not poll_filter.filter(
        record(
            '10.145.8.50 - - "GET /sensors/previews/job/latest.jpg?t=1 '
            'HTTP/1.1" 200 -'
        )
    )
    assert poll_filter.filter(
        record('10.145.8.50 - - "GET /monitoring/webcam HTTP/1.1" 200 -')
    )
    assert poll_filter.filter(
        record('10.145.8.50 - - "GET /monitoring/webcam HTTP/1.1" 500 -')
    )
    assert poll_filter.filter(
        record('10.145.8.50 - - "POST /monitoring/webcam HTTP/1.1" 202 -')
    )


def test_pipeline_stage_and_sequence_endpoints_hide_downstream_ids() -> None:
    client = app.test_client()

    stages = client.get("/pipeline/stages").get_json()["stages"]
    sequences = client.get("/pipeline/sequences").get_json()["sequences"]

    stage_ids = {stage["id"] for stage in stages}
    sequence_ids = {sequence["id"] for sequence in sequences}
    assert "bop_export" in stage_ids
    assert "bop_evaluation" not in stage_ids
    assert "metric_report_export" not in stage_ids
    assert "fake_capture_to_bop_dataset_dry_run" not in sequence_ids
    assert "real_full_capture_validation" in sequence_ids
    assert "capture_to_bop_dataset_dry_run" in sequence_ids
    assert "foundationpose_to_bop_eval_dry_run" not in sequence_ids
    assert "foundationpose_runtime_to_bop_eval" not in sequence_ids


def test_start_iiwa_command_queues_real_robot_target(monkeypatch) -> None:
    class FakeJob:
        id = "robotjob1"
        status = "queued"

        def to_dict(self):
            return {
                "id": self.id,
                "status": self.status,
            }

    class FakeRunner:
        def __init__(self):
            self.submitted = []

        def submit(self, **kwargs):
            self.submitted.append(kwargs)
            return FakeJob()

    fake_runner = FakeRunner()
    monkeypatch.setattr(web_legacy, "job_runner", fake_runner)
    client = app.test_client()

    response = client.post(
        "/run-command",
        json={
            "command": "start_iiwa",
            "robot_ip": "172.31.1.150",
            "robot_port": 30305,
            "allow_real_robot": True,
            "allow_cameras": True,
        },
    )

    payload = response.get_json()
    assert response.status_code == 202
    assert payload["job_id"] == "robotjob1"
    assert fake_runner.submitted[0]["resources"] == ["robot_command"]
    assert fake_runner.submitted[0]["command"] == [
        "uv",
        "run",
        "python",
        "start_iiwa.py",
        "--ip_robot",
        "172.31.1.150",
        "--port_robot",
        "30305",
        "--manual-test-speed",
        "--allow-real-robot",
        "--allow-cameras",
    ]
    assert fake_runner.submitted[0]["parameters"]["robot_ip"] == "172.31.1.150"
    assert fake_runner.submitted[0]["parameters"]["robot_port"] == 30305
    assert fake_runner.submitted[0]["parameters"]["commanded_velocity_m_s"] == 0.1
    assert fake_runner.submitted[0]["parameters"]["allow_real_robot"] is True
    assert fake_runner.submitted[0]["parameters"]["allow_cameras"] is True


def test_start_iiwa_command_requires_both_execution_gates(monkeypatch) -> None:
    class FakeRunner:
        def __init__(self):
            self.submitted = []

        def submit(self, **kwargs):
            self.submitted.append(kwargs)
            raise AssertionError("ungated robot start should not queue a job")

    fake_runner = FakeRunner()
    monkeypatch.setattr(web_legacy, "job_runner", fake_runner)
    client = app.test_client()

    missing_both = client.post(
        "/run-command",
        json={"command": "start_iiwa"},
    )
    missing_camera_gate = client.post(
        "/run-command",
        json={"command": "start_iiwa", "allow_real_robot": True},
    )
    false_string = client.post(
        "/run-command",
        json={
            "command": "start_iiwa",
            "allow_real_robot": "true",
            "allow_cameras": "false",
        },
    )
    true_strings = client.post(
        "/run-command",
        json={
            "command": "start_iiwa",
            "allow_real_robot": "true",
            "allow_cameras": "true",
        },
    )

    assert missing_both.status_code == 400
    assert missing_camera_gate.status_code == 400
    assert false_string.status_code == 400
    assert true_strings.status_code == 400
    assert missing_both.get_json()["output"] == (
        "start_iiwa requires allow_real_robot=true and allow_cameras=true"
    )
    assert fake_runner.submitted == []


def test_stop_iiwa_command_queues_real_robot_target(monkeypatch) -> None:
    class FakeJob:
        id = "robotjob2"
        status = "queued"

        def to_dict(self):
            return {
                "id": self.id,
                "status": self.status,
            }

    class FakeRunner:
        def __init__(self):
            self.submitted = []

        def submit(self, **kwargs):
            self.submitted.append(kwargs)
            return FakeJob()

    fake_runner = FakeRunner()
    monkeypatch.setattr(web_legacy, "job_runner", fake_runner)
    client = app.test_client()

    response = client.post(
        "/run-command",
        json={
            "command": "stop_iiwa",
            "robot_ip": "172.31.1.151",
            "robot_port": "30306",
        },
    )

    assert response.status_code == 202
    assert fake_runner.submitted[0]["resources"] == ["robot_command"]
    assert fake_runner.submitted[0]["command"] == [
        "uv",
        "run",
        "python",
        "stop_iiwa.py",
        "--ip_robot",
        "172.31.1.151",
        "--port_robot",
        "30306",
    ]
    assert fake_runner.submitted[0]["parameters"]["robot_port"] == 30306


def test_iiwa_command_rejects_invalid_robot_target(monkeypatch) -> None:
    class FakeRunner:
        def __init__(self):
            self.submitted = []

        def submit(self, **kwargs):
            self.submitted.append(kwargs)
            raise AssertionError("invalid robot target should not queue a job")

    fake_runner = FakeRunner()
    monkeypatch.setattr(web_legacy, "job_runner", fake_runner)
    client = app.test_client()

    invalid_ip = client.post(
        "/run-command",
        json={
            "command": "start_iiwa",
            "robot_ip": "not-an-ip",
            "robot_port": 30300,
        },
    )
    invalid_port = client.post(
        "/run-command",
        json={
            "command": "stop_iiwa",
            "robot_ip": LAB_ROBOT_IP,
            "robot_port": 70000,
        },
    )

    assert invalid_ip.status_code == 400
    assert invalid_port.status_code == 400
    assert fake_runner.submitted == []


def test_run_command_unknown_command_still_returns_404() -> None:
    client = app.test_client()

    response = client.post("/run-command", json={"command": "does_not_exist"})

    assert response.status_code == 404
    assert response.get_json()["output"] == "Unknown command"


def test_artifact_endpoints_are_not_registered(tmp_path: Path) -> None:
    client = app.test_client()
    run_root = tmp_path / "run"
    run_root.mkdir()

    for path in (
        "/artifacts",
        "/artifacts/preview",
        "/artifacts/file",
        "/artifacts/bop-scene",
        "/artifacts/bop-frame",
        "/artifacts/bop-frame-overlay",
    ):
        assert client.get(path, query_string={"run_root": run_root.as_posix()}).status_code == 404


def test_pipeline_recommendations_endpoint_is_acquisition_only(tmp_path: Path) -> None:
    client = app.test_client()
    run_root = tmp_path / "run"
    write_run_config(run_root, create_run_config(run_root=run_root))

    response = client.get(
        "/pipeline/recommendations",
        query_string={"run_root": run_root.as_posix()},
    )

    payload = response.get_json()
    ids = {item["id"] for item in payload["recommendations"]}
    assert response.status_code == 200
    assert "write_run_preflight" in ids
    assert "evaluate_bop_results" not in ids
    assert "plan_foundationpose" not in ids


def test_run_config_endpoint_round_trips_realsense_inverted(tmp_path: Path) -> None:
    client = app.test_client()
    run_root = tmp_path / "run-inverted"

    response = client.post(
        "/run-config",
        json={
            "run_root": run_root.as_posix(),
            "sequence": "sync_aruco",
            "resolution": "720p",
            "fps": 6,
            "velocity": 0.2,
            "sensors": [
                {
                    "sensor_type": "realsense",
                    "device_id": "123",
                    "mounting_mode": "static",
                    "display_name": "Cell RealSense",
                    "inverted": True,
                },
                {
                    "sensor_type": "oak",
                    "device_id": "auto",
                    "mounting_mode": "static",
                    "display_name": "Cell OAK-D Pro",
                    "enabled": False,
                },
            ],
        },
    )

    payload = response.get_json()
    assert response.status_code == 201
    assert payload["config"]["capture"]["sensors"][0]["inverted"] is True
    assert payload["config"]["capture"]["sensors"][1]["inverted"] is False
    assert payload["config"]["capture"]["sensors"][1]["enabled"] is False

    loaded = client.get(
        "/run-config",
        query_string={"run_root": run_root.as_posix()},
    ).get_json()

    assert loaded["config"]["capture"]["sensors"][0]["inverted"] is True
    assert loaded["config"]["capture"]["sensors"][0]["sensor_type"] == "realsense_d435"
    assert loaded["config"]["capture"]["sensors"][1]["enabled"] is False


def test_run_config_partial_post_preserves_existing_operator_contract(
    tmp_path: Path,
) -> None:
    client = app.test_client()
    run_root = tmp_path / "partial-config"
    calibration_target = {
        "target_id": "target-1",
        "bundle_path": "calibration_targets/target-1",
        "source_sha256": "a" * 64,
        "spec_sha256": "b" * 64,
        "pdf_sha256": "c" * 64,
        "configuration_sha256": "d" * 64,
        "geometry_sha256": "e" * 64,
        "placement": {"mode": "unknown"},
    }
    pose_template = {
        "selection_artifact": "pose_template_selection.json",
        "template_uuid": "template-1",
    }
    initial = create_run_config(
        run_root=run_root,
        run_name="Research combined-view run",
        resolution="720p",
        fps=6,
        velocity_m_s=0.123,
        sensors=(
            SensorRunConfig(
                "realsense_d435",
                "static-1",
                "Static D435",
                mounting_mode="static",
            ),
        ),
        dataset_mode="pose_template",
        pose_template=pose_template,
        calibration_profiles="calibration/full.json",
        intrinsic_calibration_profiles="calibration/intrinsics.json",
        calibration_target=calibration_target,
        sequence_id="sync_aruco",
        sequence_options={
            "sync_quality": {
                "min_match_ratio": 0.75,
            }
        },
        plan_only=False,
        fixed_transforms=(
            FixedFrameTransform(
                from_frame="robot_flange",
                to_frame="tcp",
                rotation_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                translation_mm=(0.0, 0.0, 125.0),
                source="tool_measurement",
            ),
        ),
    )
    write_run_config(run_root, initial)

    response = client.post(
        "/run-config",
        json={
            "run_root": run_root.as_posix(),
            "fps": 12,
        },
    )

    assert response.status_code == 201, response.get_json()
    config = response.get_json()["config"]
    assert config["run_name"] == "Research combined-view run"
    assert config["capture"]["fps"] == 12
    assert config["capture"]["velocity_m_s"] == 0.123
    assert config["robot_profile"]["cartesian_velocity_m_s"] == 0.123
    assert config["pipeline"] == {
        "sequence_id": "sync_aruco",
        "plan_only": False,
        "options": {
            "sync_quality": {
                "min_match_ratio": 0.75,
            }
        },
    }
    assert config["frames"] == initial.to_dict()["frames"]
    assert config["calibration_profiles"] == "calibration/full.json"
    assert config["intrinsic_calibration_profiles"] == "calibration/intrinsics.json"
    assert config["calibration_target"] == calibration_target
    assert config["dataset_mode"] == "pose_template"
    assert config["pose_template"] == pose_template


def test_run_config_endpoint_preserves_hardware_trigger_and_freezes_it_after_raw_evidence(
    tmp_path: Path,
) -> None:
    client = app.test_client()
    run_root = tmp_path / "run-hardware-trigger"
    sensors = [
        {
            "sensor_type": "realsense_d435",
            "device_id": "wrist-1",
            "mounting_mode": "eye_in_hand",
            "display_name": "Wrist D435",
        },
        {
            "sensor_type": "realsense_d435",
            "device_id": "static-1",
            "mounting_mode": "static",
            "display_name": "Static D435",
        },
    ]
    synchronization = {
        "schema_version": "capture_synchronization.v1",
        "mode": "hardware_trigger",
        "implementation": "realsense_inter_cam_sync",
        "scope": "depth_exposure",
        "group_id": "mixed-depth-rig",
        "master_sensor_key": "realsense_d435:wrist-1",
        "max_depth_timestamp_skew_ms": 2.0,
    }

    created = client.post(
        "/run-config",
        json={
            "run_root": run_root.as_posix(),
            "sensors": sensors,
            "synchronization": synchronization,
        },
    )

    assert created.status_code == 201
    assert created.get_json()["config"]["schema_version"] == "run_config.v3"
    assert (
        created.get_json()["config"]["capture"]["synchronization"]
        == synchronization
    )

    preserved = client.post(
        "/run-config",
        json={
            "run_root": run_root.as_posix(),
            "sensors": sensors,
            "fps": 8,
        },
    )

    assert preserved.status_code == 201
    assert (
        preserved.get_json()["config"]["capture"]["synchronization"]
        == synchronization
    )
    assert preserved.get_json()["config"]["capture"]["fps"] == 8

    preserved_without_sensor_payload = client.post(
        "/run-config",
        json={
            "run_root": run_root.as_posix(),
            "fps": 9,
        },
    )

    assert preserved_without_sensor_payload.status_code == 201
    preserved_capture = preserved_without_sensor_payload.get_json()["config"][
        "capture"
    ]
    assert preserved_capture["synchronization"] == synchronization
    assert [
        {
            key: sensor[key]
            for key in (
                "sensor_type",
                "device_id",
                "mounting_mode",
                "display_name",
            )
        }
        for sensor in preserved_capture["sensors"]
    ] == sensors
    assert preserved_capture["fps"] == 9

    (run_root / "raw_robot_ee_poses.json").write_text("{}")
    rejected = client.post(
        "/run-config",
        json={
            "run_root": run_root.as_posix(),
            "sensors": sensors,
            "synchronization": {
                "schema_version": "capture_synchronization.v1",
                "mode": "timestamp_aligned",
            },
        },
    )

    assert rejected.status_code == 400
    assert "Cannot change the hardware_trigger policy" in rejected.get_json()["output"]

    changed_membership = client.post(
        "/run-config",
        json={
            "run_root": run_root.as_posix(),
            "sensors": [
                sensors[0],
                {
                    **sensors[1],
                    "device_id": "static-2",
                },
            ],
        },
    )
    assert changed_membership.status_code == 400
    assert "camera membership" in changed_membership.get_json()["output"]

    loaded = client.get(
        "/run-config",
        query_string={"run_root": run_root.as_posix()},
    ).get_json()
    assert loaded["config"]["capture"]["synchronization"] == synchronization


def test_run_config_endpoint_rejects_truthy_string_enabled(tmp_path: Path) -> None:
    response = app.test_client().post(
        "/run-config",
        json={
            "run_root": (tmp_path / "run-string-enabled").as_posix(),
            "sensors": [
                {
                    "sensor_type": "realsense",
                    "device_id": "123",
                    "mounting_mode": "eye_in_hand",
                    "display_name": "Wrist RealSense",
                    "enabled": "false",
                }
            ],
        },
    )

    assert response.status_code == 400
    assert "literal JSON boolean" in response.get_json()["output"]


def test_run_config_explicit_redetection_replaces_preserved_sensors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = app.test_client()
    run_root = tmp_path / "redetected-run"
    write_run_config(
        run_root,
        create_run_config(
            run_root=run_root,
            sensors=(
                SensorRunConfig(
                    "realsense_d435",
                    "old",
                    "Old D435",
                    mounting_mode="static",
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "posetestbot.web.legacy.collect_sensor_status",
        lambda: {
            "families": [
                {
                    "sensor_type": "realsense_d435",
                        "devices": [
                            {
                                "sensor_type": "realsense_d435",
                                "device_id": "new",
                            "display_name": "Detected D435",
                            "connected": True,
                        }
                    ],
                }
            ]
        },
    )

    response = client.post(
        "/run-config",
        json={
            "run_root": run_root.as_posix(),
            "from_detected_sensors": True,
            "mounting_mode": "eye_in_hand",
        },
    )

    assert response.status_code == 201
    sensors = response.get_json()["config"]["capture"]["sensors"]
    assert [(sensor["device_id"], sensor["mounting_mode"]) for sensor in sensors] == [
        ("new", "eye_in_hand")
    ]


def test_run_config_endpoint_rejects_retired_robot_mode(tmp_path: Path) -> None:
    response = app.test_client().post(
        "/run-config",
        json={
            "run_root": (tmp_path / "run-retired-mode").as_posix(),
            "robot_mode": "real",
        },
    )

    assert response.status_code == 400
    assert "robot_mode is retired" in response.get_json()["output"]


def test_capture_execution_endpoint_rejects_retired_mode(tmp_path: Path) -> None:
    response = app.test_client().post(
        "/capture-plan/execution",
        json={
            "run_root": (tmp_path / "run-retired-execution-mode").as_posix(),
            "mode": "full",
        },
    )

    assert response.status_code == 400
    assert "execution mode is retired" in response.get_json()["output"]


def test_capture_plan_post_endpoints_require_literal_execution_gates(
    tmp_path: Path,
) -> None:
    client = app.test_client()
    preflight_root = tmp_path / "strict-preflight-web"
    execution_root = tmp_path / "strict-execution-web"

    preflight = client.post(
        "/capture-plan/preflight",
        json={
            "run_root": preflight_root.as_posix(),
            "allow_real_robot": "true",
        },
    )
    execution = client.post(
        "/capture-plan/execution",
        json={
            "run_root": execution_root.as_posix(),
            "allow_real_robot": "true",
            "allow_cameras": "true",
        },
    )

    assert preflight.status_code == 400
    assert execution.status_code == 400
    assert "literal JSON boolean" in preflight.get_json()["output"]
    assert "literal JSON boolean" in execution.get_json()["output"]
    assert not preflight_root.exists()
    assert not execution_root.exists()


def test_pipeline_and_sequence_web_boundaries_reject_string_gates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeRunner:
        def __init__(self):
            self.submitted = []

        def submit(self, **kwargs):
            self.submitted.append(kwargs)
            raise AssertionError("string acknowledgements must not queue a job")

    fake_runner = FakeRunner()
    monkeypatch.setattr(web_legacy, "job_runner", fake_runner)
    client = app.test_client()
    string_options = {
        "capture_plan_preflight": {"allow_real_robot": "true"},
        "capture_execution_plan": {
            "allow_cameras": "true",
            "allow_real_robot": "true",
        },
        "capture_execution": {
            "allow_cameras": "true",
            "allow_real_robot": "true",
        },
    }

    stage = client.post(
        "/pipeline/run",
        json={
            "run_root": (tmp_path / "strict-stage").as_posix(),
            "stage": "capture_execution",
            "options": string_options["capture_execution"],
        },
    )
    sequence = client.post(
        "/pipeline/run-sequence",
        json={
            "run_root": (tmp_path / "strict-sequence").as_posix(),
            "sequence": "real_full_capture_validation",
            "options": string_options,
        },
    )

    assert stage.status_code == 400
    assert sequence.status_code == 400
    assert fake_runner.submitted == []


def test_real_sequence_web_submission_passes_ephemeral_gates_only_in_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeJob:
        id = "sequence-job"
        status = "queued"

        def to_dict(self):
            return {"id": self.id, "status": self.status}

    class FakeRunner:
        def __init__(self):
            self.submitted = []

        def submit(self, **kwargs):
            self.submitted.append(kwargs)
            return FakeJob()

    fake_runner = FakeRunner()
    monkeypatch.setattr(web_legacy, "job_runner", fake_runner)
    client = app.test_client()
    options = {
        "capture_plan_preflight": {"allow_real_robot": True},
        "capture_execution_plan": {
            "allow_cameras": True,
            "allow_real_robot": True,
        },
        "capture_execution": {
            "allow_cameras": True,
            "allow_real_robot": True,
        },
    }

    response = client.post(
        "/pipeline/run-sequence",
        json={
            "run_root": (tmp_path / "ephemeral-sequence").as_posix(),
            "sequence": "real_full_capture_validation",
            "options": options,
        },
    )

    assert response.status_code == 202
    submission = fake_runner.submitted[0]
    serialized = json.dumps(
        {"command": submission["command"], "parameters": submission["parameters"]}
    )
    assert "allow_cameras" not in serialized
    assert "allow_real_robot" not in serialized
    assert "POSETESTBOT_SEQUENCE_EXECUTION_ACKNOWLEDGEMENTS" in submission["env"]


def test_sensor_alias_endpoint_round_trips_lab_local_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    alias_path = tmp_path / "sensor_aliases.json"
    monkeypatch.setattr(web_sensors, "DEFAULT_SENSOR_ALIASES_PATH", alias_path)
    client = app.test_client()

    response = client.put(
        "/sensors/aliases",
        json={
            "aliases": {
                "realsense_d435:123": {
                    "alias": "Wrist Camera",
                    "mounting_mode": "eye_in_hand",
                    "inverted": True,
                }
            }
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["aliases"]["realsense_d435:123"]["alias"] == "Wrist Camera"

    loaded = client.get("/sensors/aliases").get_json()
    assert loaded["aliases"]["realsense_d435:123"]["inverted"] is True


def test_overview_endpoint_reports_sequence_steps(tmp_path: Path) -> None:
    client = app.test_client()
    run_root = tmp_path / "run-overview"
    write_run_config(
        run_root,
        create_run_config(run_root=run_root, sequence_id="sync_to_bop_dry_run"),
    )

    response = client.get(
        "/ui/overview",
        query_string={"run_root": run_root.as_posix()},
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["config"]["pipeline"]["sequence_id"] == "sync_to_bop_dry_run"
    assert payload["calibration_sync"] == {
        "status": "not_configured",
        "sensors": [],
    }
    assert any(step["stage_id"] == "sync_quality" for step in payload["steps"])
    assert any(section["id"] == "sensors" for section in payload["sidebar"])


def test_overview_uses_validated_sync_quality_as_run_level_sync_evidence(
    tmp_path: Path,
) -> None:
    client = app.test_client()
    run_root = tmp_path / "run-sync-overview"
    write_run_config(
        run_root,
        create_run_config(run_root=run_root, sequence_id="sync_to_bop_dry_run"),
    )
    write_json(
        run_root / "sync_quality_report.json",
        {
            "schema_version": "sync_quality_report.v2",
            "overall_status": "ok",
            "sensor_count": 1,
            "sensors": [{"sensor_name": "realsense_123"}],
            "checks": [],
        },
    )

    payload = client.get(
        "/ui/overview", query_string={"run_root": run_root.as_posix()}
    ).get_json()

    sync_section = next(item for item in payload["sidebar"] if item["id"] == "sync")
    sync_step = next(item for item in payload["steps"] if item["stage_id"] == "sync_run")
    assert sync_section["artifacts"] == [
        {"path": "sync_quality_report.json", "exists": True, "status": "ok"}
    ]
    assert sync_step["status"] == "complete"


def test_overview_rejects_canceled_or_malformed_completion_evidence(
    tmp_path: Path,
) -> None:
    client = app.test_client()
    run_root = tmp_path / "run-invalid-overview"
    write_run_config(run_root, create_run_config(run_root=run_root))
    write_json(
        run_root / "capture_execution_report.json",
        {
            "schema_version": "capture_execution_report.v1",
            "status": "canceled",
        },
    )
    (run_root / "calibration_profiles.json").write_text("{not json\n")
    bop_manifest = run_root / "bop" / "bop_export_manifest.json"
    write_json(bop_manifest, {"schema_version": "bop_export_manifest.v3"})

    payload = client.get(
        "/ui/overview", query_string={"run_root": run_root.as_posix()}
    ).get_json()

    chips = {
        chip["path"]: chip
        for section in payload["sidebar"]
        for chip in section["artifacts"]
    }
    assert chips["capture_execution_report.json"]["status"] == "canceled"
    assert chips["calibration_profiles.json"]["status"] == "invalid"
    assert chips["bop/bop_export_manifest.json"]["status"] == "invalid"
    assert next(
        item for item in payload["sidebar"] if item["id"] == "capture"
    )["status"] == "blocked"


def test_overview_endpoint_treats_missing_run_config_as_empty_setup(tmp_path: Path) -> None:
    client = app.test_client()
    run_root = tmp_path / "empty-web-run"
    run_root.mkdir()

    response = client.get(
        "/ui/overview",
        query_string={"run_root": run_root.as_posix()},
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["config"] is None
    assert payload["config_error"] is None
    assert payload["steps"] == []


def test_web_rejects_run_root_symlink_escape(tmp_path: Path) -> None:
    escape = tmp_path / "escape"
    escape.symlink_to("/etc", target_is_directory=True)

    response = app.test_client().get(
        "/ui/overview", query_string={"run_root": escape.as_posix()}
    )

    assert response.status_code == 400
    assert "allowed root" in response.get_json()["output"]


def test_web_rejects_invalid_boolean_instead_of_using_truthiness(tmp_path: Path) -> None:
    response = app.test_client().post(
        "/run-config",
        json={
            "run_root": (tmp_path / "invalid-bool").as_posix(),
            "plan_only": "definitely",
        },
    )

    assert response.status_code == 400
    assert "plan_only must be" in response.get_json()["output"]


def test_web_accepts_recognized_false_boolean_string(tmp_path: Path) -> None:
    response = app.test_client().post(
        "/run-config",
        json={
            "run_root": (tmp_path / "false-bool").as_posix(),
            "plan_only": "false",
        },
    )

    assert response.status_code == 201
    assert response.get_json()["config"]["pipeline"]["plan_only"] is False


def test_web_pipeline_output_path_must_remain_under_run_root(tmp_path: Path) -> None:
    run_root = tmp_path / "scoped-run"
    response = app.test_client().post(
        "/pipeline/run",
        json={
            "run_root": run_root.as_posix(),
            "stage": "bop_export",
            "options": {"output_folder": "../outside"},
        },
    )

    assert response.status_code == 400
    assert "output_folder" in response.get_json()["output"]


def test_web_pipeline_input_path_must_use_run_or_input_roots(tmp_path: Path) -> None:
    response = app.test_client().post(
        "/pipeline/run",
        json={
            "run_root": (tmp_path / "scoped-input-run").as_posix(),
            "stage": "bop_export",
            "options": {"calibration_profiles": "/etc/profiles.json"},
        },
    )

    assert response.status_code == 400
    assert "calibration_profiles" in response.get_json()["output"]


def test_sensor_snapshot_submission_queues_camera_job(monkeypatch, tmp_path: Path) -> None:
    class FakeJob:
        id = "job123"
        status = "queued"
        parameters = {"snapshot_root": (tmp_path / "snapshots").as_posix()}

        def to_dict(self):
            return {
                "id": self.id,
                "status": self.status,
                "parameters": self.parameters,
            }

    class FakeRunner:
        def __init__(self):
            self.submitted = None

        def submit(self, **kwargs):
            self.submitted = kwargs
            return FakeJob()

    fake_runner = FakeRunner()
    monkeypatch.setattr(web_sensors, "job_runner", fake_runner)
    monkeypatch.setattr(
        web_sensors,
        "snapshot_batch_root",
        lambda: tmp_path / "snapshots",
    )
    monkeypatch.setattr(
        web_sensors,
        "collect_sensor_status",
        lambda: {
            "schema_version": "sensor_status.v1",
            "families": [
                {
                    "devices": [
                        {
                            "sensor_type": "realsense_d435",
                            "device_id": "123",
                            "display_name": "RealSense 123",
                            "effective_display_name": "Wrist Camera",
                            "connected": True,
                            "metadata": {},
                        }
                    ]
                }
            ],
            "total_connected": 1,
            "all_expected_connected": True,
            "expected_counts_requested": False,
        },
    )
    client = app.test_client()

    response = client.post(
        "/sensors/snapshots",
        json={"selected": ["realsense_d435:123"]},
    )

    payload = response.get_json()
    assert response.status_code == 202
    assert payload["job_id"] == "job123"
    assert fake_runner.submitted["resources"] == [
        "camera:realsense_d435:123",
        "disk_io",
    ]
    assert fake_runner.submitted["command"][:4] == [
        "uv",
        "run",
        "python",
        "scripts/capture_sensor_snapshot.py",
    ]


def test_sensor_preview_submission_queues_one_job_per_selected_sensor(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeJob:
        def __init__(self, job_id: str, parameters: dict):
            self.id = job_id
            self.status = "queued"
            self.parameters = parameters
            self.message = None

        def to_dict(self):
            return {
                "id": self.id,
                "status": self.status,
                "parameters": self.parameters,
                "message": self.message,
            }

    class FakeRunner:
        def __init__(self):
            self.submitted = []

        def list(self):
            return []

        def submit(self, **kwargs):
            self.submitted.append(kwargs)
            return FakeJob(f"job{len(self.submitted)}", dict(kwargs["parameters"]))

    roots = iter([tmp_path / "preview-1", tmp_path / "preview-2"])
    fake_runner = FakeRunner()
    monkeypatch.setattr(web_sensors, "job_runner", fake_runner)
    monkeypatch.setattr(web_sensors, "preview_stream_root", lambda: next(roots))
    monkeypatch.setattr(
        web_sensors,
        "collect_sensor_status",
        lambda: {
            "schema_version": "sensor_status.v1",
            "families": [
                {
                    "devices": [
                        {
                            "sensor_type": "realsense_d435",
                            "device_id": "825412070181",
                            "display_name": "RealSense 1",
                            "connected": True,
                            "metadata": {"video_nodes": []},
                        },
                        {
                            "sensor_type": "realsense_d435",
                            "device_id": "923322072633",
                            "display_name": "RealSense 2",
                            "connected": True,
                            "metadata": {"video_nodes": []},
                        },
                    ]
                }
            ],
            "total_connected": 2,
        },
    )
    client = app.test_client()

    response = client.post(
        "/sensors/previews",
        json={
            "selected": [
                "realsense_d435:825412070181",
                "realsense_d435:923322072633",
            ]
        },
    )

    payload = response.get_json()
    assert response.status_code == 202
    assert len(payload["jobs"]) == 2
    assert [item["resources"] for item in fake_runner.submitted] == [
        ["camera:realsense_d435:825412070181"],
        ["camera:realsense_d435:923322072633"],
    ]
    assert fake_runner.submitted[0]["command"][:4] == [
        "uv",
        "run",
        "python",
        "scripts/stream_sensor_rgb_preview.py",
    ]


def test_sensor_preview_rejects_family_without_live_preview(monkeypatch) -> None:
    class FakeRunner:
        def __init__(self):
            self.submitted = []

        def list(self):
            return []

        def submit(self, **kwargs):
            self.submitted.append(kwargs)
            raise AssertionError("unsupported preview should not queue a job")

    fake_runner = FakeRunner()
    monkeypatch.setattr(web_sensors, "job_runner", fake_runner)
    client = app.test_client()

    response = client.post(
        "/sensors/previews",
        json={
            "sensors": [
                {
                    "sensor_type": "zed_2i",
                    "device_id": "zed-001",
                    "display_name": "ZED 2i",
                }
            ]
        },
    )

    assert response.status_code == 409
    assert response.get_json()["errors"] == [
        {
            "sensor_key": "zed_2i:zed-001",
            "error": "Live RGB preview is not supported for Stereolabs ZED 2i.",
        }
    ]
    assert fake_runner.submitted == []


def test_sensor_status_keeps_preview_claimed_oak_visible(monkeypatch) -> None:
    class FakeJob:
        id = "oak-preview"
        status = "running"
        parameters = {
            "preview_root": "working_data/sensor_previews/oak-preview",
            "sensor_key": "oak_d_pro:18443010314F3B1300",
            "sensor_preview": True,
            "sensor_spec": {
                "sensor_type": "oak_d_pro",
                "device_id": "18443010314F3B1300",
                "display_name": "OAK-D Pro 18443010314F3B1300",
                "metadata": {"state": "X_LINK_UNBOOTED"},
            },
        }

    class FakeRunner:
        def list(self):
            return [FakeJob()]

    monkeypatch.setattr(web_sensors, "job_runner", FakeRunner())
    monkeypatch.setattr(
        web_sensors,
        "collect_sensor_status",
        lambda **_kwargs: {
            "schema_version": "sensor_status.v1",
            "families": [
                {
                    "sensor_type": "oak_d_pro",
                    "display_name": "Luxonis OAK-D Pro",
                    "devices": [],
                    "connected_count": 0,
                    "expected_count": 1,
                    "meets_expected": False,
                    "diagnostics": [
                        {
                            "code": "expected_count_not_met",
                            "message": "Connected 0 of expected 1 device(s).",
                        }
                    ],
                }
            ],
            "total_connected": 0,
            "all_expected_connected": False,
            "expected_counts_requested": True,
        },
    )

    response = app.test_client().get("/sensors/status")

    payload = response.get_json()
    family = payload["families"][0]
    assert response.status_code == 200
    assert payload["total_connected"] == 1
    assert payload["total_capture_ready"] == 1
    assert payload["all_expected_connected"] is True
    assert family["connected_count"] == 1
    assert family["capture_ready_count"] == 1
    assert family["meets_expected"] is True
    assert family["diagnostics"] == []
    assert family["devices"] == [
        {
            "capture_readiness_reason": None,
            "capture_ready": True,
            "connected": True,
            "device_id": "18443010314F3B1300",
            "discovery_state": "claimed_by_preview",
            "display_name": "OAK-D Pro 18443010314F3B1300",
            "metadata": {"state": "X_LINK_UNBOOTED"},
            "sensor_type": "oak_d_pro",
        }
    ]


def test_sensor_preview_submission_passes_explicit_inverted_spec(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeJob:
        def __init__(self, job_id: str, parameters: dict):
            self.id = job_id
            self.status = "queued"
            self.parameters = parameters
            self.message = None

        def to_dict(self):
            return {
                "id": self.id,
                "status": self.status,
                "parameters": self.parameters,
                "message": self.message,
            }

    class FakeRunner:
        def __init__(self):
            self.submitted = []

        def list(self):
            return []

        def submit(self, **kwargs):
            self.submitted.append(kwargs)
            return FakeJob("job1", dict(kwargs["parameters"]))

    fake_runner = FakeRunner()
    monkeypatch.setattr(web_sensors, "job_runner", fake_runner)
    monkeypatch.setattr(web_sensors, "preview_stream_root", lambda: tmp_path / "preview")
    client = app.test_client()

    response = client.post(
        "/sensors/previews",
        json={
            "sensors": [
                {
                    "sensor_type": "realsense_d435",
                    "device_id": "825412070181",
                    "display_name": "Wrist RealSense",
                    "inverted": True,
                    "metadata": {"video_nodes": [{"path": "/dev/video4"}]},
                }
            ]
        },
    )

    payload = response.get_json()
    assert response.status_code == 202
    assert len(payload["jobs"]) == 1
    command = fake_runner.submitted[0]["command"]
    sensor_json = command[command.index("--sensor-json") + 1]
    spec = json.loads(sensor_json)
    assert spec["inverted"] is True
    assert spec["metadata"]["video_nodes"][0]["path"] == "/dev/video4"
    assert fake_runner.submitted[0]["parameters"]["inverted"] is True
    assert fake_runner.submitted[0]["parameters"]["sensor_spec"]["metadata"][
        "video_nodes"
    ][0]["path"] == "/dev/video4"


def test_sensor_preview_detail_reports_missing_status_payload(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeJob:
        id = "preview-missing"
        status = "running"
        message = None
        parameters = {
            "preview_root": (tmp_path / "preview").as_posix(),
            "sensor_key": "realsense_d435:825412070181",
            "sensor_type": "realsense_d435",
            "device_id": "825412070181",
            "inverted": True,
            "sensor_preview": True,
        }

        def to_dict(self):
            return {
                "id": self.id,
                "status": self.status,
                "message": self.message,
                "parameters": self.parameters,
            }

    class FakeRunner:
        def get(self, job_id):
            assert job_id == "preview-missing"
            return FakeJob()

    monkeypatch.setattr(web_sensors, "job_runner", FakeRunner())
    client = app.test_client()

    response = client.get("/sensors/previews/preview-missing")

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["preview_status"]["status"] == "waiting"
    assert payload["preview_status"]["sensor_key"] == "realsense_d435:825412070181"
    assert payload["preview_status"]["inverted"] is True
    assert payload["preview_status"]["latest_image"] is None


def test_sensor_preview_detail_reports_failed_status_payload(
    monkeypatch,
    tmp_path: Path,
) -> None:
    preview_root = tmp_path / "preview"
    write_json(
        preview_root / "preview_status.json",
        {
            "schema_version": "sensor_rgb_preview.v1",
            "status": "failed",
            "sensor_key": "realsense_d435:825412070181",
            "frame_count": 0,
            "latest_image": None,
            "error": "RuntimeError: camera missing",
        },
    )

    class FakeJob:
        id = "preview-failed"
        status = "failed"
        message = "Command exited with status 2."
        parameters = {
            "preview_root": preview_root.as_posix(),
            "sensor_key": "realsense_d435:825412070181",
            "sensor_preview": True,
        }

        def to_dict(self):
            return {
                "id": self.id,
                "status": self.status,
                "message": self.message,
                "parameters": self.parameters,
            }

    class FakeRunner:
        def get(self, job_id):
            assert job_id == "preview-failed"
            return FakeJob()

    monkeypatch.setattr(web_sensors, "job_runner", FakeRunner())
    client = app.test_client()

    response = client.get("/sensors/previews/preview-failed")

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["preview_status"]["status"] == "failed"
    assert payload["preview_status"]["error"] == "RuntimeError: camera missing"


def test_sensor_preview_list_hides_stale_active_image_during_cleanup(
    monkeypatch,
    tmp_path: Path,
) -> None:
    preview_root = tmp_path / "stale-preview"
    write_png(preview_root / "latest.jpg")
    write_json(
        preview_root / "preview_status.json",
        {
            "schema_version": "sensor_rgb_preview.v1",
            "status": "running",
            "sensor_key": "realsense_d435:825412070181",
            "heartbeat_at": "2000-01-01T00:00:00Z",
            "frame_count": 12,
            "latest_image": "latest.jpg",
            "error": None,
        },
    )

    class FakeJob:
        id = "stale-preview"
        status = "running"
        message = None
        parameters = {
            "preview_root": preview_root.as_posix(),
            "sensor_key": "realsense_d435:825412070181",
            "sensor_preview": True,
        }

        def to_dict(self):
            return {
                "id": self.id,
                "status": self.status,
                "message": self.message,
                "parameters": self.parameters,
            }

    class FakeRunner:
        def __init__(self):
            self.job = FakeJob()
            self.cancelled = threading.Event()

        def list(self):
            return [self.job]

        def cancel(self, job_id):
            assert job_id == self.job.id
            self.job.status = "canceled"
            self.cancelled.set()
            return self.job

    runner = FakeRunner()
    monkeypatch.setattr(web_sensors, "job_runner", runner)
    client = app.test_client()

    response = client.get("/sensors/previews?include_terminal=true")

    assert response.status_code == 200
    assert response.get_json()["jobs"] == []
    assert runner.cancelled.wait(timeout=1)


def test_sensor_preview_image_disables_browser_cache(monkeypatch, tmp_path: Path) -> None:
    preview_root = tmp_path / "preview"
    write_png(preview_root / "latest.jpg")

    class FakeJob:
        id = "preview-live"
        parameters = {
            "preview_root": preview_root.as_posix(),
            "sensor_preview": True,
        }

    class FakeRunner:
        def get(self, job_id):
            assert job_id == "preview-live"
            return FakeJob()

    monkeypatch.setattr(web_sensors, "job_runner", FakeRunner())
    client = app.test_client()

    response = client.get("/sensors/previews/preview-live/latest.jpg")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store, max-age=0"


def test_snapshot_worker_reports_inaccessible_realsense_video_nodes(
    tmp_path: Path,
) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "capture_sensor_snapshot.py"
    spec = importlib.util.spec_from_file_location(
        "posetestbot_snapshot_worker_test",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    record = module._capture_one(
        snapshot_root=tmp_path / "snapshots",
        spec={
            "sensor_type": "realsense_d435",
            "device_id": "123",
            "metadata": {
                "video_nodes": [{"path": "/dev/video0", "accessible": False}],
                "video_accessible": False,
            },
        },
        fps=6,
        resolution="720p",
        max_frames=1,
    )

    assert record["status"] == "failed"
    assert "not accessible" in record["error"]
    assert record["diagnostics"][0]["code"] == "video_permission_denied"
