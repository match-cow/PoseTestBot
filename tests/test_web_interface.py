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
from posetestbot.pipeline.run_config import create_run_config, write_run_config
from posetestbot.web import legacy as web_legacy
from posetestbot.web.app import _PreviewPollLogFilter
from posetestbot.web.routes import monitoring as web_monitoring
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


def test_index_serves_bundled_spa_without_cdn_dependencies() -> None:
    client = app.test_client()

    response = client.get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert '<div id="root"></div>' in html
    assert "/static/ui/assets/index-" in html
    assert "cdn.jsdelivr.net" not in html
    assert "bootstrap" not in html.lower()


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


def test_pipeline_sequence_api_lists_acquisition_sequences_only() -> None:
    client = app.test_client()

    response = client.get("/pipeline/sequences")
    sequence_ids = {item["id"] for item in response.get_json()["sequences"]}

    assert response.status_code == 200
    assert "real_full_capture_validation" in sequence_ids
    assert "capture_to_bop_dataset_dry_run" in sequence_ids
    assert "fake_capture_to_bop_dataset_dry_run" not in sequence_ids
    assert "foundationpose_runtime_to_bop_eval" not in sequence_ids


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


def test_ugreen_sidebar_monitor_queues_webrtc_worker(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeJob:
        id = "webcam-job"
        status = "queued"
        message = None

        def __init__(self, parameters: dict):
            self.parameters = parameters

        def to_dict(self):
            return {
                "id": self.id,
                "status": self.status,
                "message": self.message,
                "parameters": self.parameters,
            }

    class FakeRunner:
        def __init__(self):
            self.submitted = []

        def list(self):
            return []

        def submit(self, **kwargs):
            self.submitted.append(kwargs)
            return FakeJob(dict(kwargs["parameters"]))

    fake_runner = FakeRunner()
    monkeypatch.setattr(web_monitoring, "job_runner", fake_runner)
    monkeypatch.setattr(
        web_monitoring,
        "monitor_stream_root",
        lambda: tmp_path / "webcam-monitor",
    )
    client = app.test_client()

    response = client.post("/monitoring/webcam")

    assert response.status_code == 202
    submission = fake_runner.submitted[0]
    assert submission["resources"] == ["monitoring_camera:0c45:2283"]
    assert submission["parameters"]["monitor_webcam"] is True
    command = submission["command"]
    assert command[:4] == [
        "uv",
        "run",
        "python",
        "scripts/run_monitor_webrtc.py",
    ]
    assert command[command.index("--fps") + 1] == "30"
    assert command[command.index("--width") + 1] == "640"
    assert command[command.index("--height") + 1] == "480"
    assert command[command.index("--vendor-id") + 1] == "0c45"
    assert command[command.index("--product-id") + 1] == "2283"


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
    assert "foundationpose_to_bop_eval_dry_run" not in sequence_ids


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
    ]
    assert fake_runner.submitted[0]["parameters"]["robot_ip"] == "172.31.1.150"
    assert fake_runner.submitted[0]["parameters"]["robot_port"] == 30305
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

    assert missing_both.status_code == 400
    assert missing_camera_gate.status_code == 400
    assert false_string.status_code == 400
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
            "object_folder": "object_models",
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
                },
            ],
        },
    )

    payload = response.get_json()
    assert response.status_code == 201
    assert payload["config"]["capture"]["sensors"][0]["inverted"] is True
    assert payload["config"]["capture"]["sensors"][1]["inverted"] is False

    loaded = client.get(
        "/run-config",
        query_string={"run_root": run_root.as_posix()},
    ).get_json()

    assert loaded["config"]["capture"]["sensors"][0]["inverted"] is True
    assert loaded["config"]["capture"]["sensors"][0]["sensor_type"] == "realsense_d435"


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
    assert any(step["stage_id"] == "sync_quality" for step in payload["steps"])
    assert any(section["id"] == "sensors" for section in payload["sidebar"])


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


def test_web_rejects_run_roots_outside_configured_boundaries() -> None:
    response = app.test_client().get(
        "/ui/overview", query_string={"run_root": "/etc"}
    )

    assert response.status_code == 400
    assert "allowed root" in response.get_json()["output"]


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
            "options": {"object_folder": "/etc"},
        },
    )

    assert response.status_code == 400
    assert "object_folder" in response.get_json()["output"]


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
    assert payload["all_expected_connected"] is True
    assert family["connected_count"] == 1
    assert family["meets_expected"] is True
    assert family["diagnostics"] == []
    assert family["devices"] == [
        {
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
