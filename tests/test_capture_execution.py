from __future__ import annotations

import json
import subprocess
from pathlib import Path

from posetestbot.io.artifacts import (
    CAPTURE_EXECUTION_LOGS_DIR,
    CAPTURE_EXECUTION_PLAN,
    CAPTURE_EXECUTION_REPORT,
    CAPTURE_EXECUTION_STATUS,
    CAPTURE_PLAN,
    DATASET_MANIFEST,
    RAW_ROBOT_EE_POSES,
)
from posetestbot.pipeline.capture_execution import (
    build_capture_execution_plan,
    run_capture_execution,
    write_capture_execution_plan_with_manifest,
)
from posetestbot.pipeline.run_config import (
    create_run_config,
    sensor_config_from_token,
    write_run_config,
)


class FakeBackgroundProcess:
    def __init__(self, command: list[str], log_file):
        self.command = command
        self.returncode = None
        self.log_file = log_file
        self.pid = 12345
        self.log_file.write("fake background started\n")

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        self.log_file.write("fake background finished\n")
        return 0


class FakePersistentProcess:
    def __init__(self, command: list[str], log_file):
        self.command = command
        self.returncode = None
        self.log_file = log_file
        self.pid = 23456
        self.log_file.write("fake persistent background started\n")

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        raise subprocess.TimeoutExpired(self.command, timeout)


def fake_sensor_status() -> dict:
    return {
        "schema_version": "sensor_status.v1",
        "families": [
            {
                "sensor_type": "realsense_d435",
                "sdk_available": True,
                "devices": [
                    {
                        "device_id": "123",
                        "display_name": "RealSense 123",
                        "connected": True,
                    }
                ],
                "error": None,
            }
        ],
        "overall_status": "ok",
        "checks": [],
    }


def test_capture_execution_plan_selects_pose_only_fake_roles(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
    )
    write_run_config(run_root, config)

    plan = build_capture_execution_plan(run_root)

    assert plan["schema_version"] == "capture_execution_plan.v1"
    assert plan["status"] == "ok"
    assert plan["mode"] == "pose_only_fake"
    assert plan["ready_to_execute"] is True
    assert plan["preflight_status"] == "warning"
    assert plan["selected_roles"] == ["robot_controller", "robot_pose_receiver"]
    assert [command["role"] for command in plan["selected_commands"]] == [
        "robot_controller",
        "robot_pose_receiver",
    ]
    assert [command["role"] for command in plan["skipped_commands"]] == [
        "sensor_capture"
    ]
    assert plan["skipped_commands"][0]["skip_reason"] == (
        "camera_hardware_gated_for_pose_only_fake"
    )
    assert plan["selected_resources"] == ["disk_io", "robot_command"]
    gates = {gate["name"]: gate for gate in plan["gates"]}
    assert gates["camera_permission"]["status"] == "ok"
    assert gates["capture_plan_preflight"]["status"] == "ok"


def test_capture_execution_plan_blocks_full_mode_until_cameras_allowed(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
    )
    write_run_config(run_root, config)

    plan = build_capture_execution_plan(
        run_root,
        mode="full",
        include_sensor_status=False,
    )

    assert plan["status"] == "error"
    assert plan["ready_to_execute"] is False
    assert plan["selected_commands"] == []
    assert len(plan["skipped_commands"]) == 3
    gates = {gate["name"]: gate for gate in plan["gates"]}
    assert gates["camera_permission"]["status"] == "error"
    assert gates["command_selection"]["status"] == "error"


def test_capture_execution_plan_writes_manifest_artifact(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
    )
    write_run_config(run_root, config)

    path, plan = write_capture_execution_plan_with_manifest(run_root)

    assert path == run_root / CAPTURE_EXECUTION_PLAN
    assert plan["status"] == "ok"
    assert (run_root / CAPTURE_PLAN).is_file()
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage for stage in manifest["stages"] if stage["name"] == "capture_execution_plan"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][CAPTURE_EXECUTION_PLAN] == CAPTURE_EXECUTION_PLAN
    assert stage["artifacts"][CAPTURE_PLAN] == CAPTURE_PLAN


def test_capture_execution_plan_cli_writes_artifact(tmp_path: Path) -> None:
    run_root = tmp_path / "run-cli"
    repo_root = Path(__file__).resolve().parents[1]
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
    )
    write_run_config(run_root, config)

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_capture_execution_plan.py",
            run_root.as_posix(),
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert f"Wrote {run_root / CAPTURE_EXECUTION_PLAN}" in result.stdout
    assert "Capture execution plan: ok (pose_only_fake)" in result.stdout
    data = json.loads((run_root / CAPTURE_EXECUTION_PLAN).read_text())
    assert data["selected_roles"] == ["robot_controller", "robot_pose_receiver"]


def test_capture_execution_runs_pose_only_fake_plan_with_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "run-execute"
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
    )
    write_run_config(run_root, config)
    background_commands: list[list[str]] = []
    receiver_commands: list[list[str]] = []

    def fake_popen(command, **kwargs):
        background_commands.append(list(command))
        return FakeBackgroundProcess(list(command), kwargs["stdout"])

    def fake_run(command, **kwargs):
        receiver_commands.append(list(command))
        stdout = kwargs["stdout"]
        stdout.write("pose receiver started\n")
        (run_root / RAW_ROBOT_EE_POSES).write_text(
            json.dumps(
                {
                    "0": {"motion": "circ_far", "pose": {"X": 1}},
                    "1": {"motion": "zoom", "pose": {"X": 2}},
                }
            )
        )
        stdout.write("pose receiver finished\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("posetestbot.pipeline.capture_execution.subprocess.Popen", fake_popen)
    monkeypatch.setattr("posetestbot.pipeline.capture_execution.subprocess.run", fake_run)
    monkeypatch.setattr("posetestbot.pipeline.capture_execution.time.sleep", lambda _: None)

    report_path, report = run_capture_execution(
        run_root,
        mode="pose_only_fake",
        timeout_s=5,
    )

    assert report_path == run_root / CAPTURE_EXECUTION_REPORT
    status_data = json.loads((run_root / CAPTURE_EXECUTION_STATUS).read_text())
    assert status_data["schema_version"] == "capture_execution_status.v1"
    assert status_data["status"] == "succeeded"
    assert status_data["active_process_count"] == 0
    assert status_data["process_count"] == 2
    assert status_data["raw_pose_count"] == 2
    assert status_data["selected_roles"] == [
        "robot_controller",
        "robot_pose_receiver",
    ]
    status_processes = {process["role"]: process for process in status_data["processes"]}
    assert status_processes["robot_controller"]["pid"] == 12345
    assert status_processes["robot_controller"]["started_at"]
    assert status_processes["robot_controller"]["ended_at"]
    assert status_processes["robot_controller"]["elapsed_s"] >= 0
    assert report["schema_version"] == "capture_execution_report.v1"
    assert report["status"] == "succeeded"
    assert report["raw_pose_count"] == 2
    assert report["mode"] == "pose_only_fake"
    assert report["capture_execution_plan"]["selected_roles"] == [
        "robot_controller",
        "robot_pose_receiver",
    ]
    assert [process["role"] for process in report["processes"]] == [
        "robot_controller",
        "robot_pose_receiver",
    ]
    assert report["processes"][0]["returncode"] == 0
    assert report["processes"][1]["returncode"] == 0
    assert report["processes"][0]["pid"] == 12345
    assert report["processes"][0]["started_at"]
    assert report["processes"][0]["ended_at"]
    assert report["processes"][0]["elapsed_s"] >= 0
    assert report["processes"][1]["pid"] is None
    assert report["processes"][1]["started_at"]
    assert report["processes"][1]["ended_at"]
    assert report["processes"][1]["elapsed_s"] >= 0
    assert background_commands[0][:4] == [
        "uv",
        "run",
        "python",
        "iiwa/fake_iiwa_controller.py",
    ]
    assert receiver_commands[0][:4] == [
        "uv",
        "run",
        "python",
        "scripts/pose_receiver_udp_json.py",
    ]
    assert (run_root / CAPTURE_EXECUTION_LOGS_DIR).is_dir()
    assert any((run_root / CAPTURE_EXECUTION_LOGS_DIR).glob("*fake_iiwa*.log"))

    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage for stage in manifest["stages"] if stage["name"] == "capture_execution"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][CAPTURE_EXECUTION_REPORT] == CAPTURE_EXECUTION_REPORT
    assert stage["artifacts"][CAPTURE_EXECUTION_PLAN] == CAPTURE_EXECUTION_PLAN
    assert stage["artifacts"][CAPTURE_EXECUTION_STATUS] == CAPTURE_EXECUTION_STATUS
    assert stage["artifacts"][CAPTURE_EXECUTION_LOGS_DIR] == CAPTURE_EXECUTION_LOGS_DIR
    assert stage["artifacts"][RAW_ROBOT_EE_POSES] == RAW_ROBOT_EE_POSES


def test_capture_execution_full_mode_stops_sensor_process_after_receiver(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "run-full-execute"
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
    )
    write_run_config(run_root, config)
    background_commands: list[list[str]] = []
    receiver_commands: list[list[str]] = []
    terminated_commands: list[list[str]] = []

    def fake_popen(command, **kwargs):
        background_commands.append(list(command))
        if any(item.endswith("capture_realsense_720p.py") for item in command):
            return FakePersistentProcess(list(command), kwargs["stdout"])
        return FakeBackgroundProcess(list(command), kwargs["stdout"])

    def fake_run(command, **kwargs):
        receiver_commands.append(list(command))
        stdout = kwargs["stdout"]
        stdout.write("pose receiver started\n")
        (run_root / RAW_ROBOT_EE_POSES).write_text(
            json.dumps({"0": {"motion": "circ_far", "pose": {"X": 1}}})
        )
        stdout.write("pose receiver finished\n")
        return subprocess.CompletedProcess(command, 0)

    def fake_terminate(process, *, timeout_s):
        terminated_commands.append(list(process.command))
        process.returncode = -15
        process.log_file.write("fake supervisor stopped process\n")

    monkeypatch.setattr("posetestbot.pipeline.capture_execution.subprocess.Popen", fake_popen)
    monkeypatch.setattr("posetestbot.pipeline.capture_execution.subprocess.run", fake_run)
    monkeypatch.setattr("posetestbot.pipeline.capture_execution.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution._terminate_process_group",
        fake_terminate,
    )

    report_path, report = run_capture_execution(
        run_root,
        mode="full",
        allow_cameras=True,
        collect_sensors=fake_sensor_status,
        timeout_s=5,
    )

    assert report_path == run_root / CAPTURE_EXECUTION_REPORT
    assert report["status"] == "succeeded"
    assert report["mode"] == "full"
    assert report["capture_execution_plan"]["selected_roles"] == [
        "robot_controller",
        "sensor_capture",
        "robot_pose_receiver",
    ]
    processes = {process["role"]: process for process in report["processes"]}
    assert processes["robot_controller"]["status"] == "succeeded"
    assert processes["sensor_capture"]["status"] == "stopped"
    assert processes["sensor_capture"]["pid"] == 23456
    assert processes["sensor_capture"]["started_at"]
    assert processes["sensor_capture"]["ended_at"]
    assert processes["sensor_capture"]["elapsed_s"] >= 0
    assert processes["sensor_capture"]["termination_reason"] == (
        "stopped_after_receiver_exit"
    )
    assert processes["robot_pose_receiver"]["termination_reason"] == (
        "receiver_completed"
    )
    assert terminated_commands
    assert any(
        any(item.endswith("capture_realsense_720p.py") for item in command)
        for command in terminated_commands
    )
    assert any(
        any(item.endswith("capture_realsense_720p.py") for item in command)
        for command in background_commands
    )
    assert receiver_commands[0][:4] == [
        "uv",
        "run",
        "python",
        "scripts/pose_receiver_udp_json.py",
    ]
