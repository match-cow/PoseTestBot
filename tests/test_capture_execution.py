from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path

import pytest

from posetestbot.io.artifacts import (
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


class FakeSignalProcess(FakePersistentProcess):
    def wait(self, timeout=None):
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("SIGTERM handler should interrupt receiver wait")


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


def test_capture_execution_plan_selects_full_capture_roles(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
    )
    write_run_config(run_root, config)

    plan = build_capture_execution_plan(
        run_root,
        allow_cameras=True,
        allow_real_robot=True,
        collect_sensors=fake_sensor_status,
    )

    assert plan["schema_version"] == "capture_execution_plan.v1"
    assert plan["status"] == "ok"
    assert plan["mode"] == "full"
    assert plan["ready_to_execute"] is True
    assert plan["preflight_status"] == "ok"
    assert plan["selected_roles"] == ["sensor_capture", "robot_pose_receiver"]
    assert [command["role"] for command in plan["selected_commands"]] == [
        "sensor_capture",
        "robot_pose_receiver",
    ]
    assert plan["skipped_commands"] == []
    assert plan["selected_resources"] == ["camera", "disk_io", "robot_command"]
    gates = {gate["name"]: gate for gate in plan["gates"]}
    assert gates["camera_permission"]["status"] == "ok"
    assert gates["capture_plan_preflight"]["status"] == "ok"


def test_capture_execution_plan_blocks_until_both_permissions_are_allowed(
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
        include_sensor_status=False,
    )

    assert plan["status"] == "error"
    assert plan["ready_to_execute"] is False
    assert [command["role"] for command in plan["selected_commands"]] == [
        "sensor_capture",
        "robot_pose_receiver",
    ]
    gates = {gate["name"]: gate for gate in plan["gates"]}
    assert gates["camera_permission"]["status"] == "error"
    assert gates["real_robot_permission"]["status"] == "error"


@pytest.mark.parametrize(
    ("allow_cameras", "allow_real_robot", "blocked_gate"),
    [
        (False, True, "camera_permission"),
        (True, False, "real_robot_permission"),
    ],
)
def test_capture_execution_plan_blocks_when_either_permission_is_absent(
    tmp_path: Path,
    allow_cameras: bool,
    allow_real_robot: bool,
    blocked_gate: str,
) -> None:
    run_root = tmp_path / blocked_gate
    write_run_config(
        run_root,
        create_run_config(
            run_root=run_root,
            sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
        ),
    )

    plan = build_capture_execution_plan(
        run_root,
        allow_cameras=allow_cameras,
        allow_real_robot=allow_real_robot,
        collect_sensors=fake_sensor_status,
    )

    gates = {gate["name"]: gate for gate in plan["gates"]}
    assert plan["ready_to_execute"] is False
    assert gates[blocked_gate]["status"] == "error"


def test_capture_execution_plan_writes_manifest_artifact(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
    )
    write_run_config(run_root, config)

    path, plan = write_capture_execution_plan_with_manifest(
        run_root,
        allow_cameras=True,
        allow_real_robot=True,
        collect_sensors=fake_sensor_status,
    )

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
            "--allow-cameras",
            "--allow-real-robot",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert f"Wrote {run_root / CAPTURE_EXECUTION_PLAN}" in result.stdout
    assert "Capture execution plan: warning (full)" in result.stdout
    data = json.loads((run_root / CAPTURE_EXECUTION_PLAN).read_text())
    assert data["selected_roles"] == ["sensor_capture", "robot_pose_receiver"]


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
        if any(item.endswith("pose_receiver_udp_json.py") for item in command):
            receiver_commands.append(list(command))
            (run_root / RAW_ROBOT_EE_POSES).write_text(
                json.dumps({"0": {"motion": "circ_far", "pose": {"X": 1}}})
            )
            return FakeBackgroundProcess(list(command), kwargs["stdout"])
        background_commands.append(list(command))
        if any(item.endswith("capture_realsense_720p.py") for item in command):
            return FakePersistentProcess(list(command), kwargs["stdout"])
        return FakeBackgroundProcess(list(command), kwargs["stdout"])

    def fake_terminate(process, *, timeout_s):
        terminated_commands.append(list(process.command))
        process.returncode = -15
        process.log_file.write("fake supervisor stopped process\n")

    monkeypatch.setattr("posetestbot.pipeline.capture_execution.subprocess.Popen", fake_popen)
    monkeypatch.setattr("posetestbot.pipeline.capture_execution.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution._terminate_process_group",
        fake_terminate,
    )

    report_path, report = run_capture_execution(
        run_root,
        allow_cameras=True,
        allow_real_robot=True,
        collect_sensors=fake_sensor_status,
        timeout_s=5,
    )

    assert report_path == run_root / CAPTURE_EXECUTION_REPORT
    assert report["status"] == "succeeded"
    assert report["mode"] == "full"
    assert report["capture_execution_plan"]["selected_roles"] == [
        "sensor_capture",
        "robot_pose_receiver",
    ]
    processes = {process["role"]: process for process in report["processes"]}
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


def test_capture_execution_sigterm_cancels_every_spawned_process(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "run-canceled"
    write_run_config(
        run_root,
        create_run_config(
            run_root=run_root,
            sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
        ),
    )
    spawned: list[FakePersistentProcess] = []
    terminated: list[FakePersistentProcess] = []

    def fake_popen(command, **kwargs):
        process: FakePersistentProcess
        if any(item.endswith("pose_receiver_udp_json.py") for item in command):
            process = FakeSignalProcess(list(command), kwargs["stdout"])
        else:
            process = FakePersistentProcess(list(command), kwargs["stdout"])
        spawned.append(process)
        return process

    def fake_terminate(process, *, timeout_s):
        del timeout_s
        process.returncode = -signal.SIGTERM
        terminated.append(process)

    monkeypatch.setattr("posetestbot.pipeline.capture_execution.subprocess.Popen", fake_popen)
    monkeypatch.setattr("posetestbot.pipeline.capture_execution.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution._terminate_process_group",
        fake_terminate,
    )

    with pytest.raises(RuntimeError, match="canceled by SIGTERM"):
        run_capture_execution(
            run_root,
            allow_cameras=True,
            allow_real_robot=True,
            collect_sensors=fake_sensor_status,
            timeout_s=5,
        )

    report = json.loads((run_root / CAPTURE_EXECUTION_REPORT).read_text())
    assert report["status"] == "canceled"
    assert "SIGTERM" in report["message"]
    assert len(spawned) == 2
    assert terminated == spawned
    assert all(process["status"] == "terminated" for process in report["processes"])
    persisted = json.loads((run_root / CAPTURE_EXECUTION_STATUS).read_text())
    assert persisted["status"] == "canceled"
    assert persisted["active_process_count"] == 0
