from __future__ import annotations

import json
import subprocess
from pathlib import Path

from posetestbot.io.artifacts import (
    CAPTURE_REHEARSAL_REPORT,
    DATASET_MANIFEST,
    RAW_ROBOT_EE_POSES,
)
from posetestbot.pipeline.capture_rehearsal import (
    build_capture_rehearsal_commands,
    build_capture_rehearsal_plan,
    run_capture_rehearsal,
)
from posetestbot.pipeline.run_config import create_run_config, sensor_config_from_token


class FakeControllerProcess:
    def __init__(self, command: list[str]):
        self.command = command
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def communicate(self, timeout=None):
        self.returncode = 0
        return ("Fake iiwa controller listening\nMock motion finished\n", None)

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.returncode = -9


def test_capture_rehearsal_builds_fake_uv_commands(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
    ).to_dict()

    commands = build_capture_rehearsal_commands(
        config,
        duration_s=0.1,
        sample_ms=20.0,
        robot_port=30301,
        receiver_port=8081,
    ).to_dict()

    assert commands["fake_controller"][:4] == [
        "uv",
        "run",
        "python",
        "iiwa/fake_iiwa_controller.py",
    ]
    assert "--duration" in commands["fake_controller"]
    assert "0.1" in commands["fake_controller"]
    assert "--robot-port" in commands["fake_controller"]
    assert "30301" in commands["fake_controller"]
    assert commands["pose_receiver"][:4] == [
        "uv",
        "run",
        "python",
        "scripts/pose_receiver_udp_json.py",
    ]
    assert "--port_robot" in commands["pose_receiver"]
    assert "30301" in commands["pose_receiver"]

    plan = build_capture_rehearsal_plan(
        config,
        duration_s=0.1,
        sample_ms=20.0,
        robot_port=30301,
        receiver_port=8081,
    ).to_dict()
    assert plan["schema_version"] == "capture_plan.v1"
    assert [command["role"] for command in plan["commands"]] == [
        "robot_controller",
        "sensor_capture",
        "robot_pose_receiver",
    ]
    assert plan["commands"][0]["command"] == commands["fake_controller"]
    assert plan["commands"][-1]["command"] == commands["pose_receiver"]


def test_capture_rehearsal_writes_report_and_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "run"
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
    ).to_dict()
    controller_processes: list[FakeControllerProcess] = []
    receiver_commands: list[list[str]] = []

    def fake_popen(command, **kwargs):
        process = FakeControllerProcess(list(command))
        controller_processes.append(process)
        return process

    def fake_run(command, **kwargs):
        receiver_commands.append(list(command))
        (run_root / RAW_ROBOT_EE_POSES).write_text(
            json.dumps(
                {
                    "0": {"motion": "circ_far", "pose": {"X": 1}},
                    "1": {"motion": "zoom", "pose": {"X": 2}},
                }
            )
        )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="Listening on 127.0.0.1:8081\nReceived poses: 2\n",
            stderr="",
        )

    monkeypatch.setattr("posetestbot.pipeline.capture_rehearsal.subprocess.Popen", fake_popen)
    monkeypatch.setattr("posetestbot.pipeline.capture_rehearsal.subprocess.run", fake_run)
    monkeypatch.setattr("posetestbot.pipeline.capture_rehearsal.time.sleep", lambda _: None)

    report_path, report = run_capture_rehearsal(
        config,
        duration_s=0.1,
        sample_ms=20.0,
        robot_port=30301,
        receiver_port=8081,
    )

    assert report_path == run_root / CAPTURE_REHEARSAL_REPORT
    assert report["schema_version"] == "capture_rehearsal_report.v1"
    assert report["status"] == "succeeded"
    assert report["raw_pose_count"] == 2
    assert report["capture_plan"]["schema_version"] == "capture_plan.v1"
    assert report["capture_plan"]["commands"][0]["role"] == "robot_controller"
    assert report["processes"]["fake_controller_returncode"] == 0
    assert report["processes"]["pose_receiver_returncode"] == 0
    assert controller_processes[0].command[:4] == [
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

    report_on_disk = json.loads(report_path.read_text())
    assert report_on_disk["raw_pose_count"] == 2
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(stage for stage in manifest["stages"] if stage["name"] == "capture_rehearsal")
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][CAPTURE_REHEARSAL_REPORT] == CAPTURE_REHEARSAL_REPORT
    assert stage["artifacts"][RAW_ROBOT_EE_POSES] == RAW_ROBOT_EE_POSES
    assert manifest["robot_profile"]["mode"] == "fake"
    assert manifest["capture_config"]["fps"] == 6
