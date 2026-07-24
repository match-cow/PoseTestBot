from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from posetestbot.config import (
    DEFAULT_CAPTURE_VELOCITY_M_S,
    DEFAULT_RECEIVER_PORT,
    DEFAULT_ROBOT_PORT,
    LAB_ROBOT_IP,
    LAB_ROBOT_RECEIVER_IP,
    MANUAL_TEST_COMMAND_VELOCITY_M_S,
    MAX_CAPTURE_COMMAND_VELOCITY_M_S,
    RobotProfile,
    robot_profile,
)
from posetestbot.robot import udp


def test_robot_profile_defaults_to_real_lab_robot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POSETESTBOT_ROBOT_IP", raising=False)
    monkeypatch.delenv("POSETESTBOT_RECEIVER_IP", raising=False)

    profile = robot_profile()

    assert profile.mode == "real"
    assert profile.robot_ip == LAB_ROBOT_IP
    assert profile.receiver_ip == LAB_ROBOT_RECEIVER_IP
    assert profile.command_port == DEFAULT_ROBOT_PORT
    assert profile.receiver_port == DEFAULT_RECEIVER_PORT
    assert profile.cartesian_velocity_m_s == DEFAULT_CAPTURE_VELOCITY_M_S


def test_robot_profile_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSETESTBOT_ROBOT_IP", "172.31.1.200")
    monkeypatch.setenv("POSETESTBOT_ROBOT_PORT", "30301")
    monkeypatch.setenv("POSETESTBOT_RECEIVER_IP", "172.31.1.201")
    monkeypatch.setenv("POSETESTBOT_RECEIVER_PORT", "18080")
    monkeypatch.setenv("POSETESTBOT_CAPTURE_VEL", "0.15")

    profile = robot_profile()

    assert profile.mode == "real"
    assert profile.robot_ip == "172.31.1.200"
    assert profile.command_port == 30301
    assert profile.receiver_ip == "172.31.1.201"
    assert profile.receiver_port == 18080
    assert profile.cartesian_velocity_m_s == 0.15


def test_robot_udp_command_shapes() -> None:
    assert udp.legacy_start_command(0.2) == {"start": 0.2}
    assert udp.legacy_start_command(0.2, receiver_ip=LAB_ROBOT_RECEIVER_IP) == {
        "start": 0.2,
        "receiver_ip": LAB_ROBOT_RECEIVER_IP,
    }
    assert udp.legacy_start_command(0.2, receiver_port=18080) == {
        "start": 0.2,
        "receiver_port": 18080,
    }
    assert udp.legacy_stop_command() == {"stop": True}
    assert udp.structured_start_command(0.2, "run-1") == {
        "schema_version": "robot_command.v1",
        "command": "start_capture",
        "cartesian_velocity_m_s": 0.2,
        "run_id": "run-1",
    }
    assert udp.structured_start_command(
        0.2,
        "run-1",
        receiver_ip=LAB_ROBOT_RECEIVER_IP,
        receiver_port=18080,
    ) == {
        "schema_version": "robot_command.v1",
        "command": "start_capture",
        "cartesian_velocity_m_s": 0.2,
        "run_id": "run-1",
        "receiver_ip": LAB_ROBOT_RECEIVER_IP,
        "receiver_port": 18080,
    }
    assert udp.structured_stop_command("pause_capture") == {
        "schema_version": "robot_command.v1",
        "command": "pause_capture",
    }


def test_send_start_uses_selected_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = []

    def fake_send(message, ip, port):
        sent.append((message, ip, port))

    monkeypatch.setattr(udp, "send_udp_json", fake_send)
    profile = RobotProfile(
        mode="real",
        robot_ip="127.0.0.1",
        command_port=30301,
        receiver_ip="127.0.0.1",
        receiver_port=18080,
        cartesian_velocity_m_s=0.12,
    )

    message = udp.send_start(profile, protocol="v1", run_id="run-1")

    assert message["command"] == "start_capture"
    assert (
        message["cartesian_velocity_m_s"]
        == MAX_CAPTURE_COMMAND_VELOCITY_M_S
    )
    assert message["receiver_ip"] == "127.0.0.1"
    assert message["receiver_port"] == 18080
    assert sent == [(message, "127.0.0.1", 30301)]


def test_send_start_omits_wildcard_receiver_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent = []

    def fake_send(message, ip, port):
        sent.append((message, ip, port))

    monkeypatch.setattr(udp, "send_udp_json", fake_send)
    profile = RobotProfile(
        mode="real",
        robot_ip="172.31.1.147",
        command_port=30300,
        receiver_ip="0.0.0.0",
        receiver_port=18080,
        cartesian_velocity_m_s=0.12,
    )

    message = udp.send_start(profile)

    assert message == {
        "start": MAX_CAPTURE_COMMAND_VELOCITY_M_S,
        "receiver_port": 18080,
    }
    assert sent == [(message, "172.31.1.147", 30300)]


def test_send_start_accepts_explicit_manual_test_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent = []

    def fake_send(message, ip, port):
        sent.append((message, ip, port))

    monkeypatch.setattr(udp, "send_udp_json", fake_send)
    profile = RobotProfile(
        mode="real",
        robot_ip="172.31.1.147",
        command_port=30300,
        receiver_ip="172.31.1.169",
        receiver_port=8080,
        cartesian_velocity_m_s=MANUAL_TEST_COMMAND_VELOCITY_M_S,
    )

    message = udp.send_start(
        profile,
        maximum_velocity_m_s=MANUAL_TEST_COMMAND_VELOCITY_M_S,
    )

    assert message["start"] == 0.1
    assert sent == [(message, "172.31.1.147", 30300)]


def test_direct_start_cli_requires_both_fresh_acknowledgements() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "start_iiwa.py",
            "--ip_robot",
            "192.0.2.10",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        env={**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache"},
    )

    assert result.returncode == 1
    assert "fresh --allow-real-robot and --allow-cameras" in result.stdout


@pytest.mark.parametrize(
    ("script", "arguments", "retired_flag"),
    [
        ("start_iiwa.py", ["--robot_mode", "real"], "--robot_mode"),
        ("stop_iiwa.py", ["--robot_mode", "real"], "--robot_mode"),
        (
            "scripts/pose_receiver_udp_json.py",
            ["/tmp/unused-pose-output", "--robot_mode", "real"],
            "--robot_mode",
        ),
        (
            "scripts/pose_receiver_udp_json.py",
            ["/tmp/unused-pose-output", "--test"],
            "--test",
        ),
        (
            "scripts/run_capture_execution_plan.py",
            ["/tmp/unused-capture-run", "--mode", "full"],
            "--mode",
        ),
        (
            "scripts/run_capture_execution_stage.py",
            ["/tmp/unused-capture-run", "--mode", "full"],
            "--mode",
        ),
    ],
)
def test_robot_and_execution_clis_reject_retired_flags(
    script: str,
    arguments: list[str],
    retired_flag: str,
) -> None:
    result = subprocess.run(
        ["uv", "run", "python", script, *arguments],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        env={**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache"},
    )

    assert result.returncode != 0
    assert retired_flag in result.stderr
    assert "unrecognized arguments" in result.stderr
