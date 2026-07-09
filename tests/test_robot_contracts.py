from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from posetestbot.config import (
    DEFAULT_RECEIVER_PORT,
    DEFAULT_ROBOT_PORT,
    LAB_ROBOT_IP,
    LAB_ROBOT_RECEIVER_IP,
    RobotProfile,
    robot_profile,
)
from posetestbot.robot import udp


def load_fake_controller_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "iiwa" / "fake_iiwa_controller.py"
    )
    spec = importlib.util.spec_from_file_location("fake_iiwa_controller", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_robot_profile_defaults_to_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POSETESTBOT_ROBOT_MODE", raising=False)
    monkeypatch.delenv("POSETESTBOT_ROBOT_IP", raising=False)
    monkeypatch.delenv("POSETESTBOT_RECEIVER_IP", raising=False)

    profile = robot_profile()

    assert profile.mode == "fake"
    assert profile.robot_ip == "127.0.0.1"
    assert profile.receiver_ip == "127.0.0.1"
    assert profile.command_port == DEFAULT_ROBOT_PORT
    assert profile.receiver_port == DEFAULT_RECEIVER_PORT


def test_robot_profile_real_lab_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POSETESTBOT_ROBOT_IP", raising=False)
    monkeypatch.delenv("POSETESTBOT_RECEIVER_IP", raising=False)

    profile = robot_profile("real")

    assert profile.mode == "real"
    assert profile.robot_ip == LAB_ROBOT_IP
    assert profile.receiver_ip == LAB_ROBOT_RECEIVER_IP


def test_robot_profile_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSETESTBOT_ROBOT_MODE", "real")
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
        mode="fake",
        robot_ip="127.0.0.1",
        command_port=30301,
        receiver_ip="127.0.0.1",
        receiver_port=18080,
        cartesian_velocity_m_s=0.12,
    )

    message = udp.send_start(profile, protocol="v1", run_id="run-1")

    assert message["command"] == "start_capture"
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

    assert message == {"start": 0.12, "receiver_port": 18080}
    assert sent == [(message, "172.31.1.147", 30300)]


def test_fake_controller_accepts_legacy_and_v1_commands() -> None:
    fake_controller = load_fake_controller_module()

    assert fake_controller.start_value_from_command({"start": 0.2}) == 0.2
    assert (
        fake_controller.start_value_from_command(
            {
                "schema_version": "robot_command.v1",
                "command": "start_capture",
                "cartesian_velocity_m_s": 0.12,
            }
        )
        == 0.12
    )
    assert fake_controller.is_stop_command({"stop": True})
    assert fake_controller.is_stop_command(
        {
            "schema_version": "robot_command.v1",
            "command": "stop_after_current_motion",
        }
    )
    assert fake_controller.start_value_from_command({"command": "noop"}) is None
