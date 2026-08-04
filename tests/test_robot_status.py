from __future__ import annotations

from posetestbot.config import (
    DEFAULT_CAPTURE_VELOCITY_M_S,
    LAB_NORMAL_NETWORK_IP,
    LAB_ROBOT_IP,
    MAX_CAPTURE_COMMAND_VELOCITY_M_S,
)
from posetestbot.robot import status as robot_status


def test_collect_robot_status_reports_fixed_real_profile(monkeypatch) -> None:
    for key in robot_status.ROBOT_ENV_VARS:
        monkeypatch.delenv(key, raising=False)

    status = robot_status.collect_robot_status(env={})

    assert status["schema_version"] == robot_status.SCHEMA_VERSION
    assert status["selected_profile"]["mode"] == "real"
    assert status["selected_profile"]["robot_ip"] == LAB_ROBOT_IP
    assert status["normal_network_ip"] == LAB_NORMAL_NETWORK_IP
    assert "fake_first" not in status
    assert "profiles" not in status
    assert status["env_overrides"] == {}
    assert "robot_command.v1" in status["command_protocols"]
    assert status["capture_velocity"] == {
        "requested_m_s": DEFAULT_CAPTURE_VELOCITY_M_S,
        "commanded_m_s": DEFAULT_CAPTURE_VELOCITY_M_S,
        "host_command_cap_m_s": MAX_CAPTURE_COMMAND_VELOCITY_M_S,
    }


def test_collect_robot_status_reports_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("POSETESTBOT_ROBOT_IP", "172.31.1.200")
    monkeypatch.setenv("POSETESTBOT_RECEIVER_IP", "172.31.1.201")

    status = robot_status.collect_robot_status()

    assert status["selected_profile"]["mode"] == "real"
    assert status["selected_profile"]["robot_ip"] == "172.31.1.200"
    assert "POSETESTBOT_ROBOT_MODE" not in status["env_overrides"]
    assert status["env_overrides"]["POSETESTBOT_RECEIVER_IP"] == "172.31.1.201"


def test_collect_robot_status_exposes_capture_command_cap(monkeypatch) -> None:
    monkeypatch.setenv("POSETESTBOT_CAPTURE_VEL", "0.2")

    status = robot_status.collect_robot_status()

    assert status["capture_velocity"]["requested_m_s"] == 0.2
    assert status["capture_velocity"]["commanded_m_s"] == 0.03
