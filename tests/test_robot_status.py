from __future__ import annotations

from posetestbot.config import LAB_NORMAL_NETWORK_IP, LAB_ROBOT_IP
from posetestbot.robot import status as robot_status


def test_collect_robot_status_defaults_to_fake(monkeypatch) -> None:
    for key in robot_status.ROBOT_ENV_VARS:
        monkeypatch.delenv(key, raising=False)

    status = robot_status.collect_robot_status(env={})

    assert status["schema_version"] == robot_status.SCHEMA_VERSION
    assert status["selected_profile"]["mode"] == "fake"
    assert status["fake_first"] is True
    assert status["profiles"]["real"]["robot_ip"] == LAB_ROBOT_IP
    assert status["real_robot"]["normal_network_ip"] == LAB_NORMAL_NETWORK_IP
    assert status["env_overrides"] == {}
    assert "robot_command.v1" in status["command_protocols"]


def test_collect_robot_status_reports_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("POSETESTBOT_ROBOT_MODE", "real")
    monkeypatch.setenv("POSETESTBOT_ROBOT_IP", "172.31.1.200")
    monkeypatch.setenv("POSETESTBOT_RECEIVER_IP", "172.31.1.201")

    status = robot_status.collect_robot_status()

    assert status["selected_profile"]["mode"] == "real"
    assert status["selected_profile"]["robot_ip"] == "172.31.1.200"
    assert status["fake_first"] is False
    assert status["env_overrides"]["POSETESTBOT_ROBOT_MODE"] == "real"
    assert status["env_overrides"]["POSETESTBOT_RECEIVER_IP"] == "172.31.1.201"
