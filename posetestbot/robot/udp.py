"""UDP helpers for the current and rewrite iiwa command protocols."""

from __future__ import annotations

import json
import socket
from typing import Any

from posetestbot.config import RobotProfile


def legacy_start_command(cartesian_velocity_m_s: float) -> dict[str, float]:
    return {"start": cartesian_velocity_m_s}


def structured_start_command(
    cartesian_velocity_m_s: float, run_id: str | None = None
) -> dict[str, Any]:
    command: dict[str, Any] = {
        "schema_version": "robot_command.v1",
        "command": "start_capture",
        "cartesian_velocity_m_s": cartesian_velocity_m_s,
    }
    if run_id:
        command["run_id"] = run_id
    return command


def legacy_stop_command() -> dict[str, bool]:
    return {"stop": True}


def structured_stop_command(intent: str = "stop_after_current_motion") -> dict[str, str]:
    return {
        "schema_version": "robot_command.v1",
        "command": intent,
    }


def send_udp_json(message: dict[str, Any], ip: str, port: int) -> None:
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(payload, (ip, port))


def send_start(
    profile: RobotProfile,
    *,
    protocol: str = "legacy",
    run_id: str | None = None,
) -> dict[str, Any]:
    if protocol == "v1":
        message = structured_start_command(profile.cartesian_velocity_m_s, run_id)
    elif protocol == "legacy":
        message = legacy_start_command(profile.cartesian_velocity_m_s)
    else:
        raise ValueError("protocol must be 'legacy' or 'v1'")

    send_udp_json(message, profile.robot_ip, profile.command_port)
    return message


def send_stop(
    profile: RobotProfile,
    *,
    protocol: str = "legacy",
    intent: str = "stop_after_current_motion",
) -> dict[str, Any]:
    if protocol == "v1":
        message = structured_stop_command(intent)
    elif protocol == "legacy":
        message = legacy_stop_command()
    else:
        raise ValueError("protocol must be 'legacy' or 'v1'")

    send_udp_json(message, profile.robot_ip, profile.command_port)
    return message

