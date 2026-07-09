"""UDP helpers for the current and rewrite iiwa command protocols."""

from __future__ import annotations

import json
import socket
from typing import Any

from posetestbot.config import RobotProfile


def _advertised_receiver_ip(receiver_ip: str) -> str | None:
    normalized = receiver_ip.strip()
    if normalized in {"", "0.0.0.0", "::"}:
        return None
    return normalized


def legacy_start_command(
    cartesian_velocity_m_s: float,
    *,
    receiver_ip: str | None = None,
    receiver_port: int | None = None,
) -> dict[str, float | int | str]:
    command: dict[str, float | int | str] = {"start": cartesian_velocity_m_s}
    if receiver_ip:
        command["receiver_ip"] = receiver_ip
    if receiver_port is not None:
        command["receiver_port"] = receiver_port
    return command


def structured_start_command(
    cartesian_velocity_m_s: float,
    run_id: str | None = None,
    *,
    receiver_ip: str | None = None,
    receiver_port: int | None = None,
) -> dict[str, Any]:
    command: dict[str, Any] = {
        "schema_version": "robot_command.v1",
        "command": "start_capture",
        "cartesian_velocity_m_s": cartesian_velocity_m_s,
    }
    if run_id:
        command["run_id"] = run_id
    if receiver_ip:
        command["receiver_ip"] = receiver_ip
    if receiver_port is not None:
        command["receiver_port"] = receiver_port
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
    receiver_ip = _advertised_receiver_ip(profile.receiver_ip)
    if protocol == "v1":
        message = structured_start_command(
            profile.cartesian_velocity_m_s,
            run_id,
            receiver_ip=receiver_ip,
            receiver_port=profile.receiver_port,
        )
    elif protocol == "legacy":
        message = legacy_start_command(
            profile.cartesian_velocity_m_s,
            receiver_ip=receiver_ip,
            receiver_port=profile.receiver_port,
        )
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
