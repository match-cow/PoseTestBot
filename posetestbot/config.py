"""Local runtime configuration defaults for PoseTestBot.

The rewrite is fake-iiwa-first while preserving the real lab robot profile.
Environment variables can override these defaults without editing scripts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace


FAKE_ROBOT_IP = "127.0.0.1"
FAKE_RECEIVER_IP = "127.0.0.1"

LAB_ROBOT_IP = "172.31.1.147"
LAB_ROBOT_RECEIVER_IP = "172.31.1.169"
LAB_NORMAL_NETWORK_IP = "10.145.8.132"

DEFAULT_ROBOT_PORT = 30300
DEFAULT_RECEIVER_PORT = 8080
DEFAULT_CAPTURE_VELOCITY_M_S = 0.2


@dataclass(frozen=True)
class RobotProfile:
    """Network and motion defaults for one iiwa controller target."""

    mode: str
    robot_ip: str
    command_port: int
    receiver_ip: str
    receiver_port: int
    cartesian_velocity_m_s: float

    def with_overrides(
        self,
        *,
        robot_ip: str | None = None,
        command_port: int | None = None,
        receiver_ip: str | None = None,
        receiver_port: int | None = None,
        cartesian_velocity_m_s: float | None = None,
    ) -> "RobotProfile":
        return replace(
            self,
            robot_ip=robot_ip if robot_ip is not None else self.robot_ip,
            command_port=(
                command_port if command_port is not None else self.command_port
            ),
            receiver_ip=receiver_ip if receiver_ip is not None else self.receiver_ip,
            receiver_port=(
                receiver_port if receiver_port is not None else self.receiver_port
            ),
            cartesian_velocity_m_s=(
                cartesian_velocity_m_s
                if cartesian_velocity_m_s is not None
                else self.cartesian_velocity_m_s
            ),
        )


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _normalise_mode(mode: str | None) -> str:
    resolved = mode or os.getenv("POSETESTBOT_ROBOT_MODE", "fake")
    resolved = resolved.strip().lower()
    if resolved not in {"fake", "real"}:
        raise ValueError("robot mode must be 'fake' or 'real'")
    return resolved


def robot_profile(mode: str | None = None) -> RobotProfile:
    """Return the configured iiwa profile.

    Defaults are intentionally fake-first. Set ``POSETESTBOT_ROBOT_MODE=real`` or
    pass ``mode="real"`` to target the lab robot at 172.31.1.147.
    """

    resolved_mode = _normalise_mode(mode)

    if resolved_mode == "real":
        robot_ip = LAB_ROBOT_IP
        receiver_ip = LAB_ROBOT_RECEIVER_IP
    else:
        robot_ip = FAKE_ROBOT_IP
        receiver_ip = FAKE_RECEIVER_IP

    return RobotProfile(
        mode=resolved_mode,
        robot_ip=os.getenv("POSETESTBOT_ROBOT_IP", robot_ip),
        command_port=_env_int("POSETESTBOT_ROBOT_PORT", DEFAULT_ROBOT_PORT),
        receiver_ip=os.getenv("POSETESTBOT_RECEIVER_IP", receiver_ip),
        receiver_port=_env_int("POSETESTBOT_RECEIVER_PORT", DEFAULT_RECEIVER_PORT),
        cartesian_velocity_m_s=_env_float(
            "POSETESTBOT_CAPTURE_VEL", DEFAULT_CAPTURE_VELOCITY_M_S
        ),
    )

