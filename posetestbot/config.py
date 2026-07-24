"""Real lab robot configuration defaults for PoseTestBot."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace


LAB_ROBOT_IP = "172.31.1.147"
LAB_ROBOT_RECEIVER_IP = "172.31.1.169"
LAB_NORMAL_NETWORK_IP = "10.145.8.132"

DEFAULT_ROBOT_PORT = 30300
DEFAULT_RECEIVER_PORT = 8080
DEFAULT_CAPTURE_VELOCITY_M_S = 0.01
MAX_CAPTURE_COMMAND_VELOCITY_M_S = 0.03
MANUAL_TEST_COMMAND_VELOCITY_M_S = 0.1


def bounded_capture_velocity_m_s(
    requested_velocity_m_s: float,
    *,
    maximum_velocity_m_s: float = MAX_CAPTURE_COMMAND_VELOCITY_M_S,
) -> float:
    """Return a finite positive command bounded by the selected command path.

    Acquisition uses the conservative 0.03 default. Explicit manual motion
    tests may supply their separately acknowledged command limit.
    """

    if isinstance(requested_velocity_m_s, bool):
        raise ValueError("Capture velocity must be a finite positive number")
    velocity = float(requested_velocity_m_s)
    if not math.isfinite(velocity) or velocity <= 0.0:
        raise ValueError("Capture velocity must be a finite positive number")
    maximum = float(maximum_velocity_m_s)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("Maximum capture velocity must be a finite positive number")
    return min(velocity, maximum)


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


def robot_profile() -> RobotProfile:
    """Return the real lab iiwa profile with environment overrides."""

    return RobotProfile(
        mode="real",
        robot_ip=os.getenv("POSETESTBOT_ROBOT_IP", LAB_ROBOT_IP),
        command_port=_env_int("POSETESTBOT_ROBOT_PORT", DEFAULT_ROBOT_PORT),
        receiver_ip=os.getenv("POSETESTBOT_RECEIVER_IP", LAB_ROBOT_RECEIVER_IP),
        receiver_port=_env_int("POSETESTBOT_RECEIVER_PORT", DEFAULT_RECEIVER_PORT),
        cartesian_velocity_m_s=_env_float(
            "POSETESTBOT_CAPTURE_VEL", DEFAULT_CAPTURE_VELOCITY_M_S
        ),
    )
