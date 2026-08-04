"""JSON-friendly iiwa robot profile status snapshots."""

from __future__ import annotations

import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Mapping

from posetestbot.config import (
    LAB_NORMAL_NETWORK_IP,
    LAB_ROBOT_IP,
    LAB_ROBOT_RECEIVER_IP,
    MAX_CAPTURE_COMMAND_VELOCITY_M_S,
    RobotProfile,
    bounded_capture_velocity_m_s,
    robot_profile,
)

SCHEMA_VERSION = "robot_status.v2"

ROBOT_ENV_VARS = (
    "POSETESTBOT_ROBOT_IP",
    "POSETESTBOT_ROBOT_PORT",
    "POSETESTBOT_RECEIVER_IP",
    "POSETESTBOT_RECEIVER_PORT",
    "POSETESTBOT_CAPTURE_VEL",
)


def robot_profile_dict(profile: RobotProfile) -> dict:
    return asdict(profile)


def robot_env_overrides(env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = env or os.environ
    return {
        key: env[key]
        for key in ROBOT_ENV_VARS
        if key in env and str(env[key]).strip() != ""
    }


def collect_robot_status(
    *,
    env: Mapping[str, str] | None = None,
) -> dict:
    """Return the fixed real iiwa profile without commanding the robot."""

    env = env or os.environ
    selected = robot_profile()
    commanded_velocity_m_s = bounded_capture_velocity_m_s(
        selected.cartesian_velocity_m_s
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selected_profile": robot_profile_dict(selected),
        "capture_velocity": {
            "requested_m_s": selected.cartesian_velocity_m_s,
            "commanded_m_s": commanded_velocity_m_s,
            "host_command_cap_m_s": MAX_CAPTURE_COMMAND_VELOCITY_M_S,
        },
        "normal_network_ip": LAB_NORMAL_NETWORK_IP,
        "env_overrides": robot_env_overrides(env),
        "command_protocols": ["legacy", "robot_command.v1"],
        "default_command_protocol": "legacy",
        "notes": [
            "Status is read-only and does not send UDP commands.",
            f"The lab robot is {LAB_ROBOT_IP}; the receiver is {LAB_ROBOT_RECEIVER_IP}.",
            "Environment overrides change addresses, ports, or capture velocity only.",
            "The host bounds every transmitted capture START value independently.",
        ],
    }
