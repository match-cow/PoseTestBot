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
    RobotProfile,
    robot_profile,
)

SCHEMA_VERSION = "robot_status.v1"

ROBOT_ENV_VARS = (
    "POSETESTBOT_ROBOT_MODE",
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
    selected_mode: str | None = None,
) -> dict:
    """Return the selected fake/real iiwa profile without commanding the robot."""

    env = env or os.environ
    selected = robot_profile(selected_mode)
    fake = robot_profile("fake")
    real = robot_profile("real")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selected_profile": robot_profile_dict(selected),
        "profiles": {
            "fake": robot_profile_dict(fake),
            "real": robot_profile_dict(real),
        },
        "fake_first": selected.mode == "fake",
        "real_robot": {
            "robot_ip": LAB_ROBOT_IP,
            "command_port": real.command_port,
            "receiver_ip": LAB_ROBOT_RECEIVER_IP,
            "receiver_port": real.receiver_port,
            "normal_network_ip": LAB_NORMAL_NETWORK_IP,
        },
        "env_overrides": robot_env_overrides(env),
        "command_protocols": ["legacy", "robot_command.v1"],
        "default_command_protocol": "legacy",
        "notes": [
            "Status is read-only and does not send UDP commands.",
            "Use fake mode for development and early testing.",
            "Select real mode intentionally with POSETESTBOT_ROBOT_MODE=real.",
        ],
    }
