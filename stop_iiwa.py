#!/usr/bin/env python3

import argparse
import sys

from posetestbot.config import robot_profile
from posetestbot.robot.udp import send_stop


def send_stop_message(
    *,
    robot_mode: str,
    ip_robot: str | None,
    port_robot: int | None,
    protocol: str,
    intent: str,
) -> bool:
    """Send a stop-like control message to the configured iiwa controller."""

    profile = robot_profile(robot_mode).with_overrides(
        robot_ip=ip_robot,
        command_port=port_robot,
    )

    try:
        stop_message = send_stop(profile, protocol=protocol, intent=intent)
        print(f"Sent stop message to {profile.robot_ip}:{profile.command_port}")
        print(f"Message: {stop_message}")
        return True
    except OSError as exc:
        print(f"Socket error: {exc}")
        return False
    except Exception as exc:
        print(f"Error sending stop message: {exc}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Send stop message to robot via UDP")
    parser.add_argument(
        "--robot_mode",
        choices=("fake", "real"),
        default=None,
        help="Robot target profile. Defaults to POSETESTBOT_ROBOT_MODE or fake.",
    )
    parser.add_argument(
        "--ip_robot",
        type=str,
        default=None,
        help="Override robot IP address from the selected profile.",
    )
    parser.add_argument(
        "--port_robot",
        type=int,
        default=None,
        help="Override robot UDP command port from the selected profile.",
    )
    parser.add_argument(
        "--protocol",
        choices=("legacy", "v1"),
        default="legacy",
        help="Robot command protocol. Use legacy for the current Sunrise app.",
    )
    parser.add_argument(
        "--intent",
        choices=("pause_capture", "stop_after_current_motion", "emergency_stop"),
        default="stop_after_current_motion",
        help="Structured v1 stop intent. Legacy protocol still sends {'stop': true}.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()
    selected_profile = robot_profile(args.robot_mode).with_overrides(
        robot_ip=args.ip_robot,
        command_port=args.port_robot,
    )

    if args.verbose:
        print(
            "Target robot: "
            f"{selected_profile.mode} "
            f"{selected_profile.robot_ip}:{selected_profile.command_port}"
        )

    success = send_stop_message(
        robot_mode=selected_profile.mode,
        ip_robot=selected_profile.robot_ip,
        port_robot=selected_profile.command_port,
        protocol=args.protocol,
        intent=args.intent,
    )

    if success:
        print("Stop message sent successfully.")
        sys.exit(0)
    else:
        print("Failed to send stop message.")
        sys.exit(1)

if __name__ == "__main__":
    main()
