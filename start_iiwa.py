#!/usr/bin/env python3

import argparse
import sys

from posetestbot.config import robot_profile
from posetestbot.robot.udp import send_start


def send_start_message(
    *,
    ip_robot: str | None,
    port_robot: int | None,
    capture_vel: float | None,
    protocol: str,
    run_id: str | None,
    allow_real_robot: bool = False,
    allow_cameras: bool = False,
) -> bool:
    """Send a capture-start message to the configured iiwa controller."""

    if allow_real_robot is not True or allow_cameras is not True:
        print(
            "Starting the iiwa requires fresh --allow-real-robot and "
            "--allow-cameras acknowledgements."
        )
        return False

    profile = robot_profile().with_overrides(
        robot_ip=ip_robot,
        command_port=port_robot,
        cartesian_velocity_m_s=capture_vel,
    )

    try:
        start_message = send_start(profile, protocol=protocol, run_id=run_id)
        print(f"Sent start message to {profile.robot_ip}:{profile.command_port}")
        print(f"Message: {start_message}")
        return True
    except OSError as exc:
        print(f"Socket error: {exc}")
        return False
    except Exception as exc:
        print(f"Error sending start message: {exc}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Send start message to iiwa via UDP")
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
        "--capture_vel",
        type=float,
        default=None,
        help=(
            "Override the requested Cartesian capture velocity in m/s. "
            "The transmitted numeric value is capped at 0.03."
        ),
    )
    parser.add_argument(
        "--protocol",
        choices=("legacy", "v1"),
        default="legacy",
        help="Robot command protocol. Use legacy for the current Sunrise app.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional run identifier included with v1 commands.",
    )
    parser.add_argument(
        "--allow-real-robot",
        action="store_true",
        help="Fresh acknowledgement that this invocation may start robot motion.",
    )
    parser.add_argument(
        "--allow-cameras",
        action="store_true",
        help="Fresh acknowledgement that camera acquisition is authorized.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()
    selected_profile = robot_profile().with_overrides(
        robot_ip=args.ip_robot,
        command_port=args.port_robot,
        cartesian_velocity_m_s=args.capture_vel,
    )

    if args.verbose:
        print(
            "Target robot: "
            f"{selected_profile.mode} "
            f"{selected_profile.robot_ip}:{selected_profile.command_port}"
        )

    success = send_start_message(
        ip_robot=selected_profile.robot_ip,
        port_robot=selected_profile.command_port,
        capture_vel=selected_profile.cartesian_velocity_m_s,
        protocol=args.protocol,
        run_id=args.run_id,
        allow_real_robot=args.allow_real_robot,
        allow_cameras=args.allow_cameras,
    )

    if success:
        print("Start message sent successfully.")
        sys.exit(0)
    else:
        print("Failed to send start message.")
        sys.exit(1)

if __name__ == "__main__":
    main()
