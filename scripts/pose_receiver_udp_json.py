#!/usr/bin/env python3
"""Receive one explicitly authorized iiwa UDP pose stream."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from posetestbot.config import (
    MAX_CAPTURE_COMMAND_VELOCITY_M_S,
    robot_profile,
)
from posetestbot.robot.pose_receiver import (
    DEFAULT_RECEIVE_IDLE_TIMEOUT_S,
    DEFAULT_RECEIVE_START_TIMEOUT_S,
    PoseReceiverCanceled,
    run_pose_receiver,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", help="Path to folder for received data")
    parser.add_argument(
        "--ip",
        type=str,
        default=None,
        help="IP address to bind the receiver socket to.",
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Port to bind the receiver socket to."
    )
    parser.add_argument(
        "--capture_vel",
        type=float,
        default=None,
        help=(
            "Requested capture velocity in m/s. Defaults to the selected "
            "robot profile; the transmitted value is capped at 0.03 by "
            "default. Canonical object-dataset plans may supply a larger "
            "versioned command limit."
        ),
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
        "--maximum-command-velocity-m-s",
        type=float,
        default=None,
        help=(
            "Maximum transmitted capture request in m/s. Values above the "
            "conservative 0.03 legacy limit require --protocol v1. The "
            "canonical object-dataset capture plan supplies this explicitly."
        ),
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
        "--receive-start-timeout-s",
        type=float,
        default=DEFAULT_RECEIVE_START_TIMEOUT_S,
        help="Seconds to wait for the first robot pose packet.",
    )
    parser.add_argument(
        "--receive-idle-timeout-s",
        type=float,
        default=DEFAULT_RECEIVE_IDLE_TIMEOUT_S,
        help="Seconds to wait between robot pose packets.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print received frame diagnostics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output_path)
    if args.output_path == "out":
        output_path = Path(__file__).resolve().parent / "out"
    profile = robot_profile().with_overrides(
        robot_ip=args.ip_robot,
        command_port=args.port_robot,
        receiver_ip=args.ip,
        receiver_port=args.port,
        cartesian_velocity_m_s=args.capture_vel,
    )

    try:
        result = run_pose_receiver(
            output_path,
            profile=profile,
            protocol=args.protocol,
            verbose=args.verbose,
            allow_real_robot=args.allow_real_robot,
            allow_cameras=args.allow_cameras,
            maximum_command_velocity_m_s=(
                args.maximum_command_velocity_m_s
                if args.maximum_command_velocity_m_s is not None
                else MAX_CAPTURE_COMMAND_VELOCITY_M_S
            ),
            receive_start_timeout_s=args.receive_start_timeout_s,
            receive_idle_timeout_s=args.receive_idle_timeout_s,
        )
    except PoseReceiverCanceled as exc:
        print(str(exc), file=sys.stderr)
        return 130
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Wrote {result.raw_pose_path} ({result.pose_count} poses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
