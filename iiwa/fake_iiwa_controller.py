#!/usr/bin/env python3
"""UDP stand-in for the KUKA iiwa Sunrise pose streaming application."""

from __future__ import annotations

import argparse
import json
import math
import select
import socket
import time
from typing import Iterable, NamedTuple


class Pose(NamedTuple):
    x: float
    y: float
    z: float
    a: float
    b: float
    c: float


MOTIONS = ("circ_far", "circ_close", "zoom")

RIGHT = Pose(
    420.0, -210.0, 540.0, math.radians(-25.0), math.radians(30.0), math.radians(178.99)
)
CENTER = Pose(
    0.0, -320.0, 430.0, math.radians(-90.0), math.radians(30.0), math.radians(180.0)
)
LEFT = Pose(
    -420.0,
    -210.0,
    540.0,
    math.radians(-165.0),
    math.radians(30.0),
    math.radians(178.99),
)

RIGHT_CLOSE = Pose(
    230.0, -110.0, 350.0, math.radians(-30.0), math.radians(30.0), math.radians(175.53)
)
CENTER_CLOSE = Pose(
    0.0, -250.0, 350.0, math.radians(-90.0), math.radians(30.0), math.radians(178.99)
)
LEFT_CLOSE = Pose(
    -230.0,
    -110.0,
    350.0,
    math.radians(-170.0),
    math.radians(30.0),
    math.radians(175.53),
)

TOP = Pose(0.0, -320.0, 840.0, math.radians(-90.0), math.radians(18.0), math.radians(180.0))
BOTTOM = Pose(0.0, -160.0, 350.0, math.radians(-90.0), math.radians(18.0), math.radians(180.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fake iiwa robot controller that speaks the current UDP pose protocol."
    )
    parser.add_argument(
        "--bind-ip",
        default="0.0.0.0",
        help="IP address to bind the fake robot command socket to.",
    )
    parser.add_argument(
        "--robot-port",
        type=int,
        default=30300,
        help="UDP port that receives start/stop robot commands.",
    )
    parser.add_argument(
        "--receiver-ip",
        default="127.0.0.1",
        help="IP address of pose_receiver_udp_json.py.",
    )
    parser.add_argument(
        "--receiver-port",
        type=int,
        default=8080,
        help="UDP port of pose_receiver_udp_json.py.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Total mock motion duration in seconds before sending the end packet.",
    )
    parser.add_argument(
        "--sample-ms",
        type=float,
        default=10.0,
        help="Pose packet interval in milliseconds.",
    )
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=0.2,
        help="Seconds to wait after a start command before streaming poses.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Exit after one start command finishes instead of waiting for more commands.",
    )
    return parser.parse_args()


def lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * t


def lerp_pose(start: Pose, end: Pose, t: float) -> Pose:
    return Pose(*(lerp(a, b, t) for a, b in zip(start, end)))


def quadratic_pose(start: Pose, control: Pose, end: Pose, t: float) -> Pose:
    first = lerp_pose(start, control, t)
    second = lerp_pose(control, end, t)
    return lerp_pose(first, second, t)


def pose_to_packet(motion: str, pose: Pose) -> dict[str, float | str]:
    return {
        "motion": motion,
        "X": pose.x,
        "Y": pose.y,
        "Z": pose.z,
        "A": pose.a,
        "B": pose.b,
        "C": pose.c,
    }


def mock_pose(motion: str, t: float) -> Pose:
    eased = 0.5 - 0.5 * math.cos(math.pi * t)

    if motion == "circ_far":
        return quadratic_pose(RIGHT, CENTER, LEFT, eased)
    if motion == "circ_close":
        return quadratic_pose(RIGHT_CLOSE, CENTER_CLOSE, LEFT_CLOSE, eased)
    if motion == "zoom":
        return lerp_pose(TOP, BOTTOM, eased)

    raise ValueError(f"Unknown mock motion: {motion}")


def send_packet(
    sock: socket.socket,
    receiver: tuple[str, int],
    motion: str,
    pose: Pose,
) -> None:
    payload = json.dumps(pose_to_packet(motion, pose), separators=(",", ":")).encode()
    sock.sendto(payload, receiver)


def receive_command(sock: socket.socket, timeout: float | None) -> dict | None:
    readable, _, _ = select.select([sock], [], [], timeout)
    if not readable:
        return None

    data, addr = sock.recvfrom(1024)
    try:
        command = json.loads(data.decode())
    except json.JSONDecodeError:
        print(f"Ignoring invalid JSON from {addr}: {data!r}")
        return None

    if not isinstance(command, dict):
        print(f"Ignoring non-object JSON command from {addr}: {command!r}")
        return None

    return command


def start_value_from_command(command: dict) -> float | None:
    if "start" in command:
        return float(command["start"])

    if command.get("command") == "start_capture":
        return float(command.get("cartesian_velocity_m_s", 0.2))

    return None


def is_stop_command(command: dict) -> bool:
    if command.get("stop") is True:
        return True

    return command.get("command") in {
        "pause_capture",
        "stop_after_current_motion",
        "emergency_stop",
    }


def wait_for_start(sock: socket.socket) -> float:
    while True:
        command = receive_command(sock, None)
        if command is None:
            continue

        start_value = start_value_from_command(command)
        if start_value is not None:
            print(f"Received start command: {start_value!r}")
            return start_value

        if is_stop_command(command):
            print("Received stop command while idle; nothing is running.")
            continue

        print(f"Ignoring command without a known robot command key: {command!r}")


def iter_motion_samples(duration: float, sample_interval: float) -> Iterable[tuple[str, float]]:
    segment_duration = max(duration, 0.0) / len(MOTIONS)
    samples_per_segment = max(1, int(math.ceil(segment_duration / sample_interval)))

    for motion in MOTIONS:
        for sample_index in range(samples_per_segment):
            if samples_per_segment == 1:
                t = 1.0
            else:
                t = sample_index / (samples_per_segment - 1)
            yield motion, t


def run_motion(
    sock: socket.socket,
    receiver: tuple[str, int],
    duration: float,
    sample_interval: float,
    startup_delay: float,
) -> bool:
    if startup_delay > 0:
        time.sleep(startup_delay)

    last_pose = mock_pose(MOTIONS[0], 0.0)
    next_send = time.monotonic()

    for motion, t in iter_motion_samples(duration, sample_interval):
        command = receive_command(sock, max(0.0, next_send - time.monotonic()))
        if command and is_stop_command(command):
            print("Received stop command; ending mock motion early.")
            send_packet(sock, receiver, "end", last_pose)
            return False

        if command and start_value_from_command(command) is not None:
            print("Ignoring start command because a mock motion is already running.")

        pose = mock_pose(motion, t)
        send_packet(sock, receiver, motion, pose)
        last_pose = pose
        next_send += sample_interval

    send_packet(sock, receiver, "end", last_pose)
    print("Mock motion finished; sent end packet.")
    return True


def main() -> None:
    args = parse_args()

    if args.duration < 0:
        raise SystemExit("--duration must be greater than or equal to 0.")
    if args.sample_ms <= 0:
        raise SystemExit("--sample-ms must be greater than 0.")
    if args.startup_delay < 0:
        raise SystemExit("--startup-delay must be greater than or equal to 0.")

    receiver = (args.receiver_ip, args.receiver_port)
    sample_interval = args.sample_ms / 1000.0

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((args.bind_ip, args.robot_port))
        print(f"Fake iiwa controller listening on {args.bind_ip}:{args.robot_port}")
        print(f"Streaming mock poses to {args.receiver_ip}:{args.receiver_port}")

        try:
            while True:
                wait_for_start(sock)
                run_motion(
                    sock,
                    receiver,
                    args.duration,
                    sample_interval,
                    args.startup_delay,
                )

                if args.once:
                    break
        except KeyboardInterrupt:
            print("\nFake iiwa controller stopped.")


if __name__ == "__main__":
    main()
