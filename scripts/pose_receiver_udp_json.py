import argparse
import json
import os
import socket
import time

from posetestbot.config import robot_profile
from posetestbot.io.atomic import atomic_write_json
from posetestbot.io.artifacts import RAW_ROBOT_EE_POSES
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    record_raw_robot_pose_artifact,
    upsert_stage,
    write_run_manifest,
)
from posetestbot.robot.udp import send_start


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_path",
        help="Path to folder for received data",
    )
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
        help="Capture velocity in m/s. Defaults to the selected robot profile.",
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
        "--verbose",
        action="store_true",
        help="Print received frame diagnostics.",
    )

    args = parser.parse_args()

    output_path = args.output_path
    verbose = args.verbose
    profile = robot_profile().with_overrides(
        robot_ip=args.ip_robot,
        command_port=args.port_robot,
        receiver_ip=args.ip,
        receiver_port=args.port,
        cartesian_velocity_m_s=args.capture_vel,
    )

    script_dir = os.path.dirname(os.path.relpath(__file__))
    if output_path == "out":
        output_path = os.path.join(script_dir, output_path)

    # Create the output directory if it does not exist
    os.makedirs(output_path, exist_ok=True)

    # Test if output_path is a directory
    if not os.path.isdir(output_path):
        raise ValueError("Output path is not a directory")

    manifest = load_or_create_run_manifest(
        output_path,
        robot_profile=profile,
        capture_config={
            "cartesian_velocity_m_s": profile.cartesian_velocity_m_s,
            "protocol": args.protocol,
            "mode": "real",
        },
    )
    upsert_stage(manifest, name="robot_pose_capture", status="running")
    write_run_manifest(manifest, output_path)

    # Create a UDP socket
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        # Bind before triggering the robot so early packets are not missed.
        sock.bind((profile.receiver_ip, profile.receiver_port))
        print(f"Listening on {profile.receiver_ip}:{profile.receiver_port}")

        start_message = send_start(profile, protocol=args.protocol)
        print(
            "Sent start message to "
            f"{profile.robot_ip}:{profile.command_port} "
            f"with capture vel {profile.cartesian_velocity_m_s}"
        )
        print(f"Message: {start_message}")

        poses = {}
        received_frames = 0
        previous_frame_ts = 0

        while True:
            data, addr = sock.recvfrom(1024)
            host_received_timestamp_ns = time.monotonic_ns()
            host_wall_timestamp_ns = time.time_ns()

            pose_dict = json.loads(data)
            motion = pose_dict["motion"]

            if motion == "end":
                break

            framename = int(round(host_wall_timestamp_ns / 1_000_000))

            if received_frames == 0:
                frame_delta = 0
            else:
                frame_delta = framename - int(previous_frame_ts)
            previous_frame_ts = framename

            poses[received_frames] = {
                "framename": framename,
                "host_received_timestamp_ns": host_received_timestamp_ns,
                "host_wall_timestamp_ns": host_wall_timestamp_ns,
                "frame_delta": frame_delta,
                "motion": motion,
                "pose": {
                    "X": pose_dict["X"],
                    "Y": pose_dict["Y"],
                    "Z": pose_dict["Z"],
                    "A": pose_dict["A"],
                    "B": pose_dict["B"],
                    "C": pose_dict["C"],
                },
            }

            if verbose:
                print(
                    f"framename: {framename}, addr: {addr}, "
                    f"motion: {motion}, pose_dict: {pose_dict}"
                )

            received_frames += 1
            print(f"Received poses: {received_frames}", end="\r")

    atomic_write_json(
        os.path.join(output_path, RAW_ROBOT_EE_POSES),
        poses,
        indent=4,
        sort_keys=False,
    )

    record_raw_robot_pose_artifact(manifest, output_path)
    write_run_manifest(manifest, output_path)

    return


if __name__ == "__main__":
    main()
