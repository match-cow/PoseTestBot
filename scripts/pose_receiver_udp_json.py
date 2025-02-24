import argparse
import json
import os
import socket
import time


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
        default="172.31.1.151",
        help="IP address to bind the socket to",
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port to bind the socket to"
    )
    parser.add_argument(
        "--capture_vel",
        type=int,
        default=50,
        help="Capture velocity, default is 50",
    )
    parser.add_argument(
        "--ip_robot",
        type=str,
        default="172.31.1.147",
        help="IP address of the robot.",
    )
    parser.add_argument(
        "--port_robot",
        type=int,
        default=30300,
        help="port of the robot.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Enable test mode, default is False",
    )

    args = parser.parse_args()

    ip_address = args.ip
    port = args.port
    output_path = args.output_path
    capture_vel = args.capture_vel
    ip_robot = args.ip_robot
    port_robot = args.port_robot
    test = args.test

    script_dir = os.path.dirname(os.path.relpath(__file__))
    if output_path == "out":
        output_path = os.path.join(script_dir, output_path)

    # Create the output directory if it does not exist
    os.makedirs(output_path, exist_ok=True)

    # Test if output_path is a directory
    if not os.path.isdir(output_path):
        raise ValueError("Output path is not a directory")

    # Send start message with capture vel to to the robot via tcp
    start_message = {"start": capture_vel}
    print(f"sending start message: {start_message}")

    # Convert the start_message to JSON
    start_message_json = json.dumps(start_message)

    # Send the start_message JSON to the robot via UDP
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(start_message_json.encode(), (ip_robot, port_robot))
        print(
            f"Sent start message to {ip_robot}:{port_robot} with capture vel {capture_vel}"
        )

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind the socket to a specific IP address and port
    sock.bind((ip_address, port))

    poses = {}

    received_frames = 0
    # Receive and display incoming UDP packets
    while True:
        data, addr = sock.recvfrom(1024)

        # load dict from json string
        pose_dict = json.loads(data)
        motion = pose_dict["motion"]

        if motion == "end":
            break

        # Convert the received data to a dictionary
        framename = int(round(time.time() * 1000))

        if received_frames == 0:
            previous_frame_ts = framename
            frame_delta = 0
        else:
            frame_delta = framename - int(previous_frame_ts)
            previous_frame_ts = framename

        poses[received_frames] = {
            "framename": framename,
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

        if test:
            # If test is enabled, print the received data
            print(f"framename: {framename}, motion: {motion}, pose_dict: {pose_dict}")

        received_frames += 1
        # Print the number of received frames
        print(f"Received poses: {received_frames}", end="\r")

    with open(os.path.join(output_path, f"raw_robot_ee_poses.json"), "w") as f:
        json.dump(poses, f, indent=4)

    return


if __name__ == "__main__":
    main()
