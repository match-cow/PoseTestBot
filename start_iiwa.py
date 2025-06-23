#!/usr/bin/env python3

import argparse
import json
import socket
import sys

def send_stop_message(ip_robot, port_robot):
    """Send stop message to robot via UDP"""
    
    # Stop message
    stop_message = {"start": 0.2}
    print(f"Sending stop message: {stop_message}")
    
    # Convert the stop_message to JSON
    stop_message_json = json.dumps(stop_message)
    
    try:
        # Send the stop_message JSON to the robot via UDP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(stop_message_json.encode(), (ip_robot, port_robot))
            print(f"Sent stop message to {ip_robot}:{port_robot}")
            return True
            
    except socket.error as e:
        print(f"Socket error: {e}")
        return False
    except Exception as e:
        print(f"Error sending stop message: {e}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Send stop message to robot via UDP")
    parser.add_argument(
        "--ip_robot",
        type=str,
        default="172.31.1.147",
        help="IP address of the robot (default: 172.31.1.147)",
    )
    parser.add_argument(
        "--port_robot",
        type=int,
        default=30300,
        help="Port of the robot (default: 30300)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        print(f"Target robot: {args.ip_robot}:{args.port_robot}")

    # Send stop message
    success = send_stop_message(args.ip_robot, args.port_robot)
    
    if success:
        print("✅ Stop message sent successfully!")
        sys.exit(0)
    else:
        print("❌ Failed to send stop message!")
        sys.exit(1)

if __name__ == "__main__":
    main()