import argparse
import json
import sys

from posetestbot.sensors.realsense import (
    RealSenseCaptureError,
    capture_realsense_rgbd,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realsense Capture")
    parser.add_argument(
        "output_path",
        type=str,
        nargs="?",
        help="Specify the output path for recording.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Enable test mode without recording recording.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Specify the frames per second for capturing.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Specify the maximum number of frames to capture (0 for unlimited).",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=0,
        help="Discard this many valid frames before writing capture output.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="The serial number of the device to use.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show an OpenCV RGB preview window while capturing.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the capture summary JSON after completion.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.test and not args.output_path:
        print(
            "capture_realsense_720p.py: output_path is required unless --test is set",
            file=sys.stderr,
        )
        return 2
    try:
        summary = capture_realsense_rgbd(
            args.output_path,
            device_id=args.device,
            fps=args.fps,
            max_frames=args.max_frames,
            warmup_frames=args.warmup_frames,
            preview=args.preview,
            record=not args.test,
        )
    except (RealSenseCaptureError, ValueError) as exc:
        print(f"capture_realsense_720p.py: {exc}", file=sys.stderr)
        return 2

    print(
        "RealSense capture: "
        f"{summary['status']} {summary['sensor_id']} "
        f"frames={summary['frame_count']} preview={summary['preview']}"
    )
    if args.print_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
