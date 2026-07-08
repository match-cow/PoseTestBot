#!/usr/bin/env python3
"""Capture aligned RGB-D frames from a Stereolabs ZED 2i."""

from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np

from posetestbot.sensors.contracts import CameraIntrinsics, SensorType
from posetestbot.sensors.frame_writer import (
    ensure_legacy_rgbd_folders,
    write_legacy_camera_sidecars,
    write_legacy_rgbd_frame,
)


def resolution_from_name(sl, resolution: str):
    if resolution == "720p":
        return sl.RESOLUTION.HD720
    if resolution == "360p":
        return sl.RESOLUTION.VGA
    raise SystemExit("--resolution must be 720p or 360p")


def save_camera_parameters(output_path: str, left_cam, width: int, height: int) -> None:
    write_legacy_camera_sidecars(
        output_path,
        CameraIntrinsics(
            cam_k=(
                float(left_cam.fx),
                0.0,
                float(left_cam.cx),
                0.0,
                float(left_cam.fy),
                float(left_cam.cy),
                0.0,
                0.0,
                1.0,
            ),
            width=int(width),
            height=int(height),
            depth_scale_to_mm=1.0,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZED 2i RGB-D capture")
    parser.add_argument("output_path", help="Output folder for recording.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Preview without recording frames.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Maximum number of frames to capture. Use 0 for unlimited.",
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
        help="ZED serial number to use.",
    )
    parser.add_argument(
        "--resolution",
        choices=("720p", "360p"),
        default="720p",
        help="Capture resolution profile.",
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

    try:
        import pyzed.sl as sl
    except ImportError as exc:
        raise SystemExit(
            "Stereolabs ZED SDK Python API is not installed. "
            "Install the ZED SDK and its pyzed module, then rerun with uv."
        ) from exc

    record_stream = not args.test

    if record_stream:
        ensure_legacy_rgbd_folders(args.output_path)

    init_params = sl.InitParameters()
    init_params.camera_resolution = resolution_from_name(sl, args.resolution)
    init_params.camera_fps = args.fps
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL

    if args.device:
        init_params.set_from_serial_number(int(args.device))

    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"Could not open ZED camera: {status}")

    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()
    camera_parameters_saved = False
    captured_frames = 0
    valid_frames = 0
    summary = {
        "status": "running",
        "sensor_type": SensorType.ZED_2I.value,
        "sensor_id": args.device or "default",
        "frame_count": 0,
        "preview": bool(args.preview),
        "resolution": args.resolution,
    }

    try:
        while True:
            if args.max_frames > 0 and captured_frames >= args.max_frames:
                break

            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                continue

            host_wall_timestamp_ns = time.time_ns()
            host_received_timestamp_ns = time.monotonic_ns()
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

            rgb_image = image.get_data()
            if rgb_image.shape[2] == 4:
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2BGR)

            depth_image = depth.get_data()
            depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
            depth_image = np.clip(depth_image, 0, np.iinfo(np.uint16).max).astype(
                np.uint16
            )

            valid_frames += 1
            if valid_frames <= args.warmup_frames:
                continue

            if record_stream and not camera_parameters_saved:
                camera_info = zed.get_camera_information()
                config = camera_info.camera_configuration
                left_cam = config.calibration_parameters.left_cam
                save_camera_parameters(
                    args.output_path,
                    left_cam,
                    config.resolution.width,
                    config.resolution.height,
                )
                camera_parameters_saved = True

            key = -1
            if args.preview:
                cv2.imshow("ZED 2i Capture RGB aligned", rgb_image)
                key = cv2.waitKey(1)

            if record_stream:
                sensor_timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)
                write_legacy_rgbd_frame(
                    args.output_path,
                    rgb_image=rgb_image,
                    depth_image=depth_image,
                    sensor_type=SensorType.ZED_2I,
                    sensor_id=args.device or "default",
                    frame_index=captured_frames,
                    sensor_timestamp_ns=sensor_timestamp.get_nanoseconds(),
                    host_received_timestamp_ns=host_received_timestamp_ns,
                    host_wall_timestamp_ns=host_wall_timestamp_ns,
                )

            captured_frames += 1

            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        zed.close()
        if args.preview:
            cv2.destroyAllWindows()

    summary["status"] = "succeeded"
    summary["frame_count"] = captured_frames
    print(
        "ZED 2i capture: "
        f"{summary['status']} {summary['sensor_id']} "
        f"frames={summary['frame_count']} preview={summary['preview']}"
    )
    if args.print_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
