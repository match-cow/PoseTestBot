## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

import argparse
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from posetestbot.sensors.contracts import CameraIntrinsics, SensorType
from posetestbot.sensors.frame_writer import (
    ensure_legacy_rgbd_folders,
    write_legacy_camera_sidecars,
    write_legacy_rgbd_frame,
)


def camera_intrinsics_from_realsense(intrinsics, depth_scale: float) -> CameraIntrinsics:
    return CameraIntrinsics(
        cam_k=(
            float(intrinsics.fx),
            0.0,
            float(intrinsics.ppx),
            0.0,
            float(intrinsics.fy),
            float(intrinsics.ppy),
            0.0,
            0.0,
            1.0,
        ),
        width=int(getattr(intrinsics, "width", 1280)),
        height=int(getattr(intrinsics, "height", 720)),
        depth_scale_to_mm=float(depth_scale),
    )


def save_camera_parameters(output_path, intrinsics, depth_scale):
    """Saves camera parameters to the shared legacy sidecar formats."""

    write_legacy_camera_sidecars(
        output_path,
        camera_intrinsics_from_realsense(intrinsics, depth_scale),
    )


def main():
    """Captures color and depth streams from a RealSense camera, aligns them, and saves them to disk."""
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
        "--device",
        type=str,
        default=None,
        help="The serial number of the device to use.",
    )
    args = parser.parse_args()

    output_path = args.output_path
    fps = args.fps
    max_frames = args.max_frames

    RecordStream = not args.test  # Enable or disable recording based on test mode
    if args.test:
        print("Test mode enabled")

    CameraParametersSaved = False  # Flag to check if camera parameters are saved
    captured_frames = 0

    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    # different resolutions of color and depth streams
    config = rs.config()

    if args.device:
        config.enable_device(args.device)

    # Get device product line for setting a supporting resolution
    try:
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == "RGB Camera":
            found_rgb = True
            break
    if not found_rgb:
        print("No RGB sensor found")
        exit(0)

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, fps)

    if device_product_line == "L500":
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, fps)
    else:
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, fps)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # scale depthscale
    depth_scale = depth_scale * 1000

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    if RecordStream:
        ensure_legacy_rgbd_folders(output_path)

    # Streaming loop
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            if max_frames > 0 and captured_frames > max_frames - 1:
                break

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Get instrinsics from aligned_depth_frame
            intrinsics = (
                aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            )

            # print(f"intrinsics: {intrinsics}")
            # print(f"intrinsics color: {color_frame.profile.as_video_stream_profile().intrinsics}")

            if RecordStream and not CameraParametersSaved:
                save_camera_parameters(output_path, intrinsics, depth_scale)
                CameraParametersSaved = True

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            cv2.imshow("Realsense Capture RGB algined", color_image)

            key = cv2.waitKey(1)

            if RecordStream:
                host_wall_timestamp_ns = time.time_ns()
                host_received_timestamp_ns = time.monotonic_ns()
                write_legacy_rgbd_frame(
                    output_path,
                    rgb_image=color_image,
                    depth_image=depth_image,
                    sensor_type=SensorType.REALSENSE_D435,
                    sensor_id=args.device or "default",
                    frame_index=captured_frames,
                    sensor_timestamp_ns=int(color_frame.get_timestamp() * 1_000_000),
                    depth_sensor_timestamp_ns=int(
                        aligned_depth_frame.get_timestamp() * 1_000_000
                    ),
                    host_received_timestamp_ns=host_received_timestamp_ns,
                    host_wall_timestamp_ns=host_wall_timestamp_ns,
                    extra_metadata={
                        "color_frame_number": color_frame.get_frame_number(),
                        "depth_frame_number": aligned_depth_frame.get_frame_number(),
                    },
                )

            captured_frames += 1
            # print(f"Received frames: {captured_frames}", end="\r")

            # Press esc or 'q' to close the image window
            if key & 0xFF == ord("q") or key == 27:

                cv2.destroyAllWindows()

                break
    finally:
        pipeline.stop()

    return


if __name__ == "__main__":
    main()
