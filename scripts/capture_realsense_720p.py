## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

import argparse
import json
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs


def save_camera_parameters(output_path, intrinsics, depth_scale):
    """Saves camera parameters to multiple formats.

    Args:
        output_path (str): The directory to save the camera parameters.
        intrinsics: The camera intrinsics from the RealSense sensor.
        depth_scale (float): The depth scale value from the RealSense sensor.
    """
    # FoundationPose format
    with open(os.path.join(output_path, "cam_K.txt"), "w") as f:
        f.write(f"{intrinsics.fx} {0.0} {intrinsics.ppx}\n")
        f.write(f"{0.0} {intrinsics.fy} {intrinsics.ppy}\n")
        f.write(f"{0.0} {0.0} {1.0}\n")

    with open(os.path.join(output_path, "depthscale.txt"), "w") as f:
        f.write(f"{depth_scale}\n")

    # SAM6D camera.json format
    with open(os.path.join(output_path, "camera.json"), "w") as f:
        camera_dict = {
            "cam_K": [
                intrinsics.fx,
                0.0,
                intrinsics.ppx,
                0.0,
                intrinsics.fy,
                intrinsics.ppy,
                0.0,
                0.0,
                1.0,
            ],
            "depth_scale": depth_scale,
        }
        json.dump(camera_dict, f, indent=4)

    # MegaPose camera_data.json format
    with open(os.path.join(output_path, "camera_data.json"), "w") as f:
        camera_data_dict = {
            "K": [
                [intrinsics.fx, 0.0, intrinsics.ppx],
                [0.0, intrinsics.fy, intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ],
            "resolution": [720, 1280],
        }
        json.dump(camera_data_dict, f)


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
    args = parser.parse_args()

    output_path = os.path.join(args.output_path, "realsense")
    fps = args.fps
    max_frames = args.max_frames

    RecordStream = not args.test  # Enable or disable recording based on test mode
    if args.test:
        print("Test mode enabled")

    CameraParametersSaved = False  # Flag to check if camera parameters are saved
    captured_frames = 0

    # Temporary fix for issue where camera only works every second time or so...
    # restart the usb service for the device "Intel(R) RealSense(TM) Depth Camera 435i"
    os.system("usbreset 'Intel(R) RealSense(TM) Depth Camera 435i'")
    time.sleep(0.5)

    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    # different resolutions of color and depth streams
    config = rs.config()

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
        # os.makedirs(os.path.join(output_path), exist_ok=True)
        os.makedirs(os.path.join(output_path, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "depth"), exist_ok=True)

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
                framename = int(round(time.time() * 1000))

                # Define the path to the image file within the subfolder
                image_path_depth = os.path.join(output_path, f"depth/{framename}.png")
                image_path_rgb = os.path.join(output_path, f"rgb/{framename}.png")

                cv2.imwrite(image_path_depth, depth_image)
                cv2.imwrite(image_path_rgb, color_image)

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
