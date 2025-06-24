import argparse
import json
import os
import time

import cv2
import depthai as dai
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Luxonis Capture")
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
        "--downscaleColor",
        action="store_true",
        default=True,
        help="Enable downscaled color camera resolution.",
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

    output_path = os.path.join(args.output_path, "luxonis")
    downscaleColor = args.downscaleColor
    fps = args.fps
    max_frames = args.max_frames

    captured_frames = 0

    if args.test:
        print("Test mode enabled")
        RecordStream = False
    else:
        RecordStream = True

    # get output dir from argpase and create folder if necessary
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if RecordStream:
        # os.makedirs(os.path.join(output_path), exist_ok=True)
        os.makedirs(os.path.join(output_path, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "depth"), exist_ok=True)

    # Create pipeline
    pipeline = dai.Pipeline()
    device = dai.Device()

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    left = pipeline.create(dai.node.MonoCamera)
    right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(True)
    stereo.setSubpixel(False)

    # Disabling autofocus by setting a manual focus value (0 to 255)
    # TEST: This seems to not make that much of a difference?
    camRgb.initialControl.setManualFocus(100)

    rgbOut = pipeline.create(dai.node.XLinkOut)
    depthOut = pipeline.create(dai.node.XLinkOut)

    rgbOut.setStreamName("rgb")
    depthOut.setStreamName("depth")

    # TODO: This hase a major outcome on the recorded image, might lead to false intrinincs/alignment
    # need to test this further
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    # camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)

    try:
        calibData = device.readCalibration2()
        lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.CAM_A)
        if lensPosition:
            camRgb.initialControl.setManualFocus(lensPosition)
    except:
        raise

    # The disparity is computed at this resolution, then upscaled to RGB resolution
    monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
    # Set resolution and fps for left and right cameras
    left.setResolution(monoResolution)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    left.setFps(fps)
    right.setResolution(monoResolution)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    right.setFps(fps)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # LR-check is required for depth alignment
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(1280, 720)

    # Linking
    camRgb.isp.link(rgbOut.input)
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.depth.link(depthOut.input)

    # Connect to device and start pipeline
    with device:
        device.startPipeline(pipeline)

        frameRgb = None
        depthFrameRaw = None

        # Configure windows; trackbar adjusts blending ratio of rgb/depth
        rgbWindowName = "Luxonis Capture RGB aligned"
        cv2.namedWindow(rgbWindowName)

        if RecordStream:
            calibData = device.readCalibration()

            M_rgb = np.array(
                calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 1280, 720)
            )

            # print(f"M_rgb: {M_rgb}")

            # TODO: This is mismatched, which might explain the faulty ditance for the aruco pose estimation
            # fov_from_sensor = calibData.getFov(dai.CameraBoardSocket.CAM_A)
            # print(f"fov_from_sensor: {fov_from_sensor}")
            # fov_from_intrinsics = calibData.getFov(
            #     dai.CameraBoardSocket.CAM_A, useSpec=False
            # )
            # print(f"fov_from_intrinsics: {fov_from_intrinsics}")

            # fov_delta_ratio = fov_from_intrinsics / fov_from_sensor
            # print(f"fov_delta_ratio: {fov_delta_ratio}")

            # fov_intrinsics = np.array(
            #     calibData.getCameraIntrinsics(
            #         dai.CameraBoardSocket.CAM_A,
            #         int(1280 * fov_delta_ratio),
            #         int(720 * fov_delta_ratio),
            #     )
            # )
            # print(f"fov_intrinsics: {fov_intrinsics}")
            # # M_rgb = fov_intrinsics

            # distortion = calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A)
            # print(f"Distortion: {distortion}")

            # distortion_str = " ".join(map(str, distortion))

            with open(os.path.join(output_path, "cam_K.txt"), "w") as f:
                f.write(f"{M_rgb[0][0]} {M_rgb[0][1]} {M_rgb[0][2]}\n")
                f.write(f"{M_rgb[1][0]} {M_rgb[1][1]} {M_rgb[1][2]}\n")
                f.write(f"{M_rgb[2][0]} {M_rgb[2][1]} {M_rgb[2][2]}\n")
                # f.write(f"{distortion_str}\n")

            # FoundationPose format
            with open(os.path.join(output_path, "cam_K.txt"), "w") as f:
                f.write(f"{M_rgb[0][0]} {M_rgb[0][1]} {M_rgb[0][2]}\n")
                f.write(f"{M_rgb[1][0]} {M_rgb[1][1]} {M_rgb[1][2]}\n")
                f.write(f"{M_rgb[2][0]} {M_rgb[2][1]} {M_rgb[2][2]}\n")

            with open(os.path.join(output_path, "depthscale.txt"), "w") as f:
                f.write(f"{1.0}\n")

            # SAM6D camera.json format
            with open(os.path.join(output_path, "camera.json"), "w") as f:
                camera_dict = {
                    "cam_K": [
                        M_rgb[0][0],
                        0.0,
                        M_rgb[0][2],
                        0.0,
                        M_rgb[1][1],
                        M_rgb[1][2],
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "depth_scale": 1.0,
                }
                json.dump(camera_dict, f, indent=4)

            # MegaPose camera_data.json format
            with open(os.path.join(output_path, "camera_data.json"), "w") as f:
                camera_data_dict = {
                    "K": [
                        [M_rgb[0][0], M_rgb[0][1], M_rgb[0][2]],
                        [M_rgb[1][0], M_rgb[1][1], M_rgb[1][2]],
                        [M_rgb[2][0], M_rgb[2][1], M_rgb[2][2]],
                    ],
                    "resolution": [720, 1280],
                }
                json.dump(camera_data_dict, f)

        while True:
            latestPacket = {}
            latestPacket["rgb"] = None
            latestPacket["depth"] = None

            queueEvents = device.getQueueEvents(("rgb", "depth"))

            for queueName in queueEvents:
                packets = device.getOutputQueue(queueName).tryGetAll()
                if len(packets) > 0:
                    latestPacket[queueName] = packets[-1]

            if latestPacket["rgb"] is not None:
                frameRgb = latestPacket["rgb"].getCvFrame()
                frameRgb = cv2.resize(
                    frameRgb, (1280, 720), interpolation=cv2.INTER_NEAREST
                )
                cv2.imshow(rgbWindowName, frameRgb)

            if latestPacket["depth"] is not None:
                depthFrameRaw = latestPacket["depth"].getFrame()
                depthFrame = cv2.normalize(
                    depthFrameRaw, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1
                )
                depthFrame = cv2.equalizeHist(depthFrame)
                depthFrame = cv2.applyColorMap(depthFrame, cv2.COLORMAP_HOT)

            if max_frames > 0 and captured_frames > max_frames - 1:
                break

            # Blend when both received
            if frameRgb is not None and depthFrameRaw is not None:
                # Need to have both frames in BGR format before blending
                if RecordStream:
                    framename = int(round(time.time() * 1000))

                    # Define the path to the image file within the subfolder
                    image_path_depth = os.path.join(
                        output_path, f"depth/{framename}.png"
                    )
                    image_path_rgb = os.path.join(output_path, f"rgb/{framename}.png")

                    cv2.imwrite(image_path_depth, depthFrameRaw)
                    cv2.imwrite(image_path_rgb, frameRgb)

                    captured_frames += 1
                    # print(f"Received frames: {captured_frames}", end="\r")

                # TODO: Is this required for all pose estimation methods?
                # if len(depthFrame.shape) < 3:
                #     depthFrame = cv2.cvtColor(depthFrame, cv2.COLOR_GRAY2BGR)

                frameRgb = None
                depthFrameRaw = None

            key = cv2.waitKey(1)

            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                break

    return


if __name__ == "__main__":
    main()