import argparse
import time

import cv2
import depthai as dai
import numpy as np

from posetestbot.sensors.contracts import CameraIntrinsics, SensorType
from posetestbot.sensors.frame_writer import (
    ensure_legacy_rgbd_folders,
    write_legacy_camera_sidecars,
    write_legacy_rgbd_frame,
)


def dai_timestamp_ns(packet, *, device_clock: bool) -> int | None:
    if packet is None:
        return None

    timestamp = packet.getTimestampDevice() if device_clock else packet.getTimestamp()
    if timestamp is None:
        return None
    return int(timestamp.total_seconds() * 1_000_000_000)


def camera_intrinsics_from_matrix(
    matrix: np.ndarray,
    *,
    width: int,
    height: int,
    depth_scale_to_mm: float = 1.0,
) -> CameraIntrinsics:
    return CameraIntrinsics(
        cam_k=tuple(float(value) for value in matrix.reshape(-1)),
        width=width,
        height=height,
        depth_scale_to_mm=depth_scale_to_mm,
    )


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
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="The MX ID of the Luxonis device to use.",
    )
    args = parser.parse_args()

    output_path = args.output_path
    downscaleColor = args.downscaleColor
    fps = args.fps
    max_frames = args.max_frames

    captured_frames = 0

    if args.test:
        print("Test mode enabled")
        RecordStream = False
    else:
        RecordStream = True

    if RecordStream:
        ensure_legacy_rgbd_folders(output_path)

    # Create pipeline
    pipeline = dai.Pipeline()
    device_info = dai.DeviceInfo(args.device) if args.device else None
    device = dai.Device(device_info) if device_info else dai.Device()

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
    camRgb.setFps(fps)

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
        frameRgbPacket = None
        depthFramePacket = None

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

            write_legacy_camera_sidecars(
                output_path,
                camera_intrinsics_from_matrix(M_rgb, width=1280, height=720),
            )

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
                frameRgbPacket = latestPacket["rgb"]
                frameRgb = latestPacket["rgb"].getCvFrame()
                frameRgb = cv2.resize(
                    frameRgb, (1280, 720), interpolation=cv2.INTER_NEAREST
                )
                cv2.imshow(rgbWindowName, frameRgb)

            if latestPacket["depth"] is not None:
                depthFramePacket = latestPacket["depth"]
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
                    host_wall_timestamp_ns = time.time_ns()
                    host_received_timestamp_ns = time.monotonic_ns()
                    write_legacy_rgbd_frame(
                        output_path,
                        rgb_image=frameRgb,
                        depth_image=depthFrameRaw,
                        sensor_type=SensorType.OAK_D_PRO,
                        sensor_id=args.device or "default",
                        frame_index=captured_frames,
                        sensor_timestamp_ns=dai_timestamp_ns(
                            frameRgbPacket, device_clock=True
                        ),
                        depth_sensor_timestamp_ns=dai_timestamp_ns(
                            depthFramePacket, device_clock=True
                        ),
                        host_received_timestamp_ns=host_received_timestamp_ns,
                        host_wall_timestamp_ns=host_wall_timestamp_ns,
                        extra_metadata={
                            "host_timestamp_ns": dai_timestamp_ns(
                                frameRgbPacket, device_clock=False
                            ),
                            "rgb_sequence_num": (
                                frameRgbPacket.getSequenceNum()
                                if frameRgbPacket is not None
                                else None
                            ),
                            "depth_sequence_num": (
                                depthFramePacket.getSequenceNum()
                                if depthFramePacket is not None
                                else None
                            ),
                        },
                    )

                    captured_frames += 1
                    # print(f"Received frames: {captured_frames}", end="\r")

                # TODO: Is this required for all pose estimation methods?
                # if len(depthFrame.shape) < 3:
                #     depthFrame = cv2.cvtColor(depthFrame, cv2.COLOR_GRAY2BGR)

                frameRgb = None
                depthFrameRaw = None
                frameRgbPacket = None
                depthFramePacket = None

            key = cv2.waitKey(1)

            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                break

    return


if __name__ == "__main__":
    main()
