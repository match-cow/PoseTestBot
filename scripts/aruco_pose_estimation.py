import argparse
import concurrent.futures
import json
import os
from typing import Any, Dict, List, Tuple

import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm

""" 
This script performs ArUco grid pose estimation on a set of input images.
It detects ArUco markers in the images, calculates the pose of the markers, and saves the results in a JSON file.

Based on OpenCV documentation: https://docs.opencv.org/4.9.0/db/da9/tutorial_aruco_board_detection.html
"""


def run_aruco_pose_estimation(
    image_name: str,
    frame_folder: str,
    detector: cv.aruco.ArucoDetector,
    board: cv.aruco.GridBoard,
    cam_matrix: np.ndarray,
    dist_coefficients: np.ndarray,
) -> Dict[str, Any]:
    """
    Detects ArUco markers in an image, estimates the pose, and returns the results.

    Args:
        image_name (str): The name of the image file.
        frame_folder (str): The path to the folder containing the image.
        detector (cv.aruco.ArucoDetector): The ArUco detector object.
        board (cv.aruco.GridBoard): The ArUco board object.
        cam_matrix (np.ndarray): The camera matrix.
        dist_coefficients (np.ndarray): The distortion coefficients.

    Returns:
        Dict[str, Any]: A dictionary containing the image name and the ArUco pose estimation results.
    """
    if not image_name.endswith((".png", ".jpg", ".jpeg")):
        return None

    image_path = os.path.join(frame_folder, image_name)
    image = cv.imread(image_path)

    if image is None:
        print(f"Error: unable to open image {image_path}")
        return None

    image_copy = np.copy(image)
    corners, ids, _ = detector.detectMarkers(image)
    rvec, tvec = None, None  # Initialize rvec and tvec

    if ids is not None and len(ids) > 0:
        cv.aruco.drawDetectedMarkers(image_copy, corners, ids)
        objPoints, imgPoints = board.matchImagePoints(corners, ids)
        _, rvec, tvec, _ = cv.solvePnPRansac(
            objPoints, imgPoints, cam_matrix, dist_coefficients
        )

    # Ensure rvec and tvec are not None before accessing their elements
    rvec_list = [r[0] for r in rvec] if rvec is not None else []
    tvec_list = [t[0] for t in tvec] if tvec is not None else []

    return {
        "image_name": image_name,
        "aruco_pose_estimation": {
            "rvec": rvec_list,
            "tvec": tvec_list,
            "len_ids": len(ids) if ids is not None else 0,
        },
    }


def read_camera_parameters(input_folder: str) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    Read camera parameters from a file.

    Args:
        input_folder (str): Path to the folder containing the camera parameters file.

    Returns:
        Tuple[bool, np.ndarray, np.ndarray]: A tuple containing:
            - bool: True if the camera parameters are successfully read.
            - np.ndarray: Camera matrix.
            - np.ndarray: Distortion coefficients.
    """
    filename = os.path.join(input_folder, "cam_K.txt")
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Error: file {filename} not found")

    with open(filename, "r") as f:
        lines = f.readlines()
        cam_matrix = np.zeros((3, 3))
        dist_coefficients = np.zeros((5,))  # Modified to (5,)
        for i, line in enumerate(lines):
            if i < 3:
                cam_matrix[i] = np.array([float(x) for x in line.split()])
            else:
                dist_coefficients = np.array([float(x) for x in line.split()])

    return True, cam_matrix, dist_coefficients


def process_sensor_data(
    input_folder: str,
    sensor: str,
    aruco_dict: cv.aruco.Dictionary,
    board: cv.aruco.GridBoard,
    save_images: bool,
    quiet: bool,
    wait_time: int,
):
    """
    Processes the data for a single sensor, performing ArUco pose estimation and saving the results.

    Args:
        input_folder (str): The path to the input folder containing sensor data.
        sensor (str): The name of the sensor.
        aruco_dict (cv.aruco.Dictionary): The ArUco dictionary to use.
        board (cv.aruco.GridBoard): The ArUco board to use.
        save_images (bool): Whether to save the images with detected markers.
        quiet (bool): Whether to suppress image display.
        wait_time (int): The time to wait between displaying images.
    """
    frame_folder = os.path.join(input_folder, sensor, "rgb")
    _, cam_matrix, dist_coefficients = read_camera_parameters(
        os.path.join(input_folder, sensor)
    )

    detector_parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, detector_parameters)

    json_file = os.path.join(input_folder, sensor, "match_robot_ee_poses.json")
    if not os.path.isfile(json_file):
        raise FileNotFoundError(f"Error: file {json_file} not found")

    with open(json_file, "r") as f:
        data = json.load(f)

    images = os.listdir(frame_folder)
    images = sorted(images)

    if quiet and not save_images:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            aruco_poses = list(
                tqdm(
                    executor.map(
                        lambda image_name: run_aruco_pose_estimation(
                            image_name,
                            frame_folder,
                            detector,
                            board,
                            cam_matrix,
                            dist_coefficients,
                        ),
                        images,
                    ),
                    total=len(images),
                )
            )

            aruco_poses = [pose for pose in aruco_poses if pose is not None]

        for pose in aruco_poses:
            pose_frame = pose["image_name"]
            pose_pose = pose["aruco_pose_estimation"]
            data[pose_frame].update({"aruco_pose_estimation": pose_pose})

    else:
        for image_name in tqdm(images):
            if not image_name.endswith((".png", ".jpg", ".jpeg")):
                continue

            image_path = os.path.join(frame_folder, image_name)
            image = cv.imread(image_path)

            if image is None:
                print(f"Error: unable to open image {image_path}")
                continue

            image_copy = np.copy(image)
            corners, ids, _ = detector.detectMarkers(image)
            rvec, tvec = None, None

            if ids is not None and len(ids) > 0:
                cv.aruco.drawDetectedMarkers(image_copy, corners, ids)
                objPoints, imgPoints = board.matchImagePoints(corners, ids)
                _, rvec, tvec, _ = cv.solvePnPRansac(
                    objPoints, imgPoints, cam_matrix, dist_coefficients
                )

                if (
                    _
                    and len(ids) >= 4
                    and (not quiet or save_images)
                    and rvec is not None
                    and tvec is not None
                ):
                    cv.drawFrameAxes(
                        image_copy, cam_matrix, dist_coefficients, rvec, tvec, 100
                    )

            if not quiet:
                cv.imshow("out", image_copy)

            if save_images:
                output_folder = os.path.join(input_folder, sensor, "aruco")
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, image_name)
                cv.imwrite(output_path, image_copy)

            key = cv.waitKey(wait_time)
            if key == 27:
                break

            # Ensure rvec and tvec are not None before accessing their elements
            rvec_list = [r[0] for r in rvec] if rvec is not None else []
            tvec_list = [t[0] for t in tvec] if tvec is not None else []

            data[image_name].update(
                {
                    "aruco_pose_estimation": {
                        "rvec": rvec_list,
                        "tvec": tvec_list,
                        "len_ids": len(ids) if ids is not None else 0,
                    }
                }
            )

    output_json_file = os.path.join(input_folder, sensor, "aruco_pose_estimation.json")
    with open(output_json_file, "w") as f:
        json.dump(data, f, indent=4, default=str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArUco grid pose estimation")
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the folder containing the sensor folders with images, camera parameters.",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="if used the images will be saved in an aruco folder",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="if used the images will not be displayed",
    )
    parser.add_argument(
        "--wait_time",
        type=int,
        default=1,
        help="Time in milliseconds to wait for a key press, default is 1.",
    )

    args = parser.parse_args()
    input_folder = args.input_folder
    save_images = args.save_images
    quiet = args.quiet
    wait_time = args.wait_time

    sensors = [
        folder
        for folder in os.listdir(input_folder)
        if os.path.isdir(os.path.join(input_folder, folder))
    ]

    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_50)
    board = cv.aruco.GridBoard((4, 3), 50, 65, aruco_dict)

    for sensor in sensors:
        process_sensor_data(
            input_folder, sensor, aruco_dict, board, save_images, quiet, wait_time
        )
