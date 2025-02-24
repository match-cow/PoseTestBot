import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager


def read_aruco(input_folder: str) -> Dict[int, np.ndarray]:
    """
    Reads ArUco pose estimation data from a JSON file and converts it into a dictionary of transformation matrices.

    Args:
        input_folder (str): Path to the folder containing the 'aruco_pose_estimation.json' file.

    Returns:
        Dict[int, np.ndarray]: A dictionary where keys are frame numbers (int) and values are 4x4 transformation matrices (numpy.ndarray)
                                representing the ArUco marker pose in each frame.
    """
    return_dict = {}

    with open(os.path.join(input_folder, "aruco_pose_estimation.json"), "r") as f:
        aruco_pose_estimation = json.load(f)

    for key, value in aruco_pose_estimation.items():
        frame = int(key.split(".")[0])
        rotation_vector = value["aruco_pose_estimation"]["rvec"]
        translation_vector = value["aruco_pose_estimation"]["tvec"]

        transformation_matrix = pt.transform_from(
            pr.matrix_from_compact_axis_angle(
                np.array(rotation_vector),
            ),
            np.array(translation_vector),
        )
        transformation_matrix = pt.check_transform(transformation_matrix)

        return_dict[frame] = transformation_matrix

    return return_dict


def read_gt(input_folder: str) -> Dict[int, np.ndarray]:
    """
    Reads ground truth poses from JSON files and computes the transformation from the ArUco marker to the base frame.

    Args:
        input_folder (str): Path to the folder containing the ground truth data and 'aruco_pose_estimation.json'.

    Returns:
        Dict[int, np.ndarray]: A dictionary where keys are frame numbers (int) and values are 4x4 transformation matrices (numpy.ndarray)
                                representing the ArUco marker pose in the base frame for each frame.
    """
    return_dict = {}

    # get name of sensor from last subfolder in input_folder
    sensor = os.path.basename(input_folder)

    # Get file camera_ee_transform.json from parent folder of input_folder
    camera_ee_transform_file = os.path.join(
        os.path.dirname(input_folder), "camera_ee_transform.json"
    )

    with open(camera_ee_transform_file, "r") as f:
        sensor_transformation = json.load(f)[sensor]

    transformation_cam_ee = pt.transform_from(
        pr.matrix_from_quaternion(sensor_transformation["quaternion"]),
        sensor_transformation["position"],
    )

    transformation_cam_ee = pt.check_transform(transformation_cam_ee)

    # Transformation from aruco to base
    transformation_aruco_base = pt.transform_from(
        pr.active_matrix_from_angle(0, np.deg2rad(180.0)),
        np.array([-199.5, 137.0, 0.0]),
    )

    with open(os.path.join(input_folder, "aruco_pose_estimation.json"), "r") as f:
        ee_poses = json.load(f)

    for key, value in ee_poses.items():
        frame = int(key.split(".")[0])

        ee_pose = value["robot_ee_pose"]

        transformation_ee_pose = pt.transform_from(
            pr.matrix_from_euler(
                np.array(
                    [
                        ee_pose["C"],
                        ee_pose["B"],
                        ee_pose["A"],
                    ]
                ),
                0,
                1,
                2,
                True,
            ),
            np.array(
                [
                    ee_pose["X"],
                    ee_pose["Y"],
                    ee_pose["Z"],
                ]
            ),
        )

        tm = TransformManager()
        tm.add_transform("camera", "ee", transformation_cam_ee)
        tm.add_transform("ee", "base", transformation_ee_pose)
        tm.add_transform("aruco", "base", transformation_aruco_base)

        aruco2camera = tm.get_transform("aruco", "camera")
        aruco2camera = pt.check_transform(aruco2camera)
        return_dict[frame] = aruco2camera

    return return_dict


def calc_accuracy(
    gt: Dict[int, np.ndarray],
    motions_frames: Dict[str, List[str]],
    pose_estimation_poses: Dict[int, np.ndarray],
) -> Dict[str, Dict[str, List[float]]]:
    """
    Calculates the accuracy of pose estimation by comparing estimated poses with ground truth poses.

    Args:
        gt (Dict[int, np.ndarray]): Ground truth poses, where keys are frame numbers and values are 4x4 transformation matrices.
        motions_frames (Dict[str, List[str]]): Dictionary containing lists of frames for each motion.
        pose_estimation_poses (Dict[int, np.ndarray]): Estimated poses, where keys are frame numbers and values are 4x4 transformation matrices.

    Returns:
        Dict[str, Dict[str, List[float]]]: A dictionary containing accuracy metrics for each motion and for all motions combined.
                                            The keys are motion names (or "all_motions") and the values are dictionaries of accuracy metrics.
    """
    accuracy = {}

    x = []
    y = []
    z = []
    a = []
    b = []
    c = []

    for key, value in motions_frames.items():
        motion = key
        motion_frames = value

        motions_x = []
        motions_y = []
        motions_z = []
        motions_a = []
        motions_b = []
        motions_c = []

        for mf in motion_frames:
            frame = int(mf.split(".")[0])
            try:
                pose = pose_estimation_poses[frame]
            except KeyError:
                print(f"Frame {frame} not found in pose estimation results.")
                continue

            gt_pose = np.matrix(gt[frame])

            # TRANSLATION
            gt_translation = gt_pose[:3, 3]
            pose_estimation_translation = pose[:3, 3]

            x_j = (pose_estimation_translation[0] - gt_translation[0]).item()
            y_j = (pose_estimation_translation[1] - gt_translation[1]).item()
            z_j = (pose_estimation_translation[2] - gt_translation[2]).item()

            x.append(x_j)
            y.append(y_j)
            z.append(z_j)

            motions_x.append(x_j)
            motions_y.append(y_j)
            motions_z.append(z_j)

            # ORIENTATION
            gt_rotation_matrix = gt_pose[:3, :3]
            pose_rotation_matrix = pose[:3, :3]

            # This is xc, yc, zc in the norm
            gt_x = np.array(gt_rotation_matrix[0, :]).reshape((3,))
            gt_y = np.array(gt_rotation_matrix[1, :]).reshape((3,))
            gt_z = np.array(gt_rotation_matrix[2, :]).reshape((3,))

            # This is xj, yj, zj in the norm
            pose_x = np.array(pose_rotation_matrix[0, :]).reshape((3,))
            pose_y = np.array(pose_rotation_matrix[1, :]).reshape((3,))
            pose_z = np.array(pose_rotation_matrix[2, :]).reshape((3,))

            i_a = np.rad2deg(pr.angle_between_vectors(gt_x, pose_x))
            i_b = np.rad2deg(pr.angle_between_vectors(gt_y, pose_y))
            i_c = np.rad2deg(pr.angle_between_vectors(gt_z, pose_z))

            a.append(i_a)
            b.append(i_b)
            c.append(i_c)

            motions_a.append(i_a)
            motions_b.append(i_b)
            motions_c.append(i_c)

        # Calculate accuracy for each motion
        motion_ap_x = np.mean(motions_x)
        motion_ap_y = np.mean(motions_y)
        motion_ap_z = np.mean(motions_z)

        motion_ap_a = np.mean(motions_a)
        motion_ap_b = np.mean(motions_b)
        motion_ap_c = np.mean(motions_c)

        motion_AP_p = np.sqrt(motion_ap_x**2 + motion_ap_y**2 + motion_ap_z**2)

        motion_RP_i = calc_RP_l(
            motions_x, motions_y, motions_z, motion_ap_x, motion_ap_y, motion_ap_z
        )

        motion_RP_a, motion_RP_b, motion_RP_c = calc_RP_abc(
            motions_a, motions_b, motions_c, motion_ap_a, motion_ap_b, motion_ap_c
        )

        motion_accuracy = {
            "AP_p": motion_AP_p,
            "ap_x": motion_ap_x,
            "ap_y": motion_ap_y,
            "ap_z": motion_ap_z,
            "ap_a": motion_ap_a,
            "ap_b": motion_ap_b,
            "ap_c": motion_ap_c,
            "RP_i": motion_RP_i,
            "RP_a": motion_RP_a,
            "RP_b": motion_RP_b,
            "RP_c": motion_RP_c,
            "x": motions_x,
            "y": motions_y,
            "z": motions_z,
            "a": motions_a,
            "b": motions_b,
            "c": motions_c,
        }

        accuracy[motion] = motion_accuracy

        # End of motion loop

    # Calculate accuracy for all motions
    ap_x = np.mean(x)
    ap_y = np.mean(y)
    ap_z = np.mean(z)

    ap_a = np.mean(a)
    ap_b = np.mean(b)
    ap_c = np.mean(c)

    AP_p = np.sqrt(ap_x**2 + ap_y**2 + ap_z**2)

    # save all results with information on object and sensor to the accuracy dict

    RP_i = calc_RP_l(x, y, z, ap_x, ap_y, ap_z)

    RP_a, RP_b, RP_c = calc_RP_abc(a, b, c, ap_a, ap_b, ap_c)

    accuracy["all_motions"] = {
        "AP_p": AP_p,
        "ap_x": ap_x,
        "ap_y": ap_y,
        "ap_z": ap_z,
        "ap_a": ap_a,
        "ap_b": ap_b,
        "ap_c": ap_c,
        "RP_i": RP_i,
        "RP_a": RP_a,
        "RP_b": RP_b,
        "RP_c": RP_c,
        "x": x,
        "y": y,
        "z": z,
        "a": a,
        "b": b,
        "c": c,
    }

    return accuracy


def calc_RP_l(
    x: List[float],
    y: List[float],
    z: List[float],
    ap_x: float,
    ap_y: float,
    ap_z: float,
) -> float:
    """
    Calculates the Radial Position (RP) error based on the L2 norm of the translation error.

    Args:
        x (List[float]): List of x-axis translation errors.
        y (List[float]): List of y-axis translation errors.
        z (List[float]): List of z-axis translation errors.
        ap_x (float): Mean of x-axis translation errors.
        ap_y (float): Mean of y-axis translation errors.
        ap_z (float): Mean of z-axis translation errors.

    Returns:
        float: The calculated RP error.
    """
    l = []

    for i in range(len(x)):
        error = np.sqrt((x[i] - ap_x) ** 2 + (y[i] - ap_y) ** 2 + (z[i] - ap_z) ** 2)
        l.append(error)

    l_mean = np.mean(l)

    S_l = np.sqrt(np.sum([((lj - l_mean) ** 2) for lj in l]) / (len(l) - 1))

    RP_l = l_mean + 3 * S_l

    return RP_l


def calc_RP_abc(
    a: List[float],
    b: List[float],
    c: List[float],
    ap_a: float,
    ap_b: float,
    ap_c: float,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculates the Radial Position (RP) error for orientation errors along the a, b, and c axes.

    Args:
        a (List[float]): List of orientation errors along the a-axis.
        b (List[float]): List of orientation errors along the b-axis.
        c (List[float]): List of orientation errors along the c-axis.
        ap_a (float): Mean of orientation errors along the a-axis.
        ap_b (float): Mean of orientation errors along the b-axis.
        ap_c (float): Mean of orientation errors along the c-axis.

    Returns:
        Tuple[List[float], List[float], List[float]]: A tuple containing the calculated RP errors for the a, b, and c axes.
    """
    S_a = np.sqrt(np.sum([((ai - ap_a) ** 2) for ai in a]) / (len(a) - 1))
    RP_a = [S_a * 3, -1 * S_a * 3]

    S_b = np.sqrt(np.sum([((bi - ap_b) ** 2) for bi in b]) / (len(b) - 1))
    RP_b = [S_b * 3, -1 * S_b * 3]

    S_c = np.sqrt(np.sum([((ci - ap_c) ** 2) for ci in c]) / (len(c) - 1))
    RP_c = [S_c * 3, -1 * S_c * 3]

    return RP_a, RP_b, RP_c


def main(input_folder: str, run_level: bool) -> None:
    """
    Main function to evaluate the accuracy of ArUco marker based pose estimation.

    Args:
        input_folder (str): Path to the input folder containing object folders or sensor folders directly.
        run_level (bool): If True, the input folder contains object folders, each containing sensor folders.
                          If False, the input folder directly contains sensor folders.
    """
    accuracy = {}

    if run_level:
        object_folders = [f.path for f in os.scandir(input_folder) if f.is_dir()]
        subfolders = []
        for object_folder in object_folders:
            object_name = os.path.basename(object_folder).split("-")[0]
            [
                subfolders.append([object_name, f.path])
                for f in os.scandir(object_folder)
                if f.is_dir()
            ]
    else:
        object_name = os.path.basename(input_folder).split("-")[0]
        print(f"Object name: {object_name}")

        # get list of subfolders in the input_folder
        subfolders = [
            [object_name, f.path] for f in os.scandir(input_folder) if f.is_dir()
        ]

    print(f"Subfolders: {subfolders}")

    for object, sensor in subfolders:
        sensor_folder = os.path.join(input_folder, sensor)

        gt = read_gt(sensor_folder)

        match_robot_ee_poses_file = os.path.join(
            sensor_folder, "match_robot_ee_poses.json"
        )

        with open(match_robot_ee_poses_file, "r") as f:
            match_robot_ee_poses = json.load(f)

        motions_frames: dict = {}

        for key, value in match_robot_ee_poses.items():
            motion = value["motion"]

            if motion not in motions_frames:
                motions_frames[motion] = [key]
            else:
                motions_frames[motion].append(key)

        aruco_estimation_poses = read_aruco(sensor_folder)

        method_accuracy = calc_accuracy(gt, motions_frames, aruco_estimation_poses)

        accuracy["ArUco_accuracy"] = method_accuracy

        # save dict to json
        accuracy_output_file = os.path.join(sensor, "accuracy_ArUco_HRC-Hub.json")
        with open(accuracy_output_file, "w") as f:
            json.dump(accuracy, f, indent=4, default=str)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", help="Path to the input folder")
    parser.add_argument(
        "--run_level",
        action="store_true",
        help="Choose if wrapper is for run level above object level",
    )

    args = parser.parse_args()

    main(args.input_folder, args.run_level)
