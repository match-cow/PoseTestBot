import argparse
import json
import os
from typing import Dict, List, Tuple, Union

import numpy as np
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt


def read_megapose(input_file: str) -> Dict[int, np.ndarray]:
    """
    Reads pose data from MegaPose output files.

    Args:
        input_file (str): Path to the directory containing MegaPose output.

    Returns:
        Dict[int, np.ndarray]: A dictionary where keys are frame numbers (int) and
                                values are 4x4 transformation matrices (numpy.ndarray).
    """
    return_dict = {}
    megapose_poses = os.path.join(input_file, "megapose_poses.json")

    with open(megapose_poses, "r") as f:
        megapose_data = json.load(f)

    for key, value in megapose_data.items():
        frame = int(key)
        quaternions = value[0]["TWO"][0]
        translation = value[0]["TWO"][1]
        translation = [t * 1000 for t in translation]  # Convert to mm
        pq = translation + pr.quaternion_wxyz_from_xyzw(quaternions).tolist()

        transformation_matrix = pt.transform_from_pq(pq)
        transformation_matrix = pt.check_transform(
            transformation_matrix, strict_check=False
        )

        return_dict[frame] = transformation_matrix

    return return_dict


def read_sam6d(input_file: str) -> Dict[int, np.ndarray]:
    """
    Reads pose data from SAM6D output files.

    Args:
        input_file (str): Path to the directory containing SAM6D output.

    Returns:
        Dict[int, np.ndarray]: A dictionary where keys are frame numbers (int) and
                                values are 4x4 transformation matrices (numpy.ndarray).
    """
    return_dict = {}
    files = os.listdir(os.path.join(input_file, "detections_pem"))
    file_paths = [os.path.join(input_file, "detections_pem", file) for file in files]

    for file in sorted(file_paths):
        file_name = os.path.basename(file)
        frame = int(os.path.splitext(file_name)[0].split("_")[0])

        with open(file, "r") as f:
            sam6d_data = json.load(f)

        highest_score = max(sam6d_data, key=lambda x: x["score"])

        try:
            rotation_matrix = pr.check_matrix(highest_score["R"])
            translation_vector = highest_score["t"]

            transformation_matrix = pt.transform_from(
                rotation_matrix, translation_vector
            )
            transformation_matrix = pt.check_transform(
                transformation_matrix, strict_check=False
            )

            return_dict[frame] = transformation_matrix
        except Exception as e:
            print(f"SAM6D Error in frame {frame}: {e}")

    return return_dict


def read_foundationpose(input_file: str) -> Dict[int, np.ndarray]:
    """
    Reads pose data from FoundationPose output files.

    Args:
        input_file (str): Path to the directory containing FoundationPose output.

    Returns:
        Dict[int, np.ndarray]: A dictionary where keys are frame numbers (int) and
                                values are 4x4 transformation matrices (numpy.ndarray).
    """
    return_dict = {}

    files = os.listdir(os.path.join(input_file, "ob_in_cam"))
    file_paths = [os.path.join(input_file, "ob_in_cam", file) for file in files]

    for file in sorted(file_paths):
        file_name = os.path.basename(file)
        if len(file_name) > 10:
            continue

        frame = int(os.path.splitext(file_name)[0])

        with open(file, "r") as f:
            lines = f.readlines()
            matrix = []
            for line in lines:
                row = [float(x) for x in line.strip().split()]
                matrix.append(row)

            try:
                transformation_matrix = np.matrix(matrix)
                transformation_matrix[:3, 3] *= 1000  # Convert to mm

                return_dict[frame] = transformation_matrix

            except Exception as e:
                print(f"Foundationpose Error in frame {frame}: {e}")

    return return_dict


def read_blenderproc(input_file: str) -> Dict[int, List[Dict[str, np.ndarray]]]:
    """
    Reads ground truth pose data from BlenderProc output files.

    Args:
        input_file (str): Path to the scene_gt.json file.

    Returns:
        Dict[int, List[Dict[str, np.ndarray]]]: A dictionary where keys are frame numbers (int) and
                                                 values are lists of dictionaries. Each dictionary
                                                 contains the object name (str) and its 4x4 transformation
                                                 matrix (numpy.ndarray).
    """
    return_dict = {}

    with open(input_file, "r") as f:
        scene_gt = json.load(f)

    for key, values in scene_gt.items():
        frame = int(key)
        poses = []

        for value in values:
            object_name = value["obj_id"]

            rotation_matrix = value["cam_R_m2c"]
            rotation_matrix = [
                rotation_matrix[i : i + 3] for i in range(0, len(rotation_matrix), 3)
            ]
            translation_vector = value["cam_t_m2c"]

            transformation_matrix = pt.transform_from(
                rotation_matrix, translation_vector
            )
            transformation_matrix = pt.check_transform(transformation_matrix)

            poses.append({object_name: transformation_matrix})

        return_dict[frame] = poses

    return return_dict


def calc_accuracy(
    gt: Dict[int, List[Dict[str, np.ndarray]]],
    motions_frames: Dict[str, List[str]],
    pose_estimation_poses: Dict[int, np.ndarray],
    object_name: str,
) -> Dict[str, Dict[str, Union[float, List[float]]]]:
    """
    Calculates accuracy metrics for pose estimation.

    Args:
        gt (Dict[int, List[Dict[str, np.ndarray]]]): Ground truth poses.
        motions_frames (Dict[str, List[str]]): Dictionary mapping motions to frames.
        pose_estimation_poses (Dict[int, np.ndarray]): Estimated poses.
        object_name (str): Name of the object being evaluated.

    Returns:
        Dict[str, Dict[str, Union[float, List[float]]]]: A dictionary containing accuracy metrics
                                                          for each motion and overall.
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

            gt_frame_objects = gt[frame]
            gt_object = [obj for obj in gt_frame_objects if object_name in obj][0]

            gt_pose = np.matrix(gt_object[object_name])

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

            gt_x = np.array(gt_rotation_matrix[0, :]).reshape((3,))
            gt_y = np.array(gt_rotation_matrix[1, :]).reshape((3,))
            gt_z = np.array(gt_rotation_matrix[2, :]).reshape((3,))

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

    # Calculate accuracy for all motions
    ap_x = np.mean(x)
    ap_y = np.mean(y)
    ap_z = np.mean(z)

    ap_a = np.mean(a)
    ap_b = np.mean(b)
    ap_c = np.mean(c)

    AP_p = np.sqrt(ap_x**2 + ap_y**2 + ap_z**2)

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
    Calculates the Radial Position (RP) for translation errors.

    Args:
        x (List[float]): List of x-axis translation errors.
        y (List[float]): List of y-axis translation errors.
        z (List[float]): List of z-axis translation errors.
        ap_x (float): Mean x-axis translation error.
        ap_y (float): Mean y-axis translation error.
        ap_z (float): Mean z-axis translation error.

    Returns:
        float: The calculated RP value.
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
    Calculates the Radial Position (RP) for orientation errors.

    Args:
        a (List[float]): List of rotation errors around the x-axis.
        b (List[float]): List of rotation errors around the y-axis.
        c (List[float]): List of rotation errors around the z-axis.
        ap_a (float): Mean rotation error around the x-axis.
        ap_b (float): Mean rotation error around the y-axis.
        ap_c (float): Mean rotation error around the z-axis.

    Returns:
        Tuple[List[float], List[float], List[float]]: A tuple containing the calculated RP values for each axis.
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
    Main function to evaluate pose estimation accuracy.

    Args:
        input_folder (str): Path to the input folder containing the data.
        run_level (bool): Flag indicating if the evaluation should be run at the object level or a level above.
    """
    accuracy = {}

    if run_level:
        object_folders = [f.path for f in os.scandir(input_folder) if f.is_dir()]
        subfolders = []
        for object_folder in object_folders:
            run_name = os.path.basename(object_folder).split("-")[0]
            [
                subfolders.append([run_name, f.path])
                for f in os.scandir(object_folder)
                if f.is_dir()
            ]
    else:
        run_name = os.path.basename(input_folder).split("-")[0]
        print(f"Object name: {run_name}")

        subfolders = [
            [run_name, f.path] for f in os.scandir(input_folder) if f.is_dir()
        ]

    print(f"Subfolders: {subfolders}")

    for run, sensor in subfolders:
        print(f"Sensor: {sensor}")
        sensor_folder = os.path.join(input_folder, sensor)

        gt = read_blenderproc(os.path.join(sensor, "blenderproc/output/scene_gt.json"))

        objects_file = os.path.join(sensor, "blenderproc/objects.json")
        with open(objects_file, "r") as f:
            objects = json.load(f)

        pose_methods = {
            "megapose": read_megapose,
            "sam6d": read_sam6d,
            "foundationpose": read_foundationpose,
            "foundationposeNoTracking": read_foundationpose,
        }

        match_robot_ee_poses_file = os.path.join(
            sensor_folder, "match_robot_ee_poses.json"
        )

        with open(match_robot_ee_poses_file, "r") as f:
            match_robot_ee_poses = json.load(f)

        motions_frames: Dict[str, List[str]] = {}

        for key, value in match_robot_ee_poses.items():
            motion = value["motion"]

            if motion not in motions_frames:
                motions_frames[motion] = [key]
            else:
                motions_frames[motion].append(key)

        sensor_subfolders = [f.path for f in os.scandir(sensor_folder)]
        for folder in sensor_subfolders:
            basename = os.path.basename(folder)
            if "output" not in basename:
                continue

            obj_id = int(basename.split("_")[-2][3:])

            object_name, object_data = list(objects.items())[obj_id]

            pose_estimation_method = basename.split("_")[0]
            pose_estimation_method_long = "_".join(basename.split("_")[:-1])

            pose_estimation_poses = pose_methods[pose_estimation_method](folder)

            method_accuracy = calc_accuracy(
                gt, motions_frames, pose_estimation_poses, object_name
            )

            accuracy[pose_estimation_method_long] = method_accuracy

        accuracy_output_file = os.path.join(sensor, "accuracy_HRC-Hub.json")
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
