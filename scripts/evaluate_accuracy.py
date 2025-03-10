import argparse
import json
import os
from typing import Dict, List, Union

import numpy as np
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt


def read_megapose(input_file: str) -> Dict[int, np.ndarray]:
    """Reads pose data from MegaPose output files.

    Args:
        input_file: Path to the directory containing MegaPose output.

    Returns:
        A dictionary where keys are frame numbers (int) and values are 4x4 transformation matrices (numpy arrays).
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
        pq = translation + quaternions

        transformation_matrix = pt.transform_from_pq(pq)
        transformation_matrix = pt.check_transform(
            transformation_matrix, strict_check=False
        )

        return_dict[frame] = transformation_matrix

    return return_dict


def read_sam6d(input_file: str) -> Dict[int, np.ndarray]:
    """Reads pose data from SAM6D output files.

    Args:
        input_file: Path to the directory containing SAM6D output.

    Returns:
        A dictionary where keys are frame numbers (int) and values are 4x4 transformation matrices (numpy arrays).
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
    """Reads pose data from FoundationPose output files.

    Args:
        input_file: Path to the directory containing FoundationPose output.

    Returns:
        A dictionary where keys are frame numbers (int) and values are 4x4 transformation matrices (numpy arrays).
    """
    return_dict = {}

    files = os.listdir(os.path.join(input_file, "ob_in_cam"))
    file_paths = [os.path.join(input_file, "ob_in_cam", file) for file in files]

    for file in sorted(file_paths):
        file_name = os.path.basename(file)
        frame = int(os.path.splitext(file_name)[0])

        with open(file, "r") as f:
            lines = f.readlines()
            matrix = []
            for line in lines:
                row = [float(x) for x in line.strip().split()]
                matrix.append(row)

            try:
                transformation_matrix = pt.check_transform(matrix, strict_check=False)
                transformation_matrix[:3, 3] *= 1000  # Convert translation to mm

                return_dict[frame] = transformation_matrix

            except Exception as e:
                print(f"Foundationpose Error in frame {frame}: {e}")

    return return_dict


def read_blenderproc(input_file: str) -> Dict[int, np.ndarray]:
    """Reads pose data from BlenderProc output files (scene_gt.json).

    Args:
        input_file: Path to the scene_gt.json file.

    Returns:
        A dictionary where keys are frame numbers (int) and values are 4x4 transformation matrices (numpy arrays).
    """
    return_dict = {}

    with open(input_file, "r") as f:
        scene_gt = json.load(f)

    for key, value in scene_gt.items():
        frame = int(key)

        rotation_matrix = value[0]["cam_R_m2c"]
        rotation_matrix = [
            rotation_matrix[i : i + 3] for i in range(0, len(rotation_matrix), 3)
        ]
        translation_vector = value[0]["cam_t_m2c"]

        transformation_matrix = pt.transform_from(rotation_matrix, translation_vector)
        transformation_matrix = pt.check_transform(transformation_matrix)

        return_dict[frame] = transformation_matrix

    return return_dict


def calculate_accuracy(
    gt: Dict[int, np.ndarray], pose_estimation_poses: Dict[int, np.ndarray]
) -> Dict[str, float]:
    """Calculates accuracy metrics based on ground truth and pose estimation results.

    Args:
        gt: Ground truth poses (dictionary of frame: transformation matrix).
        pose_estimation_poses: Estimated poses (dictionary of frame: transformation matrix).

    Returns:
        A dictionary containing calculated accuracy metrics.
    """
    x = []
    y = []
    z = []
    a = []
    b = []
    c = []

    for frame, pose in pose_estimation_poses.items():
        if frame not in gt:
            print(f"Warning: Frame {frame} not found in ground truth. Skipping.")
            continue

        gt_pose = gt[frame]

        gt_rotation_matrix = gt_pose[:3, :3]
        gt_axis_angle = pr.compact_axis_angle_from_matrix(
            gt_rotation_matrix, check=False
        )
        gt_axis_angle_deg = np.degrees(gt_axis_angle)

        gt_translation = gt_pose[:3, 3]

        pose_rotation_matrix = pose[:3, :3]
        pose_estimation_angle = pr.compact_axis_angle_from_matrix(
            pose_rotation_matrix, check=False
        )
        pose_estimation_angle_deg = np.degrees(pose_estimation_angle)

        pose_estimation_translation = pose[:3, 3]

        x.append(pose_estimation_translation[0] - gt_translation[0])
        y.append(pose_estimation_translation[1] - gt_translation[1])
        z.append(pose_estimation_translation[2] - gt_translation[2])

        a.append(pose_estimation_angle_deg[0] - gt_axis_angle_deg[0])
        b.append(pose_estimation_angle_deg[1] - gt_axis_angle_deg[1])
        c.append(pose_estimation_angle_deg[2] - gt_axis_angle_deg[2])

    ap_x = np.mean(x) if x else 0
    ap_y = np.mean(y) if y else 0
    ap_z = np.mean(z) if z else 0

    ap_a = np.mean(a) if a else 0
    ap_b = np.mean(b) if b else 0
    ap_c = np.mean(c) if c else 0

    l = []
    for i in range(len(x)):
        error = np.sqrt((x[i] - ap_x) ** 2 + (y[i] - ap_y) ** 2 + (z[i] - ap_z) ** 2)
        l.append(error)

    l_mean = np.mean(l) if l else 0

    S_1 = (
        np.sqrt(np.sum([((li - l_mean) ** 2) for li in l]) / (len(l) - 1))
        if len(l) > 1
        else 0
    )

    RP_1 = l_mean + 3 * S_1

    S_a = (
        np.sqrt(np.sum([((ai - ap_a) ** 2) for ai in a]) / (len(a) - 1))
        if len(a) > 1
        else 0
    )
    RP_a = [S_a * 3, -1 * S_a * 3]
    S_b = (
        np.sqrt(np.sum([((bi - ap_b) ** 2) for bi in b]) / (len(b) - 1))
        if len(b) > 1
        else 0
    )
    RP_b = [S_b * 3, -1 * S_b * 3]
    S_c = (
        np.sqrt(np.sum([((ci - ap_c) ** 2) for ci in c]) / (len(c) - 1))
        if len(c) > 1
        else 0
    )
    RP_c = [S_c * 3, -1 * S_c * 3]

    return {
        "ap_x": ap_x,
        "ap_y": ap_y,
        "ap_z": ap_z,
        "ap_a": ap_a,
        "ap_b": ap_b,
        "ap_c": ap_c,
        "RP_1": RP_1,
        "RP_a": RP_a,
        "RP_b": RP_b,
        "RP_c": RP_c,
    }


def main(input_folder: str, run_level: bool) -> None:
    """Main function to evaluate pose estimation accuracy.

    Args:
        input_folder: Path to the input folder containing object and sensor data.
        run_level: Boolean flag indicating if the script is run at the object level or a level above.
    """

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

        subfolders = [
            [object_name, f.path] for f in os.scandir(input_folder) if f.is_dir()
        ]

    print(f"Subfolders: {subfolders}")

    accuracy = {}

    pose_methods = {
        "megapose_output": read_megapose,
        "sam6d_output": read_sam6d,
        "foundationpose_output": read_foundationpose,
        "foundationposeNoTracking_output": read_foundationpose,
    }

    for object_name, sensor_path in subfolders:
        print(f"Sensor: {sensor_path}")

        gt = read_blenderproc(
            os.path.join(sensor_path, "blenderproc/output/scene_gt.json")
        )

        for output_folder, pose_reader in pose_methods.items():
            output_path = os.path.join(sensor_path, output_folder)
            if os.path.exists(output_path):
                print(f"Evaluating {output_folder}...")
                try:
                    pose_estimation_poses = pose_reader(output_path)
                    accuracy_metrics = calculate_accuracy(gt, pose_estimation_poses)

                    accuracy[output_folder] = {
                        "object": object_name,
                        "sensor": sensor_path,
                        **accuracy_metrics,
                        "Average Position Error": [
                            accuracy_metrics["ap_x"],
                            accuracy_metrics["ap_y"],
                            accuracy_metrics["ap_z"],
                        ],
                        "Average Angle Error": [
                            accuracy_metrics["ap_a"],
                            accuracy_metrics["ap_b"],
                            accuracy_metrics["ap_c"],
                        ],
                        "RP": [
                            accuracy_metrics["RP_1"],
                            accuracy_metrics["RP_a"],
                            accuracy_metrics["RP_b"],
                            accuracy_metrics["RP_c"],
                        ],
                    }
                except Exception as e:
                    print(f"Error processing {output_folder}: {e}")

    accuracy_output_file = os.path.join(input_folder, "accuracy_HRC-Hub.json")
    with open(accuracy_output_file, "w") as f:
        json.dump(accuracy, f, indent=4)
    print(f"Accuracy results saved to {accuracy_output_file}")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate pose estimation accuracy using ground truth and estimated poses."
    )
    parser.add_argument("input_folder", help="Path to the input folder")
    parser.add_argument(
        "--run_level",
        action="store_true",
        help="Choose if wrapper is for run level above object level",
    )

    args = parser.parse_args()

    main(args.input_folder, args.run_level)
