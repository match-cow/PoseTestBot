import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from pytransform3d.transform_manager import TransformManager
from tqdm import tqdm


def load_json(input_file: str) -> Dict:
    """Loads data from a JSON file.

    Args:
        input_file (str): Path to the JSON file.

    Returns:
        Dict: A dictionary containing the loaded data.
    """
    with open(input_file, "r") as f:
        data = json.load(f)
    return data


def create_aruco_to_template_transform() -> np.ndarray:
    """Creates a static transformation matrix from ArUco marker to template.

    Returns:
        np.ndarray: A 4x4 NumPy array representing the transformation matrix.
    """
    # TODO: This is static for now, but it should be read from a template config?
    return pt.transform_from(
        pr.active_matrix_from_angle(0, np.deg2rad(180.0)),
        np.array([-199.5, 137.0, 0.0]),
    )


def create_ee_to_template_transform(ee_pose: Dict) -> np.ndarray:
    """Creates a transformation matrix from end-effector to template.

    Args:
        ee_pose (Dict): A dictionary containing the end-effector pose data
            with keys 'X', 'Y', 'Z', 'A', 'B', and 'C'.

    Returns:
        np.ndarray: A 4x4 NumPy array representing the transformation matrix.
    """
    rotation_matrix = pr.matrix_from_euler(
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
    )
    translation_vector = np.array(
        [
            ee_pose["X"],
            ee_pose["Y"],
            ee_pose["Z"],
        ]
    )
    return pt.transform_from(rotation_matrix, translation_vector)


def create_aruco_to_sensor_transform(aruco_pose_estimation: Dict) -> np.ndarray:
    """Creates a transformation matrix from ArUco marker to sensor.

    Args:
        aruco_pose_estimation (Dict): A dictionary containing the ArUco pose estimation
            data with keys 'rvec' and 'tvec'.

    Returns:
        np.ndarray: A 4x4 NumPy array representing the transformation matrix.
    """
    rotation_matrix = pr.matrix_from_compact_axis_angle(
        np.array(aruco_pose_estimation["rvec"]),
    )
    translation_vector = np.array(aruco_pose_estimation["tvec"])
    return pt.transform_from(rotation_matrix, translation_vector)


def compute_sensor_to_ee_transform(
    ee2template: np.ndarray, aruco2sensor: np.ndarray
) -> List[List[float]]:
    """Computes the transformation matrix from sensor to end-effector.

    Args:
        ee2template (np.ndarray): Transformation matrix from end-effector to template.
        aruco2sensor (np.ndarray): Transformation matrix from ArUco marker to sensor.

    Returns:
        List[List[float]]: A list of lists representing the sensor to end-effector
            transformation matrix.
    """
    tm = TransformManager()
    tm.add_transform("end-effector", "template", ee2template)
    tm.add_transform("aruco", "sensor", aruco2sensor)
    sensor2ee = tm.get_transform("sensor", "end-effector")
    return sensor2ee.tolist()


def process_data(data: Dict) -> Dict:
    """Processes the input data to compute and add sensor_to_ee transformations.

    Args:
        data (Dict): A dictionary containing the input data.

    Returns:
        Dict: A dictionary containing the updated data with sensor_to_ee
            transformations.
    """
    aruco2template = create_aruco_to_template_transform()

    for frame, d in tqdm(data.items(), desc="Processing frames"):
        ee2template = create_ee_to_template_transform(d["robot_ee_pose"])
        aruco2sensor = create_aruco_to_sensor_transform(d["aruco_pose_estimation"])

        sensor2ee = compute_sensor_to_ee_transform(ee2template, aruco2sensor)
        data[frame].update({"sensor_to_ee": sensor2ee})

    return data


def save_json(data: Dict, input_file: str) -> None:
    """Saves the updated data to a new JSON file.

    Args:
        data (Dict): A dictionary containing the updated data.
        input_file (str): Path to the input JSON file.
    """
    output_file = input_file.replace(".json", "_with_sensor_to_ee.json")
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4, default=str)


def main():
    """Main function to execute the script."""
    parser = argparse.ArgumentParser(
        description="Compute and add sensor_to_ee transformation to ArUco pose data."
    )
    parser.add_argument(
        "input_file", help="Path to the input json file with the ArUco poses"
    )
    args = parser.parse_args()

    input_file = args.input_file

    if not os.path.isfile(input_file):
        raise FileNotFoundError("Input file does not exist.")

    data = load_json(input_file)
    updated_data = process_data(data)
    save_json(updated_data, input_file)


if __name__ == "__main__":
    main()
