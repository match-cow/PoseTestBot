import argparse
import json
import os

import numpy as np
import pandas as pd
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager
from tqdm import tqdm


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file", help="Path to the input json file with the ArUco poses"
    )
    args = parser.parse_args()

    # Access the arguments
    input_file = args.input_file

    # Check if input file exists
    if not os.path.isfile(input_file):
        print("Input file does not exist.")
        exit()

    # try to load data_dict from json file)
    # Load the dict from the json file
    with open(input_file, "r") as f:
        data = json.load(f)

    # TODO: This is static for now, but it should be read from a template config?
    aruco2template = pt.transform_from(
        pr.active_matrix_from_angle(0, np.deg2rad(180.0)),
        np.array([-199.5, 137.0, 0.0]),
    )

    tm = TransformManager()
    tm.add_transform("aruco", "template", aruco2template)

    for frame, d in tqdm(data.items()):
        # print(f"f: {frame}")
        # print(f"d: {d}")

        len_ids = d["aruco_pose_estimation"]["len_ids"]

        # TODO: Can len ids still be null or not exist?

        ee2template = pt.transform_from(
            pr.matrix_from_euler(
                np.array(
                    [
                        d["robot_ee_pose"]["C"],
                        d["robot_ee_pose"]["B"],
                        d["robot_ee_pose"]["A"],
                    ]
                ),
                0,
                1,
                2,
                True,
            ),
            np.array(
                [
                    d["robot_ee_pose"]["X"],
                    d["robot_ee_pose"]["Y"],
                    d["robot_ee_pose"]["Z"],
                ]
            ),
        )

        aruco2sensor = pt.transform_from(
            pr.matrix_from_compact_axis_angle(
                np.array(d["aruco_pose_estimation"]["rvec"]),
            ),
            np.array(d["aruco_pose_estimation"]["tvec"]),
        )

        tm.add_transform("end-effector", "template", ee2template)
        tm.add_transform("aruco", "sensor", aruco2sensor)

        sensor2ee = tm.get_transform("sensor", "end-effector")

        # write sensor2ee matrix to list of lists
        sensor2ee = sensor2ee.tolist()

        data[frame].update({"sensor_to_ee": sensor2ee})

    # Save the updated dict to a new json file
    output_file = input_file.replace(".json", "_with_sensor_to_ee.json")
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4, default=str)


if __name__ == "__main__":
    main()
