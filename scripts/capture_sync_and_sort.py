import argparse
import json
import os
import shutil

import pandas as pd
from tqdm import tqdm


def test_input_folder(input_folder: str) -> dict:
    """
    Validates the input folder (a sensor folder), ensuring it contains
    'rgb' and 'depth' directories.
    Args:
        input_folder: Path to the input folder.
    Returns:
        A dictionary containing paths to 'rgb' and 'depth' directories, if they exist.
        Returns None if checks fail.
    """
    if not os.path.isdir(input_folder):
        print("Error: input_folder is not a valid folder.")
        return None

    sensor_paths = {}
    rgb_path = os.path.join(input_folder, "rgb")
    depth_path = os.path.join(input_folder, "depth")

    if os.path.isdir(rgb_path):
        sensor_paths["rgb"] = rgb_path
    if os.path.isdir(depth_path):
        sensor_paths["depth"] = depth_path

    if "rgb" not in sensor_paths or "depth" not in sensor_paths:
        return None

    return sensor_paths


def find_motion_keyframes(poses_df: pd.DataFrame) -> dict:
    """
    Identifies the minimum and maximum frame numbers for each motion type
    based on the provided robot poses DataFrame.

    Args:
        poses_df: DataFrame containing robot poses with a 'motion' and 'framename' column.

    Returns:
        A dictionary where keys are motion types and values are dictionaries
        containing the 'min' and 'max' frame numbers for that motion.
    """
    keyframes = {}
    motion_groups = poses_df.groupby("motion")

    for motion, group in motion_groups:
        min_timestamp = group["framename"].min()
        max_timestamp = group["framename"].max()
        keyframes[motion] = {"min": min_timestamp, "max": max_timestamp}

    return keyframes


def process_sensor_data(
    sensor_folder: str,
    sensor_subfolders: dict,
    data: pd.DataFrame,
    keyframes: dict,
    sync_delta: int,
    dry_run: bool,
):
    """
    Processes image data for a specific sensor, synchronizing it with robot pose data,
    renaming images, and creating a JSON file containing the synchronized data.

    Args:
        sensor_folder: The path to the sensor folder.
        sensor_subfolders: Paths to the sensor's 'rgb' and 'depth' directories.
        data: DataFrame containing robot poses.
        keyframes: Dictionary of motion keyframes.
        sync_delta: Synchronization delta for the sensor.
        dry_run: If True, performs a dry run without modifying files.
    """
    print(f"Sensor folder: {sensor_folder}")

    sensor_output_dict = {}
    sensor_frame_counter = 0

    rgb_files = sorted(os.listdir(sensor_subfolders["rgb"]))

    first_frame = True
    previous_frame_ts = 0  # Initialize previous_frame_ts

    for image in tqdm(rgb_files):
        image_frame = int(os.path.splitext(image)[0])
        image_extension = os.path.splitext(image)[1]
        delayed_frame = image_frame - sync_delta

        image_motion = None
        for key, value in keyframes.items():
            if delayed_frame >= value["min"] and delayed_frame <= value["max"]:
                image_motion = key

        if image_motion is None:
            if not dry_run:
                os.remove(os.path.join(sensor_subfolders["rgb"], image))
                os.remove(os.path.join(sensor_subfolders["depth"], image))
        else:
            img_frame_counter_name = (
                str(sensor_frame_counter).zfill(6) + image_extension
            )

            if not dry_run:
                shutil.move(
                    os.path.join(sensor_subfolders["rgb"], image),
                    os.path.join(sensor_subfolders["rgb"], img_frame_counter_name),
                )
                shutil.move(
                    os.path.join(sensor_subfolders["depth"], image),
                    os.path.join(sensor_subfolders["depth"], img_frame_counter_name),
                )

            closest_pose = data.iloc[
                (data["framename"] - delayed_frame).abs().argsort()[:1]
            ]

            if first_frame:
                previous_frame_ts = image_frame
                frame_delta = 0
                first_frame = False
            else:
                frame_delta = image_frame - previous_frame_ts
                previous_frame_ts = image_frame

            sensor_output_dict[img_frame_counter_name] = {
                "motion": image_motion,
                "image_frame": image_frame,
                "delayed_frame": delayed_frame,
                "frame_delta": frame_delta,
                "robot_frame": int(closest_pose["framename"].values[0]),
                "robot_ee_pose": closest_pose["pose"].values[0],
            }

            sensor_frame_counter += 1

    if not dry_run:
        output_file = os.path.join(sensor_folder, "match_robot_ee_poses.json")
        with open(output_file, "w") as f:
            json.dump(sensor_output_dict, f, indent=4, default=str)


def copy_default_data(input_folder: str):
    """
    Copies default data files (camera_ee_transform.json and sync_data.json)
    from the 'default_data' folder to the specified input folder.

    Args:
        input_folder: The destination folder where the default data files will be copied.
    """
    script_folder = os.path.dirname(os.path.abspath(__file__))
    default_data_folder = os.path.join(script_folder, "default_data")

    if os.path.exists(default_data_folder):
        camera_ee_transform_src = os.path.join(
            default_data_folder, "camera_ee_transform.json"
        )
        sync_data_src = os.path.join(default_data_folder, "sync_data.json")
        camera_ee_transform_dst = os.path.join(input_folder, "camera_ee_transform.json")
        sync_data_dst = os.path.join(input_folder, "sync_data.json")

        shutil.copy(camera_ee_transform_src, camera_ee_transform_dst)
        shutil.copy(sync_data_src, sync_data_dst)


def main():
    """
    Main function to parse arguments, process sensor data, and synchronize
    image frames with robot poses.
    """
    parser = argparse.ArgumentParser(
        description="Process input folder, robot poses file, and dry run flag."
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the input folder containing subfolders with images.",
    )
    parser.add_argument(
        "--sync_delta",
        type=str,
        default=None,
        help="Path to the Sync Delta file for the sensor.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Enable dry run mode without moving files.",
    )

    args = parser.parse_args()

    input_folder = args.input_folder
    sync_delta_path = args.sync_delta
    dry_run = args.dry_run

    sync_delta = {}
    if sync_delta_path is None:
        print("Warning: No Sync Delta file provided, using 100ms as default value.")
        sync_delta = {"realsense": 100, "luxonis": 100}
    else:
        with open(sync_delta_path, "r") as file:
            sync_delta = json.load(file)

    sensor_paths = test_input_folder(input_folder)
    if sensor_paths is None:
        print("Error: Input folder is not a valid sensor data folder.")
        return

    sensor_name = os.path.basename(input_folder)
    sensor_type = sensor_name.split("_")[0]

    sensor_sync_delta = sync_delta.get(sensor_type)
    if sensor_sync_delta is None:
        print(f"Warning: No sync delta for {sensor_type}, using 100ms.")
        sensor_sync_delta = 100

    poses_file = os.path.join(input_folder, "raw_robot_ee_poses.json")
    data = pd.read_json(poses_file, orient="index")

    keyframes = find_motion_keyframes(data)

    process_sensor_data(
        input_folder,
        sensor_paths,
        data,
        keyframes,
        sensor_sync_delta,
        dry_run,
    )

    if not dry_run:
        copy_default_data(os.path.dirname(input_folder))


if __name__ == "__main__":
    main()
