import argparse
import json
import os
import shutil
from typing import Dict, Tuple, Union

import numpy as np
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager
from tqdm import tqdm


def read_camera_parameters(input_folder: str) -> Tuple[np.ndarray, np.ndarray]:
    """Reads camera matrix and distortion coefficients from a text file.

    Args:
        input_folder: Path to the folder containing cam_K.txt.

    Returns:
        A tuple containing the camera matrix and distortion coefficients.

    Raises:
        FileNotFoundError: If cam_K.txt is not found in the input folder.
    """
    filename = os.path.join(input_folder, "cam_K.txt")
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Error: file {filename} not found")

    with open(filename, "r") as f:
        lines = f.readlines()
        cam_matrix = np.array([list(map(float, line.split())) for line in lines[:3]])
        dist_coefficients = np.array([float(x) for x in lines[3].split()]).reshape(5, 1)

    return cam_matrix, dist_coefficients


def load_json_file(filepath: str) -> Union[Dict, None]:
    """Loads a JSON file from the given filepath.

    Args:
        filepath: Path to the JSON file.

    Returns:
        A dictionary containing the JSON data, or None if the file does not exist.
    """
    if os.path.isfile(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None


def create_directory(path: str) -> None:
    """Creates a directory if it does not already exist.

    Args:
        path: The path to the directory to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def save_numpy_array(filepath: str, array: np.ndarray) -> None:
    """Saves a NumPy array to a file.

    Args:
        filepath: The path to the file to save the array to.
        array: The NumPy array to save.
    """
    with open(filepath, "wb") as f:
        np.save(f, array)


def copy_object_files(
    object_name: str, object_folder: str, subfolder: str, subdir: str
) -> None:
    """Copies the object model and texture files to the specified subdirectory.

    Args:
        object_name: The name of the object.
        object_folder: The path to the folder containing the object files.
        subfolder: The parent folder for the subdirectory.
        subdir: The name of the subdirectory to copy the files to.

    Raises:
        SystemExit: If the object model file does not exist.
    """
    object_model = os.path.join(object_folder, f"{object_name}.ply")
    if not os.path.exists(object_model):
        print(f"Object model {object_model} does not exist.")
        exit()

    object_model_subdir = os.path.join(
        subfolder, subdir, "objects", f"{object_name}.ply"
    )
    shutil.copy(object_model, object_model_subdir)

    object_texture = os.path.join(object_folder, f"{object_name}.png")
    if os.path.exists(object_texture):
        object_texture_subdir = os.path.join(
            subfolder, subdir, "objects", f"{object_name}.png"
        )
        shutil.copy(object_texture, object_texture_subdir)


def process_object(
    object_name: str,
    object_folder: str,
    subfolder: str,
    subdir: str,
    cam_matrix: np.ndarray,
    dist_coefficients: np.ndarray,
    object2template: np.ndarray,
    camera2ee: np.ndarray,
) -> None:
    """Processes a single object, saving camera parameters, object transforms, and camera poses.

    Args:
        object_name: The name of the object.
        object_folder: The path to the folder containing the object files.
        subfolder: The parent folder for the subdirectory.
        subdir: The name of the subdirectory to save the data to.
        cam_matrix: The camera matrix.
        dist_coefficients: The distortion coefficients.
        object2template: The transformation matrix from object to template.
        camera2ee: The transformation matrix from camera to end-effector.

    Raises:
        SystemExit: If the match_robot_ee_poses.json file does not exist.
    """
    copy_object_files(object_name, object_folder, subfolder, subdir)

    save_numpy_array(os.path.join(subfolder, subdir, "camera_matrix.npy"), cam_matrix)
    save_numpy_array(
        os.path.join(subfolder, subdir, "dist_coefficients.npy"), dist_coefficients
    )
    save_numpy_array(
        os.path.join(subfolder, subdir, "objects", f"{object_name}.npy"),
        object2template,
    )

    match_robot_ee_poses_path = os.path.join(subfolder, "match_robot_ee_poses.json")
    if not os.path.exists(match_robot_ee_poses_path):
        print(f"match_robot_ee_poses.json does not exist in {subfolder}")
        exit()

    with open(match_robot_ee_poses_path, "r") as f:
        data = json.load(f)

    camera_poses = []
    for d in tqdm(data.values()):
        robot_ee_pose = d["robot_ee_pose"]
        ee2template = pt.transform_from(
            pr.matrix_from_euler(
                [robot_ee_pose["C"], robot_ee_pose["B"], robot_ee_pose["A"]],
                0,
                1,
                2,
                True,
            ),
            [robot_ee_pose["X"], robot_ee_pose["Y"], robot_ee_pose["Z"]],
        )

        tm = TransformManager()
        tm.add_transform("ee", "template", ee2template)
        tm.add_transform("camera", "ee", camera2ee)

        cam2template = tm.get_transform("camera", "template")
        cam2template_m = cam2template.copy()
        cam2template_m[:3, 3] /= 1000

        camera_poses.append(cam2template_m)

    save_numpy_array(os.path.join(subfolder, subdir, "camera_poses.npy"), camera_poses)


def main():
    """Main function to parse arguments and process objects."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_folder",
        help="Path to the input folder containing subfolders with sensors",
    )
    parser.add_argument(
        "object_folder",
        help="Path to the folder containing image 3d files and objects.json with transformations",
    )
    parser.add_argument(
        "camera_transformations",
        help="Path to the json file containing camera transformations",
    )
    parser.add_argument(
        "subdir",
        nargs="?",
        default="blenderproc",
        help="Subdirectory where the camera parameters will be saved",
    )
    args = parser.parse_args()

    input_folder = args.input_folder
    object_folder = args.object_folder
    camera_transformations = args.camera_transformations
    subdir = args.subdir

    subfolders = (
        [f.path for f in os.scandir(input_folder) if f.is_dir()]
        if os.path.exists(input_folder)
        else []
    )
    objects = load_json_file(os.path.join(object_folder, "objects.json"))
    camera_transforms = load_json_file(camera_transformations)

    if objects is None or camera_transforms is None:
        print("Error: objects.json or camera_transformations.json not found.")
        exit()

    for object_name, ob2t in objects.items():
        ob2t[:3, 3] *= 0.001  # Scale object to meters
        object2template = pt.invert_transform(ob2t)

        for subfolder in subfolders:
            sensor = os.path.basename(subfolder)
            print(f"sensor: {sensor}")

            cam_matrix, dist_coefficients = read_camera_parameters(subfolder)
            create_directory(os.path.join(subfolder, subdir, "objects"))
            shutil.copy(
                os.path.join(object_folder, "objects.json"),
                os.path.join(subfolder, subdir),
            )

            camera_transformation = camera_transforms[sensor]
            camera2ee = pt.transform_from(
                pr.matrix_from_quaternion(camera_transformation["quaternion"]),
                np.array(camera_transformation["position"]),
            )

            process_object(
                object_name,
                object_folder,
                subfolder,
                subdir,
                cam_matrix,
                dist_coefficients,
                object2template,
                camera2ee,
            )


if __name__ == "__main__":
    main()
