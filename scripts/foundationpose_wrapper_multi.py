import argparse
import json
import os
import shutil
import subprocess
from typing import Any, Dict, List

import trimesh


def run_docker_command(command: List[str]) -> None:
    """Runs a docker command and handles potential errors.

    Args:
        command: The docker command to execute.

    Raises:
        subprocess.CalledProcessError: If the docker command returns a non-zero exit code.
    """
    try:
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error executing docker command: {e}")
        raise


def prepare_object_cad(cad_path_mm: str, cad_path: str) -> None:
    """Loads a CAD model, scales it from meters to millimeters, and saves it.

    Args:
        cad_path_mm: Path to the CAD model in meters.
        cad_path: Path to save the CAD model in millimeters.
    """
    mesh = trimesh.load(cad_path_mm, process=False)
    mesh.apply_scale(0.001)
    mesh.export(cad_path)


def copy_necessary_files(sensor_folder: str, temp_folder: str, object: str) -> None:
    """Copies the CAD model, its corresponding PNG, and camera intrinsics to a temporary folder.

    Args:
        sensor_folder: Path to the sensor data folder.
        temp_folder: Path to the temporary folder.
        object: Name of the object.
    """
    # Copy CAD model
    cad_source_path = os.path.join(
        sensor_folder, "blenderproc", "objects", f"{object}.ply"
    )
    cad_destination_path = os.path.join(temp_folder, f"{object}.ply")
    shutil.copy(cad_source_path, cad_destination_path)

    # Copy object PNG if it exists
    object_png_path = os.path.join(
        sensor_folder, "blenderproc", "objects", f"{object}.png"
    )
    if os.path.exists(object_png_path):
        shutil.copy(object_png_path, os.path.join(temp_folder, f"{object}.png"))

    # Copy camera intrinsics
    shutil.copy(
        os.path.join(sensor_folder, "cam_K.txt"),
        os.path.join(temp_folder, "cam_K.txt"),
    )


def run_foundationpose_on_motion(
    foundationpose_folder: str,
    run_demo_file: str,
    cad_path: str,
    temp_folder: str,
    est_refine_iter: int,
    track_refine_iter: int,
    temp_output_folder: str,
) -> None:
    """Runs FoundationPose on a specific motion using Docker.

    Args:
        foundationpose_folder: Path to the FoundationPose directory.
        run_demo_file: Path to the FoundationPose run script.
        cad_path: Path to the CAD model.
        temp_folder: Path to the temporary folder containing input data.
        est_refine_iter: Number of estimation refinement iterations.
        track_refine_iter: Number of tracking refinement iterations.
        temp_output_folder: Path to the temporary output folder.

     Raises:
        subprocess.CalledProcessError: If the docker command returns a non-zero exit code.
    """
    command = [
        "docker",
        "exec",
        "-it",
        "foundationpose",
        "python",
        run_demo_file,
        f"--mesh_file={cad_path}",
        f"--test_scene_dir={temp_folder}",
        f"--est_refine_iter={est_refine_iter}",
        f"--track_refine_iter={track_refine_iter}",
        f"--debug=1",
        f"--debug_dir={temp_output_folder}",
    ]
    try:
        subprocess.run(
            command,
            cwd=foundationpose_folder,
            stderr=subprocess.STDOUT,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running FoundationPose in Docker: {e}")
        raise


def copy_output_files(temp_output_folder: str, output_folder: str) -> None:
    """Copies the output files from the temporary output folder to the final output folder.

    Args:
        temp_output_folder: Path to the temporary output folder.
        output_folder: Path to the final output folder.
    """
    output_subfolders = ["ob_in_cam", "track_vis"]

    for output_subfolder in output_subfolders:
        output_subfolder_path = os.path.join(temp_output_folder, output_subfolder)
        os.makedirs(os.path.join(output_folder, output_subfolder), exist_ok=True)

        if os.path.exists(output_subfolder_path):
            for file in os.listdir(output_subfolder_path):
                shutil.copy(
                    os.path.join(output_subfolder_path, file),
                    os.path.join(output_folder, output_subfolder, file),
                )


def main(
    input_folder: str,
    foundationpose_folder: str,
    no_tracking: bool,
    run_level: bool,
    est_refine_iter: int,
    track_refine_iter: int,
    object_id: int,
) -> None:
    """Main function to run FoundationPose on a dataset.

    Args:
        input_folder: Path to the input folder containing sensor data.
        foundationpose_folder: Path to the FoundationPose directory.
        no_tracking: Whether to disable tracking in FoundationPose.
        run_level: Whether the input folder is at the run level (contains multiple runs).
        est_refine_iter: Number of estimation refinement iterations.
        track_refine_iter: Number of tracking refinement iterations.
        object_id: ID of the object to process.
    """

    if run_level:
        run_folders = [f.path for f in os.scandir(input_folder) if f.is_dir()]
        subfolders = []
        for run_folder in run_folders:
            run_name = os.path.basename(run_folder).split("-")[0]
            [
                subfolders.append([run_name, f.path])
                for f in os.scandir(run_folder)
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
        objects_json_path = os.path.join(sensor, "blenderproc", "objects.json")
        if not os.path.exists(objects_json_path):
            raise FileNotFoundError(f"Error: file {objects_json_path} not found")

        with open(objects_json_path, "r") as f:
            objects_json = json.load(f)

        object = list(objects_json.keys())[object_id]
        print(f"Object: {object}")

        print(f"Running FoundationPose for run {run} and sensor {sensor} ...")

        sensor_folder = sensor
        output_folder = os.path.join(
            sensor_folder,
            f"foundationpose{'' if not no_tracking else 'NoTracking'}_est{est_refine_iter}_track{track_refine_iter}_obj{object_id}_output",
        )

        temp_folder = os.path.join(sensor_folder, "foundationpose_temp")
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(temp_folder, exist_ok=True)

        temp_output_folder = os.path.join(temp_folder, "output")
        os.makedirs(temp_output_folder, exist_ok=True)

        cad_path_mm = os.path.join(
            sensor_folder, "blenderproc", "objects", f"{object}.ply"
        )
        cad_path = os.path.join(temp_folder, f"{object}.ply")

        mesh = trimesh.load(
            cad_path_mm,
            process=False,
        )
        mesh.apply_scale(0.001)
        mesh.export(cad_path)

        if os.path.exists(
            os.path.join(sensor_folder, "blenderproc", "objects", f"{object}.png")
        ):
            shutil.copy(
                os.path.join(sensor_folder, "blenderproc", "objects", f"{object}.png"),
                os.path.join(temp_folder, f"{object}.png"),
            )

        shutil.copy(
            os.path.join(sensor_folder, "cam_K.txt"),
            os.path.join(temp_folder, "cam_K.txt"),
        )

        run_demo_file = (
            os.path.join(foundationpose_folder, "run_demo_no_tracking.py")
            if no_tracking
            else os.path.join(foundationpose_folder, "run_demo.py")
        )

        try:
            subprocess.run(
                [
                    "docker",
                    "start",
                    "foundationpose",
                ],
                stdout=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            continue

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

        for motion, frames in motions_frames.items():
            temp_rgb_folder = os.path.join(temp_folder, f"rgb")
            temp_depth_folder = os.path.join(temp_folder, f"depth")
            temp_masks_folder = os.path.join(temp_folder, f"masks")

            os.makedirs(temp_rgb_folder, exist_ok=True)
            os.makedirs(temp_depth_folder, exist_ok=True)
            os.makedirs(temp_masks_folder, exist_ok=True)

            for frame in frames:
                frame_rgb = os.path.join(sensor_folder, "rgb", frame)
                frame_depth = os.path.join(sensor_folder, "depth", frame)
                object_frame = frame.split(".")[0] + f"_{object_id:06d}.png"
                frame_masks = os.path.join(sensor_folder, "masks", object_frame)

                shutil.copy(
                    frame_rgb,
                    os.path.join(temp_rgb_folder, frame),
                )
                shutil.copy(
                    frame_depth,
                    os.path.join(temp_depth_folder, frame),
                )
                shutil.copy(
                    frame_masks,
                    os.path.join(temp_masks_folder, frame),
                )

            print(f"Running FoundationPose for motion: {motion}...")

            try:
                subprocess.run(
                    [
                        "docker",
                        "exec",
                        "-it",
                        "foundationpose",
                        "python",
                        run_demo_file,
                        f"--mesh_file={cad_path}",
                        f"--test_scene_dir={temp_folder}",
                        f"--est_refine_iter={est_refine_iter}",
                        f"--track_refine_iter={track_refine_iter}",
                        f"--debug=1",
                        f"--debug_dir={temp_output_folder}",
                    ],
                    cwd=foundationpose_folder,
                    stderr=subprocess.STDOUT,
                )
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")
                continue

            shutil.rmtree(temp_rgb_folder)
            shutil.rmtree(temp_depth_folder)
            shutil.rmtree(temp_masks_folder)

            output_subfolders = ["ob_in_cam", "track_vis"]

            for output_subfolder in output_subfolders:
                output_subfolder_path = os.path.join(
                    temp_output_folder, output_subfolder
                )
                os.makedirs(
                    os.path.join(output_folder, output_subfolder), exist_ok=True
                )

                if os.path.exists(output_subfolder_path):
                    for file in os.listdir(output_subfolder_path):
                        shutil.copy(
                            os.path.join(output_subfolder_path, file),
                            os.path.join(output_folder, output_subfolder, file),
                        )

        print(f"FoundationPose finished for object {object} and sensor {sensor}.")
        try:
            subprocess.run(
                ["docker", "exec", "-it", "foundationpose", "rm", "-rf", temp_folder],
                stdout=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            continue

        try:
            subprocess.run(
                [
                    "docker",
                    "stop",
                    "foundationpose",
                ],
                stdout=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            continue

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run FoundationPose on a dataset with multiple objects and motions."
    )
    parser.add_argument("input_folder", help="Path to the input folder")
    parser.add_argument(
        "--foundationpose_folder",
        help="Path to the foundationpose folder",
        default=f"{os.path.expanduser('~')}/FoundationPose",
    )
    parser.add_argument(
        "--no_tracking",
        type=str,
        default="n",
        help="Choose if tracking should be disabled [y/n], default is n",
    )
    parser.add_argument(
        "--run_level",
        action="store_true",
        help="Choose if wrapper is for run level above object level",
    )
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument(
        "--object_id",
        type=int,
        default=0,
        help="Object ID to be used for the foundationpose wrapper",
    )
    args = parser.parse_args()

    args.no_tracking = args.no_tracking.lower() == "y"

    main(
        args.input_folder,
        args.foundationpose_folder,
        args.no_tracking,
        args.run_level,
        args.est_refine_iter,
        args.track_refine_iter,
        args.object_id,
    )
