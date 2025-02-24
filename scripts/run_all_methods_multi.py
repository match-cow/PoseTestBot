import argparse
import json
import logging
import os
import subprocess
import time
from typing import List

logger = logging.getLogger(__name__)


def run_cmd(args: str) -> bool:
    """Runs a command in the shell.

    Args:
        args: The command to run.

    Returns:
        True if the command ran successfully, False otherwise.
    """
    logger.info(f"Running command: {args}")

    # Split args to list
    args = args.split(" ")

    try:
        subprocess.run(
            args,
            cwd=os.getcwd(),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error: {e}")
        return False


def run_foundations(input_folder: str, script_dir: str, object_id: int) -> bool:
    """Runs the foundationpose_wrapper_multi.py script with different configurations.

    Args:
        input_folder: The path to the input folder.
        script_dir: The path to the script directory.
        object_id: The object ID.

    Returns:
        True if all commands ran successfully, False otherwise.
    """

    wrapper = "foundationpose_wrapper_multi.py"
    wrapper_path = os.path.join(script_dir, wrapper)

    est_refiner_list = [5]
    tracking_refiner_list = [2]

    # Iterate over tracking
    for tracking in ["y", "n"]:
        # Iterate over est_refiner
        for est_refiner in est_refiner_list:
            # Iterate over tracking_refiner
            for tracking_refiner in tracking_refiner_list:
                # Parse command
                cmd = f"python {wrapper_path} {input_folder} --no_tracking={tracking} --est_refine_iter={est_refiner} --track_refine_iter={tracking_refiner} --object_id={object_id}"
                logger.info(f"Running command: {cmd} at {time.ctime()}")
                if not run_cmd(cmd):
                    logger.error(f"Command failed: {cmd}")
                    return False

    return True


def run_megapose(input_folder: str, script_dir: str, roi_scale: float) -> bool:
    """Runs the megapose_wrapper.py script with different model configurations.

    Args:
        input_folder: The path to the input folder.
        script_dir: The path to the script directory.
        roi_scale: The ROI scale.

    Returns:
        True if all commands ran successfully, False otherwise.
    """
    models = [
        "megapose-1.0-RGB",
        # "megapose-1.0-RGBD",
        # "megapose-1.0-RGB-multi-hypothesis",
        # "megapose-1.0-RGB-multi-hypothesis-icp",
    ]

    wrapper = "megapose_wrapper.py"
    wrapper_path = os.path.join(script_dir, wrapper)

    for model in models:
        cmd = f"conda run -n megapose python {wrapper_path} {input_folder} --model={model} --ROI_scale={roi_scale}"
        logger.info(f"Running command: {cmd} at {time.ctime()}")
        if not run_cmd(cmd):
            logger.error(f"Command failed: {cmd}")
            return False

    return True


def run_sam6d(input_folder: str, script_dir: str) -> bool:
    """Runs the sam6d_wrapper.py script with different segmentation model configurations.

    Args:
        input_folder: The path to the input folder.
        script_dir: The path to the script directory.

    Returns:
        True if all commands ran successfully, False otherwise.
    """
    wrapper = "sam6d_wrapper.py"
    wrapper_path = os.path.join(script_dir, wrapper)

    ism_models = ["sam"]

    for ism_model in ism_models:
        cmd = f"conda run -n sam6d python {wrapper_path} {input_folder} --segmentor_model={ism_model}"
        logger.info(f"Running command: {cmd} at {time.ctime()}")
        if not run_cmd(cmd):
            logger.error(f"Command failed: {cmd}")
            return False

    return True


def get_subfolders(input_folder: str, run_level: bool) -> List[str]:
    """Determines the subfolders to process based on the run level.

    Args:
        input_folder: The path to the input folder.
        run_level: A flag indicating whether to run on subfolders or the input folder directly.

    Returns:
        A list of subfolder paths.
    """
    if run_level:
        object_folders = [f.path for f in os.scandir(input_folder) if f.is_dir()]
        return object_folders
    else:
        return [input_folder]


def main(input_folder: str, script_dir: str, run_level: bool, roi_scale: float):
    """Main function that orchestrates the execution of different pose estimation methods.

    Args:
        input_folder: The path to the input folder.
        script_dir: The path to the script directory.
        run_level: A flag indicating whether to run on subfolders or the input folder directly.
        roi_scale: The ROI scale for Megapose.
    """
    logging.basicConfig(filename="run_all.log", level=logging.INFO)
    logger.info(f"STARTING at {time.ctime()}")
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Run level: {run_level}")

    subfolders = get_subfolders(input_folder, run_level)

    logger.info(f"Subfolders: {subfolders}")

    for subfolder in subfolders:
        logger.info(f"Subfolder: {subfolder}")

        # Name of subfolder is run name
        run_name = os.path.basename(subfolder)

        # Open the objects.json file from blenderproc subfolder to get the objects
        objects_json_path = os.path.join(
            subfolder, "realsense", "blenderproc", "objects.json"
        )
        if not os.path.exists(objects_json_path):
            raise FileNotFoundError(f"Error: file {objects_json_path} not found")

        with open(objects_json_path, "r") as f:
            objects_json = json.load(f)

        # Iterate over objects
        for object_id, object in enumerate(objects_json.keys()):
            logger.info(f"Object: {object}")
            logger.info(f"Object id: {object_id}")

            if not run_foundations(subfolder, script_dir, object_id):
                logger.error("Foundations failed on some methods")
                continue
            else:
                logger.info("Foundations ran successfully on all methods")

            # TODO: Implement multi for megapose and sam6d if required
            # if run_megapose(subfolder, script_dir, roi_scale, object_id):
            #     logger.info("Megapose ran successfully on all methods")
            # else:
            #     logger.error("Megapose failed on some methods", object_id)

            # if run_sam6d(subfolder, script_dir):
            #     logger.info("SAM6D ran successfully on all methods")
            # else:
            #     logger.error("SAM6D failed on some methods")

    logger.info(f"FINISHED at {time.ctime()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", help="Path to the input folder")
    parser.add_argument(
        "--run_level",
        action="store_true",
        help="If set, runs on subfolders of input_folder. Otherwise, runs on input_folder directly.",
    )
    parser.add_argument(
        "--roi_scale", type=float, default=1.0, help="ROI scale for Megapose"
    )
    args = parser.parse_args()

    # Get current dir of script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    input_folder = args.input_folder
    run_level = args.run_level
    roi_scale = args.roi_scale
    main(input_folder, script_dir, run_level, roi_scale)
