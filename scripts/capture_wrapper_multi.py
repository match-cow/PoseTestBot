import os
import shutil
import subprocess
import time
from typing import List


def get_realsense_serials() -> List[str]:
    """Gets serial numbers of connected RealSense devices."""
    serials = []
    try:
        import pyrealsense2 as rs

        context = rs.context()
        for dev in context.query_devices():
            serials.append(dev.get_info(rs.camera_info.serial_number))
    except ImportError:
        print("pyrealsense2 is not installed. Skipping RealSense devices.")
    except Exception as e:
        print(f"Error getting RealSense serial numbers: {e}")
    return serials


def get_luxonis_serials() -> List[str]:
    """Gets serial numbers of connected Luxonis devices."""
    serials = []
    try:
        import depthai as dai

        for device in dai.Device.getAllAvailableDevices():
            serials.append(device.getMxId())
    except ImportError:
        print("depthai is not installed. Skipping Luxonis devices.")
    except Exception as e:
        print(f"Error getting Luxonis serial numbers: {e}")
    return serials


def get_user_input(prompt: str, default: str) -> str:
    """Gets user input with a default value."""
    return input(f"{prompt} ([{default}]): ") or default


def run_subprocess(command: List[str]) -> None:
    """Runs a subprocess and prints its output."""
    print(f"Running: {' '.join(command)}")
    subprocess.run(command)


def capture_run(
    output_path: str,
    run_name: str,
    script_dir: str,
    capture_luxonis: bool,
    capture_realsense: bool,
    capture_vel: int,
    capture_fps: int,
    resolution: str,
) -> None:
    """Captures data for a single run."""
    run_output_folder = os.path.join(output_path, run_name)

    if os.path.exists(run_output_folder):
        print(f"Output folder {run_output_folder} already exists.")
        capture_continue = get_user_input(
            "Do you want to overwrite the existing folder? (Y/n)", "Y"
        )
        if capture_continue.lower() != "y":
            return

        shutil.rmtree(run_output_folder)

    os.makedirs(run_output_folder)

    capture_processes = []
    if capture_luxonis:
        luxonis_serials = get_luxonis_serials()
        if luxonis_serials:
            luxonis_capture = os.path.join(
                script_dir, f"capture_luxonis_{resolution}.py"
            )
            for serial in luxonis_serials:
                output_folder = os.path.join(run_output_folder, f"luxonis_{serial}")
                os.makedirs(output_folder)
                p = subprocess.Popen(
                    [
                        "python",
                        luxonis_capture,
                        output_folder,
                        f"--fps={capture_fps}",
                        f"--device={serial}",
                    ]
                )
                capture_processes.append(p)

    if capture_realsense:
        realsense_serials = get_realsense_serials()
        if realsense_serials:
            realsense_capture = os.path.join(
                script_dir, f"capture_realsense_{resolution}.py"
            )
            for serial in realsense_serials:
                output_folder = os.path.join(run_output_folder, f"realsense_{serial}")
                os.makedirs(output_folder)
                p = subprocess.Popen(
                    [
                        "python",
                        realsense_capture,
                        output_folder,
                        f"--fps={capture_fps}",
                        f"--device={serial}",
                    ]
                )
                capture_processes.append(p)

    time.sleep(2)

    # Start pose_receiver with output_folder as argument
    pose_receiver = os.path.join(script_dir, "pose_receiver_udp_json.py")
    pose_receiver_process = subprocess.Popen(
        [
            "python",
            pose_receiver,
            run_output_folder,
            f"--capture_vel={capture_vel}",
        ]
    )

    # Wait for pose_receiver to finish
    pose_receiver_process.wait()
    # Wait for 2 second so that it is more likely that the last frame was processed
    time.sleep(2)

    # Terminate luxonis_capture and realsense_capture subprocesses
    for p in capture_processes:
        p.terminate()

    # Move pose data to each sensor folder for synchronization
    pose_data_path = os.path.join(run_output_folder, "pose_data.json")
    if os.path.exists(pose_data_path):
        for f in os.scandir(run_output_folder):
            if f.is_dir():
                shutil.copy(pose_data_path, f.path)

    print(f"Finished capturing {run_name}.")


def main():
    """Main function to handle capture process."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_path = get_user_input(
        "Enter output path", os.path.join(script_dir, "../working_data")
    )

    capture_autostart_input = get_user_input(
        "Do you want to start the capture process automatically? (y/N)", "n"
    )
    capture_autostart = capture_autostart_input.lower() == "y"

    if capture_autostart:
        capture_realsense = True
        capture_luxonis = True
        capture_vel = 100
        capture_fps = 6
        resolution = "720p"
    else:
        resolution = get_user_input("Enter resolution ([720p]/360p)", "720p")
        capture_vel = int(get_user_input("Enter capture velocity in % ([100])", "100"))
        capture_fps = int(get_user_input("Enter capture fps ([6]/15/30)", "6"))
        capture_realsense_input = get_user_input(
            "Do you want to capture realsense? (Y/n)", "Y"
        )
        capture_luxonis_input = get_user_input(
            "Do you want to capture luxonis? (Y/n)", "Y"
        )
        capture_realsense = capture_realsense_input.lower() == "y"
        capture_luxonis = capture_luxonis_input.lower() == "y"

    now = time.localtime()
    output_path = os.path.join(
        output_path,
        f"{time.strftime('%Y-%m-%d_%H-%M-%S', now)}_{resolution}_{capture_vel}_{capture_fps}",
    )

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_folders = []

    start_capture_process = get_user_input(
        "Do you want to start the capture process? (Y/n)", "Y"
    )
    if start_capture_process.lower() == "y":
        while True:
            run_name = get_user_input("Enter run name", "")
            if not run_name:
                print("run_name cannot be empty.")
                continue

            capture_run(
                output_path,
                run_name,
                script_dir,
                capture_luxonis,
                capture_realsense,
                capture_vel,
                capture_fps,
                resolution,
            )

            output_folders.append(os.path.join(output_path, run_name))

            capture_continue = get_user_input(
                "Do you want to capture another object? (Y/n)", "Y"
            )
            if capture_continue.lower() != "y":
                break
    else:
        output_folders = [f.path for f in os.scandir(output_path) if f.is_dir()]

    print("Finished capturing all objects.")

    if not capture_autostart:
        capture_process = get_user_input(
            "Do you want to continue with the capture data preparation process? (Y/n)",
            "Y",
        )
        if capture_process.lower() != "y":
            return

    capture_sync_and_sort = os.path.join(script_dir, "capture_sync_and_sort.py")
    aruco_pose_estimation = os.path.join(script_dir, "aruco_pose_estimation.py")
    blencerproc_prepare = os.path.join(script_dir, "blenderproc_prepare_multi.py")
    blenderproc_wrapper = os.path.join(script_dir, "blenderproc_wrapper_multi.py")
    blenderproc_render = os.path.join(
        script_dir, f"blenderproc_render_{resolution}_multi.py"
    )

    sync_data = os.path.join(script_dir, "default_data", "sync_data.json")
    camera_ee_transform = os.path.join(
        script_dir, "default_data", "camera_ee_transform.json"
    )
    object_models = os.path.join(os.path.dirname(script_dir), "object_models")

    if capture_autostart:
        run_aruco = False
        save_aruco_images = False
    else:
        save_aruco_images = False
        run_aruco_tmp = get_user_input(
            "Do you want to run aruco pose estimation? (Y/n)", "Y"
        )
        run_aruco = run_aruco_tmp.lower() == "y"
        if run_aruco:
            save_aruco_images_tmp = get_user_input(
                "Do you want to save aruco images? (Y/n)", "Y"
            )
            save_aruco_images = save_aruco_images_tmp.lower() == "y"

    for io in output_folders:
        print(f"Processing run folder: {io}")

        # Data for each sensor is in a separate subfolder.
        # Run sync and aruco estimation on each subfolder.
        sensor_folders = [f.path for f in os.scandir(io) if f.is_dir()]
        # If no sensors were used, no subfolders will be created.
        # In that case, run on the main folder.
        if not sensor_folders:
            sensor_folders.append(io)

        for folder in sensor_folders:
            print(f"Processing sensor folder: {folder}")
            run_subprocess(
                ["python", capture_sync_and_sort, folder, f"--sync_delta={sync_data}"]
            )

            if capture_autostart:
                run_subprocess(
                    ["python", aruco_pose_estimation, folder, "--save_images", "--quiet"]
                )
            elif run_aruco:
                if save_aruco_images:
                    run_subprocess(
                        ["python", aruco_pose_estimation, folder, "--save_images"]
                    )
                else:
                    run_subprocess(["python", aruco_pose_estimation, folder, "--quiet"])

        # These steps likely need data from all sensors, so run them on the main folder.
        if not capture_autostart:
            perform_blenderproc_prepare = get_user_input(
                "Do you want to perform the blenderproc_prepare step? (Y/n)", "Y"
            )
            if perform_blenderproc_prepare.lower() == "y":
                print("Starting blenderproc_prepare.")
                run_subprocess(
                    [
                        "python",
                        blencerproc_prepare,
                        io,
                        object_models,
                        camera_ee_transform,
                    ]
                )

        if not capture_autostart:
            start_render = get_user_input("Do you want to start rendering? (Y/n)", "Y")
            if start_render.lower() == "y":
                run_subprocess(
                    ["python", blenderproc_wrapper, io, blenderproc_render],
                )


if __name__ == "__main__":
    main()
