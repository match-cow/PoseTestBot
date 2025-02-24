import blenderproc as bproc  # isort:skip

import argparse
import json
import os

import numpy as np


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Renders images of 3D objects using BlenderProc."
    )
    parser.add_argument("poses_file", help="Path to camera poses file (.npy)")
    parser.add_argument("camera_matrix", help="Path to camera matrix file (.npy)")
    parser.add_argument("output_dir", help="Path to output directory")
    return parser.parse_args()


def load_object(object_dir, object_name):
    """
    Loads a 3D object from a PLY file and sets its initial pose and scale.

    Args:
        object_dir (str): Path to the directory containing the object files.
        object_name (str): Name of the object (without extension).

    Returns:
        bproc.types.MeshObject: The loaded object.
    """
    object_path = os.path.join(object_dir, f"{object_name}.ply")
    obj2template_path = os.path.join(object_dir, f"{object_name}.npy")

    print(f"Loading object: {object_name}")
    print(f"Object Path: {object_path}")
    print(f"Object2Template Path: {obj2template_path}")

    obj = bproc.loader.load_obj(object_path)[0]
    print(f"Object: {obj}")
    print(f"Type: {type(obj)}")

    for mat in obj.get_materials():
        mat.map_vertex_color()

    object2template = np.load(obj2template_path)
    obj.set_local2world_mat(object2template)
    obj.set_scale([0.001, 0.001, 0.001])
    obj.set_cp("category_id", object_name)
    return obj


def load_objects(objects_dir, objects_json):
    """
    Loads multiple 3D objects based on a JSON configuration file.

    Args:
        objects_dir (str): Path to the directory containing the object files.
        objects_json (dict): A dictionary containing object names as keys.

    Returns:
        list: A list of loaded bproc.types.MeshObject objects.
    """
    objects_list = [
        load_object(objects_dir, object_name) for object_name in objects_json.keys()
    ]
    return objects_list


def setup_light(location=[1, -1, 1], energy=500):
    """
    Sets up a point light in the scene.

    Args:
        location (list, optional): The location of the light source. Defaults to [1, -1, 1].
        energy (int, optional): The energy/intensity of the light source. Defaults to 500.
    """
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location(location)
    light.set_energy(energy)


def setup_camera(
    camera_matrix_path, poses_file_path, resolution_x=1280, resolution_y=720
):
    """
    Sets up the camera intrinsics and poses.

    Args:
        camera_matrix_path (str): Path to the camera matrix file (.npy).
        poses_file_path (str): Path to the camera poses file (.npy).
        resolution_x (int, optional): The horizontal resolution of the rendered images. Defaults to 1280.
        resolution_y (int, optional): The vertical resolution of the rendered images. Defaults to 720.
    """
    camera_matrix = np.load(camera_matrix_path)
    bproc.camera.set_intrinsics_from_K_matrix(camera_matrix, resolution_x, resolution_y)

    camera_poses = np.load(poses_file_path)
    for camera_pose in camera_poses:
        cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(
            camera_pose, ["X", "-Y", "-Z"]
        )
        bproc.camera.add_camera_pose(cam2world)


def render_and_write(output_dir, objects_list):
    """
    Renders the scene and writes the output to BOP format.

    Args:
        output_dir (str): Path to the output directory.
        objects_list (list): A list of bproc.types.MeshObject objects in the scene.
    """
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_output_format(enable_transparency=True)

    data = bproc.renderer.render()

    bproc.writer.write_bop(
        output_dir,
        objects_list,
        data["depth"],
        data["colors"],
        annotation_unit="mm",
        frames_per_chunk=999999,
    )


def main():
    """
    Main function to set up the scene, load objects, set up camera and lighting, and render the scene.
    """
    args = parse_arguments()
    bproc.init()

    objects_json_path = os.path.join(args.output_dir, "objects.json")
    if not os.path.exists(objects_json_path):
        raise FileNotFoundError(f"Error: file {objects_json_path} not found")

    with open(objects_json_path, "r") as f:
        objects_json = json.load(f)

    objects_dir = os.path.join(args.output_dir, "objects")
    objects_list = load_objects(objects_dir, objects_json)

    setup_light()
    setup_camera(args.camera_matrix, args.poses_file)

    render_and_write(args.output_dir, objects_list)


if __name__ == "__main__":
    main()
