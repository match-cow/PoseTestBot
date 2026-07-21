import blenderproc as bproc  # isort:skip

import argparse
import importlib.metadata
import json
import os

import numpy as np


SUPPORTED_BLENDERPROC_VERSION = "2.8.0"


def validated_blenderproc_version(*, pose_template):
    """Reject unqualified renderer versions before producing GT evidence."""
    version = importlib.metadata.version("blenderproc")
    if pose_template and version != SUPPORTED_BLENDERPROC_VERSION:
        raise RuntimeError(
            "Pose-template instance GT is validated only with BlenderProc "
            f"{SUPPORTED_BLENDERPROC_VERSION}; found {version}"
        )
    return version


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


def load_object(object_dir, object_name, *, obj_id=None, instance_uuid=None, mesh=None, transform=None):
    """
    Loads a 3D object from a PLY file and sets its initial pose and scale.

    Args:
        object_dir (str): Path to the directory containing the object files.
        object_name (str): Name of the object (without extension).

    Returns:
        bproc.types.MeshObject: The loaded object.
    """
    object_path = os.path.join(object_dir, mesh or f"{object_name}.ply")
    obj2template_path = os.path.join(object_dir, transform or f"{object_name}.npy")

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
    obj.set_cp("category_id", obj_id if obj_id is not None else object_name)
    if instance_uuid is not None:
        obj.set_cp("posetestbot_instance_uuid", instance_uuid)
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
    if objects_json.get("schema_version") == "blenderproc_object_instances.v1":
        objects_list = [
            load_object(
                objects_dir,
                item["name"],
                obj_id=int(item["obj_id"]),
                instance_uuid=item["instance_uuid"],
                mesh=item["mesh"],
                transform=item["transform"],
            )
            for item in objects_json["instances"]
        ]
    else:
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


def render_and_write(output_dir, objects_list, instance_records=None, *, blenderproc_version=None):
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
    if instance_records is not None:
        scene_dir = os.path.join(output_dir, "train_pbr", "000000")
        with open(os.path.join(scene_dir, "scene_gt.json"), "r") as handle:
            scene_gt = json.load(handle)
        frames = {}
        for image_id, annotations in scene_gt.items():
            if len(annotations) != len(instance_records):
                raise RuntimeError(
                    f"BlenderProc {blenderproc_version} wrote {len(annotations)} GT "
                    f"annotations for {len(instance_records)} loaded instances in frame {image_id}"
                )
            frame_instances = []
            for gt_id, (annotation, instance) in enumerate(zip(annotations, instance_records)):
                if int(annotation["obj_id"]) != int(instance["obj_id"]):
                    raise RuntimeError(
                        "BlenderProc BOP annotation order does not preserve the validated "
                        f"instance order in frame {image_id}, GT index {gt_id}"
                    )
                frame_instances.append(
                    {
                        "gt_id": gt_id,
                        "obj_id": int(instance["obj_id"]),
                        "instance_uuid": instance["instance_uuid"],
                        "catalog_uuid": instance["catalog_uuid"],
                    }
                )
            frames[str(image_id)] = frame_instances
        with open(os.path.join(scene_dir, "posetestbot_render_instances.json"), "w") as handle:
            json.dump(
                {
                    "schema_version": "posetestbot_render_instances.v1",
                    "blenderproc_version": blenderproc_version,
                    "supported_blenderproc_version": SUPPORTED_BLENDERPROC_VERSION,
                    "identity_contract": "bop_gt_index_matches_loaded_instance_order.v1",
                    "instances": instance_records,
                    "frames": frames,
                },
                handle,
                sort_keys=True,
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

    instance_records = (
        objects_json["instances"]
        if objects_json.get("schema_version") == "blenderproc_object_instances.v1"
        else None
    )
    blenderproc_version = validated_blenderproc_version(
        pose_template=instance_records is not None
    )

    objects_dir = os.path.join(args.output_dir, "objects")
    objects_list = load_objects(objects_dir, objects_json)

    setup_light()
    setup_camera(args.camera_matrix, args.poses_file)

    render_and_write(
        args.output_dir,
        objects_list,
        instance_records,
        blenderproc_version=blenderproc_version,
    )


if __name__ == "__main__":
    main()
