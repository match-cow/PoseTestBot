import json
import os
from typing import Dict, List, Tuple

import numpy as np
import trimesh


def load_mesh_and_create_slice(
    model_path: str, plane_origin: List[float], plane_normal: List[float]
) -> Tuple[trimesh.path.Path2D, np.ndarray]:
    """
    Loads a mesh from a file, creates a slice at a given plane, and converts it to a 2D planar representation.

    Args:
        model_path (str): Path to the mesh file.
        plane_origin (List[float]): Origin of the plane.
        plane_normal (List[float]): Normal vector of the plane.

    Returns:
        Tuple[trimesh.path.Path2D, np.ndarray]: A tuple containing the 2D slice and the transformation matrix.
    """
    mesh = trimesh.load_mesh(model_path)
    print(mesh.bounds)  # Print the bounding box of the mesh
    slice_3D = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    slice_2D, to_3D = slice_3D.to_planar()
    return slice_2D, to_3D


def generate_combined_slice() -> trimesh.path.Path2D:
    """
    Generates a combined 2D slice from multiple mesh files.

    This function iterates through a list of mesh file paths, loads each mesh,
    creates a 2D slice at z=0.1, and combines them into a single 2D slice.
    The transformation matrices to recover the original 3D positions of the slices
    are stored in a dictionary and saved to a JSON file.

    Returns:
        trimesh.path.Path2D: The combined 2D slice of the meshes.
    """
    output: Dict[str, List[List[float]]] = {}
    slices_2D: List[trimesh.path.Path2D] = []
    plane_origin = [0, 0, 0.1]
    plane_normal = [0, 0, 1]
    model_names = [
        "Df4a",
        "Qf4i",
        "ITODD_6",
        "ITODD_21",
        "HOPE_3",
        "HOPE_13",
        "HOPE_26",
        "HOPE_27",
        "HOPE_28",
        "Benchy",
    ]

    for model_name in model_names:
        model_path = os.path.join("object_models", f"{model_name}.ply")
        slice_2D, to_3D = load_mesh_and_create_slice(
            model_path, plane_origin, plane_normal
        )
        output[model_name] = to_3D.tolist()
        slices_2D.append(slice_2D)

    combined = np.sum(slices_2D)

    # Save the transformation matrices to a JSON file
    with open("output.json", "w") as f:
        json.dump(output, f, default=str, indent=4)

    return combined


# Generate and display the combined slice
if __name__ == "__main__":
    combined_slice = generate_combined_slice()
    combined_slice.show()
