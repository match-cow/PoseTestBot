import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import numpy.matlib as npm
import pytransform3d.transformations as pt
from tqdm import tqdm


def average_quaternions(quaternions: np.ndarray) -> np.ndarray:
    """Averages quaternions using the Markley method.

    Args:
        quaternions: A Nx4 numpy matrix where each row is a quaternion (w, x, y, z).

    Returns:
        A numpy array representing the average quaternion.
    """
    M = quaternions.shape[0]
    A = npm.zeros((4, 4))

    for i in range(M):
        q = quaternions[i, :]
        A += np.outer(q, q)

    A /= M
    eigenValues, eigenVectors = np.linalg.eig(A)
    eigenVectors = eigenVectors[:, eigenValues.argsort()[::-1]]
    return np.real(eigenVectors[:, 0].A1)


def weighted_average_quaternions(
    quaternions: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Averages quaternions with weights using the Markley method.

    Args:
        quaternions: A Nx4 numpy matrix where each row is a quaternion (w, x, y, z).
        weights: A weight vector of the same length as the number of rows in Q.

    Returns:
        A numpy array representing the weighted average quaternion.
    """
    M = quaternions.shape[0]
    A = npm.zeros((4, 4))
    weight_sum = 0

    for i in range(M):
        q = quaternions[i, :]
        A += weights[i] * np.outer(q, q)
        weight_sum += weights[i]

    A /= weight_sum
    eigenValues, eigenVectors = np.linalg.eig(A)
    eigenVectors = eigenVectors[:, eigenValues.argsort()[::-1]]
    return np.real(eigenVectors[:, 0].A1)


def load_transformations(
    input_file: str,
) -> Tuple[List[np.ndarray], List[float], List[float], List[float]]:
    """Loads transformations from a JSON file and extracts quaternions and positions.

    Args:
        input_file: Path to the input JSON file.

    Returns:
        A tuple containing lists of quaternions, x positions, y positions, and z positions.
    """
    with open(input_file, "r") as f:
        data = json.load(f)

    quaternions = []
    px = []
    py = []
    pz = []

    for d in tqdm(data.values(), desc="Loading transformations"):
        transformation_matrix = d["sensor_to_ee"]
        transformation_matrix = pt.check_transform(transformation_matrix)
        x, y, z, qw, qx, qy, qz = pt.pq_from_transform(transformation_matrix)

        px.append(x)
        py.append(y)
        pz.append(z)

        q = np.array([qw, qx, qy, qz])
        quaternions.append(q)

    return quaternions, px, py, pz


def compute_average_transformation(input_file: str) -> dict:
    """Computes the average quaternion and position from a JSON file.

    Args:
        input_file: Path to the input JSON file.

    Returns:
        A dictionary containing the average quaternion and position.
    """
    quaternions, px, py, pz = load_transformations(input_file)

    average_quaternion = average_quaternions(np.array(quaternions)).tolist()
    average_position = [np.mean(px), np.mean(py), np.mean(pz)]

    return {
        "quaternion": average_quaternion,
        "position": average_position,
    }


def save_average_transformation(average_transformation: dict, input_file: str) -> None:
    """Saves the average transformation to a JSON file.

    Args:
        average_transformation: A dictionary containing the average quaternion and position.
        input_file: Path to the input JSON file (used to construct the output file name).
    """
    output_file = os.path.splitext(input_file)[0] + "_average.json"
    with open(output_file, "w") as f:
        json.dump(average_transformation, f, indent=4)
    print(f"Average transformation saved to {output_file}")


def main():
    """Main function to parse arguments, compute averages, and save the results."""
    parser = argparse.ArgumentParser(
        description="Compute average quaternion and position from a JSON file."
    )
    parser.add_argument("input_file", help="Path to the input JSON file.")
    args = parser.parse_args()
    input_file = args.input_file

    average_transformation = compute_average_transformation(input_file)
    save_average_transformation(average_transformation, input_file)


if __name__ == "__main__":
    main()
