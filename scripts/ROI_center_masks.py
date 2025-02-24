import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm


def find_center_of_mask(image_path: str, output_path: str) -> None:
    """
    Finds the center of a white object in a binary mask image,
    creates a new image with a white square at the center, and saves it.

    Args:
        image_path (str): Path to the input binary mask image.
        output_path (str): Path to save the output image with the center marked.
    """
    # Read the input image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Ensure the image is binary (0 or 255) using a threshold
    _, binary_mask = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Calculate moments of the binary image to find the center
    moments = cv2.moments(binary_mask)

    # Calculate the center of the white object mask
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        # Handle the case where the mask is empty by setting the center to (0, 0)
        cx, cy = 0, 0

    # Create a new black image of the same size as the input
    output_image = np.zeros_like(image)

    # Define the size of the square to mark the center
    square_size = 5  # Creates a 10x10 square (5 pixels in each direction)

    # Set a square of pixels centered at (cx, cy) to white
    for i in range(-square_size, square_size):
        for j in range(-square_size, square_size):
            # Check if the pixel is within the image bounds
            if (
                0 <= cy + i < output_image.shape[0]
                and 0 <= cx + j < output_image.shape[1]
            ):
                output_image[cy + i, cx + j] = 255

    # Save the output image
    cv2.imwrite(output_path, output_image)


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Find the center of masks in a folder and mark it with a square."
    )
    parser.add_argument(
        "input_folder", help="Path to the input folder containing the masks"
    )
    parser.add_argument(
        "--run_level",
        action="store_true",
        help="If set, the script will process subfolders of the input folder as separate objects.",
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Get the input folder and run level from the arguments
    input_folder = args.input_folder
    run_level = args.run_level

    # If run_level is set, process each subfolder as a separate object
    if run_level:
        # Get a list of object folders (subfolders of the input folder)
        object_folders = [f.path for f in os.scandir(input_folder) if f.is_dir()]
        subfolders = []
        # For each object folder, get a list of subfolders and their corresponding object names
        for object_folder in object_folders:
            object_name = os.path.basename(object_folder).split("-")[0]
            [
                subfolders.append([object_name, f.path])
                for f in os.scandir(object_folder)
                if f.is_dir()
            ]
    # Otherwise, process the input folder as a single object
    else:
        # Get the object name from the input folder name
        object_name = os.path.basename(input_folder).split("-")[0]
        print(f"Object name: {object_name}")

        # Get a list of subfolders in the input_folder and their corresponding object names
        subfolders = [
            [object_name, f.path] for f in os.scandir(input_folder) if f.is_dir()
        ]

    # Print the list of subfolders to be processed
    print(f"Subfolders: {subfolders}")

    # Iterate over the subfolders
    for object_name, subfolder in subfolders:
        # Get the path to the masks folder
        masks_folder = os.path.join(subfolder, "masks")
        # Iterate over the images in the masks folder
        for image in tqdm(os.listdir(masks_folder), desc=f"Processing {object_name}"):
            # Create the input and output image paths
            input_image_path = os.path.join(masks_folder, image)
            output_image_path = os.path.join(masks_folder, image)

            # Find the center of the mask and mark it with a square
            find_center_of_mask(input_image_path, output_image_path)
