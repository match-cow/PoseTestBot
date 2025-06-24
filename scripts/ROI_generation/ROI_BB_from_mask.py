import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm


def find_bounding_box_from_mask(image_path: str, output_path: str) -> None:
    """
    Finds the bounding box of a mask in a grayscale image, expands it with an offset,
    and saves the bounding box as a filled rectangle in a new image.

    Args:
        image_path (str): Path to the input grayscale mask image.
        output_path (str): Path to save the output image with the bounding box.
    """
    # Read the input image as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Ensure the image is binary (0 or 255) using a threshold
    _, binary_mask = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Create a black image of the same size as the input
    output_image = np.zeros_like(image)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # If contours are found
    if contours:
        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(contours[0])

        # Define an offset to expand the bounding box
        offset = 10

        # Adjust the bounding box with the offset, ensuring it stays within image bounds
        x = max(0, x - offset)
        y = max(0, y - offset)
        w = min(output_image.shape[1] - x, w + 2 * offset)
        h = min(output_image.shape[0] - y, h + 2 * offset)

        # Draw the bounding box as a filled rectangle on the output image
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 255, 255), -1)

    # Save the output image
    cv2.imwrite(output_path, output_image)


def process_folder(input_folder: str, run_level: bool) -> None:
    """
    Processes a folder of mask images to find and save bounding boxes.

    Args:
        input_folder (str): Path to the input folder containing mask images.
        run_level (bool): If True, processes at the run level (object level above).
                           If False, processes at the object level.
    """
    if run_level:
        # If run_level is True, assume a directory structure of runs containing objects
        object_folders = [f.path for f in os.scandir(input_folder) if f.is_dir()]
        subfolders = []
        for object_folder in object_folders:
            object_name = os.path.basename(object_folder).split("-")[0]
            [
                subfolders.append([object_name, f.path])
                for f in os.scandir(object_folder)
                if f.is_dir()
            ]
    else:
        # If run_level is False, assume a directory structure of objects
        object_name = os.path.basename(input_folder).split("-")[0]
        print(f"Object name: {object_name}")

        # Get list of subfolders in the input_folder
        subfolders = [
            [object_name, f.path] for f in os.scandir(input_folder) if f.is_dir()
        ]

    print(f"Subfolders: {subfolders}")

    # Iterate through the subfolders
    for object_name, subfolder in subfolders:
        masks_folder = os.path.join(subfolder, "masks")
        # Process each image in the masks folder
        for image in tqdm(os.listdir(masks_folder), desc=f"Processing {object_name}"):
            input_image_path = os.path.join(masks_folder, image)
            output_image_path = os.path.join(masks_folder, image)

            # Find and save the bounding box for the current mask image
            find_bounding_box_from_mask(input_image_path, output_image_path)


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Finds the bounding box of a mask in a grayscale image."
    )
    parser.add_argument("input_folder", help="Path to the input folder")
    parser.add_argument(
        "--run_level",
        action="store_true",
        help="Choose if wrapper is for run level above object level",
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Get the input folder and run level from the arguments
    input_folder = args.input_folder
    run_level = args.run_level

    # Process the folder
    process_folder(input_folder, run_level)
