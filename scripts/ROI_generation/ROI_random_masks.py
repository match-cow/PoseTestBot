import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm


def find_center_of_mask(image_path: str, output_path: str) -> None:
    """
    Finds the center of a mask, applies a random offset, and saves a 10x10 white square at the new center.

    Args:
        image_path (str): Path to the input mask image (grayscale).
        output_path (str): Path to save the output image with the shifted center.
    """
    # Read the input image as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Ensure the image is binary (0 or 255) using a threshold
    _, binary_mask = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Calculate moments of the binary mask to find the center
    moments = cv2.moments(binary_mask)

    # Calculate the center of the white object mask
    if moments["m00"] != 0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
    else:
        # Handle the case where the mask is empty
        center_x, center_y = 0, 0

    # Calculate the bounding rectangle of the object
    x, y, width, height = cv2.boundingRect(binary_mask)

    # Generate random offsets within 20% of the object's width and height
    max_offset_x = int(0.2 * width)
    max_offset_y = int(0.2 * height)
    offset_x = np.random.randint(-max_offset_x, max_offset_x + 1)
    offset_y = np.random.randint(-max_offset_y, max_offset_y + 1)

    # Apply the random offsets to the center coordinates
    center_x += offset_x
    center_y += offset_y

    # Create a new black image of the same size as the input
    output_image = np.zeros_like(image)

    # Define the size of the square to be placed at the center
    square_size = 5  # Creates a 10x10 square (from -5 to 4)

    # Set a square centered at (center_x, center_y) to white
    for i in range(-square_size, square_size):
        for j in range(-square_size, square_size):
            # Check if the pixel is within the image bounds
            if (
                0 <= center_y + i < output_image.shape[0]
                and 0 <= center_x + j < output_image.shape[1]
            ):
                output_image[center_y + i, center_x + j] = 255

    # Save the output image
    cv2.imwrite(output_path, output_image)


def process_folder(input_folder: str, run_level: bool) -> None:
    """
    Processes a folder of mask images, applying the center finding and shifting to each.

    Args:
        input_folder (str): Path to the input folder containing either object folders or mask folders directly.
        run_level (bool): If True, the input folder contains object folders, each containing mask folders.
                          If False, the input folder directly contains mask folders.
    """
    if run_level:
        # If run_level is True, assume a structure of input_folder/object_folder/mask_folder/images
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
        # If run_level is False, assume a structure of input_folder/mask_folder/images
        object_name = os.path.basename(input_folder).split("-")[0]
        print(f"Object name: {object_name}")

        # get list of subfolders in the input_folder
        subfolders = [
            [object_name, f.path] for f in os.scandir(input_folder) if f.is_dir()
        ]

    print(f"Subfolders: {subfolders}")

    # Iterate through the subfolders (mask folders)
    for object_name, subfolder in subfolders:
        masks_folder = os.path.join(subfolder, "masks")
        # Process each image in the mask folder
        for image in tqdm(os.listdir(masks_folder), desc=f"Processing {object_name}"):
            input_image_path = os.path.join(masks_folder, image)
            output_image_path = os.path.join(
                masks_folder, image
            )  # Overwrite the original

            find_center_of_mask(input_image_path, output_image_path)


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Finds the center of masks, applies a random offset, and saves a 10x10 white square at the new center."
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
