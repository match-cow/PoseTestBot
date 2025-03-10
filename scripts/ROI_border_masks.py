import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm


def find_center_of_mask(image_path: str, output_path: str) -> None:
    """
    Finds the center of a mask in a grayscale image, marks it with a white square,
    and saves the modified image.

    The function first reads the image and converts it to a binary mask. It then
    calculates the center of the mask using image moments. If the mask is empty,
    the center is set to (0, 0).  If a mask is detected, a random point on the
    edge of the mask is chosen as the center. Finally, a 10x10 white square is
    drawn at the calculated center on a new black image, and the result is saved
    to the specified output path.

    Args:
        image_path (str): Path to the input grayscale image.
        output_path (str): Path to save the output image with the marked center.
    """
    # Read the input image as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Convert the image to a binary mask (0 or 255)
    _, binary_mask = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Calculate moments of the binary image to find the center
    moments = cv2.moments(binary_mask)

    # Calculate the center of the white object mask
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        # Handle the case where the mask is empty
        cx, cy = 0, 0

    # Find the contours of the binary mask
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Choose a random point on the edge of the mask
    if contours:
        contour = max(contours, key=cv2.contourArea)  # Get the largest contour
        edge_points = contour[:, 0, :]  # Extract the edge points
        random_point = edge_points[np.random.choice(edge_points.shape[0])]
        cx, cy = random_point[0], random_point[1]

    # Create a new black image of the same size
    output_image = np.zeros_like(image)

    # Set a 10x10 pixel square centered at (cx, cy) to white
    square_size = 5  # Half the side length of the square
    for i in range(-square_size, square_size):
        for j in range(-square_size, square_size):
            if (
                0 <= cy + i < output_image.shape[0]
                and 0 <= cx + j < output_image.shape[1]
            ):
                output_image[cy + i, cx + j] = 255

    # Save the output image
    cv2.imwrite(output_path, output_image)


def process_folder(input_folder: str, run_level: bool) -> None:
    """
    Processes image masks in subfolders of a given input folder.

    This function identifies subfolders within the input folder and processes the
    image masks found in the "masks" subfolder of each subfolder. The processing
    involves finding the center of each mask and marking it on the image. The
    modified images are then saved back to the same location.

    Args:
        input_folder (str): The path to the main input folder containing subfolders.
        run_level (bool): A flag indicating whether the script is run at a level
            above the object level. If True, the function expects an additional
            layer of subfolders representing different objects.
    """
    if run_level:
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
        object_name = os.path.basename(input_folder).split("-")[0]
        print(f"Object name: {object_name}")

        # get list of subfolders in the input_folder
        subfolders = [
            [object_name, f.path] for f in os.scandir(input_folder) if f.is_dir()
        ]

    print(f"Subfolders: {subfolders}")

    for object_name, subfolder in subfolders:
        masks_folder = os.path.join(subfolder, "masks")
        for image in tqdm(os.listdir(masks_folder), desc=f"Processing {object_name}"):
            input_image_path = os.path.join(masks_folder, image)
            output_image_path = os.path.join(masks_folder, image)

            find_center_of_mask(input_image_path, output_image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process image masks to find and mark their centers."
    )
    parser.add_argument("input_folder", help="Path to the input folder")
    parser.add_argument(
        "--run_level",
        action="store_true",
        help="Choose if wrapper is for run level above object level",
    )

    args = parser.parse_args()

    input_folder = args.input_folder
    run_level = args.run_level

    process_folder(input_folder, run_level)
