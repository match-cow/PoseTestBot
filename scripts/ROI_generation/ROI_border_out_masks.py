import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm


def find_center_of_mask(image_path: str, output_path: str) -> None:
    """
    Finds a point outside the border of a mask in a grayscale image,
    and creates a new image with a white square at that point.

    Args:
        image_path (str): Path to the input grayscale image.
        output_path (str): Path to save the output image with the white square.
    """
    # Read the input image as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Convert the image to a binary mask (0 or 255)
    _, binary_mask = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Calculate the moments of the binary mask
    moments = cv2.moments(binary_mask)

    # Calculate the center of the mask using the moments
    if moments["m00"] != 0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
    else:
        # Handle the case where the mask is empty by setting the center to (0, 0)
        center_x, center_y = 0, 0

    # Find the contours of the binary mask
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # If contours are found, proceed to find a point outside the border
    if contours:
        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)
        # Extract the edge points from the contour
        edge_points = contour[:, 0, :]
        # Choose a random point on the edge
        random_point = edge_points[np.random.choice(edge_points.shape[0])]
        edge_x, edge_y = random_point[0], random_point[1]

        # Calculate a vector from the center to the random edge point
        vector_x = edge_x - center_x
        vector_y = edge_y - center_y

        # Normalize the vector
        length = np.sqrt(vector_x**2 + vector_y**2)
        if length != 0:
            vector_x /= length
            vector_y /= length

        # Move the point outside the mask by 10 pixels
        border_x = edge_x + int(vector_x * 10)
        border_y = edge_y + int(vector_y * 10)
    else:
        # If no contours are found, use the calculated center as the border point
        border_x, border_y = center_x, center_y

    # Create a new black image with the same dimensions as the input
    output_image = np.zeros_like(image)

    # Draw a 10x10 white square centered at the calculated border point
    square_size = 5  # Half the side length of the square
    for i in range(-square_size, square_size):
        for j in range(-square_size, square_size):
            # Check if the pixel is within the image bounds
            if (
                0 <= border_y + i < output_image.shape[0]
                and 0 <= border_x + j < output_image.shape[1]
            ):
                output_image[border_y + i, border_x + j] = 255

    # Save the output image
    cv2.imwrite(output_path, output_image)


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Process mask images to find the center and mark a border point."
    )
    parser.add_argument(
        "input_folder", help="Path to the input folder containing the images."
    )
    parser.add_argument(
        "--run_level",
        action="store_true",
        help="Specify if the script should run at the object level (above object level).",
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Get the input folder path from the arguments
    input_folder = args.input_folder
    # Determine the run level
    run_level = args.run_level

    # If running at the object level, process subfolders
    if run_level:
        # Get a list of object folders in the input folder
        object_folders = [f.path for f in os.scandir(input_folder) if f.is_dir()]
        subfolders = []
        # Iterate through each object folder
        for object_folder in object_folders:
            # Extract the object name from the folder name
            object_name = os.path.basename(object_folder).split("-")[0]
            # Get a list of subfolders within the object folder
            [
                subfolders.append([object_name, f.path])
                for f in os.scandir(object_folder)
                if f.is_dir()
            ]
    else:
        # If not running at the object level, process the input folder directly
        object_name = os.path.basename(input_folder).split("-")[0]
        print(f"Object name: {object_name}")

        # Get a list of subfolders in the input folder
        subfolders = [
            [object_name, f.path] for f in os.scandir(input_folder) if f.is_dir()
        ]

    # Print the list of subfolders to be processed
    print(f"Subfolders: {subfolders}")

    # Iterate through each subfolder
    for object_name, subfolder in subfolders:
        # Construct the path to the masks folder
        masks_folder = os.path.join(subfolder, "masks")
        # Process each image in the masks folder
        for image in tqdm(os.listdir(masks_folder), desc=f"Processing {object_name}"):
            # Construct the full input and output image paths
            input_image_path = os.path.join(masks_folder, image)
            output_image_path = os.path.join(masks_folder, image)

            # Call the function to find the center of the mask and mark the border point
            find_center_of_mask(input_image_path, output_image_path)
