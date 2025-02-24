import json
import os
from typing import Any, Dict, List


def combine_json_files(
    directory: str, results_file_name: str, results_file_name_aruco: str
) -> Dict[str, List[Dict[str, List[Dict[str, List[Any]]]]]]:
    """
    Combines JSON results from multiple files organized in a specific directory structure.

    The directory structure is expected to be:
    main_directory/experiment_name/object_name/sensor_name/{results_file_name, results_file_name_aruco}

    Args:
        directory (str): The main directory containing experiment subfolders.
        results_file_name (str): The name of the primary results file (e.g., "accuracy_HRC-Hub.json").
        results_file_name_aruco (str): The name of the ArUco results file (e.g., "accuracy_ArUco_HRC-Hub.json").

    Returns:
        Dict[str, List[Dict[str, List[Dict[str, List[Any]]]]]]: A dictionary containing the combined results,
        organized by experiment, object, and sensor. The structure is as follows:
        {
            experiment_name: [
                {
                    object_name: [
                        {
                            sensor_name: [data1, data2, ...]
                        },
                        ...
                    ]
                },
                ...
            ],
            ...
        }
    """
    combined_results: Dict[str, List[Dict[str, List[Dict[str, List[Any]]]]]] = {}

    for experiment_folder in os.listdir(directory):
        experiment_name = experiment_folder
        experiment_path = os.path.join(directory, experiment_folder)

        if os.path.isdir(experiment_path):
            experiment_results: List[Dict[str, List[Dict[str, List[Any]]]]] = []

            for object_folder in os.listdir(experiment_path):
                object_name = object_folder
                object_path = os.path.join(experiment_path, object_folder)

                if os.path.isdir(object_path):
                    object_results: List[Dict[str, List[Dict[str, List[Any]]]]] = []

                    for sensor_folder in os.listdir(object_path):
                        sensor_name = sensor_folder
                        sensor_path = os.path.join(object_path, sensor_folder)

                        if os.path.isdir(sensor_path):
                            sensor_results: List[Any] = []

                            # Helper function to read and append JSON data from a file
                            def read_json_data(
                                file_path: str, sensor_results: List[Any]
                            ):
                                if os.path.exists(file_path):
                                    with open(file_path, "r") as results_file:
                                        try:
                                            data = json.load(results_file)
                                            sensor_results.append(data)
                                        except json.JSONDecodeError:
                                            print(
                                                f"Error decoding JSON from file: {file_path}"
                                            )

                            # Read data from the primary results file
                            results_file_path = os.path.join(
                                sensor_path, results_file_name
                            )
                            read_json_data(results_file_path, sensor_results)

                            # Read data from the ArUco results file
                            results_file_path_aruco = os.path.join(
                                sensor_path, results_file_name_aruco
                            )
                            read_json_data(results_file_path_aruco, sensor_results)

                            # Add the sensor results to the object results
                            sensor_results_dict: Dict[str, List[Any]] = {
                                sensor_name: sensor_results
                            }
                            object_results.append(sensor_results_dict)

                    # Add the object results to the experiment results
                    object_results_dict: Dict[str, List[Dict[str, List[Any]]]] = {
                        object_name: object_results
                    }
                    experiment_results.append(object_results_dict)

            # Add the experiment results to the combined results
            combined_results[experiment_name] = experiment_results

    return combined_results


def save_combined_results(combined_results: Dict[str, Any], output_path: str) -> None:
    """
    Saves the combined results to a JSON file.

    Args:
        combined_results (Dict[str, Any]): The dictionary containing the combined results.
        output_path (str): The path to the output JSON file.
    """
    with open(output_path, "w") as output_file:
        json.dump(combined_results, output_file, indent=4)


if __name__ == "__main__":
    # Define the main directory containing experiment subfolders
    main_directory = "/path/to/your/working_data"
    output_file_path = os.path.join(main_directory, "all_results.json")
    results_file_name = "accuracy_HRC-Hub.json"
    results_file_name_aruco = "accuracy_ArUco_HRC-Hub.json"

    # Combine the JSON results
    combined = combine_json_files(
        main_directory, results_file_name, results_file_name_aruco
    )

    # Save the combined results to an output file
    save_combined_results(combined, output_file_path)
    print(f"Combined results saved to {output_file_path}")
