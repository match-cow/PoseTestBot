import json

import pandas as pd


def parse_run_params(run_string):
    """
    Parses the run string to extract run parameters.

    Args:
        run_string (str): A string containing run parameters separated by underscores.

    Returns:
        tuple: A tuple containing resolution, fps, capture velocity, and an optional parameter.
               If the run string does not contain the optional parameter, it returns None for that value.
    """
    run_params = run_string.split("_")
    if len(run_params) == 4:
        return run_params[0], run_params[1], run_params[2], run_params[3]
    return run_params[0], run_params[1], run_params[2], None


def create_row(
    run,
    resolution,
    fps,
    capture_vel,
    optional,
    object_name,
    sensorname,
    methodname,
    motionname,
    motiondata,
):
    """
    Creates a dictionary representing a row of data for the DataFrame.

    Args:
        run (str): The run identifier.
        resolution (str): The resolution of the run.
        fps (str): The frames per second of the run.
        capture_vel (str): The capture velocity of the run.
        optional (str): An optional parameter for the run.
        object_name (str): The name of the object being evaluated.
        sensorname (str): The name of the sensor used.
        methodname (str): The name of the method used.
        motionname (str): The name of the motion being evaluated.
        motiondata (dict): A dictionary containing motion data.

    Returns:
        dict: A dictionary containing the data for a single row.
    """
    return {
        "run": run,
        "resolution": resolution,
        "fps": fps,
        "capture_vel": capture_vel,
        "optional": optional,
        "object_name": object_name,
        "sensor": sensorname,
        "method": methodname,
        "motion": motionname,
        "AP_p": motiondata["AP_p"],
        "ap_x": motiondata["ap_x"],
        "ap_y": motiondata["ap_y"],
        "ap_z": motiondata["ap_z"],
        "ap_a": motiondata["ap_a"],
        "ap_b": motiondata["ap_b"],
        "ap_c": motiondata["ap_c"],
        "RP_i": motiondata["RP_i"],
        "RP_a": motiondata["RP_a"][0],
        "RP_b": motiondata["RP_b"][0],
        "RP_c": motiondata["RP_c"][0],
        "x": motiondata["x"],
        "y": motiondata["y"],
        "z": motiondata["z"],
        "a": motiondata["a"],
        "b": motiondata["b"],
        "c": motiondata["c"],
    }


def process_json_data(file_path):
    """
    Processes the JSON data from the given file path, parses the data, and converts it into a Pandas DataFrame.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        pandas.DataFrame: A DataFrame containing the parsed data.
    """
    header = [
        "run",
        "resolution",
        "fps",
        "capture_vel",
        "optional",
        "object_name",
        "sensor",
        "method",
        "motion",
        "AP_p",
        "ap_x",
        "ap_y",
        "ap_z",
        "ap_a",
        "ap_b",
        "ap_c",
        "RP_i",
        "RP_a",
        "RP_b",
        "RP_c",
        "x",
        "y",
        "z",
        "a",
        "b",
        "c",
    ]
    df = pd.DataFrame(columns=header)

    with open(file_path, "r") as file:
        data = json.load(file)

        for run, experiments in data.items():
            resolution, fps, capture_vel, optional = parse_run_params(run)

            for experiment in experiments:
                for object_name, sensors in experiment.items():
                    for sensor in sensors:
                        for sensorname, methods in sensor.items():
                            for method in methods:
                                for methodname, motions in method.items():
                                    for motionname, motiondata in motions.items():
                                        row = create_row(
                                            run,
                                            resolution,
                                            fps,
                                            capture_vel,
                                            optional,
                                            object_name,
                                            sensorname,
                                            methodname,
                                            motionname,
                                            motiondata,
                                        )
                                        df = pd.concat(
                                            [df, pd.DataFrame([row])], ignore_index=True
                                        )
    return df


def main():
    """
    Main function to execute the data processing and export to CSV and Excel formats.
    """
    file_path = r"/path/to/your/json/file/all_results.json"
    df = process_json_data(file_path)

    # Export the DataFrame to CSV and Excel files
    df.to_csv("all_results_v1.csv", index=False, sep=";")
    df.to_excel("all_results_v1.xlsx", index=False)

    # Round the values to 2 decimal places and export again
    rounded_df = df.round(2)
    rounded_df.to_csv("all_results_v2.csv", index=False, sep=";")
    rounded_df.to_excel("all_results_v2.xlsx", index=False)


if __name__ == "__main__":
    main()
