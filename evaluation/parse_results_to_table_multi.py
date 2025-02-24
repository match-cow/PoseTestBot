import json
import os

import pandas as pd


def parse_results_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Parses a JSON file containing experiment results and transforms it into a Pandas DataFrame.

    The function assumes a specific structure for the JSON data, where the top-level keys represent 'runs',
    and nested within each run are experiments, sensors, methods, and motion data. It extracts relevant
    information from each level and organizes it into a tabular format.

    Args:
        file_path (str): The path to the JSON file containing the experiment results.

    Returns:
        pd.DataFrame: A Pandas DataFrame where each row represents a single experiment result, and columns
                      contain information about the run, resolution, FPS, capture velocity, object name,
                      sensor, method, motion, and various performance metrics (AP, RP, x, y, z, a, b, c).
    """

    header = [
        "run",
        "resolution",
        "fps",
        "capture_vel",
        "optional",
        "run_name",
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

        for item in data:
            run = item
            run_params = run.split("_")
            optional = run_params[:1]
            run_params = run_params[1:]

            resolution, fps, capture_vel = run_params

            for experiment in data[item]:
                for experimentname, experimentdata in experiment.items():
                    run_name = experimentname

                    for sensor in experimentdata:
                        for sensorname, sensordata in sensor.items():
                            for method in sensordata:
                                for methodname, methoddata in method.items():
                                    object_name = methodname.split("_")[-1]
                                    methodname = "_".join(methodname.split("_")[:-1])

                                    for motionname, motiondata in methoddata.items():
                                        row = {
                                            "run": run,
                                            "resolution": resolution,
                                            "fps": fps,
                                            "capture_vel": capture_vel,
                                            "optional": optional,
                                            "run_name": run_name,
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
                                        df = pd.concat([df, pd.DataFrame([row])])
    return df


def save_dataframe(df: pd.DataFrame, base_filename: str):
    """
    Saves a Pandas DataFrame to both CSV and Excel formats. Additionally, saves a rounded version
    of the DataFrame to both formats as well.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        base_filename (str): The base filename to use for saving the DataFrame.
                             The function will append suffixes like "_long" and "_rounded"
                             to create the actual filenames.
    """
    long_filename_csv = f"{base_filename}_long.csv"
    long_filename_xlsx = f"{base_filename}_long.xlsx"
    rounded_filename_csv = f"{base_filename}_rounded.csv"
    rounded_filename_xlsx = f"{base_filename}_rounded.xlsx"

    df.to_csv(long_filename_csv, index=False, sep=";")
    df.to_excel(long_filename_xlsx, index=False)

    rounded_df = df.round(2)
    rounded_df.to_csv(rounded_filename_csv, index=False, sep=";")
    rounded_df.to_excel(rounded_filename_xlsx, index=False)


if __name__ == "__main__":
    file_path = "all_results.json"
    output_filename_base = "all_results"

    df = parse_results_to_dataframe(file_path)
    print(df)
    save_dataframe(df, output_filename_base)
