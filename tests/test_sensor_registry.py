from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from posetestbot.sensors.contracts import SensorType
from posetestbot.sensors.registry import (
    build_sensor_capture_command,
    capture_script_for_sensor,
    list_sensor_adapters,
    sensor_folder_name,
)


def test_sensor_adapter_registry_lists_supported_capture_scripts() -> None:
    adapters = {adapter["sensor_type"]: adapter for adapter in list_sensor_adapters()}

    assert adapters["realsense_d435"]["capture_script"] == (
        "scripts/capture_realsense_720p.py"
    )
    assert adapters["oak_d_pro"]["capture_script"] == "scripts/capture_luxonis_720p.py"
    assert adapters["zed_2i"]["capture_script"] == "scripts/capture_zed_2i.py"
    assert adapters["realsense_d435"]["supported_resolutions"] == ["720p"]
    assert adapters["zed_2i"]["supported_resolutions"] == ["720p", "360p"]
    assert adapters["oak_d_pro"]["sdk_module"] == "depthai"


def test_sensor_registry_builds_folder_names_and_uv_capture_commands() -> None:
    assert sensor_folder_name(SensorType.REALSENSE_D435, "123") == "realsense_123"
    assert sensor_folder_name(SensorType.OAK_D_PRO, "auto") == "luxonis_auto"
    assert sensor_folder_name(SensorType.ZED_2I, "default") == "zed_2i_auto"

    command = build_sensor_capture_command(
        sensor_type=SensorType.ZED_2I,
        device_id="987",
        output_folder="/tmp/run/zed_2i_987",
        fps=12,
        resolution="360p",
        max_frames=5,
    )

    assert command == [
        "uv",
        "run",
        "python",
        "scripts/capture_zed_2i.py",
        "/tmp/run/zed_2i_987",
        "--fps",
        "12",
        "--max_frames",
        "5",
        "--device",
        "987",
        "--resolution",
        "360p",
    ]


def test_sensor_registry_rejects_unsupported_resolution() -> None:
    with pytest.raises(ValueError, match="720p"):
        capture_script_for_sensor(SensorType.REALSENSE_D435, "360p")


def test_sensor_adapters_cli_outputs_json() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        ["uv", "run", "python", "scripts/sensor_adapters.py", "--json"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert {adapter["sensor_type"] for adapter in payload["adapters"]} == {
        "realsense_d435",
        "oak_d_pro",
        "zed_2i",
    }
