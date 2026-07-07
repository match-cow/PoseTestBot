from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from posetestbot.io.artifacts import CAM_K


def load_blenderproc_prepare_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "blenderproc_prepare_multi.py"
    )
    spec = importlib.util.spec_from_file_location("blenderproc_prepare_multi", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_blenderproc_read_camera_parameters_allows_missing_distortion(
    tmp_path: Path,
) -> None:
    module = load_blenderproc_prepare_module()
    sensor_folder = tmp_path / "processed" / "synchronized" / "realsense_123"
    sensor_folder.mkdir(parents=True)
    (sensor_folder / CAM_K).write_text("1 0 2\n0 3 4\n0 0 1\n")

    cam_matrix, dist_coefficients = module.read_camera_parameters(str(sensor_folder))

    np.testing.assert_allclose(cam_matrix, np.array([[1, 0, 2], [0, 3, 4], [0, 0, 1]]))
    np.testing.assert_allclose(dist_coefficients, np.zeros((5, 1)))


def test_blenderproc_camera_transform_lookup_falls_back_to_sensor_type() -> None:
    module = load_blenderproc_prepare_module()
    transforms = {
        "realsense": {"position": [1, 2, 3]},
        "luxonis": {"position": [4, 5, 6]},
        "zed_2i": {"position": [7, 8, 9]},
    }

    assert module.camera_transform_for_sensor(transforms, "realsense_123") == {
        "position": [1, 2, 3]
    }
    assert module.camera_transform_for_sensor(transforms, "luxonis_abc") == {
        "position": [4, 5, 6]
    }
    assert module.camera_transform_for_sensor(transforms, "zed_2i_42") == {
        "position": [7, 8, 9]
    }

