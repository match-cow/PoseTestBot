from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from posetestbot.aruco.grid import estimate_sensor_poses
from posetestbot.calibration.intrinsics import (
    IntrinsicCalibrationError,
    calibrate_intrinsic_profile,
    factory_intrinsic_profile,
    select_intrinsic_profile,
)
from posetestbot.calibration.targets import (
    import_aruco_gridgen_export,
    opencv_grid_board,
)
from posetestbot.io.artifacts import ARUCO_POSE_ESTIMATION


def generator_export() -> dict:
    rows, cols = 2, 3
    marker_ids = list(range(rows * cols))
    return {
        "version": "1.0",
        "timestamp": "2026-07-10T00:00:00+00:00",
        "settings": {
            "board_type": "aruco_grid",
            "paper_size": "A4",
            "orientation": "landscape",
            "dictionary": "DICT_5X5_50",
            "rows": rows,
            "cols": cols,
            "marker_size_mm": 30,
            "separation_mm": 10,
            "horizontal_scale": 100.0,
            "vertical_scale": 100.0,
        },
        "grid_info": {
            "total_markers": len(marker_ids),
            "marker_ids": marker_ids,
            "marker_positions_mm": [
                {
                    "id": marker_id,
                    "row": marker_id // cols,
                    "col": marker_id % cols,
                    "x_mm": 20 + (marker_id % cols) * 40,
                    "y_mm": 30 + (marker_id // cols) * 40,
                }
                for marker_id in marker_ids
            ],
        },
        "transformation": {
            "enabled": True,
            "matrix_4x4": [[-1, 0, 0, 999], [0, -1, 0, 999], [0, 0, 1, 999], [0, 0, 0, 1]],
        },
    }


def write_generator(path: Path, value: dict | None = None) -> bytes:
    raw = json.dumps(value or generator_export(), indent=2).encode()
    path.write_bytes(raw)
    return raw


def test_import_aruco_gridgen_target_preserves_source_and_uses_grid_frame(
    tmp_path: Path,
) -> None:
    source = tmp_path / "grid.json"
    raw = write_generator(source)

    target = import_aruco_gridgen_export(source, aligned_to_template_base=True)

    assert target["schema_version"] == "calibration_target.v2"
    assert target["grid_size"] == [3, 2]
    assert [marker["id"] for marker in target["markers"]] == list(range(6))
    assert target["markers"][0]["corners_mm"] == [
        [0.0, 0.0, 0.0],
        [30.0, 0.0, 0.0],
        [30.0, 30.0, 0.0],
        [0.0, 30.0, 0.0],
    ]
    assert target["frame"] == {
        "name": "aruco_grid",
        "origin": "compensated_outer_board_top_left",
        "axes": {"x": "right", "y": "down", "z": "into_board"},
    }
    assert target["generator_source"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert target["generator_source"]["export"]["transformation"]["enabled"] is True
    assert target["placement"]["from"] == "aruco_grid"
    assert target["placement"]["to"] == "template_base"
    assert target["placement"]["translation_mm"] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(version="2.0"), "version"),
        (lambda value: value["settings"].update(board_type="charuco"), "board_type"),
        (lambda value: value["settings"].update(dictionary="DICT_NOT_REAL"), "dictionary"),
        (lambda value: value["settings"].update(horizontal_scale=99.9), "exactly 100"),
        (lambda value: value["grid_info"].update(marker_ids=[1, 2, 3, 4, 5, 6]), "contiguous"),
    ],
)
def test_import_aruco_gridgen_rejects_invalid_contract(
    tmp_path: Path, mutation, message: str
) -> None:
    value = copy.deepcopy(generator_export())
    mutation(value)
    source = tmp_path / "grid.json"
    write_generator(source, value)

    with pytest.raises(ValueError, match=message):
        import_aruco_gridgen_export(source)


def sensor_fixture(folder: Path) -> None:
    (folder / "rgb").mkdir(parents=True)
    (folder / "cam_K.txt").write_text(
        "580 0 320\n0 585 240\n0 0 1\n0.02 -0.01 0.001 -0.001 0.003\n"
    )
    (folder / "depthscale.txt").write_text("1.0\n")
    (folder / "camera_data.json").write_text(
        json.dumps({"K": [[580, 0, 320], [0, 585, 240], [0, 0, 1]], "resolution": [480, 640]})
    )
    (folder / "frame_metadata.jsonl").write_text(
        json.dumps({"sensor_id": "SERIAL-1", "orientation": "normal"}) + "\n"
    )


def synthetic_detections(target: dict, *, centered: bool = False) -> tuple[dict, dict[str, np.ndarray]]:
    _dictionary, board = opencv_grid_board(target)
    ids = board.getIds().reshape(-1).astype(int).tolist()
    objects = [np.asarray(item, dtype=np.float32).reshape(4, 3) for item in board.getObjPoints()]
    true_k = np.array([[600.0, 0.0, 320.0], [0.0, 605.0, 240.0], [0.0, 0.0, 1.0]])
    distortion = np.array([0.02, -0.01, 0.001, -0.001, 0.003])
    frames = {}
    poses = {}
    offsets = [(0.0, 0.0)] * 18 if centered else [
        (x, y) for y in (-130.0, 0.0, 130.0) for x in (-190.0, 0.0, 190.0) for _repeat in range(2)
    ]
    for index, (tx, ty) in enumerate(offsets):
        rvec = np.array(
            [
                (-0.22, 0.08, 0.26)[index % 3],
                (-0.18, 0.20, 0.05)[(index // 3) % 3],
                -0.08 + 0.025 * (index % 7),
            ]
        )
        tvec = np.array([tx, ty, 620.0 + 8.0 * (index % 5)])
        corners = [
            cv2.projectPoints(points, rvec, tvec, true_k, distortion)[0].reshape(4, 2).tolist()
            for points in objects
        ]
        name = f"{index:06d}.png"
        frames[name] = {"ids": ids, "corners": corners, "marker_count": len(ids)}
        poses[name] = {"rvec": rvec, "tvec": tvec}
    return {
        "schema_version": "aruco_detections.v1",
        "image_size": [640, 480],
        "frames": frames,
    }, poses


def imported_target(tmp_path: Path) -> dict:
    path = tmp_path / "generator.json"
    write_generator(path)
    return import_aruco_gridgen_export(path, aligned_to_template_base=True)


def test_synthetic_intrinsic_recovery_and_enriched_pose(tmp_path: Path) -> None:
    sensor = tmp_path / "realsense_SERIAL-1"
    sensor_fixture(sensor)
    target = imported_target(tmp_path)
    detections, poses = synthetic_detections(target)

    profile = calibrate_intrinsic_profile(sensor, detections, target)

    assert profile["schema_version"] == "intrinsic_calibration.v1"
    assert profile["quality"]["accepted_view_count"] == 18
    assert len(profile["quality"]["coverage_cells"]) >= 6
    assert profile["quality"]["rms_reprojection_error_px"] < 0.1
    assert np.allclose(
        np.asarray(profile["native"]["cam_K"]).reshape(3, 3),
        np.array([[600.0, 0.0, 320.0], [0.0, 605.0, 240.0], [0.0, 0.0, 1.0]]),
        atol=1.0,
    )
    assert profile["rectified"]["distortion"] == [0.0] * 5
    assert profile["depth"]["alignment"]["recalibrated"] is False

    output = estimate_sensor_poses(sensor, detections, target, profile)
    first = output["000000.png"]["aruco_pose_estimation"]
    assert first["schema_version"] == "aruco_pose_estimation.v2"
    assert first["transform"]["from"] == "aruco_grid"
    assert first["transform"]["to"] == "camera"
    assert first["pnp_inlier_count"] >= 4
    assert first["mean_reprojection_error_px"] < 0.1
    assert np.allclose(first["tvec"], poses["000000.png"]["tvec"], atol=1.0)
    assert (sensor / ARUCO_POSE_ESTIMATION).is_file()


def test_intrinsic_coverage_failure_reports_rejected_audit(tmp_path: Path) -> None:
    sensor = tmp_path / "realsense_SERIAL-1"
    sensor_fixture(sensor)
    target = imported_target(tmp_path)
    detections, _poses = synthetic_detections(target, centered=True)

    with pytest.raises(IntrinsicCalibrationError) as captured:
        calibrate_intrinsic_profile(sensor, detections, target)

    assert captured.value.report["status"] == "rejected"
    assert "coverage" in captured.value.report["reason"]
    assert captured.value.report["accepted_views"]


def test_factory_profile_and_exact_identity_selection(tmp_path: Path) -> None:
    sensor = tmp_path / "realsense_SERIAL-1"
    sensor_fixture(sensor)
    profile = factory_intrinsic_profile(sensor)

    selected = select_intrinsic_profile(
        [profile], sensor_id="SERIAL-1", resolution=(640, 480), orientation="normal"
    )

    assert selected["source"]["mode"] == "factory"
    assert selected["depth"]["scale_source"] == "factory_sdk"
    with pytest.raises(ValueError, match="exactly one"):
        select_intrinsic_profile(
            [profile], sensor_id="SERIAL-1", resolution=(1280, 720), orientation="normal"
        )
