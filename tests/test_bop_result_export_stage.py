from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from posetestbot.evaluation.bop_results import (
    aruco_output_metadata,
    foundationpose_output_metadata,
    load_bop_export_index,
    parse_estimator_output_name,
    parse_foundationpose_output_name,
    rows_from_aruco_output,
)
from posetestbot.evaluation.bop_toolkit import validate_bop19_result_file
from posetestbot.io.artifacts import (
    ARUCO_POSE_ESTIMATION,
    BOP_DIR,
    BOP_EXPORT_MANIFEST,
    BOP_RESULT_EXPORT_MANIFEST,
    DATASET_MANIFEST,
    MODELS_DIR,
    RESULTS_DIR,
)


def write_pose(path: Path, translation_m: tuple[float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.eye(4)
    matrix[:3, 3] = translation_m
    np.savetxt(path, matrix)


def create_foundationpose_result_fixture(tmp_path: Path) -> tuple[Path, Path]:
    run_root = tmp_path / "run-1"
    sensor_folder = run_root / "processed" / "synchronized" / "realsense_123"
    blenderproc_folder = sensor_folder / "blenderproc"
    blenderproc_folder.mkdir(parents=True)
    (blenderproc_folder / "objects.json").write_text(
        json.dumps({"cube": np.eye(4).tolist(), "sphere": np.eye(4).tolist()})
    )
    output_folder = sensor_folder / "foundationpose_est5_track2_obj0_output"
    write_pose(output_folder / "ob_in_cam" / "000000.txt", (0.01, 0.02, 0.03))
    write_pose(output_folder / "ob_in_cam" / "000001.txt", (0.04, 0.05, 0.06))

    bop_root = run_root / BOP_DIR
    (bop_root / MODELS_DIR).mkdir(parents=True)
    (bop_root / BOP_EXPORT_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": "bop_export_manifest.v1",
                "exports": [
                    {
                        "sensor_name": "realsense_123",
                        "scene_id": 7,
                        "split": "test",
                        "scene_folder": (bop_root / "realsense_123" / "test" / "000007").as_posix(),
                    }
                ],
                "object_models": [
                    {
                        "object_name": "cube",
                        "obj_id": 3,
                        "source_path": "object_models/cube.ply",
                        "bop_path": "bop/models/obj_000003.ply",
                    },
                    {
                        "object_name": "sphere",
                        "obj_id": 4,
                        "source_path": "object_models/sphere.ply",
                        "bop_path": "bop/models/obj_000004.ply",
                    },
                ],
            }
        )
    )
    (bop_root / MODELS_DIR / "models_info.json").write_text(
        json.dumps(
            {
                "3": {"source_name": "cube"},
                "4": {"source_name": "sphere"},
            }
        )
    )
    return run_root, output_folder


def create_aruco_result_fixture(tmp_path: Path) -> tuple[Path, Path]:
    run_root = tmp_path / "run-aruco"
    sensor_folder = run_root / "processed" / "synchronized" / "realsense_123"
    sensor_folder.mkdir(parents=True)
    aruco_path = sensor_folder / ARUCO_POSE_ESTIMATION
    aruco_path.write_text(
        json.dumps(
            {
                "000000.png": {
                    "aruco_pose_estimation": {
                        "rvec": [0.0, 0.0, 0.0],
                        "tvec": [10.0, 20.0, 30.0],
                        "len_ids": 4,
                    }
                },
                "000001.png": {
                    "aruco_pose_estimation": {
                        "rvec": [0.0, 0.0, 0.0],
                        "tvec": [40.0, 50.0, 60.0],
                        "len_ids": 0,
                    }
                },
            }
        )
    )

    bop_root = run_root / BOP_DIR
    (bop_root / MODELS_DIR).mkdir(parents=True)
    (bop_root / BOP_EXPORT_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": "bop_export_manifest.v1",
                "exports": [
                    {
                        "sensor_name": "realsense_123",
                        "scene_id": 5,
                        "split": "test",
                        "scene_folder": (
                            bop_root / "realsense_123" / "test" / "000005"
                        ).as_posix(),
                    }
                ],
                "object_models": [
                    {
                        "object_name": "aruco",
                        "obj_id": 9,
                        "source_path": "object_models/aruco.ply",
                        "bop_path": "bop/models/obj_000009.ply",
                    }
                ],
            }
        )
    )
    (bop_root / MODELS_DIR / "models_info.json").write_text(
        json.dumps({"9": {"source_name": "aruco"}})
    )
    return run_root, aruco_path


def add_megapose_output(run_root: Path) -> Path:
    output_folder = (
        run_root
        / "processed"
        / "synchronized"
        / "realsense_123"
        / "megapose_obj0_output"
    )
    output_folder.mkdir(parents=True)
    (output_folder / "megapose_poses.json").write_text(
        json.dumps(
            {
                "000000": [
                    {
                        "TWO": [
                            [0.0, 0.0, 0.0, 1.0],
                            [0.01, 0.02, 0.03],
                        ]
                    }
                ]
            }
        )
    )
    return output_folder


def add_sam6d_output(run_root: Path) -> Path:
    output_folder = (
        run_root
        / "processed"
        / "synchronized"
        / "realsense_123"
        / "sam6d_obj0_output"
    )
    detection_folder = output_folder / "detections_pem"
    detection_folder.mkdir(parents=True)
    (detection_folder / "000000_0.json").write_text(
        json.dumps(
            [
                {
                    "score": 0.1,
                    "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "t": [1.0, 2.0, 3.0],
                },
                {
                    "score": 0.9,
                    "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "t": [4.0, 5.0, 6.0],
                },
            ]
        )
    )
    return output_folder


def test_foundationpose_output_name_parser() -> None:
    assert parse_foundationpose_output_name(
        "foundationpose_est5_track2_obj0_output"
    ) == ("foundationpose", "est5_track2", 0)
    assert parse_foundationpose_output_name(
        "foundationposeNoTracking_est5_track2_obj1_output"
    ) == ("foundationposeNoTracking", "est5_track2", 1)
    assert parse_foundationpose_output_name("notes") is None
    assert parse_estimator_output_name(
        "megapose_rgb_obj0_output",
        methods=("megapose",),
    ) == ("megapose", "rgb", 0)
    assert parse_estimator_output_name(
        "sam6d_obj0_output",
        methods=("sam6d",),
    ) == ("sam6d", None, 0)


def test_foundationpose_metadata_maps_object_and_scene(tmp_path: Path) -> None:
    run_root, output_folder = create_foundationpose_result_fixture(tmp_path)
    bop_index = load_bop_export_index(run_root / BOP_DIR)

    metadata = foundationpose_output_metadata(output_folder, bop_index)

    assert metadata.sensor_name == "realsense_123"
    assert metadata.method == "foundationpose"
    assert metadata.result_id == "est5_track2"
    assert metadata.object_name == "cube"
    assert metadata.obj_id == 3


def test_aruco_metadata_and_rows_map_object_and_scene(tmp_path: Path) -> None:
    run_root, aruco_path = create_aruco_result_fixture(tmp_path)
    bop_index = load_bop_export_index(run_root / BOP_DIR)

    metadata = aruco_output_metadata(
        aruco_path,
        bop_index,
        object_name="aruco",
    )
    rows = rows_from_aruco_output(
        metadata,
        bop_index,
        default_score=0.8,
        default_time=-1.0,
        translation_scale_to_mm=1.0,
        min_marker_count=1,
    )

    assert metadata.sensor_name == "realsense_123"
    assert metadata.obj_id == 9
    assert len(rows) == 1
    assert rows[0].scene_id == 5
    assert rows[0].im_id == 0
    assert rows[0].obj_id == 9
    assert rows[0].score == 0.8
    assert rows[0].R == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert rows[0].t == [10.0, 20.0, 30.0]


def test_bop_result_export_stage_writes_csv_manifest_and_stage(
    tmp_path: Path,
) -> None:
    run_root, _output_folder = create_foundationpose_result_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_bop_result_export_stage.py"),
            str(run_root),
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Exported 1 BOP result file" in result.stdout

    result_path = (
        run_root
        / RESULTS_DIR
        / BOP_DIR
        / "foundationpose_bop-test_est5_track2.csv"
    )
    with open(result_path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]
    assert rows[1] == [
        "7",
        "0",
        "3",
        "1",
        "1 0 0 0 1 0 0 0 1",
        "10 20 30",
        "-1",
    ]
    assert rows[2][0:4] == ["7", "1", "3", "1"]
    assert rows[2][5] == "40 50 60"
    assert validate_bop19_result_file(result_path).row_count == 2

    export_manifest = json.loads((run_root / BOP_RESULT_EXPORT_MANIFEST).read_text())
    assert export_manifest["schema_version"] == "bop_result_export_manifest.v1"
    assert export_manifest["dataset_name"] == "bop"
    assert export_manifest["results"][0]["filename"] == result_path.name
    assert export_manifest["results"][0]["row_count"] == 2

    run_manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage for stage in run_manifest["stages"] if stage["name"] == "bop_result_export"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][BOP_RESULT_EXPORT_MANIFEST] == (
        BOP_RESULT_EXPORT_MANIFEST
    )
    assert stage["artifacts"][RESULTS_DIR] == "results/bop"
    assert stage["artifacts"][f"bop_result:{result_path.name}"] == (
        f"results/bop/{result_path.name}"
    )


def test_bop_result_export_stage_writes_aruco_csv_manifest_and_stage(
    tmp_path: Path,
) -> None:
    run_root, _aruco_path = create_aruco_result_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_bop_result_export_stage.py"),
            str(run_root),
            "--source",
            "aruco",
            "--aruco-object-name",
            "aruco",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Exported 1 BOP result file" in result.stdout

    result_path = run_root / RESULTS_DIR / BOP_DIR / "aruco_bop-test.csv"
    with open(result_path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]
    assert rows[1] == [
        "5",
        "0",
        "9",
        "1",
        "1 0 0 0 1 0 0 0 1",
        "10 20 30",
        "-1",
    ]
    assert validate_bop19_result_file(result_path).row_count == 1

    export_manifest = json.loads((run_root / BOP_RESULT_EXPORT_MANIFEST).read_text())
    assert export_manifest["source_type"] == "aruco"
    assert export_manifest["translation_scale_to_mm"] == 1.0
    assert export_manifest["results"][0]["filename"] == result_path.name
    assert export_manifest["results"][0]["row_count"] == 1

    run_manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage for stage in run_manifest["stages"] if stage["name"] == "bop_result_export"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][f"bop_result:{result_path.name}"] == (
        f"results/bop/{result_path.name}"
    )


def test_bop_result_export_stage_writes_megapose_csv_manifest_and_stage(
    tmp_path: Path,
) -> None:
    run_root, _foundationpose_output = create_foundationpose_result_fixture(tmp_path)
    megapose_output = add_megapose_output(run_root)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_bop_result_export_stage.py"),
            str(run_root),
            "--source",
            "megapose",
            "--megapose-output",
            str(megapose_output),
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Exported 1 BOP result file" in result.stdout

    result_path = run_root / RESULTS_DIR / BOP_DIR / "megapose_bop-test.csv"
    with open(result_path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[1] == [
        "7",
        "0",
        "3",
        "1",
        "1 0 0 0 1 0 0 0 1",
        "10 20 30",
        "-1",
    ]
    assert validate_bop19_result_file(result_path).row_count == 1

    export_manifest = json.loads((run_root / BOP_RESULT_EXPORT_MANIFEST).read_text())
    assert export_manifest["source_type"] == "megapose"
    assert export_manifest["translation_scale_to_mm"] == 1000.0
    assert export_manifest["results"][0]["filename"] == result_path.name
    assert export_manifest["results"][0]["source_outputs"] == [
        megapose_output.as_posix()
    ]

    run_manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage for stage in run_manifest["stages"] if stage["name"] == "bop_result_export"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][f"bop_result:{result_path.name}"] == (
        f"results/bop/{result_path.name}"
    )


def test_bop_result_export_stage_writes_sam6d_csv_manifest_and_stage(
    tmp_path: Path,
) -> None:
    run_root, _foundationpose_output = create_foundationpose_result_fixture(tmp_path)
    sam6d_output = add_sam6d_output(run_root)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_bop_result_export_stage.py"),
            str(run_root),
            "--source",
            "sam6d",
            "--sam6d-output",
            str(sam6d_output),
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Exported 1 BOP result file" in result.stdout

    result_path = run_root / RESULTS_DIR / BOP_DIR / "sam6d_bop-test.csv"
    with open(result_path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[1] == [
        "7",
        "0",
        "3",
        "0.9",
        "1 0 0 0 1 0 0 0 1",
        "4 5 6",
        "-1",
    ]
    assert validate_bop19_result_file(result_path).row_count == 1

    export_manifest = json.loads((run_root / BOP_RESULT_EXPORT_MANIFEST).read_text())
    assert export_manifest["source_type"] == "sam6d"
    assert export_manifest["translation_scale_to_mm"] == 1.0
    assert export_manifest["results"][0]["filename"] == result_path.name
    assert export_manifest["results"][0]["source_outputs"] == [sam6d_output.as_posix()]

    run_manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage for stage in run_manifest["stages"] if stage["name"] == "bop_result_export"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][f"bop_result:{result_path.name}"] == (
        f"results/bop/{result_path.name}"
    )


def test_foundationpose_metadata_rejects_unexported_object(tmp_path: Path) -> None:
    run_root, output_folder = create_foundationpose_result_fixture(tmp_path)
    manifest_path = run_root / BOP_DIR / BOP_EXPORT_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["object_models"] = [
        {
            "object_name": "sphere",
            "obj_id": 4,
            "source_path": "object_models/sphere.ply",
            "bop_path": "bop/models/obj_000004.ply",
        }
    ]
    manifest_path.write_text(json.dumps(manifest))
    (run_root / BOP_DIR / MODELS_DIR / "models_info.json").write_text(
        json.dumps({"4": {"source_name": "sphere"}})
    )
    bop_index = load_bop_export_index(run_root / BOP_DIR)

    with pytest.raises(ValueError, match="Object 'cube' is not present"):
        foundationpose_output_metadata(output_folder, bop_index)
