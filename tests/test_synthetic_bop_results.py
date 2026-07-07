from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from posetestbot.evaluation.bop_toolkit import validate_bop19_result_file
from posetestbot.evaluation.synthetic_bop_results import (
    write_synthetic_bop_results_with_manifest,
)
from posetestbot.io.artifacts import (
    BOP_DIR,
    BOP_EXPORT_MANIFEST,
    BOP_FRAME_MAP_JSON,
    BOP_RESULT_EXPORT_MANIFEST,
    BOP_TARGETS_BOP19,
    DATASET_MANIFEST,
    RESULTS_DIR,
)


def create_bop_export_fixture(tmp_path: Path) -> Path:
    run_root = tmp_path / "run"
    scene_folder = run_root / BOP_DIR / "realsense_synthetic" / "test" / "000001"
    scene_folder.mkdir(parents=True)
    (scene_folder / BOP_FRAME_MAP_JSON).write_text(
        json.dumps(
            {
                "0": {"source_rgb": "rgb/000000.png"},
                "1": {"source_rgb": "rgb/000001.png"},
            }
        )
    )
    (run_root / BOP_DIR / BOP_EXPORT_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": "bop_export_manifest.v1",
                "exports": [
                    {
                        "sensor_name": "realsense_synthetic",
                        "scene_id": 1,
                        "split": "test",
                        "scene_folder": scene_folder.as_posix(),
                        "rgb_count": 2,
                    }
                ],
                "object_models": [
                    {
                        "object_name": "cube",
                        "obj_id": 7,
                        "source_path": "object_models/cube.ply",
                        "bop_path": "bop/models/obj_000007.ply",
                    }
                ],
            }
        )
    )
    return run_root


def test_synthetic_bop_results_write_csv_manifest_and_stage(tmp_path: Path) -> None:
    run_root = create_bop_export_fixture(tmp_path)

    path, manifest = write_synthetic_bop_results_with_manifest(run_root=run_root)

    assert path == run_root / BOP_RESULT_EXPORT_MANIFEST
    assert manifest.source_type == "synthetic"
    result_path = run_root / RESULTS_DIR / BOP_DIR / "synthetic_bop-test.csv"
    with open(result_path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]
    assert rows[1][0:4] == ["1", "0", "7", "1"]
    assert rows[2][0:4] == ["1", "1", "7", "1"]
    assert validate_bop19_result_file(result_path).row_count == 2
    targets = json.loads((run_root / BOP_DIR / BOP_TARGETS_BOP19).read_text())
    assert targets == [
        {"scene_id": 1, "im_id": 0, "obj_id": 7, "inst_count": 1},
        {"scene_id": 1, "im_id": 1, "obj_id": 7, "inst_count": 1},
    ]

    result_manifest = json.loads(path.read_text())
    assert result_manifest["results"][0]["filename"] == "synthetic_bop-test.csv"
    assert result_manifest["results"][0]["row_count"] == 2

    run_manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage
        for stage in run_manifest["stages"]
        if stage["name"] == "synthetic_bop_results"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][BOP_RESULT_EXPORT_MANIFEST] == BOP_RESULT_EXPORT_MANIFEST
    assert stage["artifacts"]["synthetic_bop-test.csv"] == (
        "results/bop/synthetic_bop-test.csv"
    )


def test_synthetic_bop_results_cli(tmp_path: Path) -> None:
    run_root = create_bop_export_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/create_synthetic_bop_results.py",
            run_root.as_posix(),
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Wrote 2 synthetic BOP result row" in result.stdout
    assert (run_root / RESULTS_DIR / BOP_DIR / "synthetic_bop-test.csv").is_file()
