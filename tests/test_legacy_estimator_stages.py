from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from posetestbot.io.artifacts import DATASET_MANIFEST, MEGAPOSE_PLAN, SAM6D_PLAN


def create_estimator_fixture(tmp_path: Path) -> Path:
    run_root = tmp_path / "run-1"
    blenderproc_folder = (
        run_root
        / "processed"
        / "synchronized"
        / "realsense_123"
        / "blenderproc"
    )
    blenderproc_folder.mkdir(parents=True)
    (blenderproc_folder / "objects.json").write_text(
        json.dumps(
            {
                "cube": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "sphere": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            }
        )
    )
    return run_root


def test_megapose_stage_dry_run_writes_plan_and_manifest(tmp_path: Path) -> None:
    run_root = create_estimator_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_megapose_stage.py"),
            str(run_root),
            "--wrapper-script",
            "/opt/megapose_wrapper.py",
            "--model",
            "megapose-1.0-RGBD",
            "--roi-scale",
            "1.25",
            "--object-id",
            "1",
            "--result-id",
            "rgbd",
            "--dry-run",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Dry-run MegaPose plan created for 1 sensor folder" in result.stdout

    plan = json.loads((run_root / MEGAPOSE_PLAN).read_text())
    assert plan["schema_version"] == "megapose_plan.v1"
    assert plan["dry_run"] is True
    assert plan["estimator_id"] == "megapose"
    assert plan["wrapper_script"] == "/opt/megapose_wrapper.py"
    assert plan["wrapper_exists"] is False
    assert plan["object_id"] == 1
    assert plan["result_id"] == "rgbd"
    assert plan["options"] == {"model": "megapose-1.0-RGBD", "roi_scale": 1.25}
    assert plan["jobs"][0]["sensor_name"] == "realsense_123"
    assert plan["jobs"][0]["object_name"] == "sphere"
    assert plan["jobs"][0]["expected_output_folder"].endswith(
        "realsense_123/megapose_rgbd_obj1_output"
    )
    assert plan["command"][:4] == [
        "uv",
        "run",
        "python",
        "/opt/megapose_wrapper.py",
    ]
    assert "--model=megapose-1.0-RGBD" in plan["command"]
    assert "--ROI_scale=1.25" in plan["command"]
    assert "--object_id=1" in plan["command"]

    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(stage for stage in manifest["stages"] if stage["name"] == "megapose")
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][MEGAPOSE_PLAN] == MEGAPOSE_PLAN


def test_sam6d_stage_dry_run_writes_plan_and_manifest(tmp_path: Path) -> None:
    run_root = create_estimator_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_sam6d_stage.py"),
            str(run_root),
            "--wrapper-script",
            "/opt/sam6d_wrapper.py",
            "--segmentor-model",
            "sam-hq",
            "--object-id",
            "0",
            "--result-id",
            "sam-hq",
            "--dry-run",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Dry-run SAM6D plan created for 1 sensor folder" in result.stdout

    plan = json.loads((run_root / SAM6D_PLAN).read_text())
    assert plan["schema_version"] == "sam6d_plan.v1"
    assert plan["dry_run"] is True
    assert plan["estimator_id"] == "sam6d"
    assert plan["wrapper_script"] == "/opt/sam6d_wrapper.py"
    assert plan["wrapper_exists"] is False
    assert plan["object_id"] == 0
    assert plan["result_id"] == "sam-hq"
    assert plan["options"] == {"segmentor_model": "sam-hq"}
    assert plan["jobs"][0]["sensor_name"] == "realsense_123"
    assert plan["jobs"][0]["object_name"] == "cube"
    assert plan["jobs"][0]["expected_output_folder"].endswith(
        "realsense_123/sam6d_sam-hq_obj0_output"
    )
    assert plan["command"][:4] == [
        "uv",
        "run",
        "python",
        "/opt/sam6d_wrapper.py",
    ]
    assert "--segmentor_model=sam-hq" in plan["command"]
    assert "--object_id=0" in plan["command"]

    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(stage for stage in manifest["stages"] if stage["name"] == "sam6d")
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][SAM6D_PLAN] == SAM6D_PLAN


def test_megapose_stage_rejects_missing_object_id(tmp_path: Path) -> None:
    run_root = create_estimator_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_megapose_stage.py"),
            str(run_root),
            "--object-id",
            "9",
            "--dry-run",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "Object ID 9 is not present" in result.stderr
