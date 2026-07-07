from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from posetestbot.io.artifacts import DATASET_MANIFEST, FOUNDATIONPOSE_PLAN


def create_foundationpose_fixture(tmp_path: Path) -> Path:
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


def test_foundationpose_stage_dry_run_writes_plan_and_manifest(
    tmp_path: Path,
) -> None:
    run_root = create_foundationpose_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_foundationpose_stage.py"),
            str(run_root),
            "--foundationpose-folder",
            "/opt/FoundationPose",
            "--no-tracking",
            "--object-id",
            "1",
            "--dry-run",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Dry-run FoundationPose plan created for 1 sensor folder" in result.stdout

    plan = json.loads((run_root / FOUNDATIONPOSE_PLAN).read_text())
    assert plan["schema_version"] == "foundationpose_plan.v1"
    assert plan["dry_run"] is True
    assert plan["foundationpose_folder"] == "/opt/FoundationPose"
    assert plan["no_tracking"] is True
    assert plan["object_id"] == 1
    assert plan["jobs"][0]["sensor_name"] == "realsense_123"
    assert plan["jobs"][0]["object_name"] == "sphere"
    assert plan["jobs"][0]["expected_output_folder"].endswith(
        "realsense_123/foundationposeNoTracking_est5_track2_obj1_output"
    )
    assert plan["command"][:4] == [
        "uv",
        "run",
        "python",
        "scripts/foundationpose_wrapper_multi.py",
    ]
    assert "--no_tracking=y" in plan["command"]
    assert "--object_id=1" in plan["command"]

    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(stage for stage in manifest["stages"] if stage["name"] == "foundationpose")
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][FOUNDATIONPOSE_PLAN] == FOUNDATIONPOSE_PLAN


def test_foundationpose_stage_rejects_missing_object_id(tmp_path: Path) -> None:
    run_root = create_foundationpose_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_foundationpose_stage.py"),
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
