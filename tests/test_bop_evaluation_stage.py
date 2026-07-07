from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from posetestbot.evaluation.bop_toolkit import (
    build_bop_evaluation_report,
    build_bop_evaluation_plan,
    validate_bop19_result_file,
    validate_bop_targets_file,
)
from posetestbot.io.artifacts import (
    BOP_DIR,
    BOP_EVALUATION_PLAN,
    BOP_EVALUATION_REPORT,
    BOP_TARGETS_BOP19,
    DATASET_MANIFEST,
    EVALUATION_DIR,
    MODELS_DIR,
)


def write_bop19_result(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "scene_id,im_id,obj_id,score,R,t,time",
                "1,0,1,0.9,1 0 0 0 1 0 0 0 1,10 20 30,0.05",
                "",
            ]
        )
    )


def create_bop_evaluation_fixture(
    tmp_path: Path,
    *,
    write_models: bool = True,
) -> tuple[Path, Path]:
    run_root = tmp_path / "run-1"
    bop_root = run_root / BOP_DIR
    bop_root.mkdir(parents=True)
    (bop_root / BOP_TARGETS_BOP19).write_text(
        json.dumps([{"scene_id": 1, "im_id": 0, "obj_id": 1, "inst_count": 1}])
    )
    if write_models:
        models_root = bop_root / MODELS_DIR
        models_root.mkdir()
        (models_root / "models_info.json").write_text(
            json.dumps({"1": {"source_name": "cube", "diameter": 10.0}})
        )
        (models_root / "obj_000001.ply").write_text(
            "\n".join(
                [
                    "ply",
                    "format ascii 1.0",
                    "element vertex 1",
                    "property float x",
                    "property float y",
                    "property float z",
                    "element face 0",
                    "property list uchar int vertex_indices",
                    "end_header",
                    "0 0 0",
                    "",
                ]
            )
        )
    result_file = run_root / "results" / "foundationpose_bop-test.csv"
    write_bop19_result(result_file)
    return run_root, result_file


def test_bop_evaluation_stage_dry_run_writes_plan_and_manifest(
    tmp_path: Path,
) -> None:
    run_root, result_file = create_bop_evaluation_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_bop_evaluation_stage.py"),
            str(run_root),
            "--result-file",
            str(result_file),
            "--bop-toolkit-root",
            "/opt/bop_toolkit",
            "--num-workers",
            "1",
            "--dry-run",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Dry-run BOP Toolkit evaluation plan created" in result.stdout

    plan = json.loads((run_root / BOP_EVALUATION_PLAN).read_text())
    assert plan["schema_version"] == "bop_evaluation_plan.v1"
    assert plan["dry_run"] is True
    assert plan["result"]["filename"] == "foundationpose_bop-test.csv"
    assert plan["result"]["method"] == "foundationpose"
    assert plan["result"]["dataset"] == "bop"
    assert plan["result"]["row_count"] == 1
    assert plan["environment"]["BOP_PATH"] == run_root.as_posix()
    assert plan["dataset_folder"] == (run_root / BOP_DIR).as_posix()
    assert plan["eval_path"] == (
        run_root / EVALUATION_DIR / "bop_toolkit" / "foundationpose_bop-test"
    ).as_posix()
    assert plan["command"][1] == "/opt/bop_toolkit/scripts/eval_bop19_pose.py"
    assert "--result_filenames=foundationpose_bop-test.csv" in plan["command"]
    assert f"--results_path={result_file.parent.as_posix()}" in plan["command"]
    assert f"--eval_path={plan['eval_path']}" in plan["command"]
    assert f"--targets_filename={BOP_TARGETS_BOP19}" in plan["command"]

    report = json.loads((run_root / BOP_EVALUATION_REPORT).read_text())
    assert report["schema_version"] == "bop_evaluation_report.v1"
    assert report["status"] == "planned"
    assert report["dry_run"] is True
    assert report["result"]["filename"] == "foundationpose_bop-test.csv"
    assert report["eval_path"] == plan["eval_path"]
    assert report["command"] == plan["command"]
    assert report["environment"] == plan["environment"]
    assert report["output_artifacts"] == []
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["result_file"]["ok"] is True
    assert checks["bop_root"]["ok"] is True
    assert checks["dataset_folder"]["ok"] is True
    assert checks["targets_file"]["ok"] is True
    assert checks["models_folder"]["ok"] is True
    assert checks["models_info"]["ok"] is True
    assert checks["model_files"]["ok"] is True
    assert checks["model_files"]["value"] == 1
    assert checks["eval_script"]["ok"] is False

    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(stage for stage in manifest["stages"] if stage["name"] == "bop_evaluation")
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][BOP_EVALUATION_PLAN] == BOP_EVALUATION_PLAN
    assert stage["artifacts"][BOP_EVALUATION_REPORT] == BOP_EVALUATION_REPORT
    assert stage["artifacts"]["bop_result_file"] == "results/foundationpose_bop-test.csv"
    assert stage["artifacts"]["bop_eval_path"] == (
        "evaluation/bop_toolkit/foundationpose_bop-test"
    )


def test_validate_bop19_result_file_rejects_bad_pose_rows(tmp_path: Path) -> None:
    bad_result = tmp_path / "foundationpose_bop-test.csv"
    bad_result.write_text(
        "\n".join(
            [
                "scene_id,im_id,obj_id,score,R,t,time",
                "1,0,1,0.9,1 0 0,10 20 30,0.05",
                "",
            ]
        )
    )

    with pytest.raises(ValueError, match="rotation must have 9 values"):
        validate_bop19_result_file(bad_result)


def test_bop_evaluation_plan_rejects_empty_targets_file(tmp_path: Path) -> None:
    run_root, result_file = create_bop_evaluation_fixture(tmp_path)
    (run_root / BOP_DIR / BOP_TARGETS_BOP19).write_text("[]\n")

    with pytest.raises(ValueError, match="no target rows"):
        build_bop_evaluation_plan(
            run_root=run_root,
            result_file=result_file,
            bop_toolkit_root="/opt/bop_toolkit",
            dry_run=True,
        )


def test_validate_bop_targets_file_rejects_bad_target_rows(tmp_path: Path) -> None:
    targets = tmp_path / BOP_TARGETS_BOP19
    targets.write_text(json.dumps([{"scene_id": 1, "im_id": 0, "obj_id": 1}]))

    with pytest.raises(ValueError, match="invalid 'inst_count'"):
        validate_bop_targets_file(targets)


def test_bop_evaluation_report_flags_missing_models(tmp_path: Path) -> None:
    run_root, result_file = create_bop_evaluation_fixture(
        tmp_path,
        write_models=False,
    )
    plan = build_bop_evaluation_plan(
        run_root=run_root,
        result_file=result_file,
        bop_toolkit_root="/opt/bop_toolkit",
        dry_run=True,
    )
    report = build_bop_evaluation_report(
        run_root=run_root,
        plan=plan,
        plan_path=run_root / BOP_EVALUATION_PLAN,
        status="planned",
        message="BOP Toolkit dry-run plan created.",
    ).to_dict()

    checks = {check["name"]: check for check in report["checks"]}
    assert checks["targets_file"]["ok"] is True
    assert checks["models_folder"]["ok"] is False
    assert checks["models_info"]["ok"] is False
    assert checks["model_files"]["ok"] is False
    assert checks["model_files"]["value"] == 0


def test_bop_evaluation_report_summarizes_score_outputs(tmp_path: Path) -> None:
    run_root, result_file = create_bop_evaluation_fixture(tmp_path)
    eval_path = run_root / EVALUATION_DIR / "bop_toolkit" / "foundationpose_bop-test"
    eval_path.mkdir(parents=True)
    (eval_path / "scores_bop19.json").write_text(
        json.dumps(
            {
                "bop19_average_recall": 0.75,
                "nested": {"mssd": 0.5},
                "ignored": [1, 2, 3],
            }
        )
    )
    plan = build_bop_evaluation_plan(
        run_root=run_root,
        result_file=result_file,
        eval_path=eval_path,
        bop_toolkit_root="/opt/bop_toolkit",
        dry_run=True,
    )
    plan_path = run_root / BOP_EVALUATION_PLAN
    plan_path.write_text(json.dumps(plan.to_dict()) + "\n")

    report = build_bop_evaluation_report(
        run_root=run_root,
        plan=plan,
        plan_path=plan_path,
        status="succeeded",
        message="BOP Toolkit evaluation completed.",
    ).to_dict()

    assert report["score_summary"]["score_file_count"] == 1
    assert report["score_summary"]["metrics"] == {
        "bop19_average_recall": 0.75,
        "nested.mssd": 0.5,
    }
    assert report["score_summary"]["files"][0]["relative_path"] == "scores_bop19.json"
