from __future__ import annotations

import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from posetestbot.evaluation.metric_reports import (
    METRIC_COLUMNS,
    metric_report_rows,
    write_metric_reports_with_manifest,
)
from posetestbot.io.artifacts import (
    ACCURACY_ARUCO_HRC_HUB,
    ACCURACY_HRC_HUB,
    BOP_EVALUATION_REPORT,
    DATASET_MANIFEST,
    METRIC_REPORT_CSV,
    METRIC_REPORT_JSON,
    METRIC_REPORT_XLSX,
    METRICS_DIR,
    RESULTS_DIR,
)
from posetestbot.io.artifact_browser import collect_run_artifacts
from posetestbot.io.manifest import load_run_manifest
from posetestbot.pipeline.recommendations import build_pipeline_recommendations
from posetestbot.pipeline.stages import build_pipeline_job


def create_metric_fixture(tmp_path: Path) -> Path:
    run_root = tmp_path / "run"
    sensor_root = run_root / "processed" / "synchronized" / "realsense_123"
    sensor_root.mkdir(parents=True)
    (sensor_root / ACCURACY_HRC_HUB).write_text(
        json.dumps(
            {
                "foundationpose": {
                    "motion_a": {"AP_p": 2.0, "x": [1.0, 2.0]},
                    "all_motions": {
                        "AP_p": 1.25,
                        "ap_x": 1.0,
                        "RP_i": 0.75,
                        "RP_a": [0.1, -0.1],
                        "x": [1.0, 2.0, 3.0],
                    },
                }
            }
        )
    )
    (sensor_root / ACCURACY_ARUCO_HRC_HUB).write_text(
        json.dumps(
            {
                "ArUco_accuracy": {
                    "all_motions": {
                        "AP_p": 2.5,
                        "RP_i": 1.5,
                        "x": [1.0],
                    }
                }
            }
        )
    )
    return run_root


def add_bop_score_fixture(run_root: Path) -> None:
    (run_root / BOP_EVALUATION_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "bop_evaluation_report.v1",
                "status": "succeeded",
                "dry_run": False,
                "result": {
                    "filename": "foundationpose_bop-test.csv",
                    "path": (
                        run_root / RESULTS_DIR / "bop" / "foundationpose_bop-test.csv"
                    ).as_posix(),
                },
                "checks": [
                    {"name": "result_file", "ok": True},
                    {"name": "bop_root", "ok": True},
                    {"name": "dataset_folder", "ok": True},
                    {"name": "targets_file", "ok": True},
                    {"name": "models_folder", "ok": True},
                    {"name": "models_info", "ok": True},
                    {"name": "model_files", "ok": True, "value": 1},
                    {"name": "eval_script", "ok": False},
                ],
                "output_artifacts": [],
                "score_summary": {
                    "score_file_count": 1,
                    "metrics": {
                        "bop19_average_recall": 0.75,
                        "nested.mspd": 0.6,
                    },
                    "files": [],
                },
            }
        )
    )


def test_metric_report_rows_flatten_dashboard_metrics(tmp_path: Path) -> None:
    run_root = create_metric_fixture(tmp_path)
    artifacts = write_metric_reports_with_manifest(run_root)
    report = json.loads(artifacts.json_path.read_text())

    rows = metric_report_rows(report["dashboard"])

    assert artifacts.row_count == 2
    assert rows[0]["row_type"] == "direct_method"
    assert rows[0]["method"] == "ArUco_accuracy"
    assert "AP_p" in METRIC_COLUMNS
    assert rows[1]["method"] == "foundationpose"
    assert rows[1]["AP_p"] == "1.25"


def test_metric_report_rows_include_bop_toolkit_scores(tmp_path: Path) -> None:
    run_root = create_metric_fixture(tmp_path)
    add_bop_score_fixture(run_root)

    artifacts = write_metric_reports_with_manifest(run_root)
    report = json.loads(artifacts.json_path.read_text())

    rows = metric_report_rows(report["dashboard"])

    assert artifacts.row_count == 3
    score_row = next(row for row in rows if row["row_type"] == "bop_toolkit_score")
    assert score_row["result_filename"] == "foundationpose_bop-test.csv"
    assert score_row["status"] == "succeeded"
    assert score_row["bop19_average_recall"] == "0.75"
    assert json.loads(score_row["score_metrics"]) == {
        "bop19_average_recall": 0.75,
        "nested.mspd": 0.6,
    }
    with open(artifacts.csv_path, newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert csv_rows[-1]["row_type"] == "bop_toolkit_score"
    assert csv_rows[-1]["bop19_average_recall"] == "0.75"
    with zipfile.ZipFile(artifacts.xlsx_path) as xlsx:
        summary_sheet = xlsx.read("xl/worksheets/sheet1.xml").decode()
    assert "bop_score_count" in summary_sheet
    assert "best_bop19_average_recall" in summary_sheet


def test_metric_report_export_writes_json_csv_xlsx_and_manifest(
    tmp_path: Path,
) -> None:
    run_root = create_metric_fixture(tmp_path)

    artifacts = write_metric_reports_with_manifest(run_root)

    assert artifacts.json_path == (
        run_root / RESULTS_DIR / METRICS_DIR / METRIC_REPORT_JSON
    )
    assert artifacts.csv_path == run_root / RESULTS_DIR / METRICS_DIR / METRIC_REPORT_CSV
    assert artifacts.xlsx_path == (
        run_root / RESULTS_DIR / METRICS_DIR / METRIC_REPORT_XLSX
    )
    report = json.loads(artifacts.json_path.read_text())
    assert report["schema_version"] == "metric_report.v1"
    assert report["dashboard"]["direct_method_count"] == 2

    with open(artifacts.csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["method"] == "ArUco_accuracy"
    assert rows[1]["method"] == "foundationpose"

    with zipfile.ZipFile(artifacts.xlsx_path) as xlsx:
        names = set(xlsx.namelist())
    assert "[Content_Types].xml" in names
    assert "xl/workbook.xml" in names
    assert "xl/worksheets/sheet1.xml" in names

    records = collect_run_artifacts(run_root)
    by_key_source = {(record.key, record.source): record for record in records}
    report_record = by_key_source[(METRIC_REPORT_JSON, "known")]
    assert report_record.summary["type"] == "metric_report"
    assert report_record.summary["row_count"] == 2
    assert report_record.summary["metric_report_ready_for_dashboard"] is True
    assert report_record.summary["metric_report_blocker"] is None
    assert "metric_report=ready" in report_record.to_dict()["display_label"]
    assert by_key_source[(METRIC_REPORT_CSV, "known")].summary["type"] == "csv"
    assert by_key_source[(METRIC_REPORT_XLSX, "known")].preview_type == "binary"

    manifest = load_run_manifest(run_root)
    assert manifest["artifacts"] == {}
    stage = next(
        stage for stage in manifest["stages"] if stage["name"] == "metric_report_export"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][METRIC_REPORT_JSON] == (
        f"{RESULTS_DIR}/{METRICS_DIR}/{METRIC_REPORT_JSON}"
    )
    assert stage["artifacts"][METRIC_REPORT_CSV] == (
        f"{RESULTS_DIR}/{METRICS_DIR}/{METRIC_REPORT_CSV}"
    )
    assert stage["artifacts"][METRIC_REPORT_XLSX] == (
        f"{RESULTS_DIR}/{METRICS_DIR}/{METRIC_REPORT_XLSX}"
    )


def test_metric_report_export_stage_cli_prints_json(tmp_path: Path) -> None:
    run_root = create_metric_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_metric_report_export_stage.py"),
            run_root.as_posix(),
            "--json",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["row_count"] == 2
    assert payload["json_path"].endswith(f"{RESULTS_DIR}/{METRICS_DIR}/{METRIC_REPORT_JSON}")
    assert (run_root / DATASET_MANIFEST).is_file()


def test_metric_report_pipeline_stage_and_recommendation(tmp_path: Path) -> None:
    run_root = create_metric_fixture(tmp_path)

    job = build_pipeline_job(stage_id="metric_report_export", run_root=run_root)
    assert job.command == [
        "uv",
        "run",
        "python",
        "scripts/run_metric_report_export_stage.py",
        run_root.as_posix(),
        "--group-limit",
        "200",
    ]
    assert job.resources == ["disk_io"]

    recommendations = build_pipeline_recommendations(run_root)
    recommendation = next(
        item
        for item in recommendations["recommendations"]
        if item["id"] == "export_metric_reports"
    )
    assert recommendation["stage_id"] == "metric_report_export"
    assert f"{RESULTS_DIR}/{METRICS_DIR}/{METRIC_REPORT_XLSX}" in (
        recommendation["expected_artifacts"]
    )
