"""Export legacy metric dashboard summaries as JSON, CSV, and XLSX reports."""

from __future__ import annotations

import csv
import json
import math
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from xml.sax.saxutils import escape

from posetestbot.io.artifact_browser import metric_dashboard_summary
from posetestbot.io.artifacts import (
    METRIC_REPORT_CSV,
    METRIC_REPORT_JSON,
    METRIC_REPORT_XLSX,
    METRICS_DIR,
    RESULTS_DIR,
)
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    upsert_stage,
    write_run_manifest,
)


SCHEMA_VERSION = "metric_report.v1"
METRIC_COLUMNS = (
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
)


@dataclass(frozen=True)
class MetricReportArtifacts:
    json_path: Path
    csv_path: Path
    xlsx_path: Path
    row_count: int


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def default_metric_report_folder(run_root: str | Path) -> Path:
    return Path(run_root) / RESULTS_DIR / METRICS_DIR


def _cell_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _metric_value(metrics: Mapping[str, Any], key: str) -> str:
    return _cell_value(metrics.get(key))


def metric_report_rows(dashboard: Mapping[str, Any]) -> list[dict[str, str]]:
    """Flatten dashboard metric rows for CSV/XLSX."""

    rows: list[dict[str, str]] = []
    for item in dashboard.get("direct_methods", []):
        if not isinstance(item, Mapping):
            continue
        metrics = item.get("all_motions", {})
        if not isinstance(metrics, Mapping):
            metrics = {}
        row = {
            "row_type": "direct_method",
            "context": "",
            "artifact_key": _cell_value(item.get("artifact_key")),
            "source": _cell_value(item.get("source")),
            "relative_path": _cell_value(item.get("relative_path")),
            "method": _cell_value(item.get("method")),
            "methods": _cell_value([item.get("method")] if item.get("method") else []),
            "motion_count": _cell_value(item.get("motion_count")),
            "motions": _cell_value(item.get("motions", [])),
            "sample_count": _cell_value(item.get("sample_count")),
            "best_method": _cell_value(item.get("method")),
            "best_AP_p": _metric_value(metrics, "AP_p"),
        }
        for key in METRIC_COLUMNS:
            row[key] = _metric_value(metrics, key)
        rows.append(row)

    for item in dashboard.get("combined_groups", []):
        if not isinstance(item, Mapping):
            continue
        best = item.get("best_by_AP_p")
        if not isinstance(best, Mapping):
            best = {}
        row = {
            "row_type": "combined_group",
            "context": _cell_value(item.get("context")),
            "artifact_key": _cell_value(item.get("artifact_key")),
            "source": _cell_value(item.get("source")),
            "relative_path": _cell_value(item.get("relative_path")),
            "method": "",
            "methods": _cell_value(item.get("methods", [])),
            "motion_count": "",
            "motions": "",
            "sample_count": "",
            "best_method": _cell_value(best.get("method")),
            "best_AP_p": _cell_value(best.get("AP_p")),
        }
        for key in METRIC_COLUMNS:
            row[key] = ""
        rows.append(row)

    for item in dashboard.get("bop_scores", []):
        if not isinstance(item, Mapping):
            continue
        metrics = item.get("metrics", {})
        if not isinstance(metrics, Mapping):
            metrics = {}
        row = {
            "row_type": "bop_toolkit_score",
            "context": "BOP Toolkit",
            "artifact_key": _cell_value(item.get("artifact_key")),
            "source": _cell_value(item.get("source")),
            "relative_path": _cell_value(item.get("relative_path")),
            "method": "",
            "methods": "",
            "motion_count": "",
            "motions": "",
            "sample_count": "",
            "best_method": "",
            "best_AP_p": "",
            "result_filename": _cell_value(item.get("result_filename")),
            "status": _cell_value(item.get("status")),
            "score_file_count": _cell_value(item.get("score_file_count")),
            "score_metric_count": _cell_value(item.get("score_metric_count")),
            "score_metrics": _cell_value(metrics),
            "bop19_average_recall": _metric_value(metrics, "bop19_average_recall"),
        }
        for key in METRIC_COLUMNS:
            row[key] = ""
        rows.append(row)
    return rows


def metric_report_columns(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    base_columns = [
        "row_type",
        "context",
        "artifact_key",
        "source",
        "relative_path",
        "method",
        "methods",
        "motion_count",
        "motions",
        "sample_count",
        "best_method",
        "best_AP_p",
        "result_filename",
        "status",
        "score_file_count",
        "score_metric_count",
        "bop19_average_recall",
        "score_metrics",
    ]
    columns = [*base_columns, *METRIC_COLUMNS]
    extra_columns = sorted(
        {
            str(key)
            for row in rows
            for key in row.keys()
            if str(key) not in columns
        }
    )
    return [*columns, *extra_columns]


def write_metric_csv(path: str | Path, rows: list[Mapping[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = metric_report_columns(rows)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _cell_value(row.get(column)) for column in columns})
    return output


def _column_name(index: int) -> str:
    name = ""
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        name = chr(ord("A") + remainder) + name
    return name


def _sheet_xml(rows: list[list[Any]]) -> str:
    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        "<sheetData>",
    ]
    for row_index, row in enumerate(rows, start=1):
        parts.append(f'<row r="{row_index}">')
        for column_index, value in enumerate(row):
            cell_ref = f"{_column_name(column_index)}{row_index}"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                parts.append(f'<c r="{cell_ref}"><v>{value}</v></c>')
            else:
                text = escape(_cell_value(value))
                parts.append(
                    f'<c r="{cell_ref}" t="inlineStr"><is><t>{text}</t></is></c>'
                )
        parts.append("</row>")
    parts.append("</sheetData></worksheet>")
    return "".join(parts)


def _workbook_xml(sheet_names: list[str]) -> str:
    sheets = "".join(
        f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, name in enumerate(sheet_names, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{sheets}</sheets></workbook>"
    )


def _workbook_rels_xml(sheet_names: list[str]) -> str:
    relationships = "".join(
        '<Relationship '
        f'Id="rId{index}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{index}.xml"/>'
        for index, _ in enumerate(sheet_names, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{relationships}</Relationships>"
    )


def _content_types_xml(sheet_count: int) -> str:
    sheet_overrides = "".join(
        '<Override '
        f'PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        f"{sheet_overrides}</Types>"
    )


def _root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )


def write_metric_xlsx(
    path: str | Path,
    *,
    dashboard: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = metric_report_columns(rows)
    best = dashboard.get("best_by_AP_p")
    if not isinstance(best, Mapping):
        best = {}
    bop_best = dashboard.get("best_bop19_average_recall")
    if not isinstance(bop_best, Mapping):
        bop_best = {}
    summary_rows = [
        ["field", "value"],
        ["schema_version", SCHEMA_VERSION],
        ["run_root", dashboard.get("run_root")],
        ["metric_artifact_count", dashboard.get("metric_artifact_count")],
        ["direct_method_count", dashboard.get("direct_method_count")],
        ["combined_group_count", dashboard.get("combined_group_count")],
        ["method_count", dashboard.get("method_count")],
        ["best_method", best.get("method")],
        ["best_AP_p", best.get("AP_p")],
        ["bop_score_count", dashboard.get("bop_score_count")],
        ["best_bop19_result", bop_best.get("result_filename")],
        ["best_bop19_average_recall", bop_best.get("bop19_average_recall")],
    ]
    method_rows = [columns] + [
        [_cell_value(row.get(column)) for column in columns] for row in rows
    ]
    artifact_columns = ["key", "source", "relative_path", "summary_type"]
    artifact_rows = [artifact_columns]
    for artifact in dashboard.get("artifacts", []):
        if not isinstance(artifact, Mapping):
            continue
        summary = artifact.get("summary")
        if not isinstance(summary, Mapping):
            summary = {}
        artifact_rows.append(
            [
                artifact.get("key"),
                artifact.get("source"),
                artifact.get("relative_path"),
                summary.get("type"),
            ]
        )

    sheets = {
        "summary": summary_rows,
        "methods": method_rows,
        "artifacts": artifact_rows,
    }
    sheet_names = list(sheets.keys())
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as xlsx:
        xlsx.writestr("[Content_Types].xml", _content_types_xml(len(sheet_names)))
        xlsx.writestr("_rels/.rels", _root_rels_xml())
        xlsx.writestr("xl/workbook.xml", _workbook_xml(sheet_names))
        xlsx.writestr("xl/_rels/workbook.xml.rels", _workbook_rels_xml(sheet_names))
        for index, rows_for_sheet in enumerate(sheets.values(), start=1):
            xlsx.writestr(f"xl/worksheets/sheet{index}.xml", _sheet_xml(rows_for_sheet))
    return output


def write_metric_reports(
    run_root: str | Path,
    *,
    output_folder: str | Path | None = None,
    group_limit: int = 200,
) -> MetricReportArtifacts:
    root = Path(run_root)
    folder = Path(output_folder) if output_folder is not None else default_metric_report_folder(root)
    dashboard = metric_dashboard_summary(root, group_limit=group_limit)
    rows = metric_report_rows(dashboard)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "dashboard": dashboard,
        "rows": rows,
    }

    json_path = folder / METRIC_REPORT_JSON
    csv_path = folder / METRIC_REPORT_CSV
    xlsx_path = folder / METRIC_REPORT_XLSX
    folder.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    write_metric_csv(csv_path, rows)
    write_metric_xlsx(xlsx_path, dashboard=dashboard, rows=rows)
    return MetricReportArtifacts(
        json_path=json_path,
        csv_path=csv_path,
        xlsx_path=xlsx_path,
        row_count=len(rows),
    )


def write_metric_reports_with_manifest(
    run_root: str | Path,
    *,
    output_folder: str | Path | None = None,
    group_limit: int = 200,
) -> MetricReportArtifacts:
    root = Path(run_root)
    artifacts = write_metric_reports(
        root,
        output_folder=output_folder,
        group_limit=group_limit,
    )
    manifest = load_or_create_run_manifest(root)
    upsert_stage(
        manifest,
        name="metric_report_export",
        status="succeeded",
        artifacts={
            METRIC_REPORT_JSON: artifacts.json_path,
            METRIC_REPORT_CSV: artifacts.csv_path,
            METRIC_REPORT_XLSX: artifacts.xlsx_path,
        },
        run_root=root,
        message=f"Exported {artifacts.row_count} metric report row(s).",
    )
    write_run_manifest(manifest, root)
    return artifacts
