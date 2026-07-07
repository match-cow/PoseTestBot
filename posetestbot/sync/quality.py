"""Run-level quality checks for non-destructive synchronization output."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from posetestbot.io.artifacts import (
    PROCESSED_DIR,
    SYNC_QUALITY_REPORT,
    SYNC_REPORT,
    SYNCHRONIZED_DIR,
)
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    upsert_stage,
    write_run_manifest,
)


SCHEMA_VERSION = "sync_quality_report.v1"


def _check(
    name: str,
    status: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "message": message,
        "details": dict(details or {}),
    }


def _overall_status(checks: list[Mapping[str, Any]]) -> str:
    statuses = {str(check.get("status")) for check in checks}
    if "error" in statuses:
        return "error"
    if "warning" in statuses:
        return "warning"
    return "ok"


def _generated_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Sync report must be a JSON object: {path}")
    return value


def discover_sync_reports(run_root: str | Path) -> list[Path]:
    root = Path(run_root)
    sync_root = root / PROCESSED_DIR / SYNCHRONIZED_DIR
    if not sync_root.is_dir():
        return []
    return sorted(sync_root.glob(f"*/{SYNC_REPORT}"))


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _sensor_summary(
    report_path: Path,
    report: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    total_frames = int(report.get("total_frames", 0))
    matched_frames = int(report.get("matched_frames", 0))
    dropped_frames = int(report.get("dropped_frames", 0))
    match_ratio = (matched_frames / total_frames) if total_frames else 0.0
    motion_windows = report.get("motion_windows", {})
    return {
        "sensor_name": report_path.parent.name,
        "report_path": _relative(report_path, root),
        "sensor_folder": report.get("sensor_folder"),
        "output_folder": report.get("output_folder"),
        "timestamp_source": report.get("timestamp_source"),
        "sync_delta_ms": report.get("sync_delta_ms"),
        "total_frames": total_frames,
        "matched_frames": matched_frames,
        "dropped_frames": dropped_frames,
        "match_ratio": match_ratio,
        "motion_count": (
            len(motion_windows) if isinstance(motion_windows, Mapping) else 0
        ),
        "mean_abs_nearest_pose_delta_ns": report.get("mean_abs_nearest_pose_delta_ns"),
        "max_abs_nearest_pose_delta_ns": report.get("max_abs_nearest_pose_delta_ns"),
    }


def _sensor_checks(
    sensor: Mapping[str, Any],
    *,
    min_match_ratio: float,
    max_dropped_frames: int | None,
    max_nearest_pose_delta_ms: float | None,
    require_timestamp_source: str | None,
) -> list[dict[str, Any]]:
    name = str(sensor["sensor_name"])
    checks: list[dict[str, Any]] = []
    total_frames = int(sensor["total_frames"])
    matched_frames = int(sensor["matched_frames"])
    match_ratio = float(sensor["match_ratio"])

    checks.append(
        _check(
            f"sync_frames:{name}",
            "ok" if total_frames > 0 and matched_frames > 0 else "error",
            (
                f"{name} matched {matched_frames}/{total_frames} frame(s)."
                if total_frames > 0 and matched_frames > 0
                else f"{name} has no synchronized frames."
            ),
            details={"matched_frames": matched_frames, "total_frames": total_frames},
        )
    )
    checks.append(
        _check(
            f"sync_match_ratio:{name}",
            "ok" if match_ratio >= min_match_ratio else "warning",
            (
                f"{name} match ratio is {match_ratio:.3f}."
                if match_ratio >= min_match_ratio
                else (
                    f"{name} match ratio is {match_ratio:.3f}; "
                    f"recommended minimum is {min_match_ratio:.3f}."
                )
            ),
            details={"match_ratio": match_ratio, "min_match_ratio": min_match_ratio},
        )
    )

    dropped_frames = int(sensor["dropped_frames"])
    if max_dropped_frames is not None:
        checks.append(
            _check(
                f"sync_dropped_frames:{name}",
                "ok" if dropped_frames <= max_dropped_frames else "warning",
                (
                    f"{name} dropped {dropped_frames} frame(s)."
                    if dropped_frames <= max_dropped_frames
                    else (
                        f"{name} dropped {dropped_frames} frame(s); "
                        f"threshold is {max_dropped_frames}."
                    )
                ),
                details={
                    "dropped_frames": dropped_frames,
                    "max_dropped_frames": max_dropped_frames,
                },
            )
        )

    if max_nearest_pose_delta_ms is not None:
        max_delta_ns = sensor.get("max_abs_nearest_pose_delta_ns")
        threshold_ns = int(max_nearest_pose_delta_ms * 1_000_000)
        ok = max_delta_ns is not None and int(max_delta_ns) <= threshold_ns
        checks.append(
            _check(
                f"sync_nearest_pose_delta:{name}",
                "ok" if ok else "warning",
                (
                    f"{name} max nearest-pose delta is {max_delta_ns} ns."
                    if ok
                    else (
                        f"{name} has no nearest-pose delta metric."
                        if max_delta_ns is None
                        else (
                            f"{name} max nearest-pose delta is {max_delta_ns} ns; "
                            f"threshold is {threshold_ns} ns."
                        )
                    )
                ),
                details={
                    "max_abs_nearest_pose_delta_ns": max_delta_ns,
                    "max_nearest_pose_delta_ms": max_nearest_pose_delta_ms,
                },
            )
        )

    if require_timestamp_source:
        timestamp_source = str(sensor.get("timestamp_source"))
        checks.append(
            _check(
                f"sync_timestamp_source:{name}",
                "ok" if timestamp_source == require_timestamp_source else "warning",
                (
                    f"{name} used timestamp source {timestamp_source}."
                    if timestamp_source == require_timestamp_source
                    else (
                        f"{name} used timestamp source {timestamp_source}; "
                        f"expected {require_timestamp_source}."
                    )
                ),
                details={
                    "timestamp_source": timestamp_source,
                    "require_timestamp_source": require_timestamp_source,
                },
            )
        )
    return checks


def build_sync_quality_report(
    run_root: str | Path,
    *,
    min_match_ratio: float = 0.8,
    max_dropped_frames: int | None = None,
    max_nearest_pose_delta_ms: float | None = 50.0,
    require_timestamp_source: str | None = None,
    report_paths: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    if not 0.0 <= min_match_ratio <= 1.0:
        raise ValueError("min_match_ratio must be between 0 and 1")
    if max_dropped_frames is not None and max_dropped_frames < 0:
        raise ValueError("max_dropped_frames cannot be negative")
    if max_nearest_pose_delta_ms is not None and max_nearest_pose_delta_ms < 0:
        raise ValueError("max_nearest_pose_delta_ms cannot be negative")

    root = Path(run_root)
    paths = (
        [Path(path) for path in report_paths]
        if report_paths is not None
        else discover_sync_reports(root)
    )
    checks: list[dict[str, Any]] = []
    sensors: list[dict[str, Any]] = []

    if not paths:
        checks.append(
            _check(
                "sync_reports_present",
                "error",
                "No synchronized sync_report.json files were found.",
                details={
                    "expected_root": (
                        root / PROCESSED_DIR / SYNCHRONIZED_DIR
                    ).as_posix()
                },
            )
        )
    else:
        checks.append(
            _check(
                "sync_reports_present",
                "ok",
                f"Found {len(paths)} sync report(s).",
                details={"report_count": len(paths)},
            )
        )

    for path in paths:
        resolved = path if path.is_absolute() else root / path
        try:
            report = _read_json(resolved)
            sensor = _sensor_summary(resolved, report, root)
            sensors.append(sensor)
            checks.extend(
                _sensor_checks(
                    sensor,
                    min_match_ratio=min_match_ratio,
                    max_dropped_frames=max_dropped_frames,
                    max_nearest_pose_delta_ms=max_nearest_pose_delta_ms,
                    require_timestamp_source=require_timestamp_source,
                )
            )
        except Exception as exc:
            checks.append(
                _check(
                    f"sync_report_load:{_relative(resolved, root)}",
                    "error",
                    f"Could not read sync report {resolved}: {type(exc).__name__}: {exc}",
                    details={"path": resolved.as_posix()},
                )
            )

    total_frames = sum(int(sensor["total_frames"]) for sensor in sensors)
    matched_frames = sum(int(sensor["matched_frames"]) for sensor in sensors)
    dropped_frames = sum(int(sensor["dropped_frames"]) for sensor in sensors)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_root": root.as_posix(),
        "overall_status": _overall_status(checks),
        "checks": checks,
        "sensor_count": len(sensors),
        "total_frames": total_frames,
        "matched_frames": matched_frames,
        "dropped_frames": dropped_frames,
        "overall_match_ratio": (matched_frames / total_frames) if total_frames else 0.0,
        "min_match_ratio": min_match_ratio,
        "max_dropped_frames": max_dropped_frames,
        "max_nearest_pose_delta_ms": max_nearest_pose_delta_ms,
        "require_timestamp_source": require_timestamp_source,
        "sensors": sensors,
    }


def sync_quality_report_path(run_root: str | Path) -> Path:
    return Path(run_root) / SYNC_QUALITY_REPORT


def write_sync_quality_report(
    run_root: str | Path,
    report: Mapping[str, Any],
) -> Path:
    path = sync_quality_report_path(run_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(report), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def write_sync_quality_report_with_manifest(
    run_root: str | Path,
    *,
    min_match_ratio: float = 0.8,
    max_dropped_frames: int | None = None,
    max_nearest_pose_delta_ms: float | None = 50.0,
    require_timestamp_source: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    run_root_path = Path(run_root)
    manifest = load_or_create_run_manifest(run_root_path)
    upsert_stage(manifest, name="sync_quality", status="running")
    write_run_manifest(manifest, run_root_path)
    try:
        report = build_sync_quality_report(
            run_root_path,
            min_match_ratio=min_match_ratio,
            max_dropped_frames=max_dropped_frames,
            max_nearest_pose_delta_ms=max_nearest_pose_delta_ms,
            require_timestamp_source=require_timestamp_source,
        )
        path = write_sync_quality_report(run_root_path, report)
        upsert_stage(
            manifest,
            name="sync_quality",
            status="succeeded" if report["overall_status"] != "error" else "failed",
            artifacts={SYNC_QUALITY_REPORT: path},
            run_root=run_root_path,
            message=f"Sync quality status: {report['overall_status']}.",
        )
        write_run_manifest(manifest, run_root_path)
    except Exception as exc:
        upsert_stage(manifest, name="sync_quality", status="failed", message=str(exc))
        write_run_manifest(manifest, run_root_path)
        raise
    return path, report
