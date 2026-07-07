"""Run-level coverage summaries for synchronized ArUco pose outputs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from posetestbot.io.artifacts import (
    ARUCO_COVERAGE_REPORT,
    ARUCO_POSE_ESTIMATION,
    PROCESSED_DIR,
    SYNCHRONIZED_DIR,
)
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    upsert_stage,
    write_run_manifest,
)


SCHEMA_VERSION = "aruco_coverage_report.v1"


def _generated_at() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _check(
    name: str,
    status: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {"name": name, "status": status, "message": message}
    if details is not None:
        result["details"] = dict(details)
    return result


def _overall_status(checks: list[Mapping[str, Any]]) -> str:
    statuses = {str(check.get("status")) for check in checks}
    if "error" in statuses:
        return "error"
    if "warning" in statuses:
        return "warning"
    return "ok"


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_json(path: Path) -> object:
    with open(path, "r") as f:
        return json.load(f)


def discover_aruco_outputs(run_root: str | Path) -> list[Path]:
    root = Path(run_root)
    synchronized_root = root / PROCESSED_DIR / SYNCHRONIZED_DIR
    if not synchronized_root.is_dir():
        return []
    return sorted(synchronized_root.glob(f"*/{ARUCO_POSE_ESTIMATION}"))


def _vector(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _frame_coverage(frame: Mapping[str, Any], *, min_marker_count: int) -> dict[str, Any]:
    aruco = frame.get("aruco_pose_estimation")
    if not isinstance(aruco, Mapping):
        return {
            "marker_count": 0,
            "has_detection": False,
            "has_pose": False,
            "is_valid": False,
            "reason": "missing_aruco_pose_estimation",
        }
    marker_count = int(aruco.get("len_ids", 0) or 0)
    has_detection = marker_count > 0
    has_pose = _vector(aruco.get("rvec")) is not None and _vector(aruco.get("tvec")) is not None
    if marker_count < min_marker_count:
        reason = "insufficient_markers"
    elif not has_pose:
        reason = "invalid_pose"
    else:
        reason = None
    return {
        "marker_count": marker_count,
        "has_detection": has_detection,
        "has_pose": has_pose,
        "is_valid": reason is None,
        "reason": reason,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def build_aruco_coverage_report(
    run_root: str | Path,
    *,
    min_marker_count: int = 4,
    min_valid_pose_ratio: float = 0.0,
    aruco_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Summarize per-sensor ArUco detection and valid-pose coverage."""

    if min_marker_count < 1:
        raise ValueError("min_marker_count must be at least 1")
    if not 0.0 <= min_valid_pose_ratio <= 1.0:
        raise ValueError("min_valid_pose_ratio must be between 0 and 1")

    root = Path(run_root)
    paths = (
        [Path(path) for path in aruco_paths]
        if aruco_paths is not None
        else discover_aruco_outputs(root)
    )
    checks: list[dict[str, Any]] = []
    sensors: list[dict[str, Any]] = []

    if not paths:
        checks.append(
            _check(
                "aruco_outputs_present",
                "error",
                "No synchronized aruco_pose_estimation.json files were found.",
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
                "aruco_outputs_present",
                "ok",
                f"Found {len(paths)} ArUco output file(s).",
                details={"file_count": len(paths)},
            )
        )

    for raw_path in paths:
        path = raw_path if raw_path.is_absolute() else root / raw_path
        sensor_name = path.parent.name
        try:
            frames = _read_json(path)
        except Exception as exc:
            checks.append(
                _check(
                    f"aruco_output_load:{_relative(path, root)}",
                    "error",
                    f"Could not load ArUco output {path}: {type(exc).__name__}: {exc}",
                    details={"path": path.as_posix()},
                )
            )
            continue
        if not isinstance(frames, Mapping):
            checks.append(
                _check(
                    f"aruco_output_shape:{_relative(path, root)}",
                    "error",
                    f"ArUco output must be a JSON object: {path}",
                    details={"path": path.as_posix()},
                )
            )
            continue

        frame_count = len(frames)
        detected_frame_count = 0
        pose_frame_count = 0
        valid_pose_count = 0
        missing_count = 0
        insufficient_marker_count = 0
        invalid_pose_count = 0
        max_marker_count = 0
        marker_counts: list[int] = []
        motion_names: set[str] = set()

        for frame in frames.values():
            if not isinstance(frame, Mapping):
                missing_count += 1
                continue
            motion = frame.get("motion")
            if isinstance(motion, str):
                motion_names.add(motion)
            coverage = _frame_coverage(frame, min_marker_count=min_marker_count)
            marker_count = int(coverage["marker_count"])
            marker_counts.append(marker_count)
            max_marker_count = max(max_marker_count, marker_count)
            if coverage["has_detection"]:
                detected_frame_count += 1
            if coverage["has_pose"]:
                pose_frame_count += 1
            if coverage["is_valid"]:
                valid_pose_count += 1
                continue
            reason = coverage["reason"]
            if reason == "missing_aruco_pose_estimation":
                missing_count += 1
            elif reason == "insufficient_markers":
                insufficient_marker_count += 1
            elif reason == "invalid_pose":
                invalid_pose_count += 1

        valid_pose_ratio = _ratio(valid_pose_count, frame_count)
        detection_ratio = _ratio(detected_frame_count, frame_count)
        sensor_status = (
            "ok"
            if frame_count and valid_pose_ratio >= min_valid_pose_ratio and valid_pose_count > 0
            else "warning"
        )
        sensor_summary = {
            "sensor_name": sensor_name,
            "aruco_pose_file": _relative(path, root),
            "frame_count": frame_count,
            "detected_frame_count": detected_frame_count,
            "pose_frame_count": pose_frame_count,
            "valid_pose_count": valid_pose_count,
            "missing_count": missing_count,
            "insufficient_marker_count": insufficient_marker_count,
            "invalid_pose_count": invalid_pose_count,
            "detection_ratio": detection_ratio,
            "valid_pose_ratio": valid_pose_ratio,
            "max_marker_count": max_marker_count,
            "mean_marker_count": (
                sum(marker_counts) / len(marker_counts) if marker_counts else 0.0
            ),
            "motions": sorted(motion_names),
        }
        sensors.append(sensor_summary)
        checks.append(
            _check(
                f"aruco_coverage:{sensor_name}",
                sensor_status,
                (
                    f"{sensor_name} has {valid_pose_count}/{frame_count} valid ArUco pose frame(s)."
                    if sensor_status == "ok"
                    else (
                        f"{sensor_name} has {valid_pose_count}/{frame_count} valid "
                        "ArUco pose frame(s); inspect marker visibility before calibration extraction."
                    )
                ),
                details={
                    "sensor_name": sensor_name,
                    "valid_pose_ratio": valid_pose_ratio,
                    "min_valid_pose_ratio": min_valid_pose_ratio,
                    "min_marker_count": min_marker_count,
                },
            )
        )

    frame_count = sum(int(sensor["frame_count"]) for sensor in sensors)
    detected_frame_count = sum(int(sensor["detected_frame_count"]) for sensor in sensors)
    pose_frame_count = sum(int(sensor["pose_frame_count"]) for sensor in sensors)
    valid_pose_count = sum(int(sensor["valid_pose_count"]) for sensor in sensors)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_root": root.as_posix(),
        "overall_status": _overall_status(checks),
        "min_marker_count": min_marker_count,
        "min_valid_pose_ratio": min_valid_pose_ratio,
        "source_file_count": len(paths),
        "sensor_count": len(sensors),
        "frame_count": frame_count,
        "detected_frame_count": detected_frame_count,
        "pose_frame_count": pose_frame_count,
        "valid_pose_count": valid_pose_count,
        "detection_ratio": _ratio(detected_frame_count, frame_count),
        "valid_pose_ratio": _ratio(valid_pose_count, frame_count),
        "checks": checks,
        "sensors": sensors,
    }


def aruco_coverage_report_path(run_root: str | Path) -> Path:
    return Path(run_root) / ARUCO_COVERAGE_REPORT


def write_aruco_coverage_report(
    run_root: str | Path,
    report: Mapping[str, Any],
) -> Path:
    path = aruco_coverage_report_path(run_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(report), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def write_aruco_coverage_report_with_manifest(
    run_root: str | Path,
    *,
    min_marker_count: int = 4,
    min_valid_pose_ratio: float = 0.0,
    aruco_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    root = Path(run_root)
    report = build_aruco_coverage_report(
        root,
        min_marker_count=min_marker_count,
        min_valid_pose_ratio=min_valid_pose_ratio,
        aruco_paths=aruco_paths,
    )
    path = write_aruco_coverage_report(root, report)
    manifest = load_or_create_run_manifest(root)
    upsert_stage(
        manifest,
        name="aruco_coverage",
        status="succeeded" if report["overall_status"] != "error" else "failed",
        artifacts={ARUCO_COVERAGE_REPORT: path},
        run_root=root,
        message=(
            f"ArUco coverage: {report['valid_pose_count']}/"
            f"{report['frame_count']} valid pose frame(s)."
        ),
    )
    write_run_manifest(manifest, root)
    return report
