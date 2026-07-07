"""Validate and explicitly promote calibration profile candidates."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from posetestbot.calibration.candidates import SCHEMA_VERSION as CANDIDATE_SCHEMA
from posetestbot.calibration.profiles import (
    CalibrationProfile,
    CalibrationStatus,
    load_profile_collection,
    profile_from_dict,
    profile_to_dict,
    write_profile_collection,
)
from posetestbot.io.artifacts import (
    CALIBRATION_CANDIDATES,
    CALIBRATION_PROFILES,
    CALIBRATION_PROFILES_FROM_OBSERVATIONS,
    CALIBRATION_PROFILES_SOLVED,
    CALIBRATION_SOLVER_REPORT,
    CALIBRATION_VALIDATION_REPORT,
)
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    upsert_stage,
    write_run_manifest,
)


SCHEMA_VERSION = "calibration_validation.v1"
SOLVER_SCHEMA_VERSION = "calibration_solver.v1"
DEFAULT_MIN_INLIERS = 6
DEFAULT_MAX_MEAN_TRANSLATION_RESIDUAL_MM = 10.0
DEFAULT_MAX_MEAN_ROTATION_RESIDUAL_DEG = 5.0
DEFAULT_MAX_OUTLIER_RATIO = 0.25


def _generated_at() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Calibration validation source must be a JSON object: {path}")
    return value


def _resolve_path(run_root: Path, value: str | Path | None, default_name: str) -> Path:
    path = Path(value) if value is not None else run_root / default_name
    return path if path.is_absolute() else run_root / path


def _load_candidate_profiles(
    report: Mapping[str, Any],
    profiles_path: Path,
) -> tuple[list[CalibrationProfile], str]:
    if profiles_path.is_file():
        return load_profile_collection(profiles_path), profiles_path.as_posix()
    profiles = [
        profile_from_dict(profile)
        for profile in report.get("profiles", [])
        if isinstance(profile, Mapping)
    ]
    return profiles, "embedded_candidates"


def _outlier_ratio(profile: CalibrationProfile) -> float:
    observations = profile.quality.num_observations
    if observations <= 0:
        return 0.0
    outlier_count = int(profile.metadata.get("outlier_count", 0) or 0)
    return outlier_count / observations


def _profile_validation_checks(
    profile: CalibrationProfile,
    *,
    min_inliers: int,
    max_mean_translation_residual_mm: float | None,
    max_mean_rotation_residual_deg: float | None,
    max_outlier_ratio: float | None,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    allowed_status = profile.status in {
        CalibrationStatus.NEEDS_VALIDATION,
        CalibrationStatus.VALID,
    }
    checks.append(
        _check(
            f"profile_status:{profile.profile_id}",
            "ok" if allowed_status else "error",
            (
                f"Profile {profile.profile_id} is eligible for validation."
                if allowed_status
                else (
                    f"Profile {profile.profile_id} status is "
                    f"{profile.status.value}, so it cannot be promoted."
                )
            ),
            details={
                "profile_id": profile.profile_id,
                "status": profile.status.value,
            },
        )
    )

    checks.append(
        _check(
            f"profile_inliers:{profile.profile_id}",
            "ok" if profile.quality.num_inliers >= min_inliers else "error",
            (
                f"Profile {profile.profile_id} has {profile.quality.num_inliers} inliers."
                if profile.quality.num_inliers >= min_inliers
                else (
                    f"Profile {profile.profile_id} has "
                    f"{profile.quality.num_inliers} inliers; minimum is "
                    f"{min_inliers}."
                )
            ),
            details={
                "profile_id": profile.profile_id,
                "num_inliers": profile.quality.num_inliers,
                "min_inliers": min_inliers,
            },
        )
    )

    translation_residual = profile.quality.residual_translation_mm
    if max_mean_translation_residual_mm is not None:
        translation_ok = (
            translation_residual is not None
            and float(translation_residual) <= max_mean_translation_residual_mm
        )
        checks.append(
            _check(
                f"profile_translation_residual:{profile.profile_id}",
                "ok" if translation_ok else "error",
                (
                    f"Profile {profile.profile_id} mean translation residual is "
                    f"{translation_residual} mm."
                    if translation_ok
                    else (
                        f"Profile {profile.profile_id} has no translation residual."
                        if translation_residual is None
                        else (
                            f"Profile {profile.profile_id} mean translation residual "
                            f"is {translation_residual} mm; threshold is "
                            f"{max_mean_translation_residual_mm} mm."
                        )
                    )
                ),
                details={
                    "profile_id": profile.profile_id,
                    "residual_translation_mm": translation_residual,
                    "max_mean_translation_residual_mm": max_mean_translation_residual_mm,
                },
            )
        )

    rotation_residual = profile.quality.residual_rotation_deg
    if max_mean_rotation_residual_deg is not None:
        rotation_ok = (
            rotation_residual is not None
            and float(rotation_residual) <= max_mean_rotation_residual_deg
        )
        checks.append(
            _check(
                f"profile_rotation_residual:{profile.profile_id}",
                "ok" if rotation_ok else "error",
                (
                    f"Profile {profile.profile_id} mean rotation residual is "
                    f"{rotation_residual} deg."
                    if rotation_ok
                    else (
                        f"Profile {profile.profile_id} has no rotation residual."
                        if rotation_residual is None
                        else (
                            f"Profile {profile.profile_id} mean rotation residual "
                            f"is {rotation_residual} deg; threshold is "
                            f"{max_mean_rotation_residual_deg} deg."
                        )
                    )
                ),
                details={
                    "profile_id": profile.profile_id,
                    "residual_rotation_deg": rotation_residual,
                    "max_mean_rotation_residual_deg": max_mean_rotation_residual_deg,
                },
            )
        )

    ratio = _outlier_ratio(profile)
    if max_outlier_ratio is not None:
        checks.append(
            _check(
                f"profile_outlier_ratio:{profile.profile_id}",
                "ok" if ratio <= max_outlier_ratio else "error",
                (
                    f"Profile {profile.profile_id} outlier ratio is {ratio:.3f}."
                    if ratio <= max_outlier_ratio
                    else (
                        f"Profile {profile.profile_id} outlier ratio is "
                        f"{ratio:.3f}; threshold is {max_outlier_ratio:.3f}."
                    )
                ),
                details={
                    "profile_id": profile.profile_id,
                    "outlier_ratio": ratio,
                    "max_outlier_ratio": max_outlier_ratio,
                },
            )
        )

    return checks


def build_calibration_validation(
    run_root: str | Path,
    *,
    candidates_path: str | Path | None = None,
    profiles_path: str | Path | None = None,
    min_inliers: int = DEFAULT_MIN_INLIERS,
    max_mean_translation_residual_mm: float | None = (
        DEFAULT_MAX_MEAN_TRANSLATION_RESIDUAL_MM
    ),
    max_mean_rotation_residual_deg: float | None = DEFAULT_MAX_MEAN_ROTATION_RESIDUAL_DEG,
    max_outlier_ratio: float | None = DEFAULT_MAX_OUTLIER_RATIO,
) -> dict[str, Any]:
    if min_inliers < 1:
        raise ValueError("min_inliers must be at least 1")
    if (
        max_mean_translation_residual_mm is not None
        and max_mean_translation_residual_mm < 0
    ):
        raise ValueError(
            "max_mean_translation_residual_mm must be greater than or equal to 0"
        )
    if max_mean_rotation_residual_deg is not None and max_mean_rotation_residual_deg < 0:
        raise ValueError(
            "max_mean_rotation_residual_deg must be greater than or equal to 0"
        )
    if max_outlier_ratio is not None and not 0 <= max_outlier_ratio <= 1:
        raise ValueError("max_outlier_ratio must be between 0 and 1")

    root = Path(run_root)
    candidate_path = _resolve_path(root, candidates_path, CALIBRATION_CANDIDATES)
    candidate_report = _read_json(candidate_path)
    report_schema = candidate_report.get("schema_version")
    if report_schema not in {CANDIDATE_SCHEMA, SOLVER_SCHEMA_VERSION}:
        raise ValueError(
            "Unsupported calibration candidate/solver schema: "
            f"{report_schema!r}"
        )
    default_profiles = (
        CALIBRATION_PROFILES_SOLVED
        if report_schema == SOLVER_SCHEMA_VERSION
        else CALIBRATION_PROFILES_FROM_OBSERVATIONS
    )
    profile_path = _resolve_path(root, profiles_path, default_profiles)
    profiles, profile_source = _load_candidate_profiles(candidate_report, profile_path)
    checks: list[dict[str, Any]] = []
    candidate_status = str(candidate_report.get("overall_status", "error"))
    checks.append(
        _check(
            "candidate_report_status",
            "ok" if candidate_status in {"ok", "warning"} else "error",
            (
                f"Candidate report status is {candidate_status}; validation "
                "thresholds will decide profile promotion."
                if candidate_status in {"ok", "warning"}
                else f"Candidate report status is {candidate_status}."
            ),
            details={
                "candidate_report_path": candidate_path.as_posix(),
                "candidate_report_status": candidate_status,
            },
        )
    )

    if not profiles:
        checks.append(
            _check(
                "candidate_profiles_present",
                "error",
                "No calibration candidate profiles were found.",
                details={"profiles_path": profile_path.as_posix()},
            )
        )

    profile_summaries: list[dict[str, Any]] = []
    for profile in profiles:
        profile_checks = _profile_validation_checks(
            profile,
            min_inliers=min_inliers,
            max_mean_translation_residual_mm=max_mean_translation_residual_mm,
            max_mean_rotation_residual_deg=max_mean_rotation_residual_deg,
            max_outlier_ratio=max_outlier_ratio,
        )
        checks.extend(profile_checks)
        profile_status = _overall_status(profile_checks)
        profile_summaries.append(
            {
                "profile_id": profile.profile_id,
                "sensor_id": profile.sensor_id,
                "sensor_type": profile.sensor_type.value,
                "mounting_mode": profile.mounting_mode.value,
                "source_status": profile.status.value,
                "validation_status": profile_status,
                "promotable": profile_status == "ok",
                "num_observations": profile.quality.num_observations,
                "num_inliers": profile.quality.num_inliers,
                "outlier_count": int(profile.metadata.get("outlier_count", 0) or 0),
                "outlier_ratio": _outlier_ratio(profile),
                "residual_translation_mm": profile.quality.residual_translation_mm,
                "residual_rotation_deg": profile.quality.residual_rotation_deg,
            }
        )

    overall_status = _overall_status(checks)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_root": root.as_posix(),
        "overall_status": overall_status,
        "candidate_report_path": candidate_path.as_posix(),
        "candidate_report_status": candidate_status,
        "profile_source": profile_source,
        "profile_count": len(profiles),
        "promotable_profile_count": sum(
            1 for summary in profile_summaries if summary["promotable"]
        ),
        "candidate_count": int(candidate_report.get("candidate_count", 0) or 0),
        "inlier_count": int(candidate_report.get("inlier_count", 0) or 0),
        "outlier_count": int(candidate_report.get("outlier_count", 0) or 0),
        "thresholds": {
            "min_inliers": min_inliers,
            "max_mean_translation_residual_mm": max_mean_translation_residual_mm,
            "max_mean_rotation_residual_deg": max_mean_rotation_residual_deg,
            "max_outlier_ratio": max_outlier_ratio,
        },
        "promotion": {"requested": False, "promoted": False, "path": None},
        "checks": checks,
        "profiles": profile_summaries,
    }


def calibration_validation_report_path(run_root: str | Path) -> Path:
    return Path(run_root) / CALIBRATION_VALIDATION_REPORT


def _promoted_profiles(
    profiles: list[CalibrationProfile],
    report: Mapping[str, Any],
    *,
    operator: str | None,
) -> list[CalibrationProfile]:
    generated_at = str(report["generated_at"])
    validation_report_path = str(report["candidate_report_path"])
    promoted = []
    for profile in profiles:
        metadata = dict(profile.metadata)
        metadata.update(
            {
                "validated_from_status": profile.status.value,
                "validation_schema_version": SCHEMA_VERSION,
                "validation_generated_at": generated_at,
                "validation_candidate_report": validation_report_path,
            }
        )
        promoted.append(
            replace(
                profile,
                status=CalibrationStatus.VALID,
                calibrated_at=profile.calibrated_at or generated_at,
                operator=operator or profile.operator,
                metadata=metadata,
            )
        )
    return promoted


def write_calibration_validation_report(
    run_root: str | Path,
    report: Mapping[str, Any],
) -> Path:
    path = calibration_validation_report_path(run_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(report), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def write_calibration_validation_with_manifest(
    run_root: str | Path,
    *,
    candidates_path: str | Path | None = None,
    profiles_path: str | Path | None = None,
    min_inliers: int = DEFAULT_MIN_INLIERS,
    max_mean_translation_residual_mm: float | None = (
        DEFAULT_MAX_MEAN_TRANSLATION_RESIDUAL_MM
    ),
    max_mean_rotation_residual_deg: float | None = DEFAULT_MAX_MEAN_ROTATION_RESIDUAL_DEG,
    max_outlier_ratio: float | None = DEFAULT_MAX_OUTLIER_RATIO,
    promote: bool = False,
    output_profiles_path: str | Path | None = None,
    operator: str | None = None,
) -> tuple[Path, Path | None, dict[str, Any]]:
    run_root_path = Path(run_root)
    manifest = load_or_create_run_manifest(run_root_path)
    upsert_stage(manifest, name="calibration_validation", status="running")
    write_run_manifest(manifest, run_root_path)
    try:
        report = build_calibration_validation(
            run_root_path,
            candidates_path=candidates_path,
            profiles_path=profiles_path,
            min_inliers=min_inliers,
            max_mean_translation_residual_mm=max_mean_translation_residual_mm,
            max_mean_rotation_residual_deg=max_mean_rotation_residual_deg,
            max_outlier_ratio=max_outlier_ratio,
        )
        candidate_path = Path(report["candidate_report_path"])
        candidate_report = _read_json(candidate_path)
        report_schema = candidate_report.get("schema_version")
        default_profiles = (
            CALIBRATION_PROFILES_SOLVED
            if report_schema == SOLVER_SCHEMA_VERSION
            else CALIBRATION_PROFILES_FROM_OBSERVATIONS
        )
        profile_path = _resolve_path(
            run_root_path,
            profiles_path,
            default_profiles,
        )
        profiles, _profile_source = _load_candidate_profiles(candidate_report, profile_path)
        promoted_path: Path | None = None
        if promote and report["overall_status"] == "ok":
            output_path = _resolve_path(
                run_root_path,
                output_profiles_path,
                CALIBRATION_PROFILES,
            )
            promoted_profiles = _promoted_profiles(
                profiles,
                report,
                operator=operator,
            )
            promoted_path = write_profile_collection(promoted_profiles, output_path)
            report["promotion"] = {
                "requested": True,
                "promoted": True,
                "path": promoted_path.as_posix(),
                "profile_count": len(promoted_profiles),
            }
        elif promote:
            report["promotion"] = {
                "requested": True,
                "promoted": False,
                "path": None,
                "profile_count": 0,
            }

        report_path = write_calibration_validation_report(run_root_path, report)
        artifacts: dict[str, Path] = {CALIBRATION_VALIDATION_REPORT: report_path}
        if promoted_path is not None:
            artifacts[CALIBRATION_PROFILES] = promoted_path
        upsert_stage(
            manifest,
            name="calibration_validation",
            status="succeeded" if report["overall_status"] != "error" else "failed",
            artifacts=artifacts,
            run_root=run_root_path,
            message=f"Calibration validation status: {report['overall_status']}.",
        )
        write_run_manifest(manifest, run_root_path)
    except Exception as exc:
        upsert_stage(
            manifest,
            name="calibration_validation",
            status="failed",
            message=str(exc),
        )
        write_run_manifest(manifest, run_root_path)
        raise
    return report_path, promoted_path, report
