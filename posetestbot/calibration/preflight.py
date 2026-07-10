"""Run-level calibration profile readiness checks."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from posetestbot.io.atomic import atomic_write_json
from posetestbot.calibration.profiles import (
    CalibrationProfile,
    CalibrationStatus,
    load_profile_collection,
    select_profile_for_sensor,
)
from posetestbot.io.artifacts import CALIBRATION_PREFLIGHT_REPORT, CALIBRATION_PROFILES
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    upsert_stage,
    write_run_manifest,
)
from posetestbot.pipeline.run_config import load_run_config_for_run_root
from posetestbot.sensors.contracts import MountingMode, SensorType
from posetestbot.sensors.registry import sensor_folder_name


SCHEMA_VERSION = "calibration_preflight.v1"


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


def _resolve_profile_path(run_root: Path, value: str | None) -> Path | None:
    if not value:
        default = run_root / CALIBRATION_PROFILES
        return default if default.is_file() else None
    path = Path(value)
    return path if path.is_absolute() else run_root / path


def _profile_by_id(profiles: list[CalibrationProfile]) -> dict[str, CalibrationProfile]:
    return {profile.profile_id: profile for profile in profiles}


def _quality_checks(
    profile: CalibrationProfile,
    *,
    sensor_name: str,
    require_valid: bool,
    min_observations: int,
    max_mean_reprojection_error_px: float | None,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    status_ok = profile.status == CalibrationStatus.VALID
    checks.append(
        _check(
            f"profile_status:{sensor_name}",
            "ok" if status_ok else ("error" if require_valid else "warning"),
            (
                f"Profile {profile.profile_id} is marked valid."
                if status_ok
                else f"Profile {profile.profile_id} status is {profile.status.value}."
            ),
            details={
                "profile_id": profile.profile_id,
                "status": profile.status.value,
                "require_valid": require_valid,
            },
        )
    )

    observations = profile.quality.num_observations
    checks.append(
        _check(
            f"profile_observations:{sensor_name}",
            "ok" if observations >= min_observations else "warning",
            (
                f"Profile {profile.profile_id} has {observations} observation(s)."
                if observations >= min_observations
                else (
                    f"Profile {profile.profile_id} has {observations} observation(s); "
                    f"recommended minimum is {min_observations}."
                )
            ),
            details={
                "profile_id": profile.profile_id,
                "num_observations": observations,
                "min_observations": min_observations,
            },
        )
    )

    mean_error = profile.quality.mean_reprojection_error_px
    if max_mean_reprojection_error_px is not None:
        has_metric = mean_error is not None
        ok = has_metric and float(mean_error) <= max_mean_reprojection_error_px
        checks.append(
            _check(
                f"profile_reprojection:{sensor_name}",
                "ok" if ok else "warning",
                (
                    f"Profile {profile.profile_id} mean reprojection error is {mean_error}px."
                    if ok
                    else (
                        f"Profile {profile.profile_id} has no mean reprojection metric."
                        if not has_metric
                        else (
                            f"Profile {profile.profile_id} mean reprojection error "
                            f"is {mean_error}px; threshold is "
                            f"{max_mean_reprojection_error_px}px."
                        )
                    )
                ),
                details={
                    "profile_id": profile.profile_id,
                    "mean_reprojection_error_px": mean_error,
                    "max_mean_reprojection_error_px": max_mean_reprojection_error_px,
                },
            )
        )
    return checks


def _sensor_match(
    sensor: Mapping[str, Any],
    profiles: list[CalibrationProfile],
) -> tuple[str, CalibrationProfile]:
    sensor_type = SensorType(str(sensor["sensor_type"]))
    device_id = str(sensor["device_id"])
    sensor_name = sensor_folder_name(sensor_type, device_id)
    explicit_profile_id = sensor.get("calibration_profile_id")
    if explicit_profile_id:
        by_id = _profile_by_id(profiles)
        try:
            return sensor_name, by_id[str(explicit_profile_id)]
        except KeyError as exc:
            raise KeyError(
                f"Configured profile {explicit_profile_id!r} does not exist"
            ) from exc
    mounting_mode = MountingMode(str(sensor.get("mounting_mode") or "eye_in_hand"))
    return sensor_name, select_profile_for_sensor(
        profiles,
        sensor_name,
        mounting_mode=mounting_mode,
    )


def build_calibration_preflight(
    run_root: str | Path,
    *,
    require_valid: bool = False,
    min_observations: int = 6,
    max_mean_reprojection_error_px: float | None = 2.0,
) -> dict[str, Any]:
    """Validate calibration profile coverage and quality for a saved run config."""

    if min_observations < 0:
        raise ValueError("min_observations cannot be negative")
    run_root_path = Path(run_root)
    config = load_run_config_for_run_root(run_root_path)
    profile_path = _resolve_profile_path(
        run_root_path,
        config.get("calibration_profiles"),
    )
    checks: list[dict[str, Any]] = []
    matched_sensors: list[dict[str, Any]] = []
    profiles: list[CalibrationProfile] = []

    if profile_path is None:
        checks.append(
            _check(
                "calibration_profiles_configured",
                "warning",
                "No calibration profile collection is configured or present.",
            )
        )
    elif not profile_path.is_file():
        checks.append(
            _check(
                "calibration_profiles_path",
                "error",
                f"Calibration profile collection does not exist: {profile_path}",
                details={"path": profile_path.as_posix()},
            )
        )
    else:
        try:
            profiles = load_profile_collection(profile_path)
            checks.append(
                _check(
                    "calibration_profiles_path",
                    "ok",
                    f"Loaded {len(profiles)} calibration profile(s).",
                    details={"path": profile_path.as_posix(), "profile_count": len(profiles)},
                )
            )
        except Exception as exc:
            checks.append(
                _check(
                    "calibration_profiles_load",
                    "error",
                    f"Could not load calibration profiles: {type(exc).__name__}: {exc}",
                    details={"path": profile_path.as_posix()},
                )
            )

    sensors = [
        sensor
        for sensor in config["capture"]["sensors"]
        if isinstance(sensor, Mapping) and bool(sensor.get("enabled", True))
    ]
    if profiles:
        for sensor in sensors:
            try:
                sensor_name, profile = _sensor_match(sensor, profiles)
                matched_sensors.append(
                    {
                        "sensor_name": sensor_name,
                        "sensor_type": sensor["sensor_type"],
                        "device_id": sensor["device_id"],
                        "mounting_mode": sensor.get("mounting_mode"),
                        "profile_id": profile.profile_id,
                        "profile_status": profile.status.value,
                    }
                )
                checks.append(
                    _check(
                        f"profile_match:{sensor_name}",
                        "ok",
                        f"Matched {sensor_name} to profile {profile.profile_id}.",
                        details={
                            "sensor_name": sensor_name,
                            "profile_id": profile.profile_id,
                        },
                    )
                )
                checks.extend(
                    _quality_checks(
                        profile,
                        sensor_name=sensor_name,
                        require_valid=require_valid,
                        min_observations=min_observations,
                        max_mean_reprojection_error_px=max_mean_reprojection_error_px,
                    )
                )
            except Exception as exc:
                sensor_name = sensor_folder_name(
                    str(sensor.get("sensor_type", "")),
                    str(sensor.get("device_id", "")),
                )
                checks.append(
                    _check(
                        f"profile_match:{sensor_name}",
                        "error",
                        f"No usable calibration profile for {sensor_name}: {exc}",
                        details={
                            "sensor_name": sensor_name,
                            "sensor_type": sensor.get("sensor_type"),
                            "device_id": sensor.get("device_id"),
                        },
                    )
                )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_root": run_root_path.as_posix(),
        "overall_status": _overall_status(checks),
        "checks": checks,
        "profile_path": profile_path.as_posix() if profile_path else None,
        "profile_count": len(profiles),
        "sensor_count": len(sensors),
        "matched_sensor_count": len(matched_sensors),
        "matched_sensors": matched_sensors,
        "require_valid": require_valid,
        "min_observations": min_observations,
        "max_mean_reprojection_error_px": max_mean_reprojection_error_px,
    }


def calibration_preflight_report_path(run_root: str | Path) -> Path:
    return Path(run_root) / CALIBRATION_PREFLIGHT_REPORT


def write_calibration_preflight_report(
    run_root: str | Path,
    report: Mapping[str, Any],
) -> Path:
    path = calibration_preflight_report_path(run_root)
    return atomic_write_json(path, dict(report))


def write_calibration_preflight_with_manifest(
    run_root: str | Path,
    *,
    require_valid: bool = False,
    min_observations: int = 6,
    max_mean_reprojection_error_px: float | None = 2.0,
) -> tuple[Path, dict[str, Any]]:
    run_root_path = Path(run_root)
    manifest = load_or_create_run_manifest(run_root_path)
    upsert_stage(manifest, name="calibration_preflight", status="running")
    write_run_manifest(manifest, run_root_path)
    try:
        report = build_calibration_preflight(
            run_root_path,
            require_valid=require_valid,
            min_observations=min_observations,
            max_mean_reprojection_error_px=max_mean_reprojection_error_px,
        )
        path = write_calibration_preflight_report(run_root_path, report)
        upsert_stage(
            manifest,
            name="calibration_preflight",
            status="succeeded" if report["overall_status"] != "error" else "failed",
            artifacts={CALIBRATION_PREFLIGHT_REPORT: path},
            run_root=run_root_path,
            message=f"Calibration preflight status: {report['overall_status']}.",
        )
        write_run_manifest(manifest, run_root_path)
    except Exception as exc:
        upsert_stage(
            manifest,
            name="calibration_preflight",
            status="failed",
            message=str(exc),
        )
        write_run_manifest(manifest, run_root_path)
        raise
    return path, report
