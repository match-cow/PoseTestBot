"""Run-owned physical qualification for RealSense inter-camera sync.

This module never opens a camera or contacts the robot.  It records and
validates operator-supplied physical evidence that is bound to the exact
mixed-mount D435 capture contract.  Adapter option readback and close
timestamps are intentionally insufficient: hardware-trigger capture remains
blocked until this separate qualification is present and valid.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from posetestbot.io.atomic import atomic_write_json
from posetestbot.io.artifacts import (
    CAPTURE_EXECUTION_LOGS_DIR,
    CAPTURE_EXECUTION_REPORT,
    CAPTURE_EXECUTION_STATUS,
    DATASET_MANIFEST,
    DEPTH_DIR,
    FRAME_METADATA_JSONL,
    HARDWARE_SYNC_QUALIFICATION,
    RAW_ROBOT_EE_POSES,
    RGB_DIR,
)
from posetestbot.pipeline.run_config import (
    load_run_config_for_run_root,
    normalize_inverted,
    normalize_mounting_mode,
    normalize_sensor_enabled,
    normalize_sensor_type,
    run_config_lock,
    validate_capture_synchronization,
)


SCHEMA_VERSION = "hardware_sync_qualification.v1"
CONTRACT_SCHEMA_VERSION = "hardware_sync_qualification_contract.v1"
EVIDENCE_DIR = "hardware_sync_qualification_evidence"
SUPPORTED_METHODS = frozenset(
    {
        "pulsed_light",
        "equivalent_exposure_timing",
    }
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SAFE_EVIDENCE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class HardwareSyncQualificationError(ValueError):
    """Raised when physical hardware-sync qualification cannot be trusted."""


def hardware_sync_qualification_path(run_root: str | Path) -> Path:
    """Return the canonical run-owned qualification artifact path."""

    return Path(run_root) / HARDWARE_SYNC_QUALIFICATION


def _required_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HardwareSyncQualificationError(f"{field} must be a JSON object")
    return value


def _required_text(value: Any, field: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HardwareSyncQualificationError(
            f"{field} must be a non-empty string"
        )
    normalized = value.strip()
    if len(normalized) > maximum:
        raise HardwareSyncQualificationError(
            f"{field} must be at most {maximum} characters"
        )
    return normalized


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _enabled_sensor_contracts(
    capture: Mapping[str, Any],
    master_sensor_key: str,
) -> list[dict[str, Any]]:
    raw_sensors = capture.get("sensors")
    if not isinstance(raw_sensors, Sequence) or isinstance(
        raw_sensors, str | bytes
    ):
        raise HardwareSyncQualificationError(
            "run_config.capture.sensors must be a JSON list"
        )
    sensors: list[dict[str, Any]] = []
    for index, raw_sensor in enumerate(raw_sensors):
        sensor = _required_mapping(
            raw_sensor,
            f"run_config.capture.sensors[{index}]",
        )
        if not normalize_sensor_enabled(sensor.get("enabled", True)):
            continue
        sensor_type = normalize_sensor_type(
            str(sensor.get("sensor_type", ""))
        ).value
        device_id = _required_text(
            sensor.get("device_id"),
            f"run_config.capture.sensors[{index}].device_id",
            maximum=128,
        )
        mounting_mode = normalize_mounting_mode(
            str(sensor.get("mounting_mode", ""))
        ).value
        sensor_key = f"{sensor_type}:{device_id}"
        sensors.append(
            {
                "sensor_key": sensor_key,
                "sensor_type": sensor_type,
                "device_id": device_id,
                "mounting_mode": mounting_mode,
                "inverted": normalize_inverted(
                    sensor.get("inverted", False)
                ),
                "hardware_sync_role": (
                    "master"
                    if sensor_key == master_sensor_key
                    else "subordinate"
                ),
            }
        )
    return sorted(sensors, key=lambda item: item["sensor_key"])


def hardware_sync_qualification_contract(
    run_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the canonical configuration subset a qualification certifies."""

    config = _required_mapping(run_config, "run_config")
    capture = _required_mapping(config.get("capture"), "run_config.capture")
    raw_sensors = capture.get("sensors")
    if not isinstance(raw_sensors, Sequence) or isinstance(
        raw_sensors, str | bytes
    ):
        raise HardwareSyncQualificationError(
            "run_config.capture.sensors must be a JSON list"
        )
    try:
        synchronization = validate_capture_synchronization(
            capture.get("synchronization"),
            list(raw_sensors),
        )
    except ValueError as exc:
        raise HardwareSyncQualificationError(str(exc)) from exc
    if synchronization.mode != "hardware_trigger":
        raise HardwareSyncQualificationError(
            "Physical hardware-sync qualification applies only to "
            "capture.synchronization.mode=hardware_trigger"
        )
    resolution = _required_text(
        capture.get("resolution"),
        "run_config.capture.resolution",
        maximum=64,
    )
    fps_value = capture.get("fps")
    if isinstance(fps_value, bool) or not isinstance(fps_value, int):
        raise HardwareSyncQualificationError(
            "run_config.capture.fps must be a positive integer"
        )
    if fps_value <= 0:
        raise HardwareSyncQualificationError(
            "run_config.capture.fps must be a positive integer"
        )
    master_sensor_key = str(synchronization.master_sensor_key)
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "run_config_schema_version": _required_text(
            config.get("schema_version"),
            "run_config.schema_version",
            maximum=64,
        ),
        "run_root": _required_text(
            config.get("run_root"),
            "run_config.run_root",
            maximum=4096,
        ),
        "capture": {
            "resolution": resolution,
            "fps": fps_value,
            "synchronization": synchronization.to_dict(),
            "sensors": _enabled_sensor_contracts(
                capture,
                master_sensor_key,
            ),
        },
    }


def hardware_sync_qualification_contract_sha256(
    run_config: Mapping[str, Any],
) -> str:
    """Hash the exact capture contract certified by the qualification."""

    return _canonical_sha256(hardware_sync_qualification_contract(run_config))


def _validated_observed_skew(
    value: Any,
    *,
    configured_maximum: float,
) -> float:
    if isinstance(value, bool):
        raise HardwareSyncQualificationError(
            "observed_max_depth_timestamp_skew_ms must be a finite "
            "non-negative number"
        )
    try:
        observed = float(value)
    except (TypeError, ValueError) as exc:
        raise HardwareSyncQualificationError(
            "observed_max_depth_timestamp_skew_ms must be a finite "
            "non-negative number"
        ) from exc
    if not math.isfinite(observed) or observed < 0:
        raise HardwareSyncQualificationError(
            "observed_max_depth_timestamp_skew_ms must be a finite "
            "non-negative number"
        )
    if observed > configured_maximum:
        raise HardwareSyncQualificationError(
            "Observed physical qualification skew "
            f"{observed} ms exceeds configured maximum "
            f"{configured_maximum} ms"
        )
    return observed


def _safe_evidence_name(source: Path, index: int) -> str:
    name = _SAFE_EVIDENCE_NAME.sub("_", source.name).strip("._")
    if not name:
        name = "evidence"
    return f"{index:03d}-{name[:180]}"


def _copy_evidence_file(source: Path, destination: Path) -> tuple[int, str]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        source_descriptor = os.open(source, flags)
    except OSError as exc:
        raise HardwareSyncQualificationError(
            f"Cannot open qualification evidence {source}: {exc}"
        ) from exc
    digest = hashlib.sha256()
    size = 0
    try:
        source_stat = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise HardwareSyncQualificationError(
                f"Qualification evidence must be a regular file: {source}"
            )
        with os.fdopen(source_descriptor, "rb", closefd=False) as source_handle:
            with open(destination, "xb") as destination_handle:
                while chunk := source_handle.read(1024 * 1024):
                    destination_handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
    finally:
        os.close(source_descriptor)
    return size, digest.hexdigest()


def _qualification_publication_blockers(run_root: Path) -> list[str]:
    """Find any evidence proving acquisition has already started."""

    blockers: set[str] = set()

    def add_if_present(path: Path) -> None:
        if os.path.lexists(path):
            blockers.add(path.relative_to(run_root).as_posix())

    for artifact in (
        RAW_ROBOT_EE_POSES,
        CAPTURE_EXECUTION_STATUS,
        CAPTURE_EXECUTION_REPORT,
        CAPTURE_EXECUTION_LOGS_DIR,
    ):
        add_if_present(run_root / artifact)
    raw_pose_stem = Path(RAW_ROBOT_EE_POSES).stem
    for candidate in run_root.iterdir():
        if candidate.name.startswith(raw_pose_stem):
            add_if_present(candidate)

    excluded_directories = {
        EVIDENCE_DIR,
        CAPTURE_EXECUTION_LOGS_DIR,
        "processed",
    }
    for candidate in run_root.iterdir():
        if candidate.name in excluded_directories:
            continue
        if not candidate.is_dir() and not candidate.is_symlink():
            continue
        for relative in (FRAME_METADATA_JSONL, RGB_DIR, DEPTH_DIR):
            add_if_present(candidate / relative)
    for relative in (FRAME_METADATA_JSONL, RGB_DIR, DEPTH_DIR):
        add_if_present(run_root / relative)

    manifest_path = run_root / DATASET_MANIFEST
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError):
            blockers.add(f"{DATASET_MANIFEST}#unreadable")
        else:
            stages = manifest.get("stages") if isinstance(manifest, dict) else None
            if isinstance(stages, list) and any(
                isinstance(stage, Mapping)
                and stage.get("name") == "capture_execution"
                for stage in stages
            ):
                blockers.add(f"{DATASET_MANIFEST}#capture_execution")
    return sorted(blockers)


def _assert_qualification_publication_allowed(run_root: Path) -> None:
    blockers = _qualification_publication_blockers(run_root)
    if blockers:
        raise HardwareSyncQualificationError(
            "Hardware-sync qualification is immutable once capture evidence "
            "exists; create a new run instead. Blockers: "
            + ", ".join(blockers)
        )


def record_hardware_sync_qualification(
    run_root: str | Path,
    *,
    operator: str,
    method: str,
    observed_max_depth_timestamp_skew_ms: float,
    evidence_paths: Sequence[str | Path],
    confirm_passed: bool,
) -> dict[str, Any]:
    """Serialize and publish qualification only before acquisition starts."""

    root = Path(run_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {root}")
    with run_config_lock(root) as locked_root:
        return _record_hardware_sync_qualification_locked(
            locked_root,
            operator=operator,
            method=method,
            observed_max_depth_timestamp_skew_ms=(
                observed_max_depth_timestamp_skew_ms
            ),
            evidence_paths=evidence_paths,
            confirm_passed=confirm_passed,
        )


def _record_hardware_sync_qualification_locked(
    root: Path,
    *,
    operator: str,
    method: str,
    observed_max_depth_timestamp_skew_ms: float,
    evidence_paths: Sequence[str | Path],
    confirm_passed: bool,
) -> dict[str, Any]:
    """Copy physical evidence and atomically publish a run-bound pass record."""

    if confirm_passed is not True:
        raise HardwareSyncQualificationError(
            "Recording a passed hardware-sync qualification requires the "
            "explicit confirm_passed flag"
        )
    operator_value = _required_text(operator, "operator")
    method_value = _required_text(method, "method", maximum=64)
    if method_value not in SUPPORTED_METHODS:
        raise HardwareSyncQualificationError(
            "method must be one of: " + ", ".join(sorted(SUPPORTED_METHODS))
        )
    if not isinstance(evidence_paths, Sequence) or isinstance(
        evidence_paths, str | bytes
    ) or not evidence_paths:
        raise HardwareSyncQualificationError(
            "At least one physical qualification evidence file is required"
        )

    _assert_qualification_publication_allowed(root)
    current_config = load_run_config_for_run_root(root)
    contract = hardware_sync_qualification_contract(current_config)
    contract_sha256 = _canonical_sha256(contract)
    configured_maximum = float(
        contract["capture"]["synchronization"][
            "max_depth_timestamp_skew_ms"
        ]
    )
    observed_skew = _validated_observed_skew(
        observed_max_depth_timestamp_skew_ms,
        configured_maximum=configured_maximum,
    )

    evidence_root = root / EVIDENCE_DIR
    if evidence_root.is_symlink():
        raise HardwareSyncQualificationError(
            "Managed hardware-sync qualification evidence directory must not "
            "be a symbolic link"
        )
    evidence_root.mkdir(parents=True, exist_ok=True)
    evidence_root_resolved = evidence_root.resolve()
    try:
        evidence_root_resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise HardwareSyncQualificationError(
            "Managed hardware-sync qualification evidence directory escapes "
            "the run root"
        ) from exc
    bundle_name = (
        f"{contract_sha256[:16]}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    bundle = evidence_root / bundle_name
    staging = evidence_root / f".{bundle_name}.{uuid.uuid4().hex}.tmp"
    staging.mkdir()
    evidence_records: list[dict[str, Any]] = []
    try:
        for index, raw_source in enumerate(evidence_paths, start=1):
            source = Path(raw_source)
            if source.is_symlink():
                raise HardwareSyncQualificationError(
                    f"Qualification evidence must not be a symbolic link: {source}"
                )
            try:
                source_resolved = source.resolve(strict=True)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Qualification evidence does not exist: {source}"
                ) from exc
            try:
                source_resolved.relative_to(evidence_root_resolved)
            except ValueError:
                pass
            else:
                raise HardwareSyncQualificationError(
                    "Qualification evidence source must not be inside the "
                    f"managed evidence directory: {source}"
                )
            destination = staging / _safe_evidence_name(source, index)
            size_bytes, sha256 = _copy_evidence_file(
                source_resolved,
                destination,
            )
            if size_bytes == 0:
                raise HardwareSyncQualificationError(
                    f"Qualification evidence must not be empty: {source}"
                )
            evidence_records.append(
                {
                    "path": (
                        Path(EVIDENCE_DIR)
                        / bundle_name
                        / destination.name
                    ).as_posix(),
                    "original_name": source.name,
                    "size_bytes": size_bytes,
                    "sha256": sha256,
                }
            )
        os.replace(staging, bundle)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # A concurrent config edit must never result in a newly published stale
    # pass.  The immutable evidence bundle may remain as an orphan for audit.
    final_config = load_run_config_for_run_root(root)
    if hardware_sync_qualification_contract(final_config) != contract:
        raise HardwareSyncQualificationError(
            "Run capture configuration changed while qualification evidence "
            "was being recorded; no qualification artifact was published"
        )
    _assert_qualification_publication_allowed(root)

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "qualified_at": datetime.now(timezone.utc).isoformat(),
        "operator": operator_value,
        "method": method_value,
        "observed_max_depth_timestamp_skew_ms": observed_skew,
        "configuration_contract": contract,
        "configuration_sha256": contract_sha256,
        "evidence": evidence_records,
        "claims": {
            "depth_exposure_hardware_synchronized": True,
            "rgb_exposure_hardware_synchronized": False,
        },
    }
    atomic_write_json(hardware_sync_qualification_path(root), artifact)
    validate_hardware_sync_qualification(root, run_config=final_config)
    return artifact


def _validate_evidence_record(
    record: Any,
    *,
    index: int,
    run_root: Path,
) -> dict[str, Any]:
    evidence = _required_mapping(record, f"evidence[{index}]")
    path_value = _required_text(
        evidence.get("path"),
        f"evidence[{index}].path",
        maximum=512,
    )
    relative = Path(path_value)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative.parts[0] != EVIDENCE_DIR
        or len(relative.parts) < 3
    ):
        raise HardwareSyncQualificationError(
            f"evidence[{index}].path must identify a managed run-relative file"
        )
    candidate = run_root / relative
    descendant = run_root
    for part in relative.parts:
        descendant /= part
        if descendant.is_symlink():
            raise HardwareSyncQualificationError(
                f"evidence[{index}].path must not traverse a symbolic link"
            )
    managed_evidence_root = run_root / EVIDENCE_DIR
    if managed_evidence_root.is_symlink():
        raise HardwareSyncQualificationError(
            "Managed hardware-sync qualification evidence directory must not "
            "be a symbolic link"
        )
    managed_evidence_root_resolved = managed_evidence_root.resolve()
    try:
        managed_evidence_root_resolved.relative_to(run_root.resolve())
    except ValueError as exc:
        raise HardwareSyncQualificationError(
            "Managed hardware-sync qualification evidence directory escapes "
            "the run root"
        ) from exc
    resolved = candidate.resolve()
    try:
        resolved.relative_to(managed_evidence_root_resolved)
    except ValueError as exc:
        raise HardwareSyncQualificationError(
            f"evidence[{index}].path escapes the managed evidence directory"
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Qualification evidence file does not exist: {candidate}"
        )
    size_value = evidence.get("size_bytes")
    if isinstance(size_value, bool) or not isinstance(size_value, int):
        raise HardwareSyncQualificationError(
            f"evidence[{index}].size_bytes must be a positive integer"
        )
    if size_value <= 0 or resolved.stat().st_size != size_value:
        raise HardwareSyncQualificationError(
            f"Qualification evidence size mismatch: {path_value}"
        )
    sha256 = str(evidence.get("sha256", ""))
    if not _SHA256_PATTERN.fullmatch(sha256):
        raise HardwareSyncQualificationError(
            f"evidence[{index}].sha256 must be a lowercase SHA-256 digest"
        )
    if _file_sha256(resolved) != sha256:
        raise HardwareSyncQualificationError(
            f"Qualification evidence SHA-256 mismatch: {path_value}"
        )
    return {
        "path": path_value,
        "original_name": _required_text(
            evidence.get("original_name"),
            f"evidence[{index}].original_name",
            maximum=255,
        ),
        "size_bytes": size_value,
        "sha256": sha256,
    }


def validate_hardware_sync_qualification(
    run_root: str | Path,
    *,
    run_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate physical evidence and return compact immutable provenance."""

    root = Path(run_root)
    path = hardware_sync_qualification_path(root)
    if path.is_symlink():
        raise HardwareSyncQualificationError(
            "Hardware-sync qualification artifact must not be a symbolic link"
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"Hardware-sync qualification does not exist: {path}"
        )
    raw_bytes = path.read_bytes()
    try:
        artifact = _required_mapping(
            json.loads(raw_bytes),
            "hardware-sync qualification",
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HardwareSyncQualificationError(
            f"Invalid hardware-sync qualification JSON: {exc}"
        ) from exc
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise HardwareSyncQualificationError(
            f"Hardware-sync qualification schema_version must be {SCHEMA_VERSION}"
        )
    if artifact.get("status") != "passed":
        raise HardwareSyncQualificationError(
            "Hardware-sync qualification status must be passed"
        )
    operator = _required_text(artifact.get("operator"), "operator")
    method = _required_text(artifact.get("method"), "method", maximum=64)
    if method not in SUPPORTED_METHODS:
        raise HardwareSyncQualificationError(
            "Hardware-sync qualification method is unsupported"
        )
    qualified_at = _required_text(
        artifact.get("qualified_at"),
        "qualified_at",
        maximum=64,
    )
    try:
        qualified_datetime = datetime.fromisoformat(qualified_at)
    except ValueError as exc:
        raise HardwareSyncQualificationError(
            "qualified_at must be an ISO-8601 timestamp"
        ) from exc
    if qualified_datetime.tzinfo is None:
        raise HardwareSyncQualificationError(
            "qualified_at must include a timezone"
        )

    config = (
        dict(run_config)
        if run_config is not None
        else load_run_config_for_run_root(root)
    )
    expected_contract = hardware_sync_qualification_contract(config)
    stored_contract = _required_mapping(
        artifact.get("configuration_contract"),
        "configuration_contract",
    )
    if dict(stored_contract) != expected_contract:
        raise HardwareSyncQualificationError(
            "Hardware-sync qualification is stale for the current capture "
            "configuration"
        )
    expected_contract_sha256 = _canonical_sha256(expected_contract)
    stored_contract_sha256 = str(artifact.get("configuration_sha256", ""))
    if (
        not _SHA256_PATTERN.fullmatch(stored_contract_sha256)
        or stored_contract_sha256 != expected_contract_sha256
    ):
        raise HardwareSyncQualificationError(
            "Hardware-sync qualification configuration SHA-256 is invalid"
        )
    configured_maximum = float(
        expected_contract["capture"]["synchronization"][
            "max_depth_timestamp_skew_ms"
        ]
    )
    observed_skew = _validated_observed_skew(
        artifact.get("observed_max_depth_timestamp_skew_ms"),
        configured_maximum=configured_maximum,
    )
    claims = _required_mapping(artifact.get("claims"), "claims")
    if dict(claims) != {
        "depth_exposure_hardware_synchronized": True,
        "rgb_exposure_hardware_synchronized": False,
    }:
        raise HardwareSyncQualificationError(
            "Hardware-sync qualification claims must certify depth only"
        )
    raw_evidence = artifact.get("evidence")
    if not isinstance(raw_evidence, list) or not raw_evidence:
        raise HardwareSyncQualificationError(
            "Hardware-sync qualification must reference physical evidence"
        )
    evidence = [
        _validate_evidence_record(item, index=index, run_root=root)
        for index, item in enumerate(raw_evidence)
    ]
    evidence_paths = [item["path"] for item in evidence]
    if len(evidence_paths) != len(set(evidence_paths)):
        raise HardwareSyncQualificationError(
            "Hardware-sync qualification evidence paths must be unique"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_path": HARDWARE_SYNC_QUALIFICATION,
        "artifact_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "status": "passed",
        "qualified_at": qualified_at,
        "operator": operator,
        "method": method,
        "observed_max_depth_timestamp_skew_ms": observed_skew,
        "configuration_sha256": expected_contract_sha256,
        "evidence": evidence,
        "depth_exposure_hardware_synchronized": True,
        "rgb_exposure_hardware_synchronized": False,
    }
