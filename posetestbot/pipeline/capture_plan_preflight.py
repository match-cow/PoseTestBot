"""Capture-plan validation before launching any capture processes."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from posetestbot.io.atomic import atomic_write_json
from posetestbot.io.artifacts import (
    CAPTURE_PLAN,
    CAPTURE_PLAN_PREFLIGHT_REPORT,
    RAW_ROBOT_EE_POSES,
)
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    upsert_stage,
    write_run_manifest,
)
from posetestbot.pipeline.capture_plan import (
    build_capture_plan,
    capture_plan_path,
    write_capture_plan,
)
from posetestbot.pipeline.run_config import (
    capture_synchronization_from_mapping,
    load_run_config_for_run_root,
)
from posetestbot.sensors.registry import get_sensor_adapter, sensor_folder_name
from posetestbot.sensors.hardware_sync_qualification import (
    validate_hardware_sync_qualification,
)
from posetestbot.sensors.status import (
    REALSENSE_MIN_USB_MAJOR,
    collect_sensor_status,
    realsense_usb_major_version,
)


SCHEMA_VERSION = "capture_plan_preflight.v1"
VALID_COMMAND_ROLES = {"sensor_capture", "robot_pose_receiver"}


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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _script_path(command: list[str]) -> Path | None:
    if len(command) < 4:
        return None
    if command[:3] != ["uv", "run", "python"]:
        return None
    script = Path(command[3])
    return script if script.is_absolute() else _repo_root() / script


def _sensor_families(sensor_status: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    if not sensor_status:
        return {}
    return {
        str(family.get("sensor_type")): family
        for family in sensor_status.get("families", [])
        if isinstance(family, Mapping)
    }


def _sensor_devices(family: Mapping[str, Any]) -> set[str]:
    sensor_type = str(family.get("sensor_type") or "")
    return {
        str(device.get("device_id"))
        for device in family.get("devices", [])
        if (
            isinstance(device, Mapping)
            and device.get("connected", True)
            and device.get("capture_ready") is not False
            and not _realsense_device_below_superspeed(
                sensor_type=sensor_type,
                device=device,
            )
        )
    }


def _realsense_device_below_superspeed(
    *,
    sensor_type: str,
    device: Mapping[str, Any],
) -> bool:
    if sensor_type != "realsense_d435":
        return False
    metadata = device.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    usb_major = realsense_usb_major_version(metadata.get("usb_type_descriptor"))
    return usb_major is not None and usb_major < REALSENSE_MIN_USB_MAJOR


def _sensor_family_diagnostics(family: Mapping[str, Any]) -> list[dict[str, Any]]:
    diagnostics = family.get("diagnostics", [])
    if not isinstance(diagnostics, list):
        return []
    return [
        dict(diagnostic)
        for diagnostic in diagnostics
        if isinstance(diagnostic, Mapping)
    ]


def _sensor_check_details(
    *,
    sensor_type: str,
    device_id: str,
    family: Mapping[str, Any] | None = None,
    **details: Any,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "sensor_type": sensor_type,
        "device_id": device_id,
        **details,
    }
    if family is not None:
        diagnostics = _sensor_family_diagnostics(family)
        if diagnostics:
            value["diagnostics"] = diagnostics
    return value


def _is_auto_device(device_id: str) -> bool:
    return device_id.strip().lower() in {"", "auto", "default"}


def _enabled_config_sensors(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    capture = config.get("capture", {})
    sensors = capture.get("sensors", []) if isinstance(capture, Mapping) else []
    return [
        sensor
        for sensor in sensors
        if isinstance(sensor, Mapping) and sensor.get("enabled", True) is True
    ]


def _validate_adapter_capabilities(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    capture = config.get("capture", {})
    resolution = str(capture.get("resolution", "") if isinstance(capture, Mapping) else "")
    checks: list[dict[str, Any]] = []
    for sensor in _enabled_config_sensors(config):
        sensor_type = str(sensor.get("sensor_type", ""))
        device_id = str(sensor.get("device_id", ""))
        check_name = f"sensor_adapter:{sensor_type}:{device_id}"
        try:
            adapter = get_sensor_adapter(sensor_type)
        except ValueError as exc:
            checks.append(
                _check(
                    check_name,
                    "error",
                    str(exc),
                    details={"sensor_type": sensor_type, "device_id": device_id},
                )
            )
            continue
        supported_resolutions = list(adapter.supported_resolutions)
        resolution_supported = resolution in adapter.supported_resolutions
        checks.append(
            _check(
                check_name,
                "ok" if resolution_supported else "error",
                (
                    f"{adapter.display_name} supports configured resolution {resolution}."
                    if resolution_supported
                    else (
                        f"{adapter.display_name} does not support configured "
                        f"resolution {resolution!r}; supported: "
                        f"{', '.join(supported_resolutions)}."
                    )
                ),
                details={
                    "sensor_type": sensor_type,
                    "device_id": device_id,
                    "display_name": adapter.display_name,
                    "capture_script": adapter.capture_script,
                    "configured_resolution": resolution,
                    "supported_resolutions": supported_resolutions,
                },
            )
        )
    return checks


def _validate_output_folders(
    config: Mapping[str, Any],
    *,
    run_root: Path,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    folders: dict[str, list[Mapping[str, Any]]] = {}
    for sensor in _enabled_config_sensors(config):
        sensor_type = str(sensor.get("sensor_type", ""))
        device_id = str(sensor.get("device_id", ""))
        try:
            folder_name = sensor_folder_name(sensor_type, device_id)
        except ValueError as exc:
            checks.append(
                _check(
                    f"sensor_output_folder:{sensor_type}:{device_id}",
                    "error",
                    str(exc),
                    details={"sensor_type": sensor_type, "device_id": device_id},
                )
            )
            continue
        folders.setdefault(folder_name, []).append(sensor)
        folder_path = run_root / folder_name
        if not os.path.lexists(folder_path):
            checks.append(
                _check(
                    f"sensor_output_folder:{folder_name}",
                    "ok",
                    f"Planned sensor output folder is available: {folder_name}.",
                    details={"path": folder_path.as_posix(), "exists": False},
                )
            )
            continue
        child_count = sum(1 for _ in folder_path.iterdir()) if folder_path.is_dir() else 0
        checks.append(
            _check(
                f"sensor_output_folder:{folder_name}",
                "error",
                (
                    f"Planned sensor output folder already contains {child_count} item(s): {folder_name}."
                    if child_count
                    else (
                        "Planned sensor output folder already exists, even though "
                        f"it is empty: {folder_name}. Use a new run root."
                    )
                ),
                details={
                    "path": folder_path.as_posix(),
                    "exists": True,
                    "child_count": child_count,
                },
            )
        )

    for folder_name, sensors in sorted(folders.items()):
        if len(sensors) <= 1:
            continue
        checks.append(
            _check(
                f"sensor_output_folder_duplicate:{folder_name}",
                "error",
                f"Multiple configured sensors map to output folder {folder_name}.",
                details={
                    "folder": folder_name,
                    "sensor_count": len(sensors),
                    "sensors": [
                        {
                            "sensor_type": sensor.get("sensor_type"),
                            "device_id": sensor.get("device_id"),
                            "display_name": sensor.get("display_name"),
                        }
                        for sensor in sensors
                    ],
                },
            )
        )
    return checks


def _validate_raw_pose_output(run_root: Path) -> list[dict[str, Any]]:
    path = run_root / RAW_ROBOT_EE_POSES
    return [
        _check(
            "raw_robot_pose_output",
            "error" if path.exists() else "ok",
            (
                f"Raw robot pose artifact already exists: {path}. Use a new run root."
                if path.exists()
                else f"Raw robot pose output is available: {path}."
            ),
            details={"path": path.as_posix(), "exists": path.exists()},
        )
    ]


def _validate_command_shape(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    commands = plan.get("commands", [])
    if not isinstance(commands, list):
        return [
            _check(
                "command_shape",
                "error",
                "Capture plan commands must be a list.",
            )
        ]

    checks: list[dict[str, Any]] = []
    role_counts: dict[str, int] = {}
    for index, command in enumerate(commands):
        if not isinstance(command, Mapping):
            checks.append(
                _check(
                    f"command:{index}",
                    "error",
                    "Command entry must be an object.",
                )
            )
            continue
        role = str(command.get("role", ""))
        role_counts[role] = role_counts.get(role, 0) + 1
        command_array = command.get("command")
        if role not in VALID_COMMAND_ROLES:
            checks.append(
                _check(
                    f"command:{index}",
                    "error",
                    f"Unknown command role {role!r}.",
                    details={"role": role},
                )
            )
            continue
        if not isinstance(command_array, list) or not all(
            isinstance(item, str) for item in command_array
        ):
            checks.append(
                _check(
                    f"command:{role}:{index}",
                    "error",
                    "Command must be a list of strings.",
                )
            )
            continue
        if command_array[:3] != ["uv", "run", "python"]:
            checks.append(
                _check(
                    f"command:{role}:{index}",
                    "error",
                    "Command must start with 'uv run python'.",
                    details={"command": command_array},
                )
            )
            continue
        script_path = _script_path(command_array)
        exists = bool(script_path and script_path.is_file())
        checks.append(
            _check(
                f"command:{role}:{index}",
                "ok" if exists else "error",
                (
                    f"Command script exists: {script_path}"
                    if exists
                    else f"Command script is missing: {script_path}"
                ),
                details={"script": script_path.as_posix() if script_path else None},
            )
        )

    for role in ("robot_pose_receiver",):
        count = role_counts.get(role, 0)
        checks.append(
            _check(
                f"role:{role}",
                "ok" if count == 1 else "error",
                (
                    f"Found one {role} command."
                    if count == 1
                    else f"Expected one {role} command, found {count}."
                ),
                details={"count": count},
            )
        )

    return checks


def _validate_hardware_sync_plan(
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expose the exact synchronization guarantee accepted by preflight."""

    capture = config.get("capture")
    policy = capture_synchronization_from_mapping(
        capture.get("synchronization")
        if isinstance(capture, Mapping)
        else None
    ).to_dict()
    if policy["mode"] == "timestamp_aligned":
        return [
            _check(
                "capture_synchronization",
                "ok",
                "Capture uses timestamp alignment; no hardware exposure "
                "synchronization is claimed.",
                details={
                    "mode": "timestamp_aligned",
                    "depth_exposure_synchronized": False,
                    "rgb_exposure_synchronized": False,
                },
            )
        ]

    sensor_commands = [
        command
        for command in plan.get("commands", [])
        if isinstance(command, Mapping)
        and command.get("role") == "sensor_capture"
    ]
    assignments: dict[str, Mapping[str, Any]] = {}
    errors: list[str] = []
    for command in sensor_commands:
        assignment = command.get("hardware_sync")
        if not isinstance(assignment, Mapping):
            errors.append(f"{command.get('name')} has no hardware_sync assignment")
            continue
        sensor_key = str(assignment.get("sensor_key") or "")
        if not sensor_key:
            errors.append(f"{command.get('name')} has no hardware-sync sensor key")
            continue
        if sensor_key in assignments:
            errors.append(f"duplicate hardware-sync assignment for {sensor_key}")
        assignments[sensor_key] = assignment
        expected_role = (
            "master"
            if sensor_key == policy["master_sensor_key"]
            else "subordinate"
        )
        expected_mode = 1 if expected_role == "master" else 2
        expected = {
            "implementation": policy["implementation"],
            "group_id": policy["group_id"],
            "scope": policy["scope"],
            "role": expected_role,
            "inter_cam_sync_mode_expected": expected_mode,
            "depth_exposure_synchronized": True,
            "rgb_exposure_synchronized": False,
        }
        for field_name, expected_value in expected.items():
            if assignment.get(field_name) != expected_value:
                errors.append(
                    f"{sensor_key} {field_name}={assignment.get(field_name)!r}; "
                    f"expected {expected_value!r}"
                )
        command_array = command.get("command")
        expected_flags = (
            ("--hardware-sync-role", expected_role),
            ("--hardware-sync-group-id", str(policy["group_id"])),
            ("--hardware-sync-scope", str(policy["scope"])),
        )
        if isinstance(command_array, list):
            for flag, value in expected_flags:
                try:
                    flag_index = command_array.index(flag)
                except ValueError:
                    errors.append(f"{sensor_key} command is missing {flag}")
                    continue
                if (
                    flag_index + 1 >= len(command_array)
                    or command_array[flag_index + 1] != value
                ):
                    errors.append(
                        f"{sensor_key} command has an invalid value for {flag}"
                    )

    expected_sensor_keys = {
        f"{sensor['sensor_type']}:{sensor['device_id']}"
        for sensor in _enabled_config_sensors(config)
    }
    if set(assignments) != expected_sensor_keys:
        errors.append(
            "hardware-sync assignments do not exactly cover enabled sensors"
        )
    ordered_commands = sorted(
        enumerate(sensor_commands),
        key=lambda item: (
            int(item[1].get("startup_order") or 0),
            item[0],
        ),
    )
    if (
        not ordered_commands
        or (
            ordered_commands[0][1].get("hardware_sync") or {}
        ).get("role")
        != "master"
    ):
        errors.append("hardware-sync master is not the first camera to start")

    return [
        _check(
            "capture_synchronization",
            "error" if errors else "ok",
            (
                "Hardware-sync capture plan is inconsistent: "
                + "; ".join(errors)
                if errors
                else (
                    "RealSense inter-camera depth exposures are planned with "
                    "one explicit master and complete subordinate assignments; "
                    "D435 RGB exposure is not certified as synchronized."
                )
            ),
            details={
                **policy,
                "sensor_keys": sorted(assignments),
                "depth_exposure_synchronized": not errors,
                "rgb_exposure_synchronized": False,
                "limitations": [
                    "D435 RGB rolling-shutter exposure is timestamp-associated only.",
                    "USB OAK-D Pro and USB ZED 2i cannot join this trigger group.",
                ],
                "errors": errors,
            },
        )
    ]


def _validate_hardware_sync_qualification(
    config: Mapping[str, Any],
    *,
    run_root: Path,
) -> list[dict[str, Any]]:
    """Require physical, config-bound evidence before hardware-sync capture."""

    capture = config.get("capture")
    policy = capture_synchronization_from_mapping(
        capture.get("synchronization")
        if isinstance(capture, Mapping)
        else None
    )
    if policy.mode != "hardware_trigger":
        return []
    try:
        provenance = validate_hardware_sync_qualification(
            run_root,
            run_config=config,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        return [
            _check(
                "hardware_sync_qualification",
                "error",
                (
                    "Hardware-trigger capture requires a current physical "
                    f"depth-sync qualification: {exc}"
                ),
                details={
                    "valid": False,
                    "error": str(exc),
                    "depth_exposure_hardware_synchronized": False,
                    "rgb_exposure_hardware_synchronized": False,
                },
            )
        ]
    return [
        _check(
            "hardware_sync_qualification",
            "ok",
            (
                "Physical RealSense depth-exposure synchronization "
                "qualification is present, hash-verified, and bound to the "
                "current capture configuration."
            ),
            details={
                **provenance,
                "valid": True,
            },
        )
    ]


def _validate_robot_safety(
    *,
    allow_real_robot: bool,
) -> list[dict[str, Any]]:
    allowed = allow_real_robot is True
    checks = [
        _check(
            "real_robot_permission",
            "ok" if allowed else "error",
            (
                "Real robot use was explicitly allowed for this preflight."
                if allowed
                else "Real robot use requires allow_real_robot=true."
            ),
            details={"allow_real_robot": allow_real_robot},
        )
    ]
    return checks


def _validate_sensor_readiness(
    plan: Mapping[str, Any],
    *,
    sensor_status: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if sensor_status is None:
        return [
            _check(
                "sensor_status",
                "warning",
                "Sensor status was not collected for this capture-plan preflight.",
            )
        ]

    families = _sensor_families(sensor_status)
    checks = []
    sensors = [
        sensor
        for sensor in plan.get("sensors", [])
        if isinstance(sensor, Mapping)
    ]
    sensor_commands = [
        command
        for command in plan.get("commands", [])
        if isinstance(command, Mapping) and command.get("role") == "sensor_capture"
    ]
    checks.append(
        _check(
            "sensor_command_count",
            "ok" if len(sensor_commands) == len(sensors) else "error",
            (
                f"Found one sensor command for each planned sensor ({len(sensors)})."
                if len(sensor_commands) == len(sensors)
                else (
                    f"Found {len(sensor_commands)} sensor command(s) for "
                    f"{len(sensors)} planned sensor(s)."
                )
            ),
            details={
                "sensor_count": len(sensors),
                "sensor_command_count": len(sensor_commands),
            },
        )
    )

    for sensor in sensors:
        sensor_type = str(sensor.get("sensor_type"))
        device_id = str(sensor.get("device_id", ""))
        family = families.get(sensor_type)
        check_name = f"sensor:{sensor_type}:{device_id}"
        if family is None:
            checks.append(
                _check(
                    check_name,
                    "error",
                    f"No sensor status family found for {sensor_type}.",
                    details=_sensor_check_details(
                        sensor_type=sensor_type,
                        device_id=device_id,
                    ),
                )
            )
            continue
        if family.get("error"):
            checks.append(
                _check(
                    check_name,
                    "error",
                    f"Discovery error for {sensor_type}: {family['error']}",
                    details=_sensor_check_details(
                        sensor_type=sensor_type,
                        device_id=device_id,
                        family=family,
                        error=family["error"],
                    ),
                )
            )
            continue
        if not family.get("sdk_available", False):
            checks.append(
                _check(
                    check_name,
                    "error",
                    f"SDK module is not available for {sensor_type}.",
                    details=_sensor_check_details(
                        sensor_type=sensor_type,
                        device_id=device_id,
                        family=family,
                        sdk_module=family.get("sdk_module"),
                    ),
                )
            )
            continue
        devices = _sensor_devices(family)
        if _is_auto_device(device_id):
            ok = bool(devices)
            checks.append(
                _check(
                    check_name,
                    "ok" if ok else "error",
                    (
                        f"Auto device can select from {len(devices)} capture-ready {sensor_type} device(s)."
                        if ok
                        else f"No capture-ready {sensor_type} devices for auto selection."
                    ),
                    details=_sensor_check_details(
                        sensor_type=sensor_type,
                        device_id=device_id,
                        family=family,
                        connected_devices=sorted(devices),
                    ),
                )
            )
            continue
        checks.append(
            _check(
                check_name,
                "ok" if device_id in devices else "error",
                (
                    f"Configured device {device_id} is capture-ready."
                    if device_id in devices
                    else f"Configured device {device_id} is not capture-ready."
                ),
                details=_sensor_check_details(
                    sensor_type=sensor_type,
                    device_id=device_id,
                    family=family,
                    connected_devices=sorted(devices),
                ),
            )
        )

    return checks


def capture_plan_preflight_report_path(run_root: str | Path) -> Path:
    return Path(run_root) / CAPTURE_PLAN_PREFLIGHT_REPORT


def write_capture_plan_preflight_report(
    run_root: str | Path,
    report: Mapping[str, Any],
) -> Path:
    path = capture_plan_preflight_report_path(run_root)
    return atomic_write_json(path, dict(report))


def build_capture_plan_preflight(
    run_root: str | Path,
    *,
    include_sensor_status: bool = True,
    allow_real_robot: bool = False,
    collect_sensors: Callable[[], dict] = collect_sensor_status,
    write_plan_if_missing: bool = True,
) -> dict[str, Any]:
    """Build a capture readiness report without launching robot or camera processes."""

    run_root_path = Path(run_root)
    config = load_run_config_for_run_root(run_root_path)
    plan_file = capture_plan_path(run_root_path)
    pre_plan_checks = []
    pre_plan_checks.extend(_validate_adapter_capabilities(config))
    pre_plan_checks.extend(_validate_output_folders(config, run_root=run_root_path))
    pre_plan_checks.extend(_validate_raw_pose_output(run_root_path))
    qualification_checks = _validate_hardware_sync_qualification(
        config,
        run_root=run_root_path,
    )
    pre_plan_has_errors = any(check["status"] == "error" for check in pre_plan_checks)
    sensor_status = collect_sensors() if include_sensor_status else None

    plan: dict[str, Any] | None = None
    canonical_plan: dict[str, Any] | None = None
    plan_build_error: str | None = None
    if not pre_plan_has_errors:
        try:
            canonical_plan = build_capture_plan(config).to_dict()
        except ValueError as exc:
            plan_build_error = str(exc)
    if plan_file.exists():
        with open(plan_file, "r") as f:
            plan = json.load(f)
    elif pre_plan_has_errors:
        plan_build_error = (
            "Capture plan was not built because configuration capability checks failed."
        )
    elif canonical_plan is not None:
        if write_plan_if_missing:
            write_capture_plan(
                run_root_path,
                build_capture_plan(config),
            )
        plan = canonical_plan

    plan_matches_current_config = (
        plan is not None
        and canonical_plan is not None
        and plan == canonical_plan
    )

    checks = [
        *pre_plan_checks,
        *qualification_checks,
        _check(
            "capture_plan_build",
            "ok" if plan is not None else "error",
            (
                "Capture plan is available for launch preflight."
                if plan is not None
                else plan_build_error or "Capture plan could not be built."
            ),
        ),
        _check(
            "capture_plan_current_config",
            "ok" if plan_matches_current_config else "error",
            (
                "Capture plan exactly matches the current run configuration."
                if plan_matches_current_config
                else (
                    "Persisted capture_plan.json is stale or differs from the "
                    "canonical plan for the current run configuration; "
                    "regenerate it before capture."
                    if plan is not None and canonical_plan is not None
                    else "A canonical current-config capture plan is unavailable."
                )
            ),
        ),
        _check(
            "capture_plan_schema",
            "ok"
            if isinstance(plan, Mapping) and plan.get("schema_version") == "capture_plan.v1"
            else "error",
            (
                "Capture plan schema is capture_plan.v1."
                if isinstance(plan, Mapping) and plan.get("schema_version") == "capture_plan.v1"
                else (
                    "Capture plan schema is unavailable."
                    if plan is None
                    else f"Unsupported capture plan schema: {plan.get('schema_version')!r}."
                )
            ),
            details={"schema_version": plan.get("schema_version") if isinstance(plan, Mapping) else None},
        ),
        _check(
            "capture_plan_path",
            "ok" if plan_file.exists() else "warning",
            (
                f"Capture plan exists: {plan_file}"
                if plan_file.exists()
                else "Capture plan was built in memory but not present on disk."
            ),
            details={"path": plan_file.as_posix()},
        ),
    ]
    if isinstance(plan, Mapping):
        checks.extend(_validate_command_shape(plan))
        checks.extend(_validate_hardware_sync_plan(config, plan))
        checks.extend(
            _validate_robot_safety(
                allow_real_robot=allow_real_robot,
            )
        )
        checks.extend(
            _validate_sensor_readiness(
                plan,
                sensor_status=sensor_status,
            )
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_root": run_root_path.as_posix(),
        "overall_status": _overall_status(checks),
        "checks": checks,
        "config": config,
        "capture_plan": plan,
        "capture_plan_build_error": plan_build_error,
        "sensor_status": sensor_status,
        "allow_real_robot": allow_real_robot,
    }


def write_capture_plan_preflight_with_manifest(
    run_root: str | Path,
    *,
    include_sensor_status: bool = True,
    allow_real_robot: bool = False,
    collect_sensors: Callable[[], dict] = collect_sensor_status,
    write_plan_if_missing: bool = True,
) -> tuple[Path, dict[str, Any]]:
    run_root_path = Path(run_root)
    manifest = load_or_create_run_manifest(run_root_path)
    upsert_stage(manifest, name="capture_plan_preflight", status="running")
    write_run_manifest(manifest, run_root_path)
    try:
        report = build_capture_plan_preflight(
            run_root_path,
            include_sensor_status=include_sensor_status,
            allow_real_robot=allow_real_robot,
            collect_sensors=collect_sensors,
            write_plan_if_missing=write_plan_if_missing,
        )
        path = write_capture_plan_preflight_report(run_root_path, report)
        manifest["robot_profile"] = dict(report["config"].get("robot_profile") or {})
        manifest["capture_config"] = dict(report["config"].get("capture") or {})
        artifacts: dict[str, Path] = {CAPTURE_PLAN_PREFLIGHT_REPORT: path}
        plan_path = run_root_path / CAPTURE_PLAN
        if plan_path.is_file():
            artifacts[CAPTURE_PLAN] = plan_path
        upsert_stage(
            manifest,
            name="capture_plan_preflight",
            status="succeeded" if report["overall_status"] != "error" else "failed",
            artifacts=artifacts,
            run_root=run_root_path,
            message=f"Capture plan preflight status: {report['overall_status']}.",
        )
        write_run_manifest(manifest, run_root_path)
    except Exception as exc:
        upsert_stage(
            manifest,
            name="capture_plan_preflight",
            status="failed",
            message=str(exc),
        )
        write_run_manifest(manifest, run_root_path)
        raise
    return path, report
