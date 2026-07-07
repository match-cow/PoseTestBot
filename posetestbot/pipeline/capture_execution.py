"""Capture execution planning from a validated capture plan."""

from __future__ import annotations

import json
import os
import signal
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from posetestbot.io.artifacts import (
    CAPTURE_EXECUTION_LOGS_DIR,
    CAPTURE_EXECUTION_PLAN,
    CAPTURE_EXECUTION_REPORT,
    CAPTURE_EXECUTION_STATUS,
    CAPTURE_PLAN,
    RAW_ROBOT_EE_POSES,
)
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    upsert_stage,
    write_run_manifest,
)
from posetestbot.pipeline.capture_plan_preflight import build_capture_plan_preflight
from posetestbot.sensors.status import collect_sensor_status


SCHEMA_VERSION = "capture_execution_plan.v1"
STATUS_SCHEMA_VERSION = "capture_execution_status.v1"
REPORT_SCHEMA_VERSION = "capture_execution_report.v1"
EXECUTION_MODES = ("plan_only", "pose_only_fake", "full")
POSE_ONLY_ROLES = {"robot_controller", "robot_pose_receiver"}


@dataclass(frozen=True)
class CaptureExecutionGate:
    """One readiness or operator-intent gate for capture execution."""

    name: str
    status: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["details"] = dict(self.details)
        return data


@dataclass(frozen=True)
class CaptureProcessRecord:
    """Execution metadata for one selected capture command."""

    role: str
    name: str
    command: list[str]
    command_text: str
    startup_order: int
    log_file: str
    pid: int | None = None
    started_at: str | None = None
    ended_at: str | None = None
    elapsed_s: float | None = None
    returncode: int | None = None
    status: str = "planned"
    termination_reason: str | None = None
    output_tail: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_tail"] = list(self.output_tail)
        return data


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _overall_status(gates: list[CaptureExecutionGate]) -> str:
    statuses = {gate.status for gate in gates}
    if "error" in statuses:
        return "error"
    if "warning" in statuses:
        return "warning"
    return "ok"


def _command_with_metadata(command: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    data = dict(command)
    data["plan_index"] = index
    command_array = data.get("command")
    if isinstance(command_array, list) and all(
        isinstance(item, str) for item in command_array
    ):
        data["command_text"] = shlex.join(command_array)
    return data


def _resources(commands: list[Mapping[str, Any]]) -> list[str]:
    resources: set[str] = set()
    for command in commands:
        for resource in command.get("resources", []):
            if isinstance(resource, str):
                resources.add(resource)
    return sorted(resources)


def _safe_log_stem(command: Mapping[str, Any], *, index: int) -> str:
    name = str(command.get("name") or command.get("role") or f"command_{index}")
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in name)
    return f"{index:02d}_{safe or 'command'}"


def _tail(path: Path, limit: int = 40) -> tuple[str, ...]:
    if not path.is_file():
        return ()
    return tuple(path.read_text(errors="replace").splitlines()[-limit:])


def _process_elapsed_s(info: Mapping[str, Any]) -> float | None:
    started = info.get("started_monotonic")
    if not isinstance(started, (int, float)):
        return None
    ended = info.get("ended_monotonic")
    if not isinstance(ended, (int, float)):
        ended = time.monotonic()
    return max(0.0, ended - started)


def _mark_process_ended(info: dict[str, Any]) -> None:
    if info.get("ended_at") is None:
        info["ended_at"] = _now()
    if info.get("ended_monotonic") is None:
        info["ended_monotonic"] = time.monotonic()


def _raw_pose_count(run_root: Path) -> int:
    path = run_root / RAW_ROBOT_EE_POSES
    if not path.is_file():
        return 0
    with open(path, "r") as f:
        value = json.load(f)
    return len(value) if isinstance(value, dict) else 0


def _terminate_process_group(
    process: subprocess.Popen,
    *,
    timeout_s: float,
) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.terminate()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
        process.wait(timeout=timeout_s)


def _robot_mode(preflight: Mapping[str, Any]) -> str:
    config = preflight.get("config")
    if not isinstance(config, Mapping):
        return "unknown"
    robot = config.get("robot_profile")
    if not isinstance(robot, Mapping):
        return "fake"
    return str(robot.get("mode") or "fake")


def _mode_gate(mode: str) -> CaptureExecutionGate:
    return CaptureExecutionGate(
        name="mode",
        status="ok" if mode in EXECUTION_MODES else "error",
        message=(
            f"Capture execution mode is {mode}."
            if mode in EXECUTION_MODES
            else f"Unsupported capture execution mode: {mode!r}."
        ),
        details={"mode": mode, "supported_modes": list(EXECUTION_MODES)},
    )


def _preflight_gate(
    preflight: Mapping[str, Any],
    *,
    mode: str,
) -> CaptureExecutionGate:
    preflight_status = str(preflight.get("overall_status", "error"))
    if mode == "pose_only_fake" and preflight_status == "warning":
        return CaptureExecutionGate(
            name="capture_plan_preflight",
            status="ok",
            message=(
                "Capture-plan preflight has warnings, but pose-only fake "
                "execution does not require camera readiness."
            ),
            details={"preflight_status": preflight_status},
        )
    return CaptureExecutionGate(
        name="capture_plan_preflight",
        status=preflight_status if preflight_status in {"ok", "warning"} else "error",
        message=f"Capture-plan preflight status is {preflight_status}.",
        details={"preflight_status": preflight_status},
    )


def _robot_gate(
    *,
    mode: str,
    robot_mode: str,
    allow_real_robot: bool,
) -> CaptureExecutionGate:
    if mode != "pose_only_fake":
        return CaptureExecutionGate(
            name="robot_mode",
            status="ok" if robot_mode == "fake" or allow_real_robot else "error",
            message=(
                "Fake robot mode is selected."
                if robot_mode == "fake"
                else (
                    "Real robot mode was explicitly allowed."
                    if allow_real_robot
                    else "Real robot mode requires allow_real_robot=true."
                )
            ),
            details={"robot_mode": robot_mode, "allow_real_robot": allow_real_robot},
        )
    return CaptureExecutionGate(
        name="robot_mode",
        status="ok" if robot_mode == "fake" else "error",
        message=(
            "Pose-only fake execution is using fake iiwa mode."
            if robot_mode == "fake"
            else "Pose-only fake execution refuses real iiwa mode."
        ),
        details={"robot_mode": robot_mode},
    )


def _camera_gate(
    *,
    mode: str,
    allow_cameras: bool,
) -> CaptureExecutionGate:
    if mode == "full":
        return CaptureExecutionGate(
            name="camera_permission",
            status="ok" if allow_cameras else "error",
            message=(
                "Camera process selection was explicitly allowed."
                if allow_cameras
                else "Full capture planning requires allow_cameras=true."
            ),
            details={"allow_cameras": allow_cameras},
        )
    return CaptureExecutionGate(
        name="camera_permission",
        status="ok",
        message="Camera capture commands are not selected in this mode.",
        details={"allow_cameras": allow_cameras},
    )


def _selection_for_mode(
    commands: list[Mapping[str, Any]],
    *,
    mode: str,
    allow_cameras: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], CaptureExecutionGate]:
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    if mode == "plan_only":
        for index, command in enumerate(commands):
            skipped_command = _command_with_metadata(command, index=index)
            skipped_command["skip_reason"] = "plan_only"
            skipped.append(skipped_command)
        return (
            selected,
            skipped,
            CaptureExecutionGate(
                name="command_selection",
                status="ok",
                message="No capture commands are selected in plan-only mode.",
                details={"selected_count": 0, "skipped_count": len(skipped)},
            ),
        )

    if mode == "full" and not allow_cameras:
        for index, command in enumerate(commands):
            skipped_command = _command_with_metadata(command, index=index)
            skipped_command["skip_reason"] = "camera_permission_required"
            skipped.append(skipped_command)
        return (
            selected,
            skipped,
            CaptureExecutionGate(
                name="command_selection",
                status="error",
                message=(
                    "No full-capture commands are selected until camera "
                    "execution is explicitly allowed."
                ),
                details={"selected_count": 0, "skipped_count": len(skipped)},
            ),
        )

    expected_roles = POSE_ONLY_ROLES if mode == "pose_only_fake" else None
    role_counts: dict[str, int] = {}
    for index, command in enumerate(commands):
        role = str(command.get("role", ""))
        role_counts[role] = role_counts.get(role, 0) + 1
        prepared = _command_with_metadata(command, index=index)
        if mode == "full" or role in POSE_ONLY_ROLES:
            selected.append(prepared)
        else:
            prepared["skip_reason"] = "camera_hardware_gated_for_pose_only_fake"
            skipped.append(prepared)

    missing_roles = []
    if expected_roles is not None:
        missing_roles = sorted(role for role in expected_roles if role_counts.get(role) != 1)

    status = "ok" if not missing_roles else "error"
    message = (
        "Selected fake iiwa controller and pose receiver commands."
        if mode == "pose_only_fake" and status == "ok"
        else (
            "Selected all capture-plan commands for full capture planning."
            if mode == "full"
            else f"Missing required pose-only command roles: {', '.join(missing_roles)}."
        )
    )
    return (
        selected,
        skipped,
        CaptureExecutionGate(
            name="command_selection",
            status=status,
            message=message,
            details={
                "selected_count": len(selected),
                "skipped_count": len(skipped),
                "missing_roles": missing_roles,
            },
        ),
    )


def build_capture_execution_plan(
    run_root: str | Path,
    *,
    mode: str = "pose_only_fake",
    allow_cameras: bool = False,
    allow_real_robot: bool = False,
    include_sensor_status: bool | None = None,
    collect_sensors: Callable[[], dict] = collect_sensor_status,
    write_plan_if_missing: bool = True,
) -> dict[str, Any]:
    """Build a non-executing command selection plan for capture startup."""

    if mode not in EXECUTION_MODES:
        raise ValueError(
            "mode must be one of: " + ", ".join(EXECUTION_MODES)
        )

    run_root_path = Path(run_root)
    if include_sensor_status is None:
        include_sensor_status = mode == "full"

    preflight = build_capture_plan_preflight(
        run_root_path,
        include_sensor_status=include_sensor_status,
        allow_real_robot=allow_real_robot,
        collect_sensors=collect_sensors,
        write_plan_if_missing=write_plan_if_missing,
    )
    capture_plan = preflight["capture_plan"]
    commands = [
        command
        for command in capture_plan.get("commands", [])
        if isinstance(command, Mapping)
    ]

    selected, skipped, selection_gate = _selection_for_mode(
        commands,
        mode=mode,
        allow_cameras=allow_cameras,
    )
    robot_mode = _robot_mode(preflight)
    gates = [
        _mode_gate(mode),
        _robot_gate(
            mode=mode,
            robot_mode=robot_mode,
            allow_real_robot=allow_real_robot,
        ),
        _camera_gate(mode=mode, allow_cameras=allow_cameras),
        _preflight_gate(preflight, mode=mode),
        selection_gate,
    ]
    status = _overall_status(gates)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "run_root": run_root_path.as_posix(),
        "mode": mode,
        "status": status,
        "message": (
            "Capture execution plan is ready."
            if status == "ok"
            else (
                "Capture execution plan has warnings."
                if status == "warning"
                else "Capture execution plan is blocked by safety gates."
            )
        ),
        "allow_cameras": allow_cameras,
        "allow_real_robot": allow_real_robot,
        "include_sensor_status": include_sensor_status,
        "ready_to_execute": status == "ok",
        "preflight_status": preflight.get("overall_status"),
        "selected_roles": [
            str(command.get("role"))
            for command in selected
            if isinstance(command.get("role"), str)
        ],
        "selected_resources": _resources(selected),
        "selected_commands": selected,
        "skipped_commands": skipped,
        "gates": [gate.to_dict() for gate in gates],
        "execution_strategy": {
            "supervisor": "planned_process_group",
            "working_directory": ".",
            "start_order": "ascending startup_order",
            "stop_policy": (
                "After robot_pose_receiver exits, terminate remaining selected "
                "camera/controller processes by process group."
            ),
        },
        "capture_plan": capture_plan,
        "preflight_report": preflight,
    }


def capture_execution_plan_path(run_root: str | Path) -> Path:
    return Path(run_root) / CAPTURE_EXECUTION_PLAN


def load_capture_execution_plan(run_root: str | Path) -> dict[str, Any]:
    path = capture_execution_plan_path(run_root)
    with open(path, "r") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Capture execution plan must be a JSON object: {path}")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported capture execution plan schema: "
            f"{value.get('schema_version')!r}"
        )
    return value


def write_capture_execution_plan(
    run_root: str | Path,
    plan: Mapping[str, Any],
) -> Path:
    path = capture_execution_plan_path(run_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(plan), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def write_capture_execution_plan_with_manifest(
    run_root: str | Path,
    *,
    mode: str = "pose_only_fake",
    allow_cameras: bool = False,
    allow_real_robot: bool = False,
    include_sensor_status: bool | None = None,
    collect_sensors: Callable[[], dict] = collect_sensor_status,
    write_plan_if_missing: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Write ``capture_execution_plan.json`` and record the stage."""

    run_root_path = Path(run_root)
    manifest = load_or_create_run_manifest(run_root_path)
    upsert_stage(manifest, name="capture_execution_plan", status="running")
    write_run_manifest(manifest, run_root_path)
    try:
        plan = build_capture_execution_plan(
            run_root_path,
            mode=mode,
            allow_cameras=allow_cameras,
            allow_real_robot=allow_real_robot,
            include_sensor_status=include_sensor_status,
            collect_sensors=collect_sensors,
            write_plan_if_missing=write_plan_if_missing,
        )
        path = write_capture_execution_plan(run_root_path, plan)
        config = plan["preflight_report"].get("config", {})
        manifest["robot_profile"] = dict(config.get("robot_profile") or {})
        manifest["capture_config"] = dict(config.get("capture") or {})
        upsert_stage(
            manifest,
            name="capture_execution_plan",
            status="succeeded" if plan["status"] != "error" else "failed",
            artifacts={
                CAPTURE_EXECUTION_PLAN: path,
                CAPTURE_PLAN: run_root_path / CAPTURE_PLAN,
            },
            run_root=run_root_path,
            message=f"Capture execution plan status: {plan['status']}.",
        )
        write_run_manifest(manifest, run_root_path)
    except Exception as exc:
        upsert_stage(
            manifest,
            name="capture_execution_plan",
            status="failed",
            message=str(exc),
        )
        write_run_manifest(manifest, run_root_path)
        raise
    return path, plan


def capture_execution_report_path(run_root: str | Path) -> Path:
    return Path(run_root) / CAPTURE_EXECUTION_REPORT


def capture_execution_status_path(run_root: str | Path) -> Path:
    return Path(run_root) / CAPTURE_EXECUTION_STATUS


def load_capture_execution_status(run_root: str | Path) -> dict[str, Any]:
    path = capture_execution_status_path(run_root)
    with open(path, "r") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Capture execution status must be a JSON object: {path}")
    if value.get("schema_version") != STATUS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported capture execution status schema: "
            f"{value.get('schema_version')!r}"
        )
    return value


def write_capture_execution_status(
    run_root: str | Path,
    status: Mapping[str, Any],
) -> Path:
    path = capture_execution_status_path(run_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(status), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def write_capture_execution_report(
    run_root: str | Path,
    report: Mapping[str, Any],
) -> Path:
    path = capture_execution_report_path(run_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(report), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def _command_array(command: Mapping[str, Any]) -> list[str]:
    value = command.get("command")
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Selected command has invalid command array: {command!r}")
    return list(value)


def _process_record(
    command: Mapping[str, Any],
    *,
    log_path: Path,
    pid: int | None,
    started_at: str | None,
    ended_at: str | None,
    elapsed_s: float | None,
    returncode: int | None,
    status: str,
    termination_reason: str | None = None,
) -> CaptureProcessRecord:
    command_array = _command_array(command)
    return CaptureProcessRecord(
        role=str(command.get("role") or ""),
        name=str(command.get("name") or ""),
        command=command_array,
        command_text=str(command.get("command_text") or shlex.join(command_array)),
        startup_order=int(command.get("startup_order") or 0),
        log_file=log_path.as_posix(),
        pid=pid,
        started_at=started_at,
        ended_at=ended_at,
        elapsed_s=elapsed_s,
        returncode=returncode,
        status=status,
        termination_reason=termination_reason,
        output_tail=_tail(log_path),
    )


def _status_process_record(info: Mapping[str, Any]) -> dict[str, Any]:
    command = info.get("command")
    if not isinstance(command, Mapping):
        command = {}
    command_array = command.get("command")
    if not isinstance(command_array, list) or not all(
        isinstance(item, str) for item in command_array
    ):
        command_array = []

    process = info.get("process")
    pid = info.get("pid")
    returncode = info.get("returncode")
    active = False
    if process is not None:
        pid = getattr(process, "pid", pid)
        try:
            polled = process.poll()
        except Exception:
            polled = getattr(process, "returncode", None)
        if returncode is None:
            returncode = polled
        active = polled is None and str(info.get("status")) == "running"
    else:
        active = str(info.get("status")) == "running"

    log_path = info.get("log_path")
    output_tail: tuple[str, ...] = ()
    if isinstance(log_path, Path):
        output_tail = _tail(log_path, limit=8)

    return {
        "role": str(command.get("role") or ""),
        "name": str(command.get("name") or ""),
        "command": command_array,
        "command_text": str(command.get("command_text") or shlex.join(command_array)),
        "startup_order": int(command.get("startup_order") or 0),
        "log_file": log_path.as_posix() if isinstance(log_path, Path) else None,
        "pid": pid if isinstance(pid, int) else None,
        "started_at": info.get("started_at"),
        "ended_at": info.get("ended_at"),
        "elapsed_s": _process_elapsed_s(info),
        "status": str(info.get("status") or "unknown"),
        "returncode": returncode,
        "termination_reason": info.get("termination_reason"),
        "active": active,
        "output_tail": list(output_tail),
    }


def _build_capture_execution_status(
    run_root: Path,
    *,
    status: str,
    message: str,
    mode: str,
    allow_cameras: bool,
    allow_real_robot: bool,
    started_monotonic: float,
    plan: Mapping[str, Any] | None,
    process_infos: list[dict[str, Any]],
    report_path: Path | None = None,
) -> dict[str, Any]:
    process_records = [_status_process_record(info) for info in process_infos]
    active_count = sum(1 for process in process_records if process["active"])
    data = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "generated_at": _now(),
        "run_root": run_root.as_posix(),
        "status": status,
        "message": message,
        "mode": mode,
        "allow_cameras": allow_cameras,
        "allow_real_robot": allow_real_robot,
        "elapsed_s": time.monotonic() - started_monotonic,
        "active_process_count": active_count,
        "process_count": len(process_records),
        "processes": process_records,
        "raw_pose_artifact": RAW_ROBOT_EE_POSES,
        "raw_pose_count": _raw_pose_count(run_root),
        "capture_execution_plan_artifact": CAPTURE_EXECUTION_PLAN,
        "capture_execution_report_artifact": (
            CAPTURE_EXECUTION_REPORT if report_path is not None else None
        ),
        "log_dir": (run_root / CAPTURE_EXECUTION_LOGS_DIR).as_posix(),
    }
    if isinstance(plan, Mapping):
        data["plan_status"] = plan.get("status")
        data["selected_roles"] = list(plan.get("selected_roles", []))
        data["ready_to_execute"] = bool(plan.get("ready_to_execute", False))
    return data


def _selected_commands_for_execution(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    selected = plan.get("selected_commands", [])
    if not isinstance(selected, list):
        raise ValueError("Capture execution plan selected_commands must be a list")
    commands = [dict(command) for command in selected if isinstance(command, Mapping)]
    return sorted(
        commands,
        key=lambda item: (
            int(item.get("startup_order") or 0),
            int(item.get("plan_index") or 0),
        ),
    )


def run_capture_execution(
    run_root: str | Path,
    *,
    mode: str = "pose_only_fake",
    allow_cameras: bool = False,
    allow_real_robot: bool = False,
    include_sensor_status: bool | None = None,
    timeout_s: float = 30.0,
    startup_wait_s: float = 0.2,
    terminate_timeout_s: float = 2.0,
    collect_sensors: Callable[[], dict] = collect_sensor_status,
    write_plan_if_missing: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Execute selected capture-plan commands with process-group supervision.

    The default ``pose_only_fake`` mode executes only fake iiwa and robot pose
    receiving. Camera-capable full execution must be selected explicitly through
    the plan gates.
    """

    if timeout_s <= 0:
        raise ValueError("timeout_s must be greater than 0")
    if startup_wait_s < 0:
        raise ValueError("startup_wait_s must be greater than or equal to 0")
    if terminate_timeout_s <= 0:
        raise ValueError("terminate_timeout_s must be greater than 0")

    run_root_path = Path(run_root)
    run_root_path.mkdir(parents=True, exist_ok=True)
    logs_dir = run_root_path / CAPTURE_EXECUTION_LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_or_create_run_manifest(run_root_path)
    upsert_stage(manifest, name="capture_execution", status="running")
    write_run_manifest(manifest, run_root_path)

    started_monotonic = time.monotonic()
    plan: dict[str, Any] | None = None
    process_infos: list[dict[str, Any]] = []
    background_processes: list[dict[str, Any]] = []
    status = "succeeded"
    message = "Capture execution completed successfully."
    report_path: Path | None = None

    def record_status(status_value: str, message_value: str) -> Path:
        return write_capture_execution_status(
            run_root_path,
            _build_capture_execution_status(
                run_root_path,
                status=status_value,
                message=message_value,
                mode=mode,
                allow_cameras=allow_cameras,
                allow_real_robot=allow_real_robot,
                started_monotonic=started_monotonic,
                plan=plan,
                process_infos=process_infos,
                report_path=report_path,
            ),
        )

    status_path = record_status("starting", "Capture execution supervisor starting.")

    try:
        plan = build_capture_execution_plan(
            run_root_path,
            mode=mode,
            allow_cameras=allow_cameras,
            allow_real_robot=allow_real_robot,
            include_sensor_status=include_sensor_status,
            collect_sensors=collect_sensors,
            write_plan_if_missing=write_plan_if_missing,
        )
        plan_path = write_capture_execution_plan(run_root_path, plan)
        if plan["status"] != "ok":
            raise RuntimeError(plan["message"])
        record_status("planning", "Capture execution plan accepted.")

        commands = _selected_commands_for_execution(plan)
        if not commands:
            raise RuntimeError("Capture execution plan selected no commands.")
        receiver_commands = [
            command for command in commands if command.get("role") == "robot_pose_receiver"
        ]
        if len(receiver_commands) != 1:
            raise RuntimeError(
                "Capture execution requires exactly one robot_pose_receiver "
                f"command; found {len(receiver_commands)}."
            )
        receiver_command = receiver_commands[0]
        receiver_order = int(receiver_command.get("startup_order") or 0)
        late_commands = [
            command
            for command in commands
            if command is not receiver_command
            and int(command.get("startup_order") or 0) > receiver_order
        ]
        if late_commands:
            names = ", ".join(str(command.get("name")) for command in late_commands)
            raise RuntimeError(
                "Capture execution does not yet support commands that start "
                f"after the pose receiver: {names}."
            )

        for index, command in enumerate(commands):
            if command is receiver_command:
                continue
            command_array = _command_array(command)
            log_path = logs_dir / f"{_safe_log_stem(command, index=index)}.log"
            log_file = open(log_path, "w", buffering=1)
            log_file.write(f"$ {shlex.join(command_array)}\n")
            process = subprocess.Popen(
                command_array,
                cwd=_repo_root(),
                env=os.environ.copy(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=(os.name != "nt"),
            )
            info = {
                "command": command,
                "log_path": log_path,
                "log_file": log_file,
                "process": process,
                "pid": getattr(process, "pid", None),
                "started_at": _now(),
                "started_monotonic": time.monotonic(),
                "ended_at": None,
                "ended_monotonic": None,
                "status": "running",
                "termination_reason": None,
            }
            background_processes.append(info)
            process_infos.append(info)
            record_status(
                "running",
                f"Started background capture command: {command.get('name')}.",
            )

        if startup_wait_s:
            time.sleep(startup_wait_s)
        for info in background_processes:
            process = info["process"]
            if process.poll() is not None:
                info["status"] = "failed"
                _mark_process_ended(info)
                info["termination_reason"] = "exited_before_receiver_start"
                raise RuntimeError(
                    f"Capture command exited before pose receiver start: "
                    f"{info['command'].get('name')}"
                )
        record_status("running", "Background capture commands are ready.")

        receiver_array = _command_array(receiver_command)
        receiver_index = commands.index(receiver_command)
        receiver_log = logs_dir / f"{_safe_log_stem(receiver_command, index=receiver_index)}.log"
        receiver_info = {
            "command": receiver_command,
            "log_path": receiver_log,
            "process": None,
            "pid": None,
            "started_at": _now(),
            "started_monotonic": time.monotonic(),
            "ended_at": None,
            "ended_monotonic": None,
            "returncode": None,
            "status": "running",
            "termination_reason": None,
        }
        process_infos.append(receiver_info)
        record_status("running", "Robot pose receiver is running.")
        with open(receiver_log, "w", buffering=1) as log_file:
            log_file.write(f"$ {shlex.join(receiver_array)}\n")
            result = subprocess.run(
                receiver_array,
                cwd=_repo_root(),
                env=os.environ.copy(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        receiver_info["returncode"] = result.returncode
        _mark_process_ended(receiver_info)
        receiver_info["status"] = "succeeded" if result.returncode == 0 else "failed"
        receiver_info["termination_reason"] = "receiver_completed"
        record_status(
            "running" if result.returncode == 0 else "failed",
            f"Robot pose receiver exited with status {result.returncode}.",
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Robot pose receiver exited with status "
                f"{result.returncode}."
            )

        for info in background_processes:
            process = info["process"]
            try:
                process.wait(timeout=terminate_timeout_s)
                _mark_process_ended(info)
                info["status"] = "succeeded" if process.returncode == 0 else "failed"
                info["termination_reason"] = "exited_after_receiver"
            except subprocess.TimeoutExpired:
                _terminate_process_group(process, timeout_s=terminate_timeout_s)
                _mark_process_ended(info)
                info["status"] = "stopped"
                info["termination_reason"] = "stopped_after_receiver_exit"

            if info.get("log_file") is not None:
                info["log_file"].close()
            record_status(
                "running",
                f"Background command finished: {info['command'].get('name')}.",
            )

            if (
                mode == "pose_only_fake"
                and info["command"].get("role") == "robot_controller"
                and process.returncode != 0
            ):
                raise RuntimeError(
                    "Fake iiwa controller exited with status "
                    f"{process.returncode}."
                )
    except Exception as exc:
        status = "failed"
        message = str(exc)
        for info in background_processes:
            process = info["process"]
            if process.poll() is None:
                _terminate_process_group(process, timeout_s=terminate_timeout_s)
                _mark_process_ended(info)
                info["status"] = "terminated"
                info["termination_reason"] = "failure_cleanup"
            elif info.get("status") == "running":
                _mark_process_ended(info)
                info["status"] = (
                    "succeeded" if process.returncode == 0 else "failed"
                )
                info["termination_reason"] = "exited_during_failure_cleanup"
            if info.get("log_file") is not None and not info["log_file"].closed:
                info["log_file"].close()
        record_status("failed", message)
        if plan is None:
            plan = {
                "schema_version": SCHEMA_VERSION,
                "run_root": run_root_path.as_posix(),
                "mode": mode,
                "status": "error",
                "message": message,
            }
        plan_path = run_root_path / CAPTURE_EXECUTION_PLAN

    process_records = []
    for info in process_infos:
        process = info.get("process")
        returncode = info.get("returncode")
        if process is not None:
            returncode = process.returncode
        process_records.append(
            _process_record(
                info["command"],
                log_path=info["log_path"],
                pid=info.get("pid") if isinstance(info.get("pid"), int) else None,
                started_at=info.get("started_at"),
                ended_at=info.get("ended_at"),
                elapsed_s=_process_elapsed_s(info),
                returncode=returncode,
                status=str(info.get("status") or "unknown"),
                termination_reason=info.get("termination_reason"),
            ).to_dict()
        )

    elapsed_s = time.monotonic() - started_monotonic
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": _now(),
        "run_root": run_root_path.as_posix(),
        "status": status,
        "message": message,
        "mode": mode,
        "allow_cameras": allow_cameras,
        "allow_real_robot": allow_real_robot,
        "timeout_s": timeout_s,
        "startup_wait_s": startup_wait_s,
        "terminate_timeout_s": terminate_timeout_s,
        "elapsed_s": elapsed_s,
        "raw_pose_artifact": RAW_ROBOT_EE_POSES,
        "raw_pose_count": _raw_pose_count(run_root_path),
        "log_dir": (run_root_path / CAPTURE_EXECUTION_LOGS_DIR).as_posix(),
        "supervisor_stop_policy": (
            "Background capture/controller commands are allowed to run while "
            "the robot pose receiver is active. After the receiver exits, the "
            "supervisor waits for them briefly and then stops remaining process "
            "groups."
        ),
        "capture_execution_plan_artifact": CAPTURE_EXECUTION_PLAN,
        "capture_execution_plan": plan,
        "processes": process_records,
    }
    report_path = write_capture_execution_report(run_root_path, report)
    status_path = record_status(status, message)

    config = {}
    if isinstance(plan, Mapping):
        preflight = plan.get("preflight_report")
        if isinstance(preflight, Mapping) and isinstance(preflight.get("config"), Mapping):
            config = dict(preflight["config"])
    manifest["robot_profile"] = dict(config.get("robot_profile") or {})
    manifest["capture_config"] = dict(config.get("capture") or {})
    artifacts: dict[str, str | Path] = {
        CAPTURE_EXECUTION_REPORT: report_path,
        CAPTURE_EXECUTION_PLAN: plan_path,
        CAPTURE_EXECUTION_STATUS: status_path,
        CAPTURE_EXECUTION_LOGS_DIR: logs_dir,
    }
    raw_pose_path = run_root_path / RAW_ROBOT_EE_POSES
    if raw_pose_path.is_file():
        artifacts[RAW_ROBOT_EE_POSES] = raw_pose_path
    upsert_stage(
        manifest,
        name="capture_execution",
        status="succeeded" if status == "succeeded" else "failed",
        artifacts=artifacts,
        run_root=run_root_path,
        message=message,
    )
    write_run_manifest(manifest, run_root_path)

    if status != "succeeded":
        raise RuntimeError(message)
    return report_path, report
