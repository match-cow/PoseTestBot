"""Fake-iiwa capture rehearsal helpers."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from posetestbot.io.artifacts import CAPTURE_REHEARSAL_REPORT, RAW_ROBOT_EE_POSES
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    upsert_stage,
    write_run_manifest,
)
from posetestbot.pipeline.capture_plan import CapturePlan, build_capture_plan
from posetestbot.pipeline.run_config import validate_run_config


SCHEMA_VERSION = "capture_rehearsal_report.v1"


@dataclass(frozen=True)
class CaptureRehearsalCommands:
    """Command arrays used by the pose-only fake capture rehearsal."""

    fake_controller: list[str]
    pose_receiver: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fake_controller": self.fake_controller,
            "fake_controller_text": shlex.join(self.fake_controller),
            "pose_receiver": self.pose_receiver,
            "pose_receiver_text": shlex.join(self.pose_receiver),
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tail(text: str, limit: int = 40) -> list[str]:
    return text.splitlines()[-limit:]


def _required_plan_command(plan: CapturePlan, role: str) -> list[str]:
    matches = [command.command for command in plan.commands if command.role == role]
    if len(matches) != 1:
        raise ValueError(
            f"Capture plan must contain exactly one {role!r} command; "
            f"found {len(matches)}"
        )
    return list(matches[0])


def build_capture_rehearsal_commands(
    config: Mapping[str, Any],
    *,
    duration_s: float = 0.3,
    sample_ms: float = 25.0,
    startup_delay_s: float = 0.0,
    robot_port: int | None = None,
    receiver_port: int | None = None,
    robot_ip: str | None = None,
    receiver_ip: str | None = None,
) -> CaptureRehearsalCommands:
    """Build rehearsal commands by selecting entries from a capture plan."""

    validate_run_config(config)
    robot = dict(config.get("robot_profile") or {})
    if str(robot.get("mode") or "fake") != "fake":
        raise ValueError("Capture rehearsal only supports fake robot mode")

    plan = build_capture_rehearsal_plan(
        config,
        duration_s=duration_s,
        sample_ms=sample_ms,
        startup_delay_s=startup_delay_s,
        robot_port=robot_port,
        receiver_port=receiver_port,
        robot_ip=robot_ip,
        receiver_ip=receiver_ip,
    )
    return CaptureRehearsalCommands(
        fake_controller=_required_plan_command(plan, "robot_controller"),
        pose_receiver=_required_plan_command(plan, "robot_pose_receiver"),
    )


def build_capture_rehearsal_plan(
    config: Mapping[str, Any],
    *,
    duration_s: float = 0.3,
    sample_ms: float = 25.0,
    startup_delay_s: float = 0.0,
    robot_port: int | None = None,
    receiver_port: int | None = None,
    robot_ip: str | None = None,
    receiver_ip: str | None = None,
) -> CapturePlan:
    """Build the capture plan variant used for a pose-only fake rehearsal."""

    robot = dict(config.get("robot_profile") or {})
    if str(robot.get("mode") or "fake") != "fake":
        raise ValueError("Capture rehearsal only supports fake robot mode")
    return build_capture_plan(
        config,
        robot_ip=robot_ip,
        receiver_ip=receiver_ip,
        robot_port=robot_port,
        receiver_port=receiver_port,
        fake_controller_duration_s=duration_s,
        fake_controller_sample_ms=sample_ms,
        fake_controller_startup_delay_s=startup_delay_s,
    )


def capture_rehearsal_report_path(run_root: str | Path) -> Path:
    return Path(run_root) / CAPTURE_REHEARSAL_REPORT


def _write_report(run_root: Path, report: Mapping[str, Any]) -> Path:
    path = capture_rehearsal_report_path(run_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(report), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def _raw_pose_count(run_root: Path) -> int:
    path = run_root / RAW_ROBOT_EE_POSES
    if not path.is_file():
        return 0
    with open(path, "r") as f:
        value = json.load(f)
    return len(value) if isinstance(value, dict) else 0


def run_capture_rehearsal(
    config: Mapping[str, Any],
    *,
    duration_s: float = 0.3,
    sample_ms: float = 25.0,
    startup_delay_s: float = 0.0,
    timeout_s: float = 10.0,
    robot_port: int | None = None,
    receiver_port: int | None = None,
    robot_ip: str | None = None,
    receiver_ip: str | None = None,
    controller_startup_wait_s: float = 0.2,
) -> tuple[Path, dict[str, Any]]:
    """Run fake iiwa plus pose receiver and write a manifest-tracked report."""

    if timeout_s <= 0:
        raise ValueError("timeout_s must be greater than 0")
    if controller_startup_wait_s < 0:
        raise ValueError("controller_startup_wait_s must be greater than or equal to 0")

    validate_run_config(config)
    run_root = Path(str(config["run_root"]))
    run_root.mkdir(parents=True, exist_ok=True)
    plan = build_capture_rehearsal_plan(
        config,
        duration_s=duration_s,
        sample_ms=sample_ms,
        startup_delay_s=startup_delay_s,
        robot_port=robot_port,
        receiver_port=receiver_port,
        robot_ip=robot_ip,
        receiver_ip=receiver_ip,
    )
    commands = CaptureRehearsalCommands(
        fake_controller=_required_plan_command(plan, "robot_controller"),
        pose_receiver=_required_plan_command(plan, "robot_pose_receiver"),
    )

    manifest = load_or_create_run_manifest(
        run_root,
        robot_profile=config.get("robot_profile"),
        capture_config=config.get("capture"),
    )
    manifest["robot_profile"] = dict(config.get("robot_profile") or {})
    manifest["capture_config"] = dict(config.get("capture") or {})
    upsert_stage(manifest, name="capture_rehearsal", status="running")
    write_run_manifest(manifest, run_root)

    controller = None
    receiver_result = None
    controller_output = ""
    status = "succeeded"
    message = "Fake capture rehearsal completed successfully."
    started_monotonic = time.monotonic()

    try:
        controller = subprocess.Popen(
            commands.fake_controller,
            cwd=_repo_root(),
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(os.name != "nt"),
        )
        time.sleep(controller_startup_wait_s)
        if controller.poll() is not None:
            controller_output = controller.communicate(timeout=1)[0] or ""
            raise RuntimeError("Fake iiwa controller exited before pose receiver start")

        receiver_result = subprocess.run(
            commands.pose_receiver,
            cwd=_repo_root(),
            env=os.environ.copy(),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        try:
            controller_output = controller.communicate(timeout=timeout_s)[0] or ""
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Fake iiwa controller did not exit after rehearsal") from exc

        if receiver_result.returncode != 0:
            raise RuntimeError(
                "Pose receiver exited with status "
                f"{receiver_result.returncode}"
            )
        if controller.returncode != 0:
            raise RuntimeError(
                "Fake iiwa controller exited with status "
                f"{controller.returncode}"
            )
    except Exception as exc:
        status = "failed"
        message = str(exc)
        if controller is not None and controller.poll() is None:
            controller.terminate()
            try:
                controller_output = controller.communicate(timeout=2)[0] or ""
            except subprocess.TimeoutExpired:
                controller.kill()
                controller_output = controller.communicate(timeout=2)[0] or ""

    elapsed_s = time.monotonic() - started_monotonic
    raw_pose_count = _raw_pose_count(run_root)
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_root": run_root.as_posix(),
        "status": status,
        "message": message,
        "mode": "pose_only_fake",
        "duration_s": duration_s,
        "sample_ms": sample_ms,
        "startup_delay_s": startup_delay_s,
        "timeout_s": timeout_s,
        "elapsed_s": elapsed_s,
        "raw_pose_artifact": RAW_ROBOT_EE_POSES,
        "raw_pose_count": raw_pose_count,
        "capture_plan": plan.to_dict(),
        "commands": commands.to_dict(),
        "processes": {
            "fake_controller_returncode": (
                controller.returncode if controller is not None else None
            ),
            "pose_receiver_returncode": (
                receiver_result.returncode if receiver_result is not None else None
            ),
        },
        "output_tail": {
            "fake_controller": _tail(controller_output),
            "pose_receiver": (
                _tail((receiver_result.stdout or "") + (receiver_result.stderr or ""))
                if receiver_result is not None
                else []
            ),
        },
    }
    report_path = _write_report(run_root, report)

    upsert_stage(
        manifest,
        name="capture_rehearsal",
        status=status,
        artifacts={
            CAPTURE_REHEARSAL_REPORT: report_path,
            RAW_ROBOT_EE_POSES: run_root / RAW_ROBOT_EE_POSES,
        },
        run_root=run_root,
        message=message,
    )
    write_run_manifest(manifest, run_root)

    if status != "succeeded":
        raise RuntimeError(message)

    return report_path, report
