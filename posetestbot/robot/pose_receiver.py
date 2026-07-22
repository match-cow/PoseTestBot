"""Safety-hardened UDP robot-pose acquisition.

The reusable capture plan deliberately omits execution acknowledgements.  They
must be supplied to this module for every invocation before it binds a socket
or sends the robot start message.
"""

from __future__ import annotations

import json
import ipaddress
import math
import os
import signal
import socket
import stat
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from posetestbot.config import RobotProfile
from posetestbot.io.atomic import atomic_write_json
from posetestbot.io.artifacts import RAW_ROBOT_EE_POSES
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    set_manifest_artifact,
    upsert_stage,
    write_run_manifest,
)
from posetestbot.robot.udp import send_start


DEFAULT_RECEIVE_START_TIMEOUT_S = 120.0
DEFAULT_RECEIVE_IDLE_TIMEOUT_S = 60.0
PARTIAL_SCHEMA_VERSION = "raw_robot_ee_poses_partial.v1"
CLAIM_SCHEMA_VERSION = "raw_robot_ee_poses_claim.v1"
MAX_PACKET_BYTES = 65_535


class PoseReceiverError(RuntimeError):
    """Base error for an incomplete robot-pose capture."""


class PoseReceiverPermissionError(PoseReceiverError):
    """Raised before I/O when fresh execution acknowledgements are absent."""


class PoseReceiverOverwriteError(PoseReceiverError):
    """Raised when the canonical raw-pose artifact already exists."""


class PoseReceiverTimeout(PoseReceiverError):
    """Raised when the first or next pose packet does not arrive in time."""


class PoseReceiverPacketError(PoseReceiverError):
    """Raised when a robot-pose datagram violates the packet contract."""


class PoseReceiverCanceled(PoseReceiverError):
    """Raised on an operator or supervisor interruption."""


@dataclass(frozen=True)
class PoseReceiverResult:
    """Successful pose-receiver result."""

    raw_pose_path: Path
    pose_count: int
    start_message: Mapping[str, Any]


@dataclass(frozen=True)
class RawPoseClaim:
    """Exclusive ownership token for the canonical raw-pose path."""

    path: Path
    claim_id: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _validate_execution_boundary(
    *,
    allow_real_robot: bool,
    allow_cameras: bool,
    receive_start_timeout_s: float,
    receive_idle_timeout_s: float,
    protocol: str,
) -> None:
    missing = []
    if allow_real_robot is not True:
        missing.append("--allow-real-robot")
    if allow_cameras is not True:
        missing.append("--allow-cameras")
    if missing:
        raise PoseReceiverPermissionError(
            "Pose receiver execution requires fresh acknowledgements: "
            + ", ".join(missing)
            + "."
        )
    for name, value in (
        ("receive_start_timeout_s", receive_start_timeout_s),
        ("receive_idle_timeout_s", receive_idle_timeout_s),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite value greater than 0")
    if protocol not in {"legacy", "v1"}:
        raise ValueError("protocol must be 'legacy' or 'v1'")


def _stage_artifact_paths(
    manifest: Mapping[str, Any], run_root: Path
) -> dict[str, Path]:
    for stage in manifest.get("stages", []):
        if not isinstance(stage, Mapping) or stage.get("name") != "robot_pose_capture":
            continue
        artifacts = stage.get("artifacts")
        if not isinstance(artifacts, Mapping):
            return {}
        paths: dict[str, Path] = {}
        for name, value in artifacts.items():
            if not isinstance(name, str) or not isinstance(value, str):
                continue
            path = Path(value)
            paths[name] = path if path.is_absolute() else run_root / path
        return paths
    return {}


def _partial_path(run_root: Path) -> Path:
    return run_root / (
        f"raw_robot_ee_poses.partial.{time.time_ns()}.{uuid.uuid4().hex}.json"
    )


def _write_partial_evidence(
    manifest: dict[str, Any],
    run_root: Path,
    *,
    status: str,
    message: str,
    poses: Mapping[int, Mapping[str, Any]],
    started_at: str,
    last_packet_preview: str | None,
    last_sender: tuple[Any, ...] | None,
) -> Path:
    path = _partial_path(run_root)
    evidence: dict[str, Any] = {
        "schema_version": PARTIAL_SCHEMA_VERSION,
        "status": status,
        "started_at": started_at,
        "ended_at": _now(),
        "message": message,
        "received_pose_count": len(poses),
        "poses": dict(poses),
    }
    if last_packet_preview is not None:
        evidence["last_packet_preview"] = last_packet_preview
    if last_sender is not None:
        evidence["last_sender"] = [str(value) for value in last_sender]
    atomic_write_json(path, evidence, indent=2, sort_keys=False)

    set_manifest_artifact(manifest, path.name, path, run_root=run_root)
    artifacts = _stage_artifact_paths(manifest, run_root)
    artifacts[path.name] = path
    upsert_stage(
        manifest,
        name="robot_pose_capture",
        status=status,
        artifacts=artifacts,
        run_root=run_root,
        message=message,
    )
    write_run_manifest(manifest, run_root)
    return path


def _claim_raw_pose_artifact(path: Path) -> RawPoseClaim:
    """Atomically reserve the canonical path before any network operation."""

    claim = RawPoseClaim(path=path, claim_id=uuid.uuid4().hex)
    payload = {
        "schema_version": CLAIM_SCHEMA_VERSION,
        "status": "reserved",
        "claim_id": claim.claim_id,
        "created_at": _now(),
    }
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise PoseReceiverOverwriteError(
            f"Refusing to replace existing raw pose artifact: {path}"
        ) from exc
    claimed_inode = os.fstat(descriptor)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            current_inode = path.lstat()
        except FileNotFoundError:
            pass
        else:
            if (
                current_inode.st_dev == claimed_inode.st_dev
                and current_inode.st_ino == claimed_inode.st_ino
            ):
                path.unlink(missing_ok=True)
        raise
    return claim


def _owns_raw_pose_claim(claim: RawPoseClaim) -> bool:
    try:
        metadata = claim.path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            return False
        with open(claim.path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(value, dict)
        and value.get("schema_version") == CLAIM_SCHEMA_VERSION
        and value.get("claim_id") == claim.claim_id
        and value.get("status") == "reserved"
    )


def _promote_raw_pose_claim(
    claim: RawPoseClaim,
    poses: Mapping[int, Mapping[str, Any]],
) -> Path:
    """Replace only this receiver's verified claim with complete pose data."""

    if not _owns_raw_pose_claim(claim):
        raise PoseReceiverOverwriteError(
            "Raw pose reservation ownership changed before promotion; refusing "
            f"to replace {claim.path}."
        )
    pending = claim.path.with_name(
        f".{claim.path.name}.{claim.claim_id}.{uuid.uuid4().hex}.pending"
    )
    atomic_write_json(pending, dict(poses), indent=4, sort_keys=False)
    try:
        if not _owns_raw_pose_claim(claim):
            raise PoseReceiverOverwriteError(
                "Raw pose reservation ownership changed during promotion; "
                f"refusing to replace {claim.path}."
            )
        os.replace(pending, claim.path)
    finally:
        pending.unlink(missing_ok=True)
    return claim.path


def _cleanup_raw_pose_claim(claim: RawPoseClaim) -> None:
    """Remove a failed receiver's reservation, but never a foreign artifact."""

    if _owns_raw_pose_claim(claim):
        claim.path.unlink(missing_ok=True)


def _decode_packet(data: bytes) -> tuple[str, dict[str, int | float] | None]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PoseReceiverPacketError(
            f"Malformed robot pose packet: invalid JSON ({exc})."
        ) from exc
    if not isinstance(value, dict):
        raise PoseReceiverPacketError(
            "Malformed robot pose packet: expected a JSON object."
        )

    motion = value.get("motion")
    if not isinstance(motion, str) or not motion.strip():
        raise PoseReceiverPacketError(
            "Malformed robot pose packet: motion must be a non-empty string."
        )
    if motion == "end":
        return motion, None

    pose: dict[str, int | float] = {}
    for axis in ("X", "Y", "Z", "A", "B", "C"):
        coordinate = value.get(axis)
        if (
            isinstance(coordinate, bool)
            or not isinstance(coordinate, (int, float))
            or not math.isfinite(float(coordinate))
        ):
            raise PoseReceiverPacketError(
                f"Malformed robot pose packet: {axis} must be a finite number."
            )
        pose[axis] = coordinate
    return motion, pose


def _validate_sender(
    sender: Any,
    *,
    expected_robot_ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> tuple[Any, ...]:
    if not isinstance(sender, tuple) or len(sender) < 2:
        raise PoseReceiverPacketError(
            "Malformed robot pose sender address: expected an IP/port tuple."
        )
    try:
        sender_ip = ipaddress.ip_address(str(sender[0]))
    except ValueError as exc:
        raise PoseReceiverPacketError(
            f"Malformed robot pose sender IP: {sender[0]!r}."
        ) from exc
    if sender_ip != expected_robot_ip:
        raise PoseReceiverPacketError(
            "Rejected robot pose packet from unexpected sender IP "
            f"{sender_ip}; expected {expected_robot_ip}."
        )
    return tuple(sender)


@contextmanager
def _cancellation_signal_handlers(enabled: bool) -> Iterator[None]:
    previous_handlers: dict[int, Any] = {}

    def cancel(signum: int, _frame: Any) -> None:
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        raise PoseReceiverCanceled(f"Robot pose capture canceled by {name}.")

    if enabled:
        for receiver_signal in (signal.SIGINT, signal.SIGTERM):
            try:
                previous_handlers[receiver_signal] = signal.getsignal(receiver_signal)
                signal.signal(receiver_signal, cancel)
            except (OSError, ValueError):
                previous_handlers.pop(receiver_signal, None)
    try:
        yield
    finally:
        for receiver_signal, handler in previous_handlers.items():
            signal.signal(receiver_signal, handler)


def run_pose_receiver(
    output_path: str | Path,
    *,
    profile: RobotProfile,
    protocol: str = "legacy",
    verbose: bool = False,
    allow_real_robot: bool = False,
    allow_cameras: bool = False,
    receive_start_timeout_s: float = DEFAULT_RECEIVE_START_TIMEOUT_S,
    receive_idle_timeout_s: float = DEFAULT_RECEIVE_IDLE_TIMEOUT_S,
    socket_factory: Callable[..., Any] = socket.socket,
    send_start_command: Callable[..., Mapping[str, Any]] = send_start,
    install_signal_handlers: bool = True,
) -> PoseReceiverResult:
    """Receive one pose stream after validating fresh execution permissions."""

    _validate_execution_boundary(
        allow_real_robot=allow_real_robot,
        allow_cameras=allow_cameras,
        receive_start_timeout_s=receive_start_timeout_s,
        receive_idle_timeout_s=receive_idle_timeout_s,
        protocol=protocol,
    )

    run_root = Path(output_path)
    run_root.mkdir(parents=True, exist_ok=True)
    if not run_root.is_dir():
        raise ValueError(f"Output path is not a directory: {run_root}")
    raw_pose_path = run_root / RAW_ROBOT_EE_POSES
    try:
        expected_robot_ip = ipaddress.ip_address(profile.robot_ip)
    except ValueError as exc:
        raise ValueError(
            f"Robot profile robot_ip must be an IP address: {profile.robot_ip!r}"
        ) from exc
    claim = _claim_raw_pose_artifact(raw_pose_path)

    started_at = _now()
    manifest: dict[str, Any] | None = None
    poses: dict[int, dict[str, Any]] = {}
    previous_frame_ts = 0
    last_packet_preview: str | None = None
    last_sender: tuple[Any, ...] | None = None
    start_message: Mapping[str, Any] = {}

    try:
        manifest = load_or_create_run_manifest(
            run_root,
            robot_profile=profile,
            capture_config={
                "cartesian_velocity_m_s": profile.cartesian_velocity_m_s,
                "protocol": protocol,
                "mode": "real",
            },
        )
        upsert_stage(manifest, name="robot_pose_capture", status="running")
        write_run_manifest(manifest, run_root)
        with _cancellation_signal_handlers(install_signal_handlers):
            with socket_factory(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.bind((profile.receiver_ip, profile.receiver_port))
                sock.settimeout(receive_start_timeout_s)
                print(f"Listening on {profile.receiver_ip}:{profile.receiver_port}")

                start_message = send_start_command(profile, protocol=protocol)
                print(
                    "Sent start message to "
                    f"{profile.robot_ip}:{profile.command_port} "
                    f"with capture vel {profile.cartesian_velocity_m_s}"
                )
                print(f"Message: {start_message}")

                received_any_packet = False
                while True:
                    try:
                        data, sender = sock.recvfrom(MAX_PACKET_BYTES)
                    except socket.timeout as exc:
                        if received_any_packet:
                            message = (
                                "Timed out waiting for the next robot pose packet "
                                f"after {receive_idle_timeout_s:g} seconds."
                            )
                        else:
                            message = (
                                "Timed out waiting for the first robot pose packet "
                                f"after {receive_start_timeout_s:g} seconds."
                            )
                        raise PoseReceiverTimeout(message) from exc

                    host_received_timestamp_ns = time.monotonic_ns()
                    host_wall_timestamp_ns = time.time_ns()
                    received_any_packet = True
                    last_sender = tuple(sender) if isinstance(sender, tuple) else None
                    last_packet_preview = data[:4096].decode("utf-8", errors="replace")
                    last_sender = _validate_sender(
                        sender,
                        expected_robot_ip=expected_robot_ip,
                    )
                    motion, pose = _decode_packet(data)
                    if motion == "end":
                        if not poses:
                            raise PoseReceiverPacketError(
                                "Robot pose stream ended before any pose packet was "
                                "captured."
                            )
                        break

                    if not poses:
                        sock.settimeout(receive_idle_timeout_s)
                    framename = int(round(host_wall_timestamp_ns / 1_000_000))
                    frame_delta = (
                        0 if not poses else framename - int(previous_frame_ts)
                    )
                    previous_frame_ts = framename
                    poses[len(poses)] = {
                        "framename": framename,
                        "host_received_timestamp_ns": host_received_timestamp_ns,
                        "host_wall_timestamp_ns": host_wall_timestamp_ns,
                        "frame_delta": frame_delta,
                        "motion": motion,
                        "pose": pose,
                    }

                    if verbose:
                        print(
                            f"framename: {framename}, addr: {sender}, "
                            f"motion: {motion}, pose: {pose}"
                        )
                    print(f"Received poses: {len(poses)}", end="\r", flush=True)

        _promote_raw_pose_claim(claim, poses)
        set_manifest_artifact(
            manifest,
            RAW_ROBOT_EE_POSES,
            raw_pose_path,
            run_root=run_root,
        )
        artifacts = _stage_artifact_paths(manifest, run_root)
        artifacts[RAW_ROBOT_EE_POSES] = raw_pose_path
        upsert_stage(
            manifest,
            name="robot_pose_capture",
            status="succeeded",
            artifacts=artifacts,
            run_root=run_root,
            message=f"Captured {len(poses)} robot poses.",
        )
        write_run_manifest(manifest, run_root)
    except (PoseReceiverCanceled, KeyboardInterrupt, InterruptedError) as exc:
        canceled = (
            exc
            if isinstance(exc, PoseReceiverCanceled)
            else PoseReceiverCanceled("Robot pose capture was interrupted.")
        )
        try:
            if manifest is not None:
                _write_partial_evidence(
                    manifest,
                    run_root,
                    status="canceled",
                    message=str(canceled),
                    poses=poses,
                    started_at=started_at,
                    last_packet_preview=last_packet_preview,
                    last_sender=last_sender,
                )
        finally:
            _cleanup_raw_pose_claim(claim)
        if canceled is exc:
            raise
        raise canceled from exc
    except Exception as exc:
        try:
            if manifest is not None:
                _write_partial_evidence(
                    manifest,
                    run_root,
                    status="failed",
                    message=str(exc),
                    poses=poses,
                    started_at=started_at,
                    last_packet_preview=last_packet_preview,
                    last_sender=last_sender,
                )
        finally:
            _cleanup_raw_pose_claim(claim)
        raise

    if poses:
        print()
    return PoseReceiverResult(
        raw_pose_path=raw_pose_path,
        pose_count=len(poses),
        start_message=start_message,
    )
