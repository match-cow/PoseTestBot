"""Safety-gated, monitor-only UGREEN WebRTC hardware smoke test."""

from __future__ import annotations

import asyncio
import os
import re
import signal
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aiohttp import ClientSession
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

from posetestbot.io.atomic import atomic_write_json
from posetestbot.io.manifest import utc_now_iso
from posetestbot.monitoring.webrtc import (
    MONITOR_STATUS_SCHEMA,
    build_monitor_webrtc_command,
    load_monitor_status,
    prefer_vp8,
)


MONITOR_SMOKE_SCHEMA = "monitor_webrtc_smoke.v1"
MONITOR_SMOKE_REPORT_NAME = "monitor_webrtc_smoke_report.json"
DEFAULT_EXPECTED_NODE = "/dev/video0"
DEFAULT_FRAME_TARGET = 35
DEFAULT_TIMEOUT_S = 20.0


def validate_execution_gates(
    *,
    operator_authorized: bool,
    allow_cameras: bool,
    allow_real_robot: bool,
) -> None:
    """Require the repository's explicit physical-execution acknowledgements."""

    missing = [
        label
        for enabled, label in (
            (operator_authorized, "--operator-authorized"),
            (allow_cameras, "--allow-cameras"),
            (allow_real_robot, "--allow-real-robot"),
        )
        if not enabled
    ]
    if missing:
        raise ValueError(
            "Physical monitor smoke execution requires: " + ", ".join(missing)
        )


def build_smoke_plan(
    smoke_root: str | Path,
    *,
    expected_node: str = DEFAULT_EXPECTED_NODE,
    frame_target: int = DEFAULT_FRAME_TARGET,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    root = Path(smoke_root).resolve()
    worker_root = root / "worker"
    return {
        "schema_version": MONITOR_SMOKE_SCHEMA,
        "mode": "monitor_only",
        "smoke_root": root.as_posix(),
        "worker_root": worker_root.as_posix(),
        "expected_node": expected_node,
        "expected_capture": {
            "pixel_format": "MJPG",
            "width": 640,
            "height": 480,
            "fps": 30.0,
        },
        "frame_target": max(1, frame_target),
        "timeout_s": max(1.0, timeout_s),
        "worker_command": build_monitor_webrtc_command(monitor_root=worker_root),
        "execution_gates": [
            "--operator-authorized",
            "--allow-cameras",
            "--allow-real-robot",
        ],
        "robot_commands": [],
        "acquisition_pipeline_commands": [],
    }


def parse_v4l2_negotiation(output: str) -> dict[str, Any]:
    size = re.search(r"Width/Height\s*:\s*(\d+)\s*/\s*(\d+)", output)
    pixel_format = re.search(r"Pixel Format\s*:\s*'([^']+)'", output)
    fps = re.search(r"Frames per second\s*:\s*([0-9.]+)", output)
    if size is None or pixel_format is None or fps is None:
        raise ValueError("Could not parse active V4L2 format and frame rate")
    return {
        "width": int(size.group(1)),
        "height": int(size.group(2)),
        "pixel_format": pixel_format.group(1),
        "fps": float(fps.group(1)),
    }


def device_holders(device_path: str | Path) -> list[int]:
    """Return process IDs whose file descriptors currently reference a device."""

    target = Path(device_path).resolve()
    holders: set[int] = set()
    proc_root = Path("/proc")
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit():
            continue
        fd_dir = process_dir / "fd"
        try:
            descriptors = list(fd_dir.iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        for descriptor in descriptors:
            try:
                if descriptor.resolve() == target:
                    holders.add(int(process_dir.name))
                    break
            except (FileNotFoundError, PermissionError, OSError):
                continue
    return sorted(holders)


async def _wait_for_worker_ready(
    worker_root: Path,
    process: subprocess.Popen[Any],
    *,
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = load_monitor_status(worker_root)
        if status is not None:
            if status.get("status") == "failed":
                raise RuntimeError(str(status.get("error") or "Monitor worker failed"))
            if status.get("signaling_ready") and status.get("signaling_port"):
                return status
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"Monitor worker exited early with code {return_code}")
        await asyncio.sleep(0.1)
    raise TimeoutError("Timed out waiting for monitor WebRTC signaling")


async def _wait_for_peer_state(
    peer: RTCPeerConnection,
    desired: set[str],
    *,
    timeout_s: float,
) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if peer.connectionState in desired:
            return peer.connectionState
        if peer.connectionState in {"failed", "closed"}:
            raise RuntimeError(
                f"WebRTC connection entered {peer.connectionState!r} state"
            )
        await asyncio.sleep(0.05)
    raise TimeoutError(
        "Timed out waiting for WebRTC state: " + ", ".join(sorted(desired))
    )


async def _receive_frames(
    signaling_port: int,
    *,
    frame_target: int,
    timeout_s: float,
) -> dict[str, Any]:
    peer = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
    track_future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

    @peer.on("track")
    def on_track(track: Any) -> None:
        if track.kind == "video" and not track_future.done():
            track_future.set_result(track)

    try:
        peer.addTransceiver("video", direction="recvonly")
        prefer_vp8(peer)
        offer = await peer.createOffer()
        await peer.setLocalDescription(offer)
        if peer.localDescription is None:
            raise RuntimeError("WebRTC client did not produce a local offer")

        async with ClientSession() as session:
            async with session.post(
                f"http://127.0.0.1:{signaling_port}/offer",
                json={
                    "type": peer.localDescription.type,
                    "sdp": peer.localDescription.sdp,
                },
                timeout=timeout_s,
            ) as response:
                payload = await response.json()
                if response.status != 200:
                    raise RuntimeError(
                        f"Worker signaling returned HTTP {response.status}: {payload}"
                    )
        if not isinstance(payload, Mapping):
            raise RuntimeError("Worker signaling response was not a JSON object")
        answer_type = payload.get("type")
        answer_sdp = payload.get("sdp")
        if answer_type != "answer" or not isinstance(answer_sdp, str):
            raise RuntimeError("Worker signaling response was not an SDP answer")
        await peer.setRemoteDescription(
            RTCSessionDescription(type=answer_type, sdp=answer_sdp)
        )

        track = await asyncio.wait_for(track_future, timeout=timeout_s)
        connected_state = await _wait_for_peer_state(
            peer,
            {"connected"},
            timeout_s=timeout_s,
        )
        first_pts = None
        last_pts = None
        width = None
        height = None
        for _index in range(frame_target):
            frame = await asyncio.wait_for(track.recv(), timeout=timeout_s)
            if first_pts is None:
                first_pts = frame.pts
            last_pts = frame.pts
            width = frame.width
            height = frame.height
        return {
            "connection_state": connected_state,
            "received_frames": frame_target,
            "first_pts": first_pts,
            "last_pts": last_pts,
            "width": width,
            "height": height,
        }
    finally:
        await peer.close()


def _query_v4l2(device_path: str, *, timeout_s: float) -> tuple[str, dict[str, Any]]:
    result = subprocess.run(
        [
            "v4l2-ctl",
            "--device",
            device_path,
            "--get-fmt-video",
            "--get-parm",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    output = result.stdout.strip()
    return output, parse_v4l2_negotiation(output)


async def _stop_worker(
    process: subprocess.Popen[Any],
    *,
    timeout_s: float = 5.0,
) -> int:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        return await asyncio.to_thread(process.wait, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        return await asyncio.to_thread(process.wait, timeout=timeout_s)


def _assert_expected_capture(negotiated: Mapping[str, Any]) -> None:
    if negotiated.get("pixel_format") != "MJPG":
        raise RuntimeError(f"Expected MJPG capture, got {negotiated!r}")
    if (negotiated.get("width"), negotiated.get("height")) != (640, 480):
        raise RuntimeError(f"Expected 640x480 capture, got {negotiated!r}")
    if abs(float(negotiated.get("fps", 0.0)) - 30.0) > 0.1:
        raise RuntimeError(f"Expected 30 fps capture, got {negotiated!r}")


async def run_monitor_webrtc_smoke(
    smoke_root: str | Path,
    *,
    operator_authorized: bool,
    allow_cameras: bool,
    allow_real_robot: bool,
    expected_node: str = DEFAULT_EXPECTED_NODE,
    frame_target: int = DEFAULT_FRAME_TARGET,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    repo_root: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Run the monitor worker and a local aiortc receiver, then release the camera."""

    validate_execution_gates(
        operator_authorized=operator_authorized,
        allow_cameras=allow_cameras,
        allow_real_robot=allow_real_robot,
    )
    plan = build_smoke_plan(
        smoke_root,
        expected_node=expected_node,
        frame_target=frame_target,
        timeout_s=timeout_s,
    )
    root = Path(smoke_root)
    worker_root = Path(plan["worker_root"])
    report_path = root / MONITOR_SMOKE_REPORT_NAME
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "monitor_webrtc_worker.log"
    preexisting_holders = device_holders(expected_node)
    report: dict[str, Any] = {
        **plan,
        "generated_at": utc_now_iso(),
        "status": "running",
        "passed": False,
        "preexisting_device_holders": preexisting_holders,
        "robot_commands_executed": False,
        "acquisition_pipeline_executed": False,
        "error": None,
    }
    atomic_write_json(report_path, report)
    process: subprocess.Popen[Any] | None = None
    worker_log = open(log_path, "w", encoding="utf-8")
    caught: BaseException | None = None
    try:
        process = subprocess.Popen(
            plan["worker_command"],
            cwd=Path(repo_root),
            stdout=worker_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        ready_status = await _wait_for_worker_ready(
            worker_root,
            process,
            timeout_s=float(plan["timeout_s"]),
        )
        selected_node = ready_status.get("selected_node")
        if not isinstance(selected_node, Mapping):
            raise RuntimeError("Worker did not report a selected V4L2 node")
        selected_path = selected_node.get("path")
        if selected_path != expected_node:
            raise RuntimeError(
                f"Expected worker to select {expected_node}, got {selected_path}"
            )
        v4l2_output, negotiated = await asyncio.to_thread(
            _query_v4l2,
            expected_node,
            timeout_s=float(plan["timeout_s"]),
        )
        _assert_expected_capture(negotiated)
        received = await _receive_frames(
            int(ready_status["signaling_port"]),
            frame_target=int(plan["frame_target"]),
            timeout_s=float(plan["timeout_s"]),
        )
        if (received["width"], received["height"]) != (640, 480):
            raise RuntimeError(f"Expected decoded 640x480 frames, got {received!r}")

        deadline = time.monotonic() + float(plan["timeout_s"])
        advancing_status = load_monitor_status(worker_root)
        while (
            advancing_status is None
            or int(advancing_status.get("frame_count", 0)) < 1
        ):
            if time.monotonic() >= deadline:
                raise TimeoutError("Worker frame_count did not advance to the target")
            await asyncio.sleep(0.1)
            advancing_status = load_monitor_status(worker_root)
        report.update(
            {
                "selected_node": dict(selected_node),
                "v4l2_output": v4l2_output,
                "negotiated_capture": negotiated,
                "receiver": received,
                "advancing_frame_count": advancing_status["frame_count"],
            }
        )
    except BaseException as exc:
        caught = exc
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        worker_exit_code = None
        if process is not None:
            try:
                worker_exit_code = await _stop_worker(process)
            except BaseException as stop_exc:
                if caught is None:
                    caught = stop_exc
                    report["error"] = f"{type(stop_exc).__name__}: {stop_exc}"
        worker_log.close()
        await asyncio.sleep(0.2)
        final_status = load_monitor_status(worker_root)
        final_holders = device_holders(expected_node)
        new_holders = sorted(set(final_holders) - set(preexisting_holders))
        clean_exit_codes = {0, -signal.SIGTERM, 128 + signal.SIGTERM}
        cleanup_ok = (
            worker_exit_code in clean_exit_codes
            and final_status is not None
            and final_status.get("schema_version") == MONITOR_STATUS_SCHEMA
            and final_status.get("status") == "stopped"
            and final_status.get("signaling_ready") is False
            and final_status.get("peer_count") == 0
            and not new_holders
        )
        if caught is None and not cleanup_ok:
            caught = RuntimeError("Monitor worker did not shut down cleanly")
            report["error"] = f"{type(caught).__name__}: {caught}"
        report.update(
            {
                "generated_at": utc_now_iso(),
                "status": "passed" if caught is None else "failed",
                "passed": caught is None,
                "worker_exit_code": worker_exit_code,
                "final_worker_status": final_status,
                "final_device_holders": final_holders,
                "new_device_holders": new_holders,
                "cleanup_ok": cleanup_ok,
            }
        )
        atomic_write_json(report_path, report)

    if caught is not None:
        raise RuntimeError(str(report["error"])) from caught
    return report_path, report
