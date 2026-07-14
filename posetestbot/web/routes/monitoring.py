"""Queued WebRTC room-monitor routes."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any, Mapping

from flask import Blueprint, jsonify, request

from posetestbot.jobs.runner import ResourceBusyError, TERMINAL_STATUSES
from posetestbot.monitoring.webrtc import (
    MAX_SDP_BYTES,
    MONITOR_STATUS_SCHEMA,
    UGREEN_USB_PRODUCT_ID,
    UGREEN_USB_VENDOR_ID,
    build_monitor_webrtc_command,
    load_monitor_status,
    monitor_stream_root,
    public_monitor_status,
)
from posetestbot.web.legacy import APP_ROOT, job_runner


monitoring_bp = Blueprint("monitoring", __name__)

UGREEN_DEVICE_ID = f"{UGREEN_USB_VENDOR_ID}:{UGREEN_USB_PRODUCT_ID}"
ACTIVE_JOB_STATUSES = {"queued", "running", "canceling"}


def _monitor_jobs() -> list[Any]:
    return [job for job in job_runner.list() if job.parameters.get("monitor_webrtc")]


def _active_monitor_job() -> Any | None:
    return next(
        (job for job in _monitor_jobs() if job.status in ACTIVE_JOB_STATUSES),
        None,
    )


def _default_status(job: Any) -> dict[str, Any]:
    return {
        "schema_version": MONITOR_STATUS_SCHEMA,
        "transport": "webrtc",
        "status": "starting" if job.status in ACTIVE_JOB_STATUSES else job.status,
        "signaling_ready": False,
        "peer_count": 0,
        "frame_count": 0,
        "selected_node": None,
        "error": job.message if job.status == "failed" else None,
    }


def _private_monitor_status(job: Any) -> dict[str, Any]:
    status = _default_status(job)
    monitor_root = job.parameters.get("monitor_root")
    if not monitor_root:
        status.update(status="failed", error="Monitor job is missing its status root.")
        return status
    try:
        loaded = load_monitor_status(monitor_root)
        if loaded is not None:
            status = loaded
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        status.update(status="failed", signaling_ready=False, error=str(exc))
    return status


def _monitor_payload(job: Any | None) -> dict[str, Any]:
    if job is None:
        return {"job": None, "webrtc_status": None}
    return {
        "job": job.to_dict(),
        "webrtc_status": public_monitor_status(_private_monitor_status(job)),
    }


def _proxy_webrtc_offer(port: int, payload: Mapping[str, str]) -> dict[str, Any]:
    body = json.dumps(dict(payload)).encode("utf-8")
    upstream = urllib.request.Request(
        f"http://127.0.0.1:{port}/offer",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(upstream, timeout=10) as response:
            response_body = response.read(MAX_SDP_BYTES + 4096)
    except (OSError, TimeoutError, socket.timeout, urllib.error.URLError) as exc:
        raise RuntimeError(f"Monitor signaling request failed: {exc}") from exc
    try:
        answer = json.loads(response_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Monitor signaling returned malformed JSON.") from exc
    if not isinstance(answer, dict):
        raise RuntimeError("Monitor signaling returned a non-object response.")
    return answer


@monitoring_bp.get("/monitoring/webcam")
def get_monitor_webcam():
    active = _active_monitor_job()
    if active is not None:
        return jsonify(_monitor_payload(active))
    jobs = _monitor_jobs()
    return jsonify(_monitor_payload(jobs[0] if jobs else None))


@monitoring_bp.post("/monitoring/webcam")
def start_monitor_webcam():
    existing = _active_monitor_job()
    if existing is not None:
        return jsonify(_monitor_payload(existing))

    root = monitor_stream_root()
    try:
        job = job_runner.submit(
            name="monitor-webrtc:ugreen",
            command=build_monitor_webrtc_command(monitor_root=root),
            cwd=APP_ROOT,
            resources=[f"monitoring_camera:{UGREEN_DEVICE_ID}"],
            parameters={
                "monitor_root": root.as_posix(),
                "sensor_key": f"monitor_webcam:{UGREEN_DEVICE_ID}",
                "sensor_type": "monitor_webcam",
                "device_id": UGREEN_DEVICE_ID,
                "monitor_webcam": True,
                "monitor_webrtc": True,
                "transport": "webrtc",
            },
        )
    except ResourceBusyError as exc:
        return jsonify({"output": str(exc)}), 409
    return jsonify(_monitor_payload(job)), 202


@monitoring_bp.post("/monitoring/webcam/<job_id>/webrtc/offer")
def offer_monitor_webcam(job_id: str):
    try:
        job = job_runner.get(job_id)
    except KeyError:
        return jsonify({"output": "Unknown monitor job"}), 404
    if not job.parameters.get("monitor_webrtc"):
        return jsonify({"output": "Unknown monitor job"}), 404
    if job.status in TERMINAL_STATUSES:
        return jsonify({"output": "Monitor job is terminal"}), 409
    if job.status != "running":
        return jsonify({"output": "Monitor worker is not running"}), 503

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"output": "Offer must be a JSON object"}), 400
    offer_type = payload.get("type")
    sdp = payload.get("sdp")
    if offer_type != "offer" or not isinstance(sdp, str) or not sdp.strip():
        return jsonify({"output": "Expected a non-empty SDP offer"}), 400
    if len(sdp.encode("utf-8")) > MAX_SDP_BYTES:
        return jsonify({"output": "SDP offer is too large"}), 400

    status = _private_monitor_status(job)
    port = status.get("signaling_port")
    if (
        status.get("schema_version") != MONITOR_STATUS_SCHEMA
        or status.get("signaling_ready") is not True
        or not isinstance(port, int)
        or isinstance(port, bool)
        or not 1 <= port <= 65535
    ):
        return jsonify({"output": "Monitor signaling is not ready"}), 503

    try:
        answer = _proxy_webrtc_offer(port, {"type": "offer", "sdp": sdp})
    except RuntimeError as exc:
        return jsonify({"output": str(exc)}), 503
    answer_sdp = answer.get("sdp")
    if (
        answer.get("type") != "answer"
        or not isinstance(answer_sdp, str)
        or not answer_sdp.strip()
        or len(answer_sdp.encode("utf-8")) > MAX_SDP_BYTES
    ):
        return jsonify({"output": "Monitor signaling returned an invalid answer"}), 503
    return jsonify({"type": "answer", "sdp": answer_sdp})
