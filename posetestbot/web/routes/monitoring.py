"""Queued sidebar monitoring-camera routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, send_file

from posetestbot.jobs.runner import ResourceBusyError
from posetestbot.sensors.previews import (
    build_preview_command,
    load_preview_status,
    preview_stream_root,
    resolve_preview_image,
)
from posetestbot.web.legacy import APP_ROOT, job_runner


monitoring_bp = Blueprint("monitoring", __name__)

UGREEN_USB_VENDOR_ID = "0c45"
UGREEN_USB_PRODUCT_ID = "2283"
UGREEN_DEVICE_ID = f"{UGREEN_USB_VENDOR_ID}:{UGREEN_USB_PRODUCT_ID}"
ACTIVE_JOB_STATUSES = {"queued", "running", "canceling"}


def _monitor_jobs() -> list[Any]:
    return [job for job in job_runner.list() if job.parameters.get("monitor_webcam")]


def _active_monitor_job() -> Any | None:
    return next(
        (job for job in _monitor_jobs() if job.status in ACTIVE_JOB_STATUSES),
        None,
    )


def _monitor_payload(job: Any | None) -> dict[str, Any]:
    if job is None:
        return {"job": None, "preview_status": None}
    preview_status = None
    preview_root = job.parameters.get("preview_root")
    if preview_root:
        preview_status = {
            "status": "waiting" if job.status in ACTIVE_JOB_STATUSES else job.status,
            "frame_count": 0,
            "latest_image": None,
            "selected_node": None,
            "error": job.message if job.status == "failed" else None,
        }
        try:
            loaded = load_preview_status(preview_root)
            if loaded is not None:
                preview_status = loaded
        except (OSError, ValueError) as exc:
            preview_status.update({"status": "error", "error": str(exc)})
    return {"job": job.to_dict(), "preview_status": preview_status}


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

    root = preview_stream_root()
    spec = {
        "sensor_type": "monitor_webcam",
        "device_id": UGREEN_DEVICE_ID,
        "display_name": "UGREEN Camera",
        "effective_display_name": "UGREEN Camera",
        "metadata": {
            "usb_vendor_id": UGREEN_USB_VENDOR_ID,
            "usb_product_id": UGREEN_USB_PRODUCT_ID,
        },
    }
    try:
        job = job_runner.submit(
            name="monitor-webcam:ugreen",
            command=build_preview_command(
                preview_root=root,
                spec=spec,
                fps=5,
                width=320,
                height=240,
                jpeg_quality=75,
            ),
            cwd=APP_ROOT,
            resources=[f"monitoring_camera:{UGREEN_DEVICE_ID}"],
            parameters={
                "preview_root": root.as_posix(),
                "sensor_key": f"monitor_webcam:{UGREEN_DEVICE_ID}",
                "sensor_type": "monitor_webcam",
                "device_id": UGREEN_DEVICE_ID,
                "sensor_spec": spec,
                "monitor_webcam": True,
            },
        )
    except ResourceBusyError as exc:
        return jsonify({"output": str(exc)}), 409
    return jsonify(_monitor_payload(job)), 202


@monitoring_bp.get("/monitoring/webcam/<job_id>/latest.jpg")
def get_monitor_webcam_image(job_id: str):
    try:
        job = job_runner.get(job_id)
    except KeyError:
        return jsonify({"output": "Unknown job"}), 404
    if not job.parameters.get("monitor_webcam"):
        return jsonify({"output": "Job is not a monitoring webcam"}), 400
    preview_root = job.parameters.get("preview_root")
    if not preview_root:
        return jsonify({"output": "Missing preview root"}), 400
    try:
        path = resolve_preview_image(Path(preview_root))
    except FileNotFoundError as exc:
        return jsonify({"output": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"output": str(exc)}), 400
    return send_file(path, mimetype="image/jpeg", conditional=False)
