"""Detection-first sensor web routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from flask import Blueprint, jsonify, request, send_file

from posetestbot.jobs.runner import ResourceBusyError
from posetestbot.pipeline.run_config import normalize_inverted
from posetestbot.sensors.aliases import (
    DEFAULT_SENSOR_ALIASES_PATH,
    save_sensor_aliases,
    sensor_alias_file_state,
)
from posetestbot.sensors.previews import (
    build_preview_command,
    load_preview_status,
    preview_stream_root,
    resolve_preview_image,
    stop_preview,
)
from posetestbot.sensors.snapshots import (
    build_snapshot_command,
    load_snapshot_manifest,
    resolve_snapshot_image,
    snapshot_batch_root,
    snapshot_specs_from_status,
)
from posetestbot.sensors.status import collect_sensor_status, parse_expected_counts
from posetestbot.web.legacy import APP_ROOT, job_runner


sensors_bp = Blueprint("sensors", __name__)
ACTIVE_JOB_STATUSES = {"queued", "running"}


def _json_payload() -> dict[str, Any]:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _status_from_request() -> tuple[dict[str, Any] | None, tuple[dict[str, str], int] | None]:
    try:
        expected_counts = (
            parse_expected_counts(request.args.getlist("expected"))
            if request.args.getlist("expected")
            else None
        )
    except ValueError as exc:
        return None, ({"output": str(exc)}, 400)
    return collect_sensor_status(expected_counts=expected_counts), None


@sensors_bp.get("/sensors/status")
def sensor_status():
    status, error = _status_from_request()
    if error is not None:
        payload, code = error
        return jsonify(payload), code
    return jsonify(status)


@sensors_bp.get("/sensors/aliases")
def get_sensor_aliases():
    return jsonify(sensor_alias_file_state(DEFAULT_SENSOR_ALIASES_PATH))


@sensors_bp.put("/sensors/aliases")
def put_sensor_aliases():
    data = _json_payload()
    aliases = data.get("aliases", data)
    if not isinstance(aliases, Mapping):
        return jsonify({"output": "aliases must be a JSON object"}), 400
    try:
        path = save_sensor_aliases(aliases, DEFAULT_SENSOR_ALIASES_PATH)
    except ValueError as exc:
        return jsonify({"output": str(exc)}), 400
    state = sensor_alias_file_state(path)
    return jsonify({"output": f"Wrote {path}", **state})


def _requested_sensor_specs(data: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict]:
    explicit_sensors = data.get("sensors")
    if isinstance(explicit_sensors, list) and explicit_sensors:
        specs = [dict(sensor) for sensor in explicit_sensors if isinstance(sensor, Mapping)]
        return specs, {"schema_version": "sensor_status.v1", "families": []}

    status = collect_sensor_status()
    raw_selected = data.get("selected") or data.get("sensor_keys") or []
    if isinstance(raw_selected, str):
        raw_selected = [raw_selected]
    selected = {str(value) for value in raw_selected} if raw_selected else None
    return snapshot_specs_from_status(status, selected=selected), status


def _sensor_key(spec: Mapping[str, Any]) -> str:
    return f"{spec.get('sensor_type')}:{spec.get('device_id')}"


def _active_preview_jobs_by_key() -> dict[str, Any]:
    active = {}
    for job in job_runner.list():
        if not job.parameters.get("sensor_preview"):
            continue
        if job.status not in ACTIVE_JOB_STATUSES:
            continue
        sensor_key = job.parameters.get("sensor_key")
        if sensor_key:
            active[str(sensor_key)] = job
    return active


def _preview_job_payload(job) -> dict[str, Any]:
    preview_root = job.parameters.get("preview_root")
    preview_status = None
    if preview_root:
        status_name = "waiting" if job.status in ACTIVE_JOB_STATUSES else job.status
        preview_status = {
            "status": status_name,
            "preview_root": preview_root,
            "sensor_key": job.parameters.get("sensor_key"),
            "sensor_type": job.parameters.get("sensor_type"),
            "device_id": job.parameters.get("device_id"),
            "inverted": job.parameters.get("inverted"),
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
    return {
        "job": job.to_dict(),
        "preview_root": preview_root,
        "preview_status": preview_status,
    }


def _stop_preview_job(job) -> dict[str, Any]:
    preview_root = job.parameters.get("preview_root")
    if preview_root:
        stop_preview(preview_root)
    if job.status in ACTIVE_JOB_STATUSES:
        try:
            job = job_runner.cancel(job.id)
        except KeyError:
            pass
    return _preview_job_payload(job)


@sensors_bp.post("/sensors/snapshots")
def post_sensor_snapshots():
    data = _json_payload()
    try:
        specs, status = _requested_sensor_specs(data)
    except ValueError as exc:
        return jsonify({"output": str(exc)}), 400
    if not specs:
        return jsonify({"output": "No connected sensors selected for snapshot"}), 400

    snapshot_root = snapshot_batch_root()
    try:
        command = build_snapshot_command(
            snapshot_root=snapshot_root,
            specs=specs,
            fps=int(data.get("fps", 6)),
            resolution=str(data.get("resolution", "720p")),
            max_frames=int(data.get("max_frames", 1)),
        )
        job = job_runner.submit(
            name="sensor-snapshot",
            command=command,
            cwd=APP_ROOT,
            resources=["camera"],
            parameters={
                "snapshot_root": snapshot_root.as_posix(),
                "sensor_keys": [
                    f"{spec.get('sensor_type')}:{spec.get('device_id')}" for spec in specs
                ],
                "sensor_count": len(specs),
                "sensor_snapshot": True,
            },
        )
    except ResourceBusyError as exc:
        return jsonify({"output": str(exc)}), 409
    except (TypeError, ValueError) as exc:
        return jsonify({"output": str(exc)}), 400

    return (
        jsonify(
            {
                "output": f"Queued sensor snapshot as job {job.id}",
                "job_id": job.id,
                "status": job.status,
                "job": job.to_dict(),
                "snapshot_root": snapshot_root.as_posix(),
                "sensor_status": status,
                "sensors": specs,
            }
        ),
        202,
    )


@sensors_bp.post("/sensors/previews")
def post_sensor_previews():
    data = _json_payload()
    try:
        specs, status = _requested_sensor_specs(data)
    except ValueError as exc:
        return jsonify({"output": str(exc)}), 400
    if not specs:
        return jsonify({"output": "No connected sensors selected for preview"}), 400

    active_by_key = _active_preview_jobs_by_key()
    jobs = []
    errors = []
    for spec in specs:
        key = _sensor_key(spec)
        existing = active_by_key.get(key)
        if existing is not None:
            jobs.append(_preview_job_payload(existing))
            continue

        preview_root = preview_stream_root()
        try:
            command = build_preview_command(
                preview_root=preview_root,
                spec=spec,
                fps=int(data.get("fps", 6)),
                width=int(data.get("width", 640)),
                height=int(data.get("height", 480)),
                jpeg_quality=int(data.get("jpeg_quality", 82)),
            )
            job = job_runner.submit(
                name=f"sensor-preview:{key}",
                command=command,
                cwd=APP_ROOT,
                resources=[f"camera-preview:{key}"],
                parameters={
                    "preview_root": preview_root.as_posix(),
                    "sensor_key": key,
                    "sensor_type": spec.get("sensor_type"),
                    "device_id": spec.get("device_id"),
                    "inverted": normalize_inverted(spec.get("inverted", False)),
                    "sensor_spec": dict(spec),
                    "sensor_preview": True,
                },
            )
        except ResourceBusyError as exc:
            errors.append({"sensor_key": key, "error": str(exc)})
            continue
        except (TypeError, ValueError) as exc:
            errors.append({"sensor_key": key, "error": str(exc)})
            continue
        jobs.append(_preview_job_payload(job))

    status_code = 202 if jobs else 409
    return (
        jsonify(
            {
                "output": f"Queued {len(jobs)} sensor preview(s)",
                "jobs": jobs,
                "errors": errors,
                "sensor_status": status,
                "sensors": specs,
            }
        ),
        status_code,
    )


@sensors_bp.get("/sensors/previews")
def get_sensor_previews():
    include_terminal = request.args.get("include_terminal", "").lower() in {
        "1",
        "true",
        "yes",
    }
    jobs = []
    for job in job_runner.list():
        if not job.parameters.get("sensor_preview"):
            continue
        if not include_terminal and job.status not in ACTIVE_JOB_STATUSES:
            continue
        jobs.append(_preview_job_payload(job))
    return jsonify({"jobs": jobs})


@sensors_bp.get("/sensors/previews/<job_id>")
def get_sensor_preview(job_id: str):
    try:
        job = job_runner.get(job_id)
    except KeyError:
        return jsonify({"output": "Unknown job"}), 404
    if not job.parameters.get("sensor_preview"):
        return jsonify({"output": "Job is not a sensor preview"}), 400
    return jsonify(_preview_job_payload(job))


@sensors_bp.post("/sensors/previews/stop")
def stop_sensor_previews():
    stopped = []
    for job in job_runner.list():
        if not job.parameters.get("sensor_preview"):
            continue
        if job.status not in ACTIVE_JOB_STATUSES:
            continue
        stopped.append(_stop_preview_job(job))
    return jsonify({"output": f"Stopping {len(stopped)} preview(s)", "jobs": stopped})


@sensors_bp.post("/sensors/previews/<job_id>/stop")
def stop_sensor_preview(job_id: str):
    try:
        job = job_runner.get(job_id)
    except KeyError:
        return jsonify({"output": "Unknown job"}), 404
    if not job.parameters.get("sensor_preview"):
        return jsonify({"output": "Job is not a sensor preview"}), 400
    return jsonify(_stop_preview_job(job))


@sensors_bp.get("/sensors/previews/<job_id>/latest.jpg")
def get_sensor_preview_image(job_id: str):
    try:
        job = job_runner.get(job_id)
    except KeyError:
        return jsonify({"output": "Unknown job"}), 404
    if not job.parameters.get("sensor_preview"):
        return jsonify({"output": "Job is not a sensor preview"}), 400
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


@sensors_bp.get("/sensors/snapshots/<job_id>")
def get_sensor_snapshot(job_id: str):
    try:
        job = job_runner.get(job_id)
    except KeyError:
        return jsonify({"output": "Unknown job"}), 404
    snapshot_root = job.parameters.get("snapshot_root")
    manifest = None
    if snapshot_root:
        try:
            manifest = load_snapshot_manifest(snapshot_root)
        except (OSError, ValueError) as exc:
            return jsonify({"output": str(exc), "job": job.to_dict()}), 400
    return jsonify(
        {
            "job": job.to_dict(),
            "snapshot_root": snapshot_root,
            "manifest": manifest,
        }
    )


@sensors_bp.get("/sensors/snapshots/<job_id>/image")
def get_sensor_snapshot_image(job_id: str):
    try:
        job = job_runner.get(job_id)
    except KeyError:
        return jsonify({"output": "Unknown job"}), 404
    snapshot_root = job.parameters.get("snapshot_root")
    image_path = request.args.get("path")
    if not snapshot_root or not image_path:
        return jsonify({"output": "Missing snapshot root or image path"}), 400
    try:
        path = resolve_snapshot_image(Path(snapshot_root), image_path)
    except FileNotFoundError as exc:
        return jsonify({"output": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"output": str(exc)}), 400
    return send_file(path, mimetype="image/png", conditional=True)
