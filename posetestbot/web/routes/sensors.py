"""Detection-first sensor web routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from flask import Blueprint, jsonify, request, send_file

from posetestbot.jobs.runner import ResourceBusyError
from posetestbot.sensors.aliases import (
    DEFAULT_SENSOR_ALIASES_PATH,
    save_sensor_aliases,
    sensor_alias_file_state,
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


def _requested_snapshot_specs(data: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict]:
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


@sensors_bp.post("/sensors/snapshots")
def post_sensor_snapshots():
    data = _json_payload()
    try:
        specs, status = _requested_snapshot_specs(data)
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
