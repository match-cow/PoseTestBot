"""Detection-first sensor web routes."""

from __future__ import annotations

from pathlib import Path
import threading
import time
from datetime import UTC, datetime
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
    preview_status_health,
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
ACTIVE_JOB_STATUSES = {"queued", "running", "canceling"}
_preview_replacement_lock = threading.Lock()
_preview_replacement_job_ids: set[str] = set()


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


def _camera_resource(spec: Mapping[str, Any]) -> str:
    return f"camera:{_sensor_key(spec)}"


def _preview_job_health(job: Any) -> tuple[bool, str | None]:
    if job.status == "queued":
        return True, None
    preview_root = job.parameters.get("preview_root")
    if not preview_root:
        return False, "Preview job is missing its artifact root."
    try:
        status = load_preview_status(preview_root)
    except (OSError, ValueError) as exc:
        return False, str(exc)
    if status is None:
        started_at = getattr(job, "started_at", None)
        if started_at is None:
            # Compatibility for lightweight queued/running test doubles.
            return True, None
        try:
            started = datetime.fromisoformat(
                str(started_at).replace("Z", "+00:00")
            )
            if started.tzinfo is None:
                started = started.replace(tzinfo=UTC)
            age = (datetime.now(UTC) - started).total_seconds()
        except ValueError:
            age = 6.0
        if age <= 5.0:
            return True, None
    return preview_status_health(preview_root, status)


def _cancel_preview_in_background(job: Any) -> None:
    def cancel() -> None:
        preview_root = job.parameters.get("preview_root")
        if preview_root:
            stop_preview(preview_root)
        cancel_job = getattr(job_runner, "cancel", None)
        if callable(cancel_job):
            try:
                cancel_job(job.id)
            except KeyError:
                pass

    threading.Thread(
        target=cancel,
        name=f"stop-stale-preview-{job.id}",
        daemon=True,
    ).start()


def _active_preview_jobs_by_key() -> tuple[dict[str, Any], dict[str, Any]]:
    active = {}
    stale = {}
    for job in job_runner.list():
        if not job.parameters.get("sensor_preview"):
            continue
        if job.status not in ACTIVE_JOB_STATUSES:
            continue
        sensor_key = job.parameters.get("sensor_key")
        healthy, _reason = _preview_job_health(job)
        if sensor_key and healthy:
            active[str(sensor_key)] = job
        elif sensor_key:
            stale[str(sensor_key)] = job
    return active, stale


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


def _preview_submission(
    spec: Mapping[str, Any],
    *,
    fps: int,
    width: int,
    height: int,
    jpeg_quality: int,
) -> Any:
    key = _sensor_key(spec)
    preview_root = preview_stream_root()
    command = build_preview_command(
        preview_root=preview_root,
        spec=spec,
        fps=fps,
        width=width,
        height=height,
        jpeg_quality=jpeg_quality,
    )
    return job_runner.submit(
        name=f"sensor-preview:{key}",
        command=command,
        cwd=APP_ROOT,
        resources=[_camera_resource(spec)],
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


def _schedule_preview_replacement(
    stale_job: Any,
    spec: Mapping[str, Any],
    *,
    fps: int,
    width: int,
    height: int,
    jpeg_quality: int,
) -> None:
    with _preview_replacement_lock:
        if stale_job.id in _preview_replacement_job_ids:
            return
        _preview_replacement_job_ids.add(stale_job.id)

    def replace() -> None:
        try:
            preview_root = stale_job.parameters.get("preview_root")
            if preview_root:
                stop_preview(preview_root)
            try:
                job_runner.cancel(stale_job.id)
            except KeyError:
                pass
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                try:
                    current = job_runner.get(stale_job.id)
                except KeyError:
                    break
                if current.status not in ACTIVE_JOB_STATUSES:
                    break
                time.sleep(0.05)
            active, _stale = _active_preview_jobs_by_key()
            if _sensor_key(spec) not in active:
                try:
                    _preview_submission(
                        spec,
                        fps=fps,
                        width=width,
                        height=height,
                        jpeg_quality=jpeg_quality,
                    )
                except (ResourceBusyError, TypeError, ValueError):
                    pass
        finally:
            with _preview_replacement_lock:
                _preview_replacement_job_ids.discard(stale_job.id)

    threading.Thread(
        target=replace,
        name=f"replace-preview-{stale_job.id}",
        daemon=True,
    ).start()


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
            resources=[*sorted({_camera_resource(spec) for spec in specs}), "disk_io"],
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

    active_by_key, stale_by_key = _active_preview_jobs_by_key()
    jobs = []
    errors = []
    for spec in specs:
        key = _sensor_key(spec)
        existing = active_by_key.get(key)
        if existing is not None:
            jobs.append(_preview_job_payload(existing))
            continue

        stale = stale_by_key.get(key)
        if stale is not None:
            _schedule_preview_replacement(
                stale,
                spec,
                fps=int(data.get("fps", 6)),
                width=int(data.get("width", 640)),
                height=int(data.get("height", 480)),
                jpeg_quality=int(data.get("jpeg_quality", 82)),
            )
            jobs.append(_preview_job_payload(stale))
            continue

        try:
            job = _preview_submission(
                spec,
                fps=int(data.get("fps", 6)),
                width=int(data.get("width", 640)),
                height=int(data.get("height", 480)),
                jpeg_quality=int(data.get("jpeg_quality", 82)),
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
        if job.status in ACTIVE_JOB_STATUSES:
            healthy, _reason = _preview_job_health(job)
            if not healthy:
                _cancel_preview_in_background(job)
                # A persisted active job can still point at a JPEG from an old
                # worker. Never expose that artifact as a current preview,
                # including on the terminal-history view used by the UI.
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
    response = send_file(path, mimetype="image/jpeg", conditional=False)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


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
