import ipaddress
import json
import os
from pathlib import Path

from flask import Blueprint, Response, jsonify, request, send_file

from posetestbot.calibration.preflight import (
    build_calibration_preflight,
    write_calibration_preflight_with_manifest,
)
from posetestbot.calibration.observations import (
    build_calibration_observations,
    write_calibration_observations_with_manifest,
)
from posetestbot.calibration.targets import normalize_calibration_target_spec
from posetestbot.calibration.candidates import (
    DEFAULT_MAX_ROTATION_RESIDUAL_DEG,
    DEFAULT_MAX_TRANSLATION_RESIDUAL_MM,
    build_calibration_candidates,
    write_calibration_candidates_with_manifest,
)
from posetestbot.calibration.solver import (
    DEFAULT_HAND_EYE_METHOD,
    HAND_EYE_METHODS,
    build_calibration_solver,
    write_calibration_solver_with_manifest,
)
from posetestbot.calibration.validation import (
    DEFAULT_MAX_MEAN_ROTATION_RESIDUAL_DEG,
    DEFAULT_MAX_MEAN_TRANSLATION_RESIDUAL_MM,
    DEFAULT_MAX_OUTLIER_RATIO,
    DEFAULT_MIN_INLIERS,
    build_calibration_validation,
    write_calibration_validation_with_manifest,
)
from posetestbot.config import DEFAULT_ROBOT_PORT, LAB_ROBOT_IP
from posetestbot.io.artifact_browser import (
    ArtifactPathError,
    bop_frame_detail,
    bop_scene_detail,
    collect_run_artifacts,
    preview_artifact,
    render_bop_frame_overlay_png,
    resolve_artifact_path,
)
from posetestbot.io.artifacts import (
    CAPTURE_EXECUTION_PLAN,
    CAPTURE_EXECUTION_STATUS,
    CAPTURE_PLAN,
    RUN_CONFIG,
    RUN_PREFLIGHT_REPORT,
)
from posetestbot.jobs.runner import LocalJobRunner, ResourceBusyError
from posetestbot.pipeline.capture_plan import (
    load_capture_plan,
    write_capture_plan_with_manifest,
)
from posetestbot.pipeline.capture_plan_preflight import (
    build_capture_plan_preflight,
    write_capture_plan_preflight_with_manifest,
)
from posetestbot.pipeline.hardware_status import (
    load_hardware_status_report,
    write_hardware_status_report_with_manifest,
)
from posetestbot.pipeline.capture_execution import (
    load_capture_execution_plan,
    load_capture_execution_status,
    write_capture_execution_plan_with_manifest,
)
from posetestbot.pipeline.run_config import (
    build_sequence_job_from_run_config,
    create_run_config,
    load_run_config_for_run_root,
    sequence_plan_from_run_config,
    sensor_configs_from_status,
    sensor_configs_from_values,
    write_run_config_with_manifest,
)
from posetestbot.pipeline.preflight import (
    build_run_preflight,
    load_run_preflight_report,
    run_preflight_queue_summary,
    write_run_preflight_with_manifest,
)
from posetestbot.pipeline.recommendations import build_pipeline_recommendations
from posetestbot.pipeline.stages import (
    PIPELINE_STAGES,
    build_pipeline_job,
    get_pipeline_stage,
    list_pipeline_stages,
)
from posetestbot.pipeline.sequences import (
    PIPELINE_SEQUENCES,
    build_sequence_job,
    get_pipeline_sequence,
    list_pipeline_sequences,
)
from posetestbot.runtime.status import collect_runtime_status
from posetestbot.sensors.registry import list_sensor_adapters
from posetestbot.sensors.status import collect_sensor_status
from posetestbot.robot.status import collect_robot_status
from posetestbot.sync.quality import (
    build_sync_quality_report,
    write_sync_quality_report_with_manifest,
)
from posetestbot.web.paths import APP_ROOT

app = Blueprint("legacy_api", __name__)
job_runner = LocalJobRunner(APP_ROOT / "working_data" / "jobs")


def _env_bool(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


WEB_HOST = os.environ.get("POSETESTBOT_WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.environ.get("POSETESTBOT_WEB_PORT", "5000"))
WEB_DEBUG = _env_bool("POSETESTBOT_WEB_DEBUG", default=False)

CAPTURE_JOB_STAGE_IDS = {
    "capture_plan",
    "capture_plan_preflight",
    "capture_execution_plan",
    "capture_execution",
    "capture_rehearsal",
    "realsense_capture_smoke",
}
CAPTURE_JOB_SEQUENCE_IDS = {
    "fake_capture_rehearsal",
    "fake_capture_execution",
}
ACTIVE_JOB_STATUSES = {"queued", "running", "canceling"}
ROBOT_CONTROL_COMMANDS = {"start_iiwa", "stop_iiwa"}

COMMANDS = {
    "start_iiwa": {
        "label": "Start IIWA",
        "command": ["uv", "run", "python", "start_iiwa.py"],
        "resources": ["robot_command"],
    },
    "stop_iiwa": {
        "label": "Stop IIWA",
        "command": ["uv", "run", "python", "stop_iiwa.py"],
        "resources": ["robot_command"],
    },
}


def command_spec(command_name: str) -> dict | None:
    value = COMMANDS.get(command_name)
    if value is None:
        return None
    if isinstance(value, list):
        return {"label": command_name, "command": value}
    return value


def _robot_control_command_args(command_name: str, data: dict) -> tuple[list[str], dict]:
    if command_name not in ROBOT_CONTROL_COMMANDS:
        return [], {}

    ip_value = data.get("robot_ip", LAB_ROBOT_IP)
    if ip_value is None:
        ip_value = LAB_ROBOT_IP
    robot_ip = str(ip_value).strip()
    try:
        ipaddress.IPv4Address(robot_ip)
    except ipaddress.AddressValueError as exc:
        raise ValueError("robot_ip must be a valid IPv4 address") from exc

    port_value = data.get("robot_port", DEFAULT_ROBOT_PORT)
    if port_value is None:
        port_value = DEFAULT_ROBOT_PORT
    if isinstance(port_value, bool):
        raise ValueError("robot_port must be an integer from 1 to 65535")
    try:
        robot_port = int(str(port_value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("robot_port must be an integer from 1 to 65535") from exc
    if not 1 <= robot_port <= 65535:
        raise ValueError("robot_port must be an integer from 1 to 65535")

    args = [
        "--robot_mode",
        "real",
        "--ip_robot",
        robot_ip,
        "--port_robot",
        str(robot_port),
    ]
    parameters = {
        "robot_mode": "real",
        "robot_ip": robot_ip,
        "robot_port": robot_port,
    }
    return args, parameters


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def _job_log_artifacts(run_root: str | Path) -> list[dict]:
    artifacts = []
    for job in job_runner.list():
        job_run_root = job.parameters.get("run_root")
        if not job_run_root:
            continue
        if not _same_path(job_run_root, run_root):
            continue
        log_path = Path(job.log_path)
        artifacts.append(
            {
                "key": "log",
                "source": f"job:{job.id}",
                "path": job.log_path,
                "relative_path": None,
                "kind": "job_log",
                "exists": log_path.is_file(),
                "preview_type": "job_log",
                "size_bytes": log_path.stat().st_size if log_path.is_file() else None,
                "modified_at": None,
                "child_count": None,
                "job_id": job.id,
                "job_name": job.name,
                "job_status": job.status,
                "log_endpoint": f"/jobs/{job.id}/log",
            }
        )
    return artifacts


def _capture_job_kind(job) -> str | None:
    parameters = job.parameters or {}
    stage = parameters.get("pipeline_stage")
    sequence = parameters.get("pipeline_sequence")
    if stage in CAPTURE_JOB_STAGE_IDS:
        return "stage"
    if sequence in CAPTURE_JOB_SEQUENCE_IDS:
        return "sequence"
    if any(
        resource == "camera" or resource.startswith("camera:")
        for resource in (job.resources or [])
    ):
        return "camera_resource"
    return None


def _capture_job_summary(job) -> dict:
    parameters = dict(job.parameters or {})
    options = parameters.get("options")
    if not isinstance(options, dict):
        options = {}
    active = job.status in ACTIVE_JOB_STATUSES
    data = {
        "id": job.id,
        "name": job.name,
        "status": job.status,
        "kind": _capture_job_kind(job),
        "stage": parameters.get("pipeline_stage"),
        "sequence": parameters.get("pipeline_sequence"),
        "run_root": parameters.get("run_root"),
        "mode": options.get("mode"),
        "resources": list(job.resources or []),
        "message": job.message,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "ended_at": job.ended_at,
        "returncode": job.returncode,
        "active": active,
        "tail": list(job.tail or []),
        "log_endpoint": f"/jobs/{job.id}/log",
        "stop_endpoint": f"/capture/jobs/{job.id}/stop" if active else None,
    }
    return data


def _capture_jobs_for_run(run_root: str | Path | None = None) -> list[dict]:
    summaries = []
    for job in job_runner.list():
        if _capture_job_kind(job) is None:
            continue
        if run_root is not None:
            job_run_root = (job.parameters or {}).get("run_root")
            if not job_run_root or not _same_path(job_run_root, run_root):
                continue
        summaries.append(_capture_job_summary(job))
    return summaries


def _json_object_from_text(value: str | None, *, label: str) -> dict:
    if value is None or value.strip() == "":
        return {}
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must be a JSON object")
    return loaded


def _run_config_from_payload(data: dict):
    run_root = data.get("run_root")
    if not run_root:
        raise ValueError("run_root is required")
    sequence_options = data.get("sequence_options", data.get("options", {}))
    if isinstance(sequence_options, str):
        sequence_options = _json_object_from_text(
            sequence_options,
            label="sequence_options",
        )
    if not isinstance(sequence_options, dict):
        raise ValueError("sequence_options must be an object")
    mounting_mode = data.get("mounting_mode", "eye_in_hand")
    if _truthy(data.get("from_detected_sensors"), default=False) and not data.get(
        "sensors"
    ):
        status = collect_sensor_status()
        sensors = sensor_configs_from_status(
            status,
            default_mounting_mode=mounting_mode,
        )
        if not sensors:
            raise ValueError("No connected sensors were detected")
    else:
        sensors = sensor_configs_from_values(
            data.get("sensors"),
            default_mounting_mode=mounting_mode,
        )
    return create_run_config(
        run_root=run_root,
        run_name=data.get("run_name"),
        robot_mode=data.get("robot_mode", "fake"),
        resolution=data.get("resolution", "720p"),
        fps=int(data.get("fps", 6)),
        velocity_m_s=float(data.get("velocity", data.get("velocity_m_s", 0.2))),
        sensors=sensors,
        object_folder=data.get("object_folder", "object_models"),
        calibration_profiles=data.get("calibration_profiles") or None,
        sequence_id=data.get("sequence", data.get("sequence_id", "sync_to_bop_dry_run")),
        sequence_options=sequence_options,
        plan_only=_truthy(data.get("plan_only"), default=True),
    )


def _optional_nonnegative_int(value, *, label: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if number < 0:
        raise ValueError(f"{label} must be greater than or equal to 0")
    return number


def _optional_nonnegative_float(value, *, label: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, str) and value.lower() in {"none", "off"}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if number < 0:
        raise ValueError(f"{label} must be greater than or equal to 0")
    return number


def _json_object_option(value, *, run_root: str | Path, label: str) -> dict | None:
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a JSON object or path")
    path = Path(value)
    if not path.is_absolute() and not path.exists():
        path = Path(run_root) / path
    with open(path, "r") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} file must contain a JSON object")
    return loaded


def _calibration_target_option(data, *, run_root: str | Path) -> dict:
    base = data.get('target')
    if base is None:
        base = data.get('target_spec')
    target = _json_object_option(base, run_root=run_root, label='target') if base else None
    return normalize_calibration_target_spec(
        target,
        target_type=data.get('target_type') or None,
        dictionary=data.get('dictionary') or None,
        grid_size=data.get('grid_size') or None,
        marker_length=_optional_nonnegative_float(
            data.get('marker_length_mm'),
            label='marker_length_mm',
        ),
        marker_separation=_optional_nonnegative_float(
            data.get('marker_separation_mm'),
            label='marker_separation_mm',
        ),
        square_length=_optional_nonnegative_float(
            data.get('square_length_mm'),
            label='square_length_mm',
        ),
        checkerboard_size=data.get('checkerboard_size') or None,
    )


def _calibration_candidate_threshold_options(data: dict) -> dict:
    if _truthy(data.get('no_residual_thresholds')):
        return {
            'max_translation_residual_mm': None,
            'max_rotation_residual_deg': None,
        }
    return {
        'max_translation_residual_mm': _optional_nonnegative_float(
            data.get(
                'max_translation_residual_mm',
                DEFAULT_MAX_TRANSLATION_RESIDUAL_MM,
            ),
            label='max_translation_residual_mm',
        ),
        'max_rotation_residual_deg': _optional_nonnegative_float(
            data.get('max_rotation_residual_deg', DEFAULT_MAX_ROTATION_RESIDUAL_DEG),
            label='max_rotation_residual_deg',
        ),
    }


def _calibration_solver_options(data: dict) -> dict:
    hand_eye_method = str(data.get('hand_eye_method', DEFAULT_HAND_EYE_METHOD))
    if hand_eye_method not in HAND_EYE_METHODS:
        choices = ", ".join(sorted(HAND_EYE_METHODS))
        raise ValueError(f"hand_eye_method must be one of: {choices}")
    compare_hand_eye_methods = _truthy(
        data.get('compare_hand_eye_methods', False),
        default=False,
    )
    holdout_fraction = _optional_nonnegative_float(
        data.get('holdout_fraction', 0.0),
        label='holdout_fraction',
    )
    holdout_fraction = 0.0 if holdout_fraction is None else holdout_fraction
    if holdout_fraction >= 1.0:
        raise ValueError('holdout_fraction must be less than 1')
    return {
        'hand_eye_method': hand_eye_method,
        'compare_hand_eye_methods': compare_hand_eye_methods,
        'holdout_fraction': holdout_fraction,
        **_calibration_candidate_threshold_options(data),
    }


def _calibration_validation_options(data: dict) -> dict:
    min_inliers = _optional_nonnegative_int(
        data.get('min_inliers', DEFAULT_MIN_INLIERS),
        label='min_inliers',
    )
    return {
        'min_inliers': DEFAULT_MIN_INLIERS if min_inliers is None else min_inliers,
        'max_mean_translation_residual_mm': _optional_nonnegative_float(
            data.get(
                'max_mean_translation_residual_mm',
                DEFAULT_MAX_MEAN_TRANSLATION_RESIDUAL_MM,
            ),
            label='max_mean_translation_residual_mm',
        ),
        'max_mean_rotation_residual_deg': _optional_nonnegative_float(
            data.get(
                'max_mean_rotation_residual_deg',
                DEFAULT_MAX_MEAN_ROTATION_RESIDUAL_DEG,
            ),
            label='max_mean_rotation_residual_deg',
        ),
        'max_outlier_ratio': _optional_nonnegative_float(
            data.get('max_outlier_ratio', DEFAULT_MAX_OUTLIER_RATIO),
            label='max_outlier_ratio',
        ),
    }


def _truthy(value, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "Boolean value must be one of: true, false, 1, 0, yes, no, on, off"
    )


def _sync_quality_options(data: dict) -> dict:
    timestamp_source = data.get('require_timestamp_source') or None
    valid_sources = {"host_received", "host_wall", "sensor", "filename"}
    if timestamp_source is not None and timestamp_source not in valid_sources:
        raise ValueError(
            "require_timestamp_source must be one of: "
            + ", ".join(sorted(valid_sources))
        )
    return {
        "min_match_ratio": float(data.get('min_match_ratio', 0.8)),
        "max_dropped_frames": _optional_nonnegative_int(
            data.get('max_dropped_frames'),
            label='max_dropped_frames',
        ),
        "max_nearest_pose_delta_ms": (
            None
            if _truthy(data.get('no_nearest_pose_threshold'))
            else _optional_nonnegative_float(
                data.get('max_nearest_pose_delta_ms', 50.0),
                label='max_nearest_pose_delta_ms',
            )
        ),
        "require_timestamp_source": timestamp_source,
    }


@app.route('/run-command', methods=['POST'])
def run_command():
    data = request.get_json()
    if not data or 'command' not in data:
        return jsonify({'output': 'Invalid request: command not found'}), 400
    command = data['command']

    spec = command_spec(command)
    if spec is None:
        return jsonify({'output': 'Unknown command'}), 404
    try:
        command_args, command_parameters = _robot_control_command_args(command, data)
    except ValueError as exc:
        return jsonify({'output': str(exc)}), 400
    command_array = list(spec["command"]) + command_args

    try:
        job = job_runner.submit(
            name=command,
            command=command_array,
            cwd=APP_ROOT,
            resources=spec.get("resources", []),
            parameters={
                "command": command,
                "label": spec.get("label", command),
                "resources": spec.get("resources", []),
                **command_parameters,
            },
        )
    except ResourceBusyError as exc:
        return jsonify({'output': str(exc)}), 409
    return jsonify(
        {
            'output': f"Queued {command} as job {job.id}",
            'job_id': job.id,
            'status': job.status,
            'job': job.to_dict(),
        }
    ), 202


@app.route('/jobs', methods=['GET'])
def list_jobs():
    return jsonify(
        {
            'jobs': [job.to_dict() for job in job_runner.list()],
            'resources': job_runner.resource_holders(),
        }
    )


@app.route('/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    try:
        job = job_runner.get(job_id)
    except KeyError:
        return jsonify({'output': 'Unknown job'}), 404
    return jsonify({'job': job.to_dict()})


@app.route('/jobs/<job_id>/log', methods=['GET'])
def get_job_log(job_id):
    try:
        text = job_runner.log_text(job_id)
    except KeyError:
        return jsonify({'output': 'Unknown job'}), 404
    return Response(text, mimetype='text/plain')


@app.route('/jobs/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    try:
        job = job_runner.cancel(job_id)
    except KeyError:
        return jsonify({'output': 'Unknown job'}), 404
    return jsonify({'job': job.to_dict()})


@app.route('/capture/jobs', methods=['GET'])
def list_capture_jobs():
    run_root = request.args.get('run_root') or None
    jobs = _capture_jobs_for_run(run_root)
    status_artifact = None
    if run_root:
        try:
            status_artifact = load_capture_execution_status(run_root)
        except FileNotFoundError:
            status_artifact = None
        except ValueError as exc:
            status_artifact = {"error": str(exc)}
    return jsonify(
        {
            'run_root': run_root,
            'jobs': jobs,
            'active_count': sum(1 for job in jobs if job['active']),
            'resources': job_runner.resource_holders(),
            'status_artifact': status_artifact,
        }
    )


@app.route('/capture/status', methods=['GET'])
def capture_execution_status():
    run_root = request.args.get('run_root')
    if not run_root:
        return jsonify({'output': 'Missing run_root'}), 400
    try:
        status = load_capture_execution_status(run_root)
    except FileNotFoundError:
        return jsonify({'output': f'Missing {CAPTURE_EXECUTION_STATUS}'}), 404
    except ValueError as exc:
        return jsonify({'output': str(exc)}), 400
    return jsonify({'run_root': str(Path(run_root)), 'status': status})


@app.route('/capture/jobs/<job_id>/stop', methods=['POST'])
def stop_capture_job(job_id):
    try:
        job = job_runner.get(job_id)
    except KeyError:
        return jsonify({'output': 'Unknown job'}), 404
    if _capture_job_kind(job) is None:
        return jsonify({'output': 'Job is not a capture job'}), 400
    job = job_runner.cancel(job_id)
    return jsonify(
        {
            'output': f"Stopped capture job {job.id}",
            'job': job.to_dict(),
            'capture_job': _capture_job_summary(job),
        }
    )


@app.route('/sensors/adapters', methods=['GET'])
def sensor_adapters():
    return jsonify({'adapters': list_sensor_adapters()})


@app.route('/runtime/status', methods=['GET'])
def runtime_status():
    return jsonify(collect_runtime_status())


@app.route('/robot/status', methods=['GET'])
def robot_status():
    return jsonify(collect_robot_status())


@app.route('/hardware/status', methods=['GET', 'POST'])
def hardware_status():
    if request.method == 'POST':
        data = request.get_json()
        if not isinstance(data, dict) or 'run_root' not in data:
            return jsonify({'output': 'Invalid request: run_root required'}), 400
        include_sensors = not _truthy(data.get('no_sensors'), default=False)
        include_runtimes = not _truthy(data.get('no_runtimes'), default=False)
        try:
            path, report = write_hardware_status_report_with_manifest(
                data['run_root'],
                include_sensor_status=include_sensors,
                include_runtime_status=include_runtimes,
            )
        except ValueError as exc:
            return jsonify({'output': str(exc)}), 400
        return jsonify(
            {
                'output': f"Wrote {path}",
                'path': path.as_posix(),
                'run_root': str(Path(data['run_root'])),
                'report': report,
            }
        ), 201 if report['overall_status'] != 'error' else 409

    run_root = request.args.get('run_root')
    if not run_root:
        return jsonify({'output': 'Missing run_root'}), 400
    try:
        report = load_hardware_status_report(run_root)
    except FileNotFoundError as exc:
        return jsonify({'output': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'output': str(exc)}), 400
    return jsonify(
        {
            'run_root': str(Path(run_root)),
            'report': report,
        }
    )


@app.route('/artifacts', methods=['GET'])
def list_artifacts():
    run_root = request.args.get('run_root')
    if not run_root:
        return jsonify({'output': 'Missing run_root'}), 400
    try:
        artifacts = [record.to_dict() for record in collect_run_artifacts(run_root)]
    except FileNotFoundError as exc:
        return jsonify({'output': str(exc)}), 404
    artifacts.extend(_job_log_artifacts(run_root))
    return jsonify({'run_root': str(Path(run_root)), 'artifacts': artifacts})


@app.route('/artifacts/preview', methods=['GET'])
def artifact_preview():
    run_root = request.args.get('run_root')
    artifact_path = request.args.get('path')
    if not run_root or not artifact_path:
        return jsonify({'output': 'Missing run_root or path'}), 400
    if not Path(run_root).exists():
        return jsonify({'output': f'Run root not found: {run_root}'}), 404
    try:
        return jsonify(preview_artifact(run_root, artifact_path))
    except ArtifactPathError as exc:
        return jsonify({'output': str(exc)}), 400


@app.route('/artifacts/file', methods=['GET'])
def artifact_file():
    run_root = request.args.get('run_root')
    artifact_path = request.args.get('path')
    if not run_root or not artifact_path:
        return jsonify({'output': 'Missing run_root or path'}), 400
    if not Path(run_root).exists():
        return jsonify({'output': f'Run root not found: {run_root}'}), 404
    try:
        path = resolve_artifact_path(run_root, artifact_path)
    except ArtifactPathError as exc:
        return jsonify({'output': str(exc)}), 400
    if not path.exists():
        return jsonify({'output': f'Artifact file not found: {path}'}), 404
    if not path.is_file():
        return jsonify({'output': f'Artifact path is not a file: {path}'}), 400

    download = request.args.get('download', '').lower() in {'1', 'true', 'yes'}
    return send_file(path, conditional=True, as_attachment=download)


@app.route('/artifacts/bop-scene', methods=['GET'])
def artifact_bop_scene():
    run_root = request.args.get('run_root')
    scene_path = request.args.get('path')
    if not run_root or not scene_path:
        return jsonify({'output': 'Missing run_root or path'}), 400
    if not Path(run_root).exists():
        return jsonify({'output': f'Run root not found: {run_root}'}), 404
    try:
        frame_limit = int(request.args.get('frame_limit', '200'))
    except ValueError:
        return jsonify({'output': 'frame_limit must be an integer'}), 400
    try:
        return jsonify(
            bop_scene_detail(run_root, scene_path, frame_limit=frame_limit)
        )
    except FileNotFoundError as exc:
        return jsonify({'output': str(exc)}), 404
    except (ArtifactPathError, ValueError) as exc:
        return jsonify({'output': str(exc)}), 400


@app.route('/artifacts/bop-frame', methods=['GET'])
def artifact_bop_frame():
    run_root = request.args.get('run_root')
    scene_path = request.args.get('path')
    if not run_root or not scene_path:
        return jsonify({'output': 'Missing run_root or path'}), 400
    if not Path(run_root).exists():
        return jsonify({'output': f'Run root not found: {run_root}'}), 404
    try:
        image_id = int(request.args.get('image_id', '0'))
        row_limit = int(request.args.get('row_limit', '100'))
    except ValueError:
        return jsonify({'output': 'image_id and row_limit must be integers'}), 400
    try:
        return jsonify(
            bop_frame_detail(
                run_root,
                scene_path,
                image_id=image_id,
                row_limit=row_limit,
            )
        )
    except FileNotFoundError as exc:
        return jsonify({'output': str(exc)}), 404
    except (ArtifactPathError, ValueError) as exc:
        return jsonify({'output': str(exc)}), 400


@app.route('/artifacts/bop-frame-overlay', methods=['GET'])
def artifact_bop_frame_overlay():
    run_root = request.args.get('run_root')
    scene_path = request.args.get('path')
    if not run_root or not scene_path:
        return jsonify({'output': 'Missing run_root or path'}), 400
    if not Path(run_root).exists():
        return jsonify({'output': f'Run root not found: {run_root}'}), 404
    try:
        image_id = int(request.args.get('image_id', '0'))
        row_limit = int(request.args.get('row_limit', '20'))
    except ValueError:
        return jsonify({'output': 'image_id and row_limit must be integers'}), 400
    try:
        overlay = render_bop_frame_overlay_png(
            run_root,
            scene_path,
            image_id=image_id,
            row_limit=row_limit,
            include_masks=not _truthy(request.args.get('no_masks'), default=False),
            include_gt=not _truthy(request.args.get('no_gt'), default=False),
        )
    except FileNotFoundError as exc:
        return jsonify({'output': str(exc)}), 404
    except (ArtifactPathError, ValueError) as exc:
        return jsonify({'output': str(exc)}), 400
    return Response(overlay, mimetype='image/png')


@app.route('/pipeline/stages', methods=['GET'])
def pipeline_stages():
    return jsonify({'stages': list_pipeline_stages(PIPELINE_STAGES)})


@app.route('/pipeline/stages/<stage_id>', methods=['GET'])
def pipeline_stage(stage_id):
    try:
        stage = get_pipeline_stage(stage_id, registry=PIPELINE_STAGES)
    except ValueError:
        return jsonify({'output': 'Unknown pipeline stage'}), 404
    return jsonify({'stage': stage.to_dict()})


@app.route('/pipeline/sequences', methods=['GET'])
def pipeline_sequences():
    return jsonify({'sequences': list_pipeline_sequences(PIPELINE_SEQUENCES)})


@app.route('/pipeline/sequences/<sequence_id>', methods=['GET'])
def pipeline_sequence(sequence_id):
    try:
        sequence = get_pipeline_sequence(sequence_id, registry=PIPELINE_SEQUENCES)
    except ValueError:
        return jsonify({'output': 'Unknown pipeline sequence'}), 404
    return jsonify({'sequence': sequence.to_dict()})


@app.route('/pipeline/recommendations', methods=['GET'])
def pipeline_recommendations():
    run_root = request.args.get('run_root')
    if not run_root:
        return jsonify({'output': 'Missing run_root'}), 400
    try:
        return jsonify(build_pipeline_recommendations(run_root))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return jsonify({'output': str(exc)}), 400


@app.route('/run-config', methods=['GET', 'POST'])
def run_config():
    if request.method == 'POST':
        data = request.get_json()
        if not isinstance(data, dict):
            return jsonify({'output': 'Invalid request: JSON object required'}), 400
        try:
            config = _run_config_from_payload(data)
            path = write_run_config_with_manifest(data['run_root'], config)
            config_dict = config.to_dict()
            plan = sequence_plan_from_run_config(config_dict)
            preflight = run_preflight_queue_summary(data['run_root'], config_dict)
        except ValueError as exc:
            return jsonify({'output': str(exc)}), 400
        return jsonify(
            {
                'output': f"Wrote {path}",
                'path': path.as_posix(),
                'run_root': str(Path(data['run_root'])),
                'config': config_dict,
                'sequence_plan': plan.to_dict(),
                'preflight': preflight,
            }
        ), 201

    run_root = request.args.get('run_root')
    if not run_root:
        return jsonify({'output': 'Missing run_root'}), 400
    try:
        config = load_run_config_for_run_root(run_root)
        plan = sequence_plan_from_run_config(config)
        preflight = run_preflight_queue_summary(run_root, config)
    except FileNotFoundError as exc:
        return jsonify({'output': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'output': str(exc)}), 400
    return jsonify(
        {
            'run_root': str(Path(run_root)),
            'config': config,
            'sequence_plan': plan.to_dict(),
            'preflight': preflight,
        }
    )


@app.route('/capture-plan', methods=['GET', 'POST'])
def capture_plan_endpoint():
    if request.method == 'POST':
        data = request.get_json()
        if not isinstance(data, dict) or 'run_root' not in data:
            return jsonify({'output': 'Invalid request: run_root required'}), 400
        try:
            max_frames = _optional_nonnegative_int(
                data.get('max_frames'),
                label='max_frames',
            )
            config = load_run_config_for_run_root(data['run_root'])
            path, plan = write_capture_plan_with_manifest(
                data['run_root'],
                config,
                max_frames=max_frames,
            )
        except FileNotFoundError as exc:
            return jsonify({'output': str(exc)}), 404
        except ValueError as exc:
            return jsonify({'output': str(exc)}), 400
        return jsonify(
            {
                'output': f"Wrote {path}",
                'path': path.as_posix(),
                'run_root': str(Path(data['run_root'])),
                'capture_plan': plan.to_dict(),
            }
        ), 201

    run_root = request.args.get('run_root')
    if not run_root:
        return jsonify({'output': 'Missing run_root'}), 400
    try:
        plan = load_capture_plan(run_root)
    except FileNotFoundError as exc:
        return jsonify({'output': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'output': str(exc)}), 400
    path = Path(run_root) / CAPTURE_PLAN
    return jsonify(
        {
            'path': path.as_posix(),
            'run_root': str(Path(run_root)),
            'capture_plan': plan,
        }
    )


@app.route('/capture-plan/preflight', methods=['GET', 'POST'])
def capture_plan_preflight_endpoint():
    if request.method == 'POST':
        data = request.get_json()
        if not isinstance(data, dict) or 'run_root' not in data:
            return jsonify({'output': 'Invalid request: run_root required'}), 400
        include_sensors = not _truthy(data.get('no_sensors'), default=False)
        allow_real_robot = _truthy(data.get('allow_real_robot'), default=False)
        try:
            path, report = write_capture_plan_preflight_with_manifest(
                data['run_root'],
                include_sensor_status=include_sensors,
                allow_real_robot=allow_real_robot,
            )
        except FileNotFoundError as exc:
            return jsonify({'output': str(exc)}), 404
        except ValueError as exc:
            return jsonify({'output': str(exc)}), 400
        return jsonify(
            {
                'output': f"Wrote {path}",
                'path': path.as_posix(),
                'run_root': str(Path(data['run_root'])),
                'report': report,
            }
        ), 201 if report['overall_status'] != 'error' else 409

    run_root = request.args.get('run_root')
    if not run_root:
        return jsonify({'output': 'Missing run_root'}), 400
    include_sensors = request.args.get('include_sensors', 'true').lower() not in {
        '0',
        'false',
        'no',
    }
    allow_real_robot = request.args.get('allow_real_robot', 'false').lower() in {
        '1',
        'true',
        'yes',
    }
    try:
        report = build_capture_plan_preflight(
            run_root,
            include_sensor_status=include_sensors,
            allow_real_robot=allow_real_robot,
            write_plan_if_missing=False,
        )
    except FileNotFoundError as exc:
        return jsonify({'output': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'output': str(exc)}), 400
    return jsonify(
        {
            'run_root': str(Path(run_root)),
            'report': report,
        }
    )


@app.route('/capture-plan/execution', methods=['GET', 'POST'])
def capture_plan_execution_endpoint():
    if request.method == 'POST':
        data = request.get_json()
        if not isinstance(data, dict) or 'run_root' not in data:
            return jsonify({'output': 'Invalid request: run_root required'}), 400
        mode = str(data.get('mode') or 'pose_only_fake')
        allow_cameras = _truthy(data.get('allow_cameras'), default=False)
        allow_real_robot = _truthy(data.get('allow_real_robot'), default=False)
        include_sensor_status = _truthy(data.get('include_sensors'), default=False)
        try:
            path, plan = write_capture_execution_plan_with_manifest(
                data['run_root'],
                mode=mode,
                allow_cameras=allow_cameras,
                allow_real_robot=allow_real_robot,
                include_sensor_status=include_sensor_status,
            )
        except FileNotFoundError as exc:
            return jsonify({'output': str(exc)}), 404
        except ValueError as exc:
            return jsonify({'output': str(exc)}), 400
        return jsonify(
            {
                'output': f"Wrote {path}",
                'path': path.as_posix(),
                'run_root': str(Path(data['run_root'])),
                'plan': plan,
            }
        ), 201 if plan['status'] != 'error' else 409

    run_root = request.args.get('run_root')
    if not run_root:
        return jsonify({'output': 'Missing run_root'}), 400
    try:
        plan = load_capture_execution_plan(run_root)
    except FileNotFoundError as exc:
        return jsonify({'output': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'output': str(exc)}), 400
    path = Path(run_root) / CAPTURE_EXECUTION_PLAN
    return jsonify(
        {
            'path': path.as_posix(),
            'run_root': str(Path(run_root)),
            'plan': plan,
        }
    )


@app.route('/calibration/preflight', methods=['GET', 'POST'])
def calibration_preflight_endpoint():
    if request.method == 'POST':
        data = request.get_json()
        if not isinstance(data, dict) or 'run_root' not in data:
            return jsonify({'output': 'Invalid request: run_root required'}), 400
        require_valid = _truthy(data.get('require_valid'), default=False)
        min_observations = int(data.get('min_observations', 6))
        max_error = data.get('max_mean_reprojection_error_px', 2.0)
        max_error = None if max_error is None else float(max_error)
        try:
            path, report = write_calibration_preflight_with_manifest(
                data['run_root'],
                require_valid=require_valid,
                min_observations=min_observations,
                max_mean_reprojection_error_px=max_error,
            )
        except FileNotFoundError as exc:
            return jsonify({'output': str(exc)}), 404
        except ValueError as exc:
            return jsonify({'output': str(exc)}), 400
        return jsonify(
            {
                'output': f"Wrote {path}",
                'path': path.as_posix(),
                'run_root': str(Path(data['run_root'])),
                'report': report,
            }
        ), 201 if report['overall_status'] != 'error' else 409

    run_root = request.args.get('run_root')
    if not run_root:
        return jsonify({'output': 'Missing run_root'}), 400
    require_valid = request.args.get('require_valid', 'false').lower() in {
        '1',
        'true',
        'yes',
    }
    min_observations = int(request.args.get('min_observations', '6'))
    max_error_arg = request.args.get('max_mean_reprojection_error_px', '2.0')
    max_error = None if max_error_arg.lower() in {'none', 'off'} else float(max_error_arg)
    try:
        report = build_calibration_preflight(
            run_root,
            require_valid=require_valid,
            min_observations=min_observations,
            max_mean_reprojection_error_px=max_error,
        )
    except FileNotFoundError as exc:
        return jsonify({'output': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'output': str(exc)}), 400
    return jsonify(
        {
            'run_root': str(Path(run_root)),
            'report': report,
        }
    )


@app.route('/calibration/observations', methods=['GET', 'POST'])
def calibration_observations_endpoint():
    if request.method == 'POST':
        data = request.get_json()
        if not isinstance(data, dict) or 'run_root' not in data:
            return jsonify({'output': 'Invalid request: run_root required'}), 400
        try:
            min_marker_count = _optional_nonnegative_int(
                data.get('min_marker_count', 4),
                label='min_marker_count',
            )
            min_observations = _optional_nonnegative_int(
                data.get('min_observations', 6),
                label='min_observations',
            )
            if min_marker_count == 0:
                raise ValueError('min_marker_count must be at least 1')
            target = _calibration_target_option(data, run_root=data['run_root'])
            path, report = write_calibration_observations_with_manifest(
                data['run_root'],
                min_marker_count=4 if min_marker_count is None else min_marker_count,
                min_observations=0 if min_observations is None else min_observations,
                target=target,
            )
        except FileNotFoundError as exc:
            return jsonify({'output': str(exc)}), 404
        except ValueError as exc:
            return jsonify({'output': str(exc)}), 400
        return jsonify(
            {
                'output': f"Wrote {path}",
                'path': path.as_posix(),
                'run_root': str(Path(data['run_root'])),
                'report': report,
            }
        ), 201 if report['overall_status'] != 'error' else 409

    run_root = request.args.get('run_root')
    if not run_root:
        return jsonify({'output': 'Missing run_root'}), 400
    try:
        min_marker_count = _optional_nonnegative_int(
            request.args.get('min_marker_count', 4),
            label='min_marker_count',
        )
        min_observations = _optional_nonnegative_int(
            request.args.get('min_observations', 6),
            label='min_observations',
        )
        if min_marker_count == 0:
            raise ValueError('min_marker_count must be at least 1')
        target = _calibration_target_option(request.args, run_root=run_root)
        report = build_calibration_observations(
            run_root,
            min_marker_count=4 if min_marker_count is None else min_marker_count,
            min_observations=0 if min_observations is None else min_observations,
            target=target,
        )
    except FileNotFoundError as exc:
        return jsonify({'output': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'output': str(exc)}), 400
    return jsonify(
        {
            'run_root': str(Path(run_root)),
            'report': report,
        }
    )


@app.route('/calibration/candidates', methods=['GET', 'POST'])
def calibration_candidates_endpoint():
    if request.method == 'POST':
        data = request.get_json()
        if not isinstance(data, dict) or 'run_root' not in data:
            return jsonify({'output': 'Invalid request: run_root required'}), 400
        try:
            min_observations = _optional_nonnegative_int(
                data.get('min_observations', 6),
                label='min_observations',
            )
            if min_observations == 0:
                raise ValueError('min_observations must be at least 1')
            target_to_reference = _json_object_option(
                data.get('target_to_reference') or data.get('target_to_reference_path'),
                run_root=data['run_root'],
                label='target_to_reference',
            )
            threshold_options = _calibration_candidate_threshold_options(data)
            report_path, profiles_path, report = (
                write_calibration_candidates_with_manifest(
                    data['run_root'],
                    observations_path=data.get('observations'),
                    min_observations=(
                        6 if min_observations is None else min_observations
                    ),
                    target_to_reference=target_to_reference,
                    **threshold_options,
                )
            )
        except FileNotFoundError as exc:
            return jsonify({'output': str(exc)}), 404
        except ValueError as exc:
            return jsonify({'output': str(exc)}), 400
        return jsonify(
            {
                'output': f"Wrote {report_path}\nWrote {profiles_path}",
                'path': report_path.as_posix(),
                'profiles_path': profiles_path.as_posix(),
                'run_root': str(Path(data['run_root'])),
                'report': report,
            }
        ), 201 if report['overall_status'] != 'error' else 409

    run_root = request.args.get('run_root')
    if not run_root:
        return jsonify({'output': 'Missing run_root'}), 400
    try:
        min_observations = _optional_nonnegative_int(
            request.args.get('min_observations', 6),
            label='min_observations',
        )
        if min_observations == 0:
            raise ValueError('min_observations must be at least 1')
        target_to_reference = _json_object_option(
            request.args.get('target_to_reference'),
            run_root=run_root,
            label='target_to_reference',
        )
        threshold_options = _calibration_candidate_threshold_options(
            {
                'max_translation_residual_mm': request.args.get(
                    'max_translation_residual_mm',
                    DEFAULT_MAX_TRANSLATION_RESIDUAL_MM,
                ),
                'max_rotation_residual_deg': request.args.get(
                    'max_rotation_residual_deg',
                    DEFAULT_MAX_ROTATION_RESIDUAL_DEG,
                ),
                'no_residual_thresholds': request.args.get('no_residual_thresholds'),
            }
        )
        report = build_calibration_candidates(
            run_root,
            observations_path=request.args.get('observations'),
            min_observations=6 if min_observations is None else min_observations,
            target_to_reference=target_to_reference,
            **threshold_options,
        )
    except FileNotFoundError as exc:
        return jsonify({'output': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'output': str(exc)}), 400
    return jsonify(
        {
            'run_root': str(Path(run_root)),
            'report': report,
        }
    )


@app.route('/calibration/solver', methods=['GET', 'POST'])
def calibration_solver_endpoint():
    if request.method == 'POST':
        data = request.get_json()
        if not isinstance(data, dict) or 'run_root' not in data:
            return jsonify({'output': 'Invalid request: run_root required'}), 400
        try:
            min_observations = _optional_nonnegative_int(
                data.get('min_observations', 6),
                label='min_observations',
            )
            if min_observations == 0:
                raise ValueError('min_observations must be at least 1')
            target_to_reference = _json_object_option(
                data.get('target_to_reference') or data.get('target_to_reference_path'),
                run_root=data['run_root'],
                label='target_to_reference',
            )
            solver_options = _calibration_solver_options(data)
            report_path, profiles_path, report = write_calibration_solver_with_manifest(
                data['run_root'],
                observations_path=data.get('observations'),
                min_observations=6 if min_observations is None else min_observations,
                target_to_reference=target_to_reference,
                **solver_options,
            )
        except FileNotFoundError as exc:
            return jsonify({'output': str(exc)}), 404
        except ValueError as exc:
            return jsonify({'output': str(exc)}), 400
        return jsonify(
            {
                'output': f"Wrote {report_path}\nWrote {profiles_path}",
                'path': report_path.as_posix(),
                'profiles_path': profiles_path.as_posix(),
                'run_root': str(Path(data['run_root'])),
                'report': report,
            }
        ), 201 if report['overall_status'] != 'error' else 409

    run_root = request.args.get('run_root')
    if not run_root:
        return jsonify({'output': 'Missing run_root'}), 400
    try:
        min_observations = _optional_nonnegative_int(
            request.args.get('min_observations', 6),
            label='min_observations',
        )
        if min_observations == 0:
            raise ValueError('min_observations must be at least 1')
        target_to_reference = _json_object_option(
            request.args.get('target_to_reference'),
            run_root=run_root,
            label='target_to_reference',
        )
        solver_options = _calibration_solver_options(
            {
                'hand_eye_method': request.args.get(
                    'hand_eye_method',
                    DEFAULT_HAND_EYE_METHOD,
                ),
                'holdout_fraction': request.args.get('holdout_fraction', 0.0),
                'compare_hand_eye_methods': request.args.get(
                    'compare_hand_eye_methods',
                    False,
                ),
                'max_translation_residual_mm': request.args.get(
                    'max_translation_residual_mm',
                    DEFAULT_MAX_TRANSLATION_RESIDUAL_MM,
                ),
                'max_rotation_residual_deg': request.args.get(
                    'max_rotation_residual_deg',
                    DEFAULT_MAX_ROTATION_RESIDUAL_DEG,
                ),
                'no_residual_thresholds': request.args.get('no_residual_thresholds'),
            }
        )
        report = build_calibration_solver(
            run_root,
            observations_path=request.args.get('observations'),
            min_observations=6 if min_observations is None else min_observations,
            target_to_reference=target_to_reference,
            **solver_options,
        )
    except FileNotFoundError as exc:
        return jsonify({'output': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'output': str(exc)}), 400
    return jsonify(
        {
            'run_root': str(Path(run_root)),
            'report': report,
        }
    )


@app.route('/calibration/validation', methods=['GET', 'POST'])
def calibration_validation_endpoint():
    if request.method == 'POST':
        data = request.get_json()
        if not isinstance(data, dict) or 'run_root' not in data:
            return jsonify({'output': 'Invalid request: run_root required'}), 400
        try:
            validation_options = _calibration_validation_options(data)
            if validation_options['min_inliers'] < 1:
                raise ValueError('min_inliers must be at least 1')
            report_path, promoted_path, report = (
                write_calibration_validation_with_manifest(
                    data['run_root'],
                    candidates_path=data.get('candidates'),
                    profiles_path=data.get('profiles'),
                    promote=_truthy(data.get('promote'), default=False),
                    output_profiles_path=data.get('output_profiles'),
                    operator=data.get('operator') or None,
                    **validation_options,
                )
            )
        except FileNotFoundError as exc:
            return jsonify({'output': str(exc)}), 404
        except ValueError as exc:
            return jsonify({'output': str(exc)}), 400
        return jsonify(
            {
                'output': (
                    f"Wrote {report_path}"
                    + (f"\nWrote {promoted_path}" if promoted_path else "")
                ),
                'path': report_path.as_posix(),
                'promoted_path': promoted_path.as_posix() if promoted_path else None,
                'run_root': str(Path(data['run_root'])),
                'report': report,
            }
        ), 201 if report['overall_status'] != 'error' else 409

    run_root = request.args.get('run_root')
    if not run_root:
        return jsonify({'output': 'Missing run_root'}), 400
    try:
        validation_options = _calibration_validation_options(
            {
                'min_inliers': request.args.get('min_inliers', DEFAULT_MIN_INLIERS),
                'max_mean_translation_residual_mm': request.args.get(
                    'max_mean_translation_residual_mm',
                    DEFAULT_MAX_MEAN_TRANSLATION_RESIDUAL_MM,
                ),
                'max_mean_rotation_residual_deg': request.args.get(
                    'max_mean_rotation_residual_deg',
                    DEFAULT_MAX_MEAN_ROTATION_RESIDUAL_DEG,
                ),
                'max_outlier_ratio': request.args.get(
                    'max_outlier_ratio',
                    DEFAULT_MAX_OUTLIER_RATIO,
                ),
            }
        )
        if validation_options['min_inliers'] < 1:
            raise ValueError('min_inliers must be at least 1')
        report = build_calibration_validation(
            run_root,
            candidates_path=request.args.get('candidates'),
            profiles_path=request.args.get('profiles'),
            **validation_options,
        )
    except FileNotFoundError as exc:
        return jsonify({'output': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'output': str(exc)}), 400
    return jsonify(
        {
            'run_root': str(Path(run_root)),
            'report': report,
        }
    )


@app.route('/sync/quality', methods=['GET', 'POST'])
def sync_quality_endpoint():
    if request.method == 'POST':
        data = request.get_json()
        if not isinstance(data, dict) or 'run_root' not in data:
            return jsonify({'output': 'Invalid request: run_root required'}), 400
        try:
            options = _sync_quality_options(data)
            path, report = write_sync_quality_report_with_manifest(
                data['run_root'],
                **options,
            )
        except FileNotFoundError as exc:
            return jsonify({'output': str(exc)}), 404
        except ValueError as exc:
            return jsonify({'output': str(exc)}), 400
        return jsonify(
            {
                'output': f"Wrote {path}",
                'path': path.as_posix(),
                'run_root': str(Path(data['run_root'])),
                'report': report,
            }
        ), 201 if report['overall_status'] != 'error' else 409

    run_root = request.args.get('run_root')
    if not run_root:
        return jsonify({'output': 'Missing run_root'}), 400
    data = {
        'min_match_ratio': request.args.get('min_match_ratio', 0.8),
        'max_dropped_frames': request.args.get('max_dropped_frames'),
        'max_nearest_pose_delta_ms': request.args.get(
            'max_nearest_pose_delta_ms',
            50.0,
        ),
        'no_nearest_pose_threshold': request.args.get(
            'no_nearest_pose_threshold',
        ),
        'require_timestamp_source': request.args.get('require_timestamp_source'),
    }
    try:
        report = build_sync_quality_report(
            run_root,
            **_sync_quality_options(data),
        )
    except FileNotFoundError as exc:
        return jsonify({'output': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'output': str(exc)}), 400
    return jsonify(
        {
            'run_root': str(Path(run_root)),
            'report': report,
        }
    )


@app.route('/pipeline/run', methods=['POST'])
def run_pipeline_stage():
    data = request.get_json()
    if not data or 'stage' not in data or 'run_root' not in data:
        return jsonify({'output': 'Invalid request: stage and run_root required'}), 400

    options = data.get('options') or {}
    if not isinstance(options, dict):
        return jsonify({'output': 'Invalid request: options must be an object'}), 400

    try:
        pipeline_job = build_pipeline_job(
            stage_id=data['stage'],
            run_root=data['run_root'],
            options=options,
            registry=PIPELINE_STAGES,
        )
        job = job_runner.submit(
            name=f"pipeline:{pipeline_job.stage_id}",
            command=pipeline_job.command,
            cwd=APP_ROOT,
            resources=pipeline_job.resources,
            parameters=pipeline_job.parameters,
        )
    except ValueError as exc:
        return jsonify({'output': str(exc)}), 400
    except ResourceBusyError as exc:
        return jsonify({'output': str(exc)}), 409

    return jsonify(
        {
            'output': f"Queued pipeline stage {pipeline_job.stage_id} as job {job.id}",
            'job_id': job.id,
            'status': job.status,
            'job': job.to_dict(),
            'pipeline': pipeline_job.to_dict(),
        }
    ), 202


@app.route('/pipeline/preflight', methods=['GET', 'POST'])
def pipeline_preflight():
    if request.method == 'POST':
        data = request.get_json()
        if not isinstance(data, dict) or 'run_root' not in data:
            return jsonify({'output': 'Invalid request: run_root required'}), 400
        include_sensor_status = _truthy(
            data.get('include_sensors'),
            default=True,
        )
        include_runtime_status = _truthy(
            data.get('include_runtimes'),
            default=True,
        )
        try:
            path, report = write_run_preflight_with_manifest(
                data['run_root'],
                include_sensor_status=include_sensor_status,
                include_runtime_status=include_runtime_status,
            )
        except FileNotFoundError as exc:
            return jsonify({'output': str(exc)}), 404
        except ValueError as exc:
            return jsonify({'output': str(exc)}), 400
        return jsonify(
            {
                'output': f"Wrote {path}",
                'path': path.as_posix(),
                'run_root': str(Path(data['run_root'])),
                'report': report,
            }
        ), 201 if report['overall_status'] != 'error' else 409

    run_root = request.args.get('run_root')
    if not run_root:
        return jsonify({'output': 'Missing run_root'}), 400
    include_sensor_status = request.args.get('include_sensors', 'true').lower() not in {
        '0',
        'false',
        'no',
    }
    include_runtime_status = request.args.get('include_runtimes', 'true').lower() not in {
        '0',
        'false',
        'no',
    }
    try:
        return jsonify(
            build_run_preflight(
                run_root,
                include_sensor_status=include_sensor_status,
                include_runtime_status=include_runtime_status,
            )
        )
    except FileNotFoundError as exc:
        return jsonify({'output': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'output': str(exc)}), 400


@app.route('/pipeline/run-sequence', methods=['POST'])
def run_pipeline_sequence():
    data = request.get_json()
    if not data or 'sequence' not in data or 'run_root' not in data:
        return jsonify({'output': 'Invalid request: sequence and run_root required'}), 400

    options = data.get('options') or {}
    if not isinstance(options, dict):
        return jsonify({'output': 'Invalid request: options must be an object'}), 400

    plan_only = data.get('plan_only', False)
    if not isinstance(plan_only, bool):
        return jsonify({'output': 'Invalid request: plan_only must be a boolean'}), 400

    try:
        sequence_job = build_sequence_job(
            sequence_id=data['sequence'],
            run_root=data['run_root'],
            options=options,
            plan_only=plan_only,
            sequence_registry=PIPELINE_SEQUENCES,
            stage_registry=PIPELINE_STAGES,
        )
        job = job_runner.submit(
            name=f"pipeline-sequence:{sequence_job.sequence_id}",
            command=sequence_job.command,
            cwd=APP_ROOT,
            resources=sequence_job.resources,
            parameters=sequence_job.parameters,
        )
    except ValueError as exc:
        return jsonify({'output': str(exc)}), 400
    except ResourceBusyError as exc:
        return jsonify({'output': str(exc)}), 409

    return jsonify(
        {
            'output': (
                f"Queued pipeline sequence {sequence_job.sequence_id} "
                f"as job {job.id}"
            ),
            'job_id': job.id,
            'status': job.status,
            'job': job.to_dict(),
            'sequence': sequence_job.to_dict(),
        }
    ), 202


@app.route('/pipeline/run-config', methods=['POST'])
def run_pipeline_from_config():
    data = request.get_json()
    if not data or 'run_root' not in data:
        return jsonify({'output': 'Invalid request: run_root required'}), 400

    try:
        config = load_run_config_for_run_root(data['run_root'])
        preflight = run_preflight_queue_summary(data['run_root'], config)
        if (
            preflight["queue_blocker"] == "missing_preflight"
            and not _truthy(data.get('allow_missing_preflight'), default=False)
        ):
            return jsonify(
                {
                    'output': (
                        f"{RUN_PREFLIGHT_REPORT} is missing; write a preflight "
                        "report before queueing, or set "
                        "allow_missing_preflight=true."
                    ),
                    'preflight': preflight,
                    'preflight_path': preflight["path"],
                }
            ), 409
        if preflight["queue_blocker"] == "invalid_preflight":
            return jsonify(
                {
                    'output': (
                        f"{RUN_PREFLIGHT_REPORT} is invalid; write a fresh "
                        "preflight report before queueing."
                    ),
                    'preflight': preflight,
                    'preflight_path': preflight["path"],
                }
            ), 409
        preflight_report = load_run_preflight_report(data['run_root'])
        if preflight_report is not None:
            if (
                preflight["queue_blocker"] == "failed_preflight"
                and not _truthy(data.get('allow_failed_preflight'), default=False)
            ):
                return jsonify(
                    {
                        'output': (
                            f"{RUN_PREFLIGHT_REPORT} has overall_status=error; "
                            "write or inspect a passing preflight report before "
                            "queueing, or set allow_failed_preflight=true."
                        ),
                        'preflight_report': preflight_report,
                        'preflight': preflight,
                        'preflight_path': preflight["path"],
                    }
                ), 409
            if (
                preflight["queue_blocker"] == "stale_preflight"
                and not _truthy(data.get('allow_stale_preflight'), default=False)
            ):
                return jsonify(
                    {
                        'output': (
                            f"{RUN_PREFLIGHT_REPORT} does not match the current "
                            f"{RUN_CONFIG}; write a fresh preflight report before "
                            "queueing, or set allow_stale_preflight=true."
                        ),
                        'preflight_report': preflight_report,
                        'preflight': preflight,
                        'preflight_path': preflight["path"],
                    }
                ), 409
        sequence_job = build_sequence_job_from_run_config(
            config,
            sequence_registry=PIPELINE_SEQUENCES,
            stage_registry=PIPELINE_STAGES,
        )
        parameters = dict(sequence_job.parameters)
        parameters["run_config"] = RUN_CONFIG
        parameters["run_config_path"] = (
            Path(data['run_root']) / RUN_CONFIG
        ).as_posix()
        job = job_runner.submit(
            name=f"pipeline-run-config:{sequence_job.sequence_id}",
            command=sequence_job.command,
            cwd=APP_ROOT,
            resources=sequence_job.resources,
            parameters=parameters,
        )
    except FileNotFoundError as exc:
        return jsonify({'output': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'output': str(exc)}), 400
    except ResourceBusyError as exc:
        return jsonify({'output': str(exc)}), 409

    return jsonify(
        {
            'output': (
                f"Queued run config sequence {sequence_job.sequence_id} "
                f"as job {job.id}"
            ),
            'job_id': job.id,
            'status': job.status,
            'job': job.to_dict(),
            'run_config': config,
            'sequence': sequence_job.to_dict(),
            'preflight': preflight,
        }
    ), 202

if __name__ == '__main__':
    from posetestbot.web.app import app as flask_app

    flask_app.run(host=WEB_HOST, port=WEB_PORT, debug=WEB_DEBUG)
