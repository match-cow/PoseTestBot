import json
import os
from html import escape
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file

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
from posetestbot.io.artifact_browser import (
    ArtifactPathError,
    bop_frame_detail,
    bop_result_detail,
    bop_scene_detail,
    collect_run_artifacts,
    metric_dashboard_summary,
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
    build_hardware_status_report,
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
from posetestbot.sensors.status import collect_sensor_status, parse_expected_counts
from posetestbot.robot.status import collect_robot_status
from posetestbot.sync.quality import (
    build_sync_quality_report,
    write_sync_quality_report_with_manifest,
)

app = Flask(__name__)
job_runner = LocalJobRunner(Path("working_data") / "jobs")


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
ACTIVE_JOB_STATUSES = {"queued", "running"}

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
    "realsense_multi": {
        "label": "Run RealSense",
        "command": ["uv", "run", "python", "realsense_multi.py"],
        "resources": ["camera"],
    },
}


def command_spec(command_name: str) -> dict | None:
    value = COMMANDS.get(command_name)
    if value is None:
        return None
    if isinstance(value, list):
        return {"label": command_name, "command": value}
    return value


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
    command = parameters.get("command")
    if stage in CAPTURE_JOB_STAGE_IDS:
        return "stage"
    if sequence in CAPTURE_JOB_SEQUENCE_IDS:
        return "sequence"
    if command == "realsense_multi":
        return "legacy_command"
    if "camera" in (job.resources or []):
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
        plan_only=bool(data.get("plan_only", True)),
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
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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


def _sequence_options_html(default_sequence_id: str = "sync_to_bop_dry_run") -> str:
    options = []
    for sequence in list_pipeline_sequences(PIPELINE_SEQUENCES):
        sequence_id = str(sequence["id"])
        selected = " selected" if sequence_id == default_sequence_id else ""
        escaped_id = escape(sequence_id, quote=True)
        options.append(
            f'<option value="{escaped_id}"{selected}>{escaped_id}</option>'
        )
    return "\n".join(options)


@app.route('/')
def index():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PoseTestBot Control</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="container mt-4">
            <h1 class="mb-4">PoseTestBot Control</h1>
            <div class="d-flex gap-2 mb-4">
                <button class="btn btn-primary" onclick="runCommand('start_iiwa')">Start IIWA</button>
                <button class="btn btn-danger" onclick="runCommand('stop_iiwa')">Stop IIWA</button>
                <button class="btn btn-success" onclick="runCommand('realsense_multi')">Run RealSense</button>
            </div>

            <section class="border rounded p-3 mb-4">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h2 class="h5 mb-0">Robot</h2>
                    <button class="btn btn-sm btn-outline-secondary" onclick="loadRobotStatus()">Refresh</button>
                </div>
                <div id="robotStatusPanel" class="list-group small">
                    <div class="list-group-item text-muted">Refresh to inspect the selected iiwa profile.</div>
                </div>
            </section>

            <section class="border rounded p-3 mb-4">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h2 class="h5 mb-0">Sensors</h2>
                    <div class="d-flex gap-2">
                        <button class="btn btn-sm btn-outline-secondary" onclick="loadSensorAdapters()">Adapters</button>
                        <button class="btn btn-sm btn-outline-secondary" onclick="loadSensorStatus()">Refresh</button>
                    </div>
                </div>
                <div id="sensorStatusPanel" class="list-group small">
                    <div class="list-group-item text-muted">Refresh to check SDK availability and connected sensors.</div>
                </div>
            </section>

            <section class="border rounded p-3 mb-4">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h2 class="h5 mb-0">Runtimes</h2>
                    <div class="d-flex gap-2">
                        <button class="btn btn-sm btn-outline-secondary" onclick="writeHardwareStatus()">Snapshot</button>
                        <button class="btn btn-sm btn-outline-secondary" onclick="loadHardwareStatus()">Load Snapshot</button>
                        <button class="btn btn-sm btn-outline-secondary" onclick="loadRuntimeStatus()">Refresh</button>
                    </div>
                </div>
                <div id="runtimeStatusPanel" class="list-group small">
                    <div class="list-group-item text-muted">Refresh to check external runtime readiness.</div>
                </div>
            </section>

            <section class="border rounded p-3 mb-4">
                <h2 class="h5 mb-3">Run Config</h2>
                <div class="row g-3">
                    <div class="col-md-6">
                        <label for="runRoot" class="form-label">Run Root</label>
                        <input id="runRoot" class="form-control" value="working_data/example_run">
                    </div>
                    <div class="col-md-3">
                        <label for="robotMode" class="form-label">Robot</label>
                        <select id="robotMode" class="form-select">
                            <option value="fake" selected>fake</option>
                            <option value="real">real</option>
                        </select>
                    </div>
                    <div class="col-md-3">
                        <label for="sequenceId" class="form-label">Sequence</label>
                        <select id="sequenceId" class="form-select">
                            __SEQUENCE_OPTIONS__
                        </select>
                    </div>
                    <div class="col-md-2">
                        <label for="resolution" class="form-label">Resolution</label>
                        <input id="resolution" class="form-control" value="720p">
                    </div>
                    <div class="col-md-2">
                        <label for="fps" class="form-label">FPS</label>
                        <input id="fps" type="number" min="1" class="form-control" value="6">
                    </div>
                    <div class="col-md-2">
                        <label for="velocity" class="form-label">Velocity</label>
                        <input id="velocity" type="number" step="0.01" min="0" class="form-control" value="0.2">
                    </div>
                    <div class="col-md-6">
                        <label for="objectFolder" class="form-label">Object Folder</label>
                        <input id="objectFolder" class="form-control" value="object_models">
                    </div>
                    <div class="col-md-6">
                        <label for="calibrationProfiles" class="form-label">Calibration Profiles</label>
                        <input id="calibrationProfiles" class="form-control" placeholder="/tmp/posetestbot_calibration_profiles.json">
                    </div>
                    <div class="col-md-6">
                        <label for="sensors" class="form-label">Sensors</label>
                        <textarea id="sensors" class="form-control" rows="5">realsense:825412070181:eye_in_hand:RealSense 825412070181
realsense:033422071805:eye_in_hand:RealSense 033422071805
realsense:923322072633:eye_in_hand:RealSense 923322072633
luxonis:auto:eye_in_hand:Luxonis OAK-D Pro
zed_2i:auto:eye_in_hand:Stereolabs ZED 2i</textarea>
                    </div>
                    <div class="col-md-6">
                        <label for="sequenceOptions" class="form-label">Sequence Options</label>
                        <textarea id="sequenceOptions" class="form-control" rows="5">{}</textarea>
                    </div>
                    <div class="col-12 d-flex gap-2 align-items-center">
                        <div class="form-check me-2">
                            <input id="planOnly" class="form-check-input" type="checkbox" checked>
                            <label for="planOnly" class="form-check-label">Plan only</label>
                        </div>
                        <div class="input-group input-group-sm w-auto">
                            <span class="input-group-text">Execution</span>
                            <select id="captureExecutionMode" class="form-select">
                                <option value="pose_only_fake" selected>pose_only_fake</option>
                                <option value="full">full</option>
                                <option value="plan_only">plan_only</option>
                            </select>
                        </div>
                        <div class="form-check me-2">
                            <input id="allowCaptureCameras" class="form-check-input" type="checkbox">
                            <label for="allowCaptureCameras" class="form-check-label">Allow cameras</label>
                        </div>
                        <div class="form-check me-2">
                            <input id="includeCaptureSensors" class="form-check-input" type="checkbox">
                            <label for="includeCaptureSensors" class="form-check-label">Check sensors</label>
                        </div>
                        <div class="form-check me-2">
                            <input id="allowCaptureRealRobot" class="form-check-input" type="checkbox">
                            <label for="allowCaptureRealRobot" class="form-check-label">Allow real robot</label>
                        </div>
                        <button class="btn btn-outline-primary" onclick="saveRunConfig()">Save Config</button>
                        <button class="btn btn-outline-secondary" onclick="loadRunConfig()">Load Config</button>
                        <button class="btn btn-outline-secondary" onclick="preflightRunConfig()">Preflight</button>
                        <button class="btn btn-outline-secondary" onclick="writeRunPreflight()">Write Preflight</button>
                        <button class="btn btn-outline-secondary" onclick="preflightCalibration()">Preflight Calibration</button>
                        <button class="btn btn-outline-secondary" onclick="buildCalibrationObservations()">Calibration Observations</button>
                        <button class="btn btn-outline-secondary" onclick="solveCalibration()">Solve Calibration</button>
                        <button class="btn btn-outline-secondary" onclick="buildCalibrationCandidates()">Calibration Candidates</button>
                        <button class="btn btn-outline-secondary" onclick="validateCalibrationCandidates()">Validate Calibration</button>
                        <button class="btn btn-outline-secondary" onclick="checkSyncQuality()">Sync Quality</button>
                        <button class="btn btn-outline-secondary" onclick="writeCapturePlan()">Write Capture Plan</button>
                        <button class="btn btn-outline-secondary" onclick="loadCapturePlan()">Load Capture Plan</button>
                        <button class="btn btn-outline-secondary" onclick="preflightCapturePlan()">Preflight Capture Plan</button>
                        <button class="btn btn-outline-secondary" onclick="writeCaptureExecutionPlan()">Plan Execution</button>
                        <button class="btn btn-outline-secondary" onclick="loadCaptureExecutionPlan()">Load Execution Plan</button>
                        <button class="btn btn-outline-secondary" onclick="queueCaptureExecution()">Queue Execution</button>
                        <button class="btn btn-outline-secondary" onclick="queueCaptureRehearsal()">Queue Fake Pose Rehearsal</button>
                        <div class="form-check me-2">
                            <input id="allowMissingPreflight" class="form-check-input" type="checkbox">
                            <label for="allowMissingPreflight" class="form-check-label">Missing preflight</label>
                        </div>
                        <div class="form-check me-2">
                            <input id="allowFailedPreflight" class="form-check-input" type="checkbox">
                            <label for="allowFailedPreflight" class="form-check-label">Failed preflight</label>
                        </div>
                        <div class="form-check me-2">
                            <input id="allowStalePreflight" class="form-check-input" type="checkbox">
                            <label for="allowStalePreflight" class="form-check-label">Stale preflight</label>
                        </div>
                        <button class="btn btn-primary" onclick="queueRunConfig()">Queue Config</button>
                    </div>
                </div>
                <div id="preflightPanel" class="list-group small mt-3">
                    <div class="list-group-item text-muted">Preflight the saved run config before queueing.</div>
                </div>
                <div id="capturePlanPanel" class="list-group small mt-3">
                    <div class="list-group-item text-muted">Write a capture plan to inspect startup commands before launching hardware.</div>
                </div>
            </section>

            <section class="border rounded p-3 mb-4">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h2 class="h5 mb-0">Recommended Steps</h2>
                    <button class="btn btn-sm btn-outline-secondary" onclick="loadRecommendations()">Refresh</button>
                </div>
                <div id="recommendationsPanel" class="list-group small">
                    <div class="list-group-item text-muted">Refresh to see artifact-driven next steps for this run.</div>
                </div>
            </section>

            <section class="border rounded p-3 mb-4">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h2 class="h5 mb-0">Capture Activity</h2>
                    <div class="d-flex gap-2">
                        <button class="btn btn-sm btn-outline-secondary" onclick="loadCaptureStatus()">Status</button>
                        <button class="btn btn-sm btn-outline-secondary" onclick="refreshCaptureJobs()">Refresh</button>
                    </div>
                </div>
                <div id="captureJobsPanel" class="list-group small">
                    <div class="list-group-item text-muted">Refresh to inspect active capture jobs and stop supervised capture runs.</div>
                </div>
            </section>

            <section class="border rounded p-3 mb-4">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h2 class="h5 mb-0">Jobs</h2>
                    <button class="btn btn-sm btn-outline-secondary" onclick="refreshJobs()">Refresh</button>
                </div>
                <div id="resourceStatus" class="small text-muted mb-2"></div>
                <div id="jobsPanel" class="list-group"></div>
            </section>

            <section class="border rounded p-3 mb-4">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h2 class="h5 mb-0">Artifacts</h2>
                    <div class="d-flex gap-2">
                        <button class="btn btn-sm btn-outline-secondary" onclick="listArtifacts()">List</button>
                        <button class="btn btn-sm btn-outline-secondary" onclick="loadMetrics()">Metrics</button>
                    </div>
                </div>
                <div class="input-group input-group-sm mb-3">
                    <span class="input-group-text">Path</span>
                    <input id="artifactPath" class="form-control" value="dataset_manifest.json">
                    <button class="btn btn-outline-secondary" onclick="previewArtifact()">Preview</button>
                </div>
                <div id="artifactsPanel" class="list-group"></div>
                <div id="metricsPanel" class="list-group small mt-3">
                    <div class="list-group-item text-muted">Load metrics for a compact dashboard.</div>
                </div>
            </section>

            <section class="border rounded p-3 mb-4">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h2 class="h5 mb-0">BOP Inspect</h2>
                    <div class="d-flex gap-2">
                        <button class="btn btn-sm btn-outline-secondary" onclick="loadBopFrame()">Frame</button>
                        <button class="btn btn-sm btn-outline-secondary" onclick="loadBopResult()">Result</button>
                    </div>
                </div>
                <div class="row g-2 mb-3">
                    <div class="col-md-6">
                        <label for="bopScenePath" class="form-label small mb-1">Scene Path</label>
                        <input id="bopScenePath" class="form-control form-control-sm" value="bop/realsense_123/test/000001">
                    </div>
                    <div class="col-md-2">
                        <label for="bopImageId" class="form-label small mb-1">Image ID</label>
                        <input id="bopImageId" type="number" min="0" class="form-control form-control-sm" value="0">
                    </div>
                    <div class="col-md-4">
                        <label for="bopResultPath" class="form-label small mb-1">Result CSV</label>
                        <input id="bopResultPath" class="form-control form-control-sm" value="results/bop/foundationpose_bop-test.csv">
                    </div>
                </div>
                <div id="bopInspectorPanel" class="border rounded p-3 small text-muted">
                    Load a BOP frame or result file for compact inspection.
                </div>
            </section>

            <section class="border rounded p-3">
                <h2 class="h5 mb-3">Output</h2>
                <pre id="output" class="bg-light border rounded p-3 mb-0"></pre>
            </section>
        </div>

        <script>
            function output(text) {
                document.getElementById('output').textContent = text;
            }

            function escapeHtml(value) {
                return String(value)
                    .replaceAll('&', '&amp;')
                    .replaceAll('<', '&lt;')
                    .replaceAll('>', '&gt;')
                    .replaceAll('"', '&quot;')
                    .replaceAll("'", '&#039;');
            }

            function runRootValue() {
                return document.getElementById('runRoot').value.trim();
            }

            function runConfigPayload() {
                const sensorLines = document.getElementById('sensors').value
                    .split('\\n')
                    .map(line => line.trim())
                    .filter(Boolean);
                let sequenceOptions = {};
                const optionsText = document.getElementById('sequenceOptions').value.trim();
                if (optionsText) {
                    sequenceOptions = JSON.parse(optionsText);
                }
                return {
                    run_root: runRootValue(),
                    robot_mode: document.getElementById('robotMode').value,
                    sequence: document.getElementById('sequenceId').value,
                    resolution: document.getElementById('resolution').value,
                    fps: Number(document.getElementById('fps').value),
                    velocity: Number(document.getElementById('velocity').value),
                    object_folder: document.getElementById('objectFolder').value.trim(),
                    calibration_profiles: document.getElementById('calibrationProfiles').value.trim() || null,
                    sensors: sensorLines,
                    sequence_options: sequenceOptions,
                    plan_only: document.getElementById('planOnly').checked,
                };
            }

            function showPayload(data) {
                output(JSON.stringify(data, null, 2));
                if (data.job_id) {
                    pollJob(data.job_id);
                    refreshJobs();
                    refreshCaptureJobs();
                }
            }

            function sensorFamilyStatus(family) {
                if (family.error) {
                    return 'ERROR';
                }
                if (family.meets_expected === true) {
                    return 'OK';
                }
                if (family.meets_expected === false) {
                    return 'MISSING';
                }
                return 'UNCHECKED';
            }

            function renderSensorStatus(status) {
                const panel = document.getElementById('sensorStatusPanel');
                panel.innerHTML = '';
                const summary = document.createElement('div');
                summary.className = 'list-group-item';
                summary.textContent = 'Connected: ' + status.total_connected
                    + ' · Expected profile: '
                    + (status.all_expected_connected ? 'complete' : 'incomplete')
                    + ' · Generated: ' + status.generated_at;
                panel.appendChild(summary);

                (status.families || []).forEach(family => {
                    const item = document.createElement('div');
                    item.className = 'list-group-item';
                    const expected = family.expected_count === null || family.expected_count === undefined
                        ? '-'
                        : String(family.expected_count);
                    const devices = (family.devices || [])
                        .map(device => device.device_id)
                        .join(', ') || '-';
                    const title = document.createElement('div');
                    title.className = 'fw-semibold';
                    title.textContent = family.display_name + ' · ' + sensorFamilyStatus(family);
                    const counts = document.createElement('div');
                    counts.className = 'text-muted';
                    counts.textContent = 'Connected ' + family.connected_count
                        + ' / expected ' + expected
                        + ' · SDK ' + family.sdk_module + ': '
                        + (family.sdk_available ? 'available' : 'missing');
                    const deviceLine = document.createElement('div');
                    deviceLine.className = 'text-muted';
                    deviceLine.textContent = 'Devices: ' + devices;
                    item.appendChild(title);
                    item.appendChild(counts);
                    item.appendChild(deviceLine);
                    if (family.error) {
                        const errorLine = document.createElement('div');
                        errorLine.className = 'text-danger';
                        errorLine.textContent = family.error;
                        item.appendChild(errorLine);
                    }
                    panel.appendChild(item);
                });
                if (!panel.children.length) {
                    panel.innerHTML = '<div class="list-group-item text-muted">No sensor status returned</div>';
                }
            }

            function loadSensorStatus() {
                fetch('/sensors/status')
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'Sensor status failed');
                    }
                    renderSensorStatus(result.data);
                    output(JSON.stringify(result.data, null, 2));
                })
                .catch(error => output(error.message));
            }

            function renderSensorAdapters(data) {
                const panel = document.getElementById('sensorStatusPanel');
                panel.innerHTML = '';
                (data.adapters || []).forEach(adapter => {
                    const item = document.createElement('div');
                    item.className = 'list-group-item';
                    const title = document.createElement('div');
                    title.className = 'fw-semibold';
                    title.textContent = adapter.display_name + ' · ' + adapter.sensor_type;
                    const script = document.createElement('div');
                    script.className = 'text-muted';
                    script.textContent = adapter.capture_script
                        + ' · SDK ' + adapter.sdk_module
                        + ' · resolutions ' + (adapter.supported_resolutions || []).join(', ');
                    const modes = document.createElement('div');
                    modes.className = 'text-muted';
                    modes.textContent = 'Mounting: ' + (adapter.mounting_modes || []).join(', ')
                        + ' · folder ' + adapter.folder_prefix + '_<device>';
                    item.appendChild(title);
                    item.appendChild(script);
                    item.appendChild(modes);
                    panel.appendChild(item);
                });
                if (!panel.children.length) {
                    panel.innerHTML = '<div class="list-group-item text-muted">No sensor adapters registered</div>';
                }
            }

            function loadSensorAdapters() {
                fetch('/sensors/adapters')
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'Sensor adapter listing failed');
                    }
                    renderSensorAdapters(result.data);
                    output(JSON.stringify(result.data, null, 2));
                })
                .catch(error => output(error.message));
            }

            function renderRuntimeStatus(status) {
                const panel = document.getElementById('runtimeStatusPanel');
                panel.innerHTML = '';
                const summary = document.createElement('div');
                summary.className = 'list-group-item';
                summary.textContent = 'Available: ' + status.available_count
                    + ' / ' + status.runtime_count
                    + ' · Generated: ' + status.generated_at;
                panel.appendChild(summary);

                (status.runtimes || []).forEach(runtime => {
                    const item = document.createElement('div');
                    item.className = 'list-group-item';
                    const title = document.createElement('div');
                    title.className = 'fw-semibold';
                    title.textContent = runtime.display_name + ' · '
                        + (runtime.available ? 'OK' : 'MISSING');
                    const requiredFor = document.createElement('div');
                    requiredFor.className = 'text-muted';
                    requiredFor.textContent = runtime.required_for;
                    item.appendChild(title);
                    item.appendChild(requiredFor);
                    (runtime.checks || []).forEach(check => {
                        const line = document.createElement('div');
                        line.className = check.ok ? 'text-muted' : 'text-danger';
                        const value = check.value === null || check.value === undefined
                            ? ''
                            : ' · ' + check.value;
                        line.textContent = (check.ok ? 'OK ' : 'Missing ')
                            + check.name + value;
                        if (!check.ok && check.hint) {
                            line.textContent += ' · ' + check.hint;
                        }
                        item.appendChild(line);
                    });
                    if (runtime.hint) {
                        const hint = document.createElement('div');
                        hint.className = 'text-muted';
                        hint.textContent = runtime.hint;
                        item.appendChild(hint);
                    }
                    panel.appendChild(item);
                });
                if (!panel.children.length) {
                    panel.innerHTML = '<div class="list-group-item text-muted">No runtime status returned</div>';
                }
            }

            function loadRuntimeStatus() {
                fetch('/runtime/status')
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'Runtime status failed');
                    }
                    renderRuntimeStatus(result.data);
                    output(JSON.stringify(result.data, null, 2));
                })
                .catch(error => output(error.message));
            }

            function renderHardwareStatus(report) {
                const panel = document.getElementById('runtimeStatusPanel');
                panel.innerHTML = '';
                const summary = document.createElement('div');
                summary.className = 'list-group-item';
                const sensors = report.sensor_status || {};
                const runtimes = report.runtime_status || {};
                const robot = (report.robot_status || {}).selected_profile || {};
                summary.textContent = 'Hardware snapshot: ' + report.overall_status
                    + ' · Robot: ' + (robot.mode || 'unknown')
                    + ' · Sensors: ' + (sensors.total_connected ?? '-')
                    + ' · Runtimes: ' + (runtimes.available_count ?? '-')
                    + ' / ' + (runtimes.runtime_count ?? '-');
                panel.appendChild(summary);
                (report.checks || []).forEach(check => {
                    const item = document.createElement('div');
                    item.className = 'list-group-item';
                    const title = document.createElement('div');
                    title.className = check.status === 'error'
                        ? 'fw-semibold text-danger'
                        : (check.status === 'warning' ? 'fw-semibold text-warning' : 'fw-semibold');
                    title.textContent = check.status.toUpperCase() + ' · ' + check.name;
                    const message = document.createElement('div');
                    message.className = 'text-muted';
                    message.textContent = check.message;
                    item.appendChild(title);
                    item.appendChild(message);
                    panel.appendChild(item);
                });
            }

            function writeHardwareStatus() {
                output('Writing hardware status snapshot...');
                fetch('/hardware/status', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({run_root: runRootValue()}),
                })
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'Hardware status snapshot failed');
                    }
                    document.getElementById('artifactPath').value = 'hardware_status_report.json';
                    renderHardwareStatus(result.data.report);
                    showPayload(result.data);
                })
                .catch(error => output(error.message));
            }

            function loadHardwareStatus() {
                fetch('/hardware/status?run_root=' + encodeURIComponent(runRootValue()))
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'Hardware status load failed');
                    }
                    renderHardwareStatus(result.data.report);
                    output(JSON.stringify(result.data, null, 2));
                })
                .catch(error => output(error.message));
            }

            function renderRobotStatus(status) {
                const panel = document.getElementById('robotStatusPanel');
                panel.innerHTML = '';
                const selected = status.selected_profile || {};
                const summary = document.createElement('div');
                summary.className = 'list-group-item';
                summary.textContent = 'Selected: ' + selected.mode
                    + ' · Command: ' + selected.robot_ip + ':' + selected.command_port
                    + ' · Receiver: ' + selected.receiver_ip + ':' + selected.receiver_port;
                panel.appendChild(summary);

                Object.entries(status.profiles || {}).forEach(([mode, profile]) => {
                    const item = document.createElement('div');
                    item.className = 'list-group-item';
                    const title = document.createElement('div');
                    title.className = 'fw-semibold';
                    title.textContent = mode + ' profile';
                    const command = document.createElement('div');
                    command.className = 'text-muted';
                    command.textContent = 'Command target: ' + profile.robot_ip + ':' + profile.command_port;
                    const receiver = document.createElement('div');
                    receiver.className = 'text-muted';
                    receiver.textContent = 'Receiver bind: ' + profile.receiver_ip + ':' + profile.receiver_port;
                    item.appendChild(title);
                    item.appendChild(command);
                    item.appendChild(receiver);
                    panel.appendChild(item);
                });

                const lab = status.real_robot || {};
                const labItem = document.createElement('div');
                labItem.className = 'list-group-item text-muted';
                labItem.textContent = 'Lab robot: ' + lab.robot_ip
                    + ' · Receiver: ' + lab.receiver_ip
                    + ' · Normal network: ' + lab.normal_network_ip
                    + ' · Default protocol: ' + status.default_command_protocol;
                panel.appendChild(labItem);

                const overrides = Object.entries(status.env_overrides || {});
                if (overrides.length) {
                    const overrideItem = document.createElement('div');
                    overrideItem.className = 'list-group-item text-muted';
                    overrideItem.textContent = 'Env overrides: '
                        + overrides.map(([key, value]) => key + '=' + value).join(', ');
                    panel.appendChild(overrideItem);
                }
            }

            function loadRobotStatus() {
                fetch('/robot/status')
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'Robot status failed');
                    }
                    renderRobotStatus(result.data);
                    output(JSON.stringify(result.data, null, 2));
                })
                .catch(error => output(error.message));
            }

            function runCommand(command) {
                output('Queueing ' + command + '...');

                fetch('/run-command', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ command: command }),
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    output(data.output);
                    if (data.job_id) {
                        pollJob(data.job_id);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    output('Error executing command: ' + command);
                });
            }

            function saveRunConfig() {
                output('Saving run config...');
                fetch('/run-config', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(runConfigPayload()),
                })
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'Run config save failed');
                    }
                    renderRunConfigPreflight(result.data.preflight);
                    showPayload(result.data);
                })
                .catch(error => output(error.message));
            }

            function loadRunConfig() {
                output('Loading run config...');
                fetch('/run-config?run_root=' + encodeURIComponent(runRootValue()))
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'Run config load failed');
                    }
                    const config = result.data.config;
                    document.getElementById('robotMode').value = config.robot_profile.mode;
                    document.getElementById('sequenceId').value = config.pipeline.sequence_id;
                    document.getElementById('resolution').value = config.capture.resolution;
                    document.getElementById('fps').value = config.capture.fps;
                    document.getElementById('velocity').value = config.capture.velocity_m_s;
                    document.getElementById('objectFolder').value = config.object_folder || 'object_models';
                    document.getElementById('calibrationProfiles').value = config.calibration_profiles || '';
                    document.getElementById('planOnly').checked = Boolean(config.pipeline.plan_only);
                    document.getElementById('sensors').value = config.capture.sensors
                        .map(sensor => [sensor.sensor_type, sensor.device_id, sensor.mounting_mode, sensor.display_name].join(':'))
                        .join('\\n');
                    document.getElementById('sequenceOptions').value = JSON.stringify(config.pipeline.options || {}, null, 2);
                    renderRunConfigPreflight(result.data.preflight);
                    showPayload(result.data);
                })
                .catch(error => output(error.message));
            }

            function queueRunConfig() {
                output('Queueing run config...');
                fetch('/pipeline/run-config', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(runConfigQueuePayload()),
                })
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    renderRunConfigPreflight(result.data.preflight);
                    if (!result.ok) {
                        showPayload(result.data);
                        throw new Error(result.data.output || 'Run config queue failed');
                    }
                    showPayload(result.data);
                })
                .catch(error => output(error.message));
            }

            function runConfigQueuePayload() {
                return {
                    run_root: runRootValue(),
                    allow_missing_preflight: document.getElementById('allowMissingPreflight').checked,
                    allow_failed_preflight: document.getElementById('allowFailedPreflight').checked,
                    allow_stale_preflight: document.getElementById('allowStalePreflight').checked,
                };
            }

            function renderCapturePlan(plan) {
                const panel = document.getElementById('capturePlanPanel');
                panel.innerHTML = '';
                const commands = (plan.commands || [])
                    .slice()
                    .sort((left, right) => left.startup_order - right.startup_order);
                const summary = document.createElement('div');
                summary.className = 'list-group-item';
                summary.textContent = 'Commands: ' + commands.length
                    + ' · Sensors: ' + ((plan.sensors || []).length)
                    + ' · Robot: ' + ((plan.robot_profile || {}).mode || 'unknown');
                panel.appendChild(summary);
                commands.forEach(command => {
                    const item = document.createElement('div');
                    item.className = 'list-group-item';
                    const title = document.createElement('div');
                    title.className = 'fw-semibold';
                    title.textContent = '[' + command.startup_order + '] '
                        + command.name + ' · ' + command.role;
                    const line = document.createElement('pre');
                    line.className = 'mb-0 small text-muted';
                    line.textContent = command.command_text || (command.command || []).join(' ');
                    item.appendChild(title);
                    item.appendChild(line);
                    panel.appendChild(item);
                });
                if (!commands.length) {
                    panel.innerHTML = '<div class="list-group-item text-muted">No capture commands in this plan.</div>';
                }
            }

            function writeCapturePlan() {
                output('Writing capture plan...');
                fetch('/capture-plan', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({run_root: runRootValue()}),
                })
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'Capture plan write failed');
                    }
                    document.getElementById('artifactPath').value = 'capture_plan.json';
                    renderCapturePlan(result.data.capture_plan);
                    showPayload(result.data);
                })
                .catch(error => output(error.message));
            }

            function loadCapturePlan() {
                output('Loading capture plan...');
                fetch('/capture-plan?run_root=' + encodeURIComponent(runRootValue()))
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'Capture plan load failed');
                    }
                    document.getElementById('artifactPath').value = 'capture_plan.json';
                    renderCapturePlan(result.data.capture_plan);
                    output(JSON.stringify(result.data, null, 2));
                })
                .catch(error => output(error.message));
            }

            function renderCapturePlanPreflight(report) {
                const panel = document.getElementById('capturePlanPanel');
                panel.innerHTML = '';
                const summary = document.createElement('div');
                summary.className = 'list-group-item';
                summary.textContent = 'Preflight: ' + report.overall_status
                    + ' · Checks: ' + ((report.checks || []).length);
                panel.appendChild(summary);
                (report.checks || []).forEach(check => {
                    const item = document.createElement('div');
                    item.className = 'list-group-item';
                    const title = document.createElement('div');
                    title.className = check.status === 'error'
                        ? 'fw-semibold text-danger'
                        : (check.status === 'warning' ? 'fw-semibold text-warning' : 'fw-semibold');
                    title.textContent = check.status.toUpperCase() + ' · ' + check.name;
                    const message = document.createElement('div');
                    message.className = 'text-muted';
                    message.textContent = check.message;
                    item.appendChild(title);
                    item.appendChild(message);
                    panel.appendChild(item);
                });
            }

            function preflightCapturePlan() {
                output('Preflighting capture plan...');
                fetch('/capture-plan/preflight', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({run_root: runRootValue()}),
                })
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'Capture plan preflight failed');
                    }
                    renderCapturePlanPreflight(result.data.report);
                    showPayload(result.data);
                })
                .catch(error => output(error.message));
            }

            function renderCaptureExecutionPlan(plan) {
                const panel = document.getElementById('capturePlanPanel');
                panel.innerHTML = '';
                const summary = document.createElement('div');
                summary.className = 'list-group-item';
                summary.textContent = 'Execution plan: ' + plan.status
                    + ' · Mode: ' + plan.mode
                    + ' · Selected: ' + ((plan.selected_commands || []).length)
                    + ' · Skipped: ' + ((plan.skipped_commands || []).length);
                panel.appendChild(summary);
                (plan.gates || []).forEach(gate => {
                    const item = document.createElement('div');
                    item.className = 'list-group-item';
                    const title = document.createElement('div');
                    title.className = gate.status === 'error'
                        ? 'fw-semibold text-danger'
                        : (gate.status === 'warning' ? 'fw-semibold text-warning' : 'fw-semibold');
                    title.textContent = gate.status.toUpperCase() + ' · ' + gate.name;
                    const message = document.createElement('div');
                    message.className = 'text-muted';
                    message.textContent = gate.message;
                    item.appendChild(title);
                    item.appendChild(message);
                    panel.appendChild(item);
                });
                (plan.selected_commands || [])
                    .slice()
                    .sort((left, right) => left.startup_order - right.startup_order)
                    .forEach(command => {
                        const item = document.createElement('div');
                        item.className = 'list-group-item';
                        const title = document.createElement('div');
                        title.className = 'fw-semibold';
                        title.textContent = '[' + command.startup_order + '] '
                            + command.name + ' · selected · ' + command.role;
                        const line = document.createElement('pre');
                        line.className = 'mb-0 small text-muted';
                        line.textContent = command.command_text || (command.command || []).join(' ');
                        item.appendChild(title);
                        item.appendChild(line);
                        panel.appendChild(item);
                    });
            }

            function captureExecutionOptions() {
                return {
                    mode: document.getElementById('captureExecutionMode').value,
                    allow_cameras: document.getElementById('allowCaptureCameras').checked,
                    allow_real_robot: document.getElementById('allowCaptureRealRobot').checked,
                    include_sensors: document.getElementById('includeCaptureSensors').checked,
                };
            }

            function writeCaptureExecutionPlan() {
                const options = captureExecutionOptions();
                output('Writing capture execution plan for ' + options.mode + '...');
                fetch('/capture-plan/execution', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(Object.assign({run_root: runRootValue()}, options)),
                })
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'Capture execution plan write failed');
                    }
                    document.getElementById('artifactPath').value = 'capture_execution_plan.json';
                    renderCaptureExecutionPlan(result.data.plan);
                    showPayload(result.data);
                })
                .catch(error => output(error.message));
            }

            function loadCaptureExecutionPlan() {
                output('Loading capture execution plan...');
                fetch('/capture-plan/execution?run_root=' + encodeURIComponent(runRootValue()))
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'Capture execution plan load failed');
                    }
                    document.getElementById('artifactPath').value = 'capture_execution_plan.json';
                    renderCaptureExecutionPlan(result.data.plan);
                    output(JSON.stringify(result.data, null, 2));
                })
                .catch(error => output(error.message));
            }

            function queueCaptureExecution() {
                const options = captureExecutionOptions();
                if (options.mode === 'plan_only') {
                    output('Plan-only capture execution does not start commands. Use Plan Execution.');
                    return;
                }
                output('Queueing supervised capture execution for ' + options.mode + '...');
                fetch('/pipeline/run', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        stage: 'capture_execution',
                        run_root: runRootValue(),
                        options: Object.assign({}, options, {
                            timeout_s: 30,
                            startup_wait_s: 0.2,
                            terminate_timeout_s: 2,
                        }),
                    }),
                })
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'Supervised fake execution queue failed');
                    }
                    showPayload(result.data);
                })
                .catch(error => output(error.message));
            }

            function queueCaptureRehearsal() {
                output('Queueing fake pose rehearsal...');
                fetch('/pipeline/run', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        stage: 'capture_rehearsal',
                        run_root: runRootValue(),
                        options: {
                            duration_s: 0.3,
                            sample_ms: 25,
                            startup_delay_s: 0,
                            timeout_s: 10,
                        },
                    }),
                })
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'Fake pose rehearsal queue failed');
                    }
                    showPayload(result.data);
                })
                .catch(error => output(error.message));
            }

            function renderRunConfigPreflight(preflight) {
                if (!preflight) {
                    return;
                }
                const panel = document.getElementById('preflightPanel');
                panel.innerHTML = '';
                const item = document.createElement('div');
                item.className = preflight.ready_for_queue
                    ? 'list-group-item'
                    : 'list-group-item list-group-item-warning';
                const title = document.createElement('div');
                title.className = 'fw-semibold';
                title.textContent = preflight.ready_for_queue
                    ? 'Run preflight ready for queue'
                    : 'Run preflight blocks queue';
                const detail = document.createElement('div');
                detail.className = 'text-muted';
                const status = preflight.overall_status || 'missing';
                const matches = preflight.matches_config === null
                    ? 'unknown'
                    : String(preflight.matches_config);
                detail.textContent = 'Status: ' + status
                    + ' · Matches config: ' + matches
                    + (preflight.queue_blocker ? ' · Blocker: ' + preflight.queue_blocker : '');
                item.appendChild(title);
                item.appendChild(detail);
                panel.appendChild(item);
            }

            function renderPreflight(preflight) {
                const panel = document.getElementById('preflightPanel');
                panel.innerHTML = '';
                const summary = document.createElement('div');
                summary.className = 'list-group-item';
                summary.textContent = 'Status: ' + preflight.overall_status
                    + ' · Sequence: ' + preflight.sequence_plan.sequence_id
                    + ' · Steps: ' + preflight.sequence_plan.steps.length;
                panel.appendChild(summary);
                (preflight.checks || []).forEach(check => {
                    const item = document.createElement('div');
                    item.className = 'list-group-item';
                    const title = document.createElement('div');
                    title.className = check.status === 'error'
                        ? 'fw-semibold text-danger'
                        : (check.status === 'warning' ? 'fw-semibold text-warning' : 'fw-semibold');
                    title.textContent = check.status.toUpperCase() + ' · ' + check.name;
                    const message = document.createElement('div');
                    message.className = 'text-muted';
                    message.textContent = check.message;
                    item.appendChild(title);
                    item.appendChild(message);
                    panel.appendChild(item);
                });
            }

            function preflightRunConfig() {
                output('Preflighting run config...');
                fetch('/pipeline/preflight?run_root=' + encodeURIComponent(runRootValue()))
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'Run preflight failed');
                    }
                    renderPreflight(result.data);
                    output(JSON.stringify(result.data, null, 2));
                })
                .catch(error => output(error.message));
            }

            function writeRunPreflight() {
                output('Writing run preflight report...');
                fetch('/pipeline/preflight', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        run_root: runRootValue(),
                    }),
                })
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'Run preflight write failed');
                    }
                    renderPreflight(result.data.report);
                    document.getElementById('artifactPath').value = 'run_preflight_report.json';
                    output(JSON.stringify(result.data, null, 2));
                    listArtifacts();
                })
                .catch(error => output(error.message));
            }

            function renderCalibrationPreflight(report) {
                const panel = document.getElementById('preflightPanel');
                panel.innerHTML = '';
                const summary = document.createElement('div');
                summary.className = 'list-group-item';
                summary.textContent = 'Calibration: ' + report.overall_status
                    + ' · Matched: ' + report.matched_sensor_count
                    + ' / ' + report.sensor_count
                    + ' · Profiles: ' + report.profile_count;
                panel.appendChild(summary);
                (report.checks || []).forEach(check => {
                    const item = document.createElement('div');
                    item.className = 'list-group-item';
                    const title = document.createElement('div');
                    title.className = check.status === 'error'
                        ? 'fw-semibold text-danger'
                        : (check.status === 'warning' ? 'fw-semibold text-warning' : 'fw-semibold');
                    title.textContent = check.status.toUpperCase() + ' · ' + check.name;
                    const message = document.createElement('div');
                    message.className = 'text-muted';
                    message.textContent = check.message;
                    item.appendChild(title);
                    item.appendChild(message);
                    panel.appendChild(item);
                });
            }

            function preflightCalibration() {
                output('Preflighting calibration profiles...');
                fetch('/calibration/preflight', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({run_root: runRootValue()}),
                })
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'Calibration preflight failed');
                    }
                    renderCalibrationPreflight(result.data.report);
                    showPayload(result.data);
                })
                .catch(error => output(error.message));
            }

            function renderCalibrationObservations(report) {
                const panel = document.getElementById('preflightPanel');
                panel.innerHTML = '';
                const summary = document.createElement('div');
                summary.className = 'list-group-item';
                summary.textContent = 'Calibration observations: ' + report.overall_status
                    + ' · Usable: ' + report.observation_count
                    + ' / ' + report.frame_count
                    + ' · Rejected: ' + report.rejected_count
                    + ' · Sensors: ' + report.sensor_count;
                panel.appendChild(summary);
                (report.sensors || []).forEach(sensor => {
                    const item = document.createElement('div');
                    item.className = 'list-group-item';
                    const title = document.createElement('div');
                    title.className = 'fw-semibold';
                    title.textContent = sensor.sensor_name + ' · '
                        + sensor.observation_count + ' observations · '
                        + (sensor.mounting_mode || 'mounting unknown');
                    const line = document.createElement('div');
                    line.className = 'text-muted';
                    line.textContent = 'Frames: ' + sensor.frame_count
                        + ' · Rejected: ' + sensor.rejected_count
                        + ' · Motions: ' + (sensor.motions || []).join(', ');
                    item.appendChild(title);
                    item.appendChild(line);
                    panel.appendChild(item);
                });
                (report.checks || []).forEach(check => {
                    if (check.status === 'ok') {
                        return;
                    }
                    const item = document.createElement('div');
                    item.className = 'list-group-item';
                    const title = document.createElement('div');
                    title.className = check.status === 'error'
                        ? 'fw-semibold text-danger'
                        : 'fw-semibold text-warning';
                    title.textContent = check.status.toUpperCase() + ' · ' + check.name;
                    const message = document.createElement('div');
                    message.className = 'text-muted';
                    message.textContent = check.message;
                    item.appendChild(title);
                    item.appendChild(message);
                    panel.appendChild(item);
                });
            }

            function buildCalibrationObservations() {
                output('Building calibration observations...');
                fetch('/calibration/observations', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({run_root: runRootValue()}),
                })
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok && !result.data.report) {
                        throw new Error(result.data.output || 'Calibration observation build failed');
                    }
                    document.getElementById('artifactPath').value = 'calibration_observations.json';
                    renderCalibrationObservations(result.data.report);
                    showPayload(result.data);
                })
                .catch(error => output(error.message));
            }

            function renderCalibrationSolver(report) {
                const panel = document.getElementById('preflightPanel');
                panel.innerHTML = '';
                const summary = document.createElement('div');
                summary.className = 'list-group-item';
                summary.textContent = 'Calibration solver: ' + report.overall_status
                    + ' · Profiles: ' + report.profile_count
                    + ' · Inliers: ' + (report.inlier_count ?? 0)
                    + ' / ' + (report.observation_count ?? 0)
                    + ' · Outliers: ' + (report.outlier_count ?? 0)
                    + ' · Holdout: ' + Number(report.holdout_fraction || 0).toFixed(2)
                    + ' · Method comparisons: ' + ((report.method_comparisons || []).length)
                    + ' · Sensors: ' + report.sensor_count;
                panel.appendChild(summary);
                (report.solutions || []).forEach(solution => {
                    const item = document.createElement('div');
                    item.className = 'list-group-item';
                    const title = document.createElement('div');
                    title.className = 'fw-semibold';
                    title.textContent = solution.sensor_name + ' · '
                        + solution.method + ' · '
                        + (solution.inlier_count ?? 0) + '/'
                        + solution.observation_count + ' inliers';
                    const residuals = solution.residuals || {};
                    const line = document.createElement('div');
                    line.className = 'text-muted';
                    line.textContent = 'Mean residual: '
                        + Number(residuals.mean_translation_mm || 0).toFixed(3)
                        + ' mm, '
                        + Number(residuals.mean_rotation_deg || 0).toFixed(3)
                        + ' deg · Transform: camera -> ' + solution.to;
                    item.appendChild(title);
                    item.appendChild(line);
                    if (solution.holdout_residuals) {
                        const holdout = document.createElement('div');
                        holdout.className = 'text-muted';
                        holdout.textContent = 'Holdout: '
                            + solution.holdout_observation_count
                            + ' frames · '
                            + Number(solution.holdout_residuals.mean_translation_mm || 0).toFixed(3)
                            + ' mm, '
                            + Number(solution.holdout_residuals.mean_rotation_deg || 0).toFixed(3)
                            + ' deg · ' + solution.holdout_status;
                        item.appendChild(holdout);
                    }
                    if ((solution.method_comparisons || []).length) {
                        const comparisons = document.createElement('div');
                        comparisons.className = 'text-muted';
                        const statuses = [...new Set(solution.method_comparisons.map(
                            row => row.status || 'unknown'
                        ))].sort().join(', ');
                        comparisons.textContent = 'Method comparison: '
                            + solution.method_comparisons.length
                            + ' method(s) · ' + statuses;
                        item.appendChild(comparisons);
                    }
                    panel.appendChild(item);
                });
                (report.checks || []).forEach(check => {
                    if (check.status === 'ok') {
                        return;
                    }
                    const item = document.createElement('div');
                    item.className = 'list-group-item';
                    const title = document.createElement('div');
                    title.className = check.status === 'error'
                        ? 'fw-semibold text-danger'
                        : 'fw-semibold text-warning';
                    title.textContent = check.status.toUpperCase() + ' · ' + check.name;
                    const message = document.createElement('div');
                    message.className = 'text-muted';
                    message.textContent = check.message;
                    item.appendChild(title);
                    item.appendChild(message);
                    panel.appendChild(item);
                });
            }

            function solveCalibration() {
                output('Solving calibration profiles...');
                fetch('/calibration/solver', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({run_root: runRootValue()}),
                })
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok && !result.data.report) {
                        throw new Error(result.data.output || 'Calibration solve failed');
                    }
                    document.getElementById('artifactPath').value = 'calibration_solver_report.json';
                    renderCalibrationSolver(result.data.report);
                    showPayload(result.data);
                })
                .catch(error => output(error.message));
            }

            function renderCalibrationCandidates(report) {
                const panel = document.getElementById('preflightPanel');
                panel.innerHTML = '';
                const summary = document.createElement('div');
                summary.className = 'list-group-item';
                summary.textContent = 'Calibration candidates: ' + report.overall_status
                    + ' · Profiles: ' + report.profile_count
                    + ' · Inliers: ' + (report.inlier_count ?? 0)
                    + ' · Outliers: ' + (report.outlier_count ?? 0)
                    + ' · Frame candidates: ' + report.candidate_count
                    + ' · Sensors: ' + report.sensor_count;
                panel.appendChild(summary);
                (report.residuals || []).forEach(sensor => {
                    const item = document.createElement('div');
                    item.className = 'list-group-item';
                    const title = document.createElement('div');
                    title.className = 'fw-semibold';
                    title.textContent = sensor.sensor_name + ' · '
                        + (sensor.inlier_count ?? 0) + '/'
                        + sensor.observation_count + ' inliers · '
                        + (sensor.mounting_mode || 'mounting unknown');
                    const line = document.createElement('div');
                    line.className = 'text-muted';
                    line.textContent = 'Mean residual: '
                        + Number(sensor.mean_translation_mm || 0).toFixed(3) + ' mm, '
                        + Number(sensor.mean_rotation_deg || 0).toFixed(3) + ' deg · Max: '
                        + Number(sensor.max_translation_mm || 0).toFixed(3) + ' mm, '
                        + Number(sensor.max_rotation_deg || 0).toFixed(3) + ' deg';
                    item.appendChild(title);
                    item.appendChild(line);
                    panel.appendChild(item);
                });
                (report.checks || []).forEach(check => {
                    if (check.status === 'ok') {
                        return;
                    }
                    const item = document.createElement('div');
                    item.className = 'list-group-item';
                    const title = document.createElement('div');
                    title.className = check.status === 'error'
                        ? 'fw-semibold text-danger'
                        : 'fw-semibold text-warning';
                    title.textContent = check.status.toUpperCase() + ' · ' + check.name;
                    const message = document.createElement('div');
                    message.className = 'text-muted';
                    message.textContent = check.message;
                    item.appendChild(title);
                    item.appendChild(message);
                    panel.appendChild(item);
                });
            }

            function buildCalibrationCandidates() {
                output('Building calibration candidates...');
                fetch('/calibration/candidates', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({run_root: runRootValue()}),
                })
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok && !result.data.report) {
                        throw new Error(result.data.output || 'Calibration candidate build failed');
                    }
                    document.getElementById('artifactPath').value = 'calibration_candidates.json';
                    renderCalibrationCandidates(result.data.report);
                    showPayload(result.data);
                })
                .catch(error => output(error.message));
            }

            function renderCalibrationValidation(report) {
                const panel = document.getElementById('preflightPanel');
                panel.innerHTML = '';
                const summary = document.createElement('div');
                summary.className = 'list-group-item';
                summary.textContent = 'Calibration validation: ' + report.overall_status
                    + ' · Promotable: ' + report.promotable_profile_count
                    + ' / ' + report.profile_count
                    + ' · Inliers: ' + report.inlier_count
                    + ' · Outliers: ' + report.outlier_count;
                panel.appendChild(summary);
                (report.profiles || []).forEach(profile => {
                    const item = document.createElement('div');
                    item.className = 'list-group-item';
                    const title = document.createElement('div');
                    title.className = profile.validation_status === 'ok'
                        ? 'fw-semibold'
                        : 'fw-semibold text-danger';
                    title.textContent = profile.profile_id + ' · '
                        + profile.validation_status + ' · '
                        + profile.num_inliers + '/' + profile.num_observations
                        + ' inliers';
                    const line = document.createElement('div');
                    line.className = 'text-muted';
                    line.textContent = 'Mean residual: '
                        + Number(profile.residual_translation_mm || 0).toFixed(3)
                        + ' mm, '
                        + Number(profile.residual_rotation_deg || 0).toFixed(3)
                        + ' deg · Outlier ratio: '
                        + Number(profile.outlier_ratio || 0).toFixed(3);
                    item.appendChild(title);
                    item.appendChild(line);
                    panel.appendChild(item);
                });
                (report.checks || []).forEach(check => {
                    if (check.status === 'ok') {
                        return;
                    }
                    const item = document.createElement('div');
                    item.className = 'list-group-item';
                    const title = document.createElement('div');
                    title.className = check.status === 'error'
                        ? 'fw-semibold text-danger'
                        : 'fw-semibold text-warning';
                    title.textContent = check.status.toUpperCase() + ' · ' + check.name;
                    const message = document.createElement('div');
                    message.className = 'text-muted';
                    message.textContent = check.message;
                    item.appendChild(title);
                    item.appendChild(message);
                    panel.appendChild(item);
                });
            }

            function validateCalibrationCandidates() {
                output('Validating calibration candidates...');
                fetch('/calibration/validation', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({run_root: runRootValue()}),
                })
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok && !result.data.report) {
                        throw new Error(result.data.output || 'Calibration validation failed');
                    }
                    document.getElementById('artifactPath').value = 'calibration_validation_report.json';
                    renderCalibrationValidation(result.data.report);
                    showPayload(result.data);
                })
                .catch(error => output(error.message));
            }

            function renderSyncQuality(report) {
                const panel = document.getElementById('preflightPanel');
                panel.innerHTML = '';
                const summary = document.createElement('div');
                summary.className = 'list-group-item';
                summary.textContent = 'Sync quality: ' + report.overall_status
                    + ' · Matched: ' + report.matched_frames
                    + ' / ' + report.total_frames
                    + ' · Dropped: ' + report.dropped_frames
                    + ' · Sensors: ' + report.sensor_count;
                panel.appendChild(summary);
                (report.sensors || []).forEach(sensor => {
                    const item = document.createElement('div');
                    item.className = 'list-group-item';
                    const title = document.createElement('div');
                    title.className = 'fw-semibold';
                    title.textContent = sensor.sensor_name + ' · '
                        + Number(sensor.match_ratio || 0).toFixed(3)
                        + ' match ratio · ' + sensor.timestamp_source;
                    const line = document.createElement('div');
                    line.className = 'text-muted';
                    line.textContent = sensor.matched_frames + '/'
                        + sensor.total_frames + ' frames · dropped '
                        + sensor.dropped_frames + ' · max pose delta ns '
                        + (sensor.max_abs_nearest_pose_delta_ns ?? '-');
                    item.appendChild(title);
                    item.appendChild(line);
                    panel.appendChild(item);
                });
                (report.checks || []).forEach(check => {
                    if (check.status === 'ok') {
                        return;
                    }
                    const item = document.createElement('div');
                    item.className = 'list-group-item';
                    const title = document.createElement('div');
                    title.className = check.status === 'error'
                        ? 'fw-semibold text-danger'
                        : 'fw-semibold text-warning';
                    title.textContent = check.status.toUpperCase() + ' · ' + check.name;
                    const message = document.createElement('div');
                    message.className = 'text-muted';
                    message.textContent = check.message;
                    item.appendChild(title);
                    item.appendChild(message);
                    panel.appendChild(item);
                });
            }

            function checkSyncQuality() {
                output('Checking sync quality...');
                fetch('/sync/quality', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({run_root: runRootValue()}),
                })
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok && !result.data.report) {
                        throw new Error(result.data.output || 'Sync quality failed');
                    }
                    document.getElementById('artifactPath').value = 'sync_quality_report.json';
                    renderSyncQuality(result.data.report);
                    showPayload(result.data);
                })
                .catch(error => output(error.message));
            }

            function renderRecommendations(data) {
                const panel = document.getElementById('recommendationsPanel');
                panel.innerHTML = '';
                (data.recommendations || []).slice(0, 10).forEach(recommendation => {
                    const item = document.createElement('button');
                    item.type = 'button';
                    item.className = 'list-group-item list-group-item-action text-start';
                    const title = document.createElement('div');
                    title.className = 'fw-semibold';
                    title.textContent = recommendation.label + ' · ' + recommendation.action_type;
                    const reason = document.createElement('div');
                    reason.className = 'text-muted';
                    reason.textContent = recommendation.reason;
                    const command = document.createElement('code');
                    command.className = 'd-block small text-muted mt-1';
                    command.textContent = (recommendation.command || []).join(' ');
                    const blockers = recommendation.blocks_on || [];
                    const blockerLine = document.createElement('div');
                    blockerLine.className = 'small text-muted mt-1';
                    blockerLine.textContent = blockers.length ? 'Blocks: ' + blockers.join(', ') : '';
                    item.appendChild(title);
                    item.appendChild(reason);
                    if (recommendation.command && recommendation.command.length) {
                        item.appendChild(command);
                    }
                    if (blockers.length) {
                        item.appendChild(blockerLine);
                    }
                    item.onclick = () => output(JSON.stringify(recommendation, null, 2));
                    panel.appendChild(item);
                });
                if (!panel.children.length) {
                    panel.innerHTML = '<div class="list-group-item text-muted">No recommendations for the current artifact state.</div>';
                }
            }

            function loadRecommendations() {
                fetch('/pipeline/recommendations?run_root=' + encodeURIComponent(runRootValue()))
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'Recommendation lookup failed');
                    }
                    renderRecommendations(result.data);
                    output(JSON.stringify(result.data, null, 2));
                })
                .catch(error => output(error.message));
            }

            function jobSummary(job) {
                const resources = (job.resources || []).join(', ');
                const tail = (job.tail || []).slice(-3).join('\\n');
                return [
                    job.name + ' (' + job.id + ')',
                    'Status: ' + job.status,
                    resources ? 'Resources: ' + resources : '',
                    job.message ? 'Message: ' + job.message : '',
                    tail,
                ].filter(Boolean).join('\\n');
            }

            function refreshJobs() {
                fetch('/jobs')
                .then(response => response.json())
                .then(data => {
                    const resourceEntries = Object.entries(data.resources || {});
                    document.getElementById('resourceStatus').textContent = resourceEntries.length
                        ? resourceEntries.map(([resource, job]) => resource + ': ' + job).join(' · ')
                        : 'No held resources';

                    const panel = document.getElementById('jobsPanel');
                    panel.innerHTML = '';
                    (data.jobs || []).forEach(job => {
                        const item = document.createElement('div');
                        item.className = 'list-group-item';
                        const cancelButton = job.status === 'queued' || job.status === 'running'
                            ? '<button class="btn btn-sm btn-outline-danger ms-2" onclick="cancelJob(\\'' + escapeHtml(job.id) + '\\')">Cancel</button>'
                            : '';
                        item.innerHTML = '<div class="d-flex justify-content-between align-items-start gap-3">'
                            + '<pre class="mb-0 flex-grow-1 small">' + escapeHtml(jobSummary(job)) + '</pre>'
                            + '<div class="d-flex gap-2">'
                            + '<button class="btn btn-sm btn-outline-secondary" onclick="showJob(\\'' + escapeHtml(job.id) + '\\')">Open</button>'
                            + cancelButton
                            + '</div></div>';
                        panel.appendChild(item);
                    });
                    if (!panel.children.length) {
                        panel.innerHTML = '<div class="list-group-item text-muted">No jobs</div>';
                    }
                })
                .catch(error => output(error.message));
            }

            function showJob(jobId) {
                fetch('/jobs/' + jobId)
                .then(response => response.json())
                .then(data => output(JSON.stringify(data, null, 2)))
                .catch(error => output(error.message));
            }

            function cancelJob(jobId) {
                fetch('/jobs/' + jobId + '/cancel', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    output(JSON.stringify(data, null, 2));
                    refreshJobs();
                    refreshCaptureJobs();
                })
                .catch(error => output(error.message));
            }

            function captureJobSummary(job) {
                const bits = [
                    job.name + ' (' + job.id + ')',
                    'Status: ' + job.status,
                    job.stage ? 'Stage: ' + job.stage : '',
                    job.sequence ? 'Sequence: ' + job.sequence : '',
                    job.mode ? 'Mode: ' + job.mode : '',
                    job.run_root ? 'Run: ' + job.run_root : '',
                    (job.resources || []).length ? 'Resources: ' + job.resources.join(', ') : '',
                    job.message ? 'Message: ' + job.message : '',
                    (job.tail || []).slice(-3).join('\\n'),
                ];
                return bits.filter(Boolean).join('\\n');
            }

            function renderCaptureJobs(data) {
                const panel = document.getElementById('captureJobsPanel');
                panel.innerHTML = '';
                const summary = document.createElement('div');
                summary.className = 'list-group-item';
                summary.textContent = 'Capture jobs: ' + ((data.jobs || []).length)
                    + ' · Active: ' + data.active_count;
                panel.appendChild(summary);
                const statusArtifact = data.status_artifact || null;
                if (statusArtifact && !statusArtifact.error) {
                    const statusItem = document.createElement('div');
                    statusItem.className = 'list-group-item';
                    statusItem.textContent = 'Latest supervisor status: '
                        + statusArtifact.status
                        + ' · Mode: ' + statusArtifact.mode
                        + ' · Active processes: ' + statusArtifact.active_process_count
                        + ' · Raw poses: ' + statusArtifact.raw_pose_count;
                    panel.appendChild(statusItem);
                } else if (statusArtifact && statusArtifact.error) {
                    const statusItem = document.createElement('div');
                    statusItem.className = 'list-group-item text-danger';
                    statusItem.textContent = 'Capture status artifact error: '
                        + statusArtifact.error;
                    panel.appendChild(statusItem);
                }

                (data.jobs || []).forEach(job => {
                    const item = document.createElement('div');
                    item.className = 'list-group-item';
                    const stopButton = job.stop_endpoint
                        ? '<button class="btn btn-sm btn-outline-danger" onclick="stopCaptureJob(\\'' + escapeHtml(job.id) + '\\')">Stop</button>'
                        : '';
                    item.innerHTML = '<div class="d-flex justify-content-between align-items-start gap-3">'
                        + '<pre class="mb-0 flex-grow-1 small">' + escapeHtml(captureJobSummary(job)) + '</pre>'
                        + '<div class="d-flex gap-2">'
                        + '<button class="btn btn-sm btn-outline-secondary" onclick="showJob(\\'' + escapeHtml(job.id) + '\\')">Open</button>'
                        + stopButton
                        + '</div></div>';
                    panel.appendChild(item);
                });
                if ((data.jobs || []).length === 0) {
                    const emptyItem = document.createElement('div');
                    emptyItem.className = 'list-group-item text-muted';
                    emptyItem.textContent = 'No capture jobs for this run';
                    panel.appendChild(emptyItem);
                }
                if (data.active_count > 0) {
                    setTimeout(refreshCaptureJobs, 1000);
                }
            }

            function refreshCaptureJobs() {
                fetch('/capture/jobs?run_root=' + encodeURIComponent(runRootValue()))
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'Capture job refresh failed');
                    }
                    renderCaptureJobs(result.data);
                })
                .catch(error => output(error.message));
            }

            function loadCaptureStatus() {
                fetch('/capture/status?run_root=' + encodeURIComponent(runRootValue()))
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'Capture status load failed');
                    }
                    output(JSON.stringify(result.data, null, 2));
                    renderCaptureJobs({
                        jobs: [],
                        active_count: 0,
                        status_artifact: result.data.status,
                    });
                })
                .catch(error => output(error.message));
            }

            function stopCaptureJob(jobId) {
                fetch('/capture/jobs/' + jobId + '/stop', {method: 'POST'})
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'Capture job stop failed');
                    }
                    output(JSON.stringify(result.data, null, 2));
                    refreshCaptureJobs();
                    refreshJobs();
                })
                .catch(error => output(error.message));
            }

            function artifactSummary(artifact) {
                if (artifact.display_label) {
                    return artifact.display_label;
                }
                const summary = artifact.summary || {};
                const bits = [
                    artifact.key,
                    artifact.source,
                    artifact.kind,
                    artifact.exists ? 'exists' : 'missing',
                    summary.type || '',
                ].filter(Boolean);
                return bits.join(' · ');
            }

            function appendArtifactNextActions(item, artifact) {
                const summary = artifact.summary || {};
                const labels = summary.next_action_labels || [];
                const commands = summary.next_action_commands || [];
                const blocksOn = summary.next_action_blocks_on || [];
                if (!labels.length || labels.length < 2) {
                    return;
                }
                const actions = document.createElement('div');
                actions.className = 'text-muted mt-1';
                labels.slice(0, 5).forEach((label, index) => {
                    const line = document.createElement('div');
                    const command = commands[index] || [];
                    const blockers = blocksOn[index] || [];
                    const commandText = command.length ? ': ' + command.join(' ') : '';
                    const blockerText = blockers.length ? ' [blocks: ' + blockers.join(', ') + ']' : '';
                    line.textContent = (index + 1) + '. ' + label + commandText + blockerText;
                    actions.appendChild(line);
                });
                item.appendChild(actions);
            }

            function renderArtifacts(artifacts) {
                const panel = document.getElementById('artifactsPanel');
                panel.innerHTML = '';
                (artifacts || []).slice(0, 80).forEach(artifact => {
                    const item = document.createElement('button');
                    item.type = 'button';
                    item.className = 'list-group-item list-group-item-action text-start small';
                    const title = document.createElement('div');
                    title.textContent = artifactSummary(artifact);
                    item.appendChild(title);
                    appendArtifactNextActions(item, artifact);
                    item.onclick = () => {
                        if (artifact.relative_path) {
                            document.getElementById('artifactPath').value = artifact.relative_path;
                            previewArtifact();
                        } else {
                            output(JSON.stringify(artifact, null, 2));
                        }
                    };
                    panel.appendChild(item);
                });
                if (!panel.children.length) {
                    panel.innerHTML = '<div class="list-group-item text-muted">No artifacts</div>';
                }
            }

            function listArtifacts() {
                fetch('/artifacts?run_root=' + encodeURIComponent(runRootValue()))
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'Artifact listing failed');
                    }
                    renderArtifacts(result.data.artifacts);
                    output(JSON.stringify(result.data, null, 2));
                })
                .catch(error => output(error.message));
            }

            function previewArtifact() {
                const path = document.getElementById('artifactPath').value.trim();
                fetch('/artifacts/preview?run_root=' + encodeURIComponent(runRootValue()) + '&path=' + encodeURIComponent(path))
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'Artifact preview failed');
                    }
                    output(JSON.stringify(result.data, null, 2));
                })
                .catch(error => output(error.message));
            }

            function loadMetrics() {
                fetch('/artifacts/metrics?run_root=' + encodeURIComponent(runRootValue()))
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'Metric summary failed');
                    }
                    renderMetrics(result.data);
                    output(JSON.stringify(result.data, null, 2));
                })
                .catch(error => output(error.message));
            }

            function clearMetricsPanel() {
                const panel = document.getElementById('metricsPanel');
                panel.className = 'list-group small mt-3';
                panel.innerHTML = '';
                return panel;
            }

            function metricText(value) {
                if (value === undefined || value === null || value === '') {
                    return '';
                }
                const number = Number(value);
                if (Number.isFinite(number)) {
                    return number.toFixed(3);
                }
                return String(value);
            }

            function metricBestText(best) {
                if (!best) {
                    return 'No AP_p result';
                }
                const context = best.context ? ' · ' + best.context : '';
                return best.method + ' AP_p=' + metricText(best.AP_p) + context;
            }

            function renderMetricRow(parent, label, values) {
                const item = document.createElement('div');
                item.className = 'list-group-item';
                const title = document.createElement('div');
                title.className = 'fw-semibold';
                title.textContent = label;
                item.appendChild(title);
                (values || []).filter(Boolean).forEach(value => {
                    const line = document.createElement('div');
                    line.className = 'text-muted';
                    line.textContent = value;
                    item.appendChild(line);
                });
                parent.appendChild(item);
            }

            function renderDirectMetricTable(parent, rows) {
                const table = document.createElement('table');
                table.className = 'table table-sm align-middle mb-0';
                const header = document.createElement('thead');
                header.innerHTML = '<tr><th>Method</th><th>AP_p</th><th>Samples</th><th>Artifact</th></tr>';
                table.appendChild(header);
                const body = document.createElement('tbody');
                (rows || []).slice(0, 12).forEach(row => {
                    const tr = document.createElement('tr');
                    const allMotions = row.all_motions || {};
                    [
                        row.method,
                        metricText(allMotions.AP_p),
                        row.sample_count,
                        row.relative_path,
                    ].forEach(value => {
                        const td = document.createElement('td');
                        td.textContent = value === undefined || value === null ? '' : String(value);
                        tr.appendChild(td);
                    });
                    body.appendChild(tr);
                });
                table.appendChild(body);
                parent.appendChild(table);
                if (!body.children.length) {
                    const empty = document.createElement('div');
                    empty.className = 'text-muted';
                    empty.textContent = 'No direct metric rows found.';
                    parent.appendChild(empty);
                }
            }

            function renderCombinedMetricGroups(parent, groups) {
                const visibleGroups = (groups || []).slice(0, 8);
                visibleGroups.forEach(group => {
                    renderMetricRow(parent, group.context || '(combined result)', [
                        'Methods: ' + (group.methods || []).join(', '),
                        'Best: ' + metricBestText(group.best_by_AP_p),
                        group.relative_path,
                    ]);
                });
                if (!visibleGroups.length) {
                    renderMetricRow(parent, 'Combined groups', ['No combined metric groups found.']);
                }
            }

            function bopScoreBestText(best) {
                if (!best) {
                    return 'No BOP19 average recall';
                }
                return (best.result_filename || 'BOP result')
                    + ' AR=' + metricText(best.bop19_average_recall);
            }

            function renderBopScoreTable(parent, rows) {
                const table = document.createElement('table');
                table.className = 'table table-sm align-middle mb-0';
                const header = document.createElement('thead');
                header.innerHTML = '<tr><th>Result</th><th>AR</th><th>Metrics</th><th>Artifact</th></tr>';
                table.appendChild(header);
                const body = document.createElement('tbody');
                (rows || []).slice(0, 12).forEach(row => {
                    const tr = document.createElement('tr');
                    const metrics = row.metrics || {};
                    [
                        row.result_filename,
                        metricText(metrics.bop19_average_recall),
                        row.score_metric_count,
                        row.relative_path,
                    ].forEach(value => {
                        const td = document.createElement('td');
                        td.textContent = value === undefined || value === null ? '' : String(value);
                        tr.appendChild(td);
                    });
                    body.appendChild(tr);
                });
                table.appendChild(body);
                parent.appendChild(table);
                if (!body.children.length) {
                    const empty = document.createElement('div');
                    empty.className = 'text-muted';
                    empty.textContent = 'No BOP Toolkit score rows found.';
                    parent.appendChild(empty);
                }
            }

            function renderMetrics(detail) {
                const panel = clearMetricsPanel();
                renderMetricRow(panel, 'Summary', [
                    'Artifacts: ' + detail.metric_artifact_count,
                    'Methods: ' + detail.method_count + ' (' + (detail.methods || []).join(', ') + ')',
                    'Direct rows: ' + detail.direct_method_count,
                    'Combined groups: ' + detail.combined_group_count,
                    'Best: ' + metricBestText(detail.best_by_AP_p),
                    'BOP score rows: ' + (detail.bop_score_count || 0),
                    'Best BOP19 AR: ' + bopScoreBestText(detail.best_bop19_average_recall),
                ]);

                const directItem = document.createElement('div');
                directItem.className = 'list-group-item';
                const directTitle = document.createElement('div');
                directTitle.className = 'fw-semibold mb-2';
                directTitle.textContent = 'Direct Methods';
                directItem.appendChild(directTitle);
                renderDirectMetricTable(directItem, detail.direct_methods);
                panel.appendChild(directItem);

                const groupTitle = document.createElement('div');
                groupTitle.className = 'list-group-item fw-semibold';
                groupTitle.textContent = 'Combined Groups';
                panel.appendChild(groupTitle);
                renderCombinedMetricGroups(panel, detail.combined_groups);

                const bopItem = document.createElement('div');
                bopItem.className = 'list-group-item';
                const bopTitle = document.createElement('div');
                bopTitle.className = 'fw-semibold mb-2';
                bopTitle.textContent = 'BOP Toolkit Scores';
                bopItem.appendChild(bopTitle);
                renderBopScoreTable(bopItem, detail.bop_scores);
                panel.appendChild(bopItem);
            }

            function bopScenePathValue() {
                return document.getElementById('bopScenePath').value.trim();
            }

            function bopResultPathValue() {
                return document.getElementById('bopResultPath').value.trim();
            }

            function artifactFileUrl(path) {
                return '/artifacts/file?run_root='
                    + encodeURIComponent(runRootValue())
                    + '&path='
                    + encodeURIComponent(path);
            }

            function bopFrameOverlayUrl() {
                const resultPath = bopResultPathValue();
                let url = '/artifacts/bop-frame-overlay?run_root='
                    + encodeURIComponent(runRootValue())
                    + '&path='
                    + encodeURIComponent(bopScenePathValue())
                    + '&image_id='
                    + encodeURIComponent(document.getElementById('bopImageId').value);
                if (resultPath) {
                    url += '&result_path=' + encodeURIComponent(resultPath);
                }
                return url;
            }

            function clearBopInspector() {
                const panel = document.getElementById('bopInspectorPanel');
                panel.className = 'border rounded p-3 small';
                panel.innerHTML = '';
                return panel;
            }

            function appendTextLine(parent, label, value) {
                if (value === undefined || value === null || value === '') {
                    return;
                }
                const line = document.createElement('div');
                const strong = document.createElement('strong');
                strong.textContent = label + ': ';
                line.appendChild(strong);
                line.appendChild(document.createTextNode(String(value)));
                parent.appendChild(line);
            }

            function appendImage(parent, label, artifact) {
                if (!artifact || !artifact.relative_path || !artifact.exists) {
                    return;
                }
                const wrapper = document.createElement('div');
                wrapper.className = 'mb-3';
                const title = document.createElement('div');
                title.className = 'fw-semibold mb-1';
                title.textContent = label;
                const image = document.createElement('img');
                image.className = 'img-fluid border rounded bg-light';
                image.alt = label;
                image.src = artifactFileUrl(artifact.relative_path);
                wrapper.appendChild(title);
                wrapper.appendChild(image);
                parent.appendChild(wrapper);
            }

            function appendImageGallery(parent, label, artifacts) {
                const existingArtifacts = (artifacts || []).filter(
                    artifact => artifact && artifact.relative_path && artifact.exists
                );
                if (!existingArtifacts.length) {
                    return;
                }
                const section = document.createElement('div');
                section.className = 'mb-3';
                const title = document.createElement('div');
                title.className = 'fw-semibold mb-1';
                title.textContent = label + ' (' + existingArtifacts.length + ')';
                section.appendChild(title);

                const grid = document.createElement('div');
                grid.className = 'row g-2';
                existingArtifacts.slice(0, 12).forEach((artifact, index) => {
                    const cell = document.createElement('div');
                    cell.className = 'col-6 col-lg-4';
                    const image = document.createElement('img');
                    image.className = 'img-fluid border rounded bg-light';
                    image.alt = label + ' ' + index;
                    image.src = artifactFileUrl(artifact.relative_path);
                    const caption = document.createElement('div');
                    caption.className = 'text-muted small text-truncate';
                    caption.textContent = artifact.name || artifact.relative_path;
                    cell.appendChild(image);
                    cell.appendChild(caption);
                    grid.appendChild(cell);
                });
                section.appendChild(grid);
                parent.appendChild(section);
            }

            function compactJson(value) {
                if (value === undefined || value === null) {
                    return '';
                }
                return JSON.stringify(value);
            }

            function renderGtRows(parent, annotations, infoRows) {
                const rows = annotations || [];
                const infos = infoRows || [];
                const title = document.createElement('div');
                title.className = 'fw-semibold mt-3 mb-1';
                title.textContent = 'Ground truth annotations: ' + rows.length;
                parent.appendChild(title);
                if (!rows.length) {
                    const empty = document.createElement('div');
                    empty.className = 'text-muted';
                    empty.textContent = 'No ground-truth annotations for this frame.';
                    parent.appendChild(empty);
                    return;
                }

                const table = document.createElement('table');
                table.className = 'table table-sm align-middle mb-0';
                const header = document.createElement('thead');
                header.innerHTML = '<tr><th>#</th><th>Obj</th><th>bbox</th><th>visible</th><th>px</th></tr>';
                table.appendChild(header);
                const body = document.createElement('tbody');
                rows.slice(0, 20).forEach((row, index) => {
                    const annotation = row || {};
                    const info = infos[index] || {};
                    const tr = document.createElement('tr');
                    [
                        index,
                        annotation.obj_id,
                        compactJson(info.bbox_obj || annotation.bbox_obj),
                        info.visib_fract,
                        info.px_count_visib,
                    ].forEach(value => {
                        const td = document.createElement('td');
                        td.textContent = value === undefined || value === null ? '' : String(value);
                        tr.appendChild(td);
                    });
                    body.appendChild(tr);
                });
                table.appendChild(body);
                parent.appendChild(table);
            }

            function renderPoseRows(parent, rows) {
                const table = document.createElement('table');
                table.className = 'table table-sm align-middle mb-0';
                const header = document.createElement('thead');
                header.innerHTML = '<tr><th>Scene</th><th>Image</th><th>Obj</th><th>Score</th><th>t</th><th>Time</th></tr>';
                table.appendChild(header);
                const body = document.createElement('tbody');
                (rows || []).slice(0, 20).forEach(row => {
                    const tr = document.createElement('tr');
                    const translation = Array.isArray(row.t)
                        ? row.t.map(value => Number(value).toFixed(2)).join(', ')
                        : '';
                    [row.scene_id, row.im_id, row.obj_id, row.score, translation, row.time].forEach(value => {
                        const td = document.createElement('td');
                        td.textContent = value === undefined || value === null ? '' : String(value);
                        tr.appendChild(td);
                    });
                    body.appendChild(tr);
                });
                table.appendChild(body);
                parent.appendChild(table);
                if (!body.children.length) {
                    const empty = document.createElement('div');
                    empty.className = 'text-muted';
                    empty.textContent = 'No pose rows in this view.';
                    parent.appendChild(empty);
                }
            }

            function renderBopFrame(detail) {
                const panel = clearBopInspector();
                const row = document.createElement('div');
                row.className = 'row g-3';
                const imageColumn = document.createElement('div');
                imageColumn.className = 'col-md-6';
                const overlayWrapper = document.createElement('div');
                overlayWrapper.className = 'mb-3';
                const overlayTitle = document.createElement('div');
                overlayTitle.className = 'fw-semibold mb-1';
                overlayTitle.textContent = 'Overlay';
                const overlayImage = document.createElement('img');
                overlayImage.className = 'img-fluid border rounded bg-light';
                overlayImage.alt = 'BOP frame overlay';
                overlayImage.src = bopFrameOverlayUrl();
                overlayWrapper.appendChild(overlayTitle);
                overlayWrapper.appendChild(overlayImage);
                imageColumn.appendChild(overlayWrapper);
                appendImage(imageColumn, 'RGB', detail.rgb);
                appendImage(imageColumn, 'Depth', detail.depth);
                appendImageGallery(imageColumn, 'Visible masks', detail.mask_visib_artifacts);
                appendImageGallery(imageColumn, 'Masks', detail.mask_artifacts);
                if (!imageColumn.children.length) {
                    imageColumn.className += ' text-muted';
                    imageColumn.textContent = 'No RGB/depth files were found for this frame.';
                }

                const infoColumn = document.createElement('div');
                infoColumn.className = 'col-md-6';
                appendTextLine(infoColumn, 'Scene', detail.relative_path);
                appendTextLine(infoColumn, 'Scene ID', detail.scene && detail.scene.scene_id);
                appendTextLine(infoColumn, 'Image ID', detail.image_id);
                appendTextLine(infoColumn, 'GT count', detail.gt_count);
                if (detail.frame_map) {
                    appendTextLine(infoColumn, 'Source sensor', detail.frame_map.sensor_name);
                    appendTextLine(infoColumn, 'Source frame', detail.frame_map.source_frame_id);
                    appendTextLine(infoColumn, 'Source RGB', detail.frame_map.source_rgb);
                    appendTextLine(infoColumn, 'Source depth', detail.frame_map.source_depth);
                }
                if (detail.camera) {
                    appendTextLine(infoColumn, 'Depth scale', detail.camera.depth_scale);
                    appendTextLine(infoColumn, 'Camera K', compactJson(detail.camera.cam_K));
                }
                renderGtRows(infoColumn, detail.gt, detail.gt_info);
                const result = detail.result;
                if (result) {
                    const heading = document.createElement('div');
                    heading.className = 'fw-semibold mt-3 mb-1';
                    heading.textContent = 'Matching pose rows: ' + result.matching_row_count + ' of ' + result.row_count;
                    infoColumn.appendChild(heading);
                    renderPoseRows(infoColumn, result.rows);
                }

                row.appendChild(imageColumn);
                row.appendChild(infoColumn);
                panel.appendChild(row);
            }

            function renderBopResult(detail) {
                const panel = clearBopInspector();
                const summary = document.createElement('div');
                summary.className = 'mb-3';
                appendTextLine(summary, 'Result', detail.relative_path);
                appendTextLine(summary, 'Method', detail.metadata && detail.metadata.method);
                appendTextLine(summary, 'Dataset', detail.metadata && detail.metadata.dataset);
                appendTextLine(summary, 'Split', detail.metadata && detail.metadata.split);
                appendTextLine(summary, 'Rows', detail.row_count);
                appendTextLine(summary, 'Scenes', detail.scene_count);
                panel.appendChild(summary);
                renderPoseRows(panel, detail.rows);
            }

            function loadBopFrame() {
                const scenePath = bopScenePathValue();
                const resultPath = bopResultPathValue();
                let url = '/artifacts/bop-frame?run_root='
                    + encodeURIComponent(runRootValue())
                    + '&path='
                    + encodeURIComponent(scenePath)
                    + '&image_id='
                    + encodeURIComponent(document.getElementById('bopImageId').value);
                if (resultPath) {
                    url += '&result_path=' + encodeURIComponent(resultPath);
                }
                fetch(url)
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'BOP frame load failed');
                    }
                    renderBopFrame(result.data);
                    output(JSON.stringify(result.data, null, 2));
                })
                .catch(error => output(error.message));
            }

            function loadBopResult() {
                const resultPath = bopResultPathValue();
                fetch('/artifacts/bop-result?run_root=' + encodeURIComponent(runRootValue()) + '&path=' + encodeURIComponent(resultPath))
                .then(response => response.json().then(data => ({ok: response.ok, data: data})))
                .then(result => {
                    if (!result.ok) {
                        throw new Error(result.data.output || 'BOP result load failed');
                    }
                    renderBopResult(result.data);
                    output(JSON.stringify(result.data, null, 2));
                })
                .catch(error => output(error.message));
            }

            function pollJob(jobId) {
                fetch('/jobs/' + jobId)
                .then(response => response.json())
                .then(data => {
                    const job = data.job;
                    const lines = [
                        'Job: ' + job.name + ' (' + job.id + ')',
                        'Status: ' + job.status,
                        job.message ? 'Message: ' + job.message : '',
                        '',
                        ...(job.tail || []),
                    ].filter(Boolean);
                    output(lines.join('\\n'));
                    if (job.status === 'queued' || job.status === 'running') {
                        setTimeout(() => pollJob(jobId), 1000);
                    } else {
                        refreshJobs();
                    }
                });
            }
            refreshJobs();
            refreshCaptureJobs();
        </script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    return html.replace("__SEQUENCE_OPTIONS__", _sequence_options_html())

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
        job = job_runner.submit(
            name=command,
            command=spec["command"],
            cwd=Path(__file__).resolve().parent,
            resources=spec.get("resources", []),
            parameters={
                "command": command,
                "label": spec.get("label", command),
                "resources": spec.get("resources", []),
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


@app.route('/sensors/status', methods=['GET'])
def sensor_status():
    try:
        expected_counts = (
            parse_expected_counts(request.args.getlist('expected'))
            if request.args.getlist('expected')
            else None
        )
    except ValueError as exc:
        return jsonify({'output': str(exc)}), 400
    return jsonify(collect_sensor_status(expected_counts=expected_counts))


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


@app.route('/artifacts/metrics', methods=['GET'])
def artifact_metrics():
    run_root = request.args.get('run_root')
    if not run_root:
        return jsonify({'output': 'Missing run_root'}), 400
    if not Path(run_root).exists():
        return jsonify({'output': f'Run root not found: {run_root}'}), 404
    try:
        group_limit = int(request.args.get('group_limit', '200'))
    except ValueError:
        return jsonify({'output': 'group_limit must be an integer'}), 400
    try:
        return jsonify(metric_dashboard_summary(run_root, group_limit=group_limit))
    except ValueError as exc:
        return jsonify({'output': str(exc)}), 400


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


@app.route('/artifacts/bop-result', methods=['GET'])
def artifact_bop_result():
    run_root = request.args.get('run_root')
    result_path = request.args.get('path')
    if not run_root or not result_path:
        return jsonify({'output': 'Missing run_root or path'}), 400
    if not Path(run_root).exists():
        return jsonify({'output': f'Run root not found: {run_root}'}), 404
    try:
        row_limit = int(request.args.get('row_limit', '500'))
    except ValueError:
        return jsonify({'output': 'row_limit must be an integer'}), 400
    try:
        return jsonify(bop_result_detail(run_root, result_path, row_limit=row_limit))
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
                result_path=request.args.get('result_path'),
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
            result_path=request.args.get('result_path'),
            row_limit=row_limit,
            include_masks=not _truthy(request.args.get('no_masks'), default=False),
            include_gt=not _truthy(request.args.get('no_gt'), default=False),
            include_results=not _truthy(
                request.args.get('no_results'),
                default=False,
            ),
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
        include_sensors = not bool(data.get('no_sensors', False))
        allow_real_robot = bool(data.get('allow_real_robot', False))
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
        require_valid = bool(data.get('require_valid', False))
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
                    promote=bool(data.get('promote', False)),
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
            cwd=Path(__file__).resolve().parent,
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
            cwd=Path(__file__).resolve().parent,
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
            cwd=Path(__file__).resolve().parent,
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
    app.run(host=WEB_HOST, port=WEB_PORT, debug=WEB_DEBUG)
