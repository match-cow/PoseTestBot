"""Typed pipeline stage presets for local PoseTestBot job submission."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


VALID_PARAMETER_KINDS = {"str", "path", "int", "float", "bool"}
VALID_PATH_SCOPES = {"run", "input", "output", "repository"}


@dataclass(frozen=True)
class PipelineParameter:
    """A CLI parameter exposed by a pipeline stage preset."""

    name: str
    flag: str
    kind: str = "str"
    path_scope: str | None = None
    required: bool = False
    default: Any = None
    choices: tuple[str, ...] = ()
    multiple: bool = False
    help: str = ""

    def __post_init__(self) -> None:
        if self.kind not in VALID_PARAMETER_KINDS:
            raise ValueError(f"Unsupported pipeline parameter kind: {self.kind}")
        if self.multiple and self.kind == "bool":
            raise ValueError("Boolean pipeline parameters cannot be multiple")
        if self.kind == "path":
            if self.path_scope not in VALID_PATH_SCOPES:
                raise ValueError(
                    f"Path parameter {self.name!r} must declare path_scope as one of: "
                    + ", ".join(sorted(VALID_PATH_SCOPES))
                )
        elif self.path_scope is not None:
            raise ValueError("path_scope is only valid for path parameters")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["choices"] = list(self.choices)
        return data


@dataclass(frozen=True)
class PipelineStageSpec:
    """A named stage that can be converted into a safe command array."""

    id: str
    label: str
    script: str
    description: str
    resources: tuple[str, ...] = ()
    parameters: tuple[PipelineParameter, ...] = ()
    stage_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "stage_name": self.stage_name or self.id,
            "script": self.script,
            "description": self.description,
            "resources": list(self.resources),
            "parameters": [parameter.to_dict() for parameter in self.parameters],
        }


@dataclass(frozen=True)
class PipelineJobSpec:
    """Concrete command submission data for one configured pipeline stage."""

    stage_id: str
    stage_label: str
    run_root: str
    command: list[str]
    resources: list[str]
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stringify_path(value: Any) -> str:
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def _coerce_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Pipeline option {name!r} must be a boolean")


def _coerce_single_value(parameter: PipelineParameter, value: Any) -> Any:
    if parameter.kind == "bool":
        return _coerce_bool(value, parameter.name)
    if parameter.kind == "int":
        if isinstance(value, bool):
            raise ValueError(f"Pipeline option {parameter.name!r} must be an integer")
        return int(value)
    if parameter.kind == "float":
        if isinstance(value, bool):
            raise ValueError(f"Pipeline option {parameter.name!r} must be a number")
        return float(value)
    if parameter.kind == "path":
        return _stringify_path(value)
    return str(value)


def _normalize_parameter_value(parameter: PipelineParameter, value: Any) -> Any:
    if parameter.multiple:
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
            raise ValueError(f"Pipeline option {parameter.name!r} must be a list")
        return [_coerce_single_value(parameter, item) for item in value]
    return _coerce_single_value(parameter, value)


def _validate_choices(parameter: PipelineParameter, value: Any) -> None:
    if not parameter.choices:
        return
    values = value if isinstance(value, list) else [value]
    invalid = [item for item in values if item not in parameter.choices]
    if invalid:
        choices = ", ".join(parameter.choices)
        raise ValueError(
            f"Pipeline option {parameter.name!r} must be one of: {choices}"
        )


def normalize_stage_options(
    stage: PipelineStageSpec, options: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate user options and apply stage defaults."""

    provided = dict(options or {})
    known_names = {parameter.name for parameter in stage.parameters}
    unknown_names = sorted(set(provided) - known_names)
    if unknown_names:
        raise ValueError(
            f"Unknown pipeline option(s) for {stage.id}: {', '.join(unknown_names)}"
        )

    normalized: dict[str, Any] = {}
    for parameter in stage.parameters:
        if parameter.name in provided:
            raw_value = provided[parameter.name]
        else:
            raw_value = parameter.default

        missing = raw_value is None or raw_value == ""
        if missing:
            if parameter.required:
                raise ValueError(f"Missing required pipeline option: {parameter.name}")
            continue

        value = _normalize_parameter_value(parameter, raw_value)
        _validate_choices(parameter, value)
        normalized[parameter.name] = value

    return normalized


def command_for_stage(
    stage: PipelineStageSpec,
    *,
    run_root: str | Path,
    options: Mapping[str, Any] | None = None,
) -> PipelineJobSpec:
    """Build the command-array representation for a pipeline stage."""

    run_root_value = _stringify_path(run_root)
    if not run_root_value:
        raise ValueError("Pipeline run_root must not be empty")

    normalized_options = normalize_stage_options(stage, options)
    command = ["uv", "run", "python", stage.script, run_root_value]

    parameters: dict[str, Any] = {
        "pipeline_stage": stage.id,
        "stage_label": stage.label,
        "run_root": run_root_value,
        "options": dict(normalized_options),
    }

    for parameter in stage.parameters:
        if parameter.name not in normalized_options:
            continue
        value = normalized_options[parameter.name]
        if parameter.kind == "bool":
            if value:
                command.append(parameter.flag)
            continue
        if parameter.multiple:
            for item in value:
                command.extend([parameter.flag, str(item)])
            continue
        command.extend([parameter.flag, str(value)])

    return PipelineJobSpec(
        stage_id=stage.id,
        stage_label=stage.label,
        run_root=run_root_value,
        command=command,
        resources=list(stage.resources),
        parameters=parameters,
    )


def list_pipeline_stages(
    registry: Mapping[str, PipelineStageSpec] | None = None,
) -> list[dict[str, Any]]:
    specs = PIPELINE_STAGES if registry is None else registry
    return [
        stage.to_dict() for stage in sorted(specs.values(), key=lambda item: item.id)
    ]


def get_pipeline_stage(
    stage_id: str,
    registry: Mapping[str, PipelineStageSpec] | None = None,
) -> PipelineStageSpec:
    specs = PIPELINE_STAGES if registry is None else registry
    try:
        return specs[stage_id]
    except KeyError as exc:
        raise ValueError(f"Unknown pipeline stage: {stage_id}") from exc


def build_pipeline_job(
    *,
    stage_id: str,
    run_root: str | Path,
    options: Mapping[str, Any] | None = None,
    registry: Mapping[str, PipelineStageSpec] | None = None,
) -> PipelineJobSpec:
    stage = get_pipeline_stage(stage_id, registry=registry)
    return command_for_stage(stage, run_root=run_root, options=options)


PIPELINE_STAGES: dict[str, PipelineStageSpec] = {
    "rewrite_gate": PipelineStageSpec(
        id="rewrite_gate",
        label="Rewrite Gate Audit",
        script="scripts/run_rewrite_gate.py",
        description=(
            "Audit whether a run folder contains concrete evidence for a "
            "rewrite milestone gate."
        ),
        resources=("disk_io",),
        parameters=(
            PipelineParameter(
                name="gate",
                flag="--gate",
                default="rewrite_full_capture.v1",
                choices=(
                    "rewrite_full_capture.v1",
                    "rewrite_calibration_validation.v1",
                    "rewrite_bop_export_readiness.v1",
                ),
            ),
            PipelineParameter(
                name="write",
                flag="--write",
                kind="bool",
                default=True,
            ),
            PipelineParameter(name="json", flag="--json", kind="bool", default=False),
        ),
    ),
    "rewrite_status": PipelineStageSpec(
        id="rewrite_status",
        label="Rewrite Status Audit",
        script="scripts/run_rewrite_status.py",
        description=(
            "Summarize all rewrite milestone gates for a run folder and write "
            "rewrite_status_report.json."
        ),
        resources=("disk_io",),
        parameters=(
            PipelineParameter(
                name="write",
                flag="--write",
                kind="bool",
                default=True,
            ),
            PipelineParameter(name="json", flag="--json", kind="bool", default=False),
            PipelineParameter(
                name="gate_run_root",
                flag="--gate-run-root",
                multiple=True,
                help="Per-gate evidence root as GATE_ID=RUN_ROOT.",
            ),
        ),
    ),
    "hardware_status": PipelineStageSpec(
        id="hardware_status",
        label="Hardware Status Snapshot",
        script="scripts/run_hardware_status_stage.py",
        description=(
            "Write hardware_status_report.json with read-only robot, sensor, "
            "and external runtime readiness for a run."
        ),
        resources=("camera", "disk_io"),
        parameters=(
            PipelineParameter(
                name="no_sensors",
                flag="--no-sensors",
                kind="bool",
                default=False,
                help="Skip camera SDK/device discovery checks.",
            ),
            PipelineParameter(
                name="no_runtimes",
                flag="--no-runtimes",
                kind="bool",
                default=False,
                help="Skip external runtime readiness checks.",
            ),
            PipelineParameter(
                name="plan_only",
                flag="--plan-only",
                kind="bool",
                default=False,
            ),
            PipelineParameter(name="json", flag="--json", kind="bool", default=False),
        ),
    ),
    "run_preflight": PipelineStageSpec(
        id="run_preflight",
        label="Run Preflight",
        script="scripts/run_preflight.py",
        description=(
            "Write run_preflight_report.json with run-config, sequence, robot, "
            "sensor, and runtime readiness checks before queueing a workflow."
        ),
        resources=("camera", "disk_io"),
        parameters=(
            PipelineParameter(
                name="no_sensors",
                flag="--no-sensors",
                kind="bool",
                default=False,
                help="Skip live sensor discovery.",
            ),
            PipelineParameter(
                name="no_runtimes",
                flag="--no-runtimes",
                kind="bool",
                default=False,
                help="Skip external runtime readiness checks.",
            ),
            PipelineParameter(
                name="check",
                flag="--check",
                kind="bool",
                default=False,
                help="Exit nonzero when preflight status is error.",
            ),
            PipelineParameter(
                name="write",
                flag="--write",
                kind="bool",
                default=True,
                help="Persist run_preflight_report.json and manifest stage.",
            ),
            PipelineParameter(name="json", flag="--json", kind="bool", default=False),
        ),
    ),
    "capture_plan": PipelineStageSpec(
        id="capture_plan",
        label="Capture Plan",
        script="scripts/run_capture_plan_stage.py",
        description=(
            "Build capture_plan.json from run_config.json without starting cameras "
            "or robot motion."
        ),
        resources=("disk_io",),
        parameters=(
            PipelineParameter(
                name="run_config", flag="--run-config", kind="path", path_scope="run"
            ),
            PipelineParameter(
                name="max_frames",
                flag="--max-frames",
                kind="int",
                help="Optional max frame count to include in planned camera commands.",
            ),
            PipelineParameter(
                name="warmup_frames",
                flag="--warmup-frames",
                kind="int",
                help=(
                    "Optional valid frame count for camera commands to discard "
                    "before writing capture output."
                ),
            ),
            PipelineParameter(
                name="print_json",
                flag="--print-json",
                kind="bool",
                default=False,
                help="Print the full capture plan JSON after writing it.",
            ),
        ),
    ),
    "capture_plan_preflight": PipelineStageSpec(
        id="capture_plan_preflight",
        label="Capture Plan Preflight",
        script="scripts/run_capture_plan_preflight.py",
        description=(
            "Validate capture_plan.json command shape, real robot safety, "
            "script availability, and optional sensor readiness."
        ),
        resources=("camera", "disk_io"),
        parameters=(
            PipelineParameter(
                name="allow_real_robot",
                flag="--allow-real-robot",
                kind="bool",
                default=False,
            ),
            PipelineParameter(
                name="no_sensors",
                flag="--no-sensors",
                kind="bool",
                default=False,
            ),
            PipelineParameter(
                name="no_write_plan_if_missing",
                flag="--no-write-plan-if-missing",
                kind="bool",
                default=False,
            ),
            PipelineParameter(name="json", flag="--json", kind="bool", default=False),
        ),
    ),
    "capture_execution_plan": PipelineStageSpec(
        id="capture_execution_plan",
        label="Capture Execution Plan",
        script="scripts/run_capture_execution_plan.py",
        description=(
            "Select all full-capture commands without "
            "starting robot or camera processes."
        ),
        resources=("camera", "disk_io"),
        parameters=(
            PipelineParameter(
                name="allow_cameras",
                flag="--allow-cameras",
                kind="bool",
                default=False,
            ),
            PipelineParameter(
                name="allow_real_robot",
                flag="--allow-real-robot",
                kind="bool",
                default=False,
            ),
            PipelineParameter(
                name="include_sensors",
                flag="--include-sensors",
                kind="bool",
                default=False,
            ),
            PipelineParameter(
                name="no_write_plan_if_missing",
                flag="--no-write-plan-if-missing",
                kind="bool",
                default=False,
            ),
            PipelineParameter(name="json", flag="--json", kind="bool", default=False),
        ),
    ),
    "capture_execution": PipelineStageSpec(
        id="capture_execution",
        label="Supervised Capture Execution",
        script="scripts/run_capture_execution_stage.py",
        description=(
            "Execute selected capture-plan commands with process-group "
            "supervision for the real robot and configured cameras."
        ),
        resources=("robot_command", "camera", "disk_io"),
        parameters=(
            PipelineParameter(
                name="allow_cameras",
                flag="--allow-cameras",
                kind="bool",
                default=False,
            ),
            PipelineParameter(
                name="allow_real_robot",
                flag="--allow-real-robot",
                kind="bool",
                default=False,
            ),
            PipelineParameter(
                name="include_sensors",
                flag="--include-sensors",
                kind="bool",
                default=False,
            ),
            PipelineParameter(
                name="timeout_s",
                flag="--timeout-s",
                kind="float",
                default=720.0,
            ),
            PipelineParameter(
                name="startup_wait_s",
                flag="--startup-wait",
                kind="float",
                default=15.0,
            ),
            PipelineParameter(
                name="camera_startup_attempts",
                flag="--camera-startup-attempts",
                kind="int",
                default=3,
            ),
            PipelineParameter(
                name="camera_startup_retry_delay_s",
                flag="--camera-startup-retry-delay-s",
                kind="float",
                default=1.0,
            ),
            PipelineParameter(
                name="terminate_timeout_s",
                flag="--terminate-timeout-s",
                kind="float",
                default=2.0,
            ),
            PipelineParameter(
                name="receive_start_timeout_s",
                flag="--receive-start-timeout-s",
                kind="float",
                default=120.0,
            ),
            PipelineParameter(
                name="receive_idle_timeout_s",
                flag="--receive-idle-timeout-s",
                kind="float",
                default=60.0,
            ),
            PipelineParameter(
                name="no_write_plan_if_missing",
                flag="--no-write-plan-if-missing",
                kind="bool",
                default=False,
            ),
            PipelineParameter(
                name="print_json",
                flag="--print-json",
                kind="bool",
                default=False,
            ),
        ),
    ),
    "realsense_capture_smoke": PipelineStageSpec(
        id="realsense_capture_smoke",
        label="RealSense Capture Smoke",
        script="scripts/run_realsense_capture_smoke.py",
        description=(
            "Validate a RealSense-only run config and capture short sequential "
            "RGB-D samples from the configured D435/D435i serials."
        ),
        resources=("camera", "disk_io"),
        parameters=(
            PipelineParameter(
                name="expected_count",
                flag="--expected-count",
                kind="int",
                default=3,
            ),
            PipelineParameter(name="fps", flag="--fps", kind="int", default=6),
            PipelineParameter(
                name="max_frames",
                flag="--max-frames",
                kind="int",
                default=30,
            ),
            PipelineParameter(
                name="warmup_frames",
                flag="--warmup-frames",
                kind="int",
                default=10,
            ),
            PipelineParameter(
                name="preview",
                flag="--preview",
                kind="bool",
                default=False,
            ),
            PipelineParameter(
                name="print_json",
                flag="--print-json",
                kind="bool",
                default=False,
            ),
        ),
    ),
    "calibration_preflight": PipelineStageSpec(
        id="calibration_preflight",
        label="Calibration Preflight",
        script="scripts/run_calibration_preflight.py",
        description=(
            "Check calibration profile coverage, status, and quality metrics "
            "for the sensors in run_config.json."
        ),
        resources=("disk_io",),
        parameters=(
            PipelineParameter(
                name="require_valid",
                flag="--require-valid",
                kind="bool",
                default=False,
            ),
            PipelineParameter(
                name="min_observations",
                flag="--min-observations",
                kind="int",
                default=6,
            ),
            PipelineParameter(
                name="max_mean_reprojection_error_px",
                flag="--max-mean-reprojection-error-px",
                kind="float",
                default=2.0,
            ),
            PipelineParameter(
                name="no_reprojection_threshold",
                flag="--no-reprojection-threshold",
                kind="bool",
                default=False,
            ),
            PipelineParameter(name="json", flag="--json", kind="bool", default=False),
        ),
    ),
    "calibration_observations": PipelineStageSpec(
        id="calibration_observations",
        label="Calibration Observations",
        script="scripts/run_calibration_observations.py",
        description=(
            "Extract solver-ready calibration observations from synchronized "
            "target-pose results and robot end-effector poses."
        ),
        resources=("disk_io",),
        parameters=(
            PipelineParameter(
                name="aruco_path",
                flag="--aruco-path",
                kind="path",
                path_scope="run",
                multiple=True,
                help="Repeat to process an explicit synchronized sensor subset.",
            ),
            PipelineParameter(
                name="output_root",
                flag="--output-root",
                kind="path",
                path_scope="output",
                help="Alternate derived observation output root.",
            ),
            PipelineParameter(
                name="min_marker_count",
                flag="--min-marker-count",
                kind="int",
                default=4,
            ),
            PipelineParameter(
                name="min_observations",
                flag="--min-observations",
                kind="int",
                default=6,
            ),
            PipelineParameter(
                name="target_spec",
                flag="--target-spec",
                kind="path",
                path_scope="run",
                default="calibration_target.json",
            ),
            PipelineParameter(name="json", flag="--json", kind="bool", default=False),
        ),
    ),
    "calibration_candidates": PipelineStageSpec(
        id="calibration_candidates",
        label="Legacy Calibration Candidates",
        script="scripts/run_calibration_candidates.py",
        description=(
            "Known-target/compatibility path for averaging calibration observations. "
            "It is not the guided moving-grid static-camera solve; use Workflow "
            "step 5 to estimate camera-to-PoseTemplateBase with a robot-carried grid."
        ),
        resources=("disk_io",),
        parameters=(
            PipelineParameter(
                name="observations",
                flag="--observations",
                kind="path",
                path_scope="run",
            ),
            PipelineParameter(
                name="min_observations",
                flag="--min-observations",
                kind="int",
                default=6,
            ),
            PipelineParameter(
                name="target_to_reference",
                flag="--target-to-reference",
                kind="path",
                path_scope="input",
            ),
            PipelineParameter(
                name="max_translation_residual_mm",
                flag="--max-translation-residual-mm",
                kind="float",
            ),
            PipelineParameter(
                name="max_rotation_residual_deg",
                flag="--max-rotation-residual-deg",
                kind="float",
            ),
            PipelineParameter(
                name="no_residual_thresholds",
                flag="--no-residual-thresholds",
                kind="bool",
                default=False,
            ),
            PipelineParameter(name="json", flag="--json", kind="bool", default=False),
        ),
    ),
    "calibration_solver": PipelineStageSpec(
        id="calibration_solver",
        label="Legacy Known-Target Solver",
        script="scripts/run_calibration_solver.py",
        description=(
            "Compatibility solver for a fixed/known target-to-reference transform. "
            "It cannot solve the guided static-camera arrangement with an unknown "
            "robot-carried grid; use Workflow step 5 for that joint solve."
        ),
        resources=("disk_io",),
        parameters=(
            PipelineParameter(
                name="mode",
                flag="--mode",
                choices=("hand_eye_unknown_target", "known_target", "compare"),
            ),
            PipelineParameter(
                name="calibration_target",
                flag="--calibration-target",
                kind="path",
                path_scope="run",
            ),
            PipelineParameter(
                name="observations",
                flag="--observations",
                kind="path",
                path_scope="run",
            ),
            PipelineParameter(
                name="min_observations",
                flag="--min-observations",
                kind="int",
                default=6,
            ),
            PipelineParameter(
                name="target_to_reference",
                flag="--target-to-reference",
                kind="path",
                path_scope="input",
            ),
            PipelineParameter(
                name="hand_eye_method",
                flag="--hand-eye-method",
                choices=("andreff", "daniilidis", "horaud", "park", "tsai"),
                default="tsai",
            ),
            PipelineParameter(
                name="max_translation_residual_mm",
                flag="--max-translation-residual-mm",
                kind="float",
            ),
            PipelineParameter(
                name="max_rotation_residual_deg",
                flag="--max-rotation-residual-deg",
                kind="float",
            ),
            PipelineParameter(
                name="no_residual_thresholds",
                flag="--no-residual-thresholds",
                kind="bool",
                default=False,
            ),
            PipelineParameter(
                name="holdout_fraction",
                flag="--holdout-fraction",
                kind="float",
            ),
            PipelineParameter(
                name="compare_hand_eye_methods",
                flag="--compare-hand-eye-methods",
                kind="bool",
                default=False,
            ),
            PipelineParameter(
                name="max_outlier_ratio",
                flag="--max-outlier-ratio",
                kind="float",
                default=0.25,
            ),
            PipelineParameter(
                name="max_cross_translation_mm",
                flag="--max-cross-translation-mm",
                kind="float",
                default=10.0,
            ),
            PipelineParameter(
                name="max_cross_rotation_deg",
                flag="--max-cross-rotation-deg",
                kind="float",
                default=5.0,
            ),
            PipelineParameter(name="json", flag="--json", kind="bool", default=False),
        ),
    ),
    "calibration_validation": PipelineStageSpec(
        id="calibration_validation",
        label="Calibration Validation",
        script="scripts/run_calibration_validation.py",
        description=(
            "Validate calibration candidate profiles against inlier, residual, "
            "and outlier-ratio gates. Promotion remains explicit."
        ),
        resources=("disk_io",),
        parameters=(
            PipelineParameter(
                name="candidates", flag="--candidates", kind="path", path_scope="run"
            ),
            PipelineParameter(
                name="profiles", flag="--profiles", kind="path", path_scope="run"
            ),
            PipelineParameter(
                name="min_inliers",
                flag="--min-inliers",
                kind="int",
                default=6,
            ),
            PipelineParameter(
                name="max_mean_translation_residual_mm",
                flag="--max-mean-translation-residual-mm",
                kind="float",
                default=10.0,
            ),
            PipelineParameter(
                name="max_mean_rotation_residual_deg",
                flag="--max-mean-rotation-residual-deg",
                kind="float",
                default=5.0,
            ),
            PipelineParameter(
                name="max_outlier_ratio",
                flag="--max-outlier-ratio",
                kind="float",
                default=0.25,
            ),
            PipelineParameter(
                name="promote",
                flag="--promote",
                kind="bool",
                default=False,
            ),
            PipelineParameter(
                name="output_profiles",
                flag="--output-profiles",
                kind="path",
                path_scope="output",
            ),
            PipelineParameter(name="operator", flag="--operator"),
            PipelineParameter(
                name="select_profile",
                flag="--select-profile",
                multiple=True,
                help="Repeatable SENSOR=PROFILE_ID promotion selection.",
            ),
            PipelineParameter(name="json", flag="--json", kind="bool", default=False),
        ),
    ),
    "sync_run": PipelineStageSpec(
        id="sync_run",
        label="Non-destructive Sync",
        script="scripts/sync_run_non_destructive.py",
        description=(
            "Synchronize all or an explicit subset of raw sensor folders into "
            "a derived output root without modifying raw captures."
        ),
        resources=("disk_io",),
        parameters=(
            PipelineParameter(
                name="sensor_folder",
                flag="--sensor-folder",
                kind="path",
                path_scope="run",
                multiple=True,
                help="Repeat to synchronize an explicit raw sensor subset.",
            ),
            PipelineParameter(
                name="output_root",
                flag="--output-root",
                kind="path",
                path_scope="output",
                help="Derived sync output root.",
            ),
            PipelineParameter(
                name="sync_delta",
                flag="--sync-delta",
                help="Sync delta in ms, or a JSON file mapping sensor types to ms.",
            ),
            PipelineParameter(
                name="timestamp_source",
                flag="--timestamp-source",
                choices=("host_received", "host_wall", "sensor", "filename"),
                help=(
                    "Timestamp source used for frame-to-robot matching. Runs with "
                    "a selected calibration use its immutable per-camera policy."
                ),
            ),
            PipelineParameter(
                name="robot_timestamp_source",
                flag="--robot-timestamp-source",
                choices=("host_received", "host_wall", "filename"),
                help="Explicit clock-compatible robot-pose timestamp source.",
            ),
            PipelineParameter(
                name="no_copy",
                flag="--no-copy",
                kind="bool",
                default=False,
                help="Write metadata only without copying RGB/depth frames.",
            ),
        ),
    ),
    "sync_quality": PipelineStageSpec(
        id="sync_quality",
        label="Sync Quality",
        script="scripts/run_sync_quality.py",
        description=(
            "Aggregate synchronized sync_report.json files and check dropped "
            "frames, nearest-pose deltas, match ratio, and timestamp source."
        ),
        resources=("disk_io",),
        parameters=(
            PipelineParameter(
                name="min_match_ratio",
                flag="--min-match-ratio",
                kind="float",
                default=0.8,
            ),
            PipelineParameter(
                name="max_dropped_frames",
                flag="--max-dropped-frames",
                kind="int",
            ),
            PipelineParameter(
                name="max_nearest_pose_delta_ms",
                flag="--max-nearest-pose-delta-ms",
                kind="float",
                help=(
                    "Manual quality threshold. Runs with a selected calibration "
                    "use its immutable per-camera threshold."
                ),
            ),
            PipelineParameter(
                name="no_nearest_pose_threshold",
                flag="--no-nearest-pose-threshold",
                kind="bool",
                default=False,
            ),
            PipelineParameter(
                name="require_timestamp_source",
                flag="--require-timestamp-source",
                choices=("host_received", "host_wall", "sensor", "filename"),
            ),
            PipelineParameter(
                name="require_robot_timestamp_source",
                flag="--require-robot-timestamp-source",
                choices=("host_received", "host_wall", "filename"),
            ),
            PipelineParameter(name="json", flag="--json", kind="bool", default=False),
        ),
    ),
    "aruco": PipelineStageSpec(
        id="aruco",
        label="ArUco Pose Estimation",
        script="scripts/run_aruco_stage.py",
        description="Run ArUco pose estimation on synchronized sensor folders.",
        resources=("cpu", "disk_io"),
        parameters=(
            PipelineParameter(
                name="calibration_target",
                flag="--calibration-target",
                kind="path",
                path_scope="run",
            ),
            PipelineParameter(
                name="intrinsics_mode",
                flag="--intrinsics-mode",
                choices=("factory", "calibrate"),
                default="factory",
            ),
            PipelineParameter(
                name="min_accepted_views",
                flag="--min-accepted-views",
                kind="int",
                default=15,
            ),
            PipelineParameter(
                name="min_coverage_cells",
                flag="--min-coverage-cells",
                kind="int",
                default=6,
            ),
            PipelineParameter(
                name="max_view_error_px",
                flag="--max-view-error-px",
                kind="float",
                default=3.0,
            ),
            PipelineParameter(
                name="max_rms_px",
                flag="--max-rms-px",
                kind="float",
                default=1.5,
            ),
            PipelineParameter(
                name="save_images",
                flag="--save-images",
                kind="bool",
                default=False,
                help="Save ArUco visualization images beside synchronized frames.",
            ),
            PipelineParameter(
                name="show",
                flag="--show",
                kind="bool",
                default=False,
                help="Display OpenCV windows while processing.",
            ),
            PipelineParameter(
                name="wait_time",
                flag="--wait-time",
                kind="int",
                help="OpenCV wait time in ms when display windows are enabled.",
            ),
        ),
    ),
    "calibration_target_import": PipelineStageSpec(
        id="calibration_target_import",
        label="Resolve Calibration Target",
        script="scripts/run_calibration_target_import.py",
        description=(
            "Validate the exact current target selected and snapshotted by run setup."
        ),
        resources=("disk_io",),
        parameters=(),
    ),
    "aruco_detection": PipelineStageSpec(
        id="aruco_detection",
        label="ArUco Grid Detection",
        script="scripts/run_aruco_detection_stage.py",
        description="Detect imported grid markers once in synchronized native RGB.",
        resources=("cpu", "disk_io"),
        parameters=(
            PipelineParameter(
                name="calibration_target",
                flag="--calibration-target",
                kind="path",
                path_scope="run",
            ),
            PipelineParameter(
                name="input_root",
                flag="--input-root",
                kind="path",
                path_scope="run",
            ),
            PipelineParameter(
                name="save_images", flag="--save-images", kind="bool", default=False
            ),
        ),
    ),
    "intrinsic_calibration": PipelineStageSpec(
        id="intrinsic_calibration",
        label="Color Intrinsic Calibration",
        script="scripts/run_intrinsic_calibration_stage.py",
        description="Wrap factory color intrinsics or calibrate from stored grid detections.",
        resources=("cpu", "disk_io"),
        parameters=(
            PipelineParameter(
                name="mode",
                flag="--mode",
                choices=("factory", "calibrate"),
                default="factory",
            ),
            PipelineParameter(
                name="calibration_target",
                flag="--calibration-target",
                kind="path",
                path_scope="run",
            ),
            PipelineParameter(
                name="input_root", flag="--input-root", kind="path", path_scope="run"
            ),
            PipelineParameter(
                name="min_accepted_views",
                flag="--min-accepted-views",
                kind="int",
                default=15,
            ),
            PipelineParameter(
                name="min_coverage_cells",
                flag="--min-coverage-cells",
                kind="int",
                default=6,
            ),
            PipelineParameter(
                name="max_view_error_px",
                flag="--max-view-error-px",
                kind="float",
                default=3.0,
            ),
            PipelineParameter(
                name="max_rms_px", flag="--max-rms-px", kind="float", default=1.5
            ),
        ),
    ),
    "aruco_pose": PipelineStageSpec(
        id="aruco_pose",
        label="ArUco Grid Pose Solve",
        script="scripts/run_aruco_pose_stage.py",
        description="Solve grid-to-camera poses from detections and native intrinsics.",
        resources=("cpu", "disk_io"),
        parameters=(
            PipelineParameter(
                name="calibration_target",
                flag="--calibration-target",
                kind="path",
                path_scope="run",
            ),
            PipelineParameter(
                name="intrinsic_profiles",
                flag="--intrinsic-profiles",
                kind="path",
                path_scope="run",
            ),
            PipelineParameter(
                name="input_root", flag="--input-root", kind="path", path_scope="run"
            ),
        ),
    ),
    "camera_rectification": PipelineStageSpec(
        id="camera_rectification",
        label="Camera Rectification",
        script="scripts/run_camera_rectification.py",
        description="Transactionally rectify synchronized RGB and aligned depth.",
        resources=("cpu", "disk_io"),
        parameters=(
            PipelineParameter(
                name="intrinsic_profiles",
                flag="--intrinsic-profiles",
                kind="path",
                path_scope="run",
            ),
            PipelineParameter(
                name="input_root", flag="--input-root", kind="path", path_scope="run"
            ),
            PipelineParameter(
                name="output_root",
                flag="--output-root",
                kind="path",
                path_scope="output",
            ),
            PipelineParameter(
                name="overwrite", flag="--overwrite", kind="bool", default=False
            ),
            PipelineParameter(name="json", flag="--json", kind="bool", default=False),
        ),
    ),
    "aruco_coverage": PipelineStageSpec(
        id="aruco_coverage",
        label="ArUco Coverage",
        script="scripts/run_aruco_coverage_stage.py",
        description=(
            "Summarize ArUco detection and valid-pose coverage from synchronized "
            "aruco_pose_estimation.json files."
        ),
        resources=("disk_io",),
        parameters=(
            PipelineParameter(
                name="min_marker_count",
                flag="--min-marker-count",
                kind="int",
                default=4,
            ),
            PipelineParameter(
                name="min_valid_pose_ratio",
                flag="--min-valid-pose-ratio",
                kind="float",
                default=0.0,
            ),
            PipelineParameter(
                name="aruco_pose_file",
                flag="--aruco-pose-file",
                kind="path",
                path_scope="run",
                multiple=True,
            ),
            PipelineParameter(name="json", flag="--json", kind="bool", default=False),
        ),
    ),
    "blenderproc_prepare": PipelineStageSpec(
        id="blenderproc_prepare",
        label="BlenderProc Preparation",
        script="scripts/run_blenderproc_prepare_stage.py",
        description="Prepare BlenderProc inputs from synchronized sensor folders.",
        resources=("cpu", "disk_io"),
        parameters=(
            PipelineParameter(
                name="input_folder",
                flag="--input-folder",
                kind="path",
                path_scope="run",
            ),
            PipelineParameter(
                name="objectless", flag="--objectless", kind="bool", default=False
            ),
            PipelineParameter(
                name="camera_transformations",
                flag="--camera-transformations",
                kind="path",
                path_scope="input",
            ),
            PipelineParameter(
                name="calibration_profiles",
                flag="--calibration-profiles",
                kind="path",
                path_scope="input",
            ),
            PipelineParameter(
                name="annotation_mode",
                flag="--annotation-mode",
                default="pose_and_masks",
                choices=("pose", "pose_and_masks"),
                help=(
                    "Bind prepared frames to the selected pose-only or "
                    "pose-plus-mask annotation product."
                ),
            ),
            PipelineParameter(name="subdir", flag="--subdir"),
        ),
    ),
    "blenderproc_render": PipelineStageSpec(
        id="blenderproc_render",
        label="BlenderProc Ground-Truth Poses",
        script="scripts/run_blenderproc_render_stage.py",
        description=(
            "Validate prepared scenes in BlenderProc and derive exact "
            "model-to-camera ground-truth poses. Mask evidence is completed "
            "during BOP export."
        ),
        resources=("render", "disk_io"),
        parameters=(
            PipelineParameter(
                name="input_folder",
                flag="--input-folder",
                kind="path",
                path_scope="run",
            ),
            PipelineParameter(
                name="render_script",
                flag="--render-script",
                kind="path",
                path_scope="repository",
            ),
            PipelineParameter(name="subdir", flag="--subdir"),
            PipelineParameter(name="blenderproc", flag="--blenderproc"),
            PipelineParameter(
                name="annotation_mode",
                flag="--annotation-mode",
                default="pose_and_masks",
                choices=("pose", "pose_and_masks"),
                help=(
                    "Select pose-only GT or the pose-plus-mask product. "
                    "BlenderProc writes the pose evidence for both."
                ),
            ),
            PipelineParameter(
                name="objectless", flag="--objectless", kind="bool", default=False
            ),
            PipelineParameter(
                name="dry_run",
                flag="--dry-run",
                kind="bool",
                default=True,
                help="Write a render plan without executing BlenderProc.",
            ),
        ),
    ),
    "bop_export": PipelineStageSpec(
        id="bop_export",
        label="BOP Dataset Export",
        script="scripts/run_bop_export_stage.py",
        description=(
            "Export synchronized sensor folders and canonical models into BOP "
            "scene folders. Rendered GT/masks are opt-in."
        ),
        resources=("disk_io",),
        parameters=(
            PipelineParameter(
                name="input_folder",
                flag="--input-folder",
                kind="path",
                path_scope="run",
            ),
            PipelineParameter(
                name="output_folder",
                flag="--output-folder",
                kind="path",
                path_scope="output",
            ),
            PipelineParameter(name="split", flag="--split"),
            PipelineParameter(
                name="objectless", flag="--objectless", kind="bool", default=False
            ),
            PipelineParameter(
                name="no_model_export",
                flag="--no-model-export",
                kind="bool",
                default=False,
            ),
            PipelineParameter(name="scene_start", flag="--scene-start", kind="int"),
            PipelineParameter(name="overwrite", flag="--overwrite", kind="bool"),
            PipelineParameter(
                name="calibration_profiles",
                flag="--calibration-profiles",
                kind="path",
                path_scope="input",
            ),
            PipelineParameter(
                name="write_coco_annotations",
                flag="--write-coco-annotations",
                kind="bool",
                default=False,
            ),
            PipelineParameter(
                name="annotation_source",
                flag="--annotation-source",
                default="none",
                choices=("none", "blenderproc"),
                help=(
                    "Use 'none' for an image/model BOP dataset without rendered "
                    "GT, or 'blenderproc' to consume optional rendered annotations."
                ),
            ),
            PipelineParameter(
                name="annotation_mode",
                flag="--annotation-mode",
                choices=("pose", "pose_and_masks"),
                help=(
                    "With BlenderProc annotations, choose pose-only scene_gt or "
                    "pose plus official BOP masks, visibility, and GT-info."
                ),
            ),
        ),
    ),
}
