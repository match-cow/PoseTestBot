"""Typed pipeline stage presets for local PoseTestBot job submission."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


VALID_PARAMETER_KINDS = {"str", "path", "int", "float", "bool"}


@dataclass(frozen=True)
class PipelineParameter:
    """A CLI parameter exposed by a pipeline stage preset."""

    name: str
    flag: str
    kind: str = "str"
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
            "Unknown pipeline option(s) for "
            f"{stage.id}: {', '.join(unknown_names)}"
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
    return [stage.to_dict() for stage in sorted(specs.values(), key=lambda item: item.id)]


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
                default="rewrite_fake_end_to_end.v1",
                choices=(
                    "rewrite_fake_end_to_end.v1",
                    "rewrite_full_capture.v1",
                    "rewrite_foundationpose_runtime.v1",
                    "rewrite_calibration_validation.v1",
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
            PipelineParameter(name="run_config", flag="--run-config", kind="path"),
            PipelineParameter(
                name="max_frames",
                flag="--max-frames",
                kind="int",
                help="Optional max frame count to include in planned camera commands.",
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
            "Validate capture_plan.json command shape, fake/real robot safety, "
            "script availability, and optional sensor readiness."
        ),
        resources=("disk_io",),
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
            "Select capture-plan commands for a safe execution mode without "
            "starting robot or camera processes."
        ),
        resources=("disk_io",),
        parameters=(
            PipelineParameter(
                name="mode",
                flag="--mode",
                default="pose_only_fake",
                choices=("plan_only", "pose_only_fake", "full"),
            ),
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
            "supervision. Defaults to pose-only fake iiwa execution."
        ),
        resources=("robot_command", "camera", "disk_io"),
        parameters=(
            PipelineParameter(
                name="mode",
                flag="--mode",
                default="pose_only_fake",
                choices=("plan_only", "pose_only_fake", "full"),
            ),
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
                default=30.0,
            ),
            PipelineParameter(
                name="startup_wait_s",
                flag="--startup-wait",
                kind="float",
                default=0.2,
            ),
            PipelineParameter(
                name="terminate_timeout_s",
                flag="--terminate-timeout-s",
                kind="float",
                default=2.0,
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
    "synthetic_rgbd_fixture": PipelineStageSpec(
        id="synthetic_rgbd_fixture",
        label="Synthetic RGB-D Fixture",
        script="scripts/create_synthetic_rgbd_fixture.py",
        description=(
            "Write a small synthetic RGB-D sensor folder aligned to existing "
            "raw robot poses so hardware-free runs can exercise sync and BOP export."
        ),
        resources=("disk_io",),
        parameters=(
            PipelineParameter(name="sensor_folder", flag="--sensor-folder"),
            PipelineParameter(name="sensor_id", flag="--sensor-id"),
            PipelineParameter(name="frame_count", flag="--frame-count", kind="int"),
            PipelineParameter(name="width", flag="--width", kind="int"),
            PipelineParameter(name="height", flag="--height", kind="int"),
            PipelineParameter(
                name="sync_delta_ms",
                flag="--sync-delta-ms",
                kind="float",
                default=100.0,
            ),
            PipelineParameter(
                name="include_end_motion",
                flag="--include-end-motion",
                kind="bool",
                default=False,
            ),
            PipelineParameter(
                name="overwrite",
                flag="--overwrite",
                kind="bool",
                default=False,
            ),
            PipelineParameter(name="json", flag="--json", kind="bool", default=False),
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
            PipelineParameter(name="target_spec", flag="--target-spec"),
            PipelineParameter(name="target_type", flag="--target-type"),
            PipelineParameter(name="dictionary", flag="--dictionary"),
            PipelineParameter(name="grid_size", flag="--grid-size"),
            PipelineParameter(
                name="marker_length_mm",
                flag="--marker-length-mm",
                kind="float",
            ),
            PipelineParameter(
                name="marker_separation_mm",
                flag="--marker-separation-mm",
                kind="float",
            ),
            PipelineParameter(
                name="square_length_mm",
                flag="--square-length-mm",
                kind="float",
            ),
            PipelineParameter(name="checkerboard_size", flag="--checkerboard-size"),
            PipelineParameter(name="json", flag="--json", kind="bool", default=False),
        ),
    ),
    "calibration_candidates": PipelineStageSpec(
        id="calibration_candidates",
        label="Calibration Candidates",
        script="scripts/run_calibration_candidates.py",
        description=(
            "Average calibration observations into validation-gated calibration "
            "profile candidates."
        ),
        resources=("disk_io",),
        parameters=(
            PipelineParameter(name="observations", flag="--observations", kind="path"),
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
        label="Calibration Solver",
        script="scripts/run_calibration_solver.py",
        description=(
            "Solve needs-validation calibration profiles from calibration "
            "observations. Eye-in-hand sensors use OpenCV hand-eye solving; "
            "static sensors use target/reference transform consistency."
        ),
        resources=("disk_io",),
        parameters=(
            PipelineParameter(name="observations", flag="--observations", kind="path"),
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
            PipelineParameter(name="candidates", flag="--candidates", kind="path"),
            PipelineParameter(name="profiles", flag="--profiles", kind="path"),
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
            ),
            PipelineParameter(name="operator", flag="--operator"),
            PipelineParameter(name="json", flag="--json", kind="bool", default=False),
        ),
    ),
    "capture_rehearsal": PipelineStageSpec(
        id="capture_rehearsal",
        label="Fake Pose Capture Rehearsal",
        script="scripts/run_capture_rehearsal_stage.py",
        description=(
            "Run fake iiwa plus the pose receiver to produce raw robot poses "
            "without starting camera hardware."
        ),
        resources=("robot_command", "disk_io"),
        parameters=(
            PipelineParameter(name="run_config", flag="--run-config", kind="path"),
            PipelineParameter(
                name="duration_s",
                flag="--duration",
                kind="float",
                default=0.3,
            ),
            PipelineParameter(
                name="sample_ms",
                flag="--sample-ms",
                kind="float",
                default=25.0,
            ),
            PipelineParameter(
                name="startup_delay_s",
                flag="--startup-delay",
                kind="float",
                default=0.0,
            ),
            PipelineParameter(
                name="timeout_s",
                flag="--timeout-s",
                kind="float",
                default=10.0,
            ),
            PipelineParameter(name="robot_port", flag="--robot-port", kind="int"),
            PipelineParameter(name="receiver_port", flag="--receiver-port", kind="int"),
            PipelineParameter(name="robot_ip", flag="--robot-ip"),
            PipelineParameter(name="receiver_ip", flag="--receiver-ip"),
            PipelineParameter(
                name="controller_startup_wait_s",
                flag="--controller-startup-wait",
                kind="float",
                default=0.2,
            ),
            PipelineParameter(
                name="print_json",
                flag="--print-json",
                kind="bool",
                default=False,
            ),
        ),
    ),
    "sync_run": PipelineStageSpec(
        id="sync_run",
        label="Non-destructive Sync",
        script="scripts/sync_run_non_destructive.py",
        description=(
            "Synchronize all raw sensor folders into processed/synchronized "
            "without modifying raw captures."
        ),
        resources=("disk_io",),
        parameters=(
            PipelineParameter(
                name="output_root",
                flag="--output-root",
                kind="path",
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
                default="host_received",
                choices=("host_received", "host_wall", "sensor", "filename"),
                help="Timestamp source used for frame-to-robot matching.",
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
                default=50.0,
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
            PipelineParameter(name="json", flag="--json", kind="bool", default=False),
        ),
    ),
    "aruco": PipelineStageSpec(
        id="aruco",
        label="ArUco Pose Estimation",
        script="scripts/run_aruco_stage.py",
        description="Run ArUco pose estimation on synchronized sensor folders.",
        resources=("cpu",),
        parameters=(
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
            PipelineParameter(name="input_folder", flag="--input-folder", kind="path"),
            PipelineParameter(
                name="object_folder",
                flag="--object-folder",
                kind="path",
                default="object_models",
            ),
            PipelineParameter(
                name="camera_transformations",
                flag="--camera-transformations",
                kind="path",
            ),
            PipelineParameter(
                name="calibration_profiles",
                flag="--calibration-profiles",
                kind="path",
            ),
            PipelineParameter(name="subdir", flag="--subdir"),
        ),
    ),
    "blenderproc_render": PipelineStageSpec(
        id="blenderproc_render",
        label="BlenderProc Render",
        script="scripts/run_blenderproc_render_stage.py",
        description=(
            "Validate or execute BlenderProc rendering for prepared synchronized "
            "sensor folders."
        ),
        resources=("render", "disk_io"),
        parameters=(
            PipelineParameter(name="input_folder", flag="--input-folder", kind="path"),
            PipelineParameter(
                name="render_script",
                flag="--render-script",
                kind="path",
            ),
            PipelineParameter(name="subdir", flag="--subdir"),
            PipelineParameter(name="blenderproc", flag="--blenderproc"),
            PipelineParameter(
                name="dry_run",
                flag="--dry-run",
                kind="bool",
                default=True,
                help="Write a render plan without executing BlenderProc.",
            ),
        ),
    ),
    "foundationpose": PipelineStageSpec(
        id="foundationpose",
        label="FoundationPose",
        script="scripts/run_foundationpose_stage.py",
        description=(
            "Plan or run FoundationPose on synchronized sensor folders through "
            "the legacy Docker wrapper."
        ),
        resources=("estimator", "disk_io"),
        parameters=(
            PipelineParameter(name="input_folder", flag="--input-folder", kind="path"),
            PipelineParameter(
                name="foundationpose_folder",
                flag="--foundationpose-folder",
                kind="path",
            ),
            PipelineParameter(name="no_tracking", flag="--no-tracking", kind="bool"),
            PipelineParameter(name="est_refine_iter", flag="--est-refine-iter", kind="int"),
            PipelineParameter(
                name="track_refine_iter",
                flag="--track-refine-iter",
                kind="int",
            ),
            PipelineParameter(name="object_id", flag="--object-id", kind="int"),
            PipelineParameter(name="run_level", flag="--run-level", kind="bool"),
            PipelineParameter(
                name="dry_run",
                flag="--dry-run",
                kind="bool",
                default=True,
                help="Write a FoundationPose plan without starting Docker.",
            ),
        ),
    ),
    "megapose": PipelineStageSpec(
        id="megapose",
        label="MegaPose",
        script="scripts/run_megapose_stage.py",
        description=(
            "Plan or run MegaPose on synchronized sensor folders through a "
            "configured legacy wrapper."
        ),
        resources=("estimator", "disk_io"),
        parameters=(
            PipelineParameter(name="input_folder", flag="--input-folder", kind="path"),
            PipelineParameter(name="wrapper_script", flag="--wrapper-script", kind="path"),
            PipelineParameter(name="model", flag="--model"),
            PipelineParameter(name="roi_scale", flag="--roi-scale", kind="float"),
            PipelineParameter(name="object_id", flag="--object-id", kind="int"),
            PipelineParameter(name="result_id", flag="--result-id"),
            PipelineParameter(
                name="dry_run",
                flag="--dry-run",
                kind="bool",
                default=True,
                help="Write a MegaPose plan without starting the wrapper.",
            ),
        ),
    ),
    "sam6d": PipelineStageSpec(
        id="sam6d",
        label="SAM6D",
        script="scripts/run_sam6d_stage.py",
        description=(
            "Plan or run SAM6D on synchronized sensor folders through a "
            "configured legacy wrapper."
        ),
        resources=("estimator", "disk_io"),
        parameters=(
            PipelineParameter(name="input_folder", flag="--input-folder", kind="path"),
            PipelineParameter(name="wrapper_script", flag="--wrapper-script", kind="path"),
            PipelineParameter(name="segmentor_model", flag="--segmentor-model"),
            PipelineParameter(name="object_id", flag="--object-id", kind="int"),
            PipelineParameter(name="result_id", flag="--result-id"),
            PipelineParameter(
                name="dry_run",
                flag="--dry-run",
                kind="bool",
                default=True,
                help="Write a SAM6D plan without starting the wrapper.",
            ),
        ),
    ),
    "bop_export": PipelineStageSpec(
        id="bop_export",
        label="BOP Dataset Export",
        script="scripts/run_bop_export_stage.py",
        description="Export synchronized sensor folders into BOP scene folders.",
        resources=("disk_io",),
        parameters=(
            PipelineParameter(name="input_folder", flag="--input-folder", kind="path"),
            PipelineParameter(name="output_folder", flag="--output-folder", kind="path"),
            PipelineParameter(name="split", flag="--split"),
            PipelineParameter(
                name="object_folder",
                flag="--object-folder",
                kind="path",
                default="object_models",
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
            ),
            PipelineParameter(
                name="write_multiview_targets",
                flag="--write-multiview-targets",
                kind="bool",
                default=False,
            ),
            PipelineParameter(
                name="write_coco_annotations",
                flag="--write-coco-annotations",
                kind="bool",
                default=False,
            ),
        ),
    ),
    "bop_result_export": PipelineStageSpec(
        id="bop_result_export",
        label="BOP Result Export",
        script="scripts/run_bop_result_export_stage.py",
        description="Convert estimator outputs into BOP19 result CSV files.",
        resources=("disk_io",),
        parameters=(
            PipelineParameter(
                name="source",
                flag="--source",
                choices=("foundationpose", "aruco", "megapose", "sam6d"),
            ),
            PipelineParameter(name="input_folder", flag="--input-folder", kind="path"),
            PipelineParameter(
                name="foundationpose_output",
                flag="--foundationpose-output",
                kind="path",
                multiple=True,
            ),
            PipelineParameter(
                name="aruco_pose_file",
                flag="--aruco-pose-file",
                kind="path",
                multiple=True,
            ),
            PipelineParameter(
                name="megapose_output",
                flag="--megapose-output",
                kind="path",
                multiple=True,
            ),
            PipelineParameter(
                name="sam6d_output",
                flag="--sam6d-output",
                kind="path",
                multiple=True,
            ),
            PipelineParameter(
                name="aruco_object_name",
                flag="--aruco-object-name",
            ),
            PipelineParameter(
                name="min_marker_count",
                flag="--min-marker-count",
                kind="int",
            ),
            PipelineParameter(name="bop_root", flag="--bop-root", kind="path"),
            PipelineParameter(name="output_folder", flag="--output-folder", kind="path"),
            PipelineParameter(name="dataset_name", flag="--dataset-name"),
            PipelineParameter(
                name="default_score",
                flag="--default-score",
                kind="float",
            ),
            PipelineParameter(
                name="default_time",
                flag="--default-time",
                kind="float",
            ),
            PipelineParameter(
                name="translation_scale_to_mm",
                flag="--translation-scale-to-mm",
                kind="float",
            ),
        ),
    ),
    "synthetic_bop_results": PipelineStageSpec(
        id="synthetic_bop_results",
        label="Synthetic BOP Results",
        script="scripts/create_synthetic_bop_results.py",
        description=(
            "Write deterministic BOP19 result CSV rows from the BOP export "
            "manifest for hardware-free rewrite validation."
        ),
        resources=("disk_io",),
        parameters=(
            PipelineParameter(name="bop_root", flag="--bop-root", kind="path"),
            PipelineParameter(name="output_folder", flag="--output-folder", kind="path"),
            PipelineParameter(name="dataset_name", flag="--dataset-name"),
            PipelineParameter(name="method", flag="--method", default="synthetic"),
            PipelineParameter(name="object_name", flag="--object-name"),
            PipelineParameter(name="score", flag="--score", kind="float", default=1.0),
            PipelineParameter(name="time", flag="--time", kind="float", default=-1.0),
            PipelineParameter(name="json", flag="--json", kind="bool", default=False),
        ),
    ),
    "bop_evaluation": PipelineStageSpec(
        id="bop_evaluation",
        label="BOP Toolkit Evaluation",
        script="scripts/run_bop_evaluation_stage.py",
        description=(
            "Validate a BOP19 result CSV and plan or run BOP Toolkit evaluation."
        ),
        resources=("evaluation",),
        parameters=(
            PipelineParameter(
                name="result_file",
                flag="--result-file",
                kind="path",
                required=True,
            ),
            PipelineParameter(name="bop_root", flag="--bop-root", kind="path"),
            PipelineParameter(name="bop_path", flag="--bop-path", kind="path"),
            PipelineParameter(name="eval_path", flag="--eval-path", kind="path"),
            PipelineParameter(name="targets_filename", flag="--targets-filename"),
            PipelineParameter(
                name="bop_toolkit_root",
                flag="--bop-toolkit-root",
                kind="path",
            ),
            PipelineParameter(name="eval_script", flag="--eval-script", kind="path"),
            PipelineParameter(
                name="python_executable",
                flag="--python-executable",
                kind="path",
            ),
            PipelineParameter(name="renderer_type", flag="--renderer-type"),
            PipelineParameter(name="num_workers", flag="--num-workers", kind="int"),
            PipelineParameter(name="use_gpu", flag="--use-gpu", kind="bool"),
            PipelineParameter(name="device", flag="--device"),
            PipelineParameter(
                name="cleanup_eval",
                flag="--cleanup-eval",
                kind="bool",
                default=False,
            ),
            PipelineParameter(
                name="dry_run",
                flag="--dry-run",
                kind="bool",
                default=True,
                help="Write the evaluation plan without executing BOP Toolkit.",
            ),
        ),
    ),
    "metric_report_export": PipelineStageSpec(
        id="metric_report_export",
        label="Metric Report Export",
        script="scripts/run_metric_report_export_stage.py",
        description=(
            "Export discovered legacy metric artifacts as JSON, CSV, and XLSX "
            "reports under results/metrics."
        ),
        resources=("disk_io",),
        parameters=(
            PipelineParameter(name="output_folder", flag="--output-folder", kind="path"),
            PipelineParameter(
                name="group_limit",
                flag="--group-limit",
                kind="int",
                default=200,
            ),
            PipelineParameter(name="json", flag="--json", kind="bool", default=False),
        ),
    ),
}
