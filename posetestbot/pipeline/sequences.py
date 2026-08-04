"""Dependency-aware pipeline sequence planning for PoseTestBot runs."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from posetestbot.io.atomic import atomic_write_json
from posetestbot.io.artifacts import PIPELINE_SEQUENCE_PLAN
from posetestbot.pipeline.stages import (
    PIPELINE_STAGES,
    PipelineStageSpec,
    build_pipeline_job,
)


SCHEMA_VERSION = "pipeline_sequence_plan.v1"
SEQUENCE_EXECUTION_ACK_ENV = "POSETESTBOT_SEQUENCE_EXECUTION_ACKNOWLEDGEMENTS"
EXECUTION_ACKNOWLEDGEMENT_KEYS = frozenset(
    {"allow_cameras", "allow_real_robot"}
)


@dataclass(frozen=True)
class PipelineSequenceStepSpec:
    """A stage invocation inside a named pipeline sequence."""

    id: str
    stage_id: str
    depends_on: tuple[str, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["depends_on"] = list(self.depends_on)
        data["options"] = dict(self.options)
        return data


@dataclass(frozen=True)
class PipelineSequenceSpec:
    """A reusable, dependency-aware workflow made of typed stage presets."""

    id: str
    label: str
    description: str
    steps: tuple[PipelineSequenceStepSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class PipelineSequenceStepPlan:
    id: str
    stage_id: str
    stage_label: str
    depends_on: list[str]
    command: list[str]
    resources: list[str]
    options: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PipelineSequencePlan:
    schema_version: str
    sequence_id: str
    sequence_label: str
    run_root: str
    plan_only: bool
    steps: list[PipelineSequenceStepPlan]
    resources: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sequence_id": self.sequence_id,
            "sequence_label": self.sequence_label,
            "run_root": self.run_root,
            "plan_only": self.plan_only,
            "steps": [step.to_dict() for step in self.steps],
            "resources": list(self.resources),
        }


@dataclass(frozen=True)
class PipelineSequenceJobSpec:
    sequence_id: str
    sequence_label: str
    run_root: str
    command: list[str]
    resources: list[str]
    parameters: dict[str, Any]
    plan: PipelineSequencePlan
    execution_environment: dict[str, str] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "sequence_label": self.sequence_label,
            "run_root": self.run_root,
            "command": list(self.command),
            "resources": list(self.resources),
            "parameters": dict(self.parameters),
            "plan": sequence_plan_without_acknowledgements(self.plan).to_dict(),
        }


def _stringify_path(value: str | Path) -> str:
    return value.as_posix() if isinstance(value, Path) else str(value)


def _sequence_specs(
    registry: Mapping[str, PipelineSequenceSpec] | None,
) -> Mapping[str, PipelineSequenceSpec]:
    return PIPELINE_SEQUENCES if registry is None else registry


def list_pipeline_sequences(
    registry: Mapping[str, PipelineSequenceSpec] | None = None,
) -> list[dict[str, Any]]:
    specs = _sequence_specs(registry)
    return [sequence.to_dict() for sequence in sorted(specs.values(), key=lambda item: item.id)]


def get_pipeline_sequence(
    sequence_id: str,
    registry: Mapping[str, PipelineSequenceSpec] | None = None,
) -> PipelineSequenceSpec:
    specs = _sequence_specs(registry)
    try:
        return specs[sequence_id]
    except KeyError as exc:
        raise ValueError(f"Unknown pipeline sequence: {sequence_id}") from exc


def _validate_sequence_spec(
    sequence: PipelineSequenceSpec,
    *,
    stage_registry: Mapping[str, PipelineStageSpec],
) -> None:
    if not sequence.steps:
        raise ValueError(f"Pipeline sequence {sequence.id} has no steps")

    step_ids = [step.id for step in sequence.steps]
    duplicate_step_ids = sorted(
        step_id for step_id in set(step_ids) if step_ids.count(step_id) > 1
    )
    if duplicate_step_ids:
        raise ValueError(
            f"Pipeline sequence {sequence.id} has duplicate step IDs: "
            f"{', '.join(duplicate_step_ids)}"
        )

    known_step_ids = set(step_ids)
    for step in sequence.steps:
        if step.stage_id not in stage_registry:
            raise ValueError(
                f"Pipeline sequence {sequence.id} references unknown stage: "
                f"{step.stage_id}"
            )
        unknown_dependencies = sorted(set(step.depends_on) - known_step_ids)
        if unknown_dependencies:
            raise ValueError(
                f"Pipeline sequence step {step.id} depends on unknown step(s): "
                f"{', '.join(unknown_dependencies)}"
            )


def _topological_steps(
    sequence: PipelineSequenceSpec,
    *,
    stage_registry: Mapping[str, PipelineStageSpec],
) -> list[PipelineSequenceStepSpec]:
    _validate_sequence_spec(sequence, stage_registry=stage_registry)

    remaining = {step.id: step for step in sequence.steps}
    ordered: list[PipelineSequenceStepSpec] = []
    completed: set[str] = set()

    while remaining:
        progressed = False
        for step in sequence.steps:
            if step.id not in remaining:
                continue
            if all(dependency in completed for dependency in step.depends_on):
                ordered.append(step)
                completed.add(step.id)
                del remaining[step.id]
                progressed = True
        if not progressed:
            cycle = ", ".join(sorted(remaining))
            raise ValueError(
                f"Pipeline sequence {sequence.id} has cyclic dependencies: {cycle}"
            )

    return ordered


def _normalize_sequence_option_groups(
    sequence: PipelineSequenceSpec,
    options: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    provided = dict(options or {})
    valid_groups = {step.id for step in sequence.steps}
    valid_groups.update(step.stage_id for step in sequence.steps)
    unknown_groups = sorted(set(provided) - valid_groups)
    if unknown_groups:
        raise ValueError(
            "Unknown pipeline sequence option group(s) for "
            f"{sequence.id}: {', '.join(unknown_groups)}"
        )

    normalized: dict[str, Mapping[str, Any]] = {}
    for key, value in provided.items():
        if not isinstance(value, Mapping):
            raise ValueError(
                f"Pipeline sequence options for {key!r} must be a JSON object"
            )
        normalized[key] = value
    return normalized


def _without_execution_acknowledgements(
    options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Remove one-shot motion/camera approvals from a reusable plan request."""

    sanitized: dict[str, Any] = {}
    for group, value in dict(options or {}).items():
        if isinstance(value, Mapping):
            sanitized[group] = {
                key: item
                for key, item in value.items()
                if key not in EXECUTION_ACKNOWLEDGEMENT_KEYS
            }
        else:
            sanitized[group] = value
    return sanitized


def _execution_acknowledgements_only(
    options: Mapping[str, Any] | None,
) -> dict[str, dict[str, bool]]:
    acknowledgements: dict[str, dict[str, bool]] = {}
    for group, value in dict(options or {}).items():
        if not isinstance(value, Mapping):
            continue
        selected = {
            key: True
            for key, item in value.items()
            if key in EXECUTION_ACKNOWLEDGEMENT_KEYS and item is True
        }
        if selected:
            acknowledgements[str(group)] = selected
    return acknowledgements


def validate_sequence_execution_options(
    *,
    sequence_id: str,
    options: Mapping[str, Any] | None,
) -> None:
    """Require literal booleans for each one-shot real-capture gate."""

    if sequence_id != "real_full_capture_validation":
        return
    provided = dict(options or {})
    required = {
        "capture_plan_preflight": ("allow_real_robot",),
        "capture_execution_plan": ("allow_cameras", "allow_real_robot"),
        "capture_execution": ("allow_cameras", "allow_real_robot"),
    }
    missing = []
    for group, names in required.items():
        values = provided.get(group)
        if not isinstance(values, Mapping):
            missing.extend(f"{group}.{name}" for name in names)
            continue
        missing.extend(
            f"{group}.{name}" for name in names if values.get(name) is not True
        )
    if missing:
        raise ValueError(
            "Real capture sequence execution requires fresh literal-true "
            "per-step acknowledgements: " + ", ".join(missing) + "."
        )


def sequence_plan_without_acknowledgements(
    plan: PipelineSequencePlan,
) -> PipelineSequencePlan:
    """Return reusable evidence with one-shot gates removed from options/commands."""

    steps = []
    for step in plan.steps:
        steps.append(
            PipelineSequenceStepPlan(
                id=step.id,
                stage_id=step.stage_id,
                stage_label=step.stage_label,
                depends_on=list(step.depends_on),
                command=[
                    item
                    for item in step.command
                    if item not in {"--allow-cameras", "--allow-real-robot"}
                ],
                resources=list(step.resources),
                options={
                    key: value
                    for key, value in step.options.items()
                    if key not in EXECUTION_ACKNOWLEDGEMENT_KEYS
                },
            )
        )
    return PipelineSequencePlan(
        schema_version=plan.schema_version,
        sequence_id=plan.sequence_id,
        sequence_label=plan.sequence_label,
        run_root=plan.run_root,
        plan_only=plan.plan_only,
        steps=steps,
        resources=list(plan.resources),
    )


def _resolve_option_placeholders(value: Any, *, run_root: str) -> Any:
    if isinstance(value, str):
        return value.replace("{run_root}", run_root)
    if isinstance(value, list):
        return [
            _resolve_option_placeholders(item, run_root=run_root)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _resolve_option_placeholders(item, run_root=run_root)
            for item in value
        )
    if isinstance(value, Mapping):
        return {
            key: _resolve_option_placeholders(item, run_root=run_root)
            for key, item in value.items()
        }
    return value


def _step_options(
    step: PipelineSequenceStepSpec,
    sequence_options: Mapping[str, Mapping[str, Any]],
    *,
    run_root: str,
) -> dict[str, Any]:
    options = dict(step.options)
    if step.stage_id in sequence_options:
        options.update(dict(sequence_options[step.stage_id]))
    if step.id in sequence_options:
        options.update(dict(sequence_options[step.id]))
    return _resolve_option_placeholders(options, run_root=run_root)


def build_sequence_plan(
    *,
    sequence_id: str,
    run_root: str | Path,
    options: Mapping[str, Any] | None = None,
    plan_only: bool = False,
    sequence_registry: Mapping[str, PipelineSequenceSpec] | None = None,
    stage_registry: Mapping[str, PipelineStageSpec] | None = None,
) -> PipelineSequencePlan:
    stage_specs = PIPELINE_STAGES if stage_registry is None else stage_registry
    sequence = get_pipeline_sequence(sequence_id, registry=sequence_registry)
    run_root_value = _stringify_path(run_root)
    if not run_root_value:
        raise ValueError("Pipeline sequence run_root must not be empty")

    ordered_steps = _topological_steps(sequence, stage_registry=stage_specs)
    effective_options = (
        _without_execution_acknowledgements(options) if plan_only else options
    )
    sequence_options = _normalize_sequence_option_groups(sequence, effective_options)
    step_plans: list[PipelineSequenceStepPlan] = []
    resource_set: set[str] = set()

    for step in ordered_steps:
        job = build_pipeline_job(
            stage_id=step.stage_id,
            run_root=run_root_value,
            options=_step_options(step, sequence_options, run_root=run_root_value),
            registry=stage_specs,
        )
        resource_set.update(job.resources)
        step_plans.append(
            PipelineSequenceStepPlan(
                id=step.id,
                stage_id=job.stage_id,
                stage_label=job.stage_label,
                depends_on=list(step.depends_on),
                command=job.command,
                resources=job.resources,
                options=dict(job.parameters["options"]),
            )
        )

    return PipelineSequencePlan(
        schema_version=SCHEMA_VERSION,
        sequence_id=sequence.id,
        sequence_label=sequence.label,
        run_root=run_root_value,
        plan_only=plan_only,
        steps=step_plans,
        resources=sorted(resource_set),
    )


def write_sequence_plan(
    run_root: str | Path,
    plan: PipelineSequencePlan,
    *,
    filename: str = PIPELINE_SEQUENCE_PLAN,
) -> Path:
    path = Path(run_root) / filename
    return atomic_write_json(
        path,
        sequence_plan_without_acknowledgements(plan).to_dict(),
    )


def execute_sequence_plan(
    plan: PipelineSequencePlan,
    *,
    cwd: str | Path | None = None,
) -> list[subprocess.CompletedProcess]:
    validate_sequence_execution_plan(plan)
    completed = set()
    results: list[subprocess.CompletedProcess] = []
    for step in plan.steps:
        missing_dependencies = sorted(set(step.depends_on) - completed)
        if missing_dependencies:
            raise RuntimeError(
                f"Pipeline sequence step {step.id} cannot run before: "
                f"{', '.join(missing_dependencies)}"
            )
        result = subprocess.run(step.command, cwd=cwd, check=True)
        results.append(result)
        completed.add(step.id)
    return results


def validate_sequence_execution_plan(
    plan: PipelineSequencePlan,
) -> None:
    if plan.sequence_id != "real_full_capture_validation":
        return
    required = {
        "capture_plan_preflight": ("allow_real_robot",),
        "capture_execution_plan": ("allow_cameras", "allow_real_robot"),
        "capture_execution": ("allow_cameras", "allow_real_robot"),
    }
    missing: list[str] = []
    for step in plan.steps:
        for acknowledgement in required.get(step.id, ()):
            if step.options.get(acknowledgement) is not True:
                missing.append(f"{step.id}.{acknowledgement}")
    if missing:
        raise ValueError(
            "Real capture sequence execution requires fresh per-step "
            "acknowledgements: " + ", ".join(missing) + "."
        )


def build_sequence_job(
    *,
    sequence_id: str,
    run_root: str | Path,
    options: Mapping[str, Any] | None = None,
    plan_only: bool = False,
    sequence_registry: Mapping[str, PipelineSequenceSpec] | None = None,
    stage_registry: Mapping[str, PipelineStageSpec] | None = None,
) -> PipelineSequenceJobSpec:
    if not plan_only:
        validate_sequence_execution_options(
            sequence_id=sequence_id,
            options=options,
        )
    effective_options = (
        _without_execution_acknowledgements(options)
        if plan_only
        else dict(options or {})
    )
    plan = build_sequence_plan(
        sequence_id=sequence_id,
        run_root=run_root,
        options=effective_options,
        plan_only=plan_only,
        sequence_registry=sequence_registry,
        stage_registry=stage_registry,
    )
    if not plan_only:
        validate_sequence_execution_plan(plan)
    persisted_options = _without_execution_acknowledgements(effective_options)
    options_json = json.dumps(persisted_options, sort_keys=True)
    command = [
        "uv",
        "run",
        "python",
        "scripts/run_pipeline_sequence.py",
        plan.run_root,
        "--sequence",
        plan.sequence_id,
        "--options-json",
        options_json,
    ]
    if plan_only:
        command.append("--plan-only")
    job_resources = ["disk_io"] if plan_only else list(plan.resources)
    evidence_plan = sequence_plan_without_acknowledgements(plan)
    parameters = {
        "pipeline_sequence": plan.sequence_id,
        "sequence_label": plan.sequence_label,
        "run_root": plan.run_root,
        "plan_only": plan_only,
        "locked_resources": list(job_resources),
        "planned_resources": list(plan.resources),
        "options": persisted_options,
        "steps": [step.to_dict() for step in evidence_plan.steps],
        "execution_acknowledgements": "validated_ephemeral",
    }
    return PipelineSequenceJobSpec(
        sequence_id=plan.sequence_id,
        sequence_label=plan.sequence_label,
        run_root=plan.run_root,
        command=command,
        resources=job_resources,
        parameters=parameters,
        plan=plan,
        execution_environment=(
            {
                SEQUENCE_EXECUTION_ACK_ENV: json.dumps(
                    _execution_acknowledgements_only(effective_options),
                    sort_keys=True,
                )
            }
            if not plan_only and sequence_id == "real_full_capture_validation"
            else {}
        ),
    )


PIPELINE_SEQUENCES: dict[str, PipelineSequenceSpec] = {
    "real_full_capture_validation": PipelineSequenceSpec(
        id="real_full_capture_validation",
        label="Real Full Capture Validation",
        description=(
            "Write run preflight and hardware snapshots, write and preflight "
            "the capture command plan, explicitly select real robot plus camera "
            "commands, run supervised full capture, then audit "
            "rewrite_full_capture.v1. This sequence is intended for an "
            "operator-controlled lab run."
        ),
        steps=(
            PipelineSequenceStepSpec(
                id="run_preflight",
                stage_id="run_preflight",
                options={"write": True, "check": True},
            ),
            PipelineSequenceStepSpec(
                id="hardware_status",
                stage_id="hardware_status",
                depends_on=("run_preflight",),
            ),
            PipelineSequenceStepSpec(
                id="capture_plan",
                stage_id="capture_plan",
                depends_on=("hardware_status",),
                options={"warmup_frames": 30},
            ),
            PipelineSequenceStepSpec(
                id="capture_plan_preflight",
                stage_id="capture_plan_preflight",
                depends_on=("capture_plan",),
            ),
            PipelineSequenceStepSpec(
                id="capture_execution_plan",
                stage_id="capture_execution_plan",
                depends_on=("capture_plan_preflight",),
                options={
                    "include_sensors": True,
                },
            ),
            PipelineSequenceStepSpec(
                id="capture_execution",
                stage_id="capture_execution",
                depends_on=("capture_execution_plan",),
                options={
                    "include_sensors": True,
                    "timeout_s": 720.0,
                    "startup_wait_s": 15.0,
                    "receive_start_timeout_s": 120.0,
                    "receive_idle_timeout_s": 60.0,
                    "camera_metadata_idle_timeout_s": 5.0,
                },
            ),
            PipelineSequenceStepSpec(
                id="rewrite_full_capture_gate",
                stage_id="rewrite_gate",
                depends_on=("capture_execution",),
                options={"gate": "rewrite_full_capture.v1", "write": True},
            ),
        ),
    ),
    "sync_aruco": PipelineSequenceSpec(
        id="sync_aruco",
        label="Synchronize And Run ArUco",
        description=(
            "Run non-destructive synchronization, then run ArUco pose estimation "
            "on the synchronized sensor folders."
        ),
        steps=(
            PipelineSequenceStepSpec(id="sync_run", stage_id="sync_run"),
            PipelineSequenceStepSpec(
                id="sync_quality",
                stage_id="sync_quality",
                depends_on=("sync_run",),
            ),
            PipelineSequenceStepSpec(
                id="aruco",
                stage_id="aruco",
                depends_on=("sync_quality",),
            ),
        ),
    ),
    "sync_aruco_calibration_observations": PipelineSequenceSpec(
        id="sync_aruco_calibration_observations",
        label="Synchronize ArUco Calibration Observations",
        description=(
            "Run non-destructive synchronization, sync quality checks, ArUco "
            "pose estimation, then extract solver-ready calibration "
            "observations."
        ),
        steps=(
            PipelineSequenceStepSpec(id="sync_run", stage_id="sync_run"),
            PipelineSequenceStepSpec(
                id="sync_quality",
                stage_id="sync_quality",
                depends_on=("sync_run",),
            ),
            PipelineSequenceStepSpec(
                id="aruco",
                stage_id="aruco",
                depends_on=("sync_quality",),
            ),
            PipelineSequenceStepSpec(
                id="calibration_observations",
                stage_id="calibration_observations",
                depends_on=("aruco",),
            ),
        ),
    ),
    "sync_aruco_calibration_candidates": PipelineSequenceSpec(
        id="sync_aruco_calibration_candidates",
        label="Legacy Fixed-Target Calibration Candidates",
        description=(
            "Run synchronization, sync quality checks, ArUco pose estimation, "
            "calibration observation extraction, then generate compatibility "
            "candidates. Static robot-carried-grid calibration belongs to guided "
            "Workflow step 5."
        ),
        steps=(
            PipelineSequenceStepSpec(id="sync_run", stage_id="sync_run"),
            PipelineSequenceStepSpec(
                id="sync_quality",
                stage_id="sync_quality",
                depends_on=("sync_run",),
            ),
            PipelineSequenceStepSpec(
                id="aruco",
                stage_id="aruco",
                depends_on=("sync_quality",),
            ),
            PipelineSequenceStepSpec(
                id="calibration_observations",
                stage_id="calibration_observations",
                depends_on=("aruco",),
            ),
            PipelineSequenceStepSpec(
                id="calibration_candidates",
                stage_id="calibration_candidates",
                depends_on=("calibration_observations",),
            ),
        ),
    ),
    "sync_aruco_calibration_solver": PipelineSequenceSpec(
        id="sync_aruco_calibration_solver",
        label="Legacy Fixed-Target Calibration Solver",
        description=(
            "Run synchronization, sync quality checks, ArUco pose estimation, "
            "calibration observation extraction, then run the compatibility "
            "solver. It is not the robot-carried-grid static-camera joint solve."
        ),
        steps=(
            PipelineSequenceStepSpec(id="sync_run", stage_id="sync_run"),
            PipelineSequenceStepSpec(
                id="sync_quality",
                stage_id="sync_quality",
                depends_on=("sync_run",),
            ),
            PipelineSequenceStepSpec(
                id="aruco",
                stage_id="aruco",
                depends_on=("sync_quality",),
            ),
            PipelineSequenceStepSpec(
                id="calibration_observations",
                stage_id="calibration_observations",
                depends_on=("aruco",),
            ),
            PipelineSequenceStepSpec(
                id="calibration_solver",
                stage_id="calibration_solver",
                depends_on=("calibration_observations",),
            ),
        ),
    ),
    "sync_aruco_calibration_validation": PipelineSequenceSpec(
        id="sync_aruco_calibration_validation",
        label="Legacy Fixed-Target Calibration Validation",
        description=(
            "Run synchronization, sync quality, ArUco pose estimation, "
            "observation extraction, compatibility candidate generation, then "
            "validate candidates without promoting them. Static profiles require "
            "canonical PoseTemplateBase provenance."
        ),
        steps=(
            PipelineSequenceStepSpec(id="sync_run", stage_id="sync_run"),
            PipelineSequenceStepSpec(
                id="sync_quality",
                stage_id="sync_quality",
                depends_on=("sync_run",),
            ),
            PipelineSequenceStepSpec(
                id="aruco",
                stage_id="aruco",
                depends_on=("sync_quality",),
            ),
            PipelineSequenceStepSpec(
                id="calibration_observations",
                stage_id="calibration_observations",
                depends_on=("aruco",),
            ),
            PipelineSequenceStepSpec(
                id="calibration_candidates",
                stage_id="calibration_candidates",
                depends_on=("calibration_observations",),
            ),
            PipelineSequenceStepSpec(
                id="calibration_validation",
                stage_id="calibration_validation",
                depends_on=("calibration_candidates",),
            ),
        ),
    ),
    "sync_to_bop_dry_run": PipelineSequenceSpec(
        id="sync_to_bop_dry_run",
        label="Synchronize To BOP Dataset",
        description=(
            "Run non-destructive synchronization and sync quality checks, then "
            "write an annotation-free BOP image/model dataset without BlenderProc."
        ),
        steps=(
            PipelineSequenceStepSpec(id="sync_run", stage_id="sync_run"),
            PipelineSequenceStepSpec(
                id="sync_quality",
                stage_id="sync_quality",
                depends_on=("sync_run",),
            ),
            PipelineSequenceStepSpec(
                id="bop_export",
                stage_id="bop_export",
                depends_on=("sync_quality",),
                options={"annotation_source": "none", "overwrite": True},
            ),
        ),
    ),
    "sync_to_bop_calibrated_dry_run": PipelineSequenceSpec(
        id="sync_to_bop_calibrated_dry_run",
        label="Synchronize To Calibrated BOP Dataset",
        description=(
            "Run non-destructive synchronization, sync quality checks, "
            "and calibration profile preflight, then write an annotation-free "
            "calibrated BOP image/model dataset without BlenderProc."
        ),
        steps=(
            PipelineSequenceStepSpec(id="sync_run", stage_id="sync_run"),
            PipelineSequenceStepSpec(
                id="sync_quality",
                stage_id="sync_quality",
                depends_on=("sync_run",),
            ),
            PipelineSequenceStepSpec(
                id="calibration_preflight",
                stage_id="calibration_preflight",
                depends_on=("sync_quality",),
                options={"require_valid": True},
            ),
            PipelineSequenceStepSpec(
                id="bop_export",
                stage_id="bop_export",
                depends_on=("sync_quality", "calibration_preflight"),
                options={
                    "calibration_profiles": (
                        "{run_root}/calibration_profiles.json"
                    ),
                    "annotation_source": "none",
                    "overwrite": True,
                },
            ),
        ),
    ),
    "capture_to_bop_dataset_dry_run": PipelineSequenceSpec(
        id="capture_to_bop_dataset_dry_run",
        label="Captured Run To BOP Dataset",
        description=(
            "For an existing captured run folder, run non-destructive "
            "synchronization and write an annotation-free BOP image/model "
            "dataset without BlenderProc."
        ),
        steps=(
            PipelineSequenceStepSpec(id="sync_run", stage_id="sync_run"),
            PipelineSequenceStepSpec(
                id="sync_quality",
                stage_id="sync_quality",
                depends_on=("sync_run",),
            ),
            PipelineSequenceStepSpec(
                id="bop_export",
                stage_id="bop_export",
                depends_on=("sync_quality",),
                options={"annotation_source": "none", "overwrite": True},
            ),
        ),
    ),
    "aruco_grid_full_calibration": PipelineSequenceSpec(
        id="aruco_grid_full_calibration",
        label="Legacy Fixed-Target ArUco Grid Calibration",
        description=(
            "Import the exact printed grid, synchronize, calibrate color intrinsics, "
            "solve the fixed-target wrist-camera compatibility methods, and require "
            "explicit validation selection. It is not the static moving-grid workflow."
        ),
        steps=(
            PipelineSequenceStepSpec(
                id="calibration_target_import",
                stage_id="calibration_target_import",
                options={
                    "source": "{run_root}/aruco_grid_config.json",
                    "aligned_to_template_base": True,
                },
            ),
            PipelineSequenceStepSpec(
                id="sync_run",
                stage_id="sync_run",
                depends_on=("calibration_target_import",),
            ),
            PipelineSequenceStepSpec(
                id="sync_quality",
                stage_id="sync_quality",
                depends_on=("sync_run",),
            ),
            PipelineSequenceStepSpec(
                id="aruco_detection",
                stage_id="aruco_detection",
                depends_on=("sync_quality",),
            ),
            PipelineSequenceStepSpec(
                id="intrinsic_calibration",
                stage_id="intrinsic_calibration",
                depends_on=("aruco_detection",),
                options={"mode": "calibrate"},
            ),
            PipelineSequenceStepSpec(
                id="aruco_pose",
                stage_id="aruco_pose",
                depends_on=("intrinsic_calibration",),
            ),
            PipelineSequenceStepSpec(
                id="calibration_observations",
                stage_id="calibration_observations",
                depends_on=("aruco_pose",),
                options={"target_spec": "{run_root}/calibration_target.json"},
            ),
            PipelineSequenceStepSpec(
                id="calibration_solver",
                stage_id="calibration_solver",
                depends_on=("calibration_observations",),
                options={"mode": "compare"},
            ),
            PipelineSequenceStepSpec(
                id="calibration_validation",
                stage_id="calibration_validation",
                depends_on=("calibration_solver",),
            ),
        ),
    ),
    "calibrated_capture_to_bop_dataset_dry_run": PipelineSequenceSpec(
        id="calibrated_capture_to_bop_dataset_dry_run",
        label="Calibrated Capture To BOP Dataset",
        description=(
            "Synchronize a captured run, rectify RGB-D with selected intrinsics, "
            "then export an annotation-free BOP image/model dataset using "
            "promoted calibration.v2 profiles without BlenderProc."
        ),
        steps=(
            PipelineSequenceStepSpec(id="sync_run", stage_id="sync_run"),
            PipelineSequenceStepSpec(
                id="sync_quality",
                stage_id="sync_quality",
                depends_on=("sync_run",),
            ),
            PipelineSequenceStepSpec(
                id="calibration_preflight",
                stage_id="calibration_preflight",
                depends_on=("sync_quality",),
                options={"require_valid": True},
            ),
            PipelineSequenceStepSpec(
                id="camera_rectification",
                stage_id="camera_rectification",
                depends_on=("sync_quality", "calibration_preflight"),
                options={"overwrite": True},
            ),
            PipelineSequenceStepSpec(
                id="bop_export",
                stage_id="bop_export",
                depends_on=("camera_rectification",),
                options={"annotation_source": "none", "overwrite": True},
            ),
        ),
    ),
}
