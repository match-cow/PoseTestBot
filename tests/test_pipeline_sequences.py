from __future__ import annotations

import json

import os

import subprocess

import sys

from pathlib import Path

import pytest

from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    load_run_manifest,
    upsert_stage,
    write_run_manifest,
)

from posetestbot.pipeline.sequences import (
    PIPELINE_SEQUENCES,
    SEQUENCE_EXECUTION_ACK_ENV,
    build_sequence_job,
    build_sequence_plan,
    list_pipeline_sequences,
    write_sequence_plan,
)

from scripts import run_pipeline_sequence


def step_ids(plan) -> list[str]:
    return [step.id for step in plan.steps]


def test_capture_to_bop_dataset_sequence_uses_sync_quality_gate(tmp_path: Path) -> None:
    plan = build_sequence_plan(
        sequence_id="capture_to_bop_dataset_dry_run",
        run_root=tmp_path / "captured-run",
    )

    assert step_ids(plan) == [
        "sync_run",
        "sync_quality",
        "bop_export",
    ]
    assert plan.steps[2].depends_on == ["sync_quality"]
    assert plan.steps[2].options["annotation_source"] == "none"
    assert plan.steps[2].options["overwrite"] is True
    assert "--annotation-source" in plan.steps[2].command
    assert (
        "blenderproc"
        not in " ".join(part for step in plan.steps for part in step.command).lower()
    )


def test_calibrated_bop_sequence_passes_profiles_to_export(tmp_path: Path) -> None:
    run_root = tmp_path / "calibrated-run"

    plan = build_sequence_plan(
        sequence_id="sync_to_bop_calibrated_dry_run",
        run_root=run_root,
    )

    bop_step = next(step for step in plan.steps if step.id == "bop_export")
    assert (
        bop_step.options["calibration_profiles"]
        == (run_root / "calibration_profiles.json").as_posix()
    )
    assert "calibration_preflight" in step_ids(plan)


FRESH_CAPTURE_ACKNOWLEDGEMENTS = {
    "capture_plan_preflight": {"allow_real_robot": True},
    "capture_execution_plan": {
        "allow_cameras": True,
        "allow_real_robot": True,
    },
    "capture_execution": {
        "allow_cameras": True,
        "allow_real_robot": True,
    },
}


def test_real_full_capture_plan_never_persists_execution_acknowledgements(
    tmp_path: Path,
) -> None:
    plan = build_sequence_plan(
        sequence_id="real_full_capture_validation",
        run_root=tmp_path / "real-run",
        options=FRESH_CAPTURE_ACKNOWLEDGEMENTS,
        plan_only=True,
    )

    assert step_ids(plan)[:4] == [
        "run_preflight",
        "hardware_status",
        "capture_plan",
        "capture_plan_preflight",
    ]
    capture_plan = next(step for step in plan.steps if step.id == "capture_plan")
    capture_execution = next(
        step for step in plan.steps if step.id == "capture_execution"
    )
    assert capture_plan.options["warmup_frames"] == 30
    assert "--warmup-frames" in capture_plan.command
    assert capture_execution.options["timeout_s"] == 720.0
    assert capture_execution.options["startup_wait_s"] == 15.0
    assert capture_execution.options["receive_start_timeout_s"] == 120.0
    assert capture_execution.options["receive_idle_timeout_s"] == 60.0
    assert "--startup-wait" in capture_execution.command
    for step in plan.steps:
        assert step.options.get("allow_cameras") is not True
        assert step.options.get("allow_real_robot") is not True
        assert "--allow-cameras" not in step.command
        assert "--allow-real-robot" not in step.command
    assert plan.steps[-1].options["gate"] == "rewrite_full_capture.v1"


def test_real_full_capture_job_requires_fresh_per_step_acknowledgements(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="fresh literal-true per-step"):
        build_sequence_job(
            sequence_id="real_full_capture_validation",
            run_root=tmp_path / "blocked-real-run",
        )

    job = build_sequence_job(
        sequence_id="real_full_capture_validation",
        run_root=tmp_path / "allowed-real-run",
        options=FRESH_CAPTURE_ACKNOWLEDGEMENTS,
    )

    capture_execution = next(
        step for step in job.plan.steps if step.id == "capture_execution"
    )
    assert capture_execution.options["allow_cameras"] is True
    assert capture_execution.options["allow_real_robot"] is True
    assert "--allow-cameras" in capture_execution.command
    assert "--allow-real-robot" in capture_execution.command

    serialized_command = json.dumps(job.command)
    serialized_parameters = json.dumps(job.parameters, sort_keys=True)
    serialized_evidence = json.dumps(job.to_dict(), sort_keys=True)
    assert "allow_cameras" not in serialized_command
    assert "allow_real_robot" not in serialized_command
    assert "allow_cameras" not in serialized_parameters
    assert "allow_real_robot" not in serialized_parameters
    assert "--allow-cameras" not in serialized_evidence
    assert "--allow-real-robot" not in serialized_evidence
    acknowledgements = json.loads(job.execution_environment[SEQUENCE_EXECUTION_ACK_ENV])
    assert acknowledgements == FRESH_CAPTURE_ACKNOWLEDGEMENTS


def test_real_capture_sequence_plan_persistence_strips_one_shot_gates(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "sanitized-plan"
    plan = build_sequence_plan(
        sequence_id="real_full_capture_validation",
        run_root=run_root,
        options=FRESH_CAPTURE_ACKNOWLEDGEMENTS,
    )

    path = write_sequence_plan(run_root, plan)
    serialized = path.read_text()

    assert "allow_cameras" not in serialized
    assert "allow_real_robot" not in serialized
    assert "--allow-cameras" not in serialized
    assert "--allow-real-robot" not in serialized


@pytest.mark.parametrize(
    "options",
    [
        {},
    ],
)
def test_direct_real_sequence_rejects_missing_or_string_gates_before_writes(
    tmp_path: Path,
    options: dict,
) -> None:
    run_root = tmp_path / "direct-rejected"
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_pipeline_sequence.py",
            run_root.as_posix(),
            "--sequence",
            "real_full_capture_validation",
            "--options-json",
            json.dumps(options),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache"},
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "fresh literal-true" in result.stderr
    assert not run_root.exists()


def test_executed_sequence_retains_child_manifest_sensor_updates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "sequence-manifest"
    sensor = {
        "sensor_type": "realsense_d435",
        "device_id": "123",
        "folder": "realsense_123",
        "display_name": "Wrist camera",
        "mounting_mode": "eye_in_hand",
        "status": "planned",
        "metadata": {},
    }

    def execute_child_steps(plan, *, cwd) -> None:
        assert plan.sequence_id == "sync_to_bop_dry_run"
        assert cwd == Path(__file__).resolve().parents[1]
        child_manifest = load_or_create_run_manifest(run_root)
        child_manifest["sensors"] = [sensor]
        upsert_stage(
            child_manifest,
            name="child_sensor_stage",
            status="succeeded",
        )
        write_run_manifest(child_manifest, run_root)

    monkeypatch.setattr(
        run_pipeline_sequence,
        "execute_sequence_plan",
        execute_child_steps,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline_sequence.py",
            run_root.as_posix(),
            "--sequence",
            "sync_to_bop_dry_run",
        ],
    )

    run_pipeline_sequence.main()

    manifest = load_run_manifest(run_root)
    assert manifest["sensors"] == [sensor]
    stages = {stage["name"]: stage for stage in manifest["stages"]}
    assert stages["child_sensor_stage"]["status"] == "succeeded"
    assert stages["pipeline_sequence:sync_to_bop_dry_run"]["status"] == "succeeded"


def test_failed_sequence_retains_child_manifest_sensor_updates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "failed-sequence-manifest"
    sensor = {
        "sensor_type": "realsense_d435",
        "device_id": "123",
        "folder": "realsense_123",
        "display_name": "Wrist camera",
        "operator_alias": "Wrist camera",
        "mounting_mode": "eye_in_hand",
        "status": "planned",
        "metadata": {},
    }

    def execute_failing_child(plan, *, cwd) -> None:
        child_manifest = load_or_create_run_manifest(run_root)
        child_manifest["sensors"] = [sensor]
        upsert_stage(
            child_manifest,
            name="child_sensor_stage",
            status="failed",
            message="child failed after planning",
        )
        write_run_manifest(child_manifest, run_root)
        raise RuntimeError("child failed after planning")

    monkeypatch.setattr(
        run_pipeline_sequence,
        "execute_sequence_plan",
        execute_failing_child,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline_sequence.py",
            run_root.as_posix(),
            "--sequence",
            "sync_to_bop_dry_run",
        ],
    )

    with pytest.raises(RuntimeError, match="child failed after planning"):
        run_pipeline_sequence.main()

    manifest = load_run_manifest(run_root)
    assert manifest["sensors"] == [sensor]
    stages = {stage["name"]: stage for stage in manifest["stages"]}
    assert stages["child_sensor_stage"]["status"] == "failed"
    assert stages["pipeline_sequence:sync_to_bop_dry_run"]["status"] == "failed"


def test_real_capture_planning_steps_claim_camera_resource(tmp_path: Path) -> None:
    plan = build_sequence_plan(
        sequence_id="real_full_capture_validation",
        run_root=tmp_path / "resource-plan",
        options=FRESH_CAPTURE_ACKNOWLEDGEMENTS,
    )
    preflight = next(step for step in plan.steps if step.id == "capture_plan_preflight")
    execution_plan = next(
        step for step in plan.steps if step.id == "capture_execution_plan"
    )

    assert preflight.resources == ["camera", "disk_io"]
    assert execution_plan.resources == ["camera", "disk_io"]


def test_removed_downstream_sequences_are_unknown(tmp_path: Path) -> None:
    removed = [
        "capture_to_bop_foundationpose_dry_run",
        "fake_capture_rehearsal",
        "fake_capture_execution",
        "fake_capture_to_bop_dataset_dry_run",
        "fake_capture_to_bop_foundationpose_dry_run",
        "fake_capture_to_bop_eval_dry_run",
        "foundationpose_to_bop_eval_dry_run",
        "foundationpose_runtime_to_bop_eval",
        "megapose_to_bop_eval_dry_run",
        "sam6d_to_bop_eval_dry_run",
    ]

    for sequence_id in removed:
        with pytest.raises(ValueError, match="Unknown pipeline sequence"):
            build_sequence_plan(sequence_id=sequence_id, run_root=tmp_path / "run")


def test_sequence_listing_is_acquisition_only() -> None:
    sequence_ids = {sequence["id"] for sequence in list_pipeline_sequences()}

    assert {
        "capture_to_bop_dataset_dry_run",
        "sync_to_bop_dry_run",
        "sync_to_bop_calibrated_dry_run",
        "real_full_capture_validation",
        "aruco_grid_full_calibration",
        "calibrated_capture_to_bop_dataset_dry_run",
    } <= sequence_ids
    assert "foundationpose_runtime_to_bop_eval" not in sequence_ids
    assert "megapose_runtime_to_bop_eval" not in sequence_ids


def test_pipeline_sequence_registry_references_known_stages() -> None:
    for sequence_id in PIPELINE_SEQUENCES:
        plan = build_sequence_plan(sequence_id=sequence_id, run_root="/tmp/run")
        assert plan.steps


def test_full_grid_calibration_keeps_sync_quality_adjacent_and_splits_phases(
    tmp_path: Path,
) -> None:
    plan = build_sequence_plan(
        sequence_id="aruco_grid_full_calibration",
        run_root=tmp_path / "run",
    )

    assert step_ids(plan)[1:4] == ["sync_run", "sync_quality", "aruco_detection"]
    assert step_ids(plan)[4:7] == [
        "intrinsic_calibration",
        "aruco_pose",
        "calibration_observations",
    ]
    solver = next(step for step in plan.steps if step.id == "calibration_solver")
    assert solver.options["mode"] == "compare"


def test_calibrated_capture_to_bop_rectifies_before_consumers(tmp_path: Path) -> None:
    plan = build_sequence_plan(
        sequence_id="calibrated_capture_to_bop_dataset_dry_run",
        run_root=tmp_path / "run",
    )

    assert step_ids(plan) == [
        "sync_run",
        "sync_quality",
        "calibration_preflight",
        "camera_rectification",
        "bop_export",
    ]
    assert plan.steps[4].depends_on == ["camera_rectification"]
    assert plan.steps[4].options["annotation_source"] == "none"
    assert plan.steps[3].options["overwrite"] is True
    assert plan.steps[4].options["overwrite"] is True
