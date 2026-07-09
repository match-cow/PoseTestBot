from __future__ import annotations

from pathlib import Path

import pytest

from posetestbot.pipeline.sequences import (
    PIPELINE_SEQUENCES,
    build_sequence_job,
    build_sequence_plan,
    list_pipeline_sequences,
)


def step_ids(plan) -> list[str]:
    return [step.id for step in plan.steps]


def test_fake_capture_to_bop_dataset_sequence_stops_at_bop_export(tmp_path: Path) -> None:
    run_root = tmp_path / "fake-run"

    plan = build_sequence_plan(
        sequence_id="fake_capture_to_bop_dataset_dry_run",
        run_root=run_root,
    )

    assert step_ids(plan) == [
        "capture_plan",
        "capture_plan_preflight",
        "capture_execution_plan",
        "capture_execution",
        "synthetic_rgbd_fixture",
        "sync_run",
        "sync_quality",
        "blenderproc_prepare",
        "blenderproc_render",
        "bop_export",
    ]
    assert plan.steps[-1].stage_id == "bop_export"
    assert not any(step.stage_id in {"foundationpose", "bop_evaluation"} for step in plan.steps)
    assert plan.resources == ["camera", "cpu", "disk_io", "render", "robot_command"]


def test_capture_to_bop_dataset_sequence_uses_sync_quality_gate(tmp_path: Path) -> None:
    plan = build_sequence_plan(
        sequence_id="capture_to_bop_dataset_dry_run",
        run_root=tmp_path / "captured-run",
    )

    assert step_ids(plan) == [
        "sync_run",
        "sync_quality",
        "blenderproc_prepare",
        "blenderproc_render",
        "bop_export",
    ]
    assert plan.steps[2].depends_on == ["sync_quality"]
    assert plan.steps[3].options["dry_run"] is True


def test_calibrated_bop_sequence_passes_profiles_to_export(tmp_path: Path) -> None:
    run_root = tmp_path / "calibrated-run"

    plan = build_sequence_plan(
        sequence_id="sync_to_bop_calibrated_dry_run",
        run_root=run_root,
    )

    bop_step = next(step for step in plan.steps if step.id == "bop_export")
    assert bop_step.options["calibration_profiles"] == (
        run_root / "calibration_profiles.json"
    ).as_posix()
    assert "calibration_preflight" in step_ids(plan)


def test_real_full_capture_sequence_keeps_explicit_hardware_gates(tmp_path: Path) -> None:
    plan = build_sequence_plan(
        sequence_id="real_full_capture_validation",
        run_root=tmp_path / "real-run",
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
    assert capture_execution.options["timeout_s"] == 120.0
    assert capture_execution.options["startup_wait_s"] == 6.0
    assert "--startup-wait" in capture_execution.command
    assert plan.steps[-1].options["gate"] == "rewrite_full_capture.v1"


def test_removed_downstream_sequences_are_unknown(tmp_path: Path) -> None:
    removed = [
        "capture_to_bop_foundationpose_dry_run",
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
        "fake_capture_rehearsal",
        "fake_capture_execution",
        "capture_to_bop_dataset_dry_run",
        "fake_capture_to_bop_dataset_dry_run",
        "sync_to_bop_dry_run",
        "sync_to_bop_calibrated_dry_run",
        "real_full_capture_validation",
    } <= sequence_ids
    assert "foundationpose_runtime_to_bop_eval" not in sequence_ids
    assert "megapose_runtime_to_bop_eval" not in sequence_ids


def test_sequence_job_plan_only_locks_disk_only(tmp_path: Path) -> None:
    job = build_sequence_job(
        sequence_id="fake_capture_to_bop_dataset_dry_run",
        run_root=tmp_path / "run",
        plan_only=True,
    )

    assert job.command[-1] == "--plan-only"
    assert job.resources == ["disk_io"]
    assert job.parameters["planned_resources"] == [
        "camera",
        "cpu",
        "disk_io",
        "render",
        "robot_command",
    ]


def test_pipeline_sequence_registry_references_known_stages() -> None:
    for sequence_id in PIPELINE_SEQUENCES:
        plan = build_sequence_plan(sequence_id=sequence_id, run_root="/tmp/run")
        assert plan.steps
