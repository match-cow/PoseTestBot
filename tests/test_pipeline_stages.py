from __future__ import annotations

from pathlib import Path

import pytest

from posetestbot.pipeline.stages import (
    PIPELINE_STAGES,
    build_pipeline_job,
    list_pipeline_stages,
)


def test_downstream_stage_ids_are_not_registered() -> None:
    removed = {
        "foundationpose",
        "megapose",
        "sam6d",
        "bop_result_export",
        "synthetic_bop_results",
        "bop_evaluation",
        "metric_report_export",
    }

    assert removed.isdisjoint(PIPELINE_STAGES)


def test_bop_export_stage_builds_dataset_export_command(tmp_path: Path) -> None:
    job = build_pipeline_job(
        stage_id="bop_export",
        run_root=tmp_path / "run",
        options={
            "input_folder": "processed/synchronized",
            "object_folder": "object_models",
            "write_multiview_targets": True,
            "write_coco_annotations": True,
        },
    )

    assert job.command[:4] == ["uv", "run", "python", "scripts/run_bop_export_stage.py"]
    assert "--input-folder" in job.command
    assert "processed/synchronized" in job.command
    assert "--write-multiview-targets" in job.command
    assert "--write-coco-annotations" in job.command
    assert job.resources == ["disk_io"]


def test_blenderproc_render_defaults_to_dry_run(tmp_path: Path) -> None:
    job = build_pipeline_job(stage_id="blenderproc_render", run_root=tmp_path / "run")

    assert "--dry-run" in job.command
    assert job.parameters["options"]["dry_run"] is True
    assert job.resources == ["render", "disk_io"]


def test_capture_plan_stage_accepts_warmup_frames(tmp_path: Path) -> None:
    job = build_pipeline_job(
        stage_id="capture_plan",
        run_root=tmp_path / "run",
        options={"warmup_frames": 30},
    )

    assert "--warmup-frames" in job.command
    assert "30" in job.command
    assert job.parameters["options"]["warmup_frames"] == 30


def test_rewrite_gate_choices_are_acquisition_only(tmp_path: Path) -> None:
    job = build_pipeline_job(stage_id="rewrite_gate", run_root=tmp_path / "run")

    assert "rewrite_fake_acquisition_to_bop.v1" in job.command
    stage = PIPELINE_STAGES["rewrite_gate"]
    gate = next(parameter for parameter in stage.parameters if parameter.name == "gate")
    assert "rewrite_foundationpose_runtime.v1" not in gate.choices
    assert "rewrite_bop_export_readiness.v1" in gate.choices


def test_unknown_downstream_stage_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown pipeline stage"):
        build_pipeline_job(stage_id="bop_evaluation", run_root=tmp_path / "run")


def test_stage_listing_contains_acquisition_stages_only() -> None:
    stage_ids = {stage["id"] for stage in list_pipeline_stages()}

    assert {
        "capture_plan",
        "sync_run",
        "sync_quality",
        "calibration_target_import",
        "aruco_detection",
        "intrinsic_calibration",
        "aruco_pose",
        "calibration_solver",
        "camera_rectification",
        "bop_export",
    } <= stage_ids
    assert "metric_report_export" not in stage_ids


def test_every_pipeline_path_parameter_declares_web_scope() -> None:
    path_parameters = [
        parameter
        for stage in PIPELINE_STAGES.values()
        for parameter in stage.parameters
        if parameter.kind == "path"
    ]

    assert path_parameters
    assert all(parameter.path_scope is not None for parameter in path_parameters)


def test_compare_and_selection_options_are_repeatable_api_contracts(
    tmp_path: Path,
) -> None:
    solver = build_pipeline_job(
        stage_id="calibration_solver",
        run_root=tmp_path / "run",
        options={"mode": "compare"},
    )
    validation = build_pipeline_job(
        stage_id="calibration_validation",
        run_root=tmp_path / "run",
        options={
            "select_profile": ["realsense_1=unknown", "realsense_2=known"],
            "promote": True,
        },
    )

    assert ["--mode", "compare"] == solver.command[5:7]
    assert validation.command.count("--select-profile") == 2
    assert "realsense_1=unknown" in validation.command
    assert "--promote" in validation.command
