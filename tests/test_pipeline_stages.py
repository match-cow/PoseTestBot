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


@pytest.mark.parametrize(
    "stage_id",
    ["capture_plan_preflight", "capture_execution_plan"],
)
def test_live_capture_planning_stages_reserve_camera(
    tmp_path: Path,
    stage_id: str,
) -> None:
    job = build_pipeline_job(stage_id=stage_id, run_root=tmp_path / stage_id)

    assert job.resources == ["camera", "disk_io"]


def test_capture_execution_uses_calibration_receiver_timeouts(
    tmp_path: Path,
) -> None:
    job = build_pipeline_job(
        stage_id="capture_execution",
        run_root=tmp_path / "capture",
        options={"allow_cameras": True, "allow_real_robot": True},
    )

    assert job.parameters["options"]["receive_start_timeout_s"] == 120.0
    assert job.parameters["options"]["receive_idle_timeout_s"] == 60.0
    assert job.command[job.command.index("--receive-start-timeout-s") + 1] == "120.0"
    assert job.command[job.command.index("--receive-idle-timeout-s") + 1] == "60.0"


def test_rewrite_gate_choices_are_acquisition_only(tmp_path: Path) -> None:
    job = build_pipeline_job(stage_id="rewrite_gate", run_root=tmp_path / "run")

    assert "rewrite_full_capture.v1" in job.command
    stage = PIPELINE_STAGES["rewrite_gate"]
    gate = next(parameter for parameter in stage.parameters if parameter.name == "gate")
    assert "rewrite_foundationpose_runtime.v1" not in gate.choices
    assert "rewrite_fake_acquisition_to_bop.v1" not in gate.choices
    assert "rewrite_bop_export_readiness.v1" in gate.choices


def test_unknown_downstream_stage_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown pipeline stage"):
        build_pipeline_job(stage_id="bop_evaluation", run_root=tmp_path / "run")


@pytest.mark.parametrize("stage_id", ["capture_rehearsal", "synthetic_rgbd_fixture"])
def test_retired_fake_stage_ids_are_rejected(
    tmp_path: Path,
    stage_id: str,
) -> None:
    with pytest.raises(ValueError, match="Unknown pipeline stage"):
        build_pipeline_job(stage_id=stage_id, run_root=tmp_path / "run")


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
    assert "capture_rehearsal" not in stage_ids
    assert "synthetic_rgbd_fixture" not in stage_ids


def test_capture_execution_stage_rejects_retired_mode_option(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown pipeline option"):
        build_pipeline_job(
            stage_id="capture_execution",
            run_root=tmp_path / "run",
            options={"mode": "plan_only"},
        )


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


def test_advanced_sync_and_observation_stages_accept_explicit_subsets(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    sync = build_pipeline_job(
        stage_id="sync_run",
        run_root=run_root,
        options={
            "sensor_folder": ["realsense_1", "luxonis_2"],
            "output_root": "processed/calibration/attempt/synchronized",
        },
    )
    observations = build_pipeline_job(
        stage_id="calibration_observations",
        run_root=run_root,
        options={
            "aruco_path": [
                "processed/calibration/attempt/synchronized/realsense_1/aruco_pose_estimation.json"
            ],
            "output_root": "processed/calibration/attempt",
        },
    )

    assert sync.command.count("--sensor-folder") == 2
    assert "realsense_1" in sync.command
    assert "luxonis_2" in sync.command
    assert "--output-root" in sync.command
    assert observations.command.count("--aruco-path") == 1
    assert "--output-root" in observations.command
