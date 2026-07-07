from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from posetestbot.io.artifacts import DATASET_MANIFEST, PIPELINE_SEQUENCE_PLAN
from posetestbot.pipeline.sequences import (
    build_sequence_job,
    build_sequence_plan,
    list_pipeline_sequences,
    write_sequence_plan,
)


def test_sequence_plan_orders_dependencies_and_merges_options(tmp_path: Path) -> None:
    run_root = tmp_path / "run"

    plan = build_sequence_plan(
        sequence_id="sync_to_bop_dry_run",
        run_root=run_root,
        options={
            "sync_run": {"timestamp_source": "sensor", "no_copy": True},
            "bop_export": {"no_model_export": True},
        },
    )

    assert [step.id for step in plan.steps] == [
        "sync_run",
        "sync_quality",
        "blenderproc_prepare",
        "blenderproc_render",
        "bop_export",
    ]
    assert plan.steps[1].depends_on == ["sync_run"]
    assert plan.steps[2].depends_on == ["sync_quality"]
    assert plan.steps[3].options == {"dry_run": True}
    assert "--timestamp-source" in plan.steps[0].command
    assert "sensor" in plan.steps[0].command
    assert "--no-copy" in plan.steps[0].command
    assert "--no-model-export" in plan.steps[4].command
    assert plan.resources == ["cpu", "disk_io", "render"]


def test_foundationpose_to_bop_eval_sequence_defaults_to_dry_run(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    plan = build_sequence_plan(
        sequence_id="foundationpose_to_bop_eval_dry_run",
        run_root=run_root,
    )

    assert [step.id for step in plan.steps] == [
        "bop_export",
        "bop_result_export",
        "bop_evaluation",
    ]
    assert plan.steps[1].depends_on == ["bop_export"]
    assert plan.steps[2].depends_on == ["bop_result_export"]
    assert "--source" in plan.steps[1].command
    assert "foundationpose" in plan.steps[1].command
    assert "--result-file" in plan.steps[2].command
    assert (
        run_root / "results" / "bop" / "foundationpose_bop-test.csv"
    ).as_posix() in plan.steps[2].command
    assert "--dry-run" in plan.steps[2].command
    assert plan.resources == ["disk_io", "evaluation"]


def test_sync_to_bop_calibrated_sequence_preflights_profiles(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    plan = build_sequence_plan(
        sequence_id="sync_to_bop_calibrated_dry_run",
        run_root=run_root,
        options={
            "calibration_preflight": {"require_valid": True},
            "bop_export": {"no_model_export": True},
        },
    )

    assert [step.id for step in plan.steps] == [
        "sync_run",
        "sync_quality",
        "calibration_preflight",
        "blenderproc_prepare",
        "blenderproc_render",
        "bop_export",
    ]
    assert plan.steps[2].depends_on == ["sync_quality"]
    assert plan.steps[3].depends_on == ["sync_quality", "calibration_preflight"]
    assert plan.steps[4].depends_on == ["blenderproc_prepare"]
    assert plan.steps[5].depends_on == ["blenderproc_render"]
    assert "--require-valid" in plan.steps[2].command
    assert (
        run_root / "calibration_profiles.json"
    ).as_posix() in plan.steps[3].command
    assert (
        run_root / "calibration_profiles.json"
    ).as_posix() in plan.steps[5].command
    assert "--no-model-export" in plan.steps[5].command
    assert "--dry-run" in plan.steps[4].command
    assert plan.resources == ["cpu", "disk_io", "render"]


def test_foundationpose_runtime_to_bop_eval_sequence_runs_runtime_stages(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    plan = build_sequence_plan(
        sequence_id="foundationpose_runtime_to_bop_eval",
        run_root=run_root,
        options={
            "foundationpose": {
                "foundationpose_folder": "/opt/FoundationPose",
                "run_level": True,
            },
            "bop_evaluation": {
                "bop_toolkit_root": "/opt/bop_toolkit",
                "num_workers": 2,
            },
        },
    )

    assert [step.id for step in plan.steps] == [
        "bop_export",
        "foundationpose",
        "bop_result_export",
        "bop_evaluation",
    ]
    assert plan.steps[1].depends_on == ["bop_export"]
    assert plan.steps[2].depends_on == ["foundationpose"]
    assert plan.steps[3].depends_on == ["bop_result_export"]
    assert plan.steps[1].options["dry_run"] is False
    assert plan.steps[3].options["dry_run"] is False
    assert "scripts/run_foundationpose_stage.py" in plan.steps[1].command
    assert "--dry-run" not in plan.steps[1].command
    assert "--foundationpose-folder" in plan.steps[1].command
    assert "/opt/FoundationPose" in plan.steps[1].command
    assert "--run-level" in plan.steps[1].command
    assert plan.steps[2].command[plan.steps[2].command.index("--source") + 1] == (
        "foundationpose"
    )
    assert "--dry-run" not in plan.steps[3].command
    assert (
        run_root / "results" / "bop" / "foundationpose_bop-test_est5_track2.csv"
    ).as_posix() in plan.steps[3].command
    assert "--bop-toolkit-root" in plan.steps[3].command
    assert "/opt/bop_toolkit" in plan.steps[3].command
    assert "--num-workers" in plan.steps[3].command
    assert "2" in plan.steps[3].command
    assert plan.resources == ["disk_io", "estimator", "evaluation"]


def test_sync_aruco_calibration_observation_sequence_orders_steps(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    plan = build_sequence_plan(
        sequence_id="sync_aruco_calibration_observations",
        run_root=run_root,
        options={"calibration_observations": {"min_observations": 2}},
    )

    assert [step.id for step in plan.steps] == [
        "sync_run",
        "sync_quality",
        "aruco",
        "calibration_observations",
    ]
    assert plan.steps[1].depends_on == ["sync_run"]
    assert plan.steps[2].depends_on == ["sync_quality"]
    assert plan.steps[3].depends_on == ["aruco"]
    assert "--min-observations" in plan.steps[3].command
    assert "2" in plan.steps[3].command
    assert plan.resources == ["cpu", "disk_io"]


def test_sync_aruco_calibration_candidate_sequence_orders_steps(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    plan = build_sequence_plan(
        sequence_id="sync_aruco_calibration_candidates",
        run_root=run_root,
        options={"calibration_candidates": {"min_observations": 2}},
    )

    assert [step.id for step in plan.steps] == [
        "sync_run",
        "sync_quality",
        "aruco",
        "calibration_observations",
        "calibration_candidates",
    ]
    assert plan.steps[4].depends_on == ["calibration_observations"]
    assert "--min-observations" in plan.steps[4].command
    assert "2" in plan.steps[4].command
    assert plan.resources == ["cpu", "disk_io"]


def test_sync_aruco_calibration_solver_sequence_orders_steps(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    plan = build_sequence_plan(
        sequence_id="sync_aruco_calibration_solver",
        run_root=run_root,
        options={"calibration_solver": {"min_observations": 3, "hand_eye_method": "park"}},
    )

    assert [step.id for step in plan.steps] == [
        "sync_run",
        "sync_quality",
        "aruco",
        "calibration_observations",
        "calibration_solver",
    ]
    assert plan.steps[4].depends_on == ["calibration_observations"]
    assert "--min-observations" in plan.steps[4].command
    assert "3" in plan.steps[4].command
    assert "--hand-eye-method" in plan.steps[4].command
    assert "park" in plan.steps[4].command
    assert plan.resources == ["cpu", "disk_io"]


def test_sync_aruco_calibration_validation_sequence_orders_steps(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    plan = build_sequence_plan(
        sequence_id="sync_aruco_calibration_validation",
        run_root=run_root,
        options={"calibration_validation": {"min_inliers": 4}},
    )

    assert [step.id for step in plan.steps] == [
        "sync_run",
        "sync_quality",
        "aruco",
        "calibration_observations",
        "calibration_candidates",
        "calibration_validation",
    ]
    assert plan.steps[5].depends_on == ["calibration_candidates"]
    assert "--min-inliers" in plan.steps[5].command
    assert "4" in plan.steps[5].command
    assert plan.resources == ["cpu", "disk_io"]


def test_capture_to_bop_foundationpose_sequence_plans_capture_bridge(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    plan = build_sequence_plan(
        sequence_id="capture_to_bop_foundationpose_dry_run",
        run_root=run_root,
        options={
            "sync_run": {"timestamp_source": "sensor"},
            "blenderproc_prepare": {
                "calibration_profiles": "{run_root}/calibration_profiles.json",
            },
            "bop_export": {
                "calibration_profiles": "{run_root}/calibration_profiles.json",
            },
        },
    )

    assert [step.id for step in plan.steps] == [
        "sync_run",
        "sync_quality",
        "blenderproc_prepare",
        "blenderproc_render",
        "bop_export",
        "foundationpose",
    ]
    assert plan.steps[1].depends_on == ["sync_run"]
    assert plan.steps[2].depends_on == ["sync_quality"]
    assert plan.steps[3].depends_on == ["blenderproc_prepare"]
    assert plan.steps[4].depends_on == ["blenderproc_render"]
    assert plan.steps[5].depends_on == ["bop_export"]
    assert "--timestamp-source" in plan.steps[0].command
    assert "sensor" in plan.steps[0].command
    assert plan.steps[3].options == {"dry_run": True}
    assert "--dry-run" in plan.steps[3].command
    assert plan.steps[5].options == {"dry_run": True}
    assert "--dry-run" in plan.steps[5].command
    assert (
        run_root / "calibration_profiles.json"
    ).as_posix() in plan.steps[2].command
    assert (
        run_root / "calibration_profiles.json"
    ).as_posix() in plan.steps[4].command
    assert plan.resources == ["cpu", "disk_io", "estimator", "render"]


def test_fake_capture_to_bop_foundationpose_sequence_inserts_synthetic_fixture(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    plan = build_sequence_plan(
        sequence_id="fake_capture_to_bop_foundationpose_dry_run",
        run_root=run_root,
        options={
            "capture_execution": {"timeout_s": 5},
            "synthetic_rgbd_fixture": {"frame_count": 2},
        },
    )

    assert [step.id for step in plan.steps] == [
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
        "foundationpose",
    ]
    assert plan.steps[4].depends_on == ["capture_execution"]
    assert plan.steps[5].depends_on == ["synthetic_rgbd_fixture"]
    assert plan.steps[4].stage_id == "synthetic_rgbd_fixture"
    assert "--frame-count" in plan.steps[4].command
    assert "2" in plan.steps[4].command
    assert "--overwrite" in plan.steps[4].command
    assert plan.steps[8].options == {"dry_run": True}
    assert plan.steps[10].options == {"dry_run": True}
    assert plan.resources == ["camera", "cpu", "disk_io", "estimator", "render", "robot_command"]


def test_fake_capture_to_bop_eval_sequence_plans_result_and_evaluation(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    plan = build_sequence_plan(
        sequence_id="fake_capture_to_bop_eval_dry_run",
        run_root=run_root,
        options={"synthetic_bop_results": {"object_name": "Df4a"}},
    )

    assert [step.id for step in plan.steps] == [
        "capture_plan",
        "capture_plan_preflight",
        "capture_execution_plan",
        "capture_execution",
        "synthetic_rgbd_fixture",
        "sync_run",
        "sync_quality",
        "bop_export",
        "synthetic_bop_results",
        "bop_evaluation",
    ]
    assert plan.steps[8].depends_on == ["bop_export"]
    assert plan.steps[8].stage_id == "synthetic_bop_results"
    assert "--object-name" in plan.steps[8].command
    assert "Df4a" in plan.steps[8].command
    assert plan.steps[9].depends_on == ["synthetic_bop_results"]
    assert (
        run_root / "results" / "bop" / "synthetic_bop-test.csv"
    ).as_posix() in plan.steps[9].command
    assert "--dry-run" in plan.steps[9].command
    assert plan.resources == ["camera", "disk_io", "evaluation", "robot_command"]


def test_fake_capture_rehearsal_sequence_plans_capture_plan_then_rehearsal(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    plan = build_sequence_plan(
        sequence_id="fake_capture_rehearsal",
        run_root=run_root,
        options={
            "capture_rehearsal": {
                "duration_s": 0.1,
                "sample_ms": 20,
            },
        },
    )

    assert [step.id for step in plan.steps] == [
        "capture_plan",
        "capture_plan_preflight",
        "capture_execution_plan",
        "capture_rehearsal",
    ]
    assert plan.steps[1].depends_on == ["capture_plan"]
    assert plan.steps[1].options["no_sensors"] is True
    assert plan.steps[2].depends_on == ["capture_plan_preflight"]
    assert plan.steps[2].options["mode"] == "pose_only_fake"
    assert plan.steps[3].depends_on == ["capture_execution_plan"]
    assert plan.steps[0].command[:4] == [
        "uv",
        "run",
        "python",
        "scripts/run_capture_plan_stage.py",
    ]
    assert plan.steps[1].command[:4] == [
        "uv",
        "run",
        "python",
        "scripts/run_capture_plan_preflight.py",
    ]
    assert "--no-sensors" in plan.steps[1].command
    assert plan.steps[2].command[:4] == [
        "uv",
        "run",
        "python",
        "scripts/run_capture_execution_plan.py",
    ]
    assert "--mode" in plan.steps[2].command
    assert "pose_only_fake" in plan.steps[2].command
    assert plan.steps[3].command[:4] == [
        "uv",
        "run",
        "python",
        "scripts/run_capture_rehearsal_stage.py",
    ]
    assert "--duration" in plan.steps[3].command
    assert "0.1" in plan.steps[3].command
    assert plan.resources == ["disk_io", "robot_command"]


def test_fake_capture_execution_sequence_plans_supervised_execution(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    plan = build_sequence_plan(
        sequence_id="fake_capture_execution",
        run_root=run_root,
        options={
            "capture_execution": {
                "timeout_s": 5,
                "startup_wait_s": 0.1,
            },
        },
    )

    assert [step.id for step in plan.steps] == [
        "capture_plan",
        "capture_plan_preflight",
        "capture_execution_plan",
        "capture_execution",
    ]
    assert plan.steps[1].depends_on == ["capture_plan"]
    assert plan.steps[1].options["no_sensors"] is True
    assert plan.steps[2].depends_on == ["capture_plan_preflight"]
    assert plan.steps[2].options["mode"] == "pose_only_fake"
    assert plan.steps[3].depends_on == ["capture_execution_plan"]
    assert plan.steps[3].options["mode"] == "pose_only_fake"
    assert plan.steps[3].options["timeout_s"] == 5.0
    assert plan.steps[3].command[:4] == [
        "uv",
        "run",
        "python",
        "scripts/run_capture_execution_stage.py",
    ]
    assert "--timeout-s" in plan.steps[3].command
    assert "5.0" in plan.steps[3].command
    assert plan.resources == ["camera", "disk_io", "robot_command"]


def test_real_full_capture_validation_sequence_runs_full_capture_then_gate(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    plan = build_sequence_plan(
        sequence_id="real_full_capture_validation",
        run_root=run_root,
        options={"capture_execution": {"timeout_s": 10}},
    )

    assert [step.id for step in plan.steps] == [
        "run_preflight",
        "hardware_status",
        "capture_plan",
        "capture_plan_preflight",
        "capture_execution_plan",
        "capture_execution",
        "rewrite_full_capture_gate",
    ]
    assert plan.steps[0].stage_id == "run_preflight"
    assert plan.steps[0].depends_on == []
    assert plan.steps[0].options["write"] is True
    assert plan.steps[0].options["check"] is True
    assert "--check" in plan.steps[0].command
    assert "--write" in plan.steps[0].command
    assert plan.steps[1].stage_id == "hardware_status"
    assert plan.steps[1].depends_on == ["run_preflight"]
    assert plan.steps[2].depends_on == ["hardware_status"]
    assert plan.steps[3].depends_on == ["capture_plan"]
    assert plan.steps[3].options["allow_real_robot"] is True
    assert "--allow-real-robot" in plan.steps[3].command
    assert "--no-sensors" not in plan.steps[3].command
    assert plan.steps[4].depends_on == ["capture_plan_preflight"]
    assert plan.steps[4].options["mode"] == "full"
    assert plan.steps[4].options["allow_cameras"] is True
    assert plan.steps[4].options["allow_real_robot"] is True
    assert plan.steps[4].options["include_sensors"] is True
    assert "--mode" in plan.steps[4].command
    assert "full" in plan.steps[4].command
    assert "--allow-cameras" in plan.steps[4].command
    assert "--allow-real-robot" in plan.steps[4].command
    assert "--include-sensors" in plan.steps[4].command
    assert plan.steps[5].depends_on == ["capture_execution_plan"]
    assert plan.steps[5].options["mode"] == "full"
    assert plan.steps[5].options["allow_cameras"] is True
    assert plan.steps[5].options["allow_real_robot"] is True
    assert plan.steps[5].options["include_sensors"] is True
    assert plan.steps[5].options["timeout_s"] == 10.0
    assert "--timeout-s" in plan.steps[5].command
    assert "10.0" in plan.steps[5].command
    assert plan.steps[6].depends_on == ["capture_execution"]
    assert plan.steps[6].stage_id == "rewrite_gate"
    assert plan.steps[6].options["gate"] == "rewrite_full_capture.v1"
    assert plan.steps[6].options["write"] is True
    assert plan.steps[6].command == [
        "uv",
        "run",
        "python",
        "scripts/run_rewrite_gate.py",
        run_root.as_posix(),
        "--gate",
        "rewrite_full_capture.v1",
        "--write",
    ]
    assert plan.resources == ["camera", "disk_io", "robot_command"]


def test_aruco_to_bop_eval_sequence_sets_aruco_source_and_result(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    plan = build_sequence_plan(
        sequence_id="aruco_to_bop_eval_dry_run",
        run_root=run_root,
        options={
            "bop_result_export": {"min_marker_count": 3},
            "bop_evaluation": {
                "result_file": "{run_root}/results/bop/aruco_custom.csv"
            },
        },
    )

    result_export_command = plan.steps[1].command
    evaluation_command = plan.steps[2].command
    assert result_export_command[result_export_command.index("--source") + 1] == "aruco"
    assert "--aruco-object-name" in result_export_command
    assert "aruco" in result_export_command
    assert "--min-marker-count" in result_export_command
    assert "3" in result_export_command
    assert (
        run_root / "results" / "bop" / "aruco_custom.csv"
    ).as_posix() in evaluation_command


def test_megapose_to_bop_eval_sequence_plans_estimator_bridge(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    plan = build_sequence_plan(
        sequence_id="megapose_to_bop_eval_dry_run",
        run_root=run_root,
        options={
            "megapose": {
                "wrapper_script": "/opt/megapose_wrapper.py",
                "model": "megapose-1.0-RGBD",
                "result_id": "rgbd",
            },
            "bop_result_export": {
                "megapose_output": [
                    "{run_root}/processed/synchronized/realsense_123/megapose_rgbd_obj0_output"
                ]
            },
            "bop_evaluation": {
                "result_file": "{run_root}/results/bop/megapose_bop-test_rgbd.csv"
            },
        },
    )

    assert [step.id for step in plan.steps] == [
        "bop_export",
        "megapose",
        "bop_result_export",
        "bop_evaluation",
    ]
    assert plan.steps[1].depends_on == ["bop_export"]
    assert plan.steps[2].depends_on == ["megapose"]
    assert plan.steps[3].depends_on == ["bop_result_export"]
    assert "scripts/run_megapose_stage.py" in plan.steps[1].command
    assert "--dry-run" in plan.steps[1].command
    assert "--wrapper-script" in plan.steps[1].command
    assert "/opt/megapose_wrapper.py" in plan.steps[1].command
    assert "--model" in plan.steps[1].command
    assert "megapose-1.0-RGBD" in plan.steps[1].command
    assert plan.steps[2].command[plan.steps[2].command.index("--source") + 1] == (
        "megapose"
    )
    assert "--megapose-output" in plan.steps[2].command
    assert (
        run_root
        / "processed"
        / "synchronized"
        / "realsense_123"
        / "megapose_rgbd_obj0_output"
    ).as_posix() in plan.steps[2].command
    assert (
        run_root / "results" / "bop" / "megapose_bop-test_rgbd.csv"
    ).as_posix() in plan.steps[3].command
    assert plan.resources == ["disk_io", "estimator", "evaluation"]


def test_megapose_runtime_to_bop_eval_sequence_runs_runtime_stages(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    plan = build_sequence_plan(
        sequence_id="megapose_runtime_to_bop_eval",
        run_root=run_root,
        options={
            "megapose": {
                "wrapper_script": "/opt/megapose_wrapper.py",
                "model": "megapose-1.0-RGBD",
            },
            "bop_evaluation": {
                "bop_toolkit_root": "/opt/bop_toolkit",
            },
        },
    )

    assert [step.id for step in plan.steps] == [
        "bop_export",
        "megapose",
        "bop_result_export",
        "bop_evaluation",
    ]
    assert plan.steps[1].depends_on == ["bop_export"]
    assert plan.steps[2].depends_on == ["megapose"]
    assert plan.steps[3].depends_on == ["bop_result_export"]
    assert plan.steps[1].options["dry_run"] is False
    assert plan.steps[3].options["dry_run"] is False
    assert "scripts/run_megapose_stage.py" in plan.steps[1].command
    assert "--dry-run" not in plan.steps[1].command
    assert "--wrapper-script" in plan.steps[1].command
    assert "/opt/megapose_wrapper.py" in plan.steps[1].command
    assert plan.steps[2].command[plan.steps[2].command.index("--source") + 1] == (
        "megapose"
    )
    assert "--dry-run" not in plan.steps[3].command
    assert (
        run_root / "results" / "bop" / "megapose_bop-test.csv"
    ).as_posix() in plan.steps[3].command
    assert "--bop-toolkit-root" in plan.steps[3].command
    assert "/opt/bop_toolkit" in plan.steps[3].command
    assert plan.resources == ["disk_io", "estimator", "evaluation"]


def test_sam6d_to_bop_eval_sequence_plans_estimator_bridge(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    plan = build_sequence_plan(
        sequence_id="sam6d_to_bop_eval_dry_run",
        run_root=run_root,
        options={
            "sam6d": {
                "wrapper_script": "/opt/sam6d_wrapper.py",
                "segmentor_model": "sam-hq",
                "result_id": "sam-hq",
            },
            "bop_result_export": {
                "sam6d_output": [
                    "{run_root}/processed/synchronized/realsense_123/sam6d_sam-hq_obj0_output"
                ]
            },
            "bop_evaluation": {
                "result_file": "{run_root}/results/bop/sam6d_bop-test_sam-hq.csv"
            },
        },
    )

    assert [step.id for step in plan.steps] == [
        "bop_export",
        "sam6d",
        "bop_result_export",
        "bop_evaluation",
    ]
    assert plan.steps[1].depends_on == ["bop_export"]
    assert plan.steps[2].depends_on == ["sam6d"]
    assert plan.steps[3].depends_on == ["bop_result_export"]
    assert "scripts/run_sam6d_stage.py" in plan.steps[1].command
    assert "--dry-run" in plan.steps[1].command
    assert "--wrapper-script" in plan.steps[1].command
    assert "/opt/sam6d_wrapper.py" in plan.steps[1].command
    assert "--segmentor-model" in plan.steps[1].command
    assert "sam-hq" in plan.steps[1].command
    assert plan.steps[2].command[plan.steps[2].command.index("--source") + 1] == (
        "sam6d"
    )
    assert "--sam6d-output" in plan.steps[2].command
    assert (
        run_root
        / "processed"
        / "synchronized"
        / "realsense_123"
        / "sam6d_sam-hq_obj0_output"
    ).as_posix() in plan.steps[2].command
    assert (
        run_root / "results" / "bop" / "sam6d_bop-test_sam-hq.csv"
    ).as_posix() in plan.steps[3].command
    assert plan.resources == ["disk_io", "estimator", "evaluation"]


def test_sam6d_runtime_to_bop_eval_sequence_runs_runtime_stages(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    plan = build_sequence_plan(
        sequence_id="sam6d_runtime_to_bop_eval",
        run_root=run_root,
        options={
            "sam6d": {
                "wrapper_script": "/opt/sam6d_wrapper.py",
                "segmentor_model": "sam-hq",
            },
            "bop_evaluation": {
                "bop_toolkit_root": "/opt/bop_toolkit",
            },
        },
    )

    assert [step.id for step in plan.steps] == [
        "bop_export",
        "sam6d",
        "bop_result_export",
        "bop_evaluation",
    ]
    assert plan.steps[1].depends_on == ["bop_export"]
    assert plan.steps[2].depends_on == ["sam6d"]
    assert plan.steps[3].depends_on == ["bop_result_export"]
    assert plan.steps[1].options["dry_run"] is False
    assert plan.steps[3].options["dry_run"] is False
    assert "scripts/run_sam6d_stage.py" in plan.steps[1].command
    assert "--dry-run" not in plan.steps[1].command
    assert "--wrapper-script" in plan.steps[1].command
    assert "/opt/sam6d_wrapper.py" in plan.steps[1].command
    assert plan.steps[2].command[plan.steps[2].command.index("--source") + 1] == (
        "sam6d"
    )
    assert "--dry-run" not in plan.steps[3].command
    assert (
        run_root / "results" / "bop" / "sam6d_bop-test.csv"
    ).as_posix() in plan.steps[3].command
    assert "--bop-toolkit-root" in plan.steps[3].command
    assert "/opt/bop_toolkit" in plan.steps[3].command
    assert plan.resources == ["disk_io", "estimator", "evaluation"]


def test_sequence_plan_rejects_unknown_option_group(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not_a_step"):
        build_sequence_plan(
            sequence_id="sync_aruco",
            run_root=tmp_path / "run",
            options={"not_a_step": {"save_images": True}},
        )


def test_sequence_job_builds_uv_command_and_parameter_snapshot(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    job = build_sequence_job(
        sequence_id="sync_aruco",
        run_root=run_root,
        options={"aruco": {"save_images": True}},
        plan_only=True,
    )

    assert job.command[:4] == [
        "uv",
        "run",
        "python",
        "scripts/run_pipeline_sequence.py",
    ]
    assert job.command[4] == run_root.as_posix()
    assert "--plan-only" in job.command
    assert job.resources == ["disk_io"]
    assert job.plan.resources == ["cpu", "disk_io"]
    assert job.parameters["pipeline_sequence"] == "sync_aruco"
    assert job.parameters["locked_resources"] == ["disk_io"]
    assert job.parameters["planned_resources"] == ["cpu", "disk_io"]
    assert job.parameters["steps"][2]["options"] == {
        "save_images": True,
        "show": False,
    }


def test_sequence_job_uses_full_resources_when_executing(tmp_path: Path) -> None:
    run_root = tmp_path / "run"

    job = build_sequence_job(
        sequence_id="sync_aruco",
        run_root=run_root,
        plan_only=False,
    )

    assert "--plan-only" not in job.command
    assert job.resources == ["cpu", "disk_io"]
    assert job.plan.resources == ["cpu", "disk_io"]
    assert job.parameters["locked_resources"] == ["cpu", "disk_io"]
    assert job.parameters["planned_resources"] == ["cpu", "disk_io"]


def test_write_sequence_plan_json(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    plan = build_sequence_plan(
        sequence_id="sync_aruco",
        run_root=run_root,
        plan_only=True,
    )

    path = write_sequence_plan(run_root, plan)

    data = json.loads(path.read_text())
    assert path.name == PIPELINE_SEQUENCE_PLAN
    assert data["schema_version"] == "pipeline_sequence_plan.v1"
    assert data["sequence_id"] == "sync_aruco"
    assert [step["id"] for step in data["steps"]] == [
        "sync_run",
        "sync_quality",
        "aruco",
    ]


def test_pipeline_sequence_listing_is_json_friendly() -> None:
    sequences = list_pipeline_sequences()
    sequence = next(item for item in sequences if item["id"] == "sync_to_bop_dry_run")
    sequence_ids = {item["id"] for item in sequences}

    assert sequence["label"] == "Synchronize To BOP Dry-Run"
    assert sequence["steps"][1]["depends_on"] == ["sync_run"]
    assert sequence["steps"][2]["depends_on"] == ["sync_quality"]
    assert "sync_aruco_calibration_observations" in sequence_ids
    assert "sync_aruco_calibration_candidates" in sequence_ids
    assert "sync_aruco_calibration_solver" in sequence_ids
    assert "sync_aruco_calibration_validation" in sequence_ids
    assert "fake_capture_rehearsal" in sequence_ids
    assert "fake_capture_execution" in sequence_ids
    assert "real_full_capture_validation" in sequence_ids
    assert "capture_to_bop_foundationpose_dry_run" in sequence_ids
    assert "fake_capture_to_bop_foundationpose_dry_run" in sequence_ids
    assert "fake_capture_to_bop_eval_dry_run" in sequence_ids
    assert "sync_to_bop_calibrated_dry_run" in sequence_ids
    assert "foundationpose_to_bop_eval_dry_run" in sequence_ids
    assert "foundationpose_runtime_to_bop_eval" in sequence_ids
    assert "aruco_to_bop_eval_dry_run" in sequence_ids
    assert "megapose_to_bop_eval_dry_run" in sequence_ids
    assert "megapose_runtime_to_bop_eval" in sequence_ids
    assert "sam6d_to_bop_eval_dry_run" in sequence_ids
    assert "sam6d_runtime_to_bop_eval" in sequence_ids


def test_run_pipeline_sequence_plan_only_writes_manifest_and_plan(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_pipeline_sequence.py",
            run_root.as_posix(),
            "--sequence",
            "sync_aruco",
            "--options-json",
            '{"sync_run": {"no_copy": true}}',
            "--plan-only",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert "Pipeline sequence sync_aruco planned" in result.stdout
    plan = json.loads((run_root / PIPELINE_SEQUENCE_PLAN).read_text())
    assert plan["plan_only"] is True
    assert "--no-copy" in plan["steps"][0]["command"]
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = manifest["stages"][0]
    assert stage["name"] == "pipeline_sequence:sync_aruco"
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][PIPELINE_SEQUENCE_PLAN] == PIPELINE_SEQUENCE_PLAN


def test_run_pipeline_sequence_cli_lists_sequence_choices() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_pipeline_sequence.py",
            "--help",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert "foundationpose_runtime_to_bop_eval" in result.stdout
    assert "megapose_runtime_to_bop_eval" in result.stdout
    assert "sam6d_runtime_to_bop_eval" in result.stdout


def test_run_pipeline_sequence_cli_rejects_unknown_sequence(tmp_path: Path) -> None:
    run_root = tmp_path / "bad-sequence"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_pipeline_sequence.py",
            run_root.as_posix(),
            "--sequence",
            "not_a_sequence",
            "--plan-only",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr
    assert not (run_root / PIPELINE_SEQUENCE_PLAN).exists()
