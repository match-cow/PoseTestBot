from __future__ import annotations

from pathlib import Path

import pytest

from posetestbot.pipeline.stages import (
    build_pipeline_job,
    list_pipeline_stages,
)


def test_pipeline_stage_builds_uv_command_with_safe_render_default(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    job = build_pipeline_job(stage_id="blenderproc_render", run_root=run_root)

    assert job.command == [
        "uv",
        "run",
        "python",
        "scripts/run_blenderproc_render_stage.py",
        run_root.as_posix(),
        "--dry-run",
    ]
    assert job.resources == ["render", "disk_io"]
    assert job.parameters["pipeline_stage"] == "blenderproc_render"
    assert job.parameters["options"] == {"dry_run": True}


def test_pipeline_stage_supports_repeated_foundationpose_outputs(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    job = build_pipeline_job(
        stage_id="bop_result_export",
        run_root=run_root,
        options={
            "foundationpose_output": [
                "processed/synchronized/realsense_1/foundationpose_output",
                "processed/synchronized/realsense_2/foundationpose_output",
            ],
            "translation_scale_to_mm": "1000",
        },
    )

    assert job.command == [
        "uv",
        "run",
        "python",
        "scripts/run_bop_result_export_stage.py",
        run_root.as_posix(),
        "--foundationpose-output",
        "processed/synchronized/realsense_1/foundationpose_output",
        "--foundationpose-output",
        "processed/synchronized/realsense_2/foundationpose_output",
        "--translation-scale-to-mm",
        "1000.0",
    ]
    assert job.parameters["options"]["translation_scale_to_mm"] == 1000.0


def test_pipeline_stage_builds_capture_plan_command(tmp_path: Path) -> None:
    run_root = tmp_path / "run"

    job = build_pipeline_job(
        stage_id="capture_plan",
        run_root=run_root,
        options={"max_frames": "2"},
    )

    assert job.command == [
        "uv",
        "run",
        "python",
        "scripts/run_capture_plan_stage.py",
        run_root.as_posix(),
        "--max-frames",
        "2",
    ]
    assert job.resources == ["disk_io"]
    assert job.parameters["pipeline_stage"] == "capture_plan"
    assert job.parameters["options"] == {
        "max_frames": 2,
        "print_json": False,
    }


def test_pipeline_stage_builds_hardware_status_command(tmp_path: Path) -> None:
    run_root = tmp_path / "run"

    job = build_pipeline_job(
        stage_id="hardware_status",
        run_root=run_root,
        options={"no_sensors": True},
    )

    assert job.command == [
        "uv",
        "run",
        "python",
        "scripts/run_hardware_status_stage.py",
        run_root.as_posix(),
        "--no-sensors",
    ]
    assert job.resources == ["camera", "disk_io"]
    assert job.parameters["pipeline_stage"] == "hardware_status"
    assert job.parameters["options"]["no_sensors"] is True
    assert job.parameters["options"]["no_runtimes"] is False


def test_pipeline_stage_builds_run_preflight_command(tmp_path: Path) -> None:
    run_root = tmp_path / "run"

    job = build_pipeline_job(
        stage_id="run_preflight",
        run_root=run_root,
        options={"check": True},
    )

    assert job.command == [
        "uv",
        "run",
        "python",
        "scripts/run_preflight.py",
        run_root.as_posix(),
        "--check",
        "--write",
    ]
    assert job.resources == ["camera", "disk_io"]
    assert job.parameters["pipeline_stage"] == "run_preflight"
    assert job.parameters["options"]["check"] is True
    assert job.parameters["options"]["write"] is True


def test_pipeline_stage_builds_capture_rehearsal_command(tmp_path: Path) -> None:
    run_root = tmp_path / "run"

    job = build_pipeline_job(
        stage_id="capture_rehearsal",
        run_root=run_root,
        options={
            "duration_s": "0.1",
            "sample_ms": "20",
            "robot_port": "30301",
            "receiver_port": "8081",
        },
    )

    assert job.command[:5] == [
        "uv",
        "run",
        "python",
        "scripts/run_capture_rehearsal_stage.py",
        run_root.as_posix(),
    ]
    assert "--duration" in job.command
    assert "0.1" in job.command
    assert "--robot-port" in job.command
    assert "30301" in job.command
    assert job.resources == ["robot_command", "disk_io"]
    assert job.parameters["pipeline_stage"] == "capture_rehearsal"
    assert job.parameters["options"]["duration_s"] == 0.1
    assert job.parameters["options"]["sample_ms"] == 20.0


def test_pipeline_stage_builds_capture_plan_preflight_command(tmp_path: Path) -> None:
    run_root = tmp_path / "run"

    job = build_pipeline_job(
        stage_id="capture_plan_preflight",
        run_root=run_root,
        options={"no_sensors": True, "json": True},
    )

    assert job.command == [
        "uv",
        "run",
        "python",
        "scripts/run_capture_plan_preflight.py",
        run_root.as_posix(),
        "--no-sensors",
        "--json",
    ]
    assert job.resources == ["disk_io"]
    assert job.parameters["pipeline_stage"] == "capture_plan_preflight"
    assert job.parameters["options"]["no_sensors"] is True
    assert job.parameters["options"]["allow_real_robot"] is False


def test_pipeline_stage_builds_capture_execution_plan_command(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    job = build_pipeline_job(
        stage_id="capture_execution_plan",
        run_root=run_root,
        options={"mode": "pose_only_fake"},
    )

    assert job.command == [
        "uv",
        "run",
        "python",
        "scripts/run_capture_execution_plan.py",
        run_root.as_posix(),
        "--mode",
        "pose_only_fake",
    ]
    assert job.resources == ["disk_io"]
    assert job.parameters["pipeline_stage"] == "capture_execution_plan"
    assert job.parameters["options"]["mode"] == "pose_only_fake"
    assert job.parameters["options"]["allow_cameras"] is False


def test_pipeline_stage_builds_capture_execution_command(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    job = build_pipeline_job(
        stage_id="capture_execution",
        run_root=run_root,
        options={
            "mode": "pose_only_fake",
            "timeout_s": "5",
            "startup_wait_s": "0.1",
        },
    )

    assert job.command == [
        "uv",
        "run",
        "python",
        "scripts/run_capture_execution_stage.py",
        run_root.as_posix(),
        "--mode",
        "pose_only_fake",
        "--timeout-s",
        "5.0",
        "--startup-wait",
        "0.1",
        "--terminate-timeout-s",
        "2.0",
    ]
    assert job.resources == ["robot_command", "camera", "disk_io"]
    assert job.parameters["pipeline_stage"] == "capture_execution"
    assert job.parameters["options"]["mode"] == "pose_only_fake"
    assert job.parameters["options"]["timeout_s"] == 5.0


def test_pipeline_stage_builds_rewrite_gate_command(tmp_path: Path) -> None:
    run_root = tmp_path / "run"

    job = build_pipeline_job(
        stage_id="rewrite_gate",
        run_root=run_root,
        options={"gate": "rewrite_full_capture.v1"},
    )

    assert job.command == [
        "uv",
        "run",
        "python",
        "scripts/run_rewrite_gate.py",
        run_root.as_posix(),
        "--gate",
        "rewrite_full_capture.v1",
        "--write",
    ]
    assert job.resources == ["disk_io"]
    assert job.parameters["pipeline_stage"] == "rewrite_gate"
    assert job.parameters["options"]["gate"] == "rewrite_full_capture.v1"
    assert job.parameters["options"]["write"] is True


def test_pipeline_stage_builds_rewrite_status_command(tmp_path: Path) -> None:
    run_root = tmp_path / "run"

    job = build_pipeline_job(
        stage_id="rewrite_status",
        run_root=run_root,
        options={
            "gate_run_root": [
                "rewrite_fake_end_to_end.v1=/tmp/fake",
                "rewrite_full_capture.v1=/tmp/real",
            ]
        },
    )

    assert job.command == [
        "uv",
        "run",
        "python",
        "scripts/run_rewrite_status.py",
        run_root.as_posix(),
        "--write",
        "--gate-run-root",
        "rewrite_fake_end_to_end.v1=/tmp/fake",
        "--gate-run-root",
        "rewrite_full_capture.v1=/tmp/real",
    ]
    assert job.resources == ["disk_io"]
    assert job.parameters["pipeline_stage"] == "rewrite_status"
    assert job.parameters["options"]["write"] is True
    assert job.parameters["options"]["gate_run_root"] == [
        "rewrite_fake_end_to_end.v1=/tmp/fake",
        "rewrite_full_capture.v1=/tmp/real",
    ]


def test_pipeline_stage_builds_calibration_preflight_command(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    job = build_pipeline_job(
        stage_id="calibration_preflight",
        run_root=run_root,
        options={"require_valid": True, "min_observations": "8"},
    )

    assert job.command == [
        "uv",
        "run",
        "python",
        "scripts/run_calibration_preflight.py",
        run_root.as_posix(),
        "--require-valid",
        "--min-observations",
        "8",
        "--max-mean-reprojection-error-px",
        "2.0",
    ]
    assert job.resources == ["disk_io"]
    assert job.parameters["pipeline_stage"] == "calibration_preflight"
    assert job.parameters["options"]["require_valid"] is True
    assert job.parameters["options"]["min_observations"] == 8


def test_pipeline_stage_builds_sync_quality_command(tmp_path: Path) -> None:
    run_root = tmp_path / "run"

    job = build_pipeline_job(
        stage_id="sync_quality",
        run_root=run_root,
        options={
            "max_dropped_frames": "2",
            "require_timestamp_source": "host_received",
        },
    )

    assert job.command == [
        "uv",
        "run",
        "python",
        "scripts/run_sync_quality.py",
        run_root.as_posix(),
        "--min-match-ratio",
        "0.8",
        "--max-dropped-frames",
        "2",
        "--max-nearest-pose-delta-ms",
        "50.0",
        "--require-timestamp-source",
        "host_received",
    ]
    assert job.resources == ["disk_io"]
    assert job.parameters["pipeline_stage"] == "sync_quality"
    assert job.parameters["options"]["max_dropped_frames"] == 2


def test_pipeline_stage_builds_calibration_observations_command(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    job = build_pipeline_job(
        stage_id="calibration_observations",
        run_root=run_root,
        options={
            "min_marker_count": "3",
            "min_observations": "4",
            "target_type": "charuco",
            "grid_size": "5x7",
            "marker_length_mm": 32.0,
            "square_length_mm": 40.0,
        },
    )

    assert job.command == [
        "uv",
        "run",
        "python",
        "scripts/run_calibration_observations.py",
        run_root.as_posix(),
        "--min-marker-count",
        "3",
        "--min-observations",
        "4",
        "--target-type",
        "charuco",
        "--grid-size",
        "5x7",
        "--marker-length-mm",
        "32.0",
        "--square-length-mm",
        "40.0",
    ]
    assert job.resources == ["disk_io"]
    assert job.parameters["pipeline_stage"] == "calibration_observations"


def test_pipeline_stage_builds_calibration_candidates_command(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    job = build_pipeline_job(
        stage_id="calibration_candidates",
        run_root=run_root,
        options={
            "observations": "calibration_observations.json",
            "min_observations": "4",
            "target_to_reference": "target_to_reference.json",
            "max_translation_residual_mm": "25",
            "max_rotation_residual_deg": "8",
        },
    )

    assert job.command == [
        "uv",
        "run",
        "python",
        "scripts/run_calibration_candidates.py",
        run_root.as_posix(),
        "--observations",
        "calibration_observations.json",
        "--min-observations",
        "4",
        "--target-to-reference",
        "target_to_reference.json",
        "--max-translation-residual-mm",
        "25.0",
        "--max-rotation-residual-deg",
        "8.0",
    ]
    assert job.resources == ["disk_io"]
    assert job.parameters["pipeline_stage"] == "calibration_candidates"


def test_pipeline_stage_builds_calibration_solver_command(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    job = build_pipeline_job(
        stage_id="calibration_solver",
        run_root=run_root,
        options={
            "observations": "calibration_observations.json",
            "min_observations": "4",
            "target_to_reference": "target_to_reference.json",
            "hand_eye_method": "park",
            "max_translation_residual_mm": "25",
            "max_rotation_residual_deg": "8",
            "holdout_fraction": "0.25",
            "compare_hand_eye_methods": "true",
        },
    )

    assert job.command == [
        "uv",
        "run",
        "python",
        "scripts/run_calibration_solver.py",
        run_root.as_posix(),
        "--observations",
        "calibration_observations.json",
        "--min-observations",
        "4",
        "--target-to-reference",
        "target_to_reference.json",
        "--hand-eye-method",
        "park",
        "--max-translation-residual-mm",
        "25.0",
        "--max-rotation-residual-deg",
        "8.0",
        "--holdout-fraction",
        "0.25",
        "--compare-hand-eye-methods",
    ]
    assert job.resources == ["disk_io"]
    assert job.parameters["pipeline_stage"] == "calibration_solver"


def test_pipeline_stage_builds_calibration_validation_command(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    job = build_pipeline_job(
        stage_id="calibration_validation",
        run_root=run_root,
        options={
            "candidates": "calibration_candidates.json",
            "profiles": "calibration_profiles_from_observations.json",
            "min_inliers": "4",
            "max_mean_translation_residual_mm": "3.5",
            "max_mean_rotation_residual_deg": "1.5",
            "max_outlier_ratio": "0.2",
            "promote": True,
            "output_profiles": "calibration_profiles.json",
            "operator": "operator-a",
        },
    )

    assert job.command == [
        "uv",
        "run",
        "python",
        "scripts/run_calibration_validation.py",
        run_root.as_posix(),
        "--candidates",
        "calibration_candidates.json",
        "--profiles",
        "calibration_profiles_from_observations.json",
        "--min-inliers",
        "4",
        "--max-mean-translation-residual-mm",
        "3.5",
        "--max-mean-rotation-residual-deg",
        "1.5",
        "--max-outlier-ratio",
        "0.2",
        "--promote",
        "--output-profiles",
        "calibration_profiles.json",
        "--operator",
        "operator-a",
    ]
    assert job.resources == ["disk_io"]
    assert job.parameters["pipeline_stage"] == "calibration_validation"


def test_pipeline_stage_builds_bop_export_multiview_command(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    job = build_pipeline_job(
        stage_id="bop_export",
        run_root=run_root,
        options={
            "object_folder": "object_models",
            "write_multiview_targets": True,
        },
    )

    assert job.command == [
        "uv",
        "run",
        "python",
        "scripts/run_bop_export_stage.py",
        run_root.as_posix(),
        "--object-folder",
        "object_models",
        "--write-multiview-targets",
    ]
    assert job.resources == ["disk_io"]
    assert job.parameters["pipeline_stage"] == "bop_export"
    assert job.parameters["options"]["write_multiview_targets"] is True


def test_pipeline_stage_builds_bop_export_coco_command(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    job = build_pipeline_job(
        stage_id="bop_export",
        run_root=run_root,
        options={
            "object_folder": "object_models",
            "write_coco_annotations": True,
        },
    )

    assert job.command == [
        "uv",
        "run",
        "python",
        "scripts/run_bop_export_stage.py",
        run_root.as_posix(),
        "--object-folder",
        "object_models",
        "--write-coco-annotations",
    ]
    assert job.resources == ["disk_io"]
    assert job.parameters["pipeline_stage"] == "bop_export"
    assert job.parameters["options"]["write_coco_annotations"] is True


def test_pipeline_stage_supports_aruco_result_export_options(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    job = build_pipeline_job(
        stage_id="bop_result_export",
        run_root=run_root,
        options={
            "source": "aruco",
            "aruco_pose_file": [
                "processed/synchronized/realsense_1/aruco_pose_estimation.json"
            ],
            "aruco_object_name": "aruco",
            "min_marker_count": "3",
            "translation_scale_to_mm": "1",
        },
    )

    assert job.command == [
        "uv",
        "run",
        "python",
        "scripts/run_bop_result_export_stage.py",
        run_root.as_posix(),
        "--source",
        "aruco",
        "--aruco-pose-file",
        "processed/synchronized/realsense_1/aruco_pose_estimation.json",
        "--aruco-object-name",
        "aruco",
        "--min-marker-count",
        "3",
        "--translation-scale-to-mm",
        "1.0",
    ]
    assert job.parameters["options"]["source"] == "aruco"
    assert job.parameters["options"]["min_marker_count"] == 3
    assert job.parameters["options"]["translation_scale_to_mm"] == 1.0


def test_pipeline_stage_supports_megapose_and_sam6d_result_export_options(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    megapose_job = build_pipeline_job(
        stage_id="bop_result_export",
        run_root=run_root,
        options={
            "source": "megapose",
            "megapose_output": [
                "processed/synchronized/realsense_1/megapose_obj0_output"
            ],
        },
    )
    sam6d_job = build_pipeline_job(
        stage_id="bop_result_export",
        run_root=run_root,
        options={
            "source": "sam6d",
            "sam6d_output": [
                "processed/synchronized/realsense_1/sam6d_obj0_output"
            ],
        },
    )

    assert "--megapose-output" in megapose_job.command
    assert megapose_job.parameters["options"]["source"] == "megapose"
    assert "--sam6d-output" in sam6d_job.command
    assert sam6d_job.parameters["options"]["source"] == "sam6d"


def test_pipeline_stage_builds_safe_foundationpose_dry_run_command(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    job = build_pipeline_job(
        stage_id="foundationpose",
        run_root=run_root,
        options={
            "foundationpose_folder": "/opt/FoundationPose",
            "no_tracking": True,
            "object_id": "1",
        },
    )

    assert job.command == [
        "uv",
        "run",
        "python",
        "scripts/run_foundationpose_stage.py",
        run_root.as_posix(),
        "--foundationpose-folder",
        "/opt/FoundationPose",
        "--no-tracking",
        "--object-id",
        "1",
        "--dry-run",
    ]
    assert job.resources == ["estimator", "disk_io"]
    assert job.parameters["options"]["dry_run"] is True
    assert job.parameters["options"]["object_id"] == 1


def test_pipeline_stage_builds_megapose_and_sam6d_dry_run_commands(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"

    megapose_job = build_pipeline_job(
        stage_id="megapose",
        run_root=run_root,
        options={
            "wrapper_script": "/opt/megapose_wrapper.py",
            "model": "megapose-1.0-RGBD",
            "roi_scale": "1.25",
            "object_id": "1",
            "result_id": "rgbd",
        },
    )
    sam6d_job = build_pipeline_job(
        stage_id="sam6d",
        run_root=run_root,
        options={
            "wrapper_script": "/opt/sam6d_wrapper.py",
            "segmentor_model": "sam-hq",
            "object_id": "0",
            "result_id": "sam-hq",
        },
    )

    assert megapose_job.command == [
        "uv",
        "run",
        "python",
        "scripts/run_megapose_stage.py",
        run_root.as_posix(),
        "--wrapper-script",
        "/opt/megapose_wrapper.py",
        "--model",
        "megapose-1.0-RGBD",
        "--roi-scale",
        "1.25",
        "--object-id",
        "1",
        "--result-id",
        "rgbd",
        "--dry-run",
    ]
    assert megapose_job.resources == ["estimator", "disk_io"]
    assert megapose_job.parameters["options"]["roi_scale"] == 1.25
    assert megapose_job.parameters["options"]["object_id"] == 1
    assert sam6d_job.command == [
        "uv",
        "run",
        "python",
        "scripts/run_sam6d_stage.py",
        run_root.as_posix(),
        "--wrapper-script",
        "/opt/sam6d_wrapper.py",
        "--segmentor-model",
        "sam-hq",
        "--object-id",
        "0",
        "--result-id",
        "sam-hq",
        "--dry-run",
    ]
    assert sam6d_job.resources == ["estimator", "disk_io"]
    assert sam6d_job.parameters["options"]["object_id"] == 0


def test_pipeline_stage_requires_bop_evaluation_result_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="result_file"):
        build_pipeline_job(stage_id="bop_evaluation", run_root=tmp_path / "run")


def test_pipeline_stage_rejects_unknown_options(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown"):
        build_pipeline_job(
            stage_id="sync_run",
            run_root=tmp_path / "run",
            options={"unknown": "value"},
        )


def test_pipeline_stage_rejects_invalid_timestamp_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timestamp_source"):
        build_pipeline_job(
            stage_id="sync_run",
            run_root=tmp_path / "run",
            options={"timestamp_source": "wall_clockish"},
        )


def test_pipeline_stage_listing_is_json_friendly() -> None:
    stages = list_pipeline_stages()
    sync_stage = next(stage for stage in stages if stage["id"] == "sync_run")
    capture_stage = next(stage for stage in stages if stage["id"] == "capture_plan")
    rehearsal_stage = next(
        stage for stage in stages if stage["id"] == "capture_rehearsal"
    )
    preflight_stage = next(
        stage for stage in stages if stage["id"] == "capture_plan_preflight"
    )
    execution_stage = next(
        stage for stage in stages if stage["id"] == "capture_execution_plan"
    )
    capture_execution_stage = next(
        stage for stage in stages if stage["id"] == "capture_execution"
    )
    synthetic_stage = next(
        stage for stage in stages if stage["id"] == "synthetic_rgbd_fixture"
    )
    calibration_stage = next(
        stage for stage in stages if stage["id"] == "calibration_preflight"
    )
    sync_quality_stage = next(
        stage for stage in stages if stage["id"] == "sync_quality"
    )
    calibration_observations_stage = next(
        stage for stage in stages if stage["id"] == "calibration_observations"
    )
    aruco_coverage_stage = next(
        stage for stage in stages if stage["id"] == "aruco_coverage"
    )
    calibration_candidates_stage = next(
        stage for stage in stages if stage["id"] == "calibration_candidates"
    )
    calibration_solver_stage = next(
        stage for stage in stages if stage["id"] == "calibration_solver"
    )
    calibration_validation_stage = next(
        stage for stage in stages if stage["id"] == "calibration_validation"
    )
    metric_report_stage = next(
        stage for stage in stages if stage["id"] == "metric_report_export"
    )
    synthetic_bop_stage = next(
        stage for stage in stages if stage["id"] == "synthetic_bop_results"
    )

    assert sync_stage["label"] == "Non-destructive Sync"
    assert sync_stage["parameters"][0]["name"] == "output_root"
    assert isinstance(sync_stage["resources"], list)
    assert capture_stage["script"] == "scripts/run_capture_plan_stage.py"
    assert preflight_stage["script"] == "scripts/run_capture_plan_preflight.py"
    assert execution_stage["script"] == "scripts/run_capture_execution_plan.py"
    assert capture_execution_stage["script"] == "scripts/run_capture_execution_stage.py"
    assert synthetic_stage["script"] == "scripts/create_synthetic_rgbd_fixture.py"
    assert calibration_stage["script"] == "scripts/run_calibration_preflight.py"
    assert sync_quality_stage["script"] == "scripts/run_sync_quality.py"
    assert calibration_observations_stage["script"] == (
        "scripts/run_calibration_observations.py"
    )
    assert aruco_coverage_stage["script"] == "scripts/run_aruco_coverage_stage.py"
    assert calibration_candidates_stage["script"] == (
        "scripts/run_calibration_candidates.py"
    )
    assert calibration_solver_stage["script"] == "scripts/run_calibration_solver.py"
    assert calibration_validation_stage["script"] == (
        "scripts/run_calibration_validation.py"
    )
    assert metric_report_stage["script"] == "scripts/run_metric_report_export_stage.py"
    assert synthetic_bop_stage["script"] == "scripts/create_synthetic_bop_results.py"
    assert rehearsal_stage["script"] == "scripts/run_capture_rehearsal_stage.py"
