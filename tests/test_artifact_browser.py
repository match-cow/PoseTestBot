from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

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
    ACCURACY_ARUCO_HRC_HUB,
    ACCURACY_HRC_HUB,
    ALL_RESULTS_JSON,
    ARUCO_COVERAGE_REPORT,
    BOP_COCO_ANNOTATIONS,
    BOP_DIR,
    BOP_EVALUATION_PLAN,
    BOP_EVALUATION_REPORT,
    BOP_EXPORT_MANIFEST,
    BOP_FRAME_MAP_JSON,
    BOP_MULTIVIEW_TARGETS,
    BOP_RESULT_EXPORT_MANIFEST,
    BOP_TARGETS_BOP19,
    CAPTURE_EXECUTION_LOGS_DIR,
    CAPTURE_EXECUTION_PLAN,
    CAPTURE_EXECUTION_REPORT,
    CAPTURE_EXECUTION_STATUS,
    CAPTURE_PLAN,
    CAPTURE_PLAN_PREFLIGHT_REPORT,
    CAPTURE_REHEARSAL_REPORT,
    CALIBRATION_CANDIDATES,
    CALIBRATION_OBSERVATIONS,
    CALIBRATION_PREFLIGHT_REPORT,
    CALIBRATION_PROFILES_FROM_OBSERVATIONS,
    CALIBRATION_PROFILES_SOLVED,
    CALIBRATION_SOLVER_REPORT,
    CALIBRATION_VALIDATION_REPORT,
    DATASET_MANIFEST,
    DEPTH_DIR,
    EVALUATION_DIR,
    FOUNDATIONPOSE_PLAN,
    HARDWARE_STATUS_REPORT,
    MEGAPOSE_PLAN,
    METRIC_REPORT_JSON,
    METRICS_DIR,
    MODELS_DIR,
    PIPELINE_SEQUENCE_PLAN,
    RGB_DIR,
    RESULTS_DIR,
    REWRITE_GATE_REPORT,
    REWRITE_STATUS_REPORT,
    RUN_CONFIG,
    RUN_PREFLIGHT_REPORT,
    SAM6D_PLAN,
    SYNC_QUALITY_REPORT,
    SYNC_REPORT,
)
from posetestbot.io.manifest import (
    create_run_manifest,
    set_manifest_artifact,
    upsert_stage,
    write_run_manifest,
)
from posetestbot.pipeline.run_config import create_run_config, write_run_config


def ready_bop_evaluation_checks() -> list[dict[str, object]]:
    return [
        {"name": "result_file", "ok": True},
        {"name": "bop_root", "ok": True},
        {"name": "dataset_folder", "ok": True},
        {"name": "targets_file", "ok": True},
        {"name": "models_folder", "ok": True},
        {"name": "models_info", "ok": True},
        {"name": "model_files", "ok": True, "value": 1},
        {"name": "eval_script", "ok": False},
    ]


def create_artifact_fixture(tmp_path: Path) -> Path:
    run_root = tmp_path / "run"
    run_root.mkdir()
    sync_report = run_root / "processed" / "synchronized" / "realsense_123" / SYNC_REPORT
    sync_report.parent.mkdir(parents=True)
    sync_report.write_text('{"matched_frames": 2}\n')
    (run_root / SYNC_QUALITY_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "sync_quality_report.v1",
                "overall_status": "warning",
                "sensor_count": 1,
                "total_frames": 3,
                "matched_frames": 2,
                "dropped_frames": 1,
                "overall_match_ratio": 2 / 3,
                "checks": [{"name": "sync_match_ratio:realsense_123", "status": "warning"}],
                "sensors": [{"sensor_name": "realsense_123"}],
            }
        )
    )
    pose_accuracy = {
        "foundationpose_est5_track2_obj0": {
            "motion_a": {
                "AP_p": "1.50",
                "RP_i": 0.75,
                "RP_a": [0.1, -0.1],
                "x": [1.0, 2.0],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
            },
            "all_motions": {
                "AP_p": "1.25",
                "ap_x": 1.0,
                "ap_y": 0.5,
                "ap_z": 0.25,
                "RP_i": "0.80",
                "RP_a": [0.1, -0.1],
                "RP_b": [0.2, -0.2],
                "RP_c": [0.3, -0.3],
                "x": [1.0, 2.0, 3.0],
                "y": [0.0, 0.0, 0.0],
                "z": [0.0, 0.0, 0.0],
            },
        }
    }
    aruco_accuracy = {
        "ArUco_accuracy": {
            "all_motions": {
                "AP_p": 2.5,
                "RP_i": 1.5,
                "x": [2.0],
                "y": [0.0],
                "z": [0.0],
            }
        }
    }
    (sync_report.parent / ACCURACY_HRC_HUB).write_text(json.dumps(pose_accuracy))
    (sync_report.parent / ACCURACY_ARUCO_HRC_HUB).write_text(
        json.dumps(aruco_accuracy)
    )
    (run_root / ALL_RESULTS_JSON).write_text(
        json.dumps(
            {
                "experiment_720p_6_0.2": [
                    {
                        "cube": [
                            {
                                "realsense_123": [
                                    pose_accuracy,
                                    aruco_accuracy,
                                ]
                            }
                        ]
                    }
                ]
            }
        )
    )
    (run_root / PIPELINE_SEQUENCE_PLAN).write_text(
        json.dumps(
            {
                "schema_version": "pipeline_sequence_plan.v1",
                "sequence_id": "sync_to_bop_calibrated_dry_run",
                "plan_only": True,
                "resources": ["cpu", "disk_io", "render"],
                "steps": [
                    {
                        "id": "sync_run",
                        "stage_id": "sync_run",
                        "command": [
                            "uv",
                            "run",
                            "python",
                            "scripts/sync_run_non_destructive.py",
                            run_root.as_posix(),
                        ],
                        "options": {},
                    },
                    {
                        "id": "blenderproc_prepare",
                        "stage_id": "blenderproc_prepare",
                        "command": [
                            "uv",
                            "run",
                            "python",
                            "scripts/run_blenderproc_prepare_stage.py",
                            run_root.as_posix(),
                        ],
                        "options": {
                            "calibration_profiles": (
                                "profiles/lab_calibration.json"
                            ),
                        },
                    },
                    {
                        "id": "bop_export",
                        "stage_id": "bop_export",
                        "command": [
                            "uv",
                            "run",
                            "python",
                            "scripts/run_bop_export_stage.py",
                            run_root.as_posix(),
                        ],
                        "options": {
                            "calibration_profiles": (
                                "profiles/lab_calibration.json"
                            ),
                        },
                    },
                ],
            }
        )
    )
    (run_root / FOUNDATIONPOSE_PLAN).write_text(
        json.dumps(
            {
                "schema_version": "foundationpose_plan.v1",
                "dry_run": True,
                "input_folder": (run_root / "processed" / "synchronized").as_posix(),
                "foundationpose_folder": "/opt/FoundationPose",
                "no_tracking": False,
                "est_refine_iter": 5,
                "track_refine_iter": 2,
                "object_id": 0,
                "command": [
                    "uv",
                    "run",
                    "python",
                    "scripts/foundationpose_wrapper_multi.py",
                ],
                "jobs": [
                    {
                        "sensor_name": "realsense_123",
                        "sensor_folder": (
                            run_root / "processed" / "synchronized" / "realsense_123"
                        ).as_posix(),
                        "object_name": "cube",
                        "object_id": 0,
                        "expected_output_folder": (
                            run_root
                            / "processed"
                            / "synchronized"
                            / "realsense_123"
                            / "foundationpose_est5_track2_obj0_output"
                        ).as_posix(),
                    }
                ],
            }
        )
    )
    (run_root / MEGAPOSE_PLAN).write_text(
        json.dumps(
            {
                "schema_version": "megapose_plan.v1",
                "dry_run": True,
                "estimator_id": "megapose",
                "input_folder": (run_root / "processed" / "synchronized").as_posix(),
                "wrapper_script": "/opt/megapose_wrapper.py",
                "wrapper_exists": False,
                "object_id": 0,
                "result_id": "rgbd",
                "command": ["uv", "run", "python", "/opt/megapose_wrapper.py"],
                "options": {"model": "megapose-1.0-RGBD", "roi_scale": 1.25},
                "jobs": [
                    {
                        "sensor_name": "realsense_123",
                        "sensor_folder": (
                            run_root / "processed" / "synchronized" / "realsense_123"
                        ).as_posix(),
                        "object_name": "cube",
                        "object_id": 0,
                        "expected_output_folder": (
                            run_root
                            / "processed"
                            / "synchronized"
                            / "realsense_123"
                            / "megapose_rgbd_obj0_output"
                        ).as_posix(),
                    }
                ],
            }
        )
    )
    (run_root / SAM6D_PLAN).write_text(
        json.dumps(
            {
                "schema_version": "sam6d_plan.v1",
                "dry_run": True,
                "estimator_id": "sam6d",
                "input_folder": (run_root / "processed" / "synchronized").as_posix(),
                "wrapper_script": "/opt/sam6d_wrapper.py",
                "wrapper_exists": False,
                "object_id": 0,
                "result_id": "sam-hq",
                "command": ["uv", "run", "python", "/opt/sam6d_wrapper.py"],
                "options": {"segmentor_model": "sam-hq"},
                "jobs": [
                    {
                        "sensor_name": "realsense_123",
                        "sensor_folder": (
                            run_root / "processed" / "synchronized" / "realsense_123"
                        ).as_posix(),
                        "object_name": "cube",
                        "object_id": 0,
                        "expected_output_folder": (
                            run_root
                            / "processed"
                            / "synchronized"
                            / "realsense_123"
                            / "sam6d_sam-hq_obj0_output"
                        ).as_posix(),
                    }
                ],
            }
        )
    )
    (run_root / RUN_CONFIG).write_text(
        json.dumps(
            {
                "schema_version": "run_config.v1",
                "run_name": "run",
                "run_root": run_root.as_posix(),
                "robot_profile": {"mode": "fake"},
                "object_folder": "object_models",
                "calibration_profiles": "profiles/lab_calibration.json",
                "capture": {
                    "resolution": "720p",
                    "fps": 6,
                    "sensors": [
                        {
                            "sensor_type": "realsense_d435",
                            "device_id": "123",
                            "mounting_mode": "eye_in_hand",
                        }
                    ],
                },
                "pipeline": {
                    "sequence_id": "sync_to_bop_calibrated_dry_run",
                    "plan_only": True,
                },
            }
        )
    )
    (run_root / RUN_PREFLIGHT_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "run_preflight.v1",
                "run_root": run_root.as_posix(),
                "overall_status": "warning",
                "checks": [
                    {"name": "run_config", "status": "ok"},
                    {"name": "sensor_status", "status": "warning"},
                    {"name": "runtime_requirements", "status": "ok"},
                ],
                "config": {
                    "robot_profile": {"mode": "fake"},
                },
                "sequence_plan": {
                    "sequence_id": "sync_to_bop_calibrated_dry_run",
                    "steps": [{"id": "sync_run"}, {"id": "sync_quality"}],
                },
                "robot_status": {"selected_profile": {"mode": "fake"}},
                "sensor_status": {
                    "total_connected": 4,
                    "all_expected_connected": False,
                },
                "runtime_status": {
                    "available_count": 3,
                    "runtime_count": 5,
                },
            }
        )
    )
    (run_root / HARDWARE_STATUS_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "hardware_status_report.v1",
                "overall_status": "warning",
                "checks": [
                    {"name": "robot_profile", "status": "ok"},
                    {"name": "sensor_status", "status": "warning"},
                ],
                "robot_status": {"selected_profile": {"mode": "fake"}},
                "sensor_status": {
                    "total_connected": 4,
                    "all_expected_connected": False,
                },
                "runtime_status": {
                    "available_count": 2,
                    "runtime_count": 4,
                },
            }
        )
    )
    (run_root / CAPTURE_PLAN).write_text(
        json.dumps(
            {
                "schema_version": "capture_plan.v1",
                "dry_run": True,
                "robot_profile": {"mode": "fake"},
                "sensors": [{"folder": "realsense_123"}],
                "commands": [
                    {"role": "robot_controller", "name": "fake_iiwa_controller"},
                    {"role": "sensor_capture", "name": "realsense_123"},
                    {"role": "robot_pose_receiver", "name": "pose_receiver"},
                ],
            }
        )
    )
    (run_root / CAPTURE_REHEARSAL_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "capture_rehearsal_report.v1",
                "status": "succeeded",
                "mode": "pose_only_fake",
                "raw_pose_count": 6,
            }
        )
    )
    (run_root / CAPTURE_PLAN_PREFLIGHT_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "capture_plan_preflight.v1",
                "overall_status": "ok",
                "checks": [{"name": "robot_mode", "status": "ok"}],
            }
        )
    )
    (run_root / CAPTURE_EXECUTION_PLAN).write_text(
        json.dumps(
            {
                "schema_version": "capture_execution_plan.v1",
                "status": "ok",
                "mode": "pose_only_fake",
                "ready_to_execute": True,
                "selected_roles": ["robot_controller", "robot_pose_receiver"],
                "selected_commands": [
                    {"role": "robot_controller"},
                    {"role": "robot_pose_receiver"},
                ],
                "skipped_commands": [{"role": "sensor_capture"}],
            }
        )
    )
    (run_root / CAPTURE_EXECUTION_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "capture_execution_report.v1",
                "status": "succeeded",
                "mode": "pose_only_fake",
                "raw_pose_count": 6,
                "processes": [
                    {
                        "role": "robot_controller",
                        "status": "succeeded",
                        "termination_reason": "exited_after_receiver",
                        "elapsed_s": 1.5,
                    },
                    {
                        "role": "robot_pose_receiver",
                        "status": "succeeded",
                        "termination_reason": "receiver_completed",
                        "elapsed_s": 1.0,
                    },
                ],
            }
        )
    )
    (run_root / CAPTURE_EXECUTION_STATUS).write_text(
        json.dumps(
            {
                "schema_version": "capture_execution_status.v1",
                "status": "running",
                "mode": "pose_only_fake",
                "active_process_count": 1,
                "raw_pose_count": 2,
                "selected_roles": ["robot_controller", "robot_pose_receiver"],
                "processes": [
                    {"role": "robot_controller", "status": "running", "active": True},
                    {
                        "role": "robot_pose_receiver",
                        "status": "planned",
                        "active": False,
                    },
                ],
            }
        )
    )
    (run_root / CAPTURE_EXECUTION_LOGS_DIR).mkdir()
    (run_root / CAPTURE_EXECUTION_LOGS_DIR / "00_fake_iiwa_controller.log").write_text(
        "fake controller log\n"
    )
    (run_root / CALIBRATION_PREFLIGHT_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "calibration_preflight.v1",
                "overall_status": "warning",
                "profile_count": 1,
                "sensor_count": 2,
                "matched_sensor_count": 1,
                "profile_path": "calibration_profiles.json",
                "matched_sensors": [
                    {
                        "sensor_name": "realsense_123",
                        "profile_id": "realsense_123_profile",
                    }
                ],
                "require_valid": True,
                "min_observations": 8,
                "max_mean_reprojection_error_px": 1.5,
                "checks": [
                    {"name": "profile_match:realsense_123", "status": "ok"},
                    {"name": "profile_status:realsense_123", "status": "warning"},
                ],
            }
        )
    )
    (run_root / CALIBRATION_OBSERVATIONS).write_text(
        json.dumps(
            {
                "schema_version": "calibration_observations.v1",
                "overall_status": "warning",
                "target": {
                    "target_type": "aruco_grid",
                    "dictionary": "DICT_5X5_50",
                },
                "sensor_count": 1,
                "frame_count": 3,
                "observation_count": 2,
                "rejected_count": 1,
                "motion_count": 1,
                "checks": [
                    {
                        "name": "calibration_observations:realsense_123",
                        "status": "warning",
                    }
                ],
            }
        )
    )
    (run_root / CALIBRATION_CANDIDATES).write_text(
        json.dumps(
            {
                "schema_version": "calibration_candidates.v1",
                "overall_status": "warning",
                "sensor_count": 1,
                "profile_count": 1,
                "candidate_count": 2,
                "inlier_count": 1,
                "outlier_count": 1,
                "checks": [
                    {
                        "name": "candidate_observations:realsense_123",
                        "status": "warning",
                    }
                ],
            }
        )
    )
    (run_root / CALIBRATION_PROFILES_FROM_OBSERVATIONS).write_text(
        json.dumps(
            {
                "schema_version": "calibration.v1",
                "profiles": [
                    {
                        "profile_id": "realsense_123_static_aruco_candidate",
                        "status": "needs_validation",
                    }
                ],
            }
        )
    )
    (run_root / CALIBRATION_SOLVER_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "calibration_solver.v1",
                "overall_status": "warning",
                "sensor_count": 1,
                "profile_count": 1,
                "observation_count": 2,
                "inlier_count": 1,
                "outlier_count": 1,
                "hand_eye_method": "tsai",
                "method_comparisons": [
                    {"method": "opencv_calibrateHandEye_tsai", "status": "ok"},
                    {"method": "opencv_calibrateHandEye_park", "status": "warning"},
                ],
                "checks": [
                    {
                        "name": "solver_inliers:realsense_123",
                        "status": "warning",
                    }
                ],
            }
        )
    )
    (run_root / CALIBRATION_PROFILES_SOLVED).write_text(
        json.dumps(
            {
                "schema_version": "calibration.v1",
                "profiles": [
                    {
                        "profile_id": "realsense_123_static_aruco_solved",
                        "status": "needs_validation",
                        "method": "static_target_reference_transform_average",
                    }
                ],
            }
        )
    )
    (run_root / CALIBRATION_VALIDATION_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "calibration_validation.v1",
                "overall_status": "ok",
                "profile_count": 1,
                "promotable_profile_count": 1,
                "candidate_count": 2,
                "inlier_count": 1,
                "outlier_count": 1,
                "promotion": {"requested": False, "promoted": False, "path": None},
                "checks": [{"name": "profile_inliers:p1", "status": "ok"}],
            }
        )
    )

    bop_scene = run_root / BOP_DIR / "realsense_123" / "test" / "000001"
    bop_scene.mkdir(parents=True)
    (bop_scene / RGB_DIR).mkdir()
    cv2.imwrite(
        (bop_scene / RGB_DIR / "000000.png").as_posix(),
        np.zeros((3, 4, 3), dtype=np.uint8),
    )
    (bop_scene / DEPTH_DIR).mkdir()
    cv2.imwrite(
        (bop_scene / DEPTH_DIR / "000000.png").as_posix(),
        np.zeros((3, 4), dtype=np.uint16),
    )
    scene_camera = bop_scene / "scene_camera.json"
    scene_camera.write_text(
        '{"0": {"cam_K": [1, 0, 2, 0, 1, 1, 0, 0, 1], "depth_scale": 1.0}}\n'
    )
    (bop_scene / "scene_gt.json").write_text('{"0": [{"obj_id": 1}]}\n')
    (bop_scene / "scene_gt_info.json").write_text(
        '{"0": [{"bbox_obj": [0, 0, 4, 3]}]}\n'
    )
    (bop_scene / BOP_FRAME_MAP_JSON).write_text(
        json.dumps(
            {
                "0": {
                    "source_rgb": "rgb/raw_000010.png",
                    "source_depth": "depth/raw_000010.png",
                    "bop_rgb": "rgb/000000.png",
                    "bop_depth": "depth/000000.png",
                }
            }
        )
    )
    (bop_scene / "mask").mkdir()
    cv2.imwrite(
        (bop_scene / "mask" / "000000_000000.png").as_posix(),
        np.ones((3, 4), dtype=np.uint8) * 255,
    )
    models_folder = run_root / BOP_DIR / MODELS_DIR
    models_folder.mkdir()
    model_path = models_folder / "obj_000001.ply"
    model_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "element face 0",
                "property list uchar int vertex_indices",
                "end_header",
                "-10 -5 0",
                "10 -5 0",
                "10 5 0",
                "-10 5 0",
                "",
            ]
        )
    )
    models_info = models_folder / "models_info.json"
    models_info.write_text(
        json.dumps(
            {
                "1": {
                    "source_name": "cube",
                    "diameter": 22.36,
                    "min_x": -10.0,
                    "min_y": -5.0,
                    "min_z": 0.0,
                    "size_x": 20.0,
                    "size_y": 10.0,
                    "size_z": 0.0,
                }
            }
        )
    )
    bop_manifest = run_root / BOP_DIR / BOP_EXPORT_MANIFEST
    bop_targets = run_root / BOP_DIR / BOP_TARGETS_BOP19
    bop_targets.write_text(
        json.dumps(
            [
                {
                    "scene_id": 1,
                    "im_id": 0,
                    "obj_id": 1,
                    "inst_count": 1,
                }
            ]
        )
    )
    multiview_targets = run_root / BOP_DIR / BOP_MULTIVIEW_TARGETS
    multiview_targets.write_text(
        json.dumps(
            {
                "schema_version": "posetestbot_bop_multiview_targets.v1",
                "split": "test",
                "scene_count": 1,
                "object_count": 1,
                "targets": [
                    {
                        "obj_id": 1,
                        "sensor_names": ["realsense_123"],
                        "scene_ids": [1],
                        "view_count": 1,
                        "instance_count": 1,
                        "views": [
                            {
                                "scene_id": 1,
                                "sensor_name": "realsense_123",
                                "im_id": 0,
                                "inst_count": 1,
                            }
                        ],
                    }
                ],
            }
        )
    )
    coco_annotations = run_root / BOP_DIR / BOP_COCO_ANNOTATIONS
    coco_annotations.write_text(
        json.dumps(
            {
                "schema_version": "posetestbot_coco_annotations.v1",
                "images": [
                    {
                        "id": 1,
                        "file_name": "realsense_123/test/000001/rgb/000000.png",
                    }
                ],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1}],
                "categories": [{"id": 1, "name": "cube"}],
            }
        )
    )
    bop_manifest.write_text(
        json.dumps(
            {
                "schema_version": "bop_export_manifest.v1",
                "exports": [
                    {
                        "sensor_name": "realsense_123",
                        "scene_id": 1,
                        "split": "test",
                        "scene_folder": bop_scene.as_posix(),
                        "artifacts": {"scene_camera": scene_camera.as_posix()},
                    }
                ],
                "object_models": [
                    {
                        "object_name": "cube",
                        "obj_id": 1,
                        "source_path": model_path.as_posix(),
                        "bop_path": model_path.as_posix(),
                    }
                ],
                "targets_path": bop_targets.as_posix(),
                "multiview_targets_path": multiview_targets.as_posix(),
                "coco_annotations_path": coco_annotations.as_posix(),
            }
        )
    )

    result_file = run_root / RESULTS_DIR / BOP_DIR / "foundationpose_bop-test.csv"
    result_file.parent.mkdir(parents=True)
    result_file.write_text(
        "scene_id,im_id,obj_id,score,R,t,time\n"
        "1,0,1,1,1 0 0 0 1 0 0 0 1,0 0 10,-1\n"
    )
    (run_root / BOP_RESULT_EXPORT_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": "bop_result_export_manifest.v1",
                "output_folder": result_file.parent.as_posix(),
                "results": [
                    {
                        "filename": result_file.name,
                        "path": result_file.as_posix(),
                    }
                ],
            }
        )
    )
    eval_path = run_root / EVALUATION_DIR / "bop_toolkit" / "foundationpose_bop-test"
    eval_path.mkdir(parents=True)
    eval_scores = eval_path / "scores_bop19.json"
    eval_scores.write_text('{"bop19_average_recall": 0.75}\n')
    (run_root / BOP_EVALUATION_PLAN).write_text(
        json.dumps(
            {
                "schema_version": "bop_evaluation_plan.v1",
                "dry_run": False,
                "bop_root": (run_root / BOP_DIR).as_posix(),
                "dataset_folder": (run_root / BOP_DIR).as_posix(),
                "eval_path": eval_path.as_posix(),
                "result": {
                    "filename": result_file.name,
                    "path": result_file.as_posix(),
                },
                "command": [
                    "python",
                    "bop_toolkit/scripts/eval_bop19_pose.py",
                    f"--result_filenames={result_file.name}",
                ],
                "environment": {"BOP_PATH": run_root.as_posix()},
            }
        )
    )
    (run_root / BOP_EVALUATION_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "bop_evaluation_report.v1",
                "status": "succeeded",
                "dry_run": False,
                "eval_path": eval_path.as_posix(),
                "result": {
                    "filename": result_file.name,
                    "path": result_file.as_posix(),
                },
                "checks": ready_bop_evaluation_checks(),
                "output_artifacts": [
                    {
                        "path": eval_scores.as_posix(),
                        "relative_path": "scores_bop19.json",
                        "size_bytes": eval_scores.stat().st_size,
                    }
                ],
                "score_summary": {
                    "score_file_count": 1,
                    "metrics": {
                        "bop19_average_recall": 0.75,
                    },
                    "files": [
                        {
                            "path": eval_scores.as_posix(),
                            "relative_path": "scores_bop19.json",
                            "metrics": {
                                "bop19_average_recall": 0.75,
                            },
                        }
                    ],
                },
            }
        )
    )

    manifest = create_run_manifest(run_root)
    set_manifest_artifact(manifest, "top_sync_report", sync_report, run_root=run_root)
    upsert_stage(
        manifest,
        name="sync:realsense_123",
        status="succeeded",
        artifacts={
            SYNC_REPORT: sync_report,
            "preview_image": f"{BOP_DIR}/realsense_123/test/000001/{RGB_DIR}/000000.png",
            "external_note": tmp_path / "outside.txt",
        },
        run_root=run_root,
    )
    write_run_manifest(manifest, run_root)
    return run_root


def test_run_config_artifact_summary_reports_invalid_preflight(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-invalid-preflight"
    config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, config)
    (run_root / RUN_PREFLIGHT_REPORT).write_text("[]\n")

    records = collect_run_artifacts(run_root)
    run_config = next(
        record
        for record in records
        if record.key == RUN_CONFIG and record.source == "known"
    )

    assert run_config.summary["preflight_exists"] is True
    assert run_config.summary["preflight_ready_for_queue"] is False
    assert run_config.summary["preflight_queue_blocker"] == "invalid_preflight"
    assert RUN_PREFLIGHT_REPORT in run_config.summary["preflight_error"]
    assert "preflight=invalid_preflight" in run_config.to_dict()["display_label"]


def test_run_config_artifact_summary_labels_invalid_config(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-invalid-config"
    run_root.mkdir()
    (run_root / RUN_CONFIG).write_text(
        json.dumps(
                {
                    "schema_version": "run_config.v1",
                    "run_root": run_root.as_posix(),
                    "object_folder": "object_models",
                    "robot_profile": {"mode": "fake"},
                    "capture": {"resolution": "720p", "fps": 6, "sensors": []},
                    "pipeline": {"sequence_id": "sync_aruco", "plan_only": True},
                }
        )
        + "\n"
    )

    records = collect_run_artifacts(run_root)
    run_config = next(
        record
        for record in records
        if record.key == RUN_CONFIG and record.source == "known"
    )

    assert run_config.summary["run_config_ready_for_pipeline"] is False
    assert run_config.summary["run_config_blocker"] == "invalid_run_config"
    assert "capture.sensors" in run_config.summary["run_config_error"]
    assert run_config.summary["preflight_ready_for_queue"] is False
    assert run_config.summary["preflight_queue_blocker"] == "invalid_run_config"
    assert "run_config=invalid_run_config" in (
        run_config.to_dict()["display_label"]
    )


def test_collect_run_artifacts_from_manifest_and_known_files(tmp_path: Path) -> None:
    run_root = create_artifact_fixture(tmp_path)

    records = collect_run_artifacts(run_root)
    by_key_source = {(record.key, record.source): record for record in records}

    assert by_key_source[(DATASET_MANIFEST, "known")].exists is True
    assert by_key_source[(PIPELINE_SEQUENCE_PLAN, "known")].preview_type == "json"
    assert by_key_source[(PIPELINE_SEQUENCE_PLAN, "known")].summary["type"] == (
        "pipeline_sequence_plan"
    )
    sequence_summary = by_key_source[(PIPELINE_SEQUENCE_PLAN, "known")].summary
    assert sequence_summary["sequence_id"] == "sync_to_bop_calibrated_dry_run"
    assert sequence_summary["step_count"] == 3
    assert sequence_summary["pipeline_sequence_plan_ready_for_queue"] is True
    assert sequence_summary["pipeline_sequence_plan_blocker"] is None
    assert sequence_summary["steps"] == [
        "sync_run",
        "blenderproc_prepare",
        "bop_export",
    ]
    assert sequence_summary["resources"] == ["cpu", "disk_io", "render"]
    assert sequence_summary["calibration_profile_steps"] == [
        "blenderproc_prepare",
        "bop_export",
    ]
    assert sequence_summary["calibration_profile_paths"] == [
        "profiles/lab_calibration.json"
    ]
    sequence_label = by_key_source[
        (PIPELINE_SEQUENCE_PLAN, "known")
    ].to_dict()["display_label"]
    assert "sequence=sync_to_bop_calibrated_dry_run" in sequence_label
    assert "steps=3" in sequence_label
    assert "resources=cpu,disk_io,render" in sequence_label
    assert "calibration=profiles/lab_calibration.json" in sequence_label
    assert "sequence_plan=ready" in sequence_label
    foundationpose_plan = by_key_source[(FOUNDATIONPOSE_PLAN, "known")]
    assert foundationpose_plan.summary["type"] == "estimator_plan"
    assert foundationpose_plan.summary["estimator_id"] == "foundationpose"
    assert foundationpose_plan.summary["job_count"] == 1
    assert foundationpose_plan.summary["estimator_plan_ready_for_jobs"] is True
    assert foundationpose_plan.summary["estimator_plan_blocker"] is None
    assert foundationpose_plan.summary["sensor_names"] == ["realsense_123"]
    assert foundationpose_plan.summary["object_names"] == ["cube"]
    assert foundationpose_plan.summary["command_uses_uv"] is True
    assert foundationpose_plan.summary["foundationpose_folder"] == "/opt/FoundationPose"
    assert foundationpose_plan.summary["no_tracking"] is False
    assert "estimator_plan=ready" in foundationpose_plan.to_dict()["display_label"]

    megapose_plan = by_key_source[(MEGAPOSE_PLAN, "known")]
    assert megapose_plan.summary["type"] == "estimator_plan"
    assert megapose_plan.summary["estimator_id"] == "megapose"
    assert megapose_plan.summary["dry_run"] is True
    assert megapose_plan.summary["object_id"] == 0
    assert megapose_plan.summary["job_count"] == 1
    assert megapose_plan.summary["estimator_plan_ready_for_jobs"] is True
    assert megapose_plan.summary["estimator_plan_blocker"] is None
    assert megapose_plan.summary["sensor_names"] == ["realsense_123"]
    assert megapose_plan.summary["object_names"] == ["cube"]
    assert megapose_plan.summary["command_uses_uv"] is True
    assert megapose_plan.summary["result_id"] == "rgbd"
    assert megapose_plan.summary["wrapper_exists"] is False
    assert megapose_plan.summary["option_keys"] == ["model", "roi_scale"]
    assert "estimator_plan=ready" in megapose_plan.to_dict()["display_label"]

    sam6d_plan = by_key_source[(SAM6D_PLAN, "known")]
    assert sam6d_plan.summary["type"] == "estimator_plan"
    assert sam6d_plan.summary["estimator_id"] == "sam6d"
    assert sam6d_plan.summary["dry_run"] is True
    assert sam6d_plan.summary["object_id"] == 0
    assert sam6d_plan.summary["job_count"] == 1
    assert sam6d_plan.summary["estimator_plan_ready_for_jobs"] is True
    assert sam6d_plan.summary["estimator_plan_blocker"] is None
    assert sam6d_plan.summary["sensor_names"] == ["realsense_123"]
    assert sam6d_plan.summary["object_names"] == ["cube"]
    assert sam6d_plan.summary["command_uses_uv"] is True
    assert sam6d_plan.summary["result_id"] == "sam-hq"
    assert sam6d_plan.summary["wrapper_exists"] is False
    assert sam6d_plan.summary["option_keys"] == ["segmentor_model"]
    assert "estimator_plan=ready" in sam6d_plan.to_dict()["display_label"]

    assert by_key_source[(RUN_CONFIG, "known")].summary == {
        "type": "run_config",
        "keys": [
            "calibration_profiles",
            "capture",
            "object_folder",
            "pipeline",
            "robot_profile",
            "run_name",
            "run_root",
            "schema_version",
        ],
        "key_count": 8,
        "schema_version": "run_config.v1",
        "run_name": "run",
        "object_folder": "object_models",
        "run_config_ready_for_pipeline": True,
        "run_config_blocker": None,
        "calibration_profiles": "profiles/lab_calibration.json",
        "has_calibration_profiles": True,
        "robot_mode": "fake",
        "sensor_count": 1,
        "sequence_id": "sync_to_bop_calibrated_dry_run",
        "plan_only": True,
        "preflight_exists": True,
        "preflight_status": "warning",
        "preflight_matches_config": False,
        "preflight_ready_for_queue": False,
        "preflight_queue_blocker": "stale_preflight",
    }
    run_config_label = by_key_source[(RUN_CONFIG, "known")].to_dict()[
        "display_label"
    ]
    assert "sequence=sync_to_bop_calibrated_dry_run" in run_config_label
    assert "robot=fake" in run_config_label
    assert "objects=object_models" in run_config_label
    assert "calibration=profiles/lab_calibration.json" in run_config_label
    assert "run_config=ready" in run_config_label
    assert "preflight=stale_preflight" in run_config_label
    assert by_key_source[(RUN_PREFLIGHT_REPORT, "known")].summary == {
        "type": "run_preflight_report",
        "keys": [
            "checks",
            "config",
            "overall_status",
            "robot_status",
            "run_root",
            "runtime_status",
            "schema_version",
            "sensor_status",
            "sequence_plan",
        ],
        "key_count": 9,
        "schema_version": "run_preflight.v1",
        "overall_status": "warning",
        "check_count": 3,
        "check_status_counts": {"ok": 2, "warning": 1},
        "sequence_id": "sync_to_bop_calibrated_dry_run",
        "step_count": 2,
        "robot_mode": "fake",
        "selected_robot_mode": "fake",
        "sensor_status_included": True,
        "runtime_status_included": True,
        "total_connected_sensors": 4,
        "all_expected_connected": False,
        "available_runtime_count": 3,
        "runtime_count": 5,
        "preflight_exists": True,
        "preflight_status": "warning",
        "preflight_matches_config": False,
        "preflight_ready_for_queue": False,
        "preflight_queue_blocker": "stale_preflight",
    }
    run_preflight_label = by_key_source[(RUN_PREFLIGHT_REPORT, "known")].to_dict()[
        "display_label"
    ]
    assert "status=warning" in run_preflight_label
    assert "sequence=sync_to_bop_calibrated_dry_run" in run_preflight_label
    assert "steps=2" in run_preflight_label
    assert "checks=3" in run_preflight_label
    assert "robot=fake" in run_preflight_label
    assert "check_status=ok:2,warning:1" in run_preflight_label
    assert "preflight=stale_preflight" in run_preflight_label
    assert by_key_source[(HARDWARE_STATUS_REPORT, "known")].summary == {
        "type": "hardware_status_report",
        "keys": [
            "checks",
            "overall_status",
            "robot_status",
            "runtime_status",
            "schema_version",
            "sensor_status",
        ],
        "key_count": 6,
        "schema_version": "hardware_status_report.v1",
        "overall_status": "warning",
        "hardware_status_ready_for_capture": True,
        "hardware_status_blocker": None,
        "check_count": 2,
        "robot_mode": "fake",
        "total_connected_sensors": 4,
        "all_expected_connected": False,
        "available_runtime_count": 2,
        "runtime_count": 4,
    }
    assert "hardware_status=ready" in by_key_source[
        (HARDWARE_STATUS_REPORT, "known")
    ].to_dict()["display_label"]
    assert by_key_source[(CAPTURE_PLAN, "known")].summary == {
        "type": "capture_plan",
        "keys": [
            "commands",
            "dry_run",
            "robot_profile",
            "schema_version",
            "sensors",
        ],
        "key_count": 5,
        "schema_version": "capture_plan.v1",
        "dry_run": True,
        "command_count": 3,
        "capture_plan_ready_for_preflight": True,
        "capture_plan_blocker": None,
        "sensor_count": 1,
        "robot_mode": "fake",
        "roles": ["robot_controller", "sensor_capture", "robot_pose_receiver"],
    }
    assert "capture_plan=ready" in by_key_source[
        (CAPTURE_PLAN, "known")
    ].to_dict()["display_label"]
    assert by_key_source[(CAPTURE_REHEARSAL_REPORT, "known")].summary == {
        "type": "capture_rehearsal_report",
        "keys": [
            "mode",
            "raw_pose_count",
            "schema_version",
            "status",
        ],
        "key_count": 4,
        "schema_version": "capture_rehearsal_report.v1",
        "status": "succeeded",
        "mode": "pose_only_fake",
        "raw_pose_count": 6,
        "capture_rehearsal_ready_for_sync": True,
        "capture_rehearsal_blocker": None,
    }
    assert "capture_rehearsal=ready" in by_key_source[
        (CAPTURE_REHEARSAL_REPORT, "known")
    ].to_dict()["display_label"]
    assert by_key_source[(CAPTURE_PLAN_PREFLIGHT_REPORT, "known")].summary == {
        "type": "capture_plan_preflight_report",
        "keys": [
            "checks",
            "overall_status",
            "schema_version",
        ],
        "key_count": 3,
        "schema_version": "capture_plan_preflight.v1",
        "overall_status": "ok",
        "capture_plan_preflight_ready": True,
        "capture_plan_preflight_blocker": None,
        "check_count": 1,
    }
    capture_preflight_label = by_key_source[
        (CAPTURE_PLAN_PREFLIGHT_REPORT, "known")
    ].to_dict()["display_label"]
    assert "capture_preflight=ready" in capture_preflight_label
    assert by_key_source[(CAPTURE_EXECUTION_PLAN, "known")].summary == {
        "type": "capture_execution_plan",
        "keys": [
            "mode",
            "ready_to_execute",
            "schema_version",
            "selected_commands",
            "selected_roles",
            "skipped_commands",
            "status",
        ],
        "key_count": 7,
        "schema_version": "capture_execution_plan.v1",
        "status": "ok",
        "mode": "pose_only_fake",
        "ready_to_execute": True,
        "capture_execution_plan_ready": True,
        "capture_execution_plan_blocker": None,
        "selected_count": 2,
        "skipped_count": 1,
        "selected_roles": ["robot_controller", "robot_pose_receiver"],
    }
    capture_execution_plan_label = by_key_source[
        (CAPTURE_EXECUTION_PLAN, "known")
    ].to_dict()["display_label"]
    assert "capture_execution_plan=ready" in capture_execution_plan_label
    assert by_key_source[(CAPTURE_EXECUTION_REPORT, "known")].summary == {
        "type": "capture_execution_report",
        "keys": [
            "mode",
            "processes",
            "raw_pose_count",
            "schema_version",
            "status",
        ],
        "key_count": 5,
        "schema_version": "capture_execution_report.v1",
        "status": "succeeded",
        "ready_for_downstream": True,
        "capture_execution_report_blocker": None,
        "mode": "pose_only_fake",
        "raw_pose_count": 6,
        "process_count": 2,
        "process_status_counts": {"succeeded": 2},
        "termination_reason_counts": {
            "exited_after_receiver": 1,
            "receiver_completed": 1,
        },
        "processes_with_timing": 2,
        "max_process_elapsed_s": 1.5,
    }
    capture_report_label = by_key_source[
        (CAPTURE_EXECUTION_REPORT, "known")
    ].to_dict()["display_label"]
    assert "capture=ready" in capture_report_label
    assert by_key_source[(CAPTURE_EXECUTION_STATUS, "known")].summary == {
        "type": "capture_execution_status",
        "keys": [
            "active_process_count",
            "mode",
            "processes",
            "raw_pose_count",
            "schema_version",
            "selected_roles",
            "status",
        ],
        "key_count": 7,
        "schema_version": "capture_execution_status.v1",
        "status": "running",
        "mode": "pose_only_fake",
        "active_process_count": 1,
        "process_count": 2,
        "raw_pose_count": 2,
        "selected_roles": ["robot_controller", "robot_pose_receiver"],
        "active_roles": ["robot_controller"],
        "process_status_counts": {"running": 1, "planned": 1},
    }
    assert by_key_source[(CAPTURE_EXECUTION_LOGS_DIR, "known")].kind == "directory"
    assert by_key_source[(CALIBRATION_PREFLIGHT_REPORT, "known")].summary == {
        "type": "calibration_preflight_report",
        "keys": [
            "checks",
            "matched_sensor_count",
            "matched_sensors",
            "max_mean_reprojection_error_px",
            "min_observations",
            "overall_status",
            "profile_count",
            "profile_path",
            "require_valid",
            "schema_version",
            "sensor_count",
        ],
        "key_count": 11,
        "schema_version": "calibration_preflight.v1",
        "overall_status": "warning",
        "calibration_preflight_ready_for_calibrated_stages": True,
        "calibration_preflight_blocker": None,
        "profile_path": "calibration_profiles.json",
        "profile_count": 1,
        "sensor_count": 2,
        "matched_sensor_count": 1,
        "check_count": 2,
        "check_status_counts": {"ok": 1, "warning": 1},
        "matched_profile_ids": ["realsense_123_profile"],
        "require_valid": True,
        "min_observations": 8,
        "max_mean_reprojection_error_px": 1.5,
    }
    calibration_preflight_label = by_key_source[
        (CALIBRATION_PREFLIGHT_REPORT, "known")
    ].to_dict()["display_label"]
    assert "status=warning" in calibration_preflight_label
    assert "sensors=2" in calibration_preflight_label
    assert "checks=2" in calibration_preflight_label
    assert "calibration=calibration_profiles.json" in calibration_preflight_label
    assert "matched=realsense_123_profile" in calibration_preflight_label
    assert "check_status=ok:1,warning:1" in calibration_preflight_label
    assert "calibration_preflight=ready" in calibration_preflight_label
    assert by_key_source[(CALIBRATION_OBSERVATIONS, "known")].summary == {
        "type": "calibration_observations",
        "keys": [
            "checks",
            "frame_count",
            "motion_count",
            "observation_count",
            "overall_status",
            "rejected_count",
            "schema_version",
            "sensor_count",
            "target",
        ],
        "key_count": 9,
        "schema_version": "calibration_observations.v1",
        "overall_status": "warning",
        "calibration_observations_ready_for_solver": True,
        "calibration_observations_blocker": None,
        "target_type": "aruco_grid",
        "dictionary": "DICT_5X5_50",
        "sensor_count": 1,
        "frame_count": 3,
        "observation_count": 2,
        "rejected_count": 1,
        "motion_count": 1,
        "check_count": 1,
    }
    calibration_observations_label = by_key_source[
        (CALIBRATION_OBSERVATIONS, "known")
    ].to_dict()["display_label"]
    assert "calibration_observations=ready" in calibration_observations_label
    assert by_key_source[(CALIBRATION_CANDIDATES, "known")].summary == {
        "type": "calibration_candidates",
        "keys": [
            "candidate_count",
            "checks",
            "inlier_count",
            "outlier_count",
            "overall_status",
            "profile_count",
            "schema_version",
            "sensor_count",
        ],
        "key_count": 8,
        "schema_version": "calibration_candidates.v1",
        "overall_status": "warning",
        "calibration_candidates_ready_for_validation": True,
        "calibration_candidates_blocker": None,
        "sensor_count": 1,
        "profile_count": 1,
        "candidate_count": 2,
        "inlier_count": 1,
        "outlier_count": 1,
        "check_count": 1,
    }
    calibration_candidates_label = by_key_source[
        (CALIBRATION_CANDIDATES, "known")
    ].to_dict()["display_label"]
    assert "calibration_candidates=ready" in calibration_candidates_label
    assert by_key_source[(CALIBRATION_PROFILES_FROM_OBSERVATIONS, "known")].summary == {
        "type": "calibration_profiles_from_observations",
        "keys": [
            "profiles",
            "schema_version",
        ],
        "key_count": 2,
        "schema_version": "calibration.v1",
        "profile_count": 1,
        "calibration_profile_collection_ready_for_validation": True,
        "calibration_profile_collection_blocker": None,
        "statuses": ["needs_validation"],
    }
    calibration_profiles_from_observations_label = by_key_source[
        (CALIBRATION_PROFILES_FROM_OBSERVATIONS, "known")
    ].to_dict()["display_label"]
    assert "calibration_profiles=ready" in (
        calibration_profiles_from_observations_label
    )
    assert by_key_source[(CALIBRATION_SOLVER_REPORT, "known")].summary == {
        "type": "calibration_solver_report",
        "keys": [
            "checks",
            "hand_eye_method",
            "inlier_count",
            "method_comparisons",
            "observation_count",
            "outlier_count",
            "overall_status",
            "profile_count",
            "schema_version",
            "sensor_count",
        ],
        "key_count": 10,
        "schema_version": "calibration_solver.v1",
        "overall_status": "warning",
        "calibration_solver_ready_for_candidates": True,
        "calibration_solver_blocker": None,
        "sensor_count": 1,
        "profile_count": 1,
        "observation_count": 2,
        "inlier_count": 1,
        "outlier_count": 1,
        "hand_eye_method": "tsai",
        "check_count": 1,
        "method_comparison_count": 2,
        "method_comparison_statuses": ["ok", "warning"],
    }
    calibration_solver_label = by_key_source[
        (CALIBRATION_SOLVER_REPORT, "known")
    ].to_dict()["display_label"]
    assert "calibration_solver=ready" in calibration_solver_label
    assert by_key_source[(CALIBRATION_PROFILES_SOLVED, "known")].summary == {
        "type": "calibration_profiles_solved",
        "keys": [
            "profiles",
            "schema_version",
        ],
        "key_count": 2,
        "schema_version": "calibration.v1",
        "profile_count": 1,
        "calibration_profile_collection_ready_for_validation": True,
        "calibration_profile_collection_blocker": None,
        "statuses": ["needs_validation"],
        "methods": ["static_target_reference_transform_average"],
    }
    calibration_profiles_solved_label = by_key_source[
        (CALIBRATION_PROFILES_SOLVED, "known")
    ].to_dict()["display_label"]
    assert "calibration_profiles=ready" in calibration_profiles_solved_label
    assert by_key_source[(CALIBRATION_VALIDATION_REPORT, "known")].summary == {
        "type": "calibration_validation_report",
        "keys": [
            "candidate_count",
            "checks",
            "inlier_count",
            "outlier_count",
            "overall_status",
            "profile_count",
            "promotable_profile_count",
            "promotion",
            "schema_version",
        ],
        "key_count": 9,
        "schema_version": "calibration_validation.v1",
        "overall_status": "ok",
        "calibration_validation_ready_for_profiles": True,
        "calibration_validation_blocker": None,
        "profile_count": 1,
        "promotable_profile_count": 1,
        "candidate_count": 2,
        "inlier_count": 1,
        "outlier_count": 1,
        "promoted": False,
        "check_count": 1,
    }
    validation_label = by_key_source[
        (CALIBRATION_VALIDATION_REPORT, "known")
    ].to_dict()["display_label"]
    assert "calibration_validation=ready" in validation_label
    assert by_key_source[(SYNC_QUALITY_REPORT, "known")].summary == {
        "type": "sync_quality_report",
        "keys": [
            "checks",
            "dropped_frames",
            "matched_frames",
            "overall_match_ratio",
            "overall_status",
            "schema_version",
            "sensor_count",
            "sensors",
            "total_frames",
        ],
        "key_count": 9,
        "schema_version": "sync_quality_report.v1",
        "overall_status": "warning",
        "sync_quality_ready_for_downstream": True,
        "sync_quality_report_blocker": None,
        "sensor_count": 1,
        "total_frames": 3,
        "matched_frames": 2,
        "dropped_frames": 1,
        "overall_match_ratio": 2 / 3,
        "check_count": 1,
        "sensor_names": ["realsense_123"],
    }
    sync_quality_label = by_key_source[
        (SYNC_QUALITY_REPORT, "known")
    ].to_dict()["display_label"]
    assert "sync_quality=ready" in sync_quality_label
    assert by_key_source[(BOP_EXPORT_MANIFEST, "known")].relative_path == (
        f"{BOP_DIR}/{BOP_EXPORT_MANIFEST}"
    )
    assert by_key_source[(BOP_EXPORT_MANIFEST, "known")].summary == {
        "type": "bop_export_manifest",
            "keys": [
                "coco_annotations_path",
                "exports",
                "multiview_targets_path",
                "object_models",
                "schema_version",
                "targets_path",
            ],
            "key_count": 6,
        "schema_version": "bop_export_manifest.v1",
        "export_count": 1,
        "bop_export_ready_for_results": True,
        "bop_export_blocker": None,
        "sensors": ["realsense_123"],
        "object_model_count": 1,
        "has_targets": True,
        "has_multiview_targets": True,
        "has_coco_annotations": True,
    }
    bop_export_label = by_key_source[(BOP_EXPORT_MANIFEST, "known")].to_dict()[
        "display_label"
    ]
    assert "bop_export=ready" in bop_export_label
    models_info_record = by_key_source[("models_info.json", "known")]
    assert models_info_record.summary == {
        "type": "bop_models_info",
        "keys": ["1"],
        "key_count": 1,
        "model_count": 1,
        "bop_models_info_ready": True,
        "bop_models_info_blocker": None,
        "object_ids": ["1"],
    }
    assert "bop_models=ready" in models_info_record.to_dict()["display_label"]
    targets_record = by_key_source[(BOP_TARGETS_BOP19, "known")]
    assert targets_record.summary == {
        "type": "bop_targets_bop19",
        "item_count": 1,
        "target_count": 1,
        "bop_targets_ready_for_evaluation": True,
        "bop_targets_blocker": None,
        "scene_count": 1,
        "object_count": 1,
    }
    assert "bop_targets=ready" in targets_record.to_dict()["display_label"]
    multiview_record = by_key_source[(BOP_MULTIVIEW_TARGETS, "known")]
    assert multiview_record.summary == {
        "type": "bop_multiview_targets",
        "keys": [
            "object_count",
            "scene_count",
            "schema_version",
            "split",
            "targets",
        ],
        "key_count": 5,
        "schema_version": "posetestbot_bop_multiview_targets.v1",
        "bop_multiview_targets_ready": True,
        "bop_multiview_targets_blocker": None,
        "split": "test",
        "scene_count": 1,
        "object_count": 1,
        "target_count": 1,
    }
    assert "bop_multiview=ready" in multiview_record.to_dict()["display_label"]
    assert by_key_source[(BOP_RESULT_EXPORT_MANIFEST, "known")].summary == {
        "type": "bop_result_export_manifest",
        "keys": [
            "output_folder",
            "results",
            "schema_version",
        ],
        "key_count": 3,
        "schema_version": "bop_result_export_manifest.v1",
        "result_count": 1,
        "usable_result_count": 1,
        "bop_result_export_ready_for_evaluation": True,
        "bop_result_export_blocker": None,
        "total_rows": 0,
    }
    bop_result_export_label = by_key_source[
        (BOP_RESULT_EXPORT_MANIFEST, "known")
    ].to_dict()["display_label"]
    assert "bop_results=ready" in bop_result_export_label
    evaluation_plan = by_key_source[(BOP_EVALUATION_PLAN, "known")]
    assert evaluation_plan.summary["type"] == "bop_evaluation_plan"
    assert evaluation_plan.summary["bop_evaluation_plan_ready_for_execution"] is True
    assert evaluation_plan.summary["bop_evaluation_plan_blocker"] is None
    assert evaluation_plan.summary["result_filename"] == (
        "foundationpose_bop-test.csv"
    )
    assert evaluation_plan.summary["command_count"] == 3
    assert evaluation_plan.summary["bop_path"] == run_root.as_posix()
    assert "bop_eval_plan=ready" in evaluation_plan.to_dict()["display_label"]
    evaluation_report = by_key_source[(BOP_EVALUATION_REPORT, "known")]
    assert evaluation_report.summary["type"] == "bop_evaluation_report"
    assert evaluation_report.summary["status"] == "succeeded"
    assert evaluation_report.summary["ready_for_metrics"] is True
    assert evaluation_report.summary["bop_evaluation_report_blocker"] is None
    assert evaluation_report.summary["result_filename"] == (
        "foundationpose_bop-test.csv"
    )
    assert evaluation_report.summary["check_count"] == 8
    assert evaluation_report.summary["failed_check_count"] == 1
    assert evaluation_report.summary["critical_failed_check_count"] == 0
    assert evaluation_report.summary["critical_missing_check_count"] == 0
    assert evaluation_report.summary["output_artifact_count"] == 1
    assert evaluation_report.summary["score_file_count"] == 1
    assert evaluation_report.summary["score_metric_count"] == 1
    assert evaluation_report.summary["bop19_average_recall"] == 0.75
    assert by_key_source[
        ("output:scores_bop19.json", "bop_evaluation_report.output")
    ].exists
    assert "bop_eval=ready" in evaluation_report.to_dict()["display_label"]
    assert by_key_source[(DATASET_MANIFEST, "known")].summary["type"] == (
        "dataset_manifest"
    )
    assert by_key_source[(SYNC_REPORT, "stage:sync:realsense_123")].relative_path == (
        "processed/synchronized/realsense_123/sync_report.json"
    )
    assert by_key_source[("preview_image", "stage:sync:realsense_123")].summary == {
        "type": "image",
        "width": 4,
        "height": 3,
        "channels": 3,
        "dtype": "uint8",
    }
    assert by_key_source[("realsense_123:scene_folder", "bop_export")].summary == {
        "type": "bop_scene",
        "image_count": 1,
        "rgb_count": 1,
        "depth_count": 1,
        "annotation_count": 1,
        "has_scene_gt_info": True,
        "has_mask": True,
        "has_mask_visib": False,
    }
    assert by_key_source[("realsense_123:scene_camera", "bop_export.scene")].exists
    coco_record = by_key_source[(BOP_COCO_ANNOTATIONS, "known")]
    assert coco_record.summary == {
        "type": "bop_coco_annotations",
        "keys": [
            "annotations",
            "categories",
            "images",
            "schema_version",
        ],
        "key_count": 4,
        "schema_version": "posetestbot_coco_annotations.v1",
        "bop_coco_annotations_ready": True,
        "bop_coco_annotations_blocker": None,
        "image_count": 1,
        "annotation_count": 1,
        "category_count": 1,
    }
    assert "bop_coco=ready" in coco_record.to_dict()["display_label"]
    assert by_key_source[("coco_annotations_path", "bop_export")].relative_path == (
        f"{BOP_DIR}/{BOP_COCO_ANNOTATIONS}"
    )
    csv_record = by_key_source[
        ("foundationpose_bop-test.csv", "bop_result_export.result")
    ]
    assert csv_record.preview_type == "text"
    assert csv_record.summary["row_count"] == 1
    assert csv_record.summary["columns"] == [
        "scene_id",
        "im_id",
        "obj_id",
        "score",
        "R",
        "t",
        "time",
    ]
    assert by_key_source[("external_note", "stage:sync:realsense_123")].kind == (
        "outside_run_root"
    )
    accuracy_record = by_key_source[(ACCURACY_HRC_HUB, "metrics.legacy_pose")]
    assert accuracy_record.relative_path == (
        f"processed/synchronized/realsense_123/{ACCURACY_HRC_HUB}"
    )
    assert accuracy_record.summary["type"] == "pose_accuracy_metrics"
    assert accuracy_record.summary["method_count"] == 1
    assert accuracy_record.summary["best_by_AP_p"] == {
        "method": "foundationpose_est5_track2_obj0",
        "AP_p": 1.25,
    }
    assert accuracy_record.summary["methods"][0]["sample_count"] == 3
    assert by_key_source[
        (ACCURACY_ARUCO_HRC_HUB, "metrics.legacy_pose")
    ].summary["methods"][0]["name"] == "ArUco_accuracy"
    combined_record = by_key_source[(ALL_RESULTS_JSON, "metrics.combined")]
    assert combined_record.summary["type"] == "combined_pose_accuracy_metrics"
    assert combined_record.summary["experiment_count"] == 1
    assert combined_record.summary["result_group_count"] == 2
    assert combined_record.summary["method_count"] == 2


def test_capture_execution_report_summary_labels_failed_readiness(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / CAPTURE_EXECUTION_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "capture_execution_report.v1",
                "status": "failed",
                "mode": "pose_only_fake",
                "processes": [{"status": "failed"}],
            }
        )
    )

    artifacts = collect_run_artifacts(run_root)
    report = next(
        artifact
        for artifact in artifacts
        if artifact.key == CAPTURE_EXECUTION_REPORT
    )

    assert report.summary is not None
    assert report.summary["ready_for_downstream"] is False
    assert report.summary["capture_execution_report_blocker"] == (
        "failed_capture_execution_report"
    )
    assert "capture=failed_capture_execution_report" in (
        report.to_dict()["display_label"]
    )


def test_rewrite_gate_report_summary_labels_blocked_gate(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "rewrite-gate-run"
    run_root.mkdir()
    (run_root / REWRITE_GATE_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "rewrite_gate_report.v1",
                "gate_id": "rewrite_full_capture.v1",
                "overall_status": "blocked",
                "summary": {
                    "ready_count": 1,
                    "blocked_count": 2,
                    "check_count": 3,
                },
                "checks": [],
                "next_blockers": [
                    {
                        "name": "capture_execution",
                        "artifact": "capture_execution_report.json",
                        "message": "Capture is not full mode.",
                        "details": {
                            "error_checks": [
                                {
                                    "name": "capture_plan_preflight",
                                    "status": "error",
                                    "message": "Capture-plan preflight failed.",
                                }
                            ],
                            "sensor_diagnostics": [
                                {
                                    "message": "RealSense discovery failed.",
                                    "hints": ["Check USB/udev access."],
                                }
                            ],
                        },
                    },
                    {
                        "name": "sensor_frames:realsense_1",
                        "artifact": "realsense_1",
                        "message": "Frames are missing.",
                    },
                ],
            }
        )
        + "\n"
    )

    artifacts = collect_run_artifacts(run_root)
    report = next(
        artifact for artifact in artifacts if artifact.key == REWRITE_GATE_REPORT
    )

    assert report.summary == {
        "type": "rewrite_gate_report",
        "keys": [
            "checks",
            "gate_id",
            "next_blockers",
            "overall_status",
            "schema_version",
            "summary",
        ],
        "key_count": 6,
        "schema_version": "rewrite_gate_report.v1",
        "gate_id": "rewrite_full_capture.v1",
        "overall_status": "blocked",
        "rewrite_gate_ready": False,
        "rewrite_gate_blocker": "blocked_rewrite_gate",
        "ready_count": 1,
        "blocked_count": 2,
        "check_count": 3,
        "next_blockers": [
            "capture_execution",
            "sensor_frames:realsense_1",
        ],
        "next_blocker_messages": [
            "Capture is not full mode.",
            "Frames are missing.",
        ],
        "next_blocker_diagnostics": ["RealSense discovery failed."],
        "next_blocker_hints": ["Check USB/udev access."],
        "next_blocker_checks": [
            "capture_plan_preflight: Capture-plan preflight failed.",
        ],
    }
    assert "rewrite_gate=blocked_rewrite_gate" in report.to_dict()[
        "display_label"
    ]
    assert "next_blocker=capture_execution" in report.to_dict()["display_label"]
    assert "next_diag=RealSense discovery failed." in report.to_dict()[
        "display_label"
    ]


def test_rewrite_status_report_summary_labels_blocked_status(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "rewrite-status-run"
    run_root.mkdir()
    (run_root / REWRITE_STATUS_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "rewrite_status_report.v1",
                "overall_status": "blocked",
                "summary": {
                    "gate_count": 4,
                    "ready_gate_count": 1,
                    "blocked_gate_count": 3,
                    "check_count": 24,
                    "ready_check_count": 9,
                    "blocked_check_count": 15,
                },
                "gates": [
                    {
                        "gate_id": "rewrite_fake_end_to_end.v1",
                        "overall_status": "ready",
                    },
                    {
                        "gate_id": "rewrite_full_capture.v1",
                        "overall_status": "blocked",
                    },
                    {
                        "gate_id": "rewrite_foundationpose_runtime.v1",
                        "overall_status": "blocked",
                    },
                ],
                "next_gate": {
                    "gate_id": "rewrite_full_capture.v1",
                    "run_root": run_root.as_posix(),
                    "overall_status": "blocked",
                    "summary": {
                        "ready_count": 1,
                        "blocked_count": 2,
                        "check_count": 3,
                    },
                },
                "next_actions": [
                    {
                        "gate_id": "rewrite_full_capture.v1",
                        "label": "Create real lab run config",
                        "command": [
                            "uv",
                            "run",
                            "python",
                            "scripts/create_run_config.py",
                            run_root.as_posix(),
                            "--robot-mode",
                            "real",
                            "--sequence",
                            "real_full_capture_validation",
                            "--print-sequence-plan",
                        ],
                        "reason": "Create the real validation config.",
                        "blocks_on": ["run_config"],
                    }
                ],
                "next_blockers": [
                    {
                        "gate_id": "rewrite_full_capture.v1",
                        "name": "hardware_status",
                        "artifact": "hardware_status_report.json",
                        "message": "Hardware status is blocked.",
                        "details": {
                            "error_checks": [
                                {
                                    "name": "sensor:realsense_d435",
                                    "status": "error",
                                    "message": "RealSense discovery failed.",
                                }
                            ],
                            "sensor_diagnostics": [
                                {
                                    "message": "RealSense discovery failed.",
                                    "hints": ["Check USB/udev access."],
                                }
                            ],
                        },
                    }
                ],
            }
        )
        + "\n"
    )

    artifacts = collect_run_artifacts(run_root)
    report = next(
        artifact for artifact in artifacts if artifact.key == REWRITE_STATUS_REPORT
    )

    assert report.summary == {
        "type": "rewrite_status_report",
        "keys": [
            "gates",
            "next_actions",
            "next_blockers",
            "next_gate",
            "overall_status",
            "schema_version",
            "summary",
        ],
        "key_count": 7,
        "schema_version": "rewrite_status_report.v1",
        "overall_status": "blocked",
        "rewrite_status_ready": False,
        "rewrite_status_blocker": "blocked_rewrite_status",
        "gate_count": 4,
        "ready_gate_count": 1,
        "blocked_gate_count": 3,
        "ready_check_count": 9,
        "check_count": 24,
        "blocked_gate_ids": [
            "rewrite_full_capture.v1",
            "rewrite_foundationpose_runtime.v1",
        ],
        "next_blockers": ["hardware_status"],
        "next_blocker_messages": ["Hardware status is blocked."],
        "next_blocker_diagnostics": ["RealSense discovery failed."],
        "next_blocker_hints": ["Check USB/udev access."],
        "next_blocker_checks": ["sensor:realsense_d435: RealSense discovery failed."],
        "next_gate_id": "rewrite_full_capture.v1",
        "next_gate_run_root": run_root.as_posix(),
        "next_action_count": 1,
        "next_action_labels": ["Create real lab run config"],
        "next_action_commands": [
            [
                "uv",
                "run",
                "python",
                "scripts/create_run_config.py",
                run_root.as_posix(),
                "--robot-mode",
                "real",
                "--sequence",
                "real_full_capture_validation",
                "--print-sequence-plan",
            ]
        ],
        "next_action_blocks_on": [["run_config"]],
        "next_action_label": "Create real lab run config",
        "next_action_command": [
            "uv",
            "run",
            "python",
            "scripts/create_run_config.py",
            run_root.as_posix(),
            "--robot-mode",
            "real",
            "--sequence",
            "real_full_capture_validation",
            "--print-sequence-plan",
        ],
    }
    display_label = report.to_dict()["display_label"]
    assert "rewrite_status=blocked_rewrite_status" in display_label
    assert "next_gate=rewrite_full_capture.v1" in display_label
    assert "next_blocker=hardware_status" in display_label
    assert "next_diag=RealSense discovery failed." in display_label


def test_rewrite_status_report_summary_lists_multiple_next_actions(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "rewrite-status-multi-action-run"
    run_root.mkdir()
    hardware_command = [
        "uv",
        "run",
        "python",
        "scripts/run_hardware_status_stage.py",
        run_root.as_posix(),
    ]
    (run_root / REWRITE_STATUS_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "rewrite_status_report.v1",
                "overall_status": "blocked",
                "summary": {
                    "gate_count": 4,
                    "ready_gate_count": 1,
                    "blocked_gate_count": 3,
                    "check_count": 26,
                    "ready_check_count": 12,
                    "blocked_check_count": 14,
                },
                "gates": [],
                "next_gate": {"gate_id": "rewrite_full_capture.v1"},
                "next_actions": [
                    {
                        "gate_id": "rewrite_full_capture.v1",
                        "label": "Inspect sensor status",
                        "command": [
                            "uv",
                            "run",
                            "python",
                            "scripts/sensor_status.py",
                            "--json",
                        ],
                        "blocks_on": [
                            "sensor:realsense_d435",
                            "sensor:oak_d_pro",
                            "sensor:zed_2i",
                        ],
                    },
                    {
                        "gate_id": "rewrite_full_capture.v1",
                        "label": "Refresh hardware status after sensor fix",
                        "command": hardware_command,
                        "blocks_on": [
                            "sensor:realsense_d435",
                            "sensor:oak_d_pro",
                            "sensor:zed_2i",
                        ],
                    },
                ],
                "next_blockers": [],
            }
        )
        + "\n"
    )

    artifacts = collect_run_artifacts(run_root)
    report = next(
        artifact for artifact in artifacts if artifact.key == REWRITE_STATUS_REPORT
    )

    assert report.summary["next_action_count"] == 2
    assert report.summary["next_action_labels"] == [
        "Inspect sensor status",
        "Refresh hardware status after sensor fix",
    ]
    assert report.summary["next_action_commands"][1] == hardware_command
    assert report.summary["next_action_blocks_on"] == [
        ["sensor:realsense_d435", "sensor:oak_d_pro", "sensor:zed_2i"],
        ["sensor:realsense_d435", "sensor:oak_d_pro", "sensor:zed_2i"],
    ]
    assert report.summary["next_action_label"] == "Inspect sensor status"


def test_pipeline_sequence_plan_summary_labels_empty_readiness(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / PIPELINE_SEQUENCE_PLAN).write_text(
        '{"schema_version": "pipeline_sequence_plan.v1", "steps": []}\n'
    )

    artifacts = collect_run_artifacts(run_root)
    report = next(
        artifact for artifact in artifacts if artifact.key == PIPELINE_SEQUENCE_PLAN
    )

    assert report.summary is not None
    assert report.summary["pipeline_sequence_plan_ready_for_queue"] is False
    assert report.summary["pipeline_sequence_plan_blocker"] == (
        "empty_pipeline_sequence_plan"
    )
    assert "sequence_plan=empty_pipeline_sequence_plan" in (
        report.to_dict()["display_label"]
    )


def test_hardware_status_summary_labels_failed_readiness(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / HARDWARE_STATUS_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "hardware_status_report.v1",
                "overall_status": "error",
                "checks": [{"name": "sensor_status", "status": "error"}],
            }
        )
    )

    artifacts = collect_run_artifacts(run_root)
    report = next(
        artifact for artifact in artifacts if artifact.key == HARDWARE_STATUS_REPORT
    )

    assert report.summary is not None
    assert report.summary["hardware_status_ready_for_capture"] is False
    assert report.summary["hardware_status_blocker"] == (
        "failed_hardware_status_report"
    )
    assert "hardware_status=failed_hardware_status_report" in (
        report.to_dict()["display_label"]
    )


def test_capture_rehearsal_summary_labels_empty_pose_readiness(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / CAPTURE_REHEARSAL_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "capture_rehearsal_report.v1",
                "status": "succeeded",
                "mode": "pose_only_fake",
                "raw_pose_count": 0,
            }
        )
    )

    artifacts = collect_run_artifacts(run_root)
    report = next(
        artifact for artifact in artifacts if artifact.key == CAPTURE_REHEARSAL_REPORT
    )

    assert report.summary is not None
    assert report.summary["capture_rehearsal_ready_for_sync"] is False
    assert report.summary["capture_rehearsal_blocker"] == (
        "empty_capture_rehearsal_report"
    )
    assert "capture_rehearsal=empty_capture_rehearsal_report" in (
        report.to_dict()["display_label"]
    )


def test_capture_plan_summary_labels_empty_readiness(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / CAPTURE_PLAN).write_text(
        '{"schema_version": "capture_plan.v1", "commands": []}\n'
    )

    artifacts = collect_run_artifacts(run_root)
    report = next(artifact for artifact in artifacts if artifact.key == CAPTURE_PLAN)

    assert report.summary is not None
    assert report.summary["capture_plan_ready_for_preflight"] is False
    assert report.summary["capture_plan_blocker"] == "empty_capture_plan"
    assert "capture_plan=empty_capture_plan" in report.to_dict()["display_label"]


def test_capture_plan_preflight_summary_labels_failed_readiness(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / CAPTURE_PLAN_PREFLIGHT_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "capture_plan_preflight.v1",
                "overall_status": "error",
                "checks": [{"name": "command_shape", "status": "error"}],
            }
        )
    )

    artifacts = collect_run_artifacts(run_root)
    report = next(
        artifact
        for artifact in artifacts
        if artifact.key == CAPTURE_PLAN_PREFLIGHT_REPORT
    )

    assert report.summary is not None
    assert report.summary["capture_plan_preflight_ready"] is False
    assert report.summary["capture_plan_preflight_blocker"] == (
        "failed_capture_plan_preflight"
    )
    assert "capture_preflight=failed_capture_plan_preflight" in (
        report.to_dict()["display_label"]
    )


def test_capture_execution_plan_summary_labels_failed_readiness(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / CAPTURE_EXECUTION_PLAN).write_text(
        json.dumps(
            {
                "schema_version": "capture_execution_plan.v1",
                "status": "error",
                "mode": "full",
                "ready_to_execute": False,
                "gates": [
                    {
                        "name": "capture_plan_preflight",
                        "status": "error",
                        "message": "Capture-plan preflight status is error.",
                    }
                ],
            }
        )
    )

    artifacts = collect_run_artifacts(run_root)
    report = next(
        artifact
        for artifact in artifacts
        if artifact.key == CAPTURE_EXECUTION_PLAN
    )

    assert report.summary is not None
    assert report.summary["capture_execution_plan_ready"] is False
    assert report.summary["capture_execution_plan_blocker"] == (
        "failed_capture_execution_plan"
    )
    assert report.summary["blocked_check_messages"] == [
        "Capture-plan preflight status is error."
    ]
    assert "capture_execution_plan=failed_capture_execution_plan" in (
        report.to_dict()["display_label"]
    )


def test_estimator_plan_summary_labels_empty_plan_readiness(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / FOUNDATIONPOSE_PLAN).write_text(
        json.dumps(
            {
                "schema_version": "foundationpose_plan.v1",
                "dry_run": True,
                "command": [
                    "uv",
                    "run",
                    "python",
                    "scripts/foundationpose_wrapper_multi.py",
                ],
                "jobs": [],
            }
        )
    )

    artifacts = collect_run_artifacts(run_root)
    report = next(
        artifact for artifact in artifacts if artifact.key == FOUNDATIONPOSE_PLAN
    )

    assert report.summary is not None
    assert report.summary["type"] == "estimator_plan"
    assert report.summary["estimator_id"] == "foundationpose"
    assert report.summary["job_count"] == 0
    assert report.summary["estimator_plan_ready_for_jobs"] is False
    assert report.summary["estimator_plan_blocker"] == "empty_estimator_plan"
    assert "estimator_plan=empty_estimator_plan" in (
        report.to_dict()["display_label"]
    )


def test_sync_quality_report_summary_labels_failed_readiness(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / SYNC_QUALITY_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "sync_quality_report.v1",
                "overall_status": "error",
                "checks": [{"name": "match_ratio", "status": "error"}],
            }
        )
    )

    artifacts = collect_run_artifacts(run_root)
    report = next(
        artifact for artifact in artifacts if artifact.key == SYNC_QUALITY_REPORT
    )

    assert report.summary is not None
    assert report.summary["sync_quality_ready_for_downstream"] is False
    assert report.summary["sync_quality_report_blocker"] == (
        "failed_sync_quality_report"
    )
    assert "sync_quality=failed_sync_quality_report" in (
        report.to_dict()["display_label"]
    )


def test_calibration_preflight_summary_labels_failed_readiness(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / CALIBRATION_PREFLIGHT_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "calibration_preflight.v1",
                "overall_status": "error",
                "checks": [{"name": "profiles", "status": "error"}],
            }
        )
    )

    artifacts = collect_run_artifacts(run_root)
    report = next(
        artifact
        for artifact in artifacts
        if artifact.key == CALIBRATION_PREFLIGHT_REPORT
    )

    assert report.summary is not None
    assert report.summary["calibration_preflight_ready_for_calibrated_stages"] is False
    assert report.summary["calibration_preflight_blocker"] == (
        "failed_calibration_preflight"
    )
    assert "calibration_preflight=failed_calibration_preflight" in (
        report.to_dict()["display_label"]
    )


def test_aruco_coverage_summary_labels_failed_readiness(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / ARUCO_COVERAGE_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "aruco_coverage_report.v1",
                "overall_status": "error",
                "sensor_count": 0,
                "frame_count": 0,
                "valid_pose_count": 0,
                "checks": [{"name": "aruco_outputs_present", "status": "error"}],
            }
        )
    )

    artifacts = collect_run_artifacts(run_root)
    report = next(
        artifact
        for artifact in artifacts
        if artifact.key == ARUCO_COVERAGE_REPORT
    )

    assert report.summary is not None
    assert report.summary["aruco_coverage_ready_for_downstream"] is False
    assert report.summary["aruco_coverage_blocker"] == (
        "failed_aruco_coverage_report"
    )
    assert "aruco_coverage=failed_aruco_coverage_report" in (
        report.to_dict()["display_label"]
    )


def test_calibration_observations_summary_labels_failed_readiness(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / CALIBRATION_OBSERVATIONS).write_text(
        json.dumps(
            {
                "schema_version": "calibration_observations.v1",
                "overall_status": "error",
                "checks": [{"name": "observations", "status": "error"}],
            }
        )
    )

    artifacts = collect_run_artifacts(run_root)
    report = next(
        artifact
        for artifact in artifacts
        if artifact.key == CALIBRATION_OBSERVATIONS
    )

    assert report.summary is not None
    assert report.summary["calibration_observations_ready_for_solver"] is False
    assert report.summary["calibration_observations_blocker"] == (
        "failed_calibration_observations"
    )
    assert "calibration_observations=failed_calibration_observations" in (
        report.to_dict()["display_label"]
    )


def test_calibration_solver_summary_labels_failed_readiness(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / CALIBRATION_SOLVER_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "calibration_solver.v1",
                "overall_status": "error",
                "checks": [{"name": "solver", "status": "error"}],
            }
        )
    )

    artifacts = collect_run_artifacts(run_root)
    report = next(
        artifact for artifact in artifacts if artifact.key == CALIBRATION_SOLVER_REPORT
    )

    assert report.summary is not None
    assert report.summary["calibration_solver_ready_for_candidates"] is False
    assert report.summary["calibration_solver_blocker"] == (
        "failed_calibration_solver"
    )
    assert "calibration_solver=failed_calibration_solver" in (
        report.to_dict()["display_label"]
    )


def test_calibration_candidates_summary_labels_failed_readiness(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / CALIBRATION_CANDIDATES).write_text(
        json.dumps(
            {
                "schema_version": "calibration_candidates.v1",
                "overall_status": "error",
                "checks": [{"name": "candidates", "status": "error"}],
            }
        )
    )

    artifacts = collect_run_artifacts(run_root)
    report = next(
        artifact for artifact in artifacts if artifact.key == CALIBRATION_CANDIDATES
    )

    assert report.summary is not None
    assert report.summary["calibration_candidates_ready_for_validation"] is False
    assert report.summary["calibration_candidates_blocker"] == (
        "failed_calibration_candidates"
    )
    assert "calibration_candidates=failed_calibration_candidates" in (
        report.to_dict()["display_label"]
    )


def test_calibration_profile_collection_summary_labels_empty_readiness(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / CALIBRATION_PROFILES_SOLVED).write_text(
        '{"schema_version": "calibration.v1", "profiles": []}\n'
    )

    artifacts = collect_run_artifacts(run_root)
    report = next(
        artifact for artifact in artifacts if artifact.key == CALIBRATION_PROFILES_SOLVED
    )

    assert report.summary is not None
    assert report.summary["calibration_profile_collection_ready_for_validation"] is False
    assert report.summary["calibration_profile_collection_blocker"] == (
        "empty_calibration_profile_collection"
    )
    assert "calibration_profiles=empty_calibration_profile_collection" in (
        report.to_dict()["display_label"]
    )


def test_calibration_validation_summary_labels_failed_readiness(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / CALIBRATION_VALIDATION_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "calibration_validation.v1",
                "overall_status": "error",
                "checks": [{"name": "inliers", "status": "error"}],
            }
        )
    )

    artifacts = collect_run_artifacts(run_root)
    report = next(
        artifact
        for artifact in artifacts
        if artifact.key == CALIBRATION_VALIDATION_REPORT
    )

    assert report.summary is not None
    assert report.summary["calibration_validation_ready_for_profiles"] is False
    assert report.summary["calibration_validation_blocker"] == (
        "failed_calibration_validation"
    )
    assert "calibration_validation=failed_calibration_validation" in (
        report.to_dict()["display_label"]
    )


def test_bop_result_export_summary_labels_missing_result_csv(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    missing_result = run_root / RESULTS_DIR / BOP_DIR / "missing_bop-test.csv"
    (run_root / BOP_RESULT_EXPORT_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": "bop_result_export_manifest.v1",
                "results": [
                    {
                        "filename": missing_result.name,
                        "path": missing_result.as_posix(),
                    }
                ],
            }
        )
    )

    artifacts = collect_run_artifacts(run_root)
    report = next(
        artifact
        for artifact in artifacts
        if artifact.key == BOP_RESULT_EXPORT_MANIFEST
    )

    assert report.summary is not None
    assert report.summary["bop_result_export_ready_for_evaluation"] is False
    assert report.summary["bop_result_export_blocker"] == "missing_bop_result_csv"
    assert report.summary["usable_result_count"] == 0
    assert "bop_results=missing_bop_result_csv" in (
        report.to_dict()["display_label"]
    )


def test_bop_export_summary_labels_empty_manifest(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    manifest = run_root / BOP_DIR / BOP_EXPORT_MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "bop_export_manifest.v1",
                "exports": [],
            }
        )
    )

    artifacts = collect_run_artifacts(run_root)
    report = next(
        artifact
        for artifact in artifacts
        if artifact.key == BOP_EXPORT_MANIFEST
    )

    assert report.summary is not None
    assert report.summary["bop_export_ready_for_results"] is False
    assert report.summary["bop_export_blocker"] == "empty_bop_export_manifest"
    assert report.summary["export_count"] == 0
    assert "bop_export=empty_bop_export_manifest" in (
        report.to_dict()["display_label"]
    )


def test_bop_sidecar_summaries_label_blockers(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    bop_root = run_root / BOP_DIR
    models_root = bop_root / MODELS_DIR
    models_root.mkdir(parents=True)
    (bop_root / BOP_TARGETS_BOP19).write_text("[]\n")
    (bop_root / BOP_MULTIVIEW_TARGETS).write_text(
        json.dumps(
            {
                "schema_version": "posetestbot_bop_multiview_targets.v1",
                "split": "test",
                "targets": [],
            }
        )
    )
    (bop_root / BOP_COCO_ANNOTATIONS).write_text(
        json.dumps(
            {
                "schema_version": "posetestbot_coco_annotations.v1",
                "images": [{"id": 1, "file_name": "rgb/000000.png"}],
                "annotations": [],
                "categories": [],
            }
        )
    )
    (models_root / "models_info.json").write_text(
        json.dumps({"cube": {"source_name": "cube"}})
    )

    artifacts = collect_run_artifacts(run_root)
    by_key_source = {(artifact.key, artifact.source): artifact for artifact in artifacts}

    targets_record = by_key_source[(BOP_TARGETS_BOP19, "known")]
    assert targets_record.summary is not None
    assert targets_record.summary["bop_targets_ready_for_evaluation"] is False
    assert targets_record.summary["bop_targets_blocker"] == "empty_bop_targets"
    assert "bop_targets=empty_bop_targets" in (
        targets_record.to_dict()["display_label"]
    )

    multiview_record = by_key_source[(BOP_MULTIVIEW_TARGETS, "known")]
    assert multiview_record.summary is not None
    assert multiview_record.summary["bop_multiview_targets_ready"] is False
    assert multiview_record.summary["bop_multiview_targets_blocker"] == (
        "empty_bop_multiview_targets"
    )
    assert "bop_multiview=empty_bop_multiview_targets" in (
        multiview_record.to_dict()["display_label"]
    )

    coco_record = by_key_source[(BOP_COCO_ANNOTATIONS, "known")]
    assert coco_record.summary is not None
    assert coco_record.summary["bop_coco_annotations_ready"] is False
    assert coco_record.summary["bop_coco_annotations_blocker"] == (
        "missing_bop_coco_categories"
    )
    assert "bop_coco=missing_bop_coco_categories" in (
        coco_record.to_dict()["display_label"]
    )

    models_record = by_key_source[("models_info.json", "known")]
    assert models_record.summary is not None
    assert models_record.summary["bop_models_info_ready"] is False
    assert models_record.summary["bop_models_info_blocker"] == (
        "invalid_bop_models_info"
    )
    assert "bop_models=invalid_bop_models_info" in (
        models_record.to_dict()["display_label"]
    )


def test_metric_report_summary_labels_empty_report(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    report_path = run_root / RESULTS_DIR / METRICS_DIR / METRIC_REPORT_JSON
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "metric_report.v1",
                "dashboard": {},
                "rows": [],
            }
        )
    )

    artifacts = collect_run_artifacts(run_root)
    report = next(
        artifact for artifact in artifacts if artifact.key == METRIC_REPORT_JSON
    )

    assert report.summary is not None
    assert report.summary["metric_report_ready_for_dashboard"] is False
    assert report.summary["metric_report_blocker"] == "empty_metric_report"
    assert report.summary["row_count"] == 0
    assert "metric_report=empty_metric_report" in (
        report.to_dict()["display_label"]
    )


def test_bop_evaluation_report_summary_labels_failed_readiness(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / BOP_EVALUATION_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "bop_evaluation_report.v1",
                "status": "failed",
                "dry_run": False,
                "checks": [],
                "output_artifacts": [],
                "score_summary": {"score_file_count": 0, "metrics": {}, "files": []},
            }
        )
    )

    artifacts = collect_run_artifacts(run_root)
    report = next(
        artifact
        for artifact in artifacts
        if artifact.key == BOP_EVALUATION_REPORT
    )

    assert report.summary is not None
    assert report.summary["ready_for_metrics"] is False
    assert report.summary["bop_evaluation_report_blocker"] == (
        "failed_bop_evaluation_report"
    )
    assert "bop_eval=failed_bop_evaluation_report" in (
        report.to_dict()["display_label"]
    )


def test_bop_evaluation_report_summary_labels_failed_prerequisites(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / BOP_EVALUATION_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "bop_evaluation_report.v1",
                "status": "planned",
                "dry_run": True,
                "checks": [
                    *[
                        check
                        for check in ready_bop_evaluation_checks()
                        if check["name"] != "model_files"
                    ],
                    {"name": "model_files", "ok": False, "value": 0},
                ],
                "output_artifacts": [],
                "score_summary": {"score_file_count": 0, "metrics": {}, "files": []},
            }
        )
    )

    artifacts = collect_run_artifacts(run_root)
    report = next(
        artifact
        for artifact in artifacts
        if artifact.key == BOP_EVALUATION_REPORT
    )

    assert report.summary is not None
    assert report.summary["ready_for_metrics"] is False
    assert report.summary["bop_evaluation_report_blocker"] == (
        "failed_bop_evaluation_prerequisites"
    )
    assert report.summary["failed_check_count"] == 2
    assert report.summary["critical_failed_check_count"] == 1
    assert report.summary["critical_missing_check_count"] == 0
    assert "bop_eval=failed_bop_evaluation_prerequisites" in (
        report.to_dict()["display_label"]
    )


def test_bop_evaluation_report_summary_labels_missing_prerequisites(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / BOP_EVALUATION_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "bop_evaluation_report.v1",
                "status": "planned",
                "dry_run": True,
                "checks": [{"name": "result_file", "ok": True}],
                "output_artifacts": [],
                "score_summary": {"score_file_count": 0, "metrics": {}, "files": []},
            }
        )
    )

    artifacts = collect_run_artifacts(run_root)
    report = next(
        artifact
        for artifact in artifacts
        if artifact.key == BOP_EVALUATION_REPORT
    )

    assert report.summary is not None
    assert report.summary["ready_for_metrics"] is False
    assert report.summary["bop_evaluation_report_blocker"] == (
        "failed_bop_evaluation_prerequisites"
    )
    assert report.summary["critical_failed_check_count"] == 0
    assert report.summary["critical_missing_check_count"] == 6
    assert "bop_eval=failed_bop_evaluation_prerequisites" in (
        report.to_dict()["display_label"]
    )


def test_bop_evaluation_plan_summary_labels_empty_plan(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    result_file = run_root / RESULTS_DIR / BOP_DIR / "foundationpose_bop-test.csv"
    result_file.parent.mkdir(parents=True)
    result_file.write_text(
        "scene_id,im_id,obj_id,score,R,t,time\n"
        "1,0,1,1,1 0 0 0 1 0 0 0 1,0 0 10,-1\n"
    )
    (run_root / BOP_EVALUATION_PLAN).write_text(
        json.dumps(
            {
                "schema_version": "bop_evaluation_plan.v1",
                "dry_run": True,
                "result": {
                    "filename": result_file.name,
                    "path": result_file.as_posix(),
                },
                "command": [],
                "environment": {"BOP_PATH": run_root.as_posix()},
            }
        )
    )

    artifacts = collect_run_artifacts(run_root)
    plan = next(
        artifact for artifact in artifacts if artifact.key == BOP_EVALUATION_PLAN
    )

    assert plan.summary is not None
    assert plan.summary["bop_evaluation_plan_ready_for_execution"] is False
    assert plan.summary["bop_evaluation_plan_blocker"] == (
        "empty_bop_evaluation_plan"
    )
    assert plan.summary["command_count"] == 0
    assert "bop_eval_plan=empty_bop_evaluation_plan" in (
        plan.to_dict()["display_label"]
    )


def test_preview_artifact_json_text_and_directory(tmp_path: Path) -> None:
    run_root = create_artifact_fixture(tmp_path)

    json_preview = preview_artifact(run_root, PIPELINE_SEQUENCE_PLAN)
    assert json_preview["preview"]["type"] == "json"
    assert json_preview["preview"]["value"]["sequence_id"] == (
        "sync_to_bop_calibrated_dry_run"
    )

    metric_preview = preview_artifact(
        run_root,
        f"processed/synchronized/realsense_123/{ACCURACY_HRC_HUB}",
    )
    assert metric_preview["artifact"]["summary"]["type"] == "pose_accuracy_metrics"
    assert metric_preview["artifact"]["summary"]["methods"][0]["all_motions"][
        "RP_a"
    ] == [0.1, -0.1]

    text_preview = preview_artifact(
        run_root,
        f"{RESULTS_DIR}/{BOP_DIR}/foundationpose_bop-test.csv",
    )
    assert text_preview["preview"]["type"] == "text"
    assert "scene_id" in text_preview["preview"]["text"]

    directory_preview = preview_artifact(run_root, BOP_DIR)
    assert directory_preview["preview"]["type"] == "directory"
    assert directory_preview["preview"]["children"][0]["name"] == BOP_EXPORT_MANIFEST

    image_preview = preview_artifact(
        run_root,
        f"{BOP_DIR}/realsense_123/test/000001/{RGB_DIR}/000000.png",
    )
    assert image_preview["preview"]["type"] == "image"
    assert image_preview["preview"]["readable"] is True
    assert image_preview["preview"]["width"] == 4
    assert image_preview["preview"]["height"] == 3
    assert image_preview["preview"]["thumbnail_png_base64"]


def test_resolve_artifact_path_rejects_path_escape(tmp_path: Path) -> None:
    run_root = create_artifact_fixture(tmp_path)

    with pytest.raises(ArtifactPathError):
        resolve_artifact_path(run_root, "../outside.txt")


def test_metric_dashboard_summary_aggregates_legacy_metrics(tmp_path: Path) -> None:
    run_root = create_artifact_fixture(tmp_path)

    dashboard = metric_dashboard_summary(run_root)

    assert dashboard["type"] == "metric_dashboard"
    assert dashboard["metric_artifact_count"] == 3
    assert dashboard["direct_method_count"] == 2
    assert dashboard["combined_group_count"] == 2
    assert dashboard["bop_score_count"] == 1
    assert dashboard["method_count"] == 2
    assert dashboard["methods"] == [
        "ArUco_accuracy",
        "foundationpose_est5_track2_obj0",
    ]
    assert dashboard["best_by_AP_p"] == {
        "method": "foundationpose_est5_track2_obj0",
        "AP_p": 1.25,
        "relative_path": f"processed/synchronized/realsense_123/{ACCURACY_HRC_HUB}",
    }
    assert dashboard["best_bop19_average_recall"] == {
        "result_filename": "foundationpose_bop-test.csv",
        "bop19_average_recall": 0.75,
        "relative_path": BOP_EVALUATION_REPORT,
    }
    assert dashboard["bop_scores"][0]["metrics"] == {
        "bop19_average_recall": 0.75,
    }
    foundationpose = next(
        row
        for row in dashboard["direct_methods"]
        if row["method"] == "foundationpose_est5_track2_obj0"
    )
    assert foundationpose["all_motions"]["AP_p"] == 1.25
    assert dashboard["combined_groups"][0]["context"].startswith(
        "experiment_720p_6_0.2"
    )


def test_metric_dashboard_summary_skips_unready_bop_scores(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / BOP_EVALUATION_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "bop_evaluation_report.v1",
                "status": "planned",
                "dry_run": True,
                "result": {"filename": "foundationpose_bop-test.csv"},
                "checks": [
                    {"name": "targets_file", "ok": True},
                    {"name": "model_files", "ok": False, "value": 0},
                ],
                "output_artifacts": [],
                "score_summary": {
                    "score_file_count": 1,
                    "metrics": {"bop19_average_recall": 0.75},
                    "files": [],
                },
            }
        )
    )

    dashboard = metric_dashboard_summary(run_root)

    assert dashboard["bop_score_count"] == 0
    assert dashboard["bop_scores"] == []
    assert dashboard["best_bop19_average_recall"] is None


def test_bop_result_detail_reports_pose_rows_and_scene_links(tmp_path: Path) -> None:
    run_root = create_artifact_fixture(tmp_path)
    result_path = f"{RESULTS_DIR}/{BOP_DIR}/foundationpose_bop-test.csv"

    detail = bop_result_detail(run_root, result_path)

    assert detail["type"] == "bop_result_detail"
    assert detail["relative_path"] == result_path
    assert detail["metadata"]["method"] == "foundationpose"
    assert detail["metadata"]["dataset"] == "bop"
    assert detail["metadata"]["split"] == "test"
    assert detail["row_count"] == 1
    assert detail["scene_count"] == 1
    assert detail["rows"][0]["scene_id"] == 1
    assert detail["rows"][0]["im_id"] == 0
    assert detail["rows"][0]["obj_id"] == 1
    assert detail["rows"][0]["R_matrix"] == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert detail["rows"][0]["t"] == [0.0, 0.0, 10.0]
    assert detail["rows"][0]["scene"]["relative_scene_folder"] == (
        f"{BOP_DIR}/realsense_123/test/000001"
    )


def test_bop_scene_detail_reports_frame_contracts(tmp_path: Path) -> None:
    run_root = create_artifact_fixture(tmp_path)
    scene_path = f"{BOP_DIR}/realsense_123/test/000001"

    detail = bop_scene_detail(run_root, scene_path)

    assert detail["type"] == "bop_scene_detail"
    assert detail["relative_path"] == scene_path
    assert detail["summary"]["type"] == "bop_scene"
    assert detail["files"]["frame_map"] is True
    assert detail["frames"][0]["image_id"] == 0
    assert detail["frames"][0]["rgb"]["exists"] is True
    assert detail["frames"][0]["rgb"]["relative_path"] == (
        f"{scene_path}/{RGB_DIR}/000000.png"
    )
    assert detail["frames"][0]["depth"]["summary"]["dtype"] == "uint16"
    assert detail["frames"][0]["gt_count"] == 1
    assert detail["frames"][0]["gt_info"] == [{"bbox_obj": [0, 0, 4, 3]}]
    assert detail["frames"][0]["mask_files"] == ["000000_000000.png"]
    assert detail["frames"][0]["mask_artifacts"][0]["relative_path"] == (
        f"{scene_path}/mask/000000_000000.png"
    )
    assert detail["frames"][0]["frame_map"]["source_rgb"] == "rgb/raw_000010.png"


def test_bop_frame_detail_joins_scene_assets_and_result_rows(
    tmp_path: Path,
) -> None:
    run_root = create_artifact_fixture(tmp_path)
    scene_path = f"{BOP_DIR}/realsense_123/test/000001"
    result_path = f"{RESULTS_DIR}/{BOP_DIR}/foundationpose_bop-test.csv"

    detail = bop_frame_detail(
        run_root,
        scene_path,
        image_id=0,
        result_path=result_path,
    )

    assert detail["type"] == "bop_frame_detail"
    assert detail["relative_path"] == scene_path
    assert detail["scene"]["scene_id"] == 1
    assert detail["image_key"] == "0"
    assert detail["rgb"]["relative_path"] == f"{scene_path}/{RGB_DIR}/000000.png"
    assert detail["depth"]["summary"]["dtype"] == "uint16"
    assert detail["camera"] == {
        "cam_K": [1, 0, 2, 0, 1, 1, 0, 0, 1],
        "depth_scale": 1.0,
    }
    assert detail["gt_count"] == 1
    assert detail["gt"] == [{"obj_id": 1}]
    assert detail["gt_info"] == [{"bbox_obj": [0, 0, 4, 3]}]
    assert detail["mask_artifacts"][0]["relative_path"] == (
        f"{scene_path}/mask/000000_000000.png"
    )
    assert detail["frame_map"]["source_depth"] == "depth/raw_000010.png"
    assert detail["result"]["relative_path"] == result_path
    assert detail["result"]["matching_row_count"] == 1
    assert detail["result"]["projected_origin_count"] == 1
    assert detail["result"]["projected_model_bbox_count"] == 1
    assert detail["result"]["rows"][0]["obj_id"] == 1
    assert detail["result"]["rows"][0]["projected_origin"] == {
        "u": 2.0,
        "v": 1.0,
        "depth": 10.0,
        "source": "bop19_t_object_origin",
    }
    assert detail["result"]["rows"][0]["projected_model_bbox"] == {
        "bbox": [1.0, 0.5, 2.0, 1.0],
        "vertex_count": 4,
        "projected_vertex_count": 4,
        "model_relative_path": f"{BOP_DIR}/{MODELS_DIR}/obj_000001.ply",
        "object_name": "cube",
        "source": "bop19_pose_model_vertices",
    }
    assert detail["result"]["rows"][0]["scene"]["relative_scene_folder"] == (
        scene_path
    )


def test_bop_frame_overlay_png_renders_masks_gt_and_result_labels(
    tmp_path: Path,
) -> None:
    run_root = create_artifact_fixture(tmp_path)
    scene_path = f"{BOP_DIR}/realsense_123/test/000001"
    result_path = f"{RESULTS_DIR}/{BOP_DIR}/foundationpose_bop-test.csv"

    png_bytes = render_bop_frame_overlay_png(
        run_root,
        scene_path,
        image_id=0,
        result_path=result_path,
    )

    image = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    assert image.shape[:2] == (3, 4)
    assert int(image.max()) > 0
