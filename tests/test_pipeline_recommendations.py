from __future__ import annotations

import json
from pathlib import Path

from posetestbot.io.artifacts import (
    ACCURACY_HRC_HUB,
    ARUCO_COVERAGE_REPORT,
    ARUCO_POSE_ESTIMATION,
    BOP_DIR,
    BOP_EVALUATION_REPORT,
    BOP_EXPORT_MANIFEST,
    BOP_RESULT_EXPORT_MANIFEST,
    BOP_TARGETS_BOP19,
    CALIBRATION_CANDIDATES,
    CALIBRATION_OBSERVATIONS,
    CALIBRATION_PREFLIGHT_REPORT,
    CALIBRATION_PROFILES,
    CALIBRATION_SOLVER_REPORT,
    CALIBRATION_VALIDATION_REPORT,
    CAPTURE_EXECUTION_PLAN,
    CAPTURE_EXECUTION_REPORT,
    CAPTURE_PLAN,
    CAPTURE_PLAN_PREFLIGHT_REPORT,
    CHARUCO_POSE_ESTIMATION,
    FOUNDATIONPOSE_PLAN,
    METRIC_REPORT_JSON,
    METRICS_DIR,
    PROCESSED_DIR,
    PIPELINE_SEQUENCE_PLAN,
    RAW_ROBOT_EE_POSES,
    REWRITE_GATE_REPORT,
    REWRITE_STATUS_REPORT,
    RESULTS_DIR,
    RUN_CONFIG,
    RUN_PREFLIGHT_REPORT,
    SYNC_QUALITY_REPORT,
    SYNCHRONIZED_DIR,
)
from posetestbot.pipeline import recommendations as recommendations_module
from posetestbot.pipeline.recommendations import build_pipeline_recommendations
from posetestbot.pipeline.rewrite_gate import write_rewrite_status_report
from posetestbot.pipeline.run_config import create_run_config, write_run_config


def recommendation_by_id(payload: dict, recommendation_id: str) -> dict:
    return next(
        recommendation
        for recommendation in payload["recommendations"]
        if recommendation["id"] == recommendation_id
    )


def recommendation_ids(payload: dict) -> set[str]:
    return {
        recommendation["id"] for recommendation in payload["recommendations"]
    }


def write_bop_result_manifest_fixture(
    run_root: Path,
    *,
    write_targets: bool = True,
) -> Path:
    result_file = run_root / RESULTS_DIR / "bop" / "foundationpose_bop-test.csv"
    result_file.parent.mkdir(parents=True)
    result_file.write_text(
        "scene_id,im_id,obj_id,score,R,t,time\n"
        "1,0,1,1,1 0 0 0 1 0 0 0 1,0 0 0,-1\n"
    )
    if write_targets:
        target_file = run_root / BOP_DIR / BOP_TARGETS_BOP19
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(
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
            + "\n"
        )
    (run_root / BOP_RESULT_EXPORT_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": "bop_result_export_manifest.v1",
                "results": [
                    {"filename": result_file.name, "path": result_file.as_posix()}
                ],
            }
        )
    )
    return result_file


def write_ready_bop_export_manifest(
    run_root: Path,
    *,
    write_object_models: bool = True,
) -> None:
    manifest = run_root / BOP_DIR / BOP_EXPORT_MANIFEST
    manifest.parent.mkdir(parents=True)
    payload = {
        "schema_version": "bop_export_manifest.v1",
        "exports": [{"sensor_name": "realsense_123", "scene_id": 1}],
    }
    if write_object_models:
        payload["object_models"] = [
            {
                "object_name": "cube",
                "obj_id": 1,
                "source_path": "object_models/cube.ply",
                "bop_path": "bop/models/obj_000001.ply",
            }
        ]
    manifest.write_text(
        json.dumps(payload)
        + "\n"
    )


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


def write_ready_aruco_coverage_report(run_root: Path) -> None:
    (run_root / ARUCO_COVERAGE_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "aruco_coverage_report.v1",
                "overall_status": "ok",
                "sensor_count": 1,
                "frame_count": 1,
                "valid_pose_count": 1,
                "checks": [{"name": "aruco_coverage:realsense_123", "status": "ok"}],
            }
        )
        + "\n"
    )


def write_ready_capture_plan(run_root: Path) -> None:
    (run_root / CAPTURE_PLAN).write_text(
        json.dumps(
            {
                "schema_version": "capture_plan.v1",
                "commands": [
                    {
                        "role": "robot_pose_receiver",
                        "command": [
                            "uv",
                            "run",
                            "python",
                            "scripts/pose_receiver_udp_json.py",
                        ],
                    }
                ],
            }
        )
        + "\n"
    )


def write_ready_pipeline_sequence_plan(run_root: Path) -> None:
    (run_root / PIPELINE_SEQUENCE_PLAN).write_text(
        json.dumps(
            {
                "schema_version": "pipeline_sequence_plan.v1",
                "sequence_id": "sync_aruco",
                "plan_only": True,
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
                    }
                ],
            }
        )
        + "\n"
    )


def write_prepared_blenderproc_fixture(run_root: Path) -> Path:
    sensor_root = run_root / PROCESSED_DIR / SYNCHRONIZED_DIR / "realsense_123"
    blenderproc_root = sensor_root / "blenderproc"
    blenderproc_root.mkdir(parents=True)
    (blenderproc_root / "objects.json").write_text('{"cube": {"obj_id": 1}}\n')
    return sensor_root


def test_recommendations_start_with_run_config_for_empty_run(tmp_path: Path) -> None:
    run_root = tmp_path / "empty-run"

    payload = build_pipeline_recommendations(run_root)

    assert payload["schema_version"] == "pipeline_recommendations.v1"
    assert payload["facts"]["has_run_config"] is False
    assert payload["facts"]["run_config_ready_for_pipeline"] is False
    assert payload["facts"]["run_config_blocker"] == "missing_run_config"
    recommendation = recommendation_by_id(payload, "create_run_config")
    assert recommendation["action_type"] == "api"
    assert recommendation["endpoint"] == "/run-config"
    assert recommendation["command"][:4] == [
        "uv",
        "run",
        "python",
        "scripts/create_run_config.py",
    ]
    assert recommendation["expected_artifacts"] == [RUN_CONFIG]


def test_recommendations_rewrite_invalid_run_config(tmp_path: Path) -> None:
    run_root = tmp_path / "invalid-run-config"
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

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_run_config"] is True
    assert payload["facts"]["run_config_ready_for_pipeline"] is False
    assert payload["facts"]["run_config_blocker"] == "invalid_run_config"
    assert "capture.sensors" in payload["facts"]["run_config_error"]
    recommendation = recommendation_by_id(payload, "create_run_config")
    assert recommendation["label"] == "Rewrite run config"
    assert recommendation["reason"].startswith(
        f"{RUN_CONFIG} is invalid and should be rewritten:"
    )
    recommendation_ids_ = recommendation_ids(payload)
    assert "write_run_preflight" not in recommendation_ids_
    assert "write_capture_plan" not in recommendation_ids_


def test_recommendations_suggest_sync_for_legacy_raw_poses(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "legacy-raw-run"
    run_root.mkdir()
    (run_root / RAW_ROBOT_EE_POSES).write_text("[]\n")

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_raw_robot_poses"] is True
    assert payload["facts"]["capture_execution_blocks_sync"] is False
    recommendation = recommendation_by_id(payload, "sync_raw_capture")
    assert recommendation["stage_id"] == "sync_run"
    assert recommendation["expected_artifacts"] == [
        f"{PROCESSED_DIR}/{SYNCHRONIZED_DIR}/<sensor>"
    ]


def test_recommendations_suggest_capture_plan_after_run_config(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "configured-run"
    config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, config)

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_run_config"] is True
    assert payload["facts"]["run_config_ready_for_pipeline"] is True
    assert payload["facts"]["run_config_blocker"] is None
    recommendation = recommendation_by_id(payload, "write_capture_plan")
    assert recommendation["stage_id"] == "capture_plan"
    assert recommendation["endpoint"] == "/capture-plan"
    assert recommendation["command"][:4] == [
        "uv",
        "run",
        "python",
        "scripts/run_capture_plan_stage.py",
    ]
    assert recommendation["expected_artifacts"] == [CAPTURE_PLAN]


def test_recommendations_suggest_rewrite_status_after_run_config(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "rewrite-status-configured-run"
    config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, config)

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["rewrite_status_expected"] is True
    assert payload["facts"]["has_rewrite_status_report"] is False
    assert payload["facts"]["rewrite_status_ready_for_inspection"] is False
    assert payload["facts"]["rewrite_status_blocker"] == (
        "missing_rewrite_status_report"
    )
    assert payload["facts"]["rewrite_status_overall_status"] == "blocked"
    assert payload["facts"]["rewrite_status_ready_gate_count"] == 0
    assert payload["facts"]["rewrite_status_gate_count"] == 4
    recommendation = recommendation_by_id(payload, "write_rewrite_status")
    assert recommendation["stage_id"] == "rewrite_status"
    assert recommendation["expected_artifacts"] == [REWRITE_STATUS_REPORT]
    assert recommendation["command"] == [
        "uv",
        "run",
        "python",
        "scripts/run_rewrite_status.py",
        run_root.as_posix(),
        "--write",
    ]


def test_recommendations_refresh_stale_rewrite_status_report(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "stale-rewrite-status-run"
    config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, config)
    (run_root / REWRITE_STATUS_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "rewrite_status_report.v1",
                "overall_status": "ready",
                "summary": {
                    "gate_count": 4,
                    "ready_gate_count": 4,
                    "blocked_gate_count": 0,
                    "check_count": 17,
                    "ready_check_count": 17,
                    "blocked_check_count": 0,
                },
                "gates": [],
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_rewrite_status_report"] is True
    assert payload["facts"]["rewrite_status_ready_for_inspection"] is False
    assert payload["facts"]["rewrite_status_blocker"] == "stale_rewrite_status_report"
    recommendation = recommendation_by_id(payload, "write_rewrite_status")
    assert recommendation["label"] == "Refresh rewrite status"
    assert recommendation["reason"] == (
        f"{REWRITE_STATUS_REPORT} is stale relative to current gate evidence."
    )


def test_recommendations_skip_current_rewrite_status_report(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "current-rewrite-status-run"
    config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, config)
    write_rewrite_status_report(run_root)

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_rewrite_status_report"] is True
    assert payload["facts"]["rewrite_status_ready_for_inspection"] is True
    assert payload["facts"]["rewrite_status_blocker"] is None
    assert "write_rewrite_status" not in recommendation_ids(payload)
    assert payload["facts"]["rewrite_status_next_action_label"] == "Audit rewrite gate"
    recommendation = recommendation_by_id(
        payload,
        "follow_rewrite_status_next_action",
    )
    assert recommendation["label"] == "Audit rewrite gate"
    assert recommendation["action_type"] == "command"
    assert recommendation["command"] == payload["facts"][
        "rewrite_status_next_action_command"
    ]
    assert "Next rewrite blocker: run_preflight." in recommendation["reason"]


def test_recommendations_surface_all_rewrite_status_next_actions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "current-rewrite-status-multi-action-run"
    config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, config)
    report_path, report = write_rewrite_status_report(run_root)
    report["next_actions"] = [
        {
            "label": "Inspect sensor status",
            "command": [
                "uv",
                "run",
                "python",
                "scripts/sensor_status.py",
                "--json",
                "--check-expected",
            ],
            "reason": "Inspect camera SDK/device visibility first.",
            "blocks_on": [
                "sensor:realsense_d435",
                "sensor:oak_d_pro",
                "sensor:zed_2i",
            ],
        },
        {
            "label": "Refresh hardware status after sensor fix",
            "command": [
                "uv",
                "run",
                "python",
                "scripts/run_hardware_status_stage.py",
                run_root.as_posix(),
            ],
            "reason": "Refresh the run-scoped hardware snapshot after fixing sensors.",
            "blocks_on": [
                "sensor:realsense_d435",
                "sensor:oak_d_pro",
                "sensor:zed_2i",
            ],
        },
    ]
    report_path.write_text(json.dumps(report) + "\n")
    monkeypatch.setattr(
        recommendations_module,
        "build_rewrite_status_report",
        lambda *args, **kwargs: report,
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["rewrite_status_next_action_labels"] == [
        "Inspect sensor status",
        "Refresh hardware status after sensor fix",
    ]
    assert payload["facts"]["rewrite_status_next_action_commands"] == [
        [
            "uv",
            "run",
            "python",
            "scripts/sensor_status.py",
            "--json",
            "--check-expected",
        ],
        [
            "uv",
            "run",
            "python",
            "scripts/run_hardware_status_stage.py",
            run_root.as_posix(),
        ],
    ]
    assert payload["recommendations"][0]["id"] == "follow_rewrite_status_next_action"
    assert payload["recommendations"][0]["label"] == "Inspect sensor status"
    assert payload["recommendations"][1]["id"] == (
        "follow_rewrite_status_next_action_2"
    )
    assert payload["recommendations"][1]["label"] == (
        "Refresh hardware status after sensor fix"
    )
    assert payload["recommendations"][0]["blocks_on"] == [
        "sensor:realsense_d435",
        "sensor:oak_d_pro",
        "sensor:zed_2i",
    ]
    assert payload["recommendations"][1]["blocks_on"] == [
        "sensor:realsense_d435",
        "sensor:oak_d_pro",
        "sensor:zed_2i",
    ]
    assert "Refresh the run-scoped hardware snapshot after fixing sensors." in (
        payload["recommendations"][1]["reason"]
    )
    assert "Next rewrite blocker:" in payload["recommendations"][1]["reason"]


def test_recommendations_skip_current_mixed_root_rewrite_status_report(
    tmp_path: Path,
) -> None:
    status_root = tmp_path / "current-mixed-rewrite-status-run"
    fake_root = tmp_path / "fake-gate-root"
    write_rewrite_status_report(
        status_root,
        gate_run_roots={"rewrite_fake_end_to_end.v1": fake_root},
    )

    payload = build_pipeline_recommendations(status_root)

    assert payload["facts"]["has_rewrite_status_report"] is True
    assert payload["facts"]["rewrite_status_ready_for_inspection"] is True
    assert payload["facts"]["rewrite_status_blocker"] is None
    assert payload["facts"]["rewrite_status_gate_run_roots"][
        "rewrite_fake_end_to_end.v1"
    ] == fake_root.as_posix()
    assert "write_rewrite_status" not in recommendation_ids(payload)
    assert "create_run_config" not in recommendation_ids(payload)
    assert payload["recommendations"][0]["id"] == "follow_rewrite_status_next_action"


def test_recommendations_refresh_stale_mixed_root_rewrite_status_with_overrides(
    tmp_path: Path,
) -> None:
    status_root = tmp_path / "stale-mixed-rewrite-status-run"
    fake_root = tmp_path / "fake-gate-root"
    status_root.mkdir()
    (status_root / REWRITE_STATUS_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "rewrite_status_report.v1",
                "run_root": status_root.as_posix(),
                "gate_run_roots": {
                    "rewrite_fake_end_to_end.v1": fake_root.as_posix(),
                },
                "overall_status": "ready",
                "summary": {
                    "gate_count": 4,
                    "ready_gate_count": 4,
                    "blocked_gate_count": 0,
                    "check_count": 17,
                    "ready_check_count": 17,
                    "blocked_check_count": 0,
                },
                "gates": [],
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(status_root)

    assert payload["facts"]["rewrite_status_ready_for_inspection"] is False
    assert payload["facts"]["rewrite_status_blocker"] == "stale_rewrite_status_report"
    recommendation = recommendation_by_id(payload, "write_rewrite_status")
    assert recommendation["command"] == [
        "uv",
        "run",
        "python",
        "scripts/run_rewrite_status.py",
        status_root.as_posix(),
        "--write",
        "--gate-run-root",
        f"rewrite_fake_end_to_end.v1={fake_root.as_posix()}",
    ]
    assert recommendation["resources"] == ["disk_io"]


def test_recommendations_suggest_run_preflight_after_run_config(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "configured-preflight-run"
    config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, config)

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_run_config"] is True
    assert payload["facts"]["has_run_preflight"] is False
    assert payload["facts"]["run_preflight_ready_for_queue"] is False
    assert payload["facts"]["run_preflight_queue_blocker"] == "missing_preflight"
    recommendation = recommendation_by_id(payload, "write_run_preflight")
    assert recommendation["action_type"] == "api"
    assert recommendation["endpoint"] == "/pipeline/preflight"
    assert recommendation["method"] == "POST"
    assert recommendation["resources"] == ["disk_io"]
    assert recommendation["expected_artifacts"] == [RUN_PREFLIGHT_REPORT]
    assert recommendation["command"] == [
        "uv",
        "run",
        "python",
        "scripts/run_preflight.py",
        run_root.as_posix(),
        "--write",
    ]
    assert "queue_saved_sequence" not in recommendation_ids(payload)


def test_recommendations_skip_run_preflight_after_report_exists(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "preflighted-run"
    config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, config)
    (run_root / RUN_PREFLIGHT_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "run_preflight.v1",
                "overall_status": "ok",
                "config": config.to_dict(),
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_run_preflight"] is True
    assert payload["facts"]["run_preflight_status"] == "ok"
    assert payload["facts"]["run_preflight_matches_config"] is True
    assert payload["facts"]["run_preflight_ready_for_queue"] is True
    assert payload["facts"]["run_preflight_queue_blocker"] is None
    assert payload["facts"]["has_pipeline_sequence_plan"] is False
    assert payload["facts"]["pipeline_sequence_plan_ready_for_queue"] is False
    assert payload["facts"]["pipeline_sequence_plan_blocker"] == (
        "missing_pipeline_sequence_plan"
    )
    assert "write_run_preflight" not in recommendation_ids(payload)
    recommendation = recommendation_by_id(payload, "queue_saved_sequence")
    assert recommendation["endpoint"] == "/pipeline/run-config"
    assert recommendation["expected_artifacts"] == [PIPELINE_SEQUENCE_PLAN]


def test_recommendations_rebuild_empty_pipeline_sequence_plan(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "empty-sequence-plan-run"
    config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, config)
    (run_root / RUN_PREFLIGHT_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "run_preflight.v1",
                "overall_status": "ok",
                "config": config.to_dict(),
            }
        )
        + "\n"
    )
    (run_root / PIPELINE_SEQUENCE_PLAN).write_text(
        '{"schema_version": "pipeline_sequence_plan.v1", "steps": []}\n'
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_pipeline_sequence_plan"] is True
    assert payload["facts"]["pipeline_sequence_plan_ready_for_queue"] is False
    assert payload["facts"]["pipeline_sequence_plan_blocker"] == (
        "empty_pipeline_sequence_plan"
    )
    assert payload["facts"]["pipeline_sequence_plan_step_count"] == 0
    recommendation = recommendation_by_id(payload, "queue_saved_sequence")
    assert recommendation["reason"] == (
        f"{PIPELINE_SEQUENCE_PLAN} is not ready for queueing "
        "(empty_pipeline_sequence_plan)."
    )


def test_recommendations_skip_ready_pipeline_sequence_plan(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "ready-sequence-plan-run"
    config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, config)
    (run_root / RUN_PREFLIGHT_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "run_preflight.v1",
                "overall_status": "ok",
                "config": config.to_dict(),
            }
        )
        + "\n"
    )
    write_ready_pipeline_sequence_plan(run_root)

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_pipeline_sequence_plan"] is True
    assert payload["facts"]["pipeline_sequence_plan_ready_for_queue"] is True
    assert payload["facts"]["pipeline_sequence_plan_blocker"] is None
    assert payload["facts"]["pipeline_sequence_plan_step_count"] == 1
    assert "queue_saved_sequence" not in recommendation_ids(payload)


def test_recommendations_refresh_failed_run_preflight(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "failed-preflight-run"
    config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, config)
    (run_root / RUN_PREFLIGHT_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "run_preflight.v1",
                "overall_status": "error",
                "config": config.to_dict(),
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["run_preflight_status"] == "error"
    assert payload["facts"]["run_preflight_matches_config"] is True
    assert payload["facts"]["run_preflight_ready_for_queue"] is False
    assert payload["facts"]["run_preflight_queue_blocker"] == "failed_preflight"
    recommendation = recommendation_by_id(payload, "write_run_preflight")
    assert recommendation["label"] == "Refresh run preflight"
    assert recommendation["reason"] == (
        f"{RUN_PREFLIGHT_REPORT} has overall_status=error."
    )
    assert "queue_saved_sequence" not in recommendation_ids(payload)


def test_recommendations_refresh_invalid_run_preflight(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "invalid-preflight-run"
    config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, config)
    (run_root / RUN_PREFLIGHT_REPORT).write_text("[]\n")

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["run_preflight_status"] is None
    assert payload["facts"]["run_preflight_matches_config"] is None
    assert payload["facts"]["run_preflight_ready_for_queue"] is False
    assert payload["facts"]["run_preflight_queue_blocker"] == "invalid_preflight"
    recommendation = recommendation_by_id(payload, "write_run_preflight")
    assert recommendation["label"] == "Refresh run preflight"
    assert recommendation["reason"] == (
        f"{RUN_PREFLIGHT_REPORT} is invalid and should be rewritten."
    )
    assert "queue_saved_sequence" not in recommendation_ids(payload)


def test_recommendations_refresh_stale_run_preflight(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "stale-preflight-run"
    original_config = create_run_config(run_root=run_root, sequence_id="sync_aruco")
    write_run_config(run_root, original_config)
    (run_root / RUN_PREFLIGHT_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "run_preflight.v1",
                "overall_status": "ok",
                "config": original_config.to_dict(),
            }
        )
        + "\n"
    )
    updated_config = create_run_config(
        run_root=run_root,
        sequence_id="sync_to_bop_dry_run",
    )
    write_run_config(run_root, updated_config)

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["run_preflight_status"] == "ok"
    assert payload["facts"]["run_preflight_matches_config"] is False
    assert payload["facts"]["run_preflight_ready_for_queue"] is False
    assert payload["facts"]["run_preflight_queue_blocker"] == "stale_preflight"
    recommendation = recommendation_by_id(payload, "write_run_preflight")
    assert recommendation["label"] == "Refresh run preflight"
    assert recommendation["reason"] == (
        f"{RUN_PREFLIGHT_REPORT} does not match the current {RUN_CONFIG}."
    )
    assert "queue_saved_sequence" not in recommendation_ids(payload)


def write_capture_execution_artifact_chain(
    run_root: Path,
    *,
    report_status: str | None = None,
    robot_mode: str = "fake",
) -> None:
    config = create_run_config(
        run_root=run_root,
        robot_mode=robot_mode,
        sequence_id="fake_capture_execution",
    )
    write_run_config(run_root, config)
    write_ready_capture_plan(run_root)
    (run_root / CAPTURE_PLAN_PREFLIGHT_REPORT).write_text(
        '{"schema_version": "capture_plan_preflight.v1", "overall_status": "ok"}\n'
    )
    (run_root / CAPTURE_EXECUTION_PLAN).write_text(
        json.dumps(
            {
                "schema_version": "capture_execution_plan.v1",
                "status": "ok",
                "mode": "pose_only_fake",
                "ready_to_execute": True,
            }
        )
        + "\n"
    )
    if report_status is not None:
        (run_root / CAPTURE_EXECUTION_REPORT).write_text(
            json.dumps(
                {
                    "schema_version": "capture_execution_report.v1",
                    "status": report_status,
                    "mode": "full" if robot_mode == "real" else "pose_only_fake",
                    "allow_cameras": robot_mode == "real",
                    "raw_pose_count": 1,
                    "selected_roles": (
                        ["sensor_capture", "robot_pose_receiver"]
                        if robot_mode == "real"
                        else ["robot_pose_receiver"]
                    ),
                }
            )
            + "\n"
        )


def test_recommendations_rerun_failed_capture_plan_preflight(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "failed-capture-plan-preflight-run"
    config = create_run_config(run_root=run_root, sequence_id="fake_capture_execution")
    write_run_config(run_root, config)
    write_ready_capture_plan(run_root)
    (run_root / CAPTURE_PLAN_PREFLIGHT_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "capture_plan_preflight.v1",
                "overall_status": "error",
                "checks": [{"name": "command_shape", "status": "error"}],
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_capture_plan"] is True
    assert payload["facts"]["has_capture_plan_preflight"] is True
    assert payload["facts"]["capture_plan_preflight_status"] == "error"
    assert payload["facts"]["capture_plan_preflight_ready"] is False
    assert payload["facts"]["capture_plan_preflight_blocker"] == (
        "failed_capture_plan_preflight"
    )
    recommendation = recommendation_by_id(payload, "preflight_capture_plan")
    assert recommendation["reason"] == (
        f"{CAPTURE_PLAN_PREFLIGHT_REPORT} has overall_status=error."
    )
    assert "plan_fake_capture_execution" not in recommendation_ids(payload)


def test_recommendations_rebuild_empty_capture_plan(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "empty-capture-plan-run"
    config = create_run_config(run_root=run_root, sequence_id="fake_capture_execution")
    write_run_config(run_root, config)
    (run_root / CAPTURE_PLAN).write_text(
        '{"schema_version": "capture_plan.v1", "commands": []}\n'
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_capture_plan"] is True
    assert payload["facts"]["capture_plan_ready_for_preflight"] is False
    assert payload["facts"]["capture_plan_blocker"] == "empty_capture_plan"
    assert payload["facts"]["capture_plan_command_count"] == 0
    recommendation = recommendation_by_id(payload, "write_capture_plan")
    assert recommendation["reason"] == (
        f"{CAPTURE_PLAN} is not ready for preflight (empty_capture_plan)."
    )
    assert "preflight_capture_plan" not in recommendation_ids(payload)


def test_recommendations_rebuild_unready_capture_execution_plan(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "unready-capture-execution-plan-run"
    config = create_run_config(run_root=run_root, sequence_id="fake_capture_execution")
    write_run_config(run_root, config)
    write_ready_capture_plan(run_root)
    (run_root / CAPTURE_PLAN_PREFLIGHT_REPORT).write_text(
        '{"schema_version": "capture_plan_preflight.v1", "overall_status": "ok"}\n'
    )
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
                        "message": "Capture-plan preflight status is error",
                    }
                ],
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_capture_execution_plan"] is True
    assert payload["facts"]["capture_execution_plan_status"] == "error"
    assert payload["facts"]["capture_execution_plan_ready"] is False
    assert payload["facts"]["capture_execution_plan_blocker"] == (
        "failed_capture_execution_plan"
    )
    assert payload["facts"]["capture_execution_plan_blocked_checks"] == [
        {
            "name": "capture_plan_preflight",
            "status": "error",
            "message": "Capture-plan preflight status is error",
        }
    ]
    assert payload["facts"]["capture_execution_blocks_sync"] is False
    recommendation = recommendation_by_id(payload, "plan_fake_capture_execution")
    assert recommendation["reason"] == (
        f"{CAPTURE_EXECUTION_PLAN} is blocked by capture_plan_preflight: "
        "Capture-plan preflight status is error."
    )
    assert "run_fake_capture_execution" not in recommendation_ids(payload)


def test_recommendations_plan_full_capture_for_real_run_config(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "real-capture-execution-plan-run"
    config = create_run_config(
        run_root=run_root,
        robot_mode="real",
        sequence_id="fake_capture_execution",
    )
    write_run_config(run_root, config)
    write_ready_capture_plan(run_root)
    (run_root / CAPTURE_PLAN_PREFLIGHT_REPORT).write_text(
        '{"schema_version": "capture_plan_preflight.v1", "overall_status": "ok"}\n'
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["run_config_robot_mode"] == "real"
    assert payload["facts"]["run_config_targets_real_robot"] is True
    recommendation = recommendation_by_id(payload, "plan_full_capture_execution")
    assert recommendation["stage_id"] == "capture_execution_plan"
    assert recommendation["command"] == [
        "uv",
        "run",
        "python",
        "scripts/run_capture_execution_plan.py",
        run_root.as_posix(),
        "--mode",
        "full",
        "--allow-cameras",
        "--allow-real-robot",
        "--include-sensors",
    ]
    assert recommendation["expected_artifacts"] == [CAPTURE_EXECUTION_PLAN]
    assert "plan_fake_capture_execution" not in recommendation_ids(payload)


def test_recommendations_rerun_failed_capture_execution_report(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "failed-capture-execution-run"
    write_capture_execution_artifact_chain(run_root, report_status="failed")
    (run_root / RAW_ROBOT_EE_POSES).write_text("[]\n")

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_capture_execution_report"] is True
    assert payload["facts"]["has_raw_robot_poses"] is True
    assert payload["facts"]["capture_execution_report_status"] == "failed"
    assert payload["facts"]["capture_execution_report_ready"] is False
    assert payload["facts"]["capture_execution_report_blocker"] == (
        "failed_capture_execution_report"
    )
    assert payload["facts"]["capture_execution_blocks_sync"] is True
    recommendation = recommendation_by_id(payload, "run_fake_capture_execution")
    assert recommendation["stage_id"] == "capture_execution"
    assert recommendation["reason"] == (
        f"{CAPTURE_EXECUTION_REPORT} has status=failed."
    )
    assert recommendation["expected_artifacts"] == [
        CAPTURE_EXECUTION_REPORT,
        RAW_ROBOT_EE_POSES,
    ]
    assert "sync_raw_capture" not in recommendation_ids(payload)


def test_recommendations_rerun_invalid_capture_execution_report(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "invalid-capture-execution-run"
    write_capture_execution_artifact_chain(run_root)
    (run_root / CAPTURE_EXECUTION_REPORT).write_text("[]\n")

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_capture_execution_report"] is True
    assert payload["facts"]["capture_execution_report_status"] is None
    assert payload["facts"]["capture_execution_report_ready"] is False
    assert payload["facts"]["capture_execution_report_blocker"] == (
        "invalid_capture_execution_report"
    )
    recommendation = recommendation_by_id(payload, "run_fake_capture_execution")
    assert recommendation["reason"] == (
        f"{CAPTURE_EXECUTION_REPORT} is invalid and should be rerun."
    )


def test_recommendations_audit_full_capture_gate_for_real_capture_report(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "real-capture-gate-run"
    write_capture_execution_artifact_chain(
        run_root,
        robot_mode="real",
        report_status="succeeded",
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["capture_execution_report_ready"] is True
    assert payload["facts"]["full_capture_gate_status"] == "blocked"
    assert "sensor_frames:realsense_825412070181" in payload["facts"][
        "full_capture_gate_next_blockers"
    ]
    recommendation = recommendation_by_id(payload, "audit_full_capture_gate")
    assert recommendation["stage_id"] == "rewrite_gate"
    assert recommendation["expected_artifacts"] == [REWRITE_GATE_REPORT]
    assert recommendation["command"] == [
        "uv",
        "run",
        "python",
        "scripts/run_rewrite_gate.py",
        run_root.as_posix(),
        "--gate",
        "rewrite_full_capture.v1",
        "--write",
    ]
    assert "run_fake_capture_execution" not in recommendation_ids(payload)


def test_recommendations_accept_succeeded_capture_execution_report(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "succeeded-capture-execution-run"
    write_capture_execution_artifact_chain(run_root, report_status="succeeded")

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["capture_execution_report_status"] == "succeeded"
    assert payload["facts"]["capture_execution_report_ready"] is True
    assert payload["facts"]["capture_execution_report_blocker"] is None
    assert payload["facts"]["capture_execution_blocks_sync"] is False
    assert "run_fake_capture_execution" not in recommendation_ids(payload)


def test_recommendations_suggest_calibration_preflight_for_configured_profiles(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "configured-calibration-run"
    config = create_run_config(
        run_root=run_root,
        sequence_id="sync_to_bop_dry_run",
        calibration_profiles="profiles/lab_calibration.json",
    )
    write_run_config(run_root, config)

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["calibration_profile_source"] == (
        "profiles/lab_calibration.json"
    )
    assert payload["facts"]["has_calibration_profiles_configured"] is True
    assert payload["facts"]["has_calibration_preflight"] is False
    recommendation = recommendation_by_id(payload, "preflight_calibration_profiles")
    assert recommendation["stage_id"] == "calibration_preflight"
    assert recommendation["endpoint"] == "/calibration/preflight"
    assert recommendation["expected_artifacts"] == [CALIBRATION_PREFLIGHT_REPORT]


def test_recommendations_suggest_calibration_preflight_for_default_profiles(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "default-calibration-run"
    config = create_run_config(run_root=run_root, sequence_id="sync_to_bop_dry_run")
    write_run_config(run_root, config)
    (run_root / CALIBRATION_PROFILES).write_text(
        '{"schema_version": "calibration_profiles.v1", "profiles": []}\n'
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["calibration_profile_source"] == CALIBRATION_PROFILES
    recommendation = recommendation_by_id(payload, "preflight_calibration_profiles")
    assert recommendation["stage_id"] == "calibration_preflight"


def test_recommendations_suggest_calibration_preflight_for_calibrated_sequence(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "calibrated-sequence-run"
    config = create_run_config(
        run_root=run_root,
        sequence_id="sync_to_bop_calibrated_dry_run",
    )
    write_run_config(run_root, config)

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["calibration_profile_source"] is None
    assert payload["facts"]["has_calibration_profiles_configured"] is False
    assert payload["facts"]["sequence_uses_calibration_preflight"] is True
    recommendation = recommendation_by_id(payload, "preflight_calibration_profiles")
    assert recommendation["stage_id"] == "calibration_preflight"
    assert recommendation["reason"] == (
        "Calibration profile preflight is expected but "
        f"{CALIBRATION_PREFLIGHT_REPORT} is missing."
    )


def test_recommendations_rerun_failed_calibration_preflight_before_calibrated_stages(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "failed-calibration-preflight-run"
    config = create_run_config(
        run_root=run_root,
        sequence_id="sync_to_bop_calibrated_dry_run",
    )
    write_run_config(run_root, config)
    sensor_root = run_root / PROCESSED_DIR / SYNCHRONIZED_DIR / "realsense_123"
    sensor_root.mkdir(parents=True)
    (run_root / CALIBRATION_PREFLIGHT_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "calibration_preflight.v1",
                "overall_status": "error",
                "checks": [{"name": "profiles", "status": "error"}],
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["calibration_preflight_expected"] is True
    assert payload["facts"]["has_calibration_preflight"] is True
    assert payload["facts"]["calibration_preflight_status"] == "error"
    assert payload["facts"]["calibration_preflight_ready_for_calibrated_stages"] is False
    assert payload["facts"]["calibration_preflight_blocker"] == (
        "failed_calibration_preflight"
    )
    assert payload["facts"]["calibration_preflight_blocks_calibrated_stages"] is True
    recommendation = recommendation_by_id(payload, "preflight_calibration_profiles")
    assert recommendation["reason"] == (
        f"{CALIBRATION_PREFLIGHT_REPORT} has overall_status=error."
    )
    recommendation_ids_ = recommendation_ids(payload)
    assert "prepare_blenderproc" not in recommendation_ids_
    assert "export_bop_dataset" not in recommendation_ids_


def test_recommendations_rebuild_empty_foundationpose_plan(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "empty-foundationpose-plan-run"
    write_prepared_blenderproc_fixture(run_root)
    (run_root / FOUNDATIONPOSE_PLAN).write_text(
        json.dumps(
            {
                "schema_version": "foundationpose_plan.v1",
                "dry_run": True,
                "input_folder": (
                    run_root / PROCESSED_DIR / SYNCHRONIZED_DIR
                ).as_posix(),
                "foundationpose_folder": "/opt/FoundationPose",
                "command": [
                    "uv",
                    "run",
                    "python",
                    "scripts/foundationpose_wrapper_multi.py",
                ],
                "jobs": [],
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_foundationpose_plan"] is True
    assert payload["facts"]["foundationpose_plan_ready_for_jobs"] is False
    assert payload["facts"]["foundationpose_plan_blocker"] == (
        "empty_foundationpose_plan"
    )
    assert payload["facts"]["foundationpose_plan_job_count"] == 0
    recommendation = recommendation_by_id(payload, "plan_foundationpose")
    assert recommendation["stage_id"] == "foundationpose"
    assert recommendation["reason"] == (
        f"{FOUNDATIONPOSE_PLAN} is not ready for FoundationPose jobs "
        "(empty_foundationpose_plan)."
    )


def test_recommendations_suggest_bop_evaluation_for_result_manifest(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "bop-run"
    result_file = write_bop_result_manifest_fixture(run_root)

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_bop_evaluation"] is False
    assert payload["facts"]["bop_evaluation_report_status"] is None
    assert payload["facts"]["bop_evaluation_report_ready"] is False
    assert payload["facts"]["bop_evaluation_report_blocker"] == (
        "missing_bop_evaluation_report"
    )
    assert payload["facts"]["has_bop_targets"] is True
    assert payload["facts"]["bop_targets_ready_for_evaluation"] is True
    assert payload["facts"]["bop_targets_blocker"] is None
    assert payload["facts"]["bop_targets_count"] == 1
    recommendation = recommendation_by_id(payload, "evaluate_bop_results")
    assert recommendation["stage_id"] == "bop_evaluation"
    assert recommendation["command"][:4] == [
        "uv",
        "run",
        "python",
        "scripts/run_bop_evaluation_stage.py",
    ]
    assert "--result-file" in recommendation["command"]
    assert result_file.as_posix() in recommendation["command"]
    assert "--dry-run" in recommendation["command"]


def test_recommendations_block_bop_evaluation_without_targets(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "bop-run-missing-targets"
    write_bop_result_manifest_fixture(run_root, write_targets=False)
    write_ready_bop_export_manifest(run_root)
    sensor_root = run_root / PROCESSED_DIR / SYNCHRONIZED_DIR / "realsense_123"
    sensor_root.mkdir(parents=True)

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_bop_targets"] is False
    assert payload["facts"]["bop_targets_ready_for_evaluation"] is False
    assert payload["facts"]["bop_targets_blocker"] == "missing_bop_targets"
    assert payload["facts"]["bop_targets_count"] == 0
    assert "evaluate_bop_results" not in recommendation_ids(payload)
    recommendation = recommendation_by_id(payload, "export_bop_dataset")
    assert recommendation["reason"] == (
        f"{BOP_DIR}/{BOP_TARGETS_BOP19} is missing and should be regenerated "
        "before BOP Toolkit evaluation."
    )


def test_recommendations_refresh_failed_bop_evaluation_report(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "failed-bop-evaluation-run"
    write_bop_result_manifest_fixture(run_root)
    (run_root / BOP_EVALUATION_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "bop_evaluation_report.v1",
                "status": "failed",
                "dry_run": False,
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_bop_evaluation"] is True
    assert payload["facts"]["bop_evaluation_report_status"] == "failed"
    assert payload["facts"]["bop_evaluation_report_ready"] is False
    assert payload["facts"]["bop_evaluation_report_blocker"] == (
        "failed_bop_evaluation_report"
    )
    recommendation = recommendation_by_id(payload, "evaluate_bop_results")
    assert recommendation["reason"] == (
        f"{BOP_EVALUATION_REPORT} has status=failed."
    )


def test_recommendations_accept_planned_bop_evaluation_report(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "planned-bop-evaluation-run"
    write_bop_result_manifest_fixture(run_root)
    (run_root / BOP_EVALUATION_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "bop_evaluation_report.v1",
                "status": "planned",
                "dry_run": True,
                "checks": ready_bop_evaluation_checks(),
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["bop_evaluation_report_status"] == "planned"
    assert payload["facts"]["bop_evaluation_report_ready"] is True
    assert payload["facts"]["bop_evaluation_report_blocker"] is None
    assert payload["facts"]["bop_evaluation_report_critical_missing_check_count"] == 0
    assert payload["facts"]["bop_evaluation_report_score_metric_count"] == 0
    assert payload["facts"]["has_bop_score_metrics"] is False
    assert "evaluate_bop_results" not in recommendation_ids(payload)


def test_recommendations_audit_foundationpose_runtime_gate_after_dry_run_eval(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "foundationpose-runtime-gate-run"
    output_folder = (
        run_root
        / PROCESSED_DIR
        / SYNCHRONIZED_DIR
        / "realsense_123"
        / "foundationpose_est5_track2_obj0_output"
    )
    ob_in_cam = output_folder / "ob_in_cam"
    ob_in_cam.mkdir(parents=True)
    (ob_in_cam / "000000.txt").write_text(
        "1 0 0 0\n0 1 0 0\n0 0 1 1\n0 0 0 1\n"
    )
    (run_root / FOUNDATIONPOSE_PLAN).write_text(
        json.dumps(
            {
                "schema_version": "foundationpose_plan.v1",
                "dry_run": True,
                "jobs": [
                    {
                        "sensor_name": "realsense_123",
                        "expected_output_folder": output_folder.as_posix(),
                    }
                ],
            }
        )
        + "\n"
    )
    result_file = run_root / RESULTS_DIR / BOP_DIR / "foundationpose_bop-test.csv"
    result_file.parent.mkdir(parents=True)
    result_file.write_text(
        "scene_id,im_id,obj_id,score,R,t,time\n"
        "1,0,1,1,1 0 0 0 1 0 0 0 1,0 0 1000,-1\n"
    )
    (run_root / BOP_RESULT_EXPORT_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": "bop_result_export_manifest.v1",
                "source_type": "foundationpose",
                "results": [
                    {
                        "filename": result_file.name,
                        "path": result_file.as_posix(),
                        "row_count": 1,
                    }
                ],
            }
        )
        + "\n"
    )
    (run_root / BOP_EVALUATION_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "bop_evaluation_report.v1",
                "status": "planned",
                "dry_run": True,
                "result": {
                    "filename": result_file.name,
                    "method": "foundationpose",
                },
                "checks": ready_bop_evaluation_checks(),
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["bop_evaluation_report_ready"] is True
    assert payload["facts"]["foundationpose_runtime_gate_expected"] is True
    assert payload["facts"]["foundationpose_runtime_gate_status"] == "blocked"
    assert "foundationpose_runtime" in payload["facts"][
        "foundationpose_runtime_gate_next_blockers"
    ]
    assert "foundationpose_bop_evaluation" in payload["facts"][
        "foundationpose_runtime_gate_next_blockers"
    ]
    recommendation = recommendation_by_id(
        payload,
        "audit_foundationpose_runtime_gate",
    )
    assert recommendation["stage_id"] == "rewrite_gate"
    assert recommendation["expected_artifacts"] == [REWRITE_GATE_REPORT]
    assert recommendation["command"] == [
        "uv",
        "run",
        "python",
        "scripts/run_rewrite_gate.py",
        run_root.as_posix(),
        "--gate",
        "rewrite_foundationpose_runtime.v1",
        "--write",
    ]
    assert "evaluate_bop_results" not in recommendation_ids(payload)


def test_recommendations_refresh_bop_evaluation_report_with_failed_prerequisites(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "planned-bop-evaluation-missing-models-run"
    write_bop_result_manifest_fixture(run_root)
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
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["bop_evaluation_report_status"] == "planned"
    assert payload["facts"]["bop_evaluation_report_ready"] is False
    assert payload["facts"]["bop_evaluation_report_blocker"] == (
        "failed_bop_evaluation_prerequisites"
    )
    assert payload["facts"]["bop_evaluation_report_critical_failed_check_count"] == 1
    assert payload["facts"]["bop_evaluation_report_critical_missing_check_count"] == 0
    assert payload["facts"]["has_bop_score_metrics"] is False
    recommendation = recommendation_by_id(payload, "evaluate_bop_results")
    assert recommendation["reason"] == (
        f"{BOP_EVALUATION_REPORT} has 1 failed and 0 missing BOP evaluation "
        "prerequisite check(s)."
    )


def test_recommendations_refresh_bop_evaluation_report_with_missing_prerequisites(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "planned-bop-evaluation-partial-checks-run"
    write_bop_result_manifest_fixture(run_root)
    (run_root / BOP_EVALUATION_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "bop_evaluation_report.v1",
                "status": "planned",
                "dry_run": True,
                "checks": [{"name": "result_file", "ok": True}],
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["bop_evaluation_report_ready"] is False
    assert payload["facts"]["bop_evaluation_report_blocker"] == (
        "failed_bop_evaluation_prerequisites"
    )
    assert payload["facts"]["bop_evaluation_report_critical_failed_check_count"] == 0
    assert payload["facts"]["bop_evaluation_report_critical_missing_check_count"] == 6
    recommendation = recommendation_by_id(payload, "evaluate_bop_results")
    assert recommendation["reason"] == (
        f"{BOP_EVALUATION_REPORT} has 0 failed and 6 missing BOP evaluation "
        "prerequisite check(s)."
    )


def test_recommendations_reexport_unusable_bop_result_manifest(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "unusable-bop-result-manifest-run"
    write_ready_bop_export_manifest(run_root)
    sensor_root = run_root / PROCESSED_DIR / SYNCHRONIZED_DIR / "realsense_123"
    sensor_root.mkdir(parents=True)
    (sensor_root / ARUCO_POSE_ESTIMATION).write_text("[]\n")
    write_ready_aruco_coverage_report(run_root)
    result_file = run_root / RESULTS_DIR / BOP_DIR / "aruco_bop-test.csv"
    result_file.parent.mkdir(parents=True)
    result_file.write_text(
        "scene_id,im_id,obj_id,score,R,t,time\n"
        "1,0,1,1,1 0 0 0 1 0 0 0 1,0 0 0,-1\n"
    )
    (run_root / BOP_RESULT_EXPORT_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": "bop_result_export_manifest.v1",
                "results": [
                    {
                        "filename": "missing_bop-test.csv",
                        "path": (result_file.parent / "missing_bop-test.csv").as_posix(),
                    }
                ],
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_bop_result_export"] is True
    assert payload["facts"]["bop_result_export_ready_for_evaluation"] is False
    assert payload["facts"]["bop_result_export_blocker"] == "missing_bop_result_csv"
    assert payload["facts"]["bop_result_export_result_count"] == 1
    assert payload["facts"]["bop_result_export_usable_result_count"] == 0
    recommendation = recommendation_by_id(payload, "export_aruco_bop_results")
    assert recommendation["reason"] == (
        f"{BOP_RESULT_EXPORT_MANIFEST} has no usable result CSV."
    )
    assert "evaluate_bop_results" not in recommendation_ids(payload)


def test_recommendations_rebuild_empty_bop_export_manifest(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "empty-bop-export-run"
    sensor_root = run_root / PROCESSED_DIR / SYNCHRONIZED_DIR / "realsense_123"
    sensor_root.mkdir(parents=True)
    (sensor_root / ARUCO_POSE_ESTIMATION).write_text("[]\n")
    manifest = run_root / BOP_DIR / BOP_EXPORT_MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"schema_version": "bop_export_manifest.v1", "exports": []})
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_bop_export"] is True
    assert payload["facts"]["bop_export_ready_for_results"] is False
    assert payload["facts"]["bop_export_blocker"] == "empty_bop_export_manifest"
    assert payload["facts"]["bop_export_count"] == 0
    recommendation = recommendation_by_id(payload, "export_bop_dataset")
    assert recommendation["reason"] == (
        f"{BOP_DIR}/{BOP_EXPORT_MANIFEST} has no exported scenes."
    )
    assert "export_aruco_bop_results" not in recommendation_ids(payload)


def test_recommendations_rebuild_empty_metric_report(tmp_path: Path) -> None:
    run_root = tmp_path / "empty-metric-report-run"
    run_root.mkdir()
    (run_root / ACCURACY_HRC_HUB).write_text("{}\n")
    report = run_root / RESULTS_DIR / METRICS_DIR / METRIC_REPORT_JSON
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "schema_version": "metric_report.v1",
                "dashboard": {},
                "rows": [],
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_legacy_metrics"] is True
    assert payload["facts"]["has_metric_sources"] is True
    assert payload["facts"]["has_metric_report"] is True
    assert payload["facts"]["metric_report_ready_for_dashboard"] is False
    assert payload["facts"]["metric_report_blocker"] == "empty_metric_report"
    assert payload["facts"]["metric_report_row_count"] == 0
    recommendation = recommendation_by_id(payload, "export_metric_reports")
    assert recommendation["reason"] == (
        f"{RESULTS_DIR}/{METRICS_DIR}/{METRIC_REPORT_JSON} has no dashboard rows."
    )


def test_recommendations_export_metric_reports_after_bop_evaluation(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "bop-evaluation-metric-source-run"
    run_root.mkdir()
    (run_root / BOP_EVALUATION_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "bop_evaluation_report.v1",
                "status": "planned",
                "checks": ready_bop_evaluation_checks(),
                "score_summary": {
                    "score_file_count": 1,
                    "metrics": {"bop19_average_recall": 0.75},
                    "files": [],
                },
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_legacy_metrics"] is False
    assert payload["facts"]["bop_evaluation_report_ready"] is True
    assert payload["facts"]["bop_evaluation_report_score_metric_count"] == 1
    assert payload["facts"]["has_bop_score_metrics"] is True
    assert payload["facts"]["has_metric_sources"] is True
    assert payload["facts"]["metric_report_blocker"] == "missing_metric_report"
    recommendation = recommendation_by_id(payload, "export_metric_reports")
    assert recommendation["stage_id"] == "metric_report_export"


def test_recommendations_do_not_export_metric_reports_for_bop_plan_without_scores(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "bop-evaluation-no-score-run"
    run_root.mkdir()
    (run_root / BOP_EVALUATION_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "bop_evaluation_report.v1",
                "status": "planned",
                "checks": ready_bop_evaluation_checks(),
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_legacy_metrics"] is False
    assert payload["facts"]["bop_evaluation_report_ready"] is True
    assert payload["facts"]["bop_evaluation_report_score_metric_count"] == 0
    assert payload["facts"]["has_bop_score_metrics"] is False
    assert payload["facts"]["has_metric_sources"] is False
    assert "export_metric_reports" not in recommendation_ids(payload)


def test_recommendations_refresh_failed_sync_quality_before_aruco(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "failed-sync-quality-run"
    sensor_root = run_root / PROCESSED_DIR / SYNCHRONIZED_DIR / "realsense_123"
    sensor_root.mkdir(parents=True)
    (run_root / SYNC_QUALITY_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "sync_quality_report.v1",
                "overall_status": "error",
                "checks": [{"name": "match_ratio", "status": "error"}],
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_sync_quality"] is True
    assert payload["facts"]["sync_quality_status"] == "error"
    assert payload["facts"]["sync_quality_ready_for_downstream"] is False
    assert payload["facts"]["sync_quality_blocker"] == "failed_sync_quality_report"
    recommendation = recommendation_by_id(payload, "check_sync_quality")
    assert recommendation["reason"] == (
        f"{SYNC_QUALITY_REPORT} has overall_status=error."
    )
    assert "run_aruco" not in recommendation_ids(payload)


def test_recommendations_allow_aruco_after_warning_sync_quality(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "warning-sync-quality-run"
    sensor_root = run_root / PROCESSED_DIR / SYNCHRONIZED_DIR / "realsense_123"
    sensor_root.mkdir(parents=True)
    (run_root / SYNC_QUALITY_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "sync_quality_report.v1",
                "overall_status": "warning",
                "checks": [{"name": "match_ratio", "status": "warning"}],
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["sync_quality_status"] == "warning"
    assert payload["facts"]["sync_quality_ready_for_downstream"] is True
    assert payload["facts"]["sync_quality_blocker"] is None
    assert "check_sync_quality" not in recommendation_ids(payload)
    recommendation = recommendation_by_id(payload, "run_aruco")
    assert recommendation["stage_id"] == "aruco"


def test_recommendations_refresh_failed_aruco_coverage_before_observations(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "failed-aruco-coverage-run"
    sensor_root = run_root / PROCESSED_DIR / SYNCHRONIZED_DIR / "realsense_123"
    sensor_root.mkdir(parents=True)
    (sensor_root / ARUCO_POSE_ESTIMATION).write_text("{}\n")
    (run_root / ARUCO_COVERAGE_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "aruco_coverage_report.v1",
                "overall_status": "error",
                "checks": [{"name": "aruco_outputs_present", "status": "error"}],
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_aruco_outputs"] is True
    assert payload["facts"]["has_aruco_coverage"] is True
    assert payload["facts"]["aruco_coverage_status"] == "error"
    assert payload["facts"]["aruco_coverage_ready_for_downstream"] is False
    assert payload["facts"]["aruco_coverage_blocker"] == (
        "failed_aruco_coverage_report"
    )
    recommendation = recommendation_by_id(payload, "check_aruco_coverage")
    assert recommendation["reason"] == (
        f"{ARUCO_COVERAGE_REPORT} has overall_status=error."
    )
    assert "build_calibration_observations" not in recommendation_ids(payload)


def test_recommendations_block_aruco_bop_result_export_until_coverage_ready(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "failed-aruco-result-export-coverage-run"
    write_ready_bop_export_manifest(run_root)
    sensor_root = run_root / PROCESSED_DIR / SYNCHRONIZED_DIR / "realsense_123"
    sensor_root.mkdir(parents=True)
    (sensor_root / ARUCO_POSE_ESTIMATION).write_text("{}\n")
    (run_root / ARUCO_COVERAGE_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "aruco_coverage_report.v1",
                "overall_status": "error",
                "checks": [{"name": "aruco_outputs_present", "status": "error"}],
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["bop_export_ready_for_results"] is True
    assert payload["facts"]["has_aruco_outputs"] is True
    assert payload["facts"]["aruco_coverage_ready_for_downstream"] is False
    assert "export_aruco_bop_results" not in recommendation_ids(payload)
    recommendation = recommendation_by_id(payload, "check_aruco_coverage")
    assert recommendation["stage_id"] == "aruco_coverage"


def test_recommendations_block_bop_result_export_without_object_metadata(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "result-export-missing-object-metadata"
    write_ready_bop_export_manifest(run_root, write_object_models=False)
    target_file = run_root / BOP_DIR / BOP_TARGETS_BOP19
    target_file.write_text(
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
        + "\n"
    )
    sensor_root = run_root / PROCESSED_DIR / SYNCHRONIZED_DIR / "realsense_123"
    megapose_output = sensor_root / "megapose_rgbd_obj0_output"
    megapose_output.mkdir(parents=True)
    (megapose_output / "megapose_poses.json").write_text("[]\n")

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["bop_export_ready_for_results"] is True
    assert payload["facts"]["bop_object_models_ready_for_result_export"] is False
    assert payload["facts"]["bop_object_models_blocker"] == (
        "missing_bop_object_models"
    )
    assert payload["facts"]["bop_object_model_count"] == 0
    assert "export_megapose_bop_results" not in recommendation_ids(payload)
    recommendation = recommendation_by_id(payload, "export_bop_dataset")
    assert recommendation["reason"] == (
        f"{BOP_DIR}/{BOP_EXPORT_MANIFEST} has no BOP object model metadata; "
        "rebuild the BOP export with model export enabled."
    )


def test_recommendations_suggest_megapose_and_sam6d_result_exports(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "estimator-run"
    sensor_root = run_root / PROCESSED_DIR / SYNCHRONIZED_DIR / "realsense_123"
    megapose_output = sensor_root / "megapose_rgbd_obj0_output"
    sam6d_output = sensor_root / "sam6d_pem_obj0_output"
    megapose_output.mkdir(parents=True)
    (megapose_output / "megapose_poses.json").write_text("[]\n")
    (sam6d_output / "detections_pem").mkdir(parents=True)
    write_ready_bop_export_manifest(run_root)

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_megapose_outputs"] is True
    assert payload["facts"]["has_sam6d_outputs"] is True
    assert payload["facts"]["bop_object_models_ready_for_result_export"] is True
    assert payload["facts"]["bop_object_models_blocker"] is None
    assert payload["facts"]["bop_object_model_count"] == 1
    megapose = recommendation_by_id(payload, "export_megapose_bop_results")
    sam6d = recommendation_by_id(payload, "export_sam6d_bop_results")
    assert megapose["stage_id"] == "bop_result_export"
    assert sam6d["stage_id"] == "bop_result_export"
    assert megapose["command"][megapose["command"].index("--source") + 1] == "megapose"
    assert sam6d["command"][sam6d["command"].index("--source") + 1] == "sam6d"
    assert megapose["expected_artifacts"] == [BOP_RESULT_EXPORT_MANIFEST]
    assert sam6d["expected_artifacts"] == [BOP_RESULT_EXPORT_MANIFEST]


def test_recommendations_ignore_incomplete_estimator_output_folders(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "incomplete-estimator-run"
    sensor_root = run_root / PROCESSED_DIR / SYNCHRONIZED_DIR / "realsense_123"
    (sensor_root / "foundationpose_est5_track2_obj0_output").mkdir(parents=True)
    (sensor_root / "megapose_rgbd_obj0_output").mkdir()
    (sensor_root / "sam6d_pem_obj0_output").mkdir()
    write_ready_bop_export_manifest(run_root)

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_foundationpose_outputs"] is False
    assert payload["facts"]["has_megapose_outputs"] is False
    assert payload["facts"]["has_sam6d_outputs"] is False
    recommendation_ids = {
        recommendation["id"] for recommendation in payload["recommendations"]
    }
    assert "export_foundationpose_bop_results" not in recommendation_ids
    assert "export_megapose_bop_results" not in recommendation_ids
    assert "export_sam6d_bop_results" not in recommendation_ids


def test_recommendations_suggest_calibration_solver_after_observations(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "calibration-run"
    run_root.mkdir()
    (run_root / CALIBRATION_OBSERVATIONS).write_text(
        '{"schema_version": "calibration_observations.v1", "overall_status": "ok"}\n'
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_calibration_observations"] is True
    assert payload["facts"]["calibration_observations_status"] == "ok"
    assert payload["facts"]["calibration_observations_ready_for_solver"] is True
    assert payload["facts"]["calibration_observations_blocker"] is None
    assert payload["facts"]["has_calibration_solver"] is False
    recommendation = recommendation_by_id(payload, "solve_calibration_profiles")
    assert recommendation["stage_id"] == "calibration_solver"
    assert recommendation["endpoint"] == "/calibration/solver"
    assert recommendation["expected_artifacts"] == [CALIBRATION_SOLVER_REPORT]
    assert recommendation["command"][:4] == [
        "uv",
        "run",
        "python",
        "scripts/run_calibration_solver.py",
    ]


def test_recommendations_rebuild_failed_calibration_observations(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "failed-calibration-observations-run"
    pose_file = (
        run_root
        / PROCESSED_DIR
        / SYNCHRONIZED_DIR
        / "realsense_123"
        / CHARUCO_POSE_ESTIMATION
    )
    pose_file.parent.mkdir(parents=True)
    pose_file.write_text("[]\n")
    (run_root / CALIBRATION_OBSERVATIONS).write_text(
        json.dumps(
            {
                "schema_version": "calibration_observations.v1",
                "overall_status": "error",
                "checks": [{"name": "observations", "status": "error"}],
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_calibration_observations"] is True
    assert payload["facts"]["calibration_observations_status"] == "error"
    assert payload["facts"]["calibration_observations_ready_for_solver"] is False
    assert payload["facts"]["calibration_observations_blocker"] == (
        "failed_calibration_observations"
    )
    recommendation = recommendation_by_id(payload, "build_calibration_observations")
    assert recommendation["reason"] == (
        f"{CALIBRATION_OBSERVATIONS} has overall_status=error."
    )
    assert "solve_calibration_profiles" not in recommendation_ids(payload)


def test_recommendations_rerun_failed_calibration_solver(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "failed-calibration-solver-run"
    run_root.mkdir()
    (run_root / CALIBRATION_OBSERVATIONS).write_text(
        '{"schema_version": "calibration_observations.v1", "overall_status": "ok"}\n'
    )
    (run_root / CALIBRATION_SOLVER_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "calibration_solver.v1",
                "overall_status": "error",
                "checks": [{"name": "solver", "status": "error"}],
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_calibration_solver"] is True
    assert payload["facts"]["calibration_solver_status"] == "error"
    assert payload["facts"]["calibration_solver_ready_for_candidates"] is False
    assert payload["facts"]["calibration_solver_blocker"] == (
        "failed_calibration_solver"
    )
    recommendation = recommendation_by_id(payload, "solve_calibration_profiles")
    assert recommendation["reason"] == (
        f"{CALIBRATION_SOLVER_REPORT} has overall_status=error."
    )
    assert "build_calibration_candidates" not in recommendation_ids(payload)


def test_recommendations_allow_candidates_after_warning_calibration_solver(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "warning-calibration-solver-run"
    run_root.mkdir()
    (run_root / CALIBRATION_OBSERVATIONS).write_text(
        '{"schema_version": "calibration_observations.v1", "overall_status": "ok"}\n'
    )
    (run_root / CALIBRATION_SOLVER_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "calibration_solver.v1",
                "overall_status": "warning",
                "checks": [{"name": "solver", "status": "warning"}],
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["calibration_solver_status"] == "warning"
    assert payload["facts"]["calibration_solver_ready_for_candidates"] is True
    assert payload["facts"]["calibration_solver_blocker"] is None
    assert "solve_calibration_profiles" not in recommendation_ids(payload)
    recommendation = recommendation_by_id(payload, "build_calibration_candidates")
    assert recommendation["stage_id"] == "calibration_candidates"


def test_recommendations_rebuild_failed_calibration_candidates(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "failed-calibration-candidates-run"
    run_root.mkdir()
    (run_root / CALIBRATION_OBSERVATIONS).write_text(
        '{"schema_version": "calibration_observations.v1", "overall_status": "ok"}\n'
    )
    (run_root / CALIBRATION_SOLVER_REPORT).write_text(
        '{"schema_version": "calibration_solver.v1", "overall_status": "ok"}\n'
    )
    (run_root / CALIBRATION_CANDIDATES).write_text(
        json.dumps(
            {
                "schema_version": "calibration_candidates.v1",
                "overall_status": "error",
                "checks": [{"name": "candidates", "status": "error"}],
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_calibration_candidates"] is True
    assert payload["facts"]["calibration_candidates_status"] == "error"
    assert payload["facts"]["calibration_candidates_ready_for_validation"] is False
    assert payload["facts"]["calibration_candidates_blocker"] == (
        "failed_calibration_candidates"
    )
    recommendation = recommendation_by_id(payload, "build_calibration_candidates")
    assert recommendation["reason"] == (
        f"{CALIBRATION_CANDIDATES} has overall_status=error."
    )
    assert "validate_calibration_candidates" not in recommendation_ids(payload)


def test_recommendations_allow_validation_after_warning_calibration_candidates(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "warning-calibration-candidates-run"
    run_root.mkdir()
    (run_root / CALIBRATION_OBSERVATIONS).write_text(
        '{"schema_version": "calibration_observations.v1", "overall_status": "ok"}\n'
    )
    (run_root / CALIBRATION_SOLVER_REPORT).write_text(
        '{"schema_version": "calibration_solver.v1", "overall_status": "ok"}\n'
    )
    (run_root / CALIBRATION_CANDIDATES).write_text(
        json.dumps(
            {
                "schema_version": "calibration_candidates.v1",
                "overall_status": "warning",
                "checks": [{"name": "candidates", "status": "warning"}],
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["calibration_candidates_status"] == "warning"
    assert payload["facts"]["calibration_candidates_ready_for_validation"] is True
    assert payload["facts"]["calibration_candidates_blocker"] is None
    assert "build_calibration_candidates" not in recommendation_ids(payload)
    recommendation = recommendation_by_id(payload, "validate_calibration_candidates")
    assert recommendation["stage_id"] == "calibration_validation"


def test_recommendations_rerun_failed_calibration_validation(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "failed-calibration-validation-run"
    run_root.mkdir()
    (run_root / CALIBRATION_OBSERVATIONS).write_text(
        '{"schema_version": "calibration_observations.v1", "overall_status": "ok"}\n'
    )
    (run_root / CALIBRATION_SOLVER_REPORT).write_text(
        '{"schema_version": "calibration_solver.v1", "overall_status": "ok"}\n'
    )
    (run_root / CALIBRATION_CANDIDATES).write_text(
        '{"schema_version": "calibration_candidates.v1", "overall_status": "ok"}\n'
    )
    (run_root / CALIBRATION_VALIDATION_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "calibration_validation.v1",
                "overall_status": "error",
                "checks": [{"name": "inliers", "status": "error"}],
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_calibration_validation"] is True
    assert payload["facts"]["calibration_validation_status"] == "error"
    assert payload["facts"]["calibration_validation_ready_for_profiles"] is False
    assert payload["facts"]["calibration_validation_blocker"] == (
        "failed_calibration_validation"
    )
    recommendation = recommendation_by_id(payload, "validate_calibration_candidates")
    assert recommendation["reason"] == (
        f"{CALIBRATION_VALIDATION_REPORT} has overall_status=error."
    )


def test_recommendations_audit_calibration_gate_after_unpromoted_validation(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "unpromoted-calibration-validation-run"
    (run_root / CALIBRATION_VALIDATION_REPORT).parent.mkdir(parents=True)
    (run_root / CALIBRATION_VALIDATION_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "calibration_validation.v1",
                "overall_status": "ok",
                "profile_count": 1,
                "promotable_profile_count": 1,
                "promotion": {
                    "requested": False,
                    "promoted": False,
                    "path": None,
                    "profile_count": 0,
                },
                "profiles": [
                    {
                        "profile_id": "realsense_123_static_candidate",
                        "sensor_id": "123",
                        "validation_status": "ok",
                        "promotable": True,
                    }
                ],
            }
        )
        + "\n"
    )

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["calibration_validation_ready_for_profiles"] is True
    assert payload["facts"]["calibration_validation_gate_expected"] is True
    assert payload["facts"]["calibration_validation_gate_status"] == "blocked"
    assert payload["facts"]["calibration_validation_gate_next_blockers"] == [
        "calibration_validation",
        "calibration_profiles",
    ]
    recommendation = recommendation_by_id(
        payload,
        "audit_calibration_validation_gate",
    )
    assert recommendation["stage_id"] == "rewrite_gate"
    assert recommendation["expected_artifacts"] == [REWRITE_GATE_REPORT]
    assert recommendation["command"] == [
        "uv",
        "run",
        "python",
        "scripts/run_rewrite_gate.py",
        run_root.as_posix(),
        "--gate",
        "rewrite_calibration_validation.v1",
        "--write",
    ]
    assert "validate_calibration_candidates" not in recommendation_ids(payload)


def test_recommendations_suggest_observations_for_charuco_pose_outputs(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "charuco-run"
    pose_file = (
        run_root
        / PROCESSED_DIR
        / SYNCHRONIZED_DIR
        / "realsense_123"
        / CHARUCO_POSE_ESTIMATION
    )
    pose_file.parent.mkdir(parents=True)
    pose_file.write_text("[]\n")

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["has_aruco_outputs"] is False
    assert payload["facts"]["has_calibration_target_pose_outputs"] is True
    assert payload["facts"]["has_non_aruco_calibration_target_pose_outputs"] is True
    recommendation = recommendation_by_id(payload, "build_calibration_observations")
    assert recommendation["stage_id"] == "calibration_observations"
    assert recommendation["endpoint"] == "/calibration/observations"
    assert recommendation["expected_artifacts"] == [CALIBRATION_OBSERVATIONS]
