from __future__ import annotations

import json
from pathlib import Path

from posetestbot.io.artifacts import (
    ARUCO_POSE_ESTIMATION,
    BOP_DIR,
    BOP_EXPORT_MANIFEST,
    BOP_TARGETS_BOP19,
    CALIBRATION_OBSERVATIONS,
    PROCESSED_DIR,
    RGB_DIR,
    DEPTH_DIR,
    RUN_CONFIG,
    RUN_PREFLIGHT_REPORT,
    SYNC_QUALITY_REPORT,
    SYNCHRONIZED_DIR,
)
from posetestbot.pipeline.recommendations import build_pipeline_recommendations
from posetestbot.pipeline.run_config import (
    SensorRunConfig,
    create_run_config,
    write_run_config,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def recommendation_ids(payload: dict) -> set[str]:
    return {item["id"] for item in payload["recommendations"]}


def recommendation_by_id(payload: dict, recommendation_id: str) -> dict:
    return next(
        item for item in payload["recommendations"] if item["id"] == recommendation_id
    )


def write_ready_run_config(run_root: Path) -> dict:
    config = create_run_config(
        run_root=run_root,
        sensors=(SensorRunConfig("realsense_d435", "123", "Enabled"),),
    )
    write_run_config(run_root, config)
    return config.to_dict()


def write_preflight(run_root: Path, config: dict) -> None:
    write_json(
        run_root / RUN_PREFLIGHT_REPORT,
        {
            "schema_version": "run_preflight.v1",
            "overall_status": "warning",
            "config": config,
        },
    )


def make_synchronized_sensor(run_root: Path) -> Path:
    sensor = run_root / PROCESSED_DIR / SYNCHRONIZED_DIR / "realsense_123"
    (sensor / RGB_DIR).mkdir(parents=True)
    (sensor / DEPTH_DIR).mkdir()
    return sensor


def write_bop_export(run_root: Path) -> None:
    scene = run_root / BOP_DIR / "realsense_123" / "test" / "000001"
    (scene / RGB_DIR).mkdir(parents=True)
    (scene / DEPTH_DIR).mkdir()
    write_json(
        scene / "scene_camera.json", {"0": {"cam_K": [1, 0, 0, 0, 1, 0, 0, 0, 1]}}
    )
    write_json(scene / "scene_gt.json", {"0": []})
    write_json(
        run_root / BOP_DIR / BOP_EXPORT_MANIFEST,
        {
            "schema_version": "bop_export_manifest.v1",
            "exports": [
                {
                    "sensor_name": "realsense_123",
                    "scene_id": 1,
                    "split": "test",
                    "scene_folder": scene.relative_to(run_root).as_posix(),
                }
            ],
            "object_models": [{"object_name": "cube", "obj_id": 1}],
        },
    )
    write_json(
        run_root / BOP_DIR / BOP_TARGETS_BOP19,
        [{"scene_id": 1, "im_id": 0, "obj_id": 1, "inst_count": 1}],
    )


def test_recommendations_create_run_config_when_missing(tmp_path: Path) -> None:
    payload = build_pipeline_recommendations(tmp_path / "new-run")

    assert payload["facts"]["has_run_config"] is False
    recommendation = recommendation_by_id(payload, "create_run_config")
    assert recommendation["command"][:4] == [
        "uv",
        "run",
        "python",
        "scripts/create_run_config.py",
    ]
    assert recommendation["expected_artifacts"] == [RUN_CONFIG]


def test_recommendations_write_preflight_for_valid_run_config(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_ready_run_config(run_root)

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["run_config_ready_for_pipeline"] is True
    recommendation = recommendation_by_id(payload, "write_run_preflight")
    assert recommendation["stage_id"] == "run_preflight"
    assert recommendation["expected_artifacts"] == [RUN_PREFLIGHT_REPORT]


def test_recommendations_sync_quality_and_bop_export_path(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    config = write_ready_run_config(run_root)
    write_preflight(run_root, config)
    make_synchronized_sensor(run_root)

    payload = build_pipeline_recommendations(run_root)
    ids = recommendation_ids(payload)

    assert "write_sync_quality" in ids
    assert "export_bop_dataset" not in ids

    write_json(
        run_root / SYNC_QUALITY_REPORT,
        {"schema_version": "sync_quality_report.v1", "overall_status": "ok"},
    )
    payload = build_pipeline_recommendations(run_root)
    ids = recommendation_ids(payload)
    assert "prepare_blenderproc" not in ids
    assert "plan_blenderproc_render" not in ids
    assert "export_bop_dataset" in ids
    export = recommendation_by_id(payload, "export_bop_dataset")
    assert export["stage_id"] == "bop_export"
    annotation_flag = export["command"].index("--annotation-source")
    assert export["command"][annotation_flag + 1] == "none"
    assert "--overwrite" in export["command"]


def test_recommendations_build_calibration_observations_from_target_outputs(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    config = write_ready_run_config(run_root)
    write_preflight(run_root, config)
    sensor = make_synchronized_sensor(run_root)
    write_json(sensor / ARUCO_POSE_ESTIMATION, [{"frame": 0}])

    payload = build_pipeline_recommendations(run_root)

    recommendation = recommendation_by_id(payload, "build_calibration_observations")
    assert recommendation["stage_id"] == "calibration_observations"
    assert recommendation["expected_artifacts"] == [CALIBRATION_OBSERVATIONS]


def test_recommendations_ignore_disabled_stale_sensor_artifacts(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    config = create_run_config(
        run_root=run_root,
        sensors=(
            SensorRunConfig("realsense_d435", "123", "Enabled"),
            SensorRunConfig("realsense_d435", "999", "Disabled", enabled=False),
        ),
    )
    write_run_config(run_root, config)
    write_preflight(run_root, config.to_dict())
    enabled = make_synchronized_sensor(run_root)
    disabled = enabled.parent / "realsense_999"
    (disabled / RGB_DIR).mkdir(parents=True)
    (disabled / DEPTH_DIR).mkdir()
    write_json(disabled / ARUCO_POSE_ESTIMATION, [{"frame": 0}])
    write_json(disabled / "blenderproc" / "objects.json", {"instances": []})

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["synchronized_sensor_count"] == 1
    assert payload["facts"]["has_aruco_outputs"] is False
    assert payload["facts"]["has_target_pose_outputs"] is False
    assert payload["facts"]["has_blenderproc_prepared"] is False
    assert "run_aruco" in recommendation_ids(payload)


def test_recommendations_report_ready_bop_dataset_without_downstream_suggestions(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    config = write_ready_run_config(run_root)
    write_preflight(run_root, config)
    make_synchronized_sensor(run_root)
    write_json(
        run_root / SYNC_QUALITY_REPORT,
        {"schema_version": "sync_quality_report.v1", "overall_status": "ok"},
    )
    write_bop_export(run_root)

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["bop_export_ready_for_dataset_use"] is True
    assert payload["facts"]["bop_export_count"] == 1
    ids = recommendation_ids(payload)
    assert "evaluate_bop_results" not in ids
    assert "export_metric_reports" not in ids
    assert "plan_foundationpose" not in ids


def test_recommendations_accept_explicit_objectless_bop_export(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    config = write_ready_run_config(run_root)
    write_preflight(run_root, config)
    make_synchronized_sensor(run_root)
    write_json(
        run_root / SYNC_QUALITY_REPORT,
        {"schema_version": "sync_quality_report.v1", "overall_status": "ok"},
    )
    write_json(
        run_root / BOP_DIR / BOP_EXPORT_MANIFEST,
        {
            "schema_version": "bop_export_manifest.v2",
            "objectless": True,
            "selected_objects": [],
            "exports": [{"sensor_name": "realsense_123", "scene_id": 1}],
            "object_models": [],
        },
    )
    write_json(run_root / BOP_DIR / BOP_TARGETS_BOP19, [])

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["bop_export_ready_for_dataset_use"] is True
    assert "export_bop_dataset" not in recommendation_ids(payload)


def test_recommendations_accept_annotation_free_object_bop_export(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    config = write_ready_run_config(run_root)
    write_preflight(run_root, config)
    make_synchronized_sensor(run_root)
    write_json(
        run_root / SYNC_QUALITY_REPORT,
        {"schema_version": "sync_quality_report.v1", "overall_status": "ok"},
    )
    write_bop_export(run_root)
    manifest_path = run_root / BOP_DIR / BOP_EXPORT_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "schema_version": "bop_export_manifest.v5",
            "annotation_source": "none",
            "targets_path": BOP_TARGETS_BOP19,
            "capabilities": {
                "pose_estimation_input": True,
                "bop19_evaluation": False,
            },
        }
    )
    write_json(manifest_path, manifest)

    payload = build_pipeline_recommendations(run_root)

    assert payload["facts"]["bop_export_ready_for_dataset_use"] is True
    assert payload["facts"]["bop_annotation_source"] == "none"
    assert payload["facts"]["bop_model_count"] == 1
    assert payload["facts"]["bop_target_count"] == 1
    assert payload["facts"]["bop_ready_for_pose_estimation"] is True
    assert payload["facts"]["bop_ready_for_bop19_evaluation"] is False
    assert "export_bop_dataset" not in recommendation_ids(payload)
