from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from posetestbot.calibration.intrinsics import factory_intrinsic_profile
from posetestbot.calibration.rectification import rectify_run
from posetestbot.io.artifacts import (
    BOP_DIR,
    BOP_EXPORT_MANIFEST,
    BOP_FRAME_MAP_JSON,
    BOP_FRAME_SETS,
    BOP_TARGETS_BOP19,
    CAM_K,
    CAMERA_DATA_JSON,
    CALIBRATION_PROFILES,
    CALIBRATION_VALIDATION_REPORT,
    CAPTURE_EXECUTION_REPORT,
    DEPTH_DIR,
    DEPTH_SCALE,
    FRAME_METADATA_JSONL,
    HARDWARE_STATUS_REPORT,
    MATCH_ROBOT_EE_POSES,
    MULTIVIEW_FRAME_GROUPS,
    RGB_DIR,
    RUN_PREFLIGHT_REPORT,
)
from posetestbot.pipeline.rewrite_gate import (
    BOP_EXPORT_READINESS_GATE_ID,
    CALIBRATION_VALIDATION_GATE_ID,
    FULL_CAPTURE_GATE_ID,
    GATE_IDS,
    build_bop_export_readiness_gate_report,
    build_calibration_validation_gate_report,
    build_rewrite_status_report,
    build_gate_report,
)
from posetestbot.pipeline.run_config import (
    SensorRunConfig,
    create_run_config,
    write_run_config,
)
from posetestbot.sensors.hardware_sync_qualification import (
    record_hardware_sync_qualification,
    validate_hardware_sync_qualification,
)
from posetestbot.sync.hardware import (
    build_hardware_sync_frame_groups,
    write_hardware_sync_frame_groups,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def populate_bop_export(
    run_root: Path,
    *,
    with_targets: bool = True,
) -> None:
    scene = run_root / BOP_DIR / "test" / "000001"
    (scene / RGB_DIR).mkdir(parents=True)
    (scene / DEPTH_DIR).mkdir()
    (scene / RGB_DIR / "000000.png").write_bytes(b"rgb")
    (scene / DEPTH_DIR / "000000.png").write_bytes(b"depth")
    write_json(
        scene / "scene_camera.json", {"0": {"cam_K": [1, 0, 0, 0, 1, 0, 0, 0, 1]}}
    )
    write_json(
        scene / "scene_gt.json",
        {
            "0": [
                {
                    "obj_id": 1,
                    "cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    "cam_t_m2c": [0, 0, 100],
                }
            ]
        },
    )
    write_json(scene / "scene_gt_info.json", {"0": [{}]})
    write_json(
        run_root / BOP_DIR / "dataset_info.json",
        {
            "schema_version": "posetestbot_bop_dataset_info.v1",
            "name": "fixture",
            "bop_format": "scenewise",
            "scene_count": 1,
            "sensors": ["realsense_123"],
        },
    )
    write_json(
        run_root / BOP_DIR / "posetestbot_bop_frame_map.json",
        {
            "schema_version": "posetestbot_bop_frame_map.v2",
            "scenes": {
                "1": {
                    "sensor_name": "realsense_123",
                    "frames": {"0": {"source_frame_id": "000000.png"}},
                }
            },
        },
    )
    write_json(
        run_root / BOP_DIR / BOP_EXPORT_MANIFEST,
        {
            "schema_version": "bop_export_manifest.v2",
            "format": "bop-scenewise",
            "validation": {"status": "ok"},
            "exports": [
                {
                    "sensor_name": "realsense_123",
                    "scene_id": 1,
                    "split": "test",
                    "scene_folder": "test/000001",
                    "calibration_profile_id": "profile-1",
                }
            ],
            "calibration_profiles": [{"profile_id": "profile-1", "status": "valid"}],
            "object_models": [
                {
                    "object_name": "cube",
                    "obj_id": 1,
                    "bop_path": "models/obj_000001.ply",
                }
            ],
        },
    )
    write_json(
        run_root / BOP_DIR / "models" / "models_info.json",
        {"1": {"diameter": 1}},
    )
    (run_root / BOP_DIR / "models" / "obj_000001.ply").write_text("ply\n")
    if with_targets:
        write_json(
            run_root / BOP_DIR / BOP_TARGETS_BOP19,
            [{"scene_id": 1, "im_id": 0, "obj_id": 1, "inst_count": 1}],
        )


def populate_hardware_sync_bop_export(
    tmp_path: Path,
    *,
    rectified: bool = False,
) -> Path:
    """Build the durable hardware groups and export them through production code."""

    run_root = tmp_path / "hardware-gate-run"
    config = create_run_config(
        run_root=run_root,
        sensors=(
            SensorRunConfig(
                "realsense_d435",
                "master",
                "Static master",
                mounting_mode="static",
            ),
            SensorRunConfig(
                "realsense_d435",
                "hand",
                "Robot-mounted subordinate",
                mounting_mode="eye_in_hand",
            ),
        ),
        synchronization={
            "schema_version": "capture_synchronization.v1",
            "mode": "hardware_trigger",
            "implementation": "realsense_inter_cam_sync",
            "scope": "depth_exposure",
            "group_id": "mixed-rig",
            "master_sensor_key": "realsense_d435:master",
            "max_depth_timestamp_skew_ms": 2.0,
        },
    )
    write_run_config(run_root, config)
    qualification_evidence = tmp_path / "hardware-gate-pulse.csv"
    qualification_evidence.write_text("time_ns,master,hand\n0,1,1\n")
    record_hardware_sync_qualification(
        run_root,
        operator="gate-test@example.test",
        method="pulsed_light",
        observed_max_depth_timestamp_skew_ms=0.1,
        evidence_paths=[qualification_evidence],
        confirm_passed=True,
    )
    qualification = validate_hardware_sync_qualification(
        run_root,
        run_config=config.to_dict(),
    )
    execution_binding = {
        "configuration_sha256": qualification["configuration_sha256"],
        "qualification_artifact_sha256": qualification["artifact_sha256"],
        "revalidated_immediately_before_receiver_spawn": True,
    }
    write_json(
        run_root / CAPTURE_EXECUTION_REPORT,
        {
            "schema_version": "capture_execution_report.v1",
            "run_root": run_root.as_posix(),
            "status": "succeeded",
            "mode": "full",
            "allow_cameras": True,
            "allow_real_robot": True,
            "hardware_sync_execution_binding": execution_binding,
        },
    )

    for device_id, role, mounting_mode, timestamp_ns, pixel_value in (
        ("master", "master", "static", 1_000_000_000, 10),
        ("hand", "subordinate", "eye_in_hand", 1_000_100_000, 20),
    ):
        raw_sensor = run_root / f"realsense_{device_id}"
        synchronized_sensor = (
            run_root
            / "processed"
            / "synchronized"
            / f"realsense_{device_id}"
        )
        for sensor_folder in (raw_sensor, synchronized_sensor):
            (sensor_folder / RGB_DIR).mkdir(parents=True)
            (sensor_folder / DEPTH_DIR).mkdir()
        rgb = np.full((5, 6, 3), pixel_value, dtype=np.uint8)
        depth = np.full((5, 6), pixel_value * 10, dtype=np.uint16)
        assert cv2.imwrite(
            (raw_sensor / RGB_DIR / "1000.png").as_posix(),
            rgb,
        )
        assert cv2.imwrite(
            (raw_sensor / DEPTH_DIR / "1000.png").as_posix(),
            depth,
        )
        assert cv2.imwrite(
            (synchronized_sensor / RGB_DIR / "000000.png").as_posix(),
            rgb,
        )
        assert cv2.imwrite(
            (synchronized_sensor / DEPTH_DIR / "000000.png").as_posix(),
            depth,
        )
        (synchronized_sensor / CAM_K).write_text(
            "10 0 3\n0 10 2.5\n0 0 1\n0 0 0 0 0\n"
        )
        (synchronized_sensor / DEPTH_SCALE).write_text("1.0\n")
        write_json(
            synchronized_sensor / CAMERA_DATA_JSON,
            {
                "K": [[10, 0, 3], [0, 10, 2.5], [0, 0, 1]],
                "resolution": [5, 6],
                "orientation": "normal",
                "distortion": [0.0] * 5,
                "distortion_model": "brown_conrady",
            },
        )
        inter_cam_sync_mode = 1 if role == "master" else 2
        write_json(
            synchronized_sensor / MATCH_ROBOT_EE_POSES,
            {
                "000000.png": {
                    "source_frame_id": "1000.png",
                    "matched_robot_pose_index": 20,
                    "robot_timestamp_ns": timestamp_ns + 50,
                    "nearest_robot_delta_ns": 50,
                    "motion": "capture",
                    "robot_ee_pose": {
                        "x": 1,
                        "y": 2,
                        "z": 3,
                        "a": 0,
                        "b": 0,
                        "c": 0,
                    },
                }
            },
        )
        metadata = {
            "schema_version": "frame_metadata.v1",
            "sensor_type": "realsense_d435",
            "sensor_id": device_id,
            "orientation": "normal",
            "frame_index": 0,
            "frame_id": "000000.png",
            "rgb_path": "rgb/000000.png",
            "depth_path": "depth/000000.png",
            "source_frame_index": 10,
            "source_frame_id": "1000.png",
            "source_rgb_path": "rgb/1000.png",
            "source_depth_path": "depth/1000.png",
            "depth_sensor_timestamp_ns": timestamp_ns,
            "depth_frame_number": 100,
            "depth_timestamp_domain": "global_time",
            "capture_group_id": "mixed-rig",
            "hardware_sync_role": role,
            "hardware_sync_scope": "depth_exposure",
            "hardware_sync_transport": "realsense_inter_cam_sync",
            "inter_cam_sync_mode_configured": inter_cam_sync_mode,
            "inter_cam_sync_mode_readback": inter_cam_sync_mode,
            "matched_robot_pose_index": 20,
            "nearest_robot_delta_ns": 50,
            "motion": "capture",
            "mounting_mode": mounting_mode,
        }
        (synchronized_sensor / FRAME_METADATA_JSONL).write_text(
            json.dumps(metadata) + "\n"
        )

    groups = build_hardware_sync_frame_groups(run_root)
    groups["hardware_sync_qualification"] = qualification
    groups["hardware_sync_execution_binding"] = execution_binding
    write_hardware_sync_frame_groups(run_root, groups)
    if rectified:
        synchronized = run_root / "processed" / "synchronized"
        rectify_run(
            run_root,
            [
                factory_intrinsic_profile(synchronized / sensor_name)
                for sensor_name in ("realsense_master", "realsense_hand")
            ],
        )
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_bop_export_stage.py"),
            str(run_root),
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    # The calibration-provenance check is orthogonal to this fixture. Keep its
    # manifest structurally ready so the complete report can exercise the
    # hardware-sync check as a positive gate rather than as an isolated helper.
    manifest_path = run_root / BOP_DIR / BOP_EXPORT_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["calibration_profiles"] = [
        {"profile_id": "hardware-gate-profile", "status": "valid"}
    ]
    for export in manifest["exports"]:
        export["calibration_profile_id"] = "hardware-gate-profile"
    write_json(manifest_path, manifest)
    return run_root


def hardware_gate_check(run_root: Path) -> dict[str, object]:
    report = build_bop_export_readiness_gate_report(run_root)
    return next(
        check
        for check in report["checks"]
        if check["name"] == "bop_hardware_sync_frame_sets"
    )


def test_bop_export_readiness_gate_blocks_missing_targets(tmp_path: Path) -> None:
    run_root = tmp_path / "bop-run"
    populate_bop_export(run_root, with_targets=False)

    report = build_bop_export_readiness_gate_report(run_root)

    assert report["gate_id"] == BOP_EXPORT_READINESS_GATE_ID
    assert report["overall_status"] == "blocked"
    blockers = {blocker["name"] for blocker in report["next_blockers"]}
    assert "bop_targets" in blockers


def test_bop_export_readiness_accepts_consistent_objectless_dataset(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "objectless-bop"
    populate_bop_export(run_root)
    bop = run_root / BOP_DIR
    scene = bop / "test" / "000001"
    write_json(scene / "scene_gt.json", {"0": []})
    write_json(scene / "scene_gt_info.json", {"0": []})
    write_json(bop / BOP_TARGETS_BOP19, [])
    shutil.rmtree(bop / "models")
    manifest = json.loads((bop / BOP_EXPORT_MANIFEST).read_text())
    manifest.update({"objectless": True, "selected_objects": [], "object_models": []})
    write_json(bop / BOP_EXPORT_MANIFEST, manifest)

    report = build_bop_export_readiness_gate_report(run_root)

    assert report["overall_status"] == "ready"


def test_bop_export_readiness_requires_frame_sets_for_hardware_run(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "hardware-bop"
    populate_bop_export(run_root)
    write_run_config(
        run_root,
        create_run_config(
            run_root=run_root,
            sensors=(
                SensorRunConfig(
                    "realsense_d435",
                    "123",
                    "Static",
                    mounting_mode="static",
                ),
                SensorRunConfig(
                    "realsense_d435",
                    "456",
                    "Robot",
                    mounting_mode="eye_in_hand",
                ),
            ),
            synchronization={
                "schema_version": "capture_synchronization.v1",
                "mode": "hardware_trigger",
                "implementation": "realsense_inter_cam_sync",
                "scope": "depth_exposure",
                "group_id": "mixed-rig",
                "master_sensor_key": "realsense_d435:123",
                "max_depth_timestamp_skew_ms": 2.0,
            },
        ),
    )

    report = build_bop_export_readiness_gate_report(run_root)

    assert report["overall_status"] == "blocked"
    blocker = next(
        item
        for item in report["next_blockers"]
        if item["name"] == "bop_hardware_sync_frame_sets"
    )
    assert "multiview_frame_groups.json" in blocker["message"]
    assert "posetestbot_frame_sets.json" in blocker["message"]


def test_bop_export_readiness_accepts_generated_hardware_sync_provenance(
    tmp_path: Path,
) -> None:
    run_root = populate_hardware_sync_bop_export(tmp_path)

    report = build_bop_export_readiness_gate_report(run_root)

    assert report["overall_status"] == "ready"
    check = hardware_gate_check(run_root)
    assert check["status"] == "ready"
    assert check["details"]["source_validation_error"] is None
    assert check["details"]["qualification_matches"] is True
    assert check["details"]["execution_binding_matches"] is True
    assert check["details"]["execution_binding_errors"] == {}
    assert check["details"]["projection_truth_errors"] == []


def test_bop_hardware_sync_gate_accepts_current_rectified_source_truth(
    tmp_path: Path,
) -> None:
    run_root = populate_hardware_sync_bop_export(tmp_path, rectified=True)

    report = build_bop_export_readiness_gate_report(run_root)

    assert report["overall_status"] == "ready"
    check = hardware_gate_check(run_root)
    assert check["status"] == "ready"
    assert check["details"]["projection_truth_errors"] == []


@pytest.mark.parametrize(
    "tamper",
    (
        "scene_im_remap",
        "source_rgb_depth",
        "bop_paths",
        "sensor_order",
        "frame_set_count",
        "max_skew",
        "source_schema",
        "group_inventory_mount",
        "group_inventory_role",
        "current_run_inventory",
        "qualification",
        "execution_binding",
        "execution_binding_shape",
        "capture_execution_binding",
        "stale_source_content",
        "malformed_scene_id",
        "malformed_source_groups",
        "malformed_frame_sets",
        "malformed_frame_map",
    ),
)
def test_bop_hardware_sync_gate_rejects_tampered_provenance(
    tmp_path: Path,
    tamper: str,
) -> None:
    run_root = populate_hardware_sync_bop_export(tmp_path)
    bop_root = run_root / BOP_DIR
    frame_sets_path = bop_root / BOP_FRAME_SETS
    frame_map_path = bop_root / BOP_FRAME_MAP_JSON
    groups_path = (
        run_root
        / "processed"
        / "synchronized"
        / MULTIVIEW_FRAME_GROUPS
    )
    frame_sets = json.loads(frame_sets_path.read_text())
    frame_map = json.loads(frame_map_path.read_text())
    groups = json.loads(groups_path.read_text())
    views = frame_sets["frame_sets"][0]["views"]

    if tamper == "scene_im_remap":
        views[0]["scene_id"] = views[1]["scene_id"]
        views[0]["im_id"] = views[1]["im_id"]
        write_json(frame_sets_path, frame_sets)
    elif tamper == "source_rgb_depth":
        first_scene = frame_map["scenes"][str(views[0]["scene_id"])]
        first_frame = first_scene["frames"][str(views[0]["im_id"])]
        first_frame["source_rgb"] = "rgb/000042.png"
        first_frame["source_depth"] = "depth/000042.png"
        write_json(frame_map_path, frame_map)
    elif tamper == "bop_paths":
        first_scene = frame_map["scenes"][str(views[0]["scene_id"])]
        first_frame = first_scene["frames"][str(views[0]["im_id"])]
        first_frame["bop_rgb"] = "rgb/000042.png"
        views[0]["bop_depth"] = "test/000001/depth/000042.png"
        write_json(frame_map_path, frame_map)
        write_json(frame_sets_path, frame_sets)
    elif tamper == "sensor_order":
        frame_sets["sensor_order"] = list(
            reversed(frame_sets["sensor_order"])
        )
        write_json(frame_sets_path, frame_sets)
    elif tamper == "frame_set_count":
        frame_sets["frame_set_count"] += 1
        write_json(frame_sets_path, frame_sets)
    elif tamper == "max_skew":
        frame_sets["max_depth_timestamp_skew_ns"] += 1
        write_json(frame_sets_path, frame_sets)
    elif tamper == "source_schema":
        frame_sets["source_schema_version"] = "wrong.v1"
        write_json(frame_sets_path, frame_sets)
    elif tamper == "group_inventory_mount":
        groups["sensors"][1]["mounting_mode"] = "static"
        write_json(groups_path, groups)
    elif tamper == "group_inventory_role":
        groups["sensors"][1]["hardware_sync_role"] = "master"
        write_json(groups_path, groups)
    elif tamper == "current_run_inventory":
        config_path = run_root / "run_config.json"
        run_config = json.loads(config_path.read_text())
        run_config["capture"]["sensors"][1]["mounting_mode"] = "static"
        write_json(config_path, run_config)
    elif tamper == "qualification":
        frame_sets["hardware_sync_qualification"][
            "artifact_sha256"
        ] = "0" * 64
        write_json(frame_sets_path, frame_sets)
    elif tamper == "execution_binding":
        frame_sets["hardware_sync_execution_binding"][
            "qualification_artifact_sha256"
        ] = "0" * 64
        write_json(frame_sets_path, frame_sets)
    elif tamper == "execution_binding_shape":
        frame_sets["hardware_sync_execution_binding"]["unexpected"] = True
        write_json(frame_sets_path, frame_sets)
    elif tamper == "capture_execution_binding":
        capture_report_path = run_root / CAPTURE_EXECUTION_REPORT
        capture_report = json.loads(capture_report_path.read_text())
        capture_report["hardware_sync_execution_binding"][
            "qualification_artifact_sha256"
        ] = "0" * 64
        write_json(capture_report_path, capture_report)
    elif tamper == "stale_source_content":
        stale_rgb = (
            run_root
            / "processed"
            / "synchronized"
            / "realsense_master"
            / RGB_DIR
            / "000000.png"
        )
        assert cv2.imwrite(
            stale_rgb.as_posix(),
            np.full((5, 6, 3), 99, dtype=np.uint8),
        )
    elif tamper == "malformed_scene_id":
        views[0]["scene_id"] = ["not", "an", "integer"]
        write_json(frame_sets_path, frame_sets)
    elif tamper == "malformed_source_groups":
        groups["groups"] = {"not": "a list"}
        write_json(groups_path, groups)
    elif tamper == "malformed_frame_sets":
        frame_sets["frame_sets"] = 42
        write_json(frame_sets_path, frame_sets)
    elif tamper == "malformed_frame_map":
        frame_map["scenes"] = ["not", "an", "object"]
        write_json(frame_map_path, frame_map)
    else:  # pragma: no cover - parameter list owns the supported cases.
        raise AssertionError(tamper)

    report = build_bop_export_readiness_gate_report(run_root)

    assert report["overall_status"] == "blocked"
    check = hardware_gate_check(run_root)
    assert check["status"] == "blocked"


def test_bop_export_readiness_requires_matching_pose_template_instance_evidence(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "template-bop"
    populate_bop_export(run_root)
    template_uuid = "11111111-1111-4111-8111-111111111111"
    instance_uuid = "22222222-2222-4222-8222-222222222222"
    catalog_uuid = "33333333-3333-4333-8333-333333333333"
    matrix = {"matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
    selection = {
        "schema_version": "pose_template_selection.v1",
        "template_uuid": template_uuid,
        "bundle_sha256": "a" * 64,
        "configuration_sha256": "b" * 64,
        "placement_confirmed": True,
        "template_base_from_pose_template": matrix,
    }
    write_json(run_root / "pose_template_selection.json", selection)
    selection_sha = hashlib.sha256(
        (run_root / "pose_template_selection.json").read_bytes()
    ).hexdigest()
    object_instances = {
        "schema_version": "object_instances.v1",
        "template_uuid": template_uuid,
        "bundle_sha256": "a" * 64,
        "selection_sha256": selection_sha,
        "instances": [
            {
                "instance_uuid": instance_uuid,
                "catalog_uuid": catalog_uuid,
                "obj_id": 1,
                "canonical_ply_sha256": "c" * 64,
            }
        ],
    }
    write_json(run_root / "object_instances.json", object_instances)
    pose_sidecar = {
        "schema_version": "posetestbot_pose_template.v1",
        "template_uuid": template_uuid,
        "bundle_sha256": "a" * 64,
        "configuration_sha256": "b" * 64,
        "template_base_from_pose_template": matrix,
    }
    write_json(run_root / BOP_DIR / "posetestbot_pose_template.json", pose_sidecar)
    write_json(
        run_root / BOP_DIR / "posetestbot_instance_map.json",
        {
            "schema_version": "posetestbot_bop_instance_map.v1",
            "instances": [
                {
                    "scene_id": 1,
                    "im_id": 0,
                    "gt_id": 0,
                    "obj_id": 1,
                    "instance_uuid": instance_uuid,
                    "catalog_uuid": catalog_uuid,
                }
            ],
        },
    )
    write_json(
        run_root
        / "processed"
        / "synchronized"
        / "realsense_123"
        / "blenderproc"
        / "output"
        / "posetestbot_render_instances.json",
        {
            "schema_version": "posetestbot_render_instances.v1",
            "blenderproc_version": "2.8.0",
            "identity_contract": "bop_gt_index_matches_loaded_instance_order.v1",
            "instances": [
                {
                    "instance_uuid": instance_uuid,
                    "catalog_uuid": catalog_uuid,
                    "obj_id": 1,
                }
            ],
        },
    )
    models_info = json.loads(
        (run_root / BOP_DIR / "models" / "models_info.json").read_text()
    )
    models_info["1"]["posetestbot_geometry"] = {"source_sha256": "c" * 64}
    write_json(run_root / BOP_DIR / "models" / "models_info.json", models_info)
    manifest_path = run_root / BOP_DIR / BOP_EXPORT_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "schema_version": "bop_export_manifest.v3",
            "dataset_mode": "pose_template",
            "pose_template": {
                "template_uuid": template_uuid,
                "bundle_sha256": "a" * 64,
            },
        }
    )
    write_json(manifest_path, manifest)

    report = build_bop_export_readiness_gate_report(run_root)

    assert report["overall_status"] == "ready"
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["bop_pose_template_evidence_agreement"]["status"] == "ready"

    instance_map = json.loads(
        (run_root / BOP_DIR / "posetestbot_instance_map.json").read_text()
    )
    instance_map["instances"][0]["instance_uuid"] = (
        "44444444-4444-4444-8444-444444444444"
    )
    write_json(run_root / BOP_DIR / "posetestbot_instance_map.json", instance_map)
    blocked = build_bop_export_readiness_gate_report(run_root)
    assert blocked["overall_status"] == "blocked"


def test_calibration_validation_gate_ready_after_promotion(tmp_path: Path) -> None:
    run_root = tmp_path / "calibration-run"
    write_run_config(
        run_root,
        create_run_config(
            run_root=run_root,
            sensors=(
                SensorRunConfig(
                    "realsense_d435",
                    "123",
                    "Wrist RealSense",
                    mounting_mode="eye_in_hand",
                ),
            ),
        ),
    )
    write_json(
        run_root / CALIBRATION_VALIDATION_REPORT,
        {
            "schema_version": "calibration_validation_report.v1",
            "overall_status": "ok",
            "profile_count": 1,
            "promotable_profile_count": 1,
            "promotion": {
                "requested": True,
                "promoted": True,
                "profile_count": 1,
                "promoted_profile_ids": ["profile-1"],
                "path": "calibration_profiles.json",
            },
        },
    )
    write_json(
        run_root / CALIBRATION_PROFILES,
        {
            "schema_version": "calibration_profiles.v1",
            "profiles": [
                {
                    "profile_id": "profile-1",
                    "sensor_id": "123",
                    "sensor_type": "realsense_d435",
                    "mounting_mode": "eye_in_hand",
                    "status": "valid",
                    "quality": {
                        "num_inliers": 8,
                        "residual_translation_mm": 1.0,
                        "residual_rotation_deg": 0.5,
                    },
                }
            ],
        },
    )

    report = build_calibration_validation_gate_report(run_root)

    assert report["gate_id"] == CALIBRATION_VALIDATION_GATE_ID
    assert report["overall_status"] == "ready"


def test_calibration_validation_gate_allows_preserved_valid_profiles(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "calibration-merged-run"
    write_run_config(
        run_root,
        create_run_config(
            run_root=run_root,
            sensors=tuple(
                SensorRunConfig(
                    "realsense_d435",
                    profile_id,
                    profile_id,
                    mounting_mode="eye_in_hand",
                )
                for profile_id in ("profile-existing", "profile-new")
            ),
        ),
    )
    write_json(
        run_root / CALIBRATION_VALIDATION_REPORT,
        {
            "schema_version": "calibration_validation_report.v1",
            "overall_status": "ok",
            "profile_count": 1,
            "promotable_profile_count": 1,
            "promotion": {
                "requested": True,
                "promoted": True,
                "profile_count": 2,
                "promoted_profile_ids": ["profile-new"],
                "preserved_profile_ids": ["profile-existing"],
                "path": "calibration_profiles.json",
            },
        },
    )
    profiles = []
    for profile_id in ("profile-existing", "profile-new"):
        profiles.append(
            {
                "profile_id": profile_id,
                "sensor_id": profile_id,
                "sensor_type": "realsense_d435",
                "mounting_mode": "eye_in_hand",
                "status": "valid",
                "quality": {
                    "num_inliers": 8,
                    "residual_translation_mm": 1.0,
                    "residual_rotation_deg": 0.5,
                },
            }
        )
    write_json(
        run_root / CALIBRATION_PROFILES,
        {"schema_version": "calibration_profiles.v1", "profiles": profiles},
    )

    report = build_calibration_validation_gate_report(run_root)

    assert report["overall_status"] == "ready"


def test_calibration_validation_gate_blocks_partial_enabled_sensor_coverage(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "partial-calibration-run"
    write_run_config(
        run_root,
        create_run_config(
            run_root=run_root,
            sensors=(
                SensorRunConfig("realsense_d435", "1", "First"),
                SensorRunConfig("realsense_d435", "2", "Second"),
            ),
        ),
    )
    write_json(
        run_root / CALIBRATION_VALIDATION_REPORT,
        {
            "overall_status": "ok",
            "profile_count": 1,
            "promotable_profile_count": 1,
            "promotion": {
                "requested": True,
                "promoted": True,
                "profile_count": 1,
                "promoted_profile_ids": ["profile-1"],
                "path": CALIBRATION_PROFILES,
            },
        },
    )
    write_json(
        run_root / CALIBRATION_PROFILES,
        {
            "profiles": [
                {
                    "profile_id": "profile-1",
                    "sensor_id": "1",
                    "sensor_type": "realsense_d435",
                    "mounting_mode": "eye_in_hand",
                    "status": "valid",
                    "quality": {
                        "num_inliers": 8,
                        "residual_translation_mm": 1.0,
                        "residual_rotation_deg": 0.5,
                    },
                }
            ]
        },
    )

    report = build_calibration_validation_gate_report(run_root)

    coverage = next(
        check
        for check in report["checks"]
        if check["name"] == "calibration_profile_sensor_coverage"
    )
    assert report["overall_status"] == "blocked"
    assert coverage["status"] == "blocked"
    assert coverage["details"]["enabled_sensor_count"] == 2
    assert coverage["details"]["covered_sensor_count"] == 1


def test_calibration_validation_gate_ignores_disabled_sensor_coverage(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "disabled-calibration-run"
    write_run_config(
        run_root,
        create_run_config(
            run_root=run_root,
            sensors=(
                SensorRunConfig("realsense_d435", "1", "First"),
                SensorRunConfig("realsense_d435", "2", "Offline", enabled=False),
            ),
        ),
    )
    write_json(
        run_root / CALIBRATION_VALIDATION_REPORT,
        {
            "overall_status": "ok",
            "profile_count": 1,
            "promotable_profile_count": 1,
            "promotion": {
                "requested": True,
                "promoted": True,
                "profile_count": 1,
                "promoted_profile_ids": ["profile-1"],
                "path": CALIBRATION_PROFILES,
            },
        },
    )
    write_json(
        run_root / CALIBRATION_PROFILES,
        {
            "profiles": [
                {
                    "profile_id": "profile-1",
                    "sensor_id": "1",
                    "sensor_type": "realsense_d435",
                    "mounting_mode": "eye_in_hand",
                    "status": "valid",
                    "quality": {
                        "num_inliers": 8,
                        "residual_translation_mm": 1.0,
                        "residual_rotation_deg": 0.5,
                    },
                }
            ]
        },
    )

    report = build_calibration_validation_gate_report(run_root)

    coverage = next(
        check
        for check in report["checks"]
        if check["name"] == "calibration_profile_sensor_coverage"
    )
    assert report["overall_status"] == "ready"
    assert coverage["status"] == "ready"
    assert coverage["details"]["enabled_sensor_count"] == 1


def test_calibration_validation_gate_ignores_invalid_disabled_sensor_profile(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "disabled-invalid-profile-run"
    write_run_config(
        run_root,
        create_run_config(
            run_root=run_root,
            sensors=(
                SensorRunConfig("realsense_d435", "1", "Enabled"),
                SensorRunConfig("realsense_d435", "2", "Disabled", enabled=False),
            ),
        ),
    )
    write_json(
        run_root / CALIBRATION_VALIDATION_REPORT,
        {
            "overall_status": "ok",
            "profile_count": 1,
            "promotable_profile_count": 1,
            "promotion": {
                "requested": True,
                "promoted": True,
                "profile_count": 2,
                "promoted_profile_ids": ["profile-enabled"],
                "path": CALIBRATION_PROFILES,
            },
        },
    )
    write_json(
        run_root / CALIBRATION_PROFILES,
        {
            "profiles": [
                {
                    "profile_id": "profile-enabled",
                    "sensor_id": "1",
                    "sensor_type": "realsense_d435",
                    "mounting_mode": "eye_in_hand",
                    "status": "valid",
                    "quality": {
                        "num_inliers": 8,
                        "residual_translation_mm": 1.0,
                        "residual_rotation_deg": 0.5,
                    },
                },
                {
                    "profile_id": "profile-disabled-needs-validation",
                    "sensor_id": "2",
                    "sensor_type": "realsense_d435",
                    "mounting_mode": "eye_in_hand",
                    "status": "needs_validation",
                    "quality": {
                        "num_inliers": 0,
                        "residual_translation_mm": None,
                        "residual_rotation_deg": None,
                    },
                },
            ]
        },
    )

    report = build_calibration_validation_gate_report(run_root)

    collection = next(
        check for check in report["checks"] if check["name"] == "calibration_profiles"
    )
    assert report["overall_status"] == "ready"
    assert collection["status"] == "ready"
    assert collection["details"]["validated_profile_ids"] == ["profile-enabled"]
    assert collection["details"]["ignored_disabled_profile_ids"] == [
        "profile-disabled-needs-validation"
    ]


def test_rewrite_status_uses_three_real_data_gate_ids(tmp_path: Path) -> None:
    run_root = tmp_path / "status-run"

    report = build_rewrite_status_report(run_root)

    gate_ids = [gate["gate_id"] for gate in report["gates"]]
    assert tuple(gate_ids) == GATE_IDS
    assert len(gate_ids) == 3
    assert "rewrite_fake_acquisition_to_bop.v1" not in gate_ids
    assert "rewrite_foundationpose_runtime.v1" not in gate_ids


def test_rewrite_status_recommendations_use_current_object_contract(
    tmp_path: Path,
) -> None:
    report = build_rewrite_status_report(
        tmp_path / "status-run",
        gate_ids=(BOP_EXPORT_READINESS_GATE_ID,),
    )
    recommendations = json.dumps(report["next_actions"])

    assert "pose-template selection or objectless contract" in recommendations
    assert "object registry" not in recommendations


def test_retired_fake_gate_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown rewrite gate"):
        build_gate_report(
            tmp_path / "run",
            gate_id="rewrite_fake_acquisition_to_bop.v1",
        )


def test_rewrite_gate_cli_accepts_bop_export_readiness_gate(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = tmp_path / "bop-cli"
    populate_bop_export(run_root)

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_rewrite_gate.py",
            run_root.as_posix(),
            "--gate",
            BOP_EXPORT_READINESS_GATE_ID,
            "--json",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["gate_id"] == BOP_EXPORT_READINESS_GATE_ID
    assert payload["overall_status"] == "ready"


def test_full_capture_gate_accepts_embedded_pre_start_capture_plan_preflight(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "embedded-preflight"
    run_root.mkdir()
    write_json(
        run_root / "capture_execution_plan.json",
        {
            "schema_version": "capture_execution_plan.v1",
            "status": "ok",
            "ready_to_execute": True,
            "preflight_status": "ok",
            "preflight_report": {
                "schema_version": "capture_plan_preflight.v1",
                "overall_status": "ok",
                "checks": [
                    {
                        "name": "sensor_output_folder:realsense_1",
                        "status": "ok",
                        "message": "Output folder was empty before capture.",
                    }
                ],
            },
        },
    )

    from posetestbot.pipeline.rewrite_gate import build_full_capture_gate_report

    report = build_full_capture_gate_report(run_root)

    preflight = next(
        check
        for check in report["checks"]
        if check["name"] == "capture_plan_preflight"
    )
    assert preflight["status"] == "ready"
    assert preflight["artifact"].endswith("capture_execution_plan.json")
    assert (
        preflight["details"]["source"]
        == "capture_execution_plan.json:preflight_report"
    )


def test_full_capture_gate_rejects_mismatched_embedded_preflight_status(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "mismatched-embedded-preflight"
    run_root.mkdir()
    write_json(
        run_root / "capture_execution_plan.json",
        {
            "schema_version": "capture_execution_plan.v1",
            "status": "ok",
            "ready_to_execute": True,
            "preflight_status": "warning",
            "preflight_report": {
                "schema_version": "capture_plan_preflight.v1",
                "overall_status": "ok",
                "checks": [],
            },
        },
    )

    from posetestbot.pipeline.rewrite_gate import build_full_capture_gate_report

    report = build_full_capture_gate_report(run_root)

    preflight = next(
        check
        for check in report["checks"]
        if check["name"] == "capture_plan_preflight"
    )
    assert preflight["status"] == "blocked"
    assert preflight["message"] == "capture_plan_preflight_report.json is missing."


def test_full_capture_gate_checks_real_hardware_snapshot(tmp_path: Path) -> None:
    run_root = tmp_path / "real-run"
    config = create_run_config(run_root=run_root)
    write_run_config(run_root, config)
    write_json(
        run_root / RUN_PREFLIGHT_REPORT,
        {
            "schema_version": "run_preflight.v1",
            "overall_status": "ok",
            "config": config.to_dict(),
        },
    )
    write_json(
        run_root / HARDWARE_STATUS_REPORT,
        {
            "schema_version": "hardware_status_report.v1",
            "overall_status": "ok",
            "robot_status": {"selected_profile": {"mode": "unexpected"}},
        },
    )

    from posetestbot.pipeline.rewrite_gate import build_full_capture_gate_report

    report = build_full_capture_gate_report(run_root)

    hardware = next(
        check for check in report["checks"] if check["name"] == "hardware_status"
    )
    assert report["gate_id"] == FULL_CAPTURE_GATE_ID
    assert hardware["status"] == "blocked"
    assert hardware["details"]["robot_mode_ok"] is False
