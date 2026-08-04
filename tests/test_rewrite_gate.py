from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from posetestbot.io.artifacts import (
    BOP_DIR,
    BOP_EXPORT_MANIFEST,
    BOP_FRAME_MAP_JSON,
    BOP_TARGETS_BOP19,
    CALIBRATION_PROFILES,
    CALIBRATION_VALIDATION_REPORT,
    DEPTH_DIR,
    HARDWARE_STATUS_REPORT,
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
from posetestbot.robot.reference_frames import POSE_TEMPLATE_BASE_SUNRISE_PATH


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


def test_bop_export_readiness_accepts_clean_annotation_free_v5_layout(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "clean-objectless-bop"
    populate_bop_export(run_root)
    bop = run_root / BOP_DIR
    scene = bop / "test" / "000001"
    (scene / "scene_gt.json").unlink()
    (scene / "scene_gt_info.json").unlink()
    (bop / BOP_TARGETS_BOP19).unlink()
    shutil.rmtree(bop / "models")
    frame_map_path = bop / BOP_FRAME_MAP_JSON
    frame_map = json.loads(frame_map_path.read_text())
    frame_map["schema_version"] = "posetestbot_bop_frame_map.v3"
    frame_map["scenes"]["1"] = {
        "sensor_name": "realsense_123",
        "split": "test",
        "scene_folder": "test/000001",
        "projection": "native",
        "input_sensor_folder": "processed/synchronized/realsense_123",
        "authoritative_source_sensor_folder": ("processed/synchronized/realsense_123"),
        "frames": {
            "0": {
                "source_rgb": "rgb/000000.png",
                "source_depth": "depth/000000.png",
                "bop_rgb": "rgb/000000.png",
                "bop_depth": "depth/000000.png",
            }
        },
    }
    write_json(frame_map_path, frame_map)
    manifest_path = bop / BOP_EXPORT_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "schema_version": "bop_export_manifest.v5",
            "objectless": True,
            "dataset_mode": "objectless",
            "annotation_source": "none",
            "annotation_state": "absent",
            "targets_path": None,
            "instance_map_path": None,
            "object_models": [],
        }
    )
    write_json(manifest_path, manifest)

    report = build_bop_export_readiness_gate_report(run_root)

    assert report["overall_status"] == "ready"
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["bop_targets"]["status"] == "ready"
    assert checks["bop_scene:realsense_123"]["details"]["annotation_layout_ok"] is True


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

    write_json(
        run_root / BOP_DIR / "posetestbot_instance_map.json",
        {
            "schema_version": "posetestbot_bop_instance_map.v1",
            "instances": [],
        },
    )
    scene = run_root / BOP_DIR / "test" / "000001"
    write_json(scene / "scene_gt.json", {"0": []})
    write_json(scene / "scene_gt_info.json", {"0": []})
    write_json(run_root / BOP_DIR / BOP_TARGETS_BOP19, [])
    (
        run_root
        / "processed"
        / "synchronized"
        / "realsense_123"
        / "blenderproc"
        / "output"
        / "posetestbot_render_instances.json"
    ).unlink()
    manifest["annotation_source"] = "none"
    write_json(manifest_path, manifest)

    annotation_free = build_bop_export_readiness_gate_report(run_root)

    assert annotation_free["overall_status"] == "ready"
    annotation_free_checks = {
        check["name"]: check for check in annotation_free["checks"]
    }
    evidence = annotation_free_checks["bop_pose_template_evidence_agreement"]
    assert evidence["status"] == "ready"
    assert evidence["details"]["annotation_source"] == "none"
    assert set(evidence["details"]["render_sensors"].values()) == {"not_required"}

    (scene / "scene_gt.json").unlink()
    (scene / "scene_gt_info.json").unlink()
    (run_root / BOP_DIR / "posetestbot_instance_map.json").unlink()
    write_json(
        run_root / BOP_DIR / BOP_TARGETS_BOP19,
        [{"scene_id": 1, "im_id": 0, "obj_id": 1, "inst_count": 1}],
    )
    valid_bop_model = "\n".join(
        (
            "ply",
            "format ascii 1.0",
            "element vertex 3",
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
            "element face 1",
            "property list uchar int vertex_indices",
            "end_header",
            "0 0 0 0 0 1",
            "1 0 0 0 0 1",
            "0 1 0 0 0 1",
            "3 0 1 2",
            "",
        )
    )
    (run_root / BOP_DIR / "models" / "obj_000001.ply").write_text(valid_bop_model)
    write_json(
        run_root / BOP_DIR / "models_eval" / "models_info.json",
        models_info,
    )
    (run_root / BOP_DIR / "models_eval" / "obj_000001.ply").write_text(valid_bop_model)
    frame_map_path = run_root / BOP_DIR / BOP_FRAME_MAP_JSON
    write_json(
        frame_map_path,
        {
            "schema_version": "posetestbot_bop_frame_map.v3",
            "scenes": {
                "1": {
                    "sensor_name": "realsense_123",
                    "split": "test",
                    "scene_folder": "test/000001",
                    "projection": "native",
                    "input_sensor_folder": "processed/synchronized/realsense_123",
                    "authoritative_source_sensor_folder": (
                        "processed/synchronized/realsense_123"
                    ),
                    "frames": {
                        "0": {
                            "source_rgb": "rgb/000000.png",
                            "source_depth": "depth/000000.png",
                            "bop_rgb": "rgb/000000.png",
                            "bop_depth": "depth/000000.png",
                        }
                    },
                }
            },
        },
    )
    manifest.update(
        {
            "schema_version": "bop_export_manifest.v5",
            "annotation_source": "none",
            "annotation_state": "absent",
            "targets_path": BOP_TARGETS_BOP19,
            "instance_map_path": None,
            "frame_map_path": BOP_FRAME_MAP_JSON,
        }
    )
    manifest["object_models"][0]["bop_eval_path"] = "models_eval/obj_000001.ply"
    write_json(manifest_path, manifest)

    clean_annotation_free = build_bop_export_readiness_gate_report(run_root)

    assert clean_annotation_free["overall_status"] == "ready"
    clean_checks = {check["name"]: check for check in clean_annotation_free["checks"]}
    assert clean_checks["bop_targets"]["status"] == "ready"
    assert clean_checks["bop_targets"]["details"]["target_count"] == 1


def _static_robot_pose_reference(
    path: str = POSE_TEMPLATE_BASE_SUNRISE_PATH,
) -> dict[str, str]:
    return {
        "schema_version": "robot_pose_reference.v1",
        "status": "verified",
        "packet_schema_version": "robot_pose.v1",
        "from": "robot_flange",
        "to": "template_base",
        "sunrise_reference_frame_path": path,
    }


def _populate_static_calibration_validation_gate(
    run_root: Path,
    *,
    robot_pose_reference: dict[str, str] | None,
) -> None:
    write_run_config(
        run_root,
        create_run_config(
            run_root=run_root,
            sensors=(
                SensorRunConfig(
                    "realsense_d435",
                    "static-123",
                    "Static RealSense",
                    mounting_mode="static",
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
                "promoted_profile_ids": ["profile-static"],
                "path": CALIBRATION_PROFILES,
            },
        },
    )
    profile: dict[str, object] = {
        "profile_id": "profile-static",
        "sensor_id": "static-123",
        "sensor_type": "realsense_d435",
        "mounting_mode": "static",
        "status": "valid",
        "quality": {
            "num_inliers": 8,
            "residual_translation_mm": 1.0,
            "residual_rotation_deg": 0.5,
        },
    }
    if robot_pose_reference is not None:
        profile["metadata"] = {"robot_pose_reference": robot_pose_reference}
    write_json(
        run_root / CALIBRATION_PROFILES,
        {
            "schema_version": "calibration_profiles.v1",
            "profiles": [profile],
        },
    )


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


def test_calibration_validation_gate_accepts_canonical_static_reference(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "canonical-static-calibration-run"
    _populate_static_calibration_validation_gate(
        run_root,
        robot_pose_reference=_static_robot_pose_reference(),
    )

    report = build_calibration_validation_gate_report(run_root)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["overall_status"] == "ready"
    assert checks["calibration_profiles"]["status"] == "ready"
    assert checks["calibration_profiles"]["details"]["profiles"][0][
        "static_reference_requirement_met"
    ]
    assert checks["calibration_profile_sensor_coverage"]["status"] == "ready"


@pytest.mark.parametrize(
    "robot_pose_reference",
    [
        None,
        _static_robot_pose_reference("/PoseTestBot/TemplateBase"),
        {
            **_static_robot_pose_reference(),
            "packet_schema_version": "robot_pose.v0",
        },
    ],
    ids=("unprovenanced", "wrong-path", "malformed-verified-evidence"),
)
def test_calibration_validation_gate_blocks_noncanonical_static_reference(
    tmp_path: Path,
    robot_pose_reference: dict[str, str] | None,
) -> None:
    run_root = tmp_path / "noncanonical-static-calibration-run"
    _populate_static_calibration_validation_gate(
        run_root,
        robot_pose_reference=robot_pose_reference,
    )

    report = build_calibration_validation_gate_report(run_root)

    checks = {check["name"]: check for check in report["checks"]}
    profile_check = checks["calibration_profiles"]
    coverage_check = checks["calibration_profile_sensor_coverage"]
    assert report["overall_status"] == "blocked"
    assert profile_check["status"] == "blocked"
    assert not profile_check["details"]["profiles"][0][
        "static_reference_requirement_met"
    ]
    assert coverage_check["status"] == "blocked"
    assert coverage_check["details"]["sensors"][0]["matching_profile_ids"] == []


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
        check for check in report["checks"] if check["name"] == "capture_plan_preflight"
    )
    assert preflight["status"] == "ready"
    assert preflight["artifact"].endswith("capture_execution_plan.json")
    assert (
        preflight["details"]["source"] == "capture_execution_plan.json:preflight_report"
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
        check for check in report["checks"] if check["name"] == "capture_plan_preflight"
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
