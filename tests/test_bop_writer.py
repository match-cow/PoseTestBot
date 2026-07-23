from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from posetestbot.bop.writer import (
    BopSceneExport,
    bop_frame_sets_from_hardware_groups,
    model_geometry_info,
)


def test_large_model_diameter_is_exact_not_aabb_diagonal(tmp_path: Path) -> None:
    vertices = [
        (0.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (0.0, 4.0, 0.0),
        (0.0, 0.0, 12.0),
    ]
    vertices.extend(vertices[index % 4] for index in range(4_997))
    path = tmp_path / "large_tetrahedron.ply"
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "element face 0",
        "property list uchar int vertex_indices",
        "end_header",
        *(f"{x} {y} {z}" for x, y, z in vertices),
        "",
    ]
    path.write_text("\n".join(lines))

    info = model_geometry_info(path)

    assert math.isclose(float(info["diameter"]), math.sqrt(160.0))
    assert not math.isclose(float(info["diameter"]), 13.0)
    geometry = info["posetestbot_geometry"]
    assert geometry["diameter_method"] == "exact_convex_hull_vertex_pairwise"
    assert geometry["vertex_count"] == 5_001
    assert geometry["convex_hull_vertex_count"] == 4


def _scene_export(sensor_name: str, scene_id: int) -> BopSceneExport:
    return BopSceneExport(
        sensor_name=sensor_name,
        scene_id=scene_id,
        split="test",
        scene_folder=f"test/{scene_id:06d}",
        rgb_count=1,
        depth_count=1,
        artifacts={},
        frame_map={
            "0": {
                "sensor_name": sensor_name,
                "scene_id": scene_id,
                "source_rgb": "rgb/000010.png",
                "source_depth": "depth/000010.png",
                "bop_rgb": "rgb/000000.png",
                "bop_depth": "depth/000000.png",
            }
        },
    )


def _hardware_groups() -> dict:
    master = "realsense_d435:master"
    subordinate = "realsense_d435:hand"
    master_pose = {"matched_robot_pose_index": 5}
    frame_refs = {
        master: {
            "sensor_key": master,
            "sensor_folder": "processed/synchronized/realsense_master",
            "mounting_mode": "static",
            "hardware_sync_role": "master",
            "synchronized_frame_index": 0,
            "synchronized_frame_id": "000010.png",
            "synchronized_rgb_path": "rgb/000010.png",
            "synchronized_depth_path": "depth/000010.png",
            "source_frame_index": 10,
            "source_frame_id": "10.png",
            "source_sensor_folder": "realsense_master",
            "source_rgb_path": "rgb/10.png",
            "source_depth_path": "depth/10.png",
            "depth_sensor_timestamp_ns": 1_000_000,
            "depth_frame_number": 10,
            "depth_timestamp_domain": "global_time",
            "depth_timestamp_skew_ns": 0,
            "abs_depth_timestamp_skew_ns": 0,
            "matched_robot_pose": master_pose,
        },
        subordinate: {
            "sensor_key": subordinate,
            "sensor_folder": "processed/synchronized/realsense_hand",
            "mounting_mode": "eye_in_hand",
            "hardware_sync_role": "subordinate",
            "synchronized_frame_index": 0,
            "synchronized_frame_id": "000010.png",
            "synchronized_rgb_path": "rgb/000010.png",
            "synchronized_depth_path": "depth/000010.png",
            "source_frame_index": 12,
            "source_frame_id": "12.png",
            "source_sensor_folder": "realsense_hand",
            "source_rgb_path": "rgb/12.png",
            "source_depth_path": "depth/12.png",
            "depth_sensor_timestamp_ns": 1_000_100,
            "depth_frame_number": 12,
            "depth_timestamp_domain": "global_time",
            "depth_timestamp_skew_ns": 100,
            "abs_depth_timestamp_skew_ns": 100,
            "matched_robot_pose": {"matched_robot_pose_index": 6},
        },
    }
    value = {
        "schema_version": "hardware_sync_frame_groups.v1",
        "run_config_path": "run_config.json",
        "group_id": "mixed-rig",
        "mode": "hardware_trigger",
        "implementation": "realsense_inter_cam_sync",
        "scope": "depth_exposure",
        "master_sensor_key": master,
        "max_depth_timestamp_skew_ms": 2.0,
        "max_depth_timestamp_skew_ns": 2_000_000,
        "hardware_sync_execution_binding": {
            "configuration_sha256": "1" * 64,
            "qualification_artifact_sha256": "2" * 64,
            "revalidated_immediately_before_receiver_spawn": True,
        },
        "sensor_order": [master, subordinate],
        "sensors": [
            {
                "sensor_key": master,
                "sensor_type": "realsense_d435",
                "device_id": "master",
                "sensor_folder": "processed/synchronized/realsense_master",
                "mounting_mode": "static",
                "hardware_sync_role": "master",
                "frame_metadata_path": (
                    "processed/synchronized/realsense_master/"
                    "frame_metadata.jsonl"
                ),
                "matched_robot_poses_path": (
                    "processed/synchronized/realsense_master/"
                    "match_robot_ee_poses.json"
                ),
                "frame_count": 1,
            },
            {
                "sensor_key": subordinate,
                "sensor_type": "realsense_d435",
                "device_id": "hand",
                "sensor_folder": "processed/synchronized/realsense_hand",
                "mounting_mode": "eye_in_hand",
                "hardware_sync_role": "subordinate",
                "frame_metadata_path": (
                    "processed/synchronized/realsense_hand/"
                    "frame_metadata.jsonl"
                ),
                "matched_robot_poses_path": (
                    "processed/synchronized/realsense_hand/"
                    "match_robot_ee_poses.json"
                ),
                "frame_count": 1,
            },
        ],
        "groups": [
            {
                "frame_group_id": "mixed-rig:000000",
                "frame_group_index": 0,
                "master_frame_ordinal": 0,
                "capture_group_id": "mixed-rig",
                "master_sensor_key": master,
                "depth_sensor_timestamp_ns": 1_000_000,
                "max_abs_depth_timestamp_skew_ns": 100,
                "depth_timestamp_span_ns": 100,
                "matched_robot_pose": master_pose,
                "frames": frame_refs,
            }
        ],
        "summary": {"complete_group_count": 1},
    }
    zero_sha = "0" * 64

    def canonical_sha256(payload: object) -> str:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    provenance_sensors = []
    for sensor in value["sensors"]:
        base_sensor = {
            "sensor_key": sensor["sensor_key"],
            "frame_metadata": {
                "path": sensor["frame_metadata_path"],
                "size_bytes": 0,
                "sha256": zero_sha,
            },
            "matched_robot_poses": {
                "path": sensor["matched_robot_poses_path"],
                "size_bytes": 0,
                "sha256": zero_sha,
            },
            "referenced_frames": {
                "file_count": 4,
                "total_size_bytes": 0,
                "manifest_sha256": zero_sha,
            },
        }
        provenance_sensors.append(
            {
                **base_sensor,
                "content_sha256": canonical_sha256(base_sensor),
            }
        )
    base_provenance = {
        "schema_version": "hardware_sync_content_provenance.v1",
        "digest_algorithm": "sha256",
        "hardware_contract": {
            "schema_version": "hardware_sync_run_contract.v1",
            "path": "run_config.json",
            "sha256": zero_sha,
        },
        "sensors": provenance_sensors,
    }
    value["content_provenance"] = {
        **base_provenance,
        "aggregate_sha256": canonical_sha256(base_provenance),
    }
    return value


def test_bop_frame_sets_map_complete_group_across_sensor_scenes() -> None:
    value = bop_frame_sets_from_hardware_groups(
        [
            _scene_export("realsense_master", 3),
            _scene_export("realsense_hand", 8),
        ],
        _hardware_groups(),
    )

    assert value["schema_version"] == "posetestbot_frame_sets.v1"
    assert value["frame_set_count"] == 1
    assert value["hardware_sync_execution_binding"] == {
        "configuration_sha256": "1" * 64,
        "qualification_artifact_sha256": "2" * 64,
        "revalidated_immediately_before_receiver_spawn": True,
    }
    assert value["synchronization_claims"] == {
        "depth_exposure_hardware_synchronized": True,
        "rgb_exposure_hardware_synchronized": False,
        "rgb_association": "same_device_frameset_timestamp_association",
        "synthetic_robot_occlusion_modeled": False,
    }
    frame_set = value["frame_sets"][0]
    assert frame_set["frame_set_id"] == "mixed-rig:000000"
    assert frame_set["depth_timestamp_span_ns"] == 100
    assert [(view["scene_id"], view["im_id"]) for view in frame_set["views"]] == [
        (3, 0),
        (8, 0),
    ]


def test_bop_frame_sets_reject_missing_exported_group_member() -> None:
    with pytest.raises(ValueError, match="exactly cover"):
        bop_frame_sets_from_hardware_groups(
            [_scene_export("realsense_master", 3)],
            _hardware_groups(),
        )


def test_bop_frame_sets_reject_malformed_capture_execution_binding() -> None:
    groups = _hardware_groups()
    groups["hardware_sync_execution_binding"]["unexpected"] = True

    with pytest.raises(
        ValueError,
        match="hardware_sync_execution_binding contains missing or unknown fields",
    ):
        bop_frame_sets_from_hardware_groups(
            [
                _scene_export("realsense_master", 3),
                _scene_export("realsense_hand", 8),
            ],
            groups,
        )
