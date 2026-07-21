from __future__ import annotations

import json
from pathlib import Path

from posetestbot.calibration.profiles import (
    SCHEMA_VERSION,
    CalibrationProfile,
    CalibrationQuality,
    CalibrationStatus,
    RigidTransform,
    TransformFrame,
    write_profile_collection,
)
from posetestbot.cell.scene import build_cell_scene, cell_timeline_page
from posetestbot.io.artifacts import BOP_DIR, BOP_EXPORT_MANIFEST
from posetestbot.pipeline.run_config import (
    FixedFrameTransform,
    create_run_config,
    sensor_config_from_token,
    write_run_config,
)
from posetestbot.sensors.contracts import CameraIntrinsics, MountingMode, SensorType
from posetestbot.web.app import create_app


def profile(sensor_id: str, mounting: MountingMode) -> CalibrationProfile:
    return CalibrationProfile(
        schema_version=SCHEMA_VERSION,
        profile_id=f"realsense_{sensor_id}_{mounting.value}",
        sensor_id=sensor_id,
        sensor_type=SensorType.REALSENSE_D435,
        mounting_mode=mounting,
        rig_position=f"slot_{sensor_id}",
        intrinsics=CameraIntrinsics(
            cam_k=(600, 0, 320, 0, 600, 240, 0, 0, 1),
            width=640,
            height=480,
        ),
        extrinsics=RigidTransform(
            from_frame=TransformFrame.CAMERA,
            to_frame=(TransformFrame.ROBOT_FLANGE if mounting == MountingMode.EYE_IN_HAND else TransformFrame.TEMPLATE_BASE),
            rotation_quaternion_wxyz=(1, 0, 0, 0),
            translation_mm=(10, 20, 30),
        ),
        status=CalibrationStatus.VALID,
        quality=CalibrationQuality(num_observations=4, num_inliers=4),
    )


def make_scene_run(tmp_path: Path) -> Path:
    profiles_path = tmp_path / "profiles.json"
    write_profile_collection([profile("111", MountingMode.EYE_IN_HAND), profile("222", MountingMode.STATIC)], profiles_path)
    run_root = tmp_path / "run"
    config = create_run_config(
        run_root=run_root,
        calibration_profiles=profiles_path.as_posix(),
        sensors=(
            sensor_config_from_token("realsense:111:eye_in_hand:Wrist camera"),
            sensor_config_from_token("realsense:222:static:Static camera"),
        ),
        fixed_transforms=(
            FixedFrameTransform("physical_robot_base", "template_base", (1, 0, 0, 0), (100, 0, 0)),
            FixedFrameTransform("tcp", "robot_flange", (1, 0, 0, 0), (0, 0, 120)),
        ),
    )
    write_run_config(run_root, config)
    for sensor in ("realsense_111", "realsense_222"):
        folder = run_root / "processed" / "synchronized" / sensor
        folder.mkdir(parents=True)
        (folder / "match_robot_ee_poses.json").write_text(json.dumps({
            f"{index:06d}.png": {"motion": "arc", "robot_ee_pose": {"X": index, "Y": 2, "Z": 3, "A": 0, "B": 0, "C": 0}}
            for index in range(3)
        }))
    (run_root / "calibration_target.json").write_text(json.dumps({
        "schema_version": "calibration_target.v1",
        "target_type": "aruco_grid",
        "grid_size": [2, 2],
        "marker_length": 40,
        "marker_separation": 50,
        "placement": {"from": "aruco_grid", "to": "template_base", "rotation_quaternion_wxyz": [1, 0, 0, 0], "translation_mm": [5, 6, 7]},
    }))
    return run_root


def test_scene_composes_frames_sensors_and_exact_timelines(tmp_path: Path) -> None:
    run_root = make_scene_run(tmp_path)
    scene = build_cell_scene(run_root)
    entities = {entity["id"]: entity for entity in scene["entities"]}

    assert scene["schema_version"] == "cell_scene.v1"
    assert scene["coordinate_system"]["up_axis"] == "+Z"
    assert entities["physical_robot_base"]["transform"]["translation_mm"] == [100.0, 0.0, 0.0]
    assert entities["tcp"]["transform"]["parent_frame"] == "robot_flange"
    assert entities["camera:realsense_111"]["transform"]["parent_frame"] == "robot_flange"
    assert entities["camera:realsense_222"]["transform"]["parent_frame"] == "template_base"
    assert not any(entity["type"] == "object" for entity in scene["entities"])
    assert scene["object_selection"]["dataset_mode"] == "objectless"
    assert entities["calibration_target"]["transform"]["translation_mm"] == [5.0, 6.0, 7.0]
    assert len(scene["timelines"]) == 2
    assert [pose["index"] for pose in scene["trajectory_preview"]] == [0, 1, 2]
    assert scene["object_selection"]["bop_export"]["status"] == "not_exported"

    timeline = cell_timeline_page(run_root, scene["default_timeline_id"], offset=1, limit=5000)
    assert timeline["schema_version"] == "cell_timeline.v1"
    assert timeline["limit"] == 2000
    assert [pose["frame_id"] for pose in timeline["poses"]] == ["000001.png", "000002.png"]
    assert timeline["poses"][0]["transform"]["translation_mm"] == [1.0, 2.0, 3.0]


def test_scene_marks_mismatched_bop_export_provenance_stale(tmp_path: Path) -> None:
    run_root = make_scene_run(tmp_path)
    manifest_path = run_root / BOP_DIR / BOP_EXPORT_MANIFEST
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "bop_export_manifest.v2",
                "dataset_mode": "pose_template",
            }
        )
    )

    scene = build_cell_scene(run_root)

    provenance = scene["object_selection"]["bop_export"]
    assert provenance["status"] == "stale"
    assert provenance["dataset_mode_matches"] is False
    assert any(
        warning["code"] == "stale_bop_export_provenance"
        for warning in scene["warnings"]
    )


def test_scene_marks_missing_calibration_and_supports_raw_fallback(tmp_path: Path) -> None:
    run_root = make_scene_run(tmp_path)
    for path in (run_root / "processed").rglob("match_robot_ee_poses.json"):
        path.unlink()
    (run_root / "raw_robot_ee_poses.json").write_text(json.dumps({"0": {"motion": "raw", "pose": {"X": 9, "Y": 8, "Z": 7, "A": 0, "B": 0, "C": 0}}}))
    config = json.loads((run_root / "run_config.json").read_text())
    config["calibration_profiles"] = None
    (run_root / "run_config.json").write_text(json.dumps(config))

    scene = build_cell_scene(run_root)

    assert scene["default_timeline_id"] == "raw:robot"
    assert any(entity["type"] == "camera" and entity["status"] == "unresolved" for entity in scene["entities"])
    assert any(warning["code"] == "missing_calibration_profiles" for warning in scene["warnings"])


def test_cell_apis_assets_and_objectless_state(tmp_path: Path, monkeypatch) -> None:
    run_root = make_scene_run(tmp_path)
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", tmp_path.as_posix())
    monkeypatch.setenv("POSETESTBOT_WEB_INPUT_ROOTS", tmp_path.as_posix())
    client = create_app().test_client()

    scene = client.get("/ui/cell-scene", query_string={"run_root": run_root}).get_json()
    rejected = client.get(f"/ui/cell-assets/cube/mesh?run_root={run_root.as_posix()}")
    timeline = client.get("/ui/cell-scene/timeline", query_string={"run_root": run_root, "timeline_id": scene["default_timeline_id"], "offset": 0, "limit": 2})

    assert scene["object_selection"]["objectless"] is True
    assert not any(entity["type"] == "object" for entity in scene["entities"])
    assert rejected.status_code == 404
    assert timeline.status_code == 200
    assert len(timeline.get_json()["poses"]) == 2


def test_retired_cell_registry_asset_route_is_absent(tmp_path: Path, monkeypatch) -> None:
    run_root = make_scene_run(tmp_path)
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", tmp_path.as_posix())
    monkeypatch.setenv("POSETESTBOT_WEB_INPUT_ROOTS", tmp_path.as_posix())
    client = create_app().test_client()

    response = client.get(f"/ui/cell-assets/cube/mesh?run_root={run_root.as_posix()}")

    assert response.status_code == 404
