from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from posetestbot.calibration.profiles import (
    SCHEMA_VERSION,
    CalibrationProfile,
    CalibrationQuality,
    CalibrationStatus,
    RigidTransform,
    TransformFrame,
    write_profile_collection,
)
from posetestbot.cell.scene import (
    _pose_template_footprint,
    build_cell_scene,
    cell_timeline_page,
)
from posetestbot.io.artifacts import BOP_DIR, BOP_EXPORT_MANIFEST
from posetestbot.pipeline.run_config import (
    FixedFrameTransform,
    create_run_config,
    sensor_config_from_token,
    write_run_config,
)
from posetestbot.sensors.contracts import CameraIntrinsics, MountingMode, SensorType
from posetestbot.web.app import create_app


def profile(
    sensor_id: str,
    mounting: MountingMode,
    *,
    profile_id: str | None = None,
    rig_position: str | None = None,
    rotation_quaternion_wxyz: tuple[float, float, float, float] = (1, 0, 0, 0),
    translation_mm: tuple[float, float, float] = (10, 20, 30),
) -> CalibrationProfile:
    return CalibrationProfile(
        schema_version=SCHEMA_VERSION,
        profile_id=profile_id or f"realsense_{sensor_id}_{mounting.value}",
        sensor_id=sensor_id,
        sensor_type=SensorType.REALSENSE_D435,
        mounting_mode=mounting,
        rig_position=rig_position or f"slot_{sensor_id}",
        intrinsics=CameraIntrinsics(
            cam_k=(600, 0, 320, 0, 600, 240, 0, 0, 1),
            width=640,
            height=480,
        ),
        extrinsics=RigidTransform(
            from_frame=TransformFrame.CAMERA,
            to_frame=(
                TransformFrame.ROBOT_FLANGE
                if mounting == MountingMode.EYE_IN_HAND
                else TransformFrame.TEMPLATE_BASE
            ),
            rotation_quaternion_wxyz=rotation_quaternion_wxyz,
            translation_mm=translation_mm,
        ),
        calibration_dataset_id="attempt-dataset-1",
        method="auto_compare:IPPE+park",
        status=CalibrationStatus.VALID,
        quality=CalibrationQuality(
            num_observations=8,
            num_inliers=7,
            mean_reprojection_error_px=0.25,
            residual_translation_mm=0.5,
            residual_rotation_deg=0.2,
        ),
        operator="cell-test",
        calibrated_at="2026-07-21T12:00:00+00:00",
        metadata={
            "target_id": "target-1",
            "intrinsic_profile_id": "intrinsic-1",
            "promotion_attempt_id": "a" * 32,
            "promotion_candidate_id": "candidate-1",
            "promotion_multi_camera_bundle_id": "joint:IPPE:park",
            "promotion_solver_provenance": {
                "solver_policy": "auto_compare",
                "pnp_method": "IPPE",
                "extrinsic_method": "park",
            },
            "promoted_at": "2026-07-21T12:00:00+00:00",
            "promoted_by": "cell-test",
            "outlier_count": 1,
            "outlier_ratio": 0.125,
            "companion_transform": {
                "from": "aruco_grid",
                "to": (
                    "template_base"
                    if mounting == MountingMode.EYE_IN_HAND
                    else "robot_flange"
                ),
                "matrix": [
                    [1, 0, 0, 1],
                    [0, 1, 0, 2],
                    [0, 0, 1, 3],
                    [0, 0, 0, 1],
                ],
                "rotation_quaternion_wxyz": [1, 0, 0, 0],
                "translation_mm": [1, 2, 3],
            },
        },
    )


def make_scene_run(tmp_path: Path) -> Path:
    profiles_path = tmp_path / "profiles.json"
    write_profile_collection(
        [
            profile(
                "111",
                MountingMode.EYE_IN_HAND,
                profile_id="promoted_wrist_111",
            ),
            profile(
                "111",
                MountingMode.EYE_IN_HAND,
                profile_id="older_wrist_111",
                rig_position="legacy_wrist_slot",
            ),
            profile(
                "111",
                MountingMode.STATIC,
                profile_id="static_profile_111",
                rig_position="static_slot",
            ),
            profile(
                "222",
                MountingMode.EYE_IN_HAND,
                profile_id="wrong_wrist_222",
                rig_position="other_wrist_slot",
            ),
            profile("222", MountingMode.STATIC),
        ],
        profiles_path,
    )
    run_root = tmp_path / "run"
    wrist = replace(
        sensor_config_from_token("realsense:111:eye_in_hand:Wrist camera"),
        calibration_profile_id="promoted_wrist_111",
    )
    config = create_run_config(
        run_root=run_root,
        calibration_profiles=profiles_path.as_posix(),
        sensors=(
            wrist,
            sensor_config_from_token("realsense:222:static:Static camera"),
        ),
        fixed_transforms=(
            FixedFrameTransform(
                "physical_robot_base", "template_base", (1, 0, 0, 0), (100, 0, 0)
            ),
            FixedFrameTransform("tcp", "robot_flange", (1, 0, 0, 0), (0, 0, 120)),
        ),
    )
    write_run_config(run_root, config)
    for sensor in ("realsense_111", "realsense_222"):
        folder = run_root / "processed" / "synchronized" / sensor
        folder.mkdir(parents=True)
        (folder / "match_robot_ee_poses.json").write_text(
            json.dumps(
                {
                    f"{index:06d}.png": {
                        "motion": "arc",
                        "robot_ee_pose": {
                            "X": index,
                            "Y": 2,
                            "Z": 3,
                            "A": 0,
                            "B": 0,
                            "C": 0,
                        },
                    }
                    for index in range(3)
                }
            )
        )
    (run_root / "calibration_target.json").write_text(
        json.dumps(
            {
                "schema_version": "calibration_target.v1",
                "target_type": "aruco_grid",
                "grid_size": [2, 2],
                "marker_length": 40,
                "marker_separation": 50,
                "placement": {
                    "from": "aruco_grid",
                    "to": "template_base",
                    "rotation_quaternion_wxyz": [1, 0, 0, 0],
                    "translation_mm": [5, 6, 7],
                },
            }
        )
    )
    return run_root


def write_wrist_run_config(
    run_root: Path,
    calibration_profiles: str,
    *,
    profile_id: str = "wrist_111",
) -> None:
    wrist = replace(
        sensor_config_from_token("realsense:111:eye_in_hand:Wrist camera"),
        calibration_profile_id=profile_id,
    )
    write_run_config(
        run_root,
        create_run_config(
            run_root=run_root,
            calibration_profiles=calibration_profiles,
            sensors=(wrist,),
        ),
    )


def test_scene_composes_frames_sensors_and_exact_timelines(tmp_path: Path) -> None:
    run_root = make_scene_run(tmp_path)
    scene = build_cell_scene(run_root)
    entities = {entity["id"]: entity for entity in scene["entities"]}

    assert scene["schema_version"] == "cell_scene.v1"
    assert scene["coordinate_system"]["up_axis"] == "-Z"
    presentation = scene["coordinate_system"]["presentation"]
    assert presentation["mode"] == "calibration_target_front"
    assert presentation["presentation_only"] is True
    assert presentation["target_frame"] == {
        "name": "aruco_grid",
        "origin": "compensated_outer_board_top_left",
        "axes": {"x": "right", "y": "down", "z": "into_board"},
    }
    target_to_reference = np.eye(4)
    target_to_reference[:3, 3] = [5, 6, 7]
    target_to_display = np.asarray(presentation["matrix"]) @ target_to_reference
    assert np.allclose(target_to_display, np.diag([1, -1, -1, 1]))
    assert np.linalg.det(target_to_display[:3, :3]) == pytest.approx(1)
    assert np.allclose(
        target_to_display @ np.asarray([0, 0, -500, 1]),
        [0, 0, 500, 1],
    )
    assert entities["physical_robot_base"]["transform"]["translation_mm"] == [
        100.0,
        0.0,
        0.0,
    ]
    assert entities["tcp"]["transform"]["parent_frame"] == "robot_flange"
    assert (
        entities["camera:realsense_111"]["transform"]["parent_frame"] == "robot_flange"
    )
    assert (
        entities["camera:realsense_222"]["transform"]["parent_frame"] == "template_base"
    )
    wrist_calibration = entities["camera:realsense_111"]["calibration"]
    assert wrist_calibration["profile_id"] == "promoted_wrist_111"
    assert wrist_calibration["status"] == "valid"
    assert wrist_calibration["extrinsics"] == {
        "from": "camera",
        "to": "robot_flange",
        "matrix": [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "rotation_quaternion_wxyz": [1, 0, 0, 0],
        "translation_mm": [10, 20, 30],
    }
    assert wrist_calibration["companion_transform"] == {
        "from": "aruco_grid",
        "to": "template_base",
        "matrix": [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "translation_mm": [1.0, 2.0, 3.0],
    }
    assert wrist_calibration["quality"]["num_inliers"] == 7
    assert wrist_calibration["quality"]["outlier_count"] == 1
    assert (
        wrist_calibration["evidence"]["profile_source"]
        == (run_root.parent / "profiles.json").as_posix()
    )
    assert wrist_calibration["evidence"]["promotion_attempt_id"] == "a" * 32
    assert wrist_calibration["evidence"]["promotion_multi_camera_bundle_id"] == (
        "joint:IPPE:park"
    )
    assert wrist_calibration["evidence"]["promotion_solver_provenance"] == {
        "solver_policy": "auto_compare",
        "pnp_method": "IPPE",
        "extrinsic_method": "park",
    }
    assert not any(entity["type"] == "object" for entity in scene["entities"])
    assert scene["object_selection"]["dataset_mode"] == "objectless"
    assert entities["calibration_target"]["transform"]["translation_mm"] == [
        5.0,
        6.0,
        7.0,
    ]
    assert entities["calibration_target"]["geometry"]["frame"] == (
        presentation["target_frame"]
    )
    assert len(scene["timelines"]) == 2
    assert [pose["index"] for pose in scene["trajectory_preview"]] == [0, 1, 2]
    assert scene["object_selection"]["bop_export"]["status"] == "not_exported"

    timeline = cell_timeline_page(
        run_root, scene["default_timeline_id"], offset=1, limit=5000
    )
    assert timeline["schema_version"] == "cell_timeline.v1"
    assert timeline["limit"] == 2000
    assert [pose["frame_id"] for pose in timeline["poses"]] == [
        "000001.png",
        "000002.png",
    ]
    assert timeline["poses"][0]["transform"]["translation_mm"] == [1.0, 2.0, 3.0]


def test_scene_omits_disabled_camera_even_when_a_valid_profile_exists(
    tmp_path: Path,
) -> None:
    run_root = make_scene_run(tmp_path)
    config_path = run_root / "run_config.json"
    config = json.loads(config_path.read_text())
    wrist = next(
        sensor
        for sensor in config["capture"]["sensors"]
        if sensor["device_id"] == "111"
    )
    wrist["enabled"] = False
    config_path.write_text(json.dumps(config))

    scene = build_cell_scene(run_root)
    camera_ids = {
        entity["id"] for entity in scene["entities"] if entity["type"] == "camera"
    }

    assert "camera:realsense_111" not in camera_ids
    assert "camera:realsense_222" in camera_ids
    assert [timeline["id"] for timeline in scene["timelines"]] == [
        "sensor:realsense_222"
    ]
    assert scene["default_timeline_id"] == "sensor:realsense_222"
    flange = next(
        entity for entity in scene["entities"] if entity["id"] == "robot_flange"
    )
    assert flange["provenance"]["source"].endswith(
        "processed/synchronized/realsense_222/match_robot_ee_poses.json"
    )


def test_scene_retains_reference_z_up_presentation_without_grid_target(
    tmp_path: Path,
) -> None:
    run_root = make_scene_run(tmp_path)
    (run_root / "calibration_target.json").unlink()

    scene = build_cell_scene(run_root)

    assert scene["coordinate_system"]["up_axis"] == "+Z"
    presentation = scene["coordinate_system"]["presentation"]
    assert presentation["mode"] == "reference_z_up"
    assert presentation["target_frame"] is None
    assert np.allclose(presentation["matrix"], np.eye(4))


def test_scene_raw_timeline_fallback_ignores_disabled_sensor_folder(
    tmp_path: Path,
) -> None:
    run_root = make_scene_run(tmp_path)
    shutil.rmtree(run_root / "processed")
    config_path = run_root / "run_config.json"
    config = json.loads(config_path.read_text())
    next(
        sensor
        for sensor in config["capture"]["sensors"]
        if sensor["device_id"] == "111"
    )["enabled"] = False
    config_path.write_text(json.dumps(config))
    for device_id, x_mm in (("111", 111), ("222", 222)):
        folder = run_root / f"realsense_{device_id}"
        folder.mkdir()
        (folder / "raw_robot_ee_poses.json").write_text(
            json.dumps(
                {
                    "0": {
                        "pose": {
                            "X": x_mm,
                            "Y": 2,
                            "Z": 3,
                            "A": 0,
                            "B": 0,
                            "C": 0,
                        }
                    }
                }
            )
        )

    scene = build_cell_scene(run_root)

    assert scene["default_timeline_id"] == "raw:robot"
    assert scene["trajectory_preview"][0]["transform"]["translation_mm"] == [
        222.0,
        2.0,
        3.0,
    ]
    flange = next(
        entity for entity in scene["entities"] if entity["id"] == "robot_flange"
    )
    assert flange["provenance"]["source"].endswith(
        "realsense_222/raw_robot_ee_poses.json"
    )


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


def test_scene_marks_missing_calibration_and_supports_raw_fallback(
    tmp_path: Path,
) -> None:
    run_root = make_scene_run(tmp_path)
    for path in (run_root / "processed").rglob("match_robot_ee_poses.json"):
        path.unlink()
    (run_root / "raw_robot_ee_poses.json").write_text(
        json.dumps(
            {
                "0": {
                    "motion": "raw",
                    "pose": {"X": 9, "Y": 8, "Z": 7, "A": 0, "B": 0, "C": 0},
                }
            }
        )
    )
    config = json.loads((run_root / "run_config.json").read_text())
    config["calibration_profiles"] = None
    config["frames"]["fixed_transforms"] = []
    (run_root / "run_config.json").write_text(json.dumps(config))

    scene = build_cell_scene(run_root)

    assert scene["default_timeline_id"] == "raw:robot"
    assert any(
        entity["type"] == "camera" and entity["status"] == "unresolved"
        for entity in scene["entities"]
    )
    assert any(
        warning["code"] == "missing_calibration_profiles"
        for warning in scene["warnings"]
    )
    entities = {item["id"]: item for item in scene["entities"]}
    assert entities["physical_robot_base"]["status"] == "not_configured"
    assert entities["physical_robot_base"]["unresolved_reason"] is None
    assert entities["tcp"]["status"] == "not_configured"


def test_scene_uses_latest_run_attempt_board_as_reference_surface(
    tmp_path: Path, monkeypatch
) -> None:
    run_root = make_scene_run(tmp_path)
    (run_root / "calibration_target.json").unlink()
    config_path = run_root / "run_config.json"
    config = json.loads(config_path.read_text())
    config["calibration_profiles"] = None
    config_path.write_text(json.dumps(config))
    attempt_id = "a" * 32
    attempt = run_root / "processed" / "calibration" / attempt_id
    bundle = attempt / "target_bundle"
    bundle.mkdir(parents=True)
    target = {
        "schema_version": "calibration_target.v2",
        "target_id": "target-from-attempt",
        "target_type": "aruco_grid",
        "display_name": "Attempt board",
        "target_bounds": {"x_mm": 0, "y_mm": 0, "width_mm": 90, "height_mm": 40},
        "grid_size": [2, 1],
        "markers": [
            {"id": 0, "corners_mm": [[0, 0, 0], [40, 0, 0], [40, 40, 0], [0, 40, 0]]},
            {"id": 1, "corners_mm": [[50, 0, 0], [90, 0, 0], [90, 40, 0], [50, 40, 0]]},
        ],
    }
    (bundle / "calibration_target.json").write_text(json.dumps(target))
    (bundle / "calibration_target.pdf").write_bytes(b"%PDF-1.4\n% cell test\n")
    (attempt / "request.json").write_text(
        json.dumps(
            {
                "attempt_id": attempt_id,
                "created_at": "2026-07-22T10:00:00Z",
                "target_mounting": {
                    "from": "aruco_grid",
                    "to": "template_base",
                    "state": "estimated",
                },
            }
        )
    )
    (attempt / "progress.json").write_text(
        json.dumps({"attempt_id": attempt_id, "status": "complete"})
    )

    scene = build_cell_scene(run_root)
    entities = {item["id"]: item for item in scene["entities"]}
    board = entities["calibration_target"]

    assert board["label"] == "Attempt board (reference placement)"
    assert board["status"] == "reference"
    assert board["transform"]["translation_mm"] == [0.0, 0.0, 0.0]
    assert len(board["geometry"]["markers"]) == 2
    assert board["geometry"]["pdf_url"].startswith(
        "/ui/cell-calibration-target-pdf?run_root="
    )
    assert board["provenance"]["attempt_id"] == attempt_id
    assert not any(item["id"] == "hri_template" for item in scene["entities"])

    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", tmp_path.as_posix())
    monkeypatch.setenv("POSETESTBOT_WEB_INPUT_ROOTS", tmp_path.as_posix())
    response = create_app().test_client().get(
        "/ui/cell-calibration-target-pdf",
        query_string={"run_root": run_root},
    )
    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    assert response.data.startswith(b"%PDF")


def test_scene_places_promoted_board_from_profile_companion_transform(
    tmp_path: Path,
) -> None:
    run_root = make_scene_run(tmp_path)
    target_path = run_root / "calibration_target.json"
    target = json.loads(target_path.read_text())
    target.pop("placement")
    target["target_id"] = "target-1"
    target_path.write_text(json.dumps(target))

    scene = build_cell_scene(run_root)
    board = next(item for item in scene["entities"] if item["id"] == "calibration_target")

    assert board["status"] == "planned"
    assert board["transform"]["translation_mm"] == [1.0, 2.0, 3.0]
    assert board["provenance"]["placement_source"] == (
        "promoted_calibration_profile_companion"
    )
    assert board["geometry"]["placement_known"] is True


def test_pose_template_footprint_uses_exact_snapshot_preview(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    snapshot = run_root / "processed" / "pose_template_selection"
    snapshot.mkdir(parents=True)
    (snapshot / "pose_template_bundle.json").write_text(
        json.dumps(
            {
                "display_name": "Fixture footprint",
                "configuration": {
                    "page": {
                        "size": "A3",
                        "orientation": "landscape",
                        "origin_from_lower_left_mm": [15, 15],
                    }
                },
            }
        )
    )
    contours = [[{"x_mm": 10, "y_mm": 20}, {"x_mm": 30, "y_mm": 20}, {"x_mm": 10, "y_mm": 40}]]
    (snapshot / "pose_template_preview.json").write_text(
        json.dumps(
            {
                "page": {"width_mm": 420, "height_mm": 297},
                "instances": [
                    {"instance_uuid": "instance-1", "compensated_contours": contours}
                ],
            }
        )
    )

    geometry = _pose_template_footprint(
        run_root,
        {
            "bundle_snapshot": "processed/pose_template_selection",
            "template_uuid": "template-1",
            "bundle_sha256": "b" * 64,
        },
    )

    assert geometry["kind"] == "pose_template_footprint"
    assert geometry["page"] == {"width_mm": 420, "height_mm": 297}
    assert geometry["contours"] == [
        {"instance_uuid": "instance-1", "contours": contours}
    ]


def test_scene_rejects_pinned_profile_for_another_mounting_identity(
    tmp_path: Path,
) -> None:
    run_root = make_scene_run(tmp_path)
    config = json.loads((run_root / "run_config.json").read_text())
    config["capture"]["sensors"][0]["calibration_profile_id"] = "static_profile_111"
    (run_root / "run_config.json").write_text(json.dumps(config))

    scene = build_cell_scene(run_root)
    wrist = next(
        entity for entity in scene["entities"] if entity["id"] == "camera:realsense_111"
    )

    assert wrist["status"] == "unresolved"
    assert wrist["transform"] is None
    assert "No eye_in_hand calibration profile matches" in wrist["unresolved_reason"]


def test_scene_rejects_pinned_profile_for_another_device(tmp_path: Path) -> None:
    run_root = make_scene_run(tmp_path)
    config = json.loads((run_root / "run_config.json").read_text())
    config["capture"]["sensors"][0]["calibration_profile_id"] = "wrong_wrist_222"
    (run_root / "run_config.json").write_text(json.dumps(config))

    scene = build_cell_scene(run_root)
    wrist = next(
        entity for entity in scene["entities"] if entity["id"] == "camera:realsense_111"
    )

    assert wrist["status"] == "unresolved"
    assert wrist["transform"] is None
    assert (
        "does not match sensor identity realsense_d435:111"
        in wrist["unresolved_reason"]
    )


def test_scene_never_reuses_family_generic_profile_across_devices(
    tmp_path: Path,
) -> None:
    profiles_path = tmp_path / "family-profiles.json"
    write_profile_collection(
        [
            profile(
                "realsense",
                MountingMode.EYE_IN_HAND,
                profile_id="family_generic_wrist",
            )
        ],
        profiles_path,
    )
    run_root = tmp_path / "generic-run"
    pinned = replace(
        sensor_config_from_token("realsense:111:eye_in_hand:Wrist one"),
        calibration_profile_id="family_generic_wrist",
    )
    write_run_config(
        run_root,
        create_run_config(
            run_root=run_root,
            calibration_profiles=profiles_path.as_posix(),
            sensors=(
                pinned,
                sensor_config_from_token("realsense:222:eye_in_hand:Wrist two"),
            ),
        ),
    )

    scene = build_cell_scene(run_root)
    cameras = [entity for entity in scene["entities"] if entity["type"] == "camera"]

    assert len(cameras) == 2
    assert all(camera["status"] == "unresolved" for camera in cameras)
    assert all(camera["transform"] is None for camera in cameras)
    assert all("calibration" not in camera for camera in cameras)


def test_scene_supports_run_relative_calibration_profile_path(tmp_path: Path) -> None:
    run_root = tmp_path / "relative-run"
    write_wrist_run_config(run_root, "calibration_profiles.json")
    profiles_path = run_root / "calibration_profiles.json"
    write_profile_collection(
        [profile("111", MountingMode.EYE_IN_HAND, profile_id="wrist_111")],
        profiles_path,
    )

    scene = build_cell_scene(run_root)
    wrist = next(entity for entity in scene["entities"] if entity["type"] == "camera")

    assert wrist["status"] == "planned"
    assert wrist["calibration"]["evidence"]["profile_source"] == (
        profiles_path.resolve().as_posix()
    )


def test_scene_supports_cli_style_cwd_relative_calibration_profile_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    run_root = tmp_path / "working_data" / "foo"
    configured_path = "working_data/foo/calibration_profiles.json"
    write_wrist_run_config(run_root, configured_path)
    profiles_path = run_root / "calibration_profiles.json"
    write_profile_collection(
        [profile("111", MountingMode.EYE_IN_HAND, profile_id="wrist_111")],
        profiles_path,
    )

    scene = build_cell_scene(run_root)
    wrist = next(entity for entity in scene["entities"] if entity["type"] == "camera")

    assert wrist["status"] == "planned"
    assert wrist["calibration"]["evidence"]["profile_source"] == (
        profiles_path.resolve().as_posix()
    )


def test_scene_rejects_ambiguous_relative_calibration_profile_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    run_root = tmp_path / "working_data" / "ambiguous"
    write_wrist_run_config(run_root, "profiles.json")
    write_profile_collection(
        [profile("111", MountingMode.EYE_IN_HAND, profile_id="wrist_111")],
        run_root / "profiles.json",
    )
    write_profile_collection(
        [profile("111", MountingMode.EYE_IN_HAND, profile_id="cwd_wrist_111")],
        tmp_path / "profiles.json",
    )

    scene = build_cell_scene(run_root)
    wrist = next(entity for entity in scene["entities"] if entity["type"] == "camera")

    assert wrist["status"] == "unresolved"
    assert any(
        warning["code"] == "invalid_calibration_profiles"
        and "Ambiguous calibration profile path" in warning["message"]
        for warning in scene["warnings"]
    )


def test_scene_coerces_legacy_numeric_evidence_to_json_numbers(
    tmp_path: Path,
) -> None:
    run_root = make_scene_run(tmp_path)
    profiles_path = tmp_path / "profiles.json"
    payload = json.loads(profiles_path.read_text())
    wrist_profile = next(
        item
        for item in payload["profiles"]
        if item["profile_id"] == "promoted_wrist_111"
    )
    wrist_profile["quality"].update(
        {
            "mean_reprojection_error_px": "0.25",
            "max_reprojection_error_px": "0.75",
            "residual_translation_mm": "0.5",
            "residual_rotation_deg": "0.2",
        }
    )
    wrist_profile["sync_delta_ms"] = "1.75"
    wrist_profile["metadata"]["outlier_count"] = "1"
    wrist_profile["metadata"]["outlier_ratio"] = "0.125"
    profiles_path.write_text(json.dumps(payload))

    scene = build_cell_scene(run_root)
    wrist = next(
        entity for entity in scene["entities"] if entity["id"] == "camera:realsense_111"
    )
    quality = wrist["calibration"]["quality"]
    evidence = wrist["calibration"]["evidence"]

    assert quality["mean_reprojection_error_px"] == 0.25
    assert quality["max_reprojection_error_px"] == 0.75
    assert quality["residual_translation_mm"] == 0.5
    assert quality["residual_rotation_deg"] == 0.2
    assert quality["outlier_count"] == 1
    assert quality["outlier_ratio"] == 0.125
    assert evidence["sync_delta_ms"] == 1.75
    assert isinstance(quality["mean_reprojection_error_px"], float)
    assert isinstance(quality["outlier_count"], int)


def test_scene_uses_wxyz_camera_to_parent_direction_for_rotated_extrinsic(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "rotated-run"
    profiles_path = tmp_path / "rotated-profiles.json"
    half_sqrt_two = 2**-0.5
    write_profile_collection(
        [
            profile(
                "111",
                MountingMode.EYE_IN_HAND,
                profile_id="wrist_111",
                rotation_quaternion_wxyz=(
                    half_sqrt_two,
                    0,
                    0,
                    half_sqrt_two,
                ),
                translation_mm=(10, 20, 30),
            )
        ],
        profiles_path,
    )
    write_wrist_run_config(run_root, profiles_path.as_posix())

    scene = build_cell_scene(run_root)
    wrist = next(entity for entity in scene["entities"] if entity["type"] == "camera")
    calibration = wrist["calibration"]
    matrix = np.asarray(calibration["extrinsics"]["matrix"])

    assert wrist["transform"]["parent_frame"] == "robot_flange"
    assert calibration["extrinsics"]["from"] == "camera"
    assert calibration["extrinsics"]["to"] == "robot_flange"
    np.testing.assert_allclose(
        calibration["extrinsics"]["rotation_quaternion_wxyz"],
        [half_sqrt_two, 0, 0, half_sqrt_two],
    )
    np.testing.assert_allclose(
        matrix,
        [
            [0, -1, 0, 10],
            [1, 0, 0, 20],
            [0, 0, 1, 30],
            [0, 0, 0, 1],
        ],
        atol=1e-12,
    )
    # Camera +X maps to parent +Y; an inverted parent-to-camera transform would not.
    np.testing.assert_allclose(matrix @ [1, 0, 0, 1], [10, 21, 30, 1])


def test_cell_apis_assets_and_objectless_state(tmp_path: Path, monkeypatch) -> None:
    run_root = make_scene_run(tmp_path)
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", tmp_path.as_posix())
    monkeypatch.setenv("POSETESTBOT_WEB_INPUT_ROOTS", tmp_path.as_posix())
    client = create_app().test_client()

    scene = client.get("/ui/cell-scene", query_string={"run_root": run_root}).get_json()
    rejected = client.get(f"/ui/cell-assets/cube/mesh?run_root={run_root.as_posix()}")
    timeline = client.get(
        "/ui/cell-scene/timeline",
        query_string={
            "run_root": run_root,
            "timeline_id": scene["default_timeline_id"],
            "offset": 0,
            "limit": 2,
        },
    )

    assert scene["object_selection"]["objectless"] is True
    assert not any(entity["type"] == "object" for entity in scene["entities"])
    assert rejected.status_code == 404
    assert timeline.status_code == 200
    assert len(timeline.get_json()["poses"]) == 2


def test_retired_cell_registry_asset_route_is_absent(
    tmp_path: Path, monkeypatch
) -> None:
    run_root = make_scene_run(tmp_path)
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", tmp_path.as_posix())
    monkeypatch.setenv("POSETESTBOT_WEB_INPUT_ROOTS", tmp_path.as_posix())
    client = create_app().test_client()

    response = client.get(f"/ui/cell-assets/cube/mesh?run_root={run_root.as_posix()}")

    assert response.status_code == 404
