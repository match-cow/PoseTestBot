from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from posetestbot.io.artifacts import (
    BOP_COCO_ANNOTATIONS,
    BOP_DIR,
    BOP_EXPORT_MANIFEST,
    BOP_FRAME_MAP_JSON,
    BOP_FRAME_SETS,
    BOP_TARGETS_BOP19,
    CAM_K,
    CAMERA_DATA_JSON,
    CAPTURE_EXECUTION_REPORT,
    DATASET_MANIFEST,
    DEPTH_DIR,
    DEPTH_SCALE,
    FRAME_METADATA_JSONL,
    MATCH_ROBOT_EE_POSES,
    MULTIVIEW_FRAME_GROUPS,
    RGB_DIR,
)
from posetestbot.calibration.intrinsics import factory_intrinsic_profile
from posetestbot.calibration.rectification import rectify_run
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


def create_synchronized_sensor_fixture(tmp_path: Path) -> Path:
    run_root = tmp_path / "run-1"
    sensor = run_root / "processed" / "synchronized" / "realsense_123"
    rgb = sensor / RGB_DIR
    depth = sensor / DEPTH_DIR
    rgb.mkdir(parents=True)
    depth.mkdir()
    for frame_id, value in ((10, 1), (20, 2)):
        assert cv2.imwrite(
            (rgb / f"{frame_id:06d}.png").as_posix(),
            np.full((5, 6, 3), value, dtype=np.uint8),
        )
        assert cv2.imwrite(
            (depth / f"{frame_id:06d}.png").as_posix(),
            np.full((5, 6), value, dtype=np.uint16),
        )
    (sensor / CAM_K).write_text("1 0 2\n0 3 4\n0 0 1\n")
    (sensor / DEPTH_SCALE).write_text("0.001\n")
    return run_root


def export_command(run_root: Path) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    return [
        sys.executable,
        str(repo_root / "scripts" / "run_bop_export_stage.py"),
        str(run_root),
    ]


def create_hardware_sync_fixture(tmp_path: Path) -> Path:
    run_root = tmp_path / "hardware-run"
    sensors = (
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
    )
    config = create_run_config(
        run_root=run_root,
        sensors=sensors,
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
    qualification_evidence = tmp_path / "hardware-sync-pulse.csv"
    qualification_evidence.write_text("time_ns,master,hand\n0,1,1\n")
    record_hardware_sync_qualification(
        run_root,
        operator="researcher@example.test",
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
    (run_root / CAPTURE_EXECUTION_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "capture_execution_report.v1",
                "run_root": run_root.as_posix(),
                "status": "succeeded",
                "mode": "full",
                "allow_cameras": True,
                "allow_real_robot": True,
                "hardware_sync_execution_binding": execution_binding,
            }
        )
    )
    for device_id, role, mounting_mode, timestamp_ns in (
        ("master", "master", "static", 1_000_000_000),
        ("hand", "subordinate", "eye_in_hand", 1_000_100_000),
    ):
        raw = run_root / f"realsense_{device_id}"
        synchronized = (
            run_root
            / "processed"
            / "synchronized"
            / f"realsense_{device_id}"
        )
        for folder in (raw, synchronized):
            (folder / RGB_DIR).mkdir(parents=True)
            (folder / DEPTH_DIR).mkdir()
        rgb = np.full((5, 6, 3), 10, dtype=np.uint8)
        depth = np.full((5, 6), 100, dtype=np.uint16)
        assert cv2.imwrite((raw / RGB_DIR / "1000.png").as_posix(), rgb)
        assert cv2.imwrite((raw / DEPTH_DIR / "1000.png").as_posix(), depth)
        assert cv2.imwrite(
            (synchronized / RGB_DIR / "000000.png").as_posix(), rgb
        )
        assert cv2.imwrite(
            (synchronized / DEPTH_DIR / "000000.png").as_posix(), depth
        )
        (synchronized / CAM_K).write_text(
            "10 0 3\n0 10 2.5\n0 0 1\n0 0 0 0 0\n"
        )
        (synchronized / DEPTH_SCALE).write_text("1.0\n")
        (synchronized / CAMERA_DATA_JSON).write_text(
            json.dumps(
                {
                    "K": [[10, 0, 3], [0, 10, 2.5], [0, 0, 1]],
                    "resolution": [5, 6],
                    "orientation": "normal",
                    "distortion": [0.0] * 5,
                    "distortion_model": "brown_conrady",
                }
            )
        )
        option = 1 if role == "master" else 2
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
            "inter_cam_sync_mode_configured": option,
            "inter_cam_sync_mode_readback": option,
            "matched_robot_pose_index": 20,
            "nearest_robot_delta_ns": 50,
            "motion": "capture",
            "mounting_mode": mounting_mode,
        }
        (synchronized / FRAME_METADATA_JSONL).write_text(
            json.dumps(metadata) + "\n"
        )
        (synchronized / MATCH_ROBOT_EE_POSES).write_text(
            json.dumps(
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
                }
            )
        )
    groups = build_hardware_sync_frame_groups(run_root)
    groups["hardware_sync_qualification"] = qualification
    groups["hardware_sync_execution_binding"] = execution_binding
    write_hardware_sync_frame_groups(run_root, groups)
    return run_root


def test_bop_export_stage_writes_objectless_dataset_and_manifest(
    tmp_path: Path,
) -> None:
    run_root = create_synchronized_sensor_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [*export_command(run_root), "--write-coco-annotations"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Exported 1 synchronized sensor folder" in result.stdout
    bop = run_root / BOP_DIR
    scene = bop / "test" / "000001"
    manifest = json.loads((bop / BOP_EXPORT_MANIFEST).read_text())
    assert manifest["schema_version"] == "bop_export_manifest.v4"
    assert manifest["dataset_mode"] == "objectless"
    assert manifest["objectless"] is True
    assert manifest["object_models"] == []
    assert manifest["stable_id_mapping"] == {}
    assert json.loads((bop / BOP_TARGETS_BOP19).read_text()) == []
    assert (bop / BOP_FRAME_MAP_JSON).is_file()
    assert len(list((scene / RGB_DIR).glob("*.png"))) == 2
    assert len(list((scene / DEPTH_DIR).glob("*.png"))) == 2
    assert all(
        rows == []
        for rows in json.loads((scene / "scene_gt.json").read_text()).values()
    )
    coco = json.loads((bop / BOP_COCO_ANNOTATIONS).read_text())
    assert coco["images"]
    assert coco["categories"] == []
    assert coco["annotations"] == []
    run_manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        item for item in run_manifest["stages"] if item["name"] == "bop_export"
    )
    assert stage["status"] == "succeeded"


def test_bop_export_default_ignores_disabled_stale_sensor_folder(
    tmp_path: Path,
) -> None:
    run_root = create_synchronized_sensor_fixture(tmp_path)
    synchronized = run_root / "processed" / "synchronized"
    shutil.copytree(synchronized / "realsense_123", synchronized / "realsense_999")
    write_run_config(
        run_root,
        create_run_config(
            run_root=run_root,
            sensors=(
                SensorRunConfig("realsense_d435", "123", "Enabled"),
                SensorRunConfig("realsense_d435", "999", "Disabled", enabled=False),
            ),
        ),
    )
    repo_root = Path(__file__).resolve().parents[1]

    default_result = subprocess.run(
        export_command(run_root),
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Exported 1 synchronized sensor folder" in default_result.stdout
    default_manifest = json.loads(
        (run_root / BOP_DIR / BOP_EXPORT_MANIFEST).read_text()
    )
    assert [item["sensor_name"] for item in default_manifest["exports"]] == [
        "realsense_123"
    ]

    explicit_output = run_root / "explicit-bop"
    explicit_result = subprocess.run(
        [
            *export_command(run_root),
            "--input-folder",
            str(synchronized),
            "--output-folder",
            str(explicit_output),
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Exported 2 synchronized sensor folder" in explicit_result.stdout
    explicit_manifest = json.loads((explicit_output / BOP_EXPORT_MANIFEST).read_text())
    assert [item["sensor_name"] for item in explicit_manifest["exports"]] == [
        "realsense_123",
        "realsense_999",
    ]


def test_bop_export_objectless_rejects_stale_object_gt(tmp_path: Path) -> None:
    run_root = create_synchronized_sensor_fixture(tmp_path)
    output = (
        run_root
        / "processed"
        / "synchronized"
        / "realsense_123"
        / "blenderproc"
        / "output"
    )
    output.mkdir(parents=True)
    (output / "scene_gt.json").write_text(
        json.dumps(
            {
                "0": [
                    {
                        "obj_id": 1,
                        "cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        "cam_t_m2c": [0, 0, 1],
                    }
                ],
                "1": [],
            }
        )
    )
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        export_command(run_root),
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "Unknown BOP obj_id" in result.stderr
    assert not (run_root / BOP_DIR).exists()


def test_bop_overwrite_failure_preserves_previous_dataset(tmp_path: Path) -> None:
    run_root = create_synchronized_sensor_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    command = export_command(run_root)
    subprocess.run(command, cwd=repo_root, check=True, capture_output=True, text=True)
    manifest_path = run_root / BOP_DIR / BOP_EXPORT_MANIFEST
    previous_manifest = manifest_path.read_bytes()

    sensor = run_root / "processed" / "synchronized" / "realsense_123"
    (sensor / DEPTH_DIR / "000020.png").unlink()
    failed = subprocess.run(
        [*command, "--overwrite"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert failed.returncode != 0
    assert manifest_path.read_bytes() == previous_manifest
    assert (run_root / BOP_DIR / "test" / "000001" / RGB_DIR / "000001.png").is_file()
    assert not list(run_root.glob(".bop.*.tmp"))


def test_hardware_bop_export_records_native_authoritative_source_truth(
    tmp_path: Path,
) -> None:
    run_root = create_hardware_sync_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        export_command(run_root),
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "authoritative hardware-synchronized frame set" in result.stdout
    bop = run_root / BOP_DIR
    manifest = json.loads((bop / BOP_EXPORT_MANIFEST).read_text())
    assert {item["projection"] for item in manifest["exports"]} == {"native"}
    frame_map = json.loads((bop / BOP_FRAME_MAP_JSON).read_text())
    for scene in frame_map["scenes"].values():
        assert scene["projection"] == "native"
        assert scene["input_sensor_folder"] == (
            scene["authoritative_source_sensor_folder"]
        )
        frame = scene["frames"]["0"]
        assert frame["projection"] == "native"
        assert frame["source_rgb"] == frame["authoritative_source_rgb"]
    frame_sets = json.loads((bop / BOP_FRAME_SETS).read_text())
    assert frame_sets["hardware_sync_qualification"]["status"] == "passed"
    assert frame_sets["hardware_sync_execution_binding"] == json.loads(
        (run_root / CAPTURE_EXECUTION_REPORT).read_text()
    )["hardware_sync_execution_binding"]
    for view in frame_sets["frame_sets"][0]["views"]:
        assert view["projection"] == "native"
        assert view["bop_input_sensor_folder"] == view[
            "authoritative_source_sensor_folder"
        ]
        assert view["bop_input_rgb_path"] == view[
            "authoritative_source_rgb_path"
        ]


def test_hardware_bop_export_revalidates_groups_immediately_before_publication(
    tmp_path: Path,
) -> None:
    run_root = create_hardware_sync_fixture(tmp_path)
    synchronized_root = run_root / "processed" / "synchronized"
    for sensor_name in ("realsense_master", "realsense_hand"):
        sensor = synchronized_root / sensor_name
        for frame_index in range(1, 301):
            frame_name = f"{frame_index:06d}.png"
            shutil.copy2(
                sensor / RGB_DIR / "000000.png",
                sensor / RGB_DIR / frame_name,
            )
            shutil.copy2(
                sensor / DEPTH_DIR / "000000.png",
                sensor / DEPTH_DIR / frame_name,
            )

    repo_root = Path(__file__).resolve().parents[1]
    process = subprocess.Popen(
        export_command(run_root),
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 10.0
    while not list(run_root.glob(".bop.*.tmp")):
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(
                "BOP export exited before its staging directory could prove "
                f"initial provenance was loaded: stdout={stdout!r}, "
                f"stderr={stderr!r}"
            )
        if time.monotonic() >= deadline:
            process.kill()
            process.communicate()
            pytest.fail("Timed out waiting for BOP export staging")
        time.sleep(0.001)

    groups_path = synchronized_root / MULTIVIEW_FRAME_GROUPS
    groups = json.loads(groups_path.read_text())
    groups["hardware_sync_execution_binding"][
        "qualification_artifact_sha256"
    ] = "0" * 64
    groups_path.write_text(json.dumps(groups))

    stdout, stderr = process.communicate(timeout=30)

    assert process.returncode != 0, stdout
    assert (
        "hardware_sync_execution_binding does not exactly match" in stderr
        or "hardware-sync frame groups changed" in stderr.lower()
    )
    assert not (run_root / BOP_DIR).exists()


def test_hardware_bop_export_rejects_arbitrary_explicit_input_root(
    tmp_path: Path,
) -> None:
    run_root = create_hardware_sync_fixture(tmp_path)
    arbitrary = run_root / "arbitrary-copy"
    shutil.copytree(run_root / "processed" / "synchronized", arbitrary)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            *export_command(run_root),
            "--input-folder",
            str(arbitrary),
            "--output-folder",
            str(run_root / "rejected-bop"),
        ],
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "must be the canonical" in result.stderr
    assert not (run_root / "rejected-bop").exists()


def test_hardware_bop_export_accepts_current_rectification_and_records_projection(
    tmp_path: Path,
) -> None:
    run_root = create_hardware_sync_fixture(tmp_path)
    synchronized = run_root / "processed" / "synchronized"
    profiles = [
        factory_intrinsic_profile(synchronized / sensor_name)
        for sensor_name in ("realsense_master", "realsense_hand")
    ]
    rectify_run(run_root, profiles)
    repo_root = Path(__file__).resolve().parents[1]

    subprocess.run(
        export_command(run_root),
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    bop = run_root / BOP_DIR
    manifest = json.loads((bop / BOP_EXPORT_MANIFEST).read_text())
    assert {item["projection"] for item in manifest["exports"]} == {"rectified"}
    frame_map = json.loads((bop / BOP_FRAME_MAP_JSON).read_text())
    for scene in frame_map["scenes"].values():
        assert scene["projection"] == "rectified"
        assert scene["input_sensor_folder"].startswith("processed/rectified/")
        assert scene["authoritative_source_sensor_folder"].startswith(
            "processed/synchronized/"
        )
        assert scene["input_fingerprint_sha256"] != (
            scene["authoritative_source_fingerprint_sha256"]
        )
    frame_sets = json.loads((bop / BOP_FRAME_SETS).read_text())
    for view in frame_sets["frame_sets"][0]["views"]:
        assert view["projection"] == "rectified"
        assert view["bop_input_sensor_folder"].startswith(
            "processed/rectified/"
        )
        assert view["sensor_folder"].startswith("processed/synchronized/")


def test_hardware_bop_export_rejects_stale_or_mutated_rectification(
    tmp_path: Path,
) -> None:
    run_root = create_hardware_sync_fixture(tmp_path)
    synchronized = run_root / "processed" / "synchronized"
    profiles = [
        factory_intrinsic_profile(synchronized / sensor_name)
        for sensor_name in ("realsense_master", "realsense_hand")
    ]
    rectify_run(run_root, profiles)
    mutated = (
        run_root
        / "processed"
        / "rectified"
        / "realsense_hand"
        / RGB_DIR
        / "000000.png"
    )
    mutated.write_bytes(mutated.read_bytes() + b"mutated")
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        export_command(run_root),
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "output fingerprint is stale or mismatched" in result.stderr
    assert not (run_root / BOP_DIR).exists()


def test_hardware_bop_export_binds_groups_to_current_mounting_inventory(
    tmp_path: Path,
) -> None:
    run_root = create_hardware_sync_fixture(tmp_path)
    write_run_config(
        run_root,
        create_run_config(
            run_root=run_root,
            sensors=(
                SensorRunConfig(
                    "realsense_d435",
                    "master",
                    "Static master",
                    mounting_mode="eye_in_hand",
                ),
                SensorRunConfig(
                    "realsense_d435",
                    "hand",
                    "Robot-mounted subordinate",
                    mounting_mode="static",
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
        ),
    )
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        export_command(run_root),
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert (
        "mounting_mode" in result.stderr
        or "hardware contract changed" in result.stderr
    )
    assert not (run_root / BOP_DIR).exists()


def test_hardware_bop_export_rejects_group_with_other_qualification_provenance(
    tmp_path: Path,
) -> None:
    run_root = create_hardware_sync_fixture(tmp_path)
    groups_path = (
        run_root
        / "processed"
        / "synchronized"
        / MULTIVIEW_FRAME_GROUPS
    )
    groups = json.loads(groups_path.read_text())
    groups["hardware_sync_qualification"]["artifact_sha256"] = "0" * 64
    groups_path.write_text(json.dumps(groups))
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        export_command(run_root),
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "hardware_sync_qualification does not exactly match" in result.stderr
    assert not (run_root / BOP_DIR).exists()
