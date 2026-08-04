from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from posetestbot.io.artifacts import CAPTURE_PLAN, DATASET_MANIFEST, RUN_CONFIG
from posetestbot.pipeline.capture_plan import build_capture_plan
from posetestbot.pipeline.run_config import (
    create_run_config,
    sensor_config_from_token,
    write_run_config,
)


def test_capture_plan_builds_sensor_commands_then_one_receiver(tmp_path: Path) -> None:
    run_root = tmp_path / "run-capture-plan"
    sensors = (
        sensor_config_from_token("realsense:123:eye_in_hand:Cell RealSense:inverted"),
        sensor_config_from_token("luxonis:auto:eye_in_hand:Cell OAK-D Pro"),
        sensor_config_from_token("zed:auto:static:Cell ZED 2i"),
    )
    config = create_run_config(
        run_root=run_root,
        sensors=sensors,
        fps=12,
        velocity_m_s=0.15,
    ).to_dict()

    plan = build_capture_plan(config, max_frames=5, warmup_frames=30).to_dict()

    assert plan["schema_version"] == "capture_plan.v1"
    assert plan["dry_run"] is True
    assert plan["capture"]["requested_velocity_m_s"] == 0.15
    assert plan["capture"]["velocity_m_s"] == 0.03
    assert plan["capture"]["command_velocity_cap_m_s"] == 0.03
    assert any(
        "0.15 m/s is reduced to the host command cap 0.03 m/s" in note
        for note in plan["notes"]
    )
    assert plan["capture"]["enabled_sensor_count"] == 3
    assert plan["capture"]["warmup_frames"] == 30
    assert [sensor["folder"] for sensor in plan["sensors"]] == [
        "realsense_123",
        "luxonis_auto",
        "zed_2i_auto",
    ]
    assert [sensor["operator_alias"] for sensor in plan["sensors"]] == [
        "Cell RealSense",
        "Cell OAK-D Pro",
        "Cell ZED 2i",
    ]
    assert [command["role"] for command in plan["commands"]] == [
        "sensor_capture",
        "sensor_capture",
        "sensor_capture",
        "robot_pose_receiver",
    ]

    realsense = plan["commands"][0]
    assert realsense["command"] == [
        "uv",
        "run",
        "python",
        "scripts/capture_realsense_720p.py",
        (run_root / "realsense_123").as_posix(),
        "--fps",
        "12",
        "--max_frames",
        "5",
        "--warmup-frames",
        "30",
        "--device",
        "123",
        "--inverted",
    ]
    assert plan["sensors"][0]["metadata"]["inverted"] is True
    assert plan["sensors"][0]["metadata"]["image_rotation_degrees"] == 180

    luxonis = plan["commands"][1]
    assert luxonis["command"] == [
        "uv",
        "run",
        "python",
        "scripts/capture_luxonis_720p.py",
        (run_root / "luxonis_auto").as_posix(),
        "--fps",
        "12",
        "--max_frames",
        "5",
        "--warmup-frames",
        "30",
    ]
    assert "--inverted" not in luxonis["command"]

    zed = plan["commands"][2]
    assert zed["command"][-2:] == ["--resolution", "720p"]
    assert "--device" not in zed["command"]
    assert "--inverted" not in zed["command"]

    receiver = plan["commands"][-1]
    assert receiver["command"][:4] == [
        "uv",
        "run",
        "python",
        "scripts/pose_receiver_udp_json.py",
    ]
    assert "--robot_mode" not in receiver["command"]
    assert "--allow-cameras" not in receiver["command"]
    assert "--allow-real-robot" not in receiver["command"]
    assert "--receive-start-timeout-s" not in receiver["command"]
    assert "--receive-idle-timeout-s" not in receiver["command"]
    velocity_index = receiver["command"].index("--capture_vel")
    assert receiver["command"][velocity_index + 1] == "0.03"


def test_object_dataset_plan_passes_extended_speed_over_structured_protocol(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-extended-dataset-speed"
    config = create_run_config(
        run_root=run_root,
        dataset_mode="pose_template",
        sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
        velocity_m_s=0.15,
    ).to_dict()

    plan = build_capture_plan(config).to_dict()

    assert plan["capture"]["requested_velocity_m_s"] == 0.15
    assert plan["capture"]["velocity_m_s"] == 0.15
    assert plan["capture"]["command_velocity_cap_m_s"] == 1.0
    assert plan["capture"]["command_protocol"] == "v1"
    assert not any("reduced to the host command cap" in note for note in plan["notes"])
    assert any("structured robot_command.v1 protocol" in note for note in plan["notes"])
    receiver = plan["commands"][-1]["command"]
    assert receiver[receiver.index("--capture_vel") + 1] == "0.15"
    assert receiver[receiver.index("--protocol") + 1] == "v1"
    assert receiver[
        receiver.index("--maximum-command-velocity-m-s") + 1
    ] == "1.0"


def test_capture_plan_uses_adapter_resolution_validation(tmp_path: Path) -> None:
    run_root = tmp_path / "run-bad-resolution"
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
        resolution="360p",
    ).to_dict()

    with pytest.raises(ValueError, match="RealSense D435"):
        build_capture_plan(config)


def test_capture_plan_excludes_disabled_sensor_without_deleting_identity(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-disabled-camera"
    config = create_run_config(
        run_root=run_root,
        sensors=(
            sensor_config_from_token("realsense:working:eye_in_hand:Working"),
            sensor_config_from_token("realsense:offline:eye_in_hand:Offline"),
        ),
    ).to_dict()
    config["capture"]["sensors"][1]["enabled"] = False

    plan = build_capture_plan(config).to_dict()

    assert len(config["capture"]["sensors"]) == 2
    assert config["capture"]["sensors"][1]["device_id"] == "offline"
    assert plan["capture"]["sensor_count"] == 2
    assert plan["capture"]["enabled_sensor_count"] == 1
    assert [sensor["device_id"] for sensor in plan["sensors"]] == ["working"]
    assert [command["role"] for command in plan["commands"]] == [
        "sensor_capture",
        "robot_pose_receiver",
    ]


def test_capture_plan_rejects_negative_warmup_frames(tmp_path: Path) -> None:
    run_root = tmp_path / "run-bad-warmup"
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
    ).to_dict()

    with pytest.raises(ValueError, match="warmup_frames"):
        build_capture_plan(config, warmup_frames=-1)


def test_capture_plan_treats_string_false_inverted_as_normal(tmp_path: Path) -> None:
    run_root = tmp_path / "run-string-false"
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
    ).to_dict()
    config["capture"]["sensors"][0]["inverted"] = "false"

    plan = build_capture_plan(config).to_dict()

    assert "--inverted" not in plan["commands"][0]["command"]
    assert plan["sensors"][0]["metadata"]["inverted"] is False


def test_capture_plan_assigns_master_first_for_mixed_mount_hardware_sync(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-hardware-sync"
    config = create_run_config(
        run_root=run_root,
        sensors=(
            sensor_config_from_token(
                "realsense:static-1:static:Static RealSense"
            ),
            sensor_config_from_token(
                "realsense:hand-1:eye_in_hand:Robot RealSense"
            ),
        ),
        synchronization={
            "schema_version": "capture_synchronization.v1",
            "mode": "hardware_trigger",
            "implementation": "realsense_inter_cam_sync",
            "scope": "depth_exposure",
            "group_id": "mixed-rig-1",
            "master_sensor_key": "realsense_d435:hand-1",
            "max_depth_timestamp_skew_ms": 2.0,
        },
    ).to_dict()

    plan = build_capture_plan(config, warmup_frames=30).to_dict()

    assert plan["capture"]["synchronization"]["mode"] == "hardware_trigger"
    cameras = plan["commands"][:-1]
    assert [command["device_id"] for command in cameras] == [
        "hand-1",
        "static-1",
    ]
    assert [command["startup_order"] for command in cameras] == [20, 21]
    assert [command["hardware_sync"]["role"] for command in cameras] == [
        "master",
        "subordinate",
    ]
    assert cameras[0]["hardware_sync"]["inter_cam_sync_mode_expected"] == 1
    assert cameras[1]["hardware_sync"]["inter_cam_sync_mode_expected"] == 2
    assert all(
        command["hardware_sync"]["depth_exposure_synchronized"] is True
        and command["hardware_sync"]["rgb_exposure_synchronized"] is False
        for command in cameras
    )
    assert cameras[0]["command"][-6:] == [
        "--hardware-sync-role",
        "master",
        "--hardware-sync-group-id",
        "mixed-rig-1",
        "--hardware-sync-scope",
        "depth_exposure",
    ]


def test_capture_plan_rejects_finite_hardware_sync_capture(tmp_path: Path) -> None:
    run_root = tmp_path / "run-finite-hardware-sync"
    config = create_run_config(
        run_root=run_root,
        sensors=(
            sensor_config_from_token("realsense:master:static:Master"),
            sensor_config_from_token("realsense:slave:eye_in_hand:Slave"),
        ),
        synchronization={
            "schema_version": "capture_synchronization.v1",
            "mode": "hardware_trigger",
            "implementation": "realsense_inter_cam_sync",
            "scope": "depth_exposure",
            "group_id": "finite-test",
            "master_sensor_key": "realsense_d435:master",
            "max_depth_timestamp_skew_ms": 2.0,
        },
    ).to_dict()

    with pytest.raises(ValueError, match="must be long-running"):
        build_capture_plan(config, max_frames=100)


def test_capture_plan_stage_writes_manifest_artifact(tmp_path: Path) -> None:
    run_root = tmp_path / "run-cli"
    repo_root = Path(__file__).resolve().parents[1]
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
        sequence_id="sync_aruco",
    )
    write_run_config(run_root, config)

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_capture_plan_stage.py",
            run_root.as_posix(),
            "--max-frames",
            "2",
            "--warmup-frames",
            "3",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert f"Wrote {run_root / CAPTURE_PLAN}" in result.stdout
    assert "capture_realsense_720p.py" in result.stdout

    plan = json.loads((run_root / CAPTURE_PLAN).read_text())
    assert plan["capture"]["max_frames"] == 2
    assert plan["capture"]["warmup_frames"] == 3
    assert plan["commands"][0]["command"][-6:] == [
        "--max_frames",
        "2",
        "--warmup-frames",
        "3",
        "--device",
        "123",
    ]

    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(stage for stage in manifest["stages"] if stage["name"] == "capture_plan")
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][CAPTURE_PLAN] == CAPTURE_PLAN
    assert manifest["artifacts"] == {}
    assert manifest["sensors"][0]["status"] == "planned"
    assert manifest["sensors"][0]["folder"] == "realsense_123"
    assert manifest["sensors"][0]["display_name"] == "Cell RealSense"
    assert manifest["sensors"][0]["operator_alias"] == "Cell RealSense"
    assert (run_root / RUN_CONFIG).is_file()
