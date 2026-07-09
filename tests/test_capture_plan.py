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


def test_capture_plan_builds_fake_first_uv_commands(tmp_path: Path) -> None:
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
    assert plan["capture"]["enabled_sensor_count"] == 3
    assert plan["capture"]["warmup_frames"] == 30
    assert [sensor["folder"] for sensor in plan["sensors"]] == [
        "realsense_123",
        "luxonis_auto",
        "zed_2i_auto",
    ]
    assert [command["role"] for command in plan["commands"]] == [
        "robot_controller",
        "sensor_capture",
        "sensor_capture",
        "sensor_capture",
        "robot_pose_receiver",
    ]

    fake_controller = plan["commands"][0]
    assert fake_controller["command"][:4] == [
        "uv",
        "run",
        "python",
        "iiwa/fake_iiwa_controller.py",
    ]
    assert "--once" in fake_controller["command"]

    realsense = plan["commands"][1]
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

    luxonis = plan["commands"][2]
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

    zed = plan["commands"][3]
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
    assert "--robot_mode" in receiver["command"]
    assert "fake" in receiver["command"]


def test_capture_plan_uses_adapter_resolution_validation(tmp_path: Path) -> None:
    run_root = tmp_path / "run-bad-resolution"
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
        resolution="360p",
    ).to_dict()

    with pytest.raises(ValueError, match="RealSense D435"):
        build_capture_plan(config)


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

    assert "--inverted" not in plan["commands"][1]["command"]
    assert plan["sensors"][0]["metadata"]["inverted"] is False


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
    assert plan["commands"][1]["command"][-6:] == [
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
    assert (run_root / RUN_CONFIG).is_file()
