from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from posetestbot.io.artifacts import DATASET_MANIFEST, RUN_CONFIG
from posetestbot.pipeline.run_config import (
    FixedFrameTransform,
    SCHEMA_VERSION,
    build_sequence_job_from_run_config,
    create_run_config,
    load_run_config_for_run_root,
    sensor_config_from_mapping,
    sensor_configs_from_status,
    sensor_config_from_token,
    sequence_plan_from_run_config,
    write_run_config,
)


def test_default_run_config_uses_fake_robot_and_lab_sensors(tmp_path: Path) -> None:
    run_root = tmp_path / "run-1"

    config = create_run_config(run_root=run_root)
    data = config.to_dict()

    assert data["schema_version"] == SCHEMA_VERSION
    assert data["robot_profile"]["mode"] == "fake"
    assert data["robot_profile"]["robot_ip"] == "127.0.0.1"
    assert data["object_folder"] == "object_models"
    assert data["pipeline"]["sequence_id"] == "sync_to_bop_dry_run"
    assert data["pipeline"]["plan_only"] is True
    assert data["frames"]["robot_pose"] == {
        "from": "robot_flange",
        "to": "template_base",
        "convention": "kuka_abc_radians",
    }
    assert data["frames"]["dataset_reference_frame"] == "template_base"
    sensors = data["capture"]["sensors"]
    assert [sensor["sensor_type"] for sensor in sensors].count("realsense_d435") == 3
    assert [sensor["sensor_type"] for sensor in sensors].count("oak_d_pro") == 1
    assert [sensor["sensor_type"] for sensor in sensors].count("zed_2i") == 1
    assert all(sensor["inverted"] is False for sensor in sensors)

    plan = sequence_plan_from_run_config(data)

    assert plan.sequence_id == "sync_to_bop_dry_run"
    assert plan.plan_only is True
    assert plan.steps[0].command[:3] == ["uv", "run", "python"]


def test_sensor_config_token_accepts_alias_and_mounting_mode() -> None:
    sensor = sensor_config_from_token("luxonis:mxid-1:static:Cell OAK-D Pro")

    assert sensor.sensor_type == "oak_d_pro"
    assert sensor.device_id == "mxid-1"
    assert sensor.mounting_mode == "static"
    assert sensor.display_name == "Cell OAK-D Pro"
    assert sensor.inverted is False


def test_sensor_config_accepts_realsense_inverted_orientation() -> None:
    token_sensor = sensor_config_from_token(
        "realsense:123:static:Cell RealSense:inverted"
    )
    mapping_sensor = sensor_config_from_mapping(
        {
            "sensor_type": "realsense",
            "device_id": "456",
            "mounting_mode": "eye_in_hand",
            "display_name": "Wrist RealSense",
            "inverted": "true",
        }
    )

    assert token_sensor.sensor_type == "realsense_d435"
    assert token_sensor.inverted is True
    assert mapping_sensor.device_id == "456"
    assert mapping_sensor.inverted is True


def test_sensor_configs_from_status_uses_alias_defaults() -> None:
    sensors = sensor_configs_from_status(
        {
            "families": [
                {
                    "devices": [
                        {
                            "sensor_type": "realsense_d435",
                            "device_id": "123",
                            "display_name": "Intel RealSense 123",
                            "effective_display_name": "Wrist Camera",
                            "mounting_mode": "eye_in_hand",
                            "inverted": True,
                            "metadata": {"model": "D435"},
                        }
                    ]
                }
            ]
        }
    )

    assert len(sensors) == 1
    assert sensors[0].device_id == "123"
    assert sensors[0].display_name == "Wrist Camera"
    assert sensors[0].inverted is True
    assert sensors[0].metadata == {"model": "D435"}


def test_sensor_config_rejects_non_realsense_inverted_orientation() -> None:
    with pytest.raises(ValueError, match="only supported for RealSense"):
        sensor_config_from_token("oak:auto:static:Cell OAK-D Pro:inverted")

    with pytest.raises(ValueError, match="only supported for RealSense"):
        sensor_config_from_mapping(
            {
                "sensor_type": "zed_2i",
                "device_id": "auto",
                "mounting_mode": "static",
                "display_name": "Cell ZED 2i",
                "inverted": True,
            }
        )


def test_sensor_config_token_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unknown sensor type"):
        sensor_config_from_token("webcam:0")


def test_run_config_rejects_empty_object_folder(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="object_folder"):
        create_run_config(run_root=tmp_path / "run-object", object_folder="")


def test_run_config_loads_from_run_root_and_builds_sequence_job(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-job"
    config = create_run_config(
        run_root=run_root,
        sequence_id="sync_aruco",
        sequence_options={"aruco": {"save_images": True}},
    )
    write_run_config(run_root, config)

    loaded = load_run_config_for_run_root(run_root)
    job = build_sequence_job_from_run_config(loaded)

    assert loaded["run_root"] == run_root.as_posix()
    assert job.sequence_id == "sync_aruco"
    assert job.command[:4] == [
        "uv",
        "run",
        "python",
        "scripts/run_pipeline_sequence.py",
    ]
    assert "--plan-only" in job.command
    assert job.parameters["options"] == {"aruco": {"save_images": True}}


def test_run_config_calibration_profiles_flow_to_calibrated_sequence(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-calibrated"
    profiles_path = run_root / "profiles" / "lab_profiles.json"
    config = create_run_config(
        run_root=run_root,
        sequence_id="sync_to_bop_calibrated_dry_run",
        calibration_profiles=profiles_path.as_posix(),
    )

    plan = sequence_plan_from_run_config(config.to_dict())
    prepare_step = next(
        step for step in plan.steps if step.id == "blenderproc_prepare"
    )
    export_step = next(step for step in plan.steps if step.id == "bop_export")

    assert prepare_step.options["calibration_profiles"] == profiles_path.as_posix()
    assert export_step.options["calibration_profiles"] == profiles_path.as_posix()
    assert profiles_path.as_posix() in prepare_step.command
    assert profiles_path.as_posix() in export_step.command


def test_create_run_config_cli_writes_config_manifest_and_plan(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-cli"
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/create_run_config.py",
            run_root.as_posix(),
            "--robot-mode",
            "fake",
            "--sensor",
            "realsense:123:static:Cell RealSense",
            "--object-folder",
            "custom_object_models",
            "--sequence",
            "sync_aruco",
            "--sequence-options-json",
            '{"aruco": {"save_images": true}}',
            "--print-sequence-plan",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert f"Wrote {run_root / RUN_CONFIG}" in result.stdout
    assert '"sequence_id": "sync_aruco"' in result.stdout
    config = json.loads((run_root / RUN_CONFIG).read_text())
    assert config["capture"]["sensors"] == [
        {
            "calibration_profile_id": None,
            "device_id": "123",
            "display_name": "Cell RealSense",
            "enabled": True,
            "inverted": False,
            "metadata": {},
            "mounting_mode": "static",
            "sensor_type": "realsense_d435",
        }
    ]
    assert config["object_folder"] == "custom_object_models"
    assert config["pipeline"]["options"] == {"aruco": {"save_images": True}}
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(stage for stage in manifest["stages"] if stage["name"] == "run_config")
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][RUN_CONFIG] == RUN_CONFIG


def test_create_run_config_cli_writes_real_full_capture_validation_plan(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "real-full-capture-cli"
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/create_run_config.py",
            run_root.as_posix(),
            "--robot-mode",
            "real",
            "--sequence",
            "real_full_capture_validation",
            "--print-sequence-plan",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    config = json.loads((run_root / RUN_CONFIG).read_text())

    assert config["robot_profile"]["mode"] == "real"
    assert config["pipeline"]["sequence_id"] == "real_full_capture_validation"
    assert config["pipeline"]["plan_only"] is True
    assert '"sequence_id": "real_full_capture_validation"' in result.stdout
    assert "scripts/run_capture_execution_stage.py" in result.stdout
    assert "--allow-cameras" in result.stdout
    assert "--allow-real-robot" in result.stdout
    assert "scripts/run_rewrite_gate.py" in result.stdout
    assert "rewrite_full_capture.v1" in result.stdout


def test_create_run_config_cli_lists_sequence_choices() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/create_run_config.py",
            "--help",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "capture_to_bop_dataset_dry_run" in result.stdout
    assert "fake_capture_to_bop_dataset_dry_run" in result.stdout
    assert "real_full_capture_validation" in result.stdout
    assert "foundationpose_runtime_to_bop_eval" not in result.stdout
    assert "sam6d_runtime_to_bop_eval" not in result.stdout


def test_create_run_config_cli_rejects_unknown_sequence(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/create_run_config.py",
            (tmp_path / "run-bad-sequence").as_posix(),
            "--sequence",
            "not_a_sequence",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr
    assert not (tmp_path / "run-bad-sequence" / RUN_CONFIG).exists()


def test_legacy_run_config_loads_with_frame_warning(tmp_path: Path) -> None:
    run_root = tmp_path / "legacy-run"
    value = create_run_config(run_root=run_root).to_dict()
    value.pop("frames")
    run_root.mkdir()
    (run_root / RUN_CONFIG).write_text(json.dumps(value))

    loaded = load_run_config_for_run_root(run_root)

    assert loaded["frames"]["robot_pose"]["from"] == "robot_flange"
    assert loaded["frames"]["robot_pose"]["to"] == "template_base"
    assert loaded["warnings"][0]["code"] == "legacy_frames_inferred"


def test_create_run_config_records_typed_fixed_frame_edges(tmp_path: Path) -> None:
    config = create_run_config(
        run_root=tmp_path / "fixed-run",
        fixed_transforms=(
            FixedFrameTransform(
                from_frame="robot_flange",
                to_frame="tcp",
                rotation_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                translation_mm=(0.0, 0.0, 125.0),
                source="tool_measurement",
            ),
        ),
    ).to_dict()

    assert config["frames"]["fixed_transforms"] == [
        {
            "from": "robot_flange",
            "to": "tcp",
            "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "translation_mm": [0.0, 0.0, 125.0],
            "source": "tool_measurement",
        }
    ]
