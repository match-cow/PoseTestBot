from __future__ import annotations

import json

import subprocess

from pathlib import Path

import pytest

from posetestbot.io.artifacts import DATASET_MANIFEST, RUN_CONFIG

from posetestbot.config import DEFAULT_CAPTURE_VELOCITY_M_S

from posetestbot.pipeline.run_config import (
    CAPTURE_SYNCHRONIZATION_SCHEMA_VERSION,
    FixedFrameTransform,
    SCHEMA_VERSION,
    build_sequence_job_from_run_config,
    create_run_config,
    load_run_config_for_run_root,
    sensor_config_from_mapping,
    sensor_configs_from_status,
    sensor_config_from_token,
    sequence_plan_from_run_config,
    validate_run_config,
    write_run_config,
)


def test_default_run_config_uses_real_robot_and_lab_sensors(tmp_path: Path) -> None:
    run_root = tmp_path / "run-1"

    config = create_run_config(run_root=run_root)
    data = config.to_dict()

    assert data["schema_version"] == SCHEMA_VERSION
    assert data["robot_profile"]["mode"] == "real"
    assert data["robot_profile"]["robot_ip"] == "172.31.1.147"
    assert (
        data["robot_profile"]["cartesian_velocity_m_s"] == DEFAULT_CAPTURE_VELOCITY_M_S
    )
    assert data["capture"]["velocity_m_s"] == DEFAULT_CAPTURE_VELOCITY_M_S
    assert data["dataset_mode"] == "objectless"
    assert "object_folder" not in data
    assert "selected_objects" not in data
    assert data["pipeline"]["sequence_id"] == "real_full_capture_validation"
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
    assert all(sensor["enabled"] is True for sensor in sensors)
    assert all(sensor["inverted"] is False for sensor in sensors)
    assert data["capture"]["synchronization"] == {
        "schema_version": CAPTURE_SYNCHRONIZATION_SCHEMA_VERSION,
        "mode": "timestamp_aligned",
    }

    plan = sequence_plan_from_run_config(data)

    assert plan.sequence_id == "real_full_capture_validation"
    assert plan.plan_only is True
    assert plan.steps[0].command[:3] == ["uv", "run", "python"]


def test_run_config_records_and_validates_expected_sunrise_reference_path(
    tmp_path: Path,
) -> None:
    value = create_run_config(
        run_root=tmp_path / "referenced-run",
        robot_pose_sunrise_reference_frame_path="/PoseTestBot/PoseTemplateBase",
    ).to_dict()

    assert value["frames"]["robot_pose"]["sunrise_reference_frame_path"] == (
        "/PoseTestBot/PoseTemplateBase"
    )
    validate_run_config(value)

    value["frames"]["robot_pose"]["sunrise_reference_frame_path"] = (
        "PoseTestBot/PoseTemplateBase"
    )
    with pytest.raises(ValueError, match="sunrise_reference_frame_path is invalid"):
        validate_run_config(value)


@pytest.mark.parametrize("velocity", [float("nan")])
def test_run_config_rejects_non_positive_or_non_finite_velocity(
    tmp_path: Path,
    velocity: float,
) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        create_run_config(
            run_root=tmp_path / "bad-velocity",
            velocity_m_s=velocity,
        )


def test_sensor_config_token_accepts_alias_and_mounting_mode() -> None:
    sensor = sensor_config_from_token("luxonis:mxid-1:static:Cell OAK-D Pro")

    assert sensor.sensor_type == "oak_d_pro"
    assert sensor.device_id == "mxid-1"
    assert sensor.mounting_mode == "static"
    assert sensor.display_name == "Cell OAK-D Pro"
    assert sensor.operator_alias == "Cell OAK-D Pro"
    assert sensor.inverted is False


def test_sensor_config_normalizes_explicit_run_operator_alias() -> None:
    sensor = sensor_config_from_mapping(
        {
            "sensor_type": "realsense",
            "device_id": "123",
            "mounting_mode": "eye_in_hand",
            "display_name": "Intel RealSense 123",
            "operator_alias": "  Run wrist camera  ",
        }
    )

    assert sensor.operator_alias == "Run wrist camera"
    assert sensor.display_name == "Run wrist camera"
    assert sensor.to_dict()["operator_alias"] == "Run wrist camera"


def test_sensor_mapping_requires_explicit_mount_or_explicit_default() -> None:
    value = {
        "sensor_type": "realsense",
        "device_id": "123",
        "display_name": "Intel RealSense 123",
    }

    with pytest.raises(ValueError, match="mounting_mode is required"):
        sensor_config_from_mapping(value)

    sensor = sensor_config_from_mapping(value, default_mounting_mode="static")
    assert sensor.mounting_mode == "static"


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


def test_sensor_config_preserves_literal_disabled_flag() -> None:
    sensor = sensor_config_from_mapping(
        {
            "sensor_type": "realsense",
            "device_id": "456",
            "mounting_mode": "eye_in_hand",
            "display_name": "Wrist RealSense",
            "enabled": False,
        }
    )

    assert sensor.enabled is False


@pytest.mark.parametrize("enabled", [None])
def test_sensor_config_rejects_non_literal_enabled_values(enabled: object) -> None:
    with pytest.raises(ValueError, match="literal JSON boolean"):
        sensor_config_from_mapping(
            {
                "sensor_type": "realsense",
                "device_id": "456",
                "mounting_mode": "eye_in_hand",
                "display_name": "Wrist RealSense",
                "enabled": enabled,
            }
        )


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
                            "alias": "Wrist Camera",
                            "effective_display_name": "Wrist Camera",
                            "mounting_mode": "static",
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
    assert sensors[0].operator_alias == "Wrist Camera"
    assert sensors[0].mounting_mode == "static"
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


def test_run_config_rejects_persisted_execution_acknowledgements(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must not persist execution gate"):
        create_run_config(
            run_root=tmp_path / "unsafe-reusable-config",
            sequence_options={
                "capture_execution": {
                    "allow_cameras": True,
                    "allow_real_robot": True,
                }
            },
        )


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
    export_step = next(step for step in plan.steps if step.id == "bop_export")

    assert export_step.options["calibration_profiles"] == profiles_path.as_posix()
    assert export_step.options["annotation_source"] == "none"
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
            "--sensor",
            "realsense:123:static:Cell RealSense",
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
    assert config["frames"]["robot_pose"]["sunrise_reference_frame_path"] == (
        "/PoseTestBot/PoseTemplateBase"
    )
    assert config["capture"]["sensors"] == [
        {
            "calibration_profile_id": None,
            "device_id": "123",
            "display_name": "Cell RealSense",
            "enabled": True,
            "inverted": False,
            "metadata": {},
            "mounting_mode": "static",
            "operator_alias": "Cell RealSense",
            "sensor_type": "realsense_d435",
        }
    ]
    assert config["dataset_mode"] == "objectless"
    assert "object_folder" not in config
    assert config["pipeline"]["options"] == {"aruco": {"save_images": True}}
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(stage for stage in manifest["stages"] if stage["name"] == "run_config")
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][RUN_CONFIG] == RUN_CONFIG


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


def test_run_config_rejects_retired_fake_robot_profile(tmp_path: Path) -> None:
    run_root = tmp_path / "retired-fake-config"
    value = create_run_config(run_root=run_root).to_dict()
    value["robot_profile"]["mode"] = "fake"
    run_root.mkdir()
    (run_root / RUN_CONFIG).write_text(json.dumps(value))

    with pytest.raises(ValueError, match="must be 'real'"):
        load_run_config_for_run_root(run_root)


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
