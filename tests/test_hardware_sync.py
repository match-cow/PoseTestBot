from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from posetestbot.io.artifacts import (
    BOP_FRAME_SETS,
    CAPTURE_EXECUTION_REPORT,
    FRAME_METADATA_JSONL,
    MATCH_ROBOT_EE_POSES,
    MULTIVIEW_FRAME_GROUPS,
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
    HardwareSyncEvidenceError,
    build_hardware_sync_frame_groups,
    hardware_sync_frame_groups_path,
    load_hardware_sync_frame_groups,
    validate_hardware_sync_frame_groups,
    write_hardware_sync_frame_groups,
)


MASTER_KEY = "realsense_d435:master"
SUBORDINATE_A_KEY = "realsense_d435:sub-a"
SUBORDINATE_B_KEY = "realsense_d435:sub-b"


def _sensor(device_id: str, mounting_mode: str) -> SensorRunConfig:
    return SensorRunConfig(
        sensor_type="realsense_d435",
        device_id=device_id,
        display_name=device_id,
        mounting_mode=mounting_mode,
    )


def _configured_run(
    root: Path,
    *,
    three_sensors: bool = False,
    max_skew_ms: float = 2.0,
) -> dict[str, Any]:
    sensors = [
        _sensor("master", "static"),
        _sensor("sub-a", "eye_in_hand"),
    ]
    if three_sensors:
        sensors.append(_sensor("sub-b", "static"))
    config = create_run_config(
        run_root=root,
        sensors=tuple(sensors),
        synchronization={
            "schema_version": "capture_synchronization.v1",
            "mode": "hardware_trigger",
            "implementation": "realsense_inter_cam_sync",
            "scope": "depth_exposure",
            "group_id": "research-rig",
            "master_sensor_key": MASTER_KEY,
            "max_depth_timestamp_skew_ms": max_skew_ms,
        },
    )
    write_run_config(root, config)
    return config.to_dict()


def _write_sensor_evidence(
    root: Path,
    *,
    device_id: str,
    role: str,
    mounting_mode: str,
    timestamps_ns: list[int],
    counters: list[int],
) -> None:
    assert len(timestamps_ns) == len(counters)
    raw_folder = root / f"realsense_{device_id}"
    synchronized_folder = (
        root / "processed" / "synchronized" / f"realsense_{device_id}"
    )
    for folder in (raw_folder, synchronized_folder):
        (folder / "rgb").mkdir(parents=True, exist_ok=True)
        (folder / "depth").mkdir(parents=True, exist_ok=True)

    records = []
    matched: dict[str, Any] = {}
    option = 1 if role == "master" else 2
    for index, (timestamp_ns, counter) in enumerate(
        zip(timestamps_ns, counters, strict=True)
    ):
        source_id = f"{1000 + index}.png"
        synchronized_id = f"{index:06d}.png"
        for folder, frame_id in (
            (raw_folder, source_id),
            (synchronized_folder, synchronized_id),
        ):
            (folder / "rgb" / frame_id).write_bytes(b"rgb")
            (folder / "depth" / frame_id).write_bytes(b"depth")
        records.append(
            {
                "schema_version": "frame_metadata.v1",
                "sensor_type": "realsense_d435",
                "sensor_id": device_id,
                "frame_index": index,
                "frame_id": synchronized_id,
                "rgb_path": f"rgb/{synchronized_id}",
                "depth_path": f"depth/{synchronized_id}",
                "source_frame_index": index + 10,
                "source_frame_id": source_id,
                "source_rgb_path": f"rgb/{source_id}",
                "source_depth_path": f"depth/{source_id}",
                "depth_sensor_timestamp_ns": timestamp_ns,
                "depth_frame_number": counter,
                "depth_timestamp_domain": "global_time",
                "capture_group_id": "research-rig",
                "hardware_sync_role": role,
                "hardware_sync_scope": "depth_exposure",
                "hardware_sync_transport": "realsense_inter_cam_sync",
                "inter_cam_sync_mode_configured": option,
                "inter_cam_sync_mode_readback": option,
                "matched_robot_pose_index": index + 100,
                "nearest_robot_delta_ns": index,
                "motion": f"pose-{index}",
                "mounting_mode": mounting_mode,
            }
        )
        matched[synchronized_id] = {
            "source_frame_id": source_id,
            "matched_robot_pose_index": index + 100,
            "robot_timestamp_ns": timestamp_ns + 50,
            "nearest_robot_delta_ns": 50,
            "motion": f"pose-{index}",
            "robot_ee_pose": {
                "x": float(index),
                "y": float(index + 1),
                "z": float(index + 2),
                "a": 0.0,
                "b": 0.0,
                "c": 0.0,
            },
        }
    (synchronized_folder / FRAME_METADATA_JSONL).write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    (synchronized_folder / MATCH_ROBOT_EE_POSES).write_text(
        json.dumps(matched)
    )


def _load_metadata(root: Path, device_id: str) -> list[dict[str, Any]]:
    path = (
        root
        / "processed"
        / "synchronized"
        / f"realsense_{device_id}"
        / FRAME_METADATA_JSONL
    )
    return [json.loads(line) for line in path.read_text().splitlines()]


def _replace_metadata(
    root: Path,
    device_id: str,
    records: list[dict[str, Any]],
) -> None:
    path = (
        root
        / "processed"
        / "synchronized"
        / f"realsense_{device_id}"
        / FRAME_METADATA_JSONL
    )
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _raw_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for folder in sorted(root.glob("realsense_*"))
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def _record_current_qualification(root: Path) -> dict[str, Any]:
    evidence = root / "physical-depth-sync-evidence.txt"
    evidence.write_text("externally measured exposure timing")
    record_hardware_sync_qualification(
        root,
        operator="test-operator",
        method="pulsed_light",
        observed_max_depth_timestamp_skew_ms=0.5,
        evidence_paths=[evidence],
        confirm_passed=True,
    )
    return validate_hardware_sync_qualification(root)


def _attach_authoritative_capture_provenance(
    root: Path,
    value: dict[str, Any],
) -> dict[str, Any]:
    qualification = validate_hardware_sync_qualification(root)
    binding = {
        "configuration_sha256": qualification["configuration_sha256"],
        "qualification_artifact_sha256": qualification["artifact_sha256"],
        "revalidated_immediately_before_receiver_spawn": True,
    }
    (root / CAPTURE_EXECUTION_REPORT).write_text(
        json.dumps(
            {
                "schema_version": "capture_execution_report.v1",
                "run_root": root.as_posix(),
                "status": "succeeded",
                "mode": "full",
                "allow_cameras": True,
                "allow_real_robot": True,
                "hardware_sync_execution_binding": binding,
            }
        )
    )
    value["hardware_sync_qualification"] = qualification
    value["hardware_sync_execution_binding"] = binding
    return value


def test_build_write_and_load_complete_hardware_frame_groups(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _configured_run(root, three_sensors=True)
    _record_current_qualification(root)
    _write_sensor_evidence(
        root,
        device_id="master",
        role="master",
        mounting_mode="static",
        timestamps_ns=[1_000_000_000, 2_000_000_000, 3_000_000_000],
        counters=[100, 101, 102],
    )
    _write_sensor_evidence(
        root,
        device_id="sub-a",
        role="subordinate",
        mounting_mode="eye_in_hand",
        timestamps_ns=[1_000_500_000, 2_001_000_000, 4_000_000_000],
        counters=[10, 11, 13],
    )
    _write_sensor_evidence(
        root,
        device_id="sub-b",
        role="subordinate",
        mounting_mode="static",
        timestamps_ns=[999_000_000, 1_999_000_000, 3_000_500_000],
        counters=[20, 21, 22],
    )
    raw_before = _raw_snapshot(root)

    value = build_hardware_sync_frame_groups(root)

    assert value["schema_version"] == "hardware_sync_frame_groups.v1"
    provenance = value["content_provenance"]
    assert provenance["schema_version"] == "hardware_sync_content_provenance.v1"
    assert provenance["digest_algorithm"] == "sha256"
    assert provenance["hardware_contract"]["path"] == "run_config.json"
    assert len(provenance["hardware_contract"]["sha256"]) == 64
    assert [item["sensor_key"] for item in provenance["sensors"]] == [
        MASTER_KEY,
        SUBORDINATE_A_KEY,
        SUBORDINATE_B_KEY,
    ]
    assert provenance["sensors"][0]["referenced_frames"]["file_count"] == 12
    assert value["sensor_order"] == [
        MASTER_KEY,
        SUBORDINATE_A_KEY,
        SUBORDINATE_B_KEY,
    ]
    assert value["mounting_modes"] == {
        "static": [MASTER_KEY, SUBORDINATE_B_KEY],
        "eye_in_hand": [SUBORDINATE_A_KEY],
    }
    assert value["summary"] == {
        "sensor_count": 3,
        "master_frame_count": 3,
        "complete_group_count": 2,
        "incomplete_master_group_count": 1,
        "unmatched_subordinate_frame_count": 1,
        "counter_discontinuity_count": 1,
    }
    assert [group["frame_group_id"] for group in value["groups"]] == [
        "research-rig:000000",
        "research-rig:000001",
    ]
    first_group = value["groups"][0]
    assert first_group["depth_sensor_timestamp_ns"] == 1_000_000_000
    assert first_group["matched_robot_pose"]["matched_robot_pose_index"] == 100
    assert first_group["matched_robot_pose"]["robot_ee_pose"]["x"] == 0.0
    assert first_group["frames"][MASTER_KEY]["depth_timestamp_skew_ns"] == 0
    assert (
        first_group["frames"][SUBORDINATE_A_KEY]["depth_timestamp_skew_ns"]
        == 500_000
    )
    assert (
        first_group["frames"][SUBORDINATE_B_KEY]["depth_timestamp_skew_ns"]
        == -1_000_000
    )
    assert first_group["frames"][SUBORDINATE_A_KEY]["source_frame_id"] == (
        "1000.png"
    )
    assert first_group["frames"][SUBORDINATE_A_KEY][
        "synchronized_frame_id"
    ] == "000000.png"
    assert first_group["frames"][SUBORDINATE_A_KEY]["sensor_folder"] == (
        "processed/synchronized/realsense_sub-a"
    )
    assert first_group["frames"][SUBORDINATE_A_KEY][
        "synchronized_rgb_path"
    ] == "rgb/000000.png"
    assert first_group["frames"][SUBORDINATE_A_KEY][
        "source_sensor_folder"
    ] == "realsense_sub-a"
    assert first_group["frames"][SUBORDINATE_A_KEY]["source_rgb_path"] == (
        "rgb/1000.png"
    )
    assert value["incomplete_master_groups"][0]["missing_sensor_keys"] == [
        SUBORDINATE_A_KEY
    ]
    assert (
        value["incomplete_master_groups"][0]["master_frame_ordinal"] == 2
    )
    unmatched = value["unmatched_subordinate_frames"][SUBORDINATE_A_KEY]
    assert len(unmatched) == 1
    assert unmatched[0]["source_frame_id"] == "1002.png"
    assert unmatched[0]["reason"] == (
        "no_master_within_max_depth_timestamp_skew"
    )
    assert value["counter_discontinuities"][SUBORDINATE_A_KEY] == [
        {
            "kind": "gap",
            "previous_synchronized_frame_id": "000001.png",
            "synchronized_frame_id": "000002.png",
            "previous_depth_frame_number": 11,
            "depth_frame_number": 13,
            "counter_delta": 2,
            "missing_frame_count": 1,
            "previous_depth_sensor_timestamp_ns": 2_001_000_000,
            "depth_sensor_timestamp_ns": 4_000_000_000,
        }
    ]
    assert value["observed_skew"][
        "maximum_abs_depth_timestamp_skew_ns"
    ] == 1_000_000
    assert first_group["depth_timestamp_span_ns"] == 1_500_000
    _attach_authoritative_capture_provenance(root, value)

    path = write_hardware_sync_frame_groups(root, value)

    assert path == (
        root
        / "processed"
        / "synchronized"
        / MULTIVIEW_FRAME_GROUPS
    )
    assert hardware_sync_frame_groups_path(root) == path
    assert load_hardware_sync_frame_groups(root) == value
    assert _raw_snapshot(root) == raw_before
    assert BOP_FRAME_SETS == "posetestbot_frame_sets.json"


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("capture_group_id", "other-rig", "capture_group_id"),
        ("hardware_sync_role", "master", "hardware_sync_role"),
        ("hardware_sync_scope", "rgb_exposure", "hardware_sync_scope"),
        ("hardware_sync_transport", "host_clock", "hardware_sync_transport"),
        ("inter_cam_sync_mode_configured", 1, "configured"),
        ("inter_cam_sync_mode_readback", 0, "readback"),
        ("depth_timestamp_domain", "hardware_clock", "timestamp_domain"),
        ("depth_sensor_timestamp_ns", None, "depth_sensor_timestamp_ns"),
        ("depth_frame_number", None, "depth_frame_number"),
    ],
)
def test_frame_evidence_must_prove_the_configured_hardware_sync(
    tmp_path: Path,
    field: str,
    replacement: Any,
    message: str,
) -> None:
    root = tmp_path / field
    _configured_run(root)
    _write_sensor_evidence(
        root,
        device_id="master",
        role="master",
        mounting_mode="static",
        timestamps_ns=[1_000_000_000],
        counters=[1],
    )
    _write_sensor_evidence(
        root,
        device_id="sub-a",
        role="subordinate",
        mounting_mode="eye_in_hand",
        timestamps_ns=[1_000_000_000],
        counters=[1],
    )
    records = _load_metadata(root, "sub-a")
    if replacement is None:
        records[0].pop(field)
    else:
        records[0][field] = replacement
    _replace_metadata(root, "sub-a", records)

    with pytest.raises(HardwareSyncEvidenceError, match=message):
        build_hardware_sync_frame_groups(root)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("schema_version", "capture_synchronization.v0", "schema_version"),
        ("mode", "timestamp_aligned", "mode"),
        ("implementation", "host_timestamps", "implementation"),
        ("scope", "rgb_exposure", "scope"),
        ("group_id", "", "group_id"),
        ("master_sensor_key", "realsense_d435:missing", "enabled sensor"),
        ("max_depth_timestamp_skew_ms", float("nan"), "finite"),
    ],
)
def test_build_rejects_unsupported_or_incomplete_configuration(
    tmp_path: Path,
    field: str,
    replacement: Any,
    message: str,
) -> None:
    root = tmp_path / field
    config = _configured_run(root)
    config["capture"]["synchronization"][field] = replacement

    with pytest.raises(HardwareSyncEvidenceError, match=message):
        build_hardware_sync_frame_groups(root, run_config=config)


def test_disabled_sensor_is_not_a_required_group_member(tmp_path: Path) -> None:
    root = tmp_path / "run"
    config = _configured_run(root)
    config["capture"]["sensors"].append(
        {
            "sensor_type": "oak_d_pro",
            "device_id": "disabled",
            "display_name": "disabled",
            "mounting_mode": "static",
            "enabled": False,
        }
    )
    _write_sensor_evidence(
        root,
        device_id="master",
        role="master",
        mounting_mode="static",
        timestamps_ns=[1_000_000_000],
        counters=[1],
    )
    _write_sensor_evidence(
        root,
        device_id="sub-a",
        role="subordinate",
        mounting_mode="eye_in_hand",
        timestamps_ns=[1_000_000_000],
        counters=[1],
    )

    value = build_hardware_sync_frame_groups(root, run_config=config)

    assert value["summary"]["sensor_count"] == 2
    assert value["summary"]["complete_group_count"] == 1


def test_one_subordinate_frame_is_never_reused_across_master_groups(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _configured_run(root, max_skew_ms=2.0)
    _write_sensor_evidence(
        root,
        device_id="master",
        role="master",
        mounting_mode="static",
        timestamps_ns=[1_000_000_000, 1_002_000_000],
        counters=[1, 2],
    )
    _write_sensor_evidence(
        root,
        device_id="sub-a",
        role="subordinate",
        mounting_mode="eye_in_hand",
        timestamps_ns=[1_001_000_000],
        counters=[10],
    )

    value = build_hardware_sync_frame_groups(root)

    assert len(value["groups"]) == 1
    assert value["groups"][0]["master_frame_ordinal"] == 0
    assert value["incomplete_master_groups"][0]["master_frame_ordinal"] == 1
    assert (
        value["sensors"][1]["matched_to_master_frame_count"]
        == value["sensors"][1]["frame_count"]
        == 1
    )


def test_complete_group_skew_uses_full_three_camera_timestamp_span(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _configured_run(root, three_sensors=True, max_skew_ms=2.0)
    _write_sensor_evidence(
        root,
        device_id="master",
        role="master",
        mounting_mode="static",
        timestamps_ns=[1_000_000_000, 2_000_000_000],
        counters=[1, 2],
    )
    _write_sensor_evidence(
        root,
        device_id="sub-a",
        role="subordinate",
        mounting_mode="eye_in_hand",
        timestamps_ns=[1_001_500_000, 2_000_500_000],
        counters=[10, 11],
    )
    _write_sensor_evidence(
        root,
        device_id="sub-b",
        role="subordinate",
        mounting_mode="static",
        timestamps_ns=[998_500_000, 1_999_500_000],
        counters=[20, 21],
    )

    value = build_hardware_sync_frame_groups(root)

    assert [group["master_frame_ordinal"] for group in value["groups"]] == [1]
    assert value["groups"][0]["depth_timestamp_span_ns"] == 1_000_000
    rejected = value["incomplete_master_groups"][0]
    assert rejected["master_frame_ordinal"] == 0
    assert rejected["missing_sensor_keys"] == []
    assert rejected["reason"] == (
        "depth_timestamp_span_exceeds_configured_maximum"
    )
    assert rejected["max_abs_depth_timestamp_skew_ns"] == 1_500_000
    assert rejected["depth_timestamp_span_ns"] == 3_000_000


def test_structural_validation_rejects_opposite_side_span_above_limit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _configured_run(root, three_sensors=True, max_skew_ms=2.0)
    for device_id, role, mounting in (
        ("master", "master", "static"),
        ("sub-a", "subordinate", "eye_in_hand"),
        ("sub-b", "subordinate", "static"),
    ):
        _write_sensor_evidence(
            root,
            device_id=device_id,
            role=role,
            mounting_mode=mounting,
            timestamps_ns=[1_000_000_000],
            counters=[1],
        )
    value = build_hardware_sync_frame_groups(root)
    group = value["groups"][0]
    group["frames"][SUBORDINATE_A_KEY]["depth_sensor_timestamp_ns"] += 1_500_000
    group["frames"][SUBORDINATE_A_KEY]["depth_timestamp_skew_ns"] = 1_500_000
    group["frames"][SUBORDINATE_A_KEY][
        "abs_depth_timestamp_skew_ns"
    ] = 1_500_000
    group["frames"][SUBORDINATE_B_KEY]["depth_sensor_timestamp_ns"] -= 1_500_000
    group["frames"][SUBORDINATE_B_KEY]["depth_timestamp_skew_ns"] = -1_500_000
    group["frames"][SUBORDINATE_B_KEY][
        "abs_depth_timestamp_skew_ns"
    ] = 1_500_000
    group["max_abs_depth_timestamp_skew_ns"] = 1_500_000
    group["depth_timestamp_span_ns"] = 3_000_000
    value["hardware_sync_execution_binding"] = {
        "configuration_sha256": "a" * 64,
        "qualification_artifact_sha256": "b" * 64,
        "revalidated_immediately_before_receiver_spawn": True,
    }

    with pytest.raises(HardwareSyncEvidenceError, match="span exceeds"):
        validate_hardware_sync_frame_groups(value)


def test_build_requires_at_least_one_complete_group(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _configured_run(root, max_skew_ms=1.0)
    _write_sensor_evidence(
        root,
        device_id="master",
        role="master",
        mounting_mode="static",
        timestamps_ns=[1_000_000_000],
        counters=[1],
    )
    _write_sensor_evidence(
        root,
        device_id="sub-a",
        role="subordinate",
        mounting_mode="eye_in_hand",
        timestamps_ns=[2_000_000_000],
        counters=[1],
    )

    with pytest.raises(HardwareSyncEvidenceError, match="no complete"):
        build_hardware_sync_frame_groups(root)

    assert not hardware_sync_frame_groups_path(root).exists()


def test_counter_duplicate_regression_and_gap_are_reported(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _configured_run(root)
    timestamps = [
        1_000_000_000,
        2_000_000_000,
        3_000_000_000,
        4_000_000_000,
    ]
    _write_sensor_evidence(
        root,
        device_id="master",
        role="master",
        mounting_mode="static",
        timestamps_ns=timestamps,
        counters=[1, 2, 3, 4],
    )
    _write_sensor_evidence(
        root,
        device_id="sub-a",
        role="subordinate",
        mounting_mode="eye_in_hand",
        timestamps_ns=timestamps,
        counters=[5, 5, 4, 7],
    )

    value = build_hardware_sync_frame_groups(root)

    assert [
        item["kind"]
        for item in value["counter_discontinuities"][SUBORDINATE_A_KEY]
    ] == ["duplicate", "regression", "gap"]
    assert value["summary"]["counter_discontinuity_count"] == 3
    assert value["summary"]["complete_group_count"] == 4


def test_missing_matched_pose_and_escaping_frame_path_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _configured_run(root)
    for device_id, role, mounting in (
        ("master", "master", "static"),
        ("sub-a", "subordinate", "eye_in_hand"),
    ):
        _write_sensor_evidence(
            root,
            device_id=device_id,
            role=role,
            mounting_mode=mounting,
            timestamps_ns=[1_000_000_000],
            counters=[1],
        )
    matched_path = (
        root
        / "processed"
        / "synchronized"
        / "realsense_sub-a"
        / MATCH_ROBOT_EE_POSES
    )
    matched_path.write_text("{}")

    with pytest.raises(HardwareSyncEvidenceError, match="matched robot pose"):
        build_hardware_sync_frame_groups(root)

    _write_sensor_evidence(
        root,
        device_id="sub-a",
        role="subordinate",
        mounting_mode="eye_in_hand",
        timestamps_ns=[1_000_000_000],
        counters=[1],
    )
    records = _load_metadata(root, "sub-a")
    records[0]["source_rgb_path"] = "../outside.png"
    _replace_metadata(root, "sub-a", records)

    with pytest.raises(HardwareSyncEvidenceError, match="source_rgb_path"):
        build_hardware_sync_frame_groups(root)


def test_write_and_load_reject_tampered_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _configured_run(root)
    _record_current_qualification(root)
    for device_id, role, mounting in (
        ("master", "master", "static"),
        ("sub-a", "subordinate", "eye_in_hand"),
    ):
        _write_sensor_evidence(
            root,
            device_id=device_id,
            role=role,
            mounting_mode=mounting,
            timestamps_ns=[1_000_000_000],
            counters=[1],
        )
    value = build_hardware_sync_frame_groups(root)
    _attach_authoritative_capture_provenance(root, value)
    invalid = copy.deepcopy(value)
    invalid["groups"][0]["frames"][SUBORDINATE_A_KEY][
        "depth_timestamp_skew_ns"
    ] = 42

    with pytest.raises(HardwareSyncEvidenceError, match="skew is inconsistent"):
        write_hardware_sync_frame_groups(root, invalid)

    missing_counter = copy.deepcopy(value)
    missing_counter["groups"][0]["frames"][SUBORDINATE_A_KEY].pop(
        "depth_frame_number"
    )
    with pytest.raises(HardwareSyncEvidenceError, match="depth_frame_number"):
        write_hardware_sync_frame_groups(root, missing_counter)

    assert not hardware_sync_frame_groups_path(root).exists()
    path = write_hardware_sync_frame_groups(root, value)
    stored = json.loads(path.read_text())
    stored["schema_version"] = "hardware_sync_frame_groups.v0"
    path.write_text(json.dumps(stored))

    with pytest.raises(HardwareSyncEvidenceError, match="schema_version"):
        load_hardware_sync_frame_groups(root)


def test_authoritative_boundaries_require_exact_current_qualification(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing"
    _configured_run(missing_root)
    for device_id, role, mounting in (
        ("master", "master", "static"),
        ("sub-a", "subordinate", "eye_in_hand"),
    ):
        _write_sensor_evidence(
            missing_root,
            device_id=device_id,
            role=role,
            mounting_mode=mounting,
            timestamps_ns=[1_000_000_000],
            counters=[1],
        )
    missing_value = build_hardware_sync_frame_groups(missing_root)

    with pytest.raises(
        HardwareSyncEvidenceError,
        match="qualification|execution_binding",
    ):
        write_hardware_sync_frame_groups(missing_root, missing_value)

    root = tmp_path / "qualified"
    _configured_run(root)
    _record_current_qualification(root)
    for device_id, role, mounting in (
        ("master", "master", "static"),
        ("sub-a", "subordinate", "eye_in_hand"),
    ):
        _write_sensor_evidence(
            root,
            device_id=device_id,
            role=role,
            mounting_mode=mounting,
            timestamps_ns=[1_000_000_000],
            counters=[1],
        )
    value = _attach_authoritative_capture_provenance(
        root,
        build_hardware_sync_frame_groups(root),
    )
    wrong_qualification = copy.deepcopy(value)
    wrong_qualification["hardware_sync_qualification"]["operator"] = "other"
    with pytest.raises(HardwareSyncEvidenceError, match="exact current"):
        write_hardware_sync_frame_groups(root, wrong_qualification)

    write_hardware_sync_frame_groups(root, value)
    qualification_path = root / "hardware_sync_qualification.json"
    qualification = json.loads(qualification_path.read_text())
    qualification["operator"] = "direct post-capture tamper"
    qualification_path.write_text(json.dumps(qualification))
    with pytest.raises(HardwareSyncEvidenceError, match="qualification"):
        load_hardware_sync_frame_groups(root)


def test_authoritative_write_rejects_noncanonical_derived_payload(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _configured_run(root)
    _record_current_qualification(root)
    for device_id, role, mounting in (
        ("master", "master", "static"),
        ("sub-a", "subordinate", "eye_in_hand"),
    ):
        _write_sensor_evidence(
            root,
            device_id=device_id,
            role=role,
            mounting_mode=mounting,
            timestamps_ns=[1_000_000_000],
            counters=[1],
        )
    value = _attach_authoritative_capture_provenance(
        root,
        build_hardware_sync_frame_groups(root),
    )

    changed_summary = copy.deepcopy(value)
    changed_summary["summary"]["master_frame_count"] = 99
    with pytest.raises(HardwareSyncEvidenceError, match="canonical groups"):
        write_hardware_sync_frame_groups(root, changed_summary)

    changed_skew = copy.deepcopy(value)
    changed_skew["max_depth_timestamp_skew_ms"] = 1.0
    changed_skew["max_depth_timestamp_skew_ns"] = 1_000_000
    with pytest.raises(HardwareSyncEvidenceError, match="canonical groups"):
        write_hardware_sync_frame_groups(root, changed_skew)

    added_field = copy.deepcopy(value)
    added_field["untrusted_annotation"] = True
    with pytest.raises(HardwareSyncEvidenceError, match="canonical groups"):
        write_hardware_sync_frame_groups(root, added_field)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_report",
        "failed_status",
        "not_revalidated",
        "configuration_digest",
        "qualification_hash",
    ],
)
def test_authoritative_load_rejects_changed_capture_execution_binding(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = tmp_path / mutation
    _configured_run(root)
    _record_current_qualification(root)
    for device_id, role, mounting in (
        ("master", "master", "static"),
        ("sub-a", "subordinate", "eye_in_hand"),
    ):
        _write_sensor_evidence(
            root,
            device_id=device_id,
            role=role,
            mounting_mode=mounting,
            timestamps_ns=[1_000_000_000],
            counters=[1],
        )
    value = _attach_authoritative_capture_provenance(
        root,
        build_hardware_sync_frame_groups(root),
    )
    write_hardware_sync_frame_groups(root, value)
    report_path = root / CAPTURE_EXECUTION_REPORT
    if mutation == "missing_report":
        report_path.unlink()
    else:
        report = json.loads(report_path.read_text())
        binding = report["hardware_sync_execution_binding"]
        if mutation == "failed_status":
            report["status"] = "failed"
        elif mutation == "not_revalidated":
            binding["revalidated_immediately_before_receiver_spawn"] = False
        elif mutation == "configuration_digest":
            binding["configuration_sha256"] = "0" * 64
        else:
            binding["qualification_artifact_sha256"] = "0" * 64
        report_path.write_text(json.dumps(report))

    with pytest.raises(
        (FileNotFoundError, HardwareSyncEvidenceError),
        match=(
            "report|succeeded|revalidation|configuration|qualification"
        ),
    ):
        load_hardware_sync_frame_groups(root)


@pytest.mark.parametrize(
    "mutation",
    [
        "timestamp",
        "pose",
        "synchronized_rgb",
        "synchronized_depth",
        "source_rgb",
        "source_depth",
        "config_mount",
        "config_identity",
    ],
)
def test_load_rejects_stale_source_or_hardware_contract_content(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = tmp_path / mutation
    _configured_run(root)
    _record_current_qualification(root)
    for device_id, role, mounting in (
        ("master", "master", "static"),
        ("sub-a", "subordinate", "eye_in_hand"),
    ):
        _write_sensor_evidence(
            root,
            device_id=device_id,
            role=role,
            mounting_mode=mounting,
            timestamps_ns=[1_000_000_000],
            counters=[1],
        )
    value = _attach_authoritative_capture_provenance(
        root,
        build_hardware_sync_frame_groups(root),
    )
    write_hardware_sync_frame_groups(
        root,
        value,
    )

    synchronized_folder = (
        root / "processed" / "synchronized" / "realsense_sub-a"
    )
    source_folder = root / "realsense_sub-a"
    if mutation == "timestamp":
        records = _load_metadata(root, "sub-a")
        records[0]["depth_sensor_timestamp_ns"] += 1
        _replace_metadata(root, "sub-a", records)
    elif mutation == "pose":
        matched_path = synchronized_folder / MATCH_ROBOT_EE_POSES
        matched = json.loads(matched_path.read_text())
        matched["000000.png"]["robot_ee_pose"]["x"] = 123.0
        matched_path.write_text(json.dumps(matched))
    elif mutation == "synchronized_rgb":
        (synchronized_folder / "rgb" / "000000.png").write_bytes(b"changed-rgb")
    elif mutation == "synchronized_depth":
        (synchronized_folder / "depth" / "000000.png").write_bytes(
            b"changed-depth"
        )
    elif mutation == "source_rgb":
        (source_folder / "rgb" / "1000.png").write_bytes(b"changed-source-rgb")
    elif mutation == "source_depth":
        (source_folder / "depth" / "1000.png").write_bytes(
            b"changed-source-depth"
        )
    else:
        config_path = root / "run_config.json"
        config = json.loads(config_path.read_text())
        if mutation == "config_mount":
            config["capture"]["sensors"][0]["mounting_mode"] = "eye_in_hand"
            config["capture"]["sensors"][1]["mounting_mode"] = "static"
        else:
            config["capture"]["sensors"][0]["device_id"] = "replacement"
            config["capture"]["synchronization"][
                "master_sensor_key"
            ] = "realsense_d435:replacement"
        config_path.write_text(json.dumps(config))

    with pytest.raises(
        HardwareSyncEvidenceError,
        match="stale|hardware contract|canonical groups",
    ):
        load_hardware_sync_frame_groups(root)
