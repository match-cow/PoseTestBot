from __future__ import annotations

import json
import os
import signal
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from posetestbot.io.artifacts import (
    CAPTURE_EXECUTION_LOGS_DIR,
    CAPTURE_EXECUTION_PLAN,
    CAPTURE_EXECUTION_REPORT,
    CAPTURE_EXECUTION_STATUS,
    CAPTURE_PLAN,
    DATASET_MANIFEST,
    FRAME_METADATA_JSONL,
    HARDWARE_SYNC_QUALIFICATION,
    RAW_ROBOT_EE_POSES,
    RUN_CONFIG,
)
from posetestbot.io.manifest import write_run_manifest
from posetestbot.pipeline.capture_plan import build_capture_plan, write_capture_plan
from posetestbot.pipeline.capture_execution import (
    CaptureExecutionPermissionError,
    _hardware_sync_overlap_evidence,
    _hardware_sync_readiness_errors,
    build_capture_execution_plan,
    run_capture_execution,
    write_capture_execution_plan_with_manifest,
)
from posetestbot.pipeline.run_config import (
    SensorRunConfig,
    create_run_config,
    sensor_config_from_token,
    write_run_config,
)
from posetestbot.sensors.hardware_sync_qualification import (
    record_hardware_sync_qualification,
)


class FakeBackgroundProcess:
    def __init__(self, command: list[str], log_file):
        self.command = command
        self.returncode = None
        self.log_file = log_file
        self.pid = 12345
        self.log_file.write("fake background started\n")

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        self.log_file.write("fake background finished\n")
        return 0


class FakePersistentProcess:
    def __init__(self, command: list[str], log_file):
        self.command = command
        self.returncode = None
        self.log_file = log_file
        self.pid = 23456
        self.log_file.write("fake persistent background started\n")

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        raise subprocess.TimeoutExpired(self.command, timeout)


class FakeSignalProcess(FakePersistentProcess):
    def wait(self, timeout=None):
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("SIGTERM handler should interrupt receiver wait")


class FakeCameraExitWhileReceiverRuns(FakePersistentProcess):
    def __init__(self, command: list[str], log_file, state: dict, returncode: int):
        super().__init__(command, log_file)
        self.state = state
        self.exit_returncode = returncode

    def poll(self):
        if self.state.get("receiver_started"):
            self.returncode = self.exit_returncode
        return self.returncode


class FakeCameraExitAfterReceiver(FakePersistentProcess):
    def __init__(self, command: list[str], log_file, returncode: int):
        super().__init__(command, log_file)
        self.exit_returncode = returncode

    def wait(self, timeout=None):
        self.returncode = self.exit_returncode
        return self.returncode


def fake_sensor_status() -> dict:
    return {
        "schema_version": "sensor_status.v1",
        "families": [
            {
                "sensor_type": "realsense_d435",
                "sdk_available": True,
                "devices": [
                    {
                        "device_id": "123",
                        "display_name": "RealSense 123",
                        "connected": True,
                    }
                ],
                "error": None,
            }
        ],
        "overall_status": "ok",
        "checks": [],
    }


def filesystem_snapshot(root: Path) -> dict[str, tuple[str, bytes | None]]:
    if not root.exists():
        return {}
    snapshot: dict[str, tuple[str, bytes | None]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        snapshot[relative] = (
            "dir" if path.is_dir() else "file",
            None if path.is_dir() else path.read_bytes(),
        )
    return snapshot


def configured_run(tmp_path: Path, name: str = "run") -> tuple[Path, dict]:
    run_root = tmp_path / name
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
    )
    write_run_config(run_root, config)
    return run_root, config.to_dict()


def mark_sensor_ready(
    run_root: Path,
    *,
    device_id: str = "123",
    record_count: int = 3,
) -> None:
    sensor_folder = run_root / f"realsense_{device_id}"
    sensor_folder.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "schema_version": "frame_metadata.v1",
            "sensor_type": "realsense_d435",
            "sensor_id": device_id,
            "frame_index": index,
            "frame_id": f"{index}.png",
            "rgb_path": f"rgb/{index}.png",
            "depth_path": f"depth/{index}.png",
            "sensor_timestamp_ns": index + 1,
            "host_received_timestamp_ns": index + 1,
            "host_wall_timestamp_ns": index + 1,
        }
        for index in range(record_count)
    ]
    (sensor_folder / FRAME_METADATA_JSONL).write_text(
        "".join(f"{json.dumps(record)}\n" for record in records)
    )


def test_hardware_sync_readiness_requires_exact_adapter_readback(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / FRAME_METADATA_JSONL
    command = {
        "hardware_sync": {
            "implementation": "realsense_inter_cam_sync",
            "role": "subordinate",
            "group_id": "mixed-rig",
            "scope": "depth_exposure",
            "inter_cam_sync_mode_expected": 2,
        }
    }
    records = [
        {
            "schema_version": "frame_metadata.v1",
            "sensor_id": "hand",
            "frame_index": index,
            "frame_id": f"{index}.png",
            "rgb_path": f"rgb/{index}.png",
            "depth_path": f"depth/{index}.png",
            "host_received_timestamp_ns": index + 1,
            "capture_group_id": "mixed-rig",
            "hardware_sync_role": "subordinate",
            "hardware_sync_scope": "depth_exposure",
            "hardware_sync_transport": "realsense_inter_cam_sync",
            "inter_cam_sync_mode_configured": 2,
            "inter_cam_sync_mode_readback": 2,
            "depth_sensor_timestamp_ns": 1000 + index,
            "depth_frame_number": 20 + index,
            "depth_timestamp_domain": "global_time",
        }
        for index in range(3)
    ]
    metadata_path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records)
    )

    assert _hardware_sync_readiness_errors(command, metadata_path) == []

    records[2]["inter_cam_sync_mode_readback"] = 0
    metadata_path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records)
    )
    errors = _hardware_sync_readiness_errors(command, metadata_path)
    assert any("inter_cam_sync_mode_readback=0, expected 2" in item for item in errors)


def _write_hardware_sync_overlap_sensor(
    root: Path,
    *,
    device_id: str,
    role: str,
    timestamp_ns: int,
    transport: str = "realsense_inter_cam_sync",
    max_skew_ms: float = 2.0,
    record_count: int = 3,
) -> dict:
    output = root / f"realsense_{device_id}"
    output.mkdir(parents=True)
    mode = 1 if role == "master" else 2
    records = [
        {
            "schema_version": "frame_metadata.v1",
            "sensor_id": device_id,
            "frame_index": index,
            "frame_id": f"{index}.png",
            "rgb_path": f"rgb/{index}.png",
            "depth_path": f"depth/{index}.png",
            "host_received_timestamp_ns": (
                timestamp_ns + index * 33_000_000
            ),
            "capture_group_id": "mixed-rig",
            "hardware_sync_role": role,
            "hardware_sync_scope": "depth_exposure",
            "hardware_sync_transport": transport,
            "inter_cam_sync_mode_configured": mode,
            "inter_cam_sync_mode_readback": mode,
            "depth_sensor_timestamp_ns": (
                timestamp_ns + index * 33_000_000
            ),
            "depth_frame_number": index + 1,
            "depth_timestamp_domain": "global_time",
        }
        for index in range(record_count)
    ]
    (output / FRAME_METADATA_JSONL).write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    return {
        "name": f"capture-{device_id}",
        "device_id": device_id,
        "output_folder": output.as_posix(),
        "hardware_sync": {
            "implementation": "realsense_inter_cam_sync",
            "role": role,
            "group_id": "mixed-rig",
            "scope": "depth_exposure",
            "sensor_key": f"realsense_d435:{device_id}",
            "inter_cam_sync_mode_expected": mode,
            "max_depth_timestamp_skew_ms": max_skew_ms,
        },
    }


def test_hardware_sync_overlap_requires_a_complete_group_within_skew(
    tmp_path: Path,
) -> None:
    commands = [
        _write_hardware_sync_overlap_sensor(
            tmp_path,
            device_id="master",
            role="master",
            timestamp_ns=10_000_000,
        ),
        _write_hardware_sync_overlap_sensor(
            tmp_path,
            device_id="wrist",
            role="subordinate",
            timestamp_ns=11_000_000,
        ),
    ]

    evidence = _hardware_sync_overlap_evidence(commands)

    assert evidence is not None
    assert evidence["master_sensor_key"] == "realsense_d435:master"
    assert evidence["observed_max_abs_depth_timestamp_skew_ns"] == 1_000_000
    assert evidence["observed_max_depth_timestamp_span_ns"] == 1_000_000
    assert evidence["observed_consecutive_group_count"] == 3
    assert all(
        group["depth_timestamp_span_ns"] == 1_000_000
        for group in evidence["groups"]
    )
    assert all(
        set(group["frames"])
        == {
            "realsense_d435:master",
            "realsense_d435:wrist",
        }
        for group in evidence["groups"]
    )

    wrist_metadata = (
        tmp_path / "realsense_wrist" / FRAME_METADATA_JSONL
    )
    wrist_records = [
        json.loads(line) for line in wrist_metadata.read_text().splitlines()
    ]
    for wrist_record in wrist_records:
        wrist_record["depth_sensor_timestamp_ns"] += 3_000_001
    wrist_metadata.write_text(
        "".join(json.dumps(record) + "\n" for record in wrist_records)
    )
    assert _hardware_sync_overlap_evidence(commands) is None

    for index, wrist_record in enumerate(wrist_records):
        wrist_record["depth_sensor_timestamp_ns"] = (
            11_000_000 + index * 33_000_000
        )
    wrist_records[1]["depth_frame_number"] = 7
    wrist_metadata.write_text(
        "".join(json.dumps(record) + "\n" for record in wrist_records)
    )
    assert _hardware_sync_overlap_evidence(commands) is None


def test_hardware_sync_overlap_enforces_full_three_camera_timestamp_span(
    tmp_path: Path,
) -> None:
    commands = [
        _write_hardware_sync_overlap_sensor(
            tmp_path,
            device_id="master",
            role="master",
            timestamp_ns=10_000_000,
            max_skew_ms=2.0,
        ),
        _write_hardware_sync_overlap_sensor(
            tmp_path,
            device_id="early",
            role="subordinate",
            timestamp_ns=8_500_000,
            max_skew_ms=2.0,
        ),
        _write_hardware_sync_overlap_sensor(
            tmp_path,
            device_id="late",
            role="subordinate",
            timestamp_ns=11_500_000,
            max_skew_ms=2.0,
        ),
    ]

    # Both subordinates are individually within 2 ms of the master, but the
    # complete group spans 3 ms and therefore must not authorize robot START.
    assert _hardware_sync_overlap_evidence(commands) is None


def test_hardware_sync_overlap_rejects_wrong_transport_and_unsafe_skew(
    tmp_path: Path,
) -> None:
    commands = [
        _write_hardware_sync_overlap_sensor(
            tmp_path,
            device_id="master",
            role="master",
            timestamp_ns=10_000_000,
        ),
        _write_hardware_sync_overlap_sensor(
            tmp_path,
            device_id="wrist",
            role="subordinate",
            timestamp_ns=11_000_000,
            transport="host_clock",
        ),
    ]

    with pytest.raises(RuntimeError, match="hardware_sync_transport"):
        _hardware_sync_overlap_evidence(commands)

    commands[1]["hardware_sync"]["max_depth_timestamp_skew_ms"] = 6.0
    with pytest.raises(RuntimeError, match="assignments disagree|within"):
        _hardware_sync_overlap_evidence(commands)


def fake_realsense_status(*device_ids: str) -> dict:
    status = fake_sensor_status()
    status["families"][0]["devices"] = [
        {
            "device_id": device_id,
            "display_name": f"RealSense {device_id}",
            "connected": True,
            "capture_ready": True,
        }
        for device_id in device_ids
    ]
    return status


def test_capture_execution_plan_selects_full_capture_roles(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
    )
    write_run_config(run_root, config)

    plan = build_capture_execution_plan(
        run_root,
        allow_cameras=True,
        allow_real_robot=True,
        collect_sensors=fake_sensor_status,
    )

    assert plan["schema_version"] == "capture_execution_plan.v1"
    assert plan["status"] == "ok"
    assert plan["mode"] == "full"
    assert plan["ready_to_execute"] is True
    assert plan["preflight_status"] == "ok"
    assert plan["selected_roles"] == ["sensor_capture", "robot_pose_receiver"]
    assert [command["role"] for command in plan["selected_commands"]] == [
        "sensor_capture",
        "robot_pose_receiver",
    ]
    assert plan["skipped_commands"] == []
    assert plan["selected_resources"] == ["camera", "disk_io", "robot_command"]
    gates = {gate["name"]: gate for gate in plan["gates"]}
    assert gates["camera_permission"]["status"] == "ok"
    assert gates["capture_plan_preflight"]["status"] == "ok"
    assert plan["execution_strategy"][
        "camera_metadata_idle_timeout_s"
    ] == 2.0
    assert plan["execution_strategy"][
        "camera_metadata_idle_timeout_source"
    ] == "capture_fps_derived"


def test_capture_execution_plan_blocks_until_both_permissions_are_allowed(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
    )
    write_run_config(run_root, config)

    plan = build_capture_execution_plan(
        run_root,
        include_sensor_status=False,
    )

    assert plan["status"] == "error"
    assert plan["ready_to_execute"] is False
    assert [command["role"] for command in plan["selected_commands"]] == [
        "sensor_capture",
        "robot_pose_receiver",
    ]
    gates = {gate["name"]: gate for gate in plan["gates"]}
    assert gates["camera_permission"]["status"] == "error"
    assert gates["real_robot_permission"]["status"] == "error"


@pytest.mark.parametrize(
    ("allow_cameras", "allow_real_robot", "blocked_gate"),
    [
        (False, True, "camera_permission"),
        (True, False, "real_robot_permission"),
    ],
)
def test_capture_execution_plan_blocks_when_either_permission_is_absent(
    tmp_path: Path,
    allow_cameras: bool,
    allow_real_robot: bool,
    blocked_gate: str,
) -> None:
    run_root = tmp_path / blocked_gate
    write_run_config(
        run_root,
        create_run_config(
            run_root=run_root,
            sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
        ),
    )

    plan = build_capture_execution_plan(
        run_root,
        allow_cameras=allow_cameras,
        allow_real_robot=allow_real_robot,
        collect_sensors=fake_sensor_status,
    )

    gates = {gate["name"]: gate for gate in plan["gates"]}
    assert plan["ready_to_execute"] is False
    assert gates[blocked_gate]["status"] == "error"


def test_capture_execution_plan_writes_manifest_artifact(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
    )
    write_run_config(run_root, config)

    path, plan = write_capture_execution_plan_with_manifest(
        run_root,
        allow_cameras=True,
        allow_real_robot=True,
        collect_sensors=fake_sensor_status,
    )

    assert path == run_root / CAPTURE_EXECUTION_PLAN
    assert plan["status"] == "ok"
    assert (run_root / CAPTURE_PLAN).is_file()
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    stage = next(
        stage for stage in manifest["stages"] if stage["name"] == "capture_execution_plan"
    )
    assert stage["status"] == "succeeded"
    assert stage["artifacts"][CAPTURE_EXECUTION_PLAN] == CAPTURE_EXECUTION_PLAN
    assert stage["artifacts"][CAPTURE_PLAN] == CAPTURE_PLAN


def test_capture_execution_plan_cli_writes_artifact(tmp_path: Path) -> None:
    run_root = tmp_path / "run-cli"
    repo_root = Path(__file__).resolve().parents[1]
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
    )
    write_run_config(run_root, config)

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_capture_execution_plan.py",
            run_root.as_posix(),
            "--allow-cameras",
            "--allow-real-robot",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert f"Wrote {run_root / CAPTURE_EXECUTION_PLAN}" in result.stdout
    assert "Capture execution plan: warning (full)" in result.stdout
    data = json.loads((run_root / CAPTURE_EXECUTION_PLAN).read_text())
    assert data["selected_roles"] == ["sensor_capture", "robot_pose_receiver"]


@pytest.mark.parametrize(
    ("allow_cameras", "allow_real_robot"),
    [
        (False, True),
        (True, False),
        (1, True),
        (True, 1),
        ("true", True),
        (True, "true"),
    ],
)
def test_capture_execution_rejects_nonliteral_gates_before_any_mutation(
    tmp_path: Path,
    monkeypatch,
    allow_cameras,
    allow_real_robot,
) -> None:
    run_root, _config = configured_run(tmp_path, "strict-boundary")
    manifest_path = run_root / DATASET_MANIFEST
    manifest_path.write_text('{"sentinel": true}\n')
    before = filesystem_snapshot(run_root)

    def forbidden_discovery():
        raise AssertionError("permission rejection must precede sensor discovery")

    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution.subprocess.Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("permission rejection must precede process startup")
        ),
    )

    with pytest.raises(CaptureExecutionPermissionError, match="fresh strict"):
        run_capture_execution(
            run_root,
            allow_cameras=allow_cameras,
            allow_real_robot=allow_real_robot,
            collect_sensors=forbidden_discovery,
        )

    assert filesystem_snapshot(run_root) == before


@pytest.mark.parametrize("blocker", ["raw_pose", "sensor_folder"])
def test_capture_execution_rejects_existing_raw_outputs_before_discovery_or_mutation(
    tmp_path: Path,
    monkeypatch,
    blocker: str,
) -> None:
    run_root, config = configured_run(tmp_path, f"existing-{blocker}")
    canonical_plan = build_capture_plan(config)
    if blocker == "raw_pose":
        (run_root / RAW_ROBOT_EE_POSES).write_text('{"preserve": true}\n')
    else:
        sensor_command = next(
            command
            for command in canonical_plan.commands
            if command.role == "sensor_capture"
        )
        assert sensor_command.output_folder is not None
        Path(sensor_command.output_folder).mkdir(parents=True)
    before = filesystem_snapshot(run_root)

    def forbidden_discovery():
        raise AssertionError("raw output rejection must precede sensor discovery")

    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution.subprocess.Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("raw output rejection must precede process startup")
        ),
    )

    with pytest.raises(FileExistsError, match="unused raw output paths"):
        run_capture_execution(
            run_root,
            allow_cameras=True,
            allow_real_robot=True,
            collect_sensors=forbidden_discovery,
        )

    assert filesystem_snapshot(run_root) == before
    assert not (run_root / CAPTURE_EXECUTION_PLAN).exists()
    assert not (run_root / CAPTURE_EXECUTION_STATUS).exists()
    assert not (run_root / CAPTURE_EXECUTION_REPORT).exists()
    assert not (run_root / CAPTURE_EXECUTION_LOGS_DIR).exists()


def test_capture_execution_validates_live_sensor_preflight_before_supervisor_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, _config = configured_run(tmp_path, "sensor-preflight-first")
    before = filesystem_snapshot(run_root)

    def failed_discovery():
        raise RuntimeError("sensor discovery failed before acceptance")

    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution.subprocess.Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("failed preflight must not start a process")
        ),
    )

    with pytest.raises(RuntimeError, match="sensor discovery failed"):
        run_capture_execution(
            run_root,
            allow_cameras=True,
            allow_real_robot=True,
            collect_sensors=failed_discovery,
        )

    assert filesystem_snapshot(run_root) == before
    assert not (run_root / CAPTURE_EXECUTION_PLAN).exists()
    assert not (run_root / CAPTURE_PLAN).exists()
    assert not (run_root / CAPTURE_EXECUTION_STATUS).exists()
    assert not (run_root / CAPTURE_EXECUTION_REPORT).exists()
    assert not (run_root / CAPTURE_EXECUTION_LOGS_DIR).exists()
    assert not (run_root / DATASET_MANIFEST).exists()


@pytest.mark.parametrize("timeout_s", [0, 5.001, float("inf"), True])
def test_capture_execution_rejects_unsafe_camera_metadata_timeout(
    tmp_path: Path,
    timeout_s,
) -> None:
    run_root, _config = configured_run(
        tmp_path,
        "invalid-camera-metadata-timeout",
    )
    before = filesystem_snapshot(run_root)

    def forbidden_discovery():
        raise AssertionError(
            "camera metadata timeout validation must precede discovery"
        )

    with pytest.raises(
        ValueError,
        match="camera_metadata_idle_timeout_s",
    ):
        run_capture_execution(
            run_root,
            allow_cameras=True,
            allow_real_robot=True,
            camera_metadata_idle_timeout_s=timeout_s,
            collect_sensors=forbidden_discovery,
        )

    assert filesystem_snapshot(run_root) == before


@pytest.mark.parametrize("tampered_role", ["sensor_capture", "robot_pose_receiver"])
def test_capture_execution_rejects_any_noncanonical_persisted_command_before_mutation(
    tmp_path: Path,
    monkeypatch,
    tampered_role: str,
) -> None:
    run_root, config = configured_run(tmp_path, f"tampered-{tampered_role}")
    plan_path = write_capture_plan(run_root, build_capture_plan(config))
    persisted = json.loads(plan_path.read_text())
    command = next(
        item for item in persisted["commands"] if item["role"] == tampered_role
    )
    command["command"][3] = "scripts/tampered_capture_command.py"
    plan_path.write_text(json.dumps(persisted, indent=2) + "\n")
    before = filesystem_snapshot(run_root)

    def forbidden_discovery():
        raise AssertionError("command identity rejection must precede discovery")

    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution.subprocess.Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("tampered command must not start a process")
        ),
    )

    with pytest.raises(ValueError, match="exactly match the canonical commands"):
        run_capture_execution(
            run_root,
            allow_cameras=True,
            allow_real_robot=True,
            collect_sensors=forbidden_discovery,
        )

    assert filesystem_snapshot(run_root) == before


def test_capture_execution_rejects_stale_plan_after_camera_is_disabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "disabled-after-plan"
    config = create_run_config(
        run_root=run_root,
        sensors=(
            sensor_config_from_token("realsense:working:eye_in_hand:Working"),
            sensor_config_from_token("realsense:offline:eye_in_hand:Offline"),
        ),
    )
    write_run_config(run_root, config)
    write_capture_plan(run_root, build_capture_plan(config.to_dict()))

    updated_sensors = (
        config.capture.sensors[0],
        replace(config.capture.sensors[1], enabled=False),
    )
    updated_config = replace(
        config,
        capture=replace(config.capture, sensors=updated_sensors),
    )
    write_run_config(run_root, updated_config)
    before = filesystem_snapshot(run_root)

    def forbidden_discovery():
        raise AssertionError("stale-plan rejection must precede sensor discovery")

    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution.subprocess.Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stale capture plan must not start a process")
        ),
    )

    with pytest.raises(ValueError, match="exactly match the canonical commands"):
        run_capture_execution(
            run_root,
            allow_cameras=True,
            allow_real_robot=True,
            collect_sensors=forbidden_discovery,
        )

    assert filesystem_snapshot(run_root) == before


def test_capture_execution_full_mode_stops_sensor_process_after_receiver(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "run-full-execute"
    config = create_run_config(
        run_root=run_root,
        sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
    )
    write_run_config(run_root, config)
    background_commands: list[list[str]] = []
    receiver_commands: list[list[str]] = []
    terminated_commands: list[list[str]] = []

    def fake_popen(command, **kwargs):
        if any(item.endswith("pose_receiver_udp_json.py") for item in command):
            receiver_commands.append(list(command))
            (run_root / RAW_ROBOT_EE_POSES).write_text(
                json.dumps({"0": {"motion": "circ_far", "pose": {"X": 1}}})
            )
            return FakeBackgroundProcess(list(command), kwargs["stdout"])
        background_commands.append(list(command))
        if any(item.endswith("capture_realsense_720p.py") for item in command):
            mark_sensor_ready(run_root)
            return FakePersistentProcess(list(command), kwargs["stdout"])
        return FakeBackgroundProcess(list(command), kwargs["stdout"])

    def fake_terminate(process, *, timeout_s):
        terminated_commands.append(list(process.command))
        process.returncode = -15
        process.log_file.write("fake supervisor stopped process\n")

    monkeypatch.setattr("posetestbot.pipeline.capture_execution.subprocess.Popen", fake_popen)
    monkeypatch.setattr("posetestbot.pipeline.capture_execution.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution._terminate_process_group",
        fake_terminate,
    )

    report_path, report = run_capture_execution(
        run_root,
        allow_cameras=True,
        allow_real_robot=True,
        collect_sensors=fake_sensor_status,
        timeout_s=5,
        receive_start_timeout_s=11,
        receive_idle_timeout_s=7,
    )

    assert report_path == run_root / CAPTURE_EXECUTION_REPORT
    assert report["status"] == "succeeded"
    assert report["mode"] == "full"
    assert report["capture_execution_plan"]["selected_roles"] == [
        "sensor_capture",
        "robot_pose_receiver",
    ]
    processes = {process["role"]: process for process in report["processes"]}
    assert processes["sensor_capture"]["status"] == "stopped"
    assert processes["sensor_capture"]["pid"] == 23456
    assert processes["sensor_capture"]["started_at"]
    assert processes["sensor_capture"]["ended_at"]
    assert processes["sensor_capture"]["elapsed_s"] >= 0
    assert processes["sensor_capture"]["termination_reason"] == (
        "stopped_after_receiver_exit"
    )
    assert processes["robot_pose_receiver"]["termination_reason"] == (
        "receiver_completed"
    )
    assert "--allow-cameras" in processes["robot_pose_receiver"]["command"]
    assert "--allow-real-robot" in processes["robot_pose_receiver"]["command"]
    assert terminated_commands
    assert any(
        any(item.endswith("capture_realsense_720p.py") for item in command)
        for command in terminated_commands
    )
    assert any(
        any(item.endswith("capture_realsense_720p.py") for item in command)
        for command in background_commands
    )
    assert receiver_commands[0][:4] == [
        "uv",
        "run",
        "python",
        "scripts/pose_receiver_udp_json.py",
    ]
    assert "--allow-cameras" in receiver_commands[0]
    assert "--allow-real-robot" in receiver_commands[0]
    assert receiver_commands[0][
        receiver_commands[0].index("--receive-start-timeout-s") + 1
    ] == "11"
    assert receiver_commands[0][
        receiver_commands[0].index("--receive-idle-timeout-s") + 1
    ] == "7"
    assert report["receive_start_timeout_s"] == 11
    assert report["receive_idle_timeout_s"] == 7
    assert report["camera_metadata_idle_timeout_s"] == 2.0
    assert report["camera_metadata_idle_timeout_source"] == (
        "capture_fps_derived"
    )
    persisted_status = json.loads(
        (run_root / CAPTURE_EXECUTION_STATUS).read_text()
    )
    assert persisted_status["receive_idle_timeout_s"] == 7
    assert persisted_status["camera_metadata_idle_timeout_s"] == 2.0
    assert persisted_status["camera_metadata_idle_timeout_source"] == (
        "capture_fps_derived"
    )
    planned_receiver = next(
        command
        for command in report["capture_execution_plan"]["selected_commands"]
        if command["role"] == "robot_pose_receiver"
    )
    assert "--allow-cameras" not in planned_receiver["command"]
    assert "--allow-real-robot" not in planned_receiver["command"]
    persisted_plan = json.loads((run_root / CAPTURE_PLAN).read_text())
    persisted_receiver = next(
        command
        for command in persisted_plan["commands"]
        if command["role"] == "robot_pose_receiver"
    )
    assert "--allow-cameras" not in persisted_receiver["command"]
    assert "--allow-real-robot" not in persisted_receiver["command"]


def test_capture_execution_starts_cameras_sequentially_after_each_is_ready(
    tmp_path: Path,
    monkeypatch,
) -> None:
    device_ids = ("111", "222", "333")
    run_root = tmp_path / "sequential-camera-startup"
    write_run_config(
        run_root,
        create_run_config(
            run_root=run_root,
            sensors=tuple(
                sensor_config_from_token(
                    f"realsense:{device_id}:eye_in_hand:RealSense {device_id}"
                )
                for device_id in device_ids
            ),
        ),
    )
    events: list[str] = []
    waiting_for_readiness: list[str] = []

    def fake_popen(command, **kwargs):
        if any(item.endswith("pose_receiver_udp_json.py") for item in command):
            assert not waiting_for_readiness
            assert all(
                (run_root / f"realsense_{device_id}" / FRAME_METADATA_JSONL).is_file()
                for device_id in device_ids
            )
            events.append("receiver")
            (run_root / RAW_ROBOT_EE_POSES).write_text(
                json.dumps({"0": {"motion": "circ_far", "pose": {"X": 1}}})
            )
            return FakeBackgroundProcess(list(command), kwargs["stdout"])
        device_id = command[command.index("--device") + 1]
        assert not waiting_for_readiness
        events.append(f"start:{device_id}")
        waiting_for_readiness.append(device_id)
        return FakePersistentProcess(list(command), kwargs["stdout"])

    def fake_sleep(_delay):
        if not waiting_for_readiness:
            return
        device_id = waiting_for_readiness.pop()
        mark_sensor_ready(run_root, device_id=device_id)
        events.append(f"ready:{device_id}")

    def fake_terminate(process, *, timeout_s):
        del timeout_s
        process.returncode = -15

    monkeypatch.setattr("posetestbot.pipeline.capture_execution.subprocess.Popen", fake_popen)
    monkeypatch.setattr("posetestbot.pipeline.capture_execution.time.sleep", fake_sleep)
    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution._terminate_process_group",
        fake_terminate,
    )

    _report_path, report = run_capture_execution(
        run_root,
        allow_cameras=True,
        allow_real_robot=True,
        collect_sensors=lambda: fake_realsense_status(*device_ids),
        timeout_s=5,
    )

    assert report["status"] == "succeeded"
    assert events == [
        "start:111",
        "ready:111",
        "start:222",
        "ready:222",
        "start:333",
        "ready:333",
        "receiver",
    ]
    camera_processes = [
        process for process in report["processes"] if process["role"] == "sensor_capture"
    ]
    assert [process["name"] for process in camera_processes] == [
        "realsense_111",
        "realsense_222",
        "realsense_333",
    ]
    assert [process["startup_attempt"] for process in camera_processes] == [1, 1, 1]
    assert all(process["readiness_record_count"] == 3 for process in camera_processes)
    strategy = report["capture_execution_plan"]["execution_strategy"]
    assert strategy["camera_startup_attempts"] == 3
    assert "start one sensor child" in strategy["start_order"]
    assert report["camera_readiness_contract"]["deadline_scope"] == (
        "per_camera_startup_attempt"
    )
    assert "never sends an iiwa STOP" in report["robot_stop_policy"]


def test_capture_execution_retries_pristine_camera_startup_then_succeeds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, _config = configured_run(tmp_path, "camera-retry-success")
    camera_spawn_count = 0
    receiver_spawn_count = 0

    def fake_popen(command, **kwargs):
        nonlocal camera_spawn_count, receiver_spawn_count
        if any(item.endswith("pose_receiver_udp_json.py") for item in command):
            receiver_spawn_count += 1
            assert camera_spawn_count == 2
            (run_root / RAW_ROBOT_EE_POSES).write_text(
                json.dumps({"0": {"motion": "circ_far", "pose": {"X": 1}}})
            )
            return FakeBackgroundProcess(list(command), kwargs["stdout"])
        camera_spawn_count += 1
        if camera_spawn_count == 2:
            mark_sensor_ready(run_root)
        return FakePersistentProcess(list(command), kwargs["stdout"])

    def fake_terminate(process, *, timeout_s):
        del timeout_s
        process.returncode = -15

    monkeypatch.setattr("posetestbot.pipeline.capture_execution.subprocess.Popen", fake_popen)
    monkeypatch.setattr("posetestbot.pipeline.capture_execution.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution._terminate_process_group",
        fake_terminate,
    )

    _report_path, report = run_capture_execution(
        run_root,
        allow_cameras=True,
        allow_real_robot=True,
        collect_sensors=fake_sensor_status,
        timeout_s=5,
        startup_wait_s=0,
        camera_startup_retry_delay_s=0,
    )

    assert report["status"] == "succeeded"
    assert camera_spawn_count == 2
    assert receiver_spawn_count == 1
    camera_processes = [
        process for process in report["processes"] if process["role"] == "sensor_capture"
    ]
    assert [process["startup_attempt"] for process in camera_processes] == [1, 2]
    assert camera_processes[0]["termination_reason"] == (
        "startup_readiness_timeout_retry"
    )
    assert camera_processes[0]["output_mutated"] is False
    assert camera_processes[1]["readiness_record_count"] == 3
    assert len({process["log_file"] for process in camera_processes}) == 2
    assert camera_processes[0]["log_file"].endswith("_attempt_01.log")
    assert camera_processes[1]["log_file"].endswith("_attempt_02.log")


def test_capture_execution_exhausts_bounded_pristine_startup_retries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, _config = configured_run(tmp_path, "camera-retry-exhausted")
    camera_spawn_count = 0
    receiver_commands: list[list[str]] = []

    def fake_popen(command, **kwargs):
        nonlocal camera_spawn_count
        if any(item.endswith("pose_receiver_udp_json.py") for item in command):
            receiver_commands.append(list(command))
            raise AssertionError("receiver must not start after retry exhaustion")
        camera_spawn_count += 1
        process = FakePersistentProcess(list(command), kwargs["stdout"])
        process.returncode = 7
        return process

    monkeypatch.setattr("posetestbot.pipeline.capture_execution.subprocess.Popen", fake_popen)
    monkeypatch.setattr("posetestbot.pipeline.capture_execution.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="exhausted 3 startup attempt"):
        run_capture_execution(
            run_root,
            allow_cameras=True,
            allow_real_robot=True,
            collect_sensors=fake_sensor_status,
            camera_startup_retry_delay_s=0,
        )

    assert camera_spawn_count == 3
    assert receiver_commands == []
    report = json.loads((run_root / CAPTURE_EXECUTION_REPORT).read_text())
    camera_processes = [
        process for process in report["processes"] if process["role"] == "sensor_capture"
    ]
    assert [process["startup_attempt"] for process in camera_processes] == [1, 2, 3]
    assert [process["termination_reason"] for process in camera_processes] == [
        "startup_exit_retry",
        "startup_exit_retry",
        "exited_before_receiver_start",
    ]
    assert all(process["output_mutated"] is False for process in camera_processes)
    assert len({process["log_file"] for process in camera_processes}) == 3


def test_capture_execution_does_not_retry_after_partial_sensor_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, _config = configured_run(tmp_path, "camera-partial-no-retry")
    camera_spawn_count = 0
    receiver_commands: list[list[str]] = []

    def fake_popen(command, **kwargs):
        nonlocal camera_spawn_count
        if any(item.endswith("pose_receiver_udp_json.py") for item in command):
            receiver_commands.append(list(command))
            raise AssertionError("receiver must not start after partial camera output")
        camera_spawn_count += 1
        mark_sensor_ready(run_root, record_count=1)
        process = FakePersistentProcess(list(command), kwargs["stdout"])
        process.returncode = 9
        return process

    monkeypatch.setattr("posetestbot.pipeline.capture_execution.subprocess.Popen", fake_popen)
    monkeypatch.setattr("posetestbot.pipeline.capture_execution.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="preserving partial raw evidence"):
        run_capture_execution(
            run_root,
            allow_cameras=True,
            allow_real_robot=True,
            collect_sensors=fake_sensor_status,
            camera_startup_retry_delay_s=0,
        )

    assert camera_spawn_count == 1
    assert receiver_commands == []
    report = json.loads((run_root / CAPTURE_EXECUTION_REPORT).read_text())
    camera_processes = [
        process for process in report["processes"] if process["role"] == "sensor_capture"
    ]
    assert len(camera_processes) == 1
    assert camera_processes[0]["startup_attempt"] == 1
    assert camera_processes[0]["readiness_record_count"] == 1
    assert camera_processes[0]["output_mutated"] is True
    assert camera_processes[0]["termination_reason"] == (
        "startup_partial_output_no_retry"
    )
    assert (run_root / "realsense_123" / FRAME_METADATA_JSONL).is_file()


def test_capture_execution_never_starts_receiver_without_first_frame_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, _config = configured_run(tmp_path, "camera-not-ready")
    receiver_commands: list[list[str]] = []
    camera_processes: list[FakePersistentProcess] = []

    def fake_popen(command, **kwargs):
        if any(item.endswith("pose_receiver_udp_json.py") for item in command):
            receiver_commands.append(list(command))
            raise AssertionError("receiver must not start before camera readiness")
        process = FakePersistentProcess(list(command), kwargs["stdout"])
        camera_processes.append(process)
        return process

    def fake_terminate(process, *, timeout_s):
        del timeout_s
        process.returncode = -15

    monkeypatch.setattr("posetestbot.pipeline.capture_execution.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution._terminate_process_group",
        fake_terminate,
    )

    with pytest.raises(RuntimeError, match="readiness deadline expired before robot START"):
        run_capture_execution(
            run_root,
            allow_cameras=True,
            allow_real_robot=True,
            collect_sensors=fake_sensor_status,
            startup_wait_s=0,
            camera_startup_attempts=1,
        )

    assert receiver_commands == []
    assert len(camera_processes) == 1
    assert camera_processes[0].returncode == -15
    assert not (run_root / RAW_ROBOT_EE_POSES).exists()
    report = json.loads((run_root / CAPTURE_EXECUTION_REPORT).read_text())
    assert report["status"] == "failed"
    assert FRAME_METADATA_JSONL in report["message"]


def test_capture_execution_never_starts_receiver_with_only_one_metadata_record(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, _config = configured_run(tmp_path, "camera-one-frame-only")
    receiver_commands: list[list[str]] = []
    camera_processes: list[FakePersistentProcess] = []

    def fake_popen(command, **kwargs):
        if any(item.endswith("pose_receiver_udp_json.py") for item in command):
            receiver_commands.append(list(command))
            raise AssertionError("receiver must not start after only one camera frame")
        mark_sensor_ready(run_root, record_count=1)
        process = FakePersistentProcess(list(command), kwargs["stdout"])
        camera_processes.append(process)
        return process

    def fake_terminate(process, *, timeout_s):
        del timeout_s
        process.returncode = -15

    monkeypatch.setattr("posetestbot.pipeline.capture_execution.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution._terminate_process_group",
        fake_terminate,
    )

    with pytest.raises(RuntimeError, match="at least 3 valid committed"):
        run_capture_execution(
            run_root,
            allow_cameras=True,
            allow_real_robot=True,
            collect_sensors=fake_sensor_status,
            startup_wait_s=0,
            camera_startup_attempts=1,
        )

    assert receiver_commands == []
    assert len(camera_processes) == 1
    assert camera_processes[0].returncode == -15
    assert not (run_root / RAW_ROBOT_EE_POSES).exists()
    report = json.loads((run_root / CAPTURE_EXECUTION_REPORT).read_text())
    assert report["status"] == "failed"
    assert report["camera_readiness_contract"][
        "minimum_valid_committed_records"
    ] == 3


def configured_hardware_sync_run(tmp_path: Path, name: str) -> Path:
    run_root = tmp_path / name
    write_run_config(
        run_root,
        create_run_config(
            run_root=run_root,
            fps=6,
            sensors=(
                SensorRunConfig(
                    "realsense_d435",
                    "master",
                    "Static D435",
                    mounting_mode="static",
                ),
                SensorRunConfig(
                    "realsense_d435",
                    "wrist",
                    "Wrist D435",
                    mounting_mode="eye_in_hand",
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
    qualification_evidence = tmp_path / f"{name}-pulse-trace.csv"
    qualification_evidence.write_text("t,master,wrist\n0,1,1\n")
    record_hardware_sync_qualification(
        run_root,
        operator="pytest",
        method="pulsed_light",
        observed_max_depth_timestamp_skew_ms=1.0,
        evidence_paths=[qualification_evidence],
        confirm_passed=True,
    )
    return run_root


def test_hardware_sync_capture_never_starts_robot_without_sustained_overlap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "hardware-no-overlap"
    write_run_config(
        run_root,
        create_run_config(
            run_root=run_root,
            fps=6,
            sensors=(
                SensorRunConfig(
                    "realsense_d435",
                    "master",
                    "Static D435",
                    mounting_mode="static",
                ),
                SensorRunConfig(
                    "realsense_d435",
                    "wrist",
                    "Wrist D435",
                    mounting_mode="eye_in_hand",
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
    qualification_evidence = tmp_path / "pulse-trace.csv"
    qualification_evidence.write_text("t,master,wrist\n0,1,1\n")
    record_hardware_sync_qualification(
        run_root,
        operator="pytest",
        method="pulsed_light",
        observed_max_depth_timestamp_skew_ms=1.0,
        evidence_paths=[qualification_evidence],
        confirm_passed=True,
    )
    receiver_commands: list[list[str]] = []
    camera_processes: list[FakePersistentProcess] = []

    def fake_popen(command, **kwargs):
        if any(item.endswith("pose_receiver_udp_json.py") for item in command):
            receiver_commands.append(list(command))
            raise AssertionError("robot receiver must not start without overlap")
        device_id = command[command.index("--device") + 1]
        role = command[command.index("--hardware-sync-role") + 1]
        _write_hardware_sync_overlap_sensor(
            run_root,
            device_id=device_id,
            role=role,
            timestamp_ns=(
                10_000_000 if role == "master" else 20_000_000
            ),
        )
        process = FakePersistentProcess(list(command), kwargs["stdout"])
        camera_processes.append(process)
        return process

    def fake_terminate(process, *, timeout_s):
        del timeout_s
        process.returncode = -15

    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution.subprocess.Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution.time.sleep",
        lambda _delay: None,
    )
    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution._terminate_process_group",
        fake_terminate,
    )

    with pytest.raises(RuntimeError, match="overlap deadline expired"):
        run_capture_execution(
            run_root,
            allow_cameras=True,
            allow_real_robot=True,
            collect_sensors=lambda: fake_realsense_status("master", "wrist"),
            startup_wait_s=0,
            camera_startup_attempts=1,
        )

    assert receiver_commands == []
    assert len(camera_processes) == 2
    assert all(process.returncode == -15 for process in camera_processes)
    assert not (run_root / RAW_ROBOT_EE_POSES).exists()
    report = json.loads((run_root / CAPTURE_EXECUTION_REPORT).read_text())
    assert report["status"] == "failed"
    assert report["hardware_sync_start_evidence"] is None


def test_hardware_sync_capture_revalidates_run_config_digest_before_receiver(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = configured_hardware_sync_run(
        tmp_path,
        "hardware-config-changed-after-cameras",
    )
    receiver_commands: list[list[str]] = []
    camera_processes: list[FakePersistentProcess] = []

    def fake_popen(command, **kwargs):
        if any(item.endswith("pose_receiver_udp_json.py") for item in command):
            receiver_commands.append(list(command))
            raise AssertionError(
                "receiver must not start after hardware contract mutation"
            )
        device_id = command[command.index("--device") + 1]
        role = command[command.index("--hardware-sync-role") + 1]
        _write_hardware_sync_overlap_sensor(
            run_root,
            device_id=device_id,
            role=role,
            timestamp_ns=(
                10_000_000 if role == "master" else 11_000_000
            ),
        )
        if device_id == "wrist":
            config = json.loads((run_root / RUN_CONFIG).read_text())
            config["capture"]["synchronization"][
                "max_depth_timestamp_skew_ms"
            ] = 1.5
            (run_root / RUN_CONFIG).write_text(json.dumps(config))
        process = FakePersistentProcess(list(command), kwargs["stdout"])
        camera_processes.append(process)
        return process

    def fake_terminate(process, *, timeout_s):
        del timeout_s
        process.returncode = -15

    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution.subprocess.Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution._terminate_process_group",
        fake_terminate,
    )

    with pytest.raises(RuntimeError, match="contract digest changed"):
        run_capture_execution(
            run_root,
            allow_cameras=True,
            allow_real_robot=True,
            collect_sensors=lambda: fake_realsense_status("master", "wrist"),
            camera_startup_attempts=1,
        )

    assert receiver_commands == []
    assert len(camera_processes) == 2
    assert all(process.returncode == -15 for process in camera_processes)
    assert not (run_root / RAW_ROBOT_EE_POSES).exists()
    report = json.loads((run_root / CAPTURE_EXECUTION_REPORT).read_text())
    assert report["status"] == "failed"
    assert report["hardware_sync_start_evidence"] is not None
    assert report["hardware_sync_execution_binding"][
        "revalidated_immediately_before_receiver_spawn"
    ] is False


def test_hardware_sync_capture_revalidates_unchanged_binding_and_starts_receiver(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = configured_hardware_sync_run(
        tmp_path,
        "hardware-binding-unchanged",
    )
    receiver_commands: list[list[str]] = []
    camera_processes: list[FakePersistentProcess] = []

    def fake_popen(command, **kwargs):
        if any(item.endswith("pose_receiver_udp_json.py") for item in command):
            receiver_commands.append(list(command))
            (run_root / RAW_ROBOT_EE_POSES).write_text(
                json.dumps({"0": {"motion": "circ_far", "pose": {"X": 1}}})
            )
            return FakeBackgroundProcess(list(command), kwargs["stdout"])
        device_id = command[command.index("--device") + 1]
        role = command[command.index("--hardware-sync-role") + 1]
        _write_hardware_sync_overlap_sensor(
            run_root,
            device_id=device_id,
            role=role,
            timestamp_ns=(
                10_000_000 if role == "master" else 11_000_000
            ),
        )
        process = FakePersistentProcess(list(command), kwargs["stdout"])
        camera_processes.append(process)
        return process

    def fake_terminate(process, *, timeout_s):
        del timeout_s
        process.returncode = -15

    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution.subprocess.Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution._terminate_process_group",
        fake_terminate,
    )

    _report_path, report = run_capture_execution(
        run_root,
        allow_cameras=True,
        allow_real_robot=True,
        collect_sensors=lambda: fake_realsense_status("master", "wrist"),
        camera_startup_attempts=1,
    )

    assert len(receiver_commands) == 1
    assert len(camera_processes) == 2
    assert all(process.returncode == -15 for process in camera_processes)
    assert report["status"] == "succeeded"
    assert report["hardware_sync_execution_binding"][
        "revalidated_immediately_before_receiver_spawn"
    ] is True
    assert report["hardware_sync_start_evidence"][
        "observed_max_depth_timestamp_span_ns"
    ] == 1_000_000


def test_hardware_sync_capture_revalidates_qualification_before_receiver(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = configured_hardware_sync_run(
        tmp_path,
        "hardware-qualification-changed-after-cameras",
    )
    original_qualification = json.loads(
        (run_root / HARDWARE_SYNC_QUALIFICATION).read_text()
    )
    receiver_commands: list[list[str]] = []
    camera_processes: list[FakePersistentProcess] = []

    def fake_popen(command, **kwargs):
        if any(item.endswith("pose_receiver_udp_json.py") for item in command):
            receiver_commands.append(list(command))
            raise AssertionError(
                "receiver must not start after qualification replacement"
            )
        device_id = command[command.index("--device") + 1]
        role = command[command.index("--hardware-sync-role") + 1]
        _write_hardware_sync_overlap_sensor(
            run_root,
            device_id=device_id,
            role=role,
            timestamp_ns=(
                10_000_000 if role == "master" else 11_000_000
            ),
        )
        if device_id == "wrist":
            # The supported recorder now shares the run-config transaction
            # and refuses publication once capture evidence exists. Simulate
            # an out-of-band file edit to prove the final pre-Popen validation
            # still detects a changed, otherwise structurally valid artifact.
            replacement = json.loads(
                (run_root / HARDWARE_SYNC_QUALIFICATION).read_text()
            )
            replacement["operator"] = "out-of-band-replacement"
            (run_root / HARDWARE_SYNC_QUALIFICATION).write_text(
                json.dumps(replacement)
            )
        process = FakePersistentProcess(list(command), kwargs["stdout"])
        camera_processes.append(process)
        return process

    def fake_terminate(process, *, timeout_s):
        del timeout_s
        process.returncode = -15

    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution.subprocess.Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution._terminate_process_group",
        fake_terminate,
    )

    with pytest.raises(RuntimeError, match="qualification changed"):
        run_capture_execution(
            run_root,
            allow_cameras=True,
            allow_real_robot=True,
            collect_sensors=lambda: fake_realsense_status("master", "wrist"),
            camera_startup_attempts=1,
        )

    assert receiver_commands == []
    assert len(camera_processes) == 2
    assert all(process.returncode == -15 for process in camera_processes)
    assert not (run_root / RAW_ROBOT_EE_POSES).exists()
    current_qualification = json.loads(
        (run_root / HARDWARE_SYNC_QUALIFICATION).read_text()
    )
    assert current_qualification != original_qualification
    report = json.loads((run_root / CAPTURE_EXECUTION_REPORT).read_text())
    assert report["status"] == "failed"
    assert report["hardware_sync_start_evidence"] is not None
    assert report["hardware_sync_execution_binding"][
        "revalidated_immediately_before_receiver_spawn"
    ] is False


def test_capture_execution_monitors_camera_exit_during_readiness_window(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, _config = configured_run(tmp_path, "camera-exits-before-ready")
    receiver_commands: list[list[str]] = []

    def fake_popen(command, **kwargs):
        if any(item.endswith("pose_receiver_udp_json.py") for item in command):
            receiver_commands.append(list(command))
            raise AssertionError("receiver must not start after early camera exit")
        process = FakePersistentProcess(list(command), kwargs["stdout"])
        process.returncode = 0
        return process

    monkeypatch.setattr("posetestbot.pipeline.capture_execution.subprocess.Popen", fake_popen)

    with pytest.raises(RuntimeError, match="exited before first-frame readiness"):
        run_capture_execution(
            run_root,
            allow_cameras=True,
            allow_real_robot=True,
            collect_sensors=fake_sensor_status,
            camera_startup_attempts=1,
        )

    assert receiver_commands == []
    report = json.loads((run_root / CAPTURE_EXECUTION_REPORT).read_text())
    camera = next(
        process for process in report["processes"] if process["role"] == "sensor_capture"
    )
    assert camera["status"] == "failed"
    assert camera["returncode"] == 0
    assert camera["termination_reason"] == "exited_before_receiver_start"


def test_capture_execution_aborts_alive_camera_with_stalled_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, _config = configured_run(tmp_path, "camera-metadata-stalls")
    spawned: list[FakePersistentProcess] = []

    def fake_popen(command, **kwargs):
        if not any(
            item.endswith("pose_receiver_udp_json.py") for item in command
        ):
            mark_sensor_ready(run_root)
        process = FakePersistentProcess(list(command), kwargs["stdout"])
        spawned.append(process)
        return process

    def fake_terminate(process, *, timeout_s):
        del timeout_s
        process.returncode = -15

    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution.subprocess.Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution._terminate_process_group",
        fake_terminate,
    )

    with pytest.raises(RuntimeError, match="stopped advancing"):
        run_capture_execution(
            run_root,
            allow_cameras=True,
            allow_real_robot=True,
            collect_sensors=fake_sensor_status,
            timeout_s=2,
            camera_metadata_idle_timeout_s=0.01,
        )

    assert len(spawned) == 2
    assert all(process.returncode == -15 for process in spawned)
    assert (
        run_root / "realsense_123" / FRAME_METADATA_JSONL
    ).is_file()
    report = json.loads((run_root / CAPTURE_EXECUTION_REPORT).read_text())
    camera = next(
        process
        for process in report["processes"]
        if process["role"] == "sensor_capture"
    )
    assert report["status"] == "failed"
    assert camera["status"] == "failed"
    assert camera["termination_reason"] == "camera_metadata_stalled"
    assert camera["metadata_record_count"] == 3
    assert camera["metadata_last_frame_index"] == 2
    assert camera["metadata_idle_elapsed_s"] >= 0.01
    assert camera["metadata_idle_timeout_s"] == 0.01
    assert not (run_root / RAW_ROBOT_EE_POSES).exists()


def test_capture_execution_rejects_camera_stall_before_short_receiver_finishes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, _config = configured_run(
        tmp_path,
        "camera-stalls-before-receiver-finishes",
    )
    clock = {"monotonic": 0.0}
    camera_processes: list[FakePersistentProcess] = []

    class FakeReceiverFinishesAfterCameraDeadline(FakeBackgroundProcess):
        def wait(self, timeout=None):
            del timeout
            # The robot-side command finishes well within its independent
            # default 60-second packet-idle limit, but after the FPS-derived
            # camera metadata freshness deadline.
            clock["monotonic"] += 3.0
            self.returncode = 0
            return 0

    def fake_popen(command, **kwargs):
        if any(item.endswith("pose_receiver_udp_json.py") for item in command):
            return FakeReceiverFinishesAfterCameraDeadline(
                list(command),
                kwargs["stdout"],
            )
        mark_sensor_ready(run_root)
        process = FakePersistentProcess(list(command), kwargs["stdout"])
        camera_processes.append(process)
        return process

    def fake_terminate(process, *, timeout_s):
        del timeout_s
        process.returncode = -15

    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution.time.monotonic",
        lambda: clock["monotonic"],
    )
    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution.subprocess.Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution._terminate_process_group",
        fake_terminate,
    )

    with pytest.raises(RuntimeError, match="stopped advancing"):
        run_capture_execution(
            run_root,
            allow_cameras=True,
            allow_real_robot=True,
            collect_sensors=fake_sensor_status,
        )

    assert len(camera_processes) == 1
    assert camera_processes[0].returncode == -15
    report = json.loads((run_root / CAPTURE_EXECUTION_REPORT).read_text())
    assert report["status"] == "failed"
    assert report["receive_idle_timeout_s"] == 60.0
    assert report["camera_metadata_idle_timeout_s"] == 2.0
    assert report["camera_metadata_idle_timeout_source"] == (
        "capture_fps_derived"
    )
    camera = next(
        process
        for process in report["processes"]
        if process["role"] == "sensor_capture"
    )
    assert camera["termination_reason"] == "camera_metadata_stalled"
    assert camera["metadata_idle_elapsed_s"] == 3.0
    assert (
        run_root / "realsense_123" / FRAME_METADATA_JSONL
    ).is_file()


def test_capture_execution_reloads_child_manifest_updates_before_final_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, _config = configured_run(tmp_path, "manifest-merge")
    partial_name = "raw_robot_ee_poses.partial.child.json"
    partial_path = run_root / partial_name

    def fake_popen(command, **kwargs):
        if any(item.endswith("pose_receiver_udp_json.py") for item in command):
            (run_root / RAW_ROBOT_EE_POSES).write_text(
                json.dumps({"0": {"motion": "circ_far", "pose": {"X": 1}}})
            )
            partial_path.write_text('{"status": "failed-child-attempt"}\n')
            child_manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
            child_manifest["artifacts"][partial_name] = partial_name
            child_manifest["artifacts"][RAW_ROBOT_EE_POSES] = RAW_ROBOT_EE_POSES
            child_manifest["stages"].append(
                {
                    "name": "robot_pose_capture",
                    "status": "succeeded",
                    "started_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:01+00:00",
                    "ended_at": "2026-01-01T00:00:01+00:00",
                    "message": "child receiver evidence",
                    "artifacts": {
                        partial_name: partial_name,
                        RAW_ROBOT_EE_POSES: RAW_ROBOT_EE_POSES,
                    },
                }
            )
            write_run_manifest(child_manifest, run_root)
            return FakeBackgroundProcess(list(command), kwargs["stdout"])
        mark_sensor_ready(run_root)
        return FakePersistentProcess(list(command), kwargs["stdout"])

    def fake_terminate(process, *, timeout_s):
        del timeout_s
        process.returncode = -15

    monkeypatch.setattr("posetestbot.pipeline.capture_execution.subprocess.Popen", fake_popen)
    monkeypatch.setattr("posetestbot.pipeline.capture_execution.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution._terminate_process_group",
        fake_terminate,
    )

    _report_path, report = run_capture_execution(
        run_root,
        allow_cameras=True,
        allow_real_robot=True,
        collect_sensors=fake_sensor_status,
        timeout_s=5,
    )

    assert report["status"] == "succeeded"
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    assert manifest["artifacts"][partial_name] == partial_name
    robot_stage = next(
        stage for stage in manifest["stages"] if stage["name"] == "robot_pose_capture"
    )
    assert robot_stage["message"] == "child receiver evidence"
    assert robot_stage["artifacts"][partial_name] == partial_name
    assert any(stage["name"] == "capture_execution" for stage in manifest["stages"])


@pytest.mark.parametrize("camera_returncode", [0, 7])
def test_capture_execution_fails_when_camera_exits_while_receiver_is_active(
    tmp_path: Path,
    monkeypatch,
    camera_returncode: int,
) -> None:
    run_root, _config = configured_run(
        tmp_path,
        f"premature-camera-{camera_returncode}",
    )
    state: dict[str, bool] = {"receiver_started": False}
    spawned = []

    def fake_popen(command, **kwargs):
        if any(item.endswith("pose_receiver_udp_json.py") for item in command):
            state["receiver_started"] = True
            process = FakePersistentProcess(list(command), kwargs["stdout"])
        else:
            mark_sensor_ready(run_root)
            process = FakeCameraExitWhileReceiverRuns(
                list(command),
                kwargs["stdout"],
                state,
                camera_returncode,
            )
        spawned.append(process)
        return process

    def fake_terminate(process, *, timeout_s):
        del timeout_s
        process.returncode = -15

    monkeypatch.setattr("posetestbot.pipeline.capture_execution.subprocess.Popen", fake_popen)
    monkeypatch.setattr("posetestbot.pipeline.capture_execution.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution._terminate_process_group",
        fake_terminate,
    )

    with pytest.raises(RuntimeError, match="exited before the robot pose receiver"):
        run_capture_execution(
            run_root,
            allow_cameras=True,
            allow_real_robot=True,
            collect_sensors=fake_sensor_status,
            timeout_s=5,
        )

    report = json.loads((run_root / CAPTURE_EXECUTION_REPORT).read_text())
    camera = next(
        process for process in report["processes"] if process["role"] == "sensor_capture"
    )
    assert report["status"] == "failed"
    assert camera["status"] == "failed"
    assert camera["returncode"] == camera_returncode
    assert camera["termination_reason"] == "camera_exited_while_receiver_active"


def test_capture_execution_fails_on_nonzero_camera_exit_after_receiver(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, _config = configured_run(tmp_path, "camera-fails-after-receiver")

    def fake_popen(command, **kwargs):
        if any(item.endswith("pose_receiver_udp_json.py") for item in command):
            (run_root / RAW_ROBOT_EE_POSES).write_text(
                json.dumps({"0": {"motion": "circ_far", "pose": {"X": 1}}})
            )
            return FakeBackgroundProcess(list(command), kwargs["stdout"])
        mark_sensor_ready(run_root)
        return FakeCameraExitAfterReceiver(list(command), kwargs["stdout"], 9)

    monkeypatch.setattr("posetestbot.pipeline.capture_execution.subprocess.Popen", fake_popen)
    monkeypatch.setattr("posetestbot.pipeline.capture_execution.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="failure after receiver completion"):
        run_capture_execution(
            run_root,
            allow_cameras=True,
            allow_real_robot=True,
            collect_sensors=fake_sensor_status,
            timeout_s=5,
        )

    report = json.loads((run_root / CAPTURE_EXECUTION_REPORT).read_text())
    camera = next(
        process for process in report["processes"] if process["role"] == "sensor_capture"
    )
    assert report["status"] == "failed"
    assert camera["returncode"] == 9
    assert camera["status"] == "failed"


def test_capture_execution_defers_camera_spawn_signal_until_child_is_tracked(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, _config = configured_run(tmp_path, "camera-spawn-signal")
    spawned: list[FakePersistentProcess] = []
    terminated: list[FakePersistentProcess] = []
    receiver_commands: list[list[str]] = []

    def fake_popen(command, **kwargs):
        if any(item.endswith("pose_receiver_udp_json.py") for item in command):
            receiver_commands.append(list(command))
            raise AssertionError("receiver must not start after camera cancellation")
        process = FakePersistentProcess(list(command), kwargs["stdout"])
        spawned.append(process)
        os.kill(os.getpid(), signal.SIGTERM)
        return process

    def fake_terminate(process, *, timeout_s):
        del timeout_s
        process.returncode = -signal.SIGTERM
        terminated.append(process)

    monkeypatch.setattr("posetestbot.pipeline.capture_execution.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution._terminate_process_group",
        fake_terminate,
    )

    with pytest.raises(RuntimeError, match="canceled by SIGTERM"):
        run_capture_execution(
            run_root,
            allow_cameras=True,
            allow_real_robot=True,
            collect_sensors=fake_sensor_status,
        )

    assert len(spawned) == 1
    assert terminated == spawned
    assert receiver_commands == []
    report = json.loads((run_root / CAPTURE_EXECUTION_REPORT).read_text())
    assert report["status"] == "canceled"
    assert len(report["processes"]) == 1
    assert report["processes"][0]["role"] == "sensor_capture"
    assert report["processes"][0]["startup_attempt"] == 1
    assert report["processes"][0]["status"] == "terminated"
    assert report["processes"][0]["termination_reason"] == (
        "cancellation_cleanup"
    )


def test_capture_execution_deferred_signal_wins_over_camera_spawn_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, _config = configured_run(tmp_path, "camera-spawn-signal-and-error")
    camera_spawn_count = 0

    def fake_popen(command, **kwargs):
        nonlocal camera_spawn_count
        del kwargs
        if any(item.endswith("pose_receiver_udp_json.py") for item in command):
            raise AssertionError("receiver must not start after camera cancellation")
        camera_spawn_count += 1
        os.kill(os.getpid(), signal.SIGTERM)
        raise OSError("simulated Popen failure after deferred signal")

    monkeypatch.setattr("posetestbot.pipeline.capture_execution.subprocess.Popen", fake_popen)
    monkeypatch.setattr("posetestbot.pipeline.capture_execution.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="canceled by SIGTERM"):
        run_capture_execution(
            run_root,
            allow_cameras=True,
            allow_real_robot=True,
            collect_sensors=fake_sensor_status,
            camera_startup_retry_delay_s=0,
        )

    assert camera_spawn_count == 1
    report = json.loads((run_root / CAPTURE_EXECUTION_REPORT).read_text())
    assert report["status"] == "canceled"
    assert len(report["processes"]) == 1
    assert report["processes"][0]["status"] == "canceled"
    assert report["processes"][0]["termination_reason"] == (
        "not_spawned_during_cancellation_cleanup"
    )


def test_capture_execution_defers_receiver_spawn_signal_until_child_is_tracked(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, _config = configured_run(tmp_path, "receiver-spawn-signal")
    spawned: list[FakePersistentProcess] = []
    terminated: list[FakePersistentProcess] = []
    receiver_spawn_count = 0

    def fake_popen(command, **kwargs):
        nonlocal receiver_spawn_count
        process = FakePersistentProcess(list(command), kwargs["stdout"])
        spawned.append(process)
        if any(item.endswith("pose_receiver_udp_json.py") for item in command):
            receiver_spawn_count += 1
            os.kill(os.getpid(), signal.SIGTERM)
        else:
            mark_sensor_ready(run_root)
        return process

    def fake_terminate(process, *, timeout_s):
        del timeout_s
        process.returncode = -signal.SIGTERM
        terminated.append(process)

    monkeypatch.setattr("posetestbot.pipeline.capture_execution.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution._terminate_process_group",
        fake_terminate,
    )

    with pytest.raises(RuntimeError, match="canceled by SIGTERM"):
        run_capture_execution(
            run_root,
            allow_cameras=True,
            allow_real_robot=True,
            collect_sensors=fake_sensor_status,
            timeout_s=5,
        )

    assert receiver_spawn_count == 1
    assert len(spawned) == 2
    assert terminated == spawned
    report = json.loads((run_root / CAPTURE_EXECUTION_REPORT).read_text())
    assert report["status"] == "canceled"
    assert len(report["processes"]) == 2
    receiver = next(
        process
        for process in report["processes"]
        if process["role"] == "robot_pose_receiver"
    )
    assert receiver["status"] == "terminated"
    assert receiver["termination_reason"] == "cancellation_cleanup"
    assert not (run_root / RAW_ROBOT_EE_POSES).exists()


def test_capture_execution_sigterm_cancels_every_spawned_process(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "run-canceled"
    write_run_config(
        run_root,
        create_run_config(
            run_root=run_root,
            sensors=(sensor_config_from_token("realsense:123:static:Cell RealSense"),),
        ),
    )
    spawned: list[FakePersistentProcess] = []
    terminated: list[FakePersistentProcess] = []

    def fake_popen(command, **kwargs):
        process: FakePersistentProcess
        if any(item.endswith("pose_receiver_udp_json.py") for item in command):
            process = FakeSignalProcess(list(command), kwargs["stdout"])
        else:
            mark_sensor_ready(run_root)
            process = FakePersistentProcess(list(command), kwargs["stdout"])
        spawned.append(process)
        return process

    def fake_terminate(process, *, timeout_s):
        del timeout_s
        process.returncode = -signal.SIGTERM
        terminated.append(process)

    monkeypatch.setattr("posetestbot.pipeline.capture_execution.subprocess.Popen", fake_popen)
    monkeypatch.setattr("posetestbot.pipeline.capture_execution.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "posetestbot.pipeline.capture_execution._terminate_process_group",
        fake_terminate,
    )

    with pytest.raises(RuntimeError, match="canceled by SIGTERM"):
        run_capture_execution(
            run_root,
            allow_cameras=True,
            allow_real_robot=True,
            collect_sensors=fake_sensor_status,
            timeout_s=5,
        )

    report = json.loads((run_root / CAPTURE_EXECUTION_REPORT).read_text())
    assert report["status"] == "canceled"
    assert "SIGTERM" in report["message"]
    assert len(spawned) == 2
    assert terminated == spawned
    assert all(process["status"] == "terminated" for process in report["processes"])
    persisted = json.loads((run_root / CAPTURE_EXECUTION_STATUS).read_text())
    assert persisted["status"] == "canceled"
    assert persisted["active_process_count"] == 0
