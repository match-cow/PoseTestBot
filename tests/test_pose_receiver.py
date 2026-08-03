from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

import pytest

from posetestbot.config import RobotProfile
from posetestbot.io.artifacts import DATASET_MANIFEST, RAW_ROBOT_EE_POSES
from posetestbot.robot.pose_receiver import (
    CLAIM_SCHEMA_VERSION,
    PARTIAL_SCHEMA_VERSION,
    POSE_PACKET_SCHEMA_VERSION,
    PoseReceiverCanceled,
    PoseReceiverOverwriteError,
    PoseReceiverPacketError,
    PoseReceiverPermissionError,
    PoseReceiverTimeout,
    run_pose_receiver,
)


class FakeDatagramSocket:
    def __init__(
        self,
        events: list[Any],
        *,
        bind_error: Exception | None = None,
        on_bind=None,
    ) -> None:
        self.events = list(events)
        self.bind_error = bind_error
        self.on_bind = on_bind
        self.bound_to: tuple[str, int] | None = None
        self.timeouts: list[float] = []

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        return False

    def bind(self, address: tuple[str, int]) -> None:
        if self.on_bind is not None:
            self.on_bind(address)
        if self.bind_error is not None:
            raise self.bind_error
        self.bound_to = address

    def settimeout(self, timeout: float) -> None:
        self.timeouts.append(timeout)

    def recvfrom(self, _size: int):
        if not self.events:
            raise AssertionError("Fake socket has no remaining receive event")
        event = self.events.pop(0)
        if isinstance(event, BaseException):
            raise event
        return event


class FakeSocketFactory:
    def __init__(self, sock: FakeDatagramSocket) -> None:
        self.sock = sock
        self.calls: list[tuple[int, int]] = []

    def __call__(self, family: int, socket_type: int) -> FakeDatagramSocket:
        self.calls.append((family, socket_type))
        return self.sock


def profile() -> RobotProfile:
    return RobotProfile(
        mode="real",
        robot_ip="192.0.2.10",
        command_port=30300,
        receiver_ip="127.0.0.1",
        receiver_port=18080,
        cartesian_velocity_m_s=0.02,
    )


def packet(motion: str = "circ_1") -> bytes:
    return json.dumps(
        {
            "motion": motion,
            "X": 1.0,
            "Y": 2.0,
            "Z": 3.0,
            "A": 0.1,
            "B": 0.2,
            "C": 0.3,
        }
    ).encode()


def end_packet() -> bytes:
    return json.dumps({"motion": "end"}).encode()


def v1_packet(
    *,
    sequence: int,
    motion: str = "a1_capture_sweep",
    reference_path: str = "/PoseTestBot/PoseTemplateBase",
    run_id: str = "run-1",
    cadence_evidence: bool = False,
) -> bytes:
    value: dict[str, Any] = {
        "schema_version": POSE_PACKET_SCHEMA_VERSION,
        "packet_kind": "end" if motion == "end" else "pose",
        "sequence": sequence,
        "sender_monotonic_ns": 1_000_000 + sequence,
        "sender_wall_timestamp_ms": 2_000_000 + sequence,
        "run_id": run_id,
        "motion": motion,
        "from_frame": "robot_flange",
        "to_frame": "template_base",
        "sunrise_reference_frame_path": reference_path,
    }
    if motion != "end":
        value.update(
            {
                "X": 1.0,
                "Y": 2.0,
                "Z": 3.0,
                "A": 0.1,
                "B": 0.2,
                "C": 0.3,
            }
        )
        if cadence_evidence:
            value.update(
                {
                    "sender_target_period_ms": 10,
                    "sender_previous_pose_delta_ns": 10_100_000,
                    "sender_pose_query_duration_ns": 800_000,
                }
            )
    return json.dumps(value).encode()


def manifest_stage(run_root: Path) -> dict[str, Any]:
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    return next(
        stage for stage in manifest["stages"] if stage["name"] == "robot_pose_capture"
    )


@pytest.mark.parametrize(
    ("allow_real_robot", "allow_cameras"),
    [
        (False, False),
        (True, False),
        (False, True),
        (1, True),
        (True, 1),
        ("true", True),
        (True, "true"),
    ],
)
def test_receiver_requires_both_fresh_acknowledgements_before_socket_io(
    tmp_path: Path,
    allow_real_robot: Any,
    allow_cameras: Any,
) -> None:
    run_root = tmp_path / "blocked"
    fake_socket = FakeDatagramSocket([])
    socket_factory = FakeSocketFactory(fake_socket)
    start_calls: list[object] = []

    with pytest.raises(PoseReceiverPermissionError, match="fresh acknowledgements"):
        run_pose_receiver(
            run_root,
            profile=profile(),
            allow_real_robot=allow_real_robot,
            allow_cameras=allow_cameras,
            socket_factory=socket_factory,
            send_start_command=lambda *args, **kwargs: start_calls.append(
                (args, kwargs)
            ),
            install_signal_handlers=False,
        )

    assert socket_factory.calls == []
    assert start_calls == []
    assert not run_root.exists()


def test_receiver_cli_rejects_direct_ungated_start_without_network_io(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "direct-blocked"
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/pose_receiver_udp_json.py",
            run_root.as_posix(),
            "--ip",
            "127.0.0.1",
            "--port",
            "18080",
            "--ip_robot",
            "192.0.2.10",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache"},
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "--allow-real-robot" in result.stderr
    assert "--allow-cameras" in result.stderr
    assert not run_root.exists()


def test_receiver_refuses_existing_raw_artifact_before_socket_or_start(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "existing"
    run_root.mkdir()
    raw_path = run_root / RAW_ROBOT_EE_POSES
    raw_path.write_text('{"preserve": true}\n')
    original = raw_path.read_bytes()
    socket_factory = FakeSocketFactory(FakeDatagramSocket([]))
    start_calls: list[object] = []

    with pytest.raises(PoseReceiverOverwriteError, match="Refusing to replace"):
        run_pose_receiver(
            run_root,
            profile=profile(),
            allow_real_robot=True,
            allow_cameras=True,
            socket_factory=socket_factory,
            send_start_command=lambda *args, **kwargs: start_calls.append(
                (args, kwargs)
            ),
            install_signal_handlers=False,
        )

    assert raw_path.read_bytes() == original
    assert socket_factory.calls == []
    assert start_calls == []
    assert list(run_root.glob("raw_robot_ee_poses.partial.*.json")) == []


def test_receiver_success_uses_start_then_idle_timeout_and_writes_canonical_raw(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "success"
    fake_socket = FakeDatagramSocket(
        [
            (packet(), ("192.0.2.10", 40001)),
            (end_packet(), ("192.0.2.10", 49999)),
        ]
    )
    starts: list[tuple[RobotProfile, str]] = []

    def fake_start(
        robot: RobotProfile,
        *,
        protocol: str,
        maximum_velocity_m_s: float,
    ):
        assert maximum_velocity_m_s == 0.03
        starts.append((robot, protocol))
        return {"start": robot.cartesian_velocity_m_s}

    result = run_pose_receiver(
        run_root,
        profile=profile(),
        allow_real_robot=True,
        allow_cameras=True,
        receive_start_timeout_s=1.25,
        receive_idle_timeout_s=2.5,
        socket_factory=FakeSocketFactory(fake_socket),
        send_start_command=fake_start,
        install_signal_handlers=False,
    )

    assert starts == [(profile(), "legacy")]
    assert fake_socket.bound_to == ("127.0.0.1", 18080)
    assert fake_socket.timeouts == [1.25, 2.5]
    assert result.raw_pose_path == run_root / RAW_ROBOT_EE_POSES
    assert result.pose_count == 1
    saved = json.loads(result.raw_pose_path.read_text())
    assert saved["0"]["motion"] == "circ_1"
    assert saved["0"]["pose"] == {
        "X": 1.0,
        "Y": 2.0,
        "Z": 3.0,
        "A": 0.1,
        "B": 0.2,
        "C": 0.3,
    }
    assert manifest_stage(run_root)["status"] == "succeeded"


def test_receiver_caps_start_velocity_and_records_requested_value(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "bounded-start"
    fake_socket = FakeDatagramSocket(
        [
            (packet(), ("192.0.2.10", 40001)),
            (end_packet(), ("192.0.2.10", 40001)),
        ]
    )
    requested_profile = profile().with_overrides(cartesian_velocity_m_s=0.2)
    starts: list[RobotProfile] = []

    def fake_start(
        robot: RobotProfile,
        *,
        protocol: str,
        maximum_velocity_m_s: float,
    ):
        assert protocol == "legacy"
        assert maximum_velocity_m_s == 0.03
        starts.append(robot)
        return {"start": robot.cartesian_velocity_m_s}

    run_pose_receiver(
        run_root,
        profile=requested_profile,
        allow_real_robot=True,
        allow_cameras=True,
        socket_factory=FakeSocketFactory(fake_socket),
        send_start_command=fake_start,
        install_signal_handlers=False,
    )

    assert starts[0].cartesian_velocity_m_s == 0.03
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    assert manifest["capture_config"]["cartesian_velocity_m_s"] == 0.03
    assert manifest["capture_config"]["requested_cartesian_velocity_m_s"] == 0.2
    assert manifest["capture_config"]["command_velocity_cap_m_s"] == 0.03


def test_receiver_passes_extended_speed_only_over_versioned_protocol(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "versioned-extended-start"
    fake_socket = FakeDatagramSocket(
        [
            (packet(), ("192.0.2.10", 40001)),
            (end_packet(), ("192.0.2.10", 40001)),
        ]
    )
    requested_profile = profile().with_overrides(cartesian_velocity_m_s=0.2)
    starts: list[tuple[RobotProfile, str, float]] = []

    def fake_start(
        robot: RobotProfile,
        *,
        protocol: str,
        maximum_velocity_m_s: float,
    ):
        starts.append((robot, protocol, maximum_velocity_m_s))
        return {
            "schema_version": "robot_command.v1",
            "cartesian_velocity_m_s": robot.cartesian_velocity_m_s,
        }

    run_pose_receiver(
        run_root,
        profile=requested_profile,
        protocol="v1",
        maximum_command_velocity_m_s=1.0,
        allow_real_robot=True,
        allow_cameras=True,
        socket_factory=FakeSocketFactory(fake_socket),
        send_start_command=fake_start,
        install_signal_handlers=False,
    )

    assert starts == [(requested_profile, "v1", 1.0)]
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    assert manifest["capture_config"]["cartesian_velocity_m_s"] == 0.2
    assert manifest["capture_config"]["command_velocity_cap_m_s"] == 1.0
    assert manifest["capture_config"]["protocol"] == "v1"


def test_receiver_rejects_extended_limit_on_legacy_protocol(tmp_path: Path) -> None:
    socket_factory = FakeSocketFactory(FakeDatagramSocket([]))

    with pytest.raises(ValueError, match="require protocol='v1'"):
        run_pose_receiver(
            tmp_path / "legacy-extended-start",
            profile=profile(),
            maximum_command_velocity_m_s=1.0,
            allow_real_robot=True,
            allow_cameras=True,
            socket_factory=socket_factory,
            install_signal_handlers=False,
        )

    assert socket_factory.calls == []


def test_receiver_retains_v1_frame_identity_and_packet_loss_evidence(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "v1-packets"
    fake_socket = FakeDatagramSocket(
        [
            (v1_packet(sequence=10), ("192.0.2.10", 40001)),
            (v1_packet(sequence=12), ("192.0.2.10", 40001)),
            (v1_packet(sequence=13, motion="end"), ("192.0.2.10", 40001)),
        ]
    )

    result = run_pose_receiver(
        run_root,
        profile=profile(),
        allow_real_robot=True,
        allow_cameras=True,
        socket_factory=FakeSocketFactory(fake_socket),
        send_start_command=lambda *_args, **_kwargs: {"start": 0.02},
        install_signal_handlers=False,
    )

    saved = json.loads(result.raw_pose_path.read_text())
    assert saved["0"]["source_packet"] == {
        "schema_version": POSE_PACKET_SCHEMA_VERSION,
        "packet_kind": "pose",
        "sequence": 10,
        "sender_monotonic_ns": 1_000_010,
        "sender_wall_timestamp_ms": 2_000_010,
        "run_id": "run-1",
        "from_frame": "robot_flange",
        "to_frame": "template_base",
        "sunrise_reference_frame_path": "/PoseTestBot/PoseTemplateBase",
        "sequence_delta": 0,
        "estimated_packets_lost": 0,
    }
    assert saved["1"]["source_packet"]["sequence_delta"] == 2
    assert saved["1"]["source_packet"]["estimated_packets_lost"] == 1


def test_receiver_retains_complete_sender_cadence_evidence(tmp_path: Path) -> None:
    run_root = tmp_path / "v1-cadence"
    fake_socket = FakeDatagramSocket(
        [
            (
                v1_packet(sequence=10, cadence_evidence=True),
                ("192.0.2.10", 40001),
            ),
            (v1_packet(sequence=11, motion="end"), ("192.0.2.10", 40001)),
        ]
    )

    result = run_pose_receiver(
        run_root,
        profile=profile(),
        allow_real_robot=True,
        allow_cameras=True,
        socket_factory=FakeSocketFactory(fake_socket),
        send_start_command=lambda *_args, **_kwargs: {"start": 0.02},
        install_signal_handlers=False,
    )

    saved = json.loads(result.raw_pose_path.read_text())
    assert saved["0"]["source_packet"]["sender_target_period_ms"] == 10
    assert saved["0"]["source_packet"]["sender_previous_pose_delta_ns"] == 10_100_000
    assert saved["0"]["source_packet"]["sender_pose_query_duration_ns"] == 800_000


def test_receiver_rejects_partial_sender_cadence_evidence(tmp_path: Path) -> None:
    value = json.loads(v1_packet(sequence=1))
    value["sender_target_period_ms"] = 10

    with pytest.raises(PoseReceiverPacketError, match="cadence evidence must include"):
        run_pose_receiver(
            tmp_path / "partial-cadence",
            profile=profile(),
            allow_real_robot=True,
            allow_cameras=True,
            socket_factory=FakeSocketFactory(
                FakeDatagramSocket(
                    [(json.dumps(value).encode(), ("192.0.2.10", 40001))]
                )
            ),
            send_start_command=lambda *_args, **_kwargs: {"start": 0.02},
            install_signal_handlers=False,
        )


def test_receiver_rejects_v1_reference_frame_change_mid_stream(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "frame-changed"

    with pytest.raises(PoseReceiverPacketError, match="identity changed"):
        run_pose_receiver(
            run_root,
            profile=profile(),
            allow_real_robot=True,
            allow_cameras=True,
            socket_factory=FakeSocketFactory(
                FakeDatagramSocket(
                    [
                        (v1_packet(sequence=1), ("192.0.2.10", 30300)),
                        (
                            v1_packet(
                                sequence=2,
                                reference_path="/PoseTestBot/TemplateBase",
                            ),
                            ("192.0.2.10", 30300),
                        ),
                    ]
                )
            ),
            send_start_command=lambda *_args, **_kwargs: {"start": 0.02},
            install_signal_handlers=False,
        )

    partial = next(run_root.glob("raw_robot_ee_poses.partial.*.json"))
    assert json.loads(partial.read_text())["received_pose_count"] == 1


def test_receiver_claims_canonical_raw_path_before_bind_or_start(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "claimed-before-network"
    raw_path = run_root / RAW_ROBOT_EE_POSES
    observed_claims: list[dict[str, Any]] = []

    def observe_claim(_address: tuple[str, int]) -> None:
        observed_claims.append(json.loads(raw_path.read_text()))

    fake_socket = FakeDatagramSocket(
        [
            (packet(), ("192.0.2.10", 41000)),
            (end_packet(), ("192.0.2.10", 42000)),
        ],
        on_bind=observe_claim,
    )

    run_pose_receiver(
        run_root,
        profile=profile(),
        allow_real_robot=True,
        allow_cameras=True,
        socket_factory=FakeSocketFactory(fake_socket),
        send_start_command=lambda *_args, **_kwargs: {"start": 0.02},
        install_signal_handlers=False,
    )

    assert len(observed_claims) == 1
    assert observed_claims[0]["schema_version"] == CLAIM_SCHEMA_VERSION
    assert observed_claims[0]["status"] == "reserved"
    assert isinstance(observed_claims[0]["claim_id"], str)


def test_receiver_rejects_datagrams_from_any_ip_except_configured_robot(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "unexpected-sender"

    with pytest.raises(PoseReceiverPacketError, match="unexpected sender IP"):
        run_pose_receiver(
            run_root,
            profile=profile(),
            allow_real_robot=True,
            allow_cameras=True,
            socket_factory=FakeSocketFactory(
                FakeDatagramSocket([(packet(), ("192.0.2.11", 30300))])
            ),
            send_start_command=lambda *_args, **_kwargs: {"start": 0.02},
            install_signal_handlers=False,
        )

    assert not (run_root / RAW_ROBOT_EE_POSES).exists()
    partial = next(run_root.glob("raw_robot_ee_poses.partial.*.json"))
    assert json.loads(partial.read_text())["last_sender"] == [
        "192.0.2.11",
        "30300",
    ]


def test_receiver_never_replaces_foreign_artifact_that_displaces_its_claim(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "claim-displaced"
    raw_path = run_root / RAW_ROBOT_EE_POSES
    foreign_bytes = b'{"foreign": true}\n'

    def replace_claim(*_args, **_kwargs):
        assert (
            json.loads(raw_path.read_text())["schema_version"] == CLAIM_SCHEMA_VERSION
        )
        raw_path.unlink()
        raw_path.write_bytes(foreign_bytes)
        return {"start": 0.02}

    with pytest.raises(PoseReceiverOverwriteError, match="ownership changed"):
        run_pose_receiver(
            run_root,
            profile=profile(),
            allow_real_robot=True,
            allow_cameras=True,
            socket_factory=FakeSocketFactory(
                FakeDatagramSocket(
                    [
                        (packet(), ("192.0.2.10", 30301)),
                        (end_packet(), ("192.0.2.10", 30302)),
                    ]
                )
            ),
            send_start_command=replace_claim,
            install_signal_handlers=False,
        )

    assert raw_path.read_bytes() == foreign_bytes
    assert len(list(run_root.glob("raw_robot_ee_poses.partial.*.json"))) == 1


def test_receiver_start_and_idle_timeouts_preserve_unique_partial_evidence(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "timeouts"

    with pytest.raises(PoseReceiverTimeout, match="first robot pose"):
        run_pose_receiver(
            run_root,
            profile=profile(),
            allow_real_robot=True,
            allow_cameras=True,
            receive_start_timeout_s=0.25,
            socket_factory=FakeSocketFactory(FakeDatagramSocket([socket.timeout()])),
            send_start_command=lambda *_args, **_kwargs: {"start": 0.02},
            install_signal_handlers=False,
        )

    idle_socket = FakeDatagramSocket(
        [(packet(), ("192.0.2.10", 30300)), socket.timeout()]
    )
    with pytest.raises(PoseReceiverTimeout, match="next robot pose"):
        run_pose_receiver(
            run_root,
            profile=profile(),
            allow_real_robot=True,
            allow_cameras=True,
            receive_idle_timeout_s=0.5,
            socket_factory=FakeSocketFactory(idle_socket),
            send_start_command=lambda *_args, **_kwargs: {"start": 0.02},
            install_signal_handlers=False,
        )

    partials = sorted(run_root.glob("raw_robot_ee_poses.partial.*.json"))
    assert len(partials) == 2
    evidence = [json.loads(path.read_text()) for path in partials]
    assert {item["received_pose_count"] for item in evidence} == {0, 1}
    assert all(item["schema_version"] == PARTIAL_SCHEMA_VERSION for item in evidence)
    assert not (run_root / RAW_ROBOT_EE_POSES).exists()
    manifest = json.loads((run_root / DATASET_MANIFEST).read_text())
    assert all(path.name in manifest["artifacts"] for path in partials)
    stage = manifest_stage(run_root)
    assert stage["status"] == "failed"
    assert all(path.name in stage["artifacts"] for path in partials)
    assert idle_socket.timeouts == [120.0, 0.5]


def test_receiver_malformed_packet_records_failed_manifest_and_partial(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "malformed"
    malformed = b'{"motion":"circ_1","X":"not-a-number"}'

    with pytest.raises(PoseReceiverPacketError, match="X must be a finite number"):
        run_pose_receiver(
            run_root,
            profile=profile(),
            allow_real_robot=True,
            allow_cameras=True,
            socket_factory=FakeSocketFactory(
                FakeDatagramSocket([(malformed, ("192.0.2.10", 30300))])
            ),
            send_start_command=lambda *_args, **_kwargs: {"start": 0.02},
            install_signal_handlers=False,
        )

    partial = next(run_root.glob("raw_robot_ee_poses.partial.*.json"))
    evidence = json.loads(partial.read_text())
    assert evidence["status"] == "failed"
    assert evidence["last_packet_preview"] == malformed.decode()
    assert manifest_stage(run_root)["status"] == "failed"
    assert not (run_root / RAW_ROBOT_EE_POSES).exists()


def test_receiver_rejects_end_marker_before_any_pose(tmp_path: Path) -> None:
    run_root = tmp_path / "empty-stream"

    with pytest.raises(PoseReceiverPacketError, match="before any pose"):
        run_pose_receiver(
            run_root,
            profile=profile(),
            allow_real_robot=True,
            allow_cameras=True,
            socket_factory=FakeSocketFactory(
                FakeDatagramSocket([(end_packet(), ("192.0.2.10", 30300))])
            ),
            send_start_command=lambda *_args, **_kwargs: {"start": 0.02},
            install_signal_handlers=False,
        )

    assert manifest_stage(run_root)["status"] == "failed"
    assert len(list(run_root.glob("raw_robot_ee_poses.partial.*.json"))) == 1
    assert not (run_root / RAW_ROBOT_EE_POSES).exists()


def test_receiver_bind_failure_records_failed_manifest_without_sending_start(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "bind-failure"
    start_calls: list[object] = []

    with pytest.raises(OSError, match="address unavailable"):
        run_pose_receiver(
            run_root,
            profile=profile(),
            allow_real_robot=True,
            allow_cameras=True,
            socket_factory=FakeSocketFactory(
                FakeDatagramSocket([], bind_error=OSError("address unavailable"))
            ),
            send_start_command=lambda *args, **kwargs: start_calls.append(
                (args, kwargs)
            ),
            install_signal_handlers=False,
        )

    assert start_calls == []
    assert manifest_stage(run_root)["status"] == "failed"
    assert len(list(run_root.glob("raw_robot_ee_poses.partial.*.json"))) == 1
    assert not (run_root / RAW_ROBOT_EE_POSES).exists()


def test_receiver_interruption_records_canceled_manifest_and_partial(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "canceled"
    fake_socket = FakeDatagramSocket(
        [(packet(), ("192.0.2.10", 30300)), KeyboardInterrupt()]
    )

    with pytest.raises(PoseReceiverCanceled, match="interrupted"):
        run_pose_receiver(
            run_root,
            profile=profile(),
            allow_real_robot=True,
            allow_cameras=True,
            socket_factory=FakeSocketFactory(fake_socket),
            send_start_command=lambda *_args, **_kwargs: {"start": 0.02},
            install_signal_handlers=False,
        )

    partial = next(run_root.glob("raw_robot_ee_poses.partial.*.json"))
    evidence = json.loads(partial.read_text())
    assert evidence["status"] == "canceled"
    assert evidence["received_pose_count"] == 1
    assert manifest_stage(run_root)["status"] == "canceled"
    assert not (run_root / RAW_ROBOT_EE_POSES).exists()
