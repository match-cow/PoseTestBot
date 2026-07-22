from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

from posetestbot.sensors import discovery
from posetestbot.sensors.aliases import (
    load_sensor_aliases,
    save_sensor_aliases,
    sensor_alias_key,
    sensor_alias_file_state,
)
from posetestbot.sensors import status as sensor_status
from posetestbot.sensors.contracts import SensorDeviceInfo, SensorType


def device(sensor_type: SensorType, device_id: str) -> SensorDeviceInfo:
    return SensorDeviceInfo(
        sensor_type=sensor_type,
        device_id=device_id,
        display_name=f"{sensor_type.value} {device_id}",
        metadata={"fixture": True},
    )


def test_collect_sensor_status_is_detection_first_by_default(monkeypatch) -> None:
    monkeypatch.setattr(sensor_status, "sdk_module_available", lambda _: True)
    discoverers = {
        SensorType.REALSENSE_D435: lambda: [
            device(SensorType.REALSENSE_D435, "rs-1"),
            device(SensorType.REALSENSE_D435, "rs-2"),
            device(SensorType.REALSENSE_D435, "rs-3"),
        ],
        SensorType.OAK_D_PRO: lambda: [device(SensorType.OAK_D_PRO, "oak-1")],
        SensorType.ZED_2I: lambda: [],
    }

    status = sensor_status.collect_sensor_status(discoverers=discoverers)

    assert status["schema_version"] == sensor_status.SCHEMA_VERSION
    assert status["total_connected"] == 4
    assert status["expected_counts_requested"] is False
    assert status["all_expected_connected"] is True

    families = {family["sensor_type"]: family for family in status["families"]}
    assert families["realsense_d435"]["meets_expected"] is None
    assert families["realsense_d435"]["expected_count"] is None
    assert families["realsense_d435"]["devices"][0]["device_id"] == "rs-1"
    assert families["oak_d_pro"]["meets_expected"] is None
    assert families["zed_2i"]["sdk_available"] is True


def test_collect_sensor_status_reports_explicit_expected_counts(monkeypatch) -> None:
    monkeypatch.setattr(sensor_status, "sdk_module_available", lambda _: True)
    status = sensor_status.collect_sensor_status(
        expected_counts={
            SensorType.REALSENSE_D435: 3,
            SensorType.OAK_D_PRO: 1,
            SensorType.ZED_2I: 1,
        },
        discoverers={
            SensorType.REALSENSE_D435: lambda: [
                device(SensorType.REALSENSE_D435, "rs-1"),
                device(SensorType.REALSENSE_D435, "rs-2"),
                device(SensorType.REALSENSE_D435, "rs-3"),
            ],
            SensorType.OAK_D_PRO: lambda: [device(SensorType.OAK_D_PRO, "oak-1")],
            SensorType.ZED_2I: lambda: [],
        },
    )

    families = {family["sensor_type"]: family for family in status["families"]}
    assert status["expected_counts_requested"] is True
    assert status["all_expected_connected"] is False
    assert families["realsense_d435"]["meets_expected"] is True
    assert families["realsense_d435"]["expected_count"] == 3
    assert families["zed_2i"]["meets_expected"] is False
    assert families["zed_2i"]["meets_expected"] is False


def test_realsense_usb_only_fallback_does_not_satisfy_expected_count(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sensor_status, "sdk_module_available", lambda _: True)
    sdk_devices = [
        SensorDeviceInfo(
            sensor_type=SensorType.REALSENSE_D435,
            device_id=serial,
            display_name=f"RealSense {serial}",
            metadata={"discovery": "librealsense"},
        )
        for serial in ("825412070181", "923322072633")
    ]
    usb_only = SensorDeviceInfo(
        sensor_type=SensorType.REALSENSE_D435,
        device_id="usb-005-005",
        display_name="Intel RealSense D435 usb-005-005",
        metadata={
            "discovery": "lsusb_descriptor",
            "serial_available": False,
            "bus": "005",
            "device": "005",
        },
    )

    status = sensor_status.collect_sensor_status(
        expected_counts={SensorType.REALSENSE_D435: 3},
        discoverers={
            SensorType.REALSENSE_D435: lambda: [*sdk_devices, usb_only],
            SensorType.OAK_D_PRO: lambda: [],
            SensorType.ZED_2I: lambda: [],
        },
    )

    family = status["families"][0]
    assert status["total_connected"] == 3
    assert status["total_capture_ready"] == 2
    assert status["all_expected_connected"] is False
    assert family["connected_count"] == 3
    assert family["capture_ready_count"] == 2
    assert family["meets_expected"] is False
    by_id = {item["device_id"]: item for item in family["devices"]}
    fallback = by_id["usb-005-005"]
    assert fallback["connected"] is True
    assert fallback["capture_ready"] is False
    assert fallback["capture_readiness_reason"] == "not_enumerated_by_sdk"
    assert fallback["metadata"]["bus"] == "005"
    assert [item["code"] for item in family["diagnostics"]] == [
        "devices_not_capture_ready",
        "expected_count_not_met",
    ]
    assert family["diagnostics"][0]["devices"] == [
        {
            "device_id": "usb-005-005",
            "reason": "not_enumerated_by_sdk",
            "discovery": "lsusb_descriptor",
        }
    ]


def test_realsense_usb_fallback_is_not_ready_without_sdk(monkeypatch) -> None:
    monkeypatch.setattr(sensor_status, "sdk_module_available", lambda _: False)
    usb_descriptor = SensorDeviceInfo(
        sensor_type=SensorType.REALSENSE_D435,
        device_id="033422071805",
        display_name="Intel RealSense D435 033422071805",
        metadata={
            "discovery": "lsusb_descriptor",
            "serial_available": True,
        },
    )

    family = sensor_status.collect_sensor_family_status(
        SensorType.REALSENSE_D435,
        expected_count=1,
        discoverer=lambda: [usb_descriptor],
    ).as_dict(aliases={})

    assert family["connected_count"] == 1
    assert family["capture_ready_count"] == 0
    assert family["meets_expected"] is False
    assert family["devices"][0]["capture_readiness_reason"] == "sdk_unavailable"
    assert [item["code"] for item in family["diagnostics"]] == [
        "sdk_unavailable",
        "devices_not_capture_ready",
        "expected_count_not_met",
    ]


def test_realsense_sdk_usb2_device_is_not_capture_ready(monkeypatch) -> None:
    monkeypatch.setattr(sensor_status, "sdk_module_available", lambda _: True)
    usb2_device = SensorDeviceInfo(
        sensor_type=SensorType.REALSENSE_D435,
        device_id="033422071805",
        display_name="Intel RealSense D435 033422071805",
        metadata={
            "discovery": "librealsense",
            "usb_type_descriptor": "2.1",
            "firmware_version": "5.16.0.1",
            "recommended_firmware_version": "5.17.0.10",
        },
    )

    family = sensor_status.collect_sensor_family_status(
        SensorType.REALSENSE_D435,
        expected_count=1,
        discoverer=lambda: [usb2_device],
    ).as_dict(aliases={})

    assert family["connected_count"] == 1
    assert family["capture_ready_count"] == 0
    assert family["meets_expected"] is False
    assert family["devices"][0]["capture_readiness_reason"] == (
        "usb_connection_below_superspeed"
    )
    assert [item["code"] for item in family["diagnostics"]] == [
        "realsense_firmware_recommendation_mismatch",
        "realsense_usb_below_superspeed",
        "devices_not_capture_ready",
        "expected_count_not_met",
    ]
    firmware = family["diagnostics"][0]
    assert firmware["severity"] == "warning"
    assert firmware["devices"] == [
        {
            "device_id": "033422071805",
            "firmware_version": "5.16.0.1",
            "recommended_firmware_version": "5.17.0.10",
            "relation": "older",
        }
    ]
    transport = family["diagnostics"][1]
    assert transport["severity"] == "error"
    assert transport["devices"] == [
        {
            "device_id": "033422071805",
            "reason": "usb_connection_below_superspeed",
            "discovery": "librealsense",
            "usb_type_descriptor": "2.1",
            "usb_major": 2,
        }
    ]


def test_realsense_missing_or_unrecognized_usb_descriptor_is_backward_compatible(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sensor_status, "sdk_module_available", lambda _: True)
    devices = [
        SensorDeviceInfo(
            sensor_type=SensorType.REALSENSE_D435,
            device_id="legacy-missing",
            display_name="Legacy RealSense",
            metadata={"discovery": "librealsense"},
        ),
        SensorDeviceInfo(
            sensor_type=SensorType.REALSENSE_D435,
            device_id="legacy-unknown",
            display_name="Legacy RealSense",
            metadata={
                "discovery": "librealsense",
                "usb_type_descriptor": "unknown",
            },
        ),
    ]

    family = sensor_status.collect_sensor_family_status(
        SensorType.REALSENSE_D435,
        expected_count=2,
        discoverer=lambda: devices,
    ).as_dict(aliases={})

    assert family["capture_ready_count"] == 2
    assert family["meets_expected"] is True
    assert all(item["capture_ready"] for item in family["devices"])
    assert family["diagnostics"] == []


def test_depthai_device_id_accepts_v3_device_info_shape() -> None:
    class FakeDepthAIV3DeviceInfo:
        deviceId = "18443010314F3B1300"
        name = "2.10"

        def getDeviceId(self) -> str:
            return self.deviceId

    assert (
        discovery._depthai_device_id(FakeDepthAIV3DeviceInfo())
        == "18443010314F3B1300"
    )


def test_depthai_tcp_ip_devices_are_not_counted_as_local_oak(monkeypatch) -> None:
    class FakeDevice:
        deviceId = "1965890851"
        name = "10.145.8.163"
        platform = "XLinkPlatform.X_LINK_RVC4"
        protocol = "XLinkProtocol.X_LINK_TCP_IP"
        state = "XLinkDeviceState.X_LINK_GATE_BOOTED"

        def getDeviceId(self) -> str:
            return self.deviceId

    fake_depthai = type(
        "FakeDepthAI",
        (),
        {
            "Device": type(
                "FakeDepthAIDevice",
                (),
                {"getAllAvailableDevices": staticmethod(lambda: [FakeDevice()])},
            )
        },
    )
    monkeypatch.setitem(sys.modules, "depthai", fake_depthai)

    assert discovery.discover_oak_d_pro() == []


def test_realsense_discovery_merges_sdk_devices_with_usb_video_nodes(
    monkeypatch,
) -> None:
    class FakeCameraInfo:
        serial_number = "serial_number"
        name = "name"
        product_line = "product_line"
        product_id = "product_id"
        firmware_version = "firmware_version"
        recommended_firmware_version = "recommended_firmware_version"
        usb_type_descriptor = "usb_type_descriptor"
        physical_port = "physical_port"

    class FakeDevice:
        def supports(self, _key):
            return True

        def get_info(self, key):
            values = {
                "serial_number": "sdk-serial-1",
                "name": "Intel RealSense D435",
                "product_line": "D400",
                "product_id": "0B07",
                "firmware_version": "5.17.0.10",
                "recommended_firmware_version": "5.17.0.10",
                "usb_type_descriptor": "3.2",
                "physical_port": "/sys/devices/usb3/3-1/3-1:1.0/video4linux/video0",
            }
            return values[key]

    class FakeContext:
        def query_devices(self):
            return [FakeDevice()]

    fake_rs = type(
        "FakeRS",
        (),
        {
            "camera_info": FakeCameraInfo,
            "context": staticmethod(lambda: FakeContext()),
        },
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)
    monkeypatch.setattr(
        discovery,
        "_udev_properties_for_node",
        lambda _path: {"ID_SERIAL_SHORT": "usb-serial-1"},
    )
    monkeypatch.setattr(
        discovery,
        "_video_node_metadata_by_serial",
        lambda: {
            "usb-serial-1": {
                "video_accessible": True,
                "video_nodes": [
                    {
                        "path": "/dev/video4",
                        "interface": "03",
                        "capabilities": ":capture:",
                        "accessible": True,
                    }
                ],
            }
        },
    )
    monkeypatch.setattr(
        discovery,
        "_discover_realsense_from_lsusb",
        lambda: [
            SensorDeviceInfo(
                sensor_type=SensorType.REALSENSE_D435,
                device_id="usb-serial-1",
                display_name="USB duplicate",
            ),
            SensorDeviceInfo(
                sensor_type=SensorType.REALSENSE_D435,
                device_id="usb-serial-2",
                display_name="USB orphan",
            ),
        ],
    )

    devices = discovery.discover_realsense_d435()

    assert [device.device_id for device in devices] == [
        "sdk-serial-1",
        "usb-serial-2",
    ]
    assert devices[0].metadata["usb_serial"] == "usb-serial-1"
    assert devices[0].metadata["discovery"] == "librealsense"
    assert devices[0].metadata["recommended_firmware_version"] == "5.17.0.10"
    assert devices[0].metadata["video_nodes"][0]["path"] == "/dev/video4"


def test_realsense_discovery_skips_unsupported_optional_camera_info(
    monkeypatch,
) -> None:
    class FakeCameraInfo:
        serial_number = "serial_number"
        name = "name"
        product_line = "product_line"
        recommended_firmware_version = "recommended_firmware_version"

    class FakeDevice:
        def supports(self, key):
            return key != "recommended_firmware_version"

        def get_info(self, key):
            if key == "recommended_firmware_version":
                raise AssertionError("unsupported camera_info was queried")
            return {
                "serial_number": "sdk-serial-1",
                "name": "Intel RealSense D435",
                "product_line": "D400",
            }[key]

    fake_rs = type(
        "FakeRS",
        (),
        {
            "camera_info": FakeCameraInfo,
            "context": staticmethod(
                lambda: type(
                    "FakeContext",
                    (),
                    {"query_devices": lambda self: [FakeDevice()]},
                )()
            ),
        },
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)
    monkeypatch.setattr(discovery, "_video_node_metadata_by_serial", lambda: {})
    monkeypatch.setattr(discovery, "_discover_realsense_from_lsusb", lambda: [])

    devices = discovery.discover_realsense_d435()

    assert len(devices) == 1
    assert devices[0].metadata["discovery"] == "librealsense"
    assert "recommended_firmware_version" not in devices[0].metadata


def test_collect_sensor_status_records_discovery_errors(monkeypatch) -> None:
    monkeypatch.setattr(sensor_status, "sdk_module_available", lambda _: True)

    def failing_discoverer() -> list[SensorDeviceInfo]:
        raise RuntimeError("camera bus unavailable")

    status = sensor_status.collect_sensor_status(
        expected_counts={
            SensorType.REALSENSE_D435: 1,
            SensorType.OAK_D_PRO: None,
            SensorType.ZED_2I: None,
        },
        discoverers={
            SensorType.REALSENSE_D435: failing_discoverer,
            SensorType.OAK_D_PRO: lambda: [],
            SensorType.ZED_2I: lambda: [],
        },
    )

    family = status["families"][0]
    assert family["sensor_type"] == "realsense_d435"
    assert family["connected_count"] == 0
    assert family["meets_expected"] is False
    assert family["error"] == "RuntimeError: camera bus unavailable"
    assert family["diagnostics"][0]["code"] == "discovery_error"
    assert family["diagnostics"][0]["severity"] == "error"
    assert family["diagnostics"][1]["code"] == "expected_count_not_met"
    assert status["all_expected_connected"] is False


def test_collect_sensor_status_adds_udev_and_sdk_diagnostics(
    monkeypatch,
) -> None:
    def sdk_available(module_name: str) -> bool:
        return module_name != "pyzed.sl"

    monkeypatch.setattr(sensor_status, "sdk_module_available", sdk_available)

    def failing_realsense() -> list[SensorDeviceInfo]:
        raise RuntimeError("could not initialize udev monitor")

    status = sensor_status.collect_sensor_status(
        expected_counts={
            SensorType.REALSENSE_D435: 1,
            SensorType.OAK_D_PRO: 1,
            SensorType.ZED_2I: 1,
        },
        discoverers={
            SensorType.REALSENSE_D435: failing_realsense,
            SensorType.OAK_D_PRO: lambda: [],
            SensorType.ZED_2I: lambda: [],
        },
    )

    families = {family["sensor_type"]: family for family in status["families"]}
    realsense_diagnostics = families["realsense_d435"]["diagnostics"]
    assert realsense_diagnostics[0]["code"] == "discovery_error"
    assert any("udev" in hint for hint in realsense_diagnostics[0]["hints"])
    oak_diagnostics = families["oak_d_pro"]["diagnostics"]
    assert oak_diagnostics[0]["code"] == "expected_count_not_met"
    assert any("USB permissions" in hint for hint in oak_diagnostics[0]["hints"])
    zed_diagnostics = families["zed_2i"]["diagnostics"]
    assert [diagnostic["code"] for diagnostic in zed_diagnostics] == [
        "sdk_unavailable",
        "expected_count_not_met",
    ]


def test_collect_sensor_status_reports_unsupported_depthai_version(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sensor_status, "sdk_module_available", lambda _: True)

    def sdk_version(module_name: str) -> str | None:
        return "2.30.0.0" if module_name == "depthai" else None

    monkeypatch.setattr(sensor_status, "sdk_module_version", sdk_version)

    status = sensor_status.collect_sensor_status(
        expected_counts={
            SensorType.REALSENSE_D435: None,
            SensorType.OAK_D_PRO: None,
            SensorType.ZED_2I: None,
        },
        discoverers={
            SensorType.REALSENSE_D435: lambda: [],
            SensorType.OAK_D_PRO: lambda: [],
            SensorType.ZED_2I: lambda: [],
        },
    )

    families = {family["sensor_type"]: family for family in status["families"]}
    oak = families["oak_d_pro"]
    assert oak["sdk_version"] == "2.30.0.0"
    assert oak["sdk_requirement"] == ">=3,<4"
    assert oak["diagnostics"][0]["code"] == "sdk_version_unsupported"


def test_sensor_status_json_stdout_survives_noisy_sdk(monkeypatch, capfd) -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "sensor_status.py"
    spec = importlib.util.spec_from_file_location(
        "posetestbot_sensor_status_cli_test",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cli
    spec.loader.exec_module(cli)

    def noisy_collect_sensor_status(*, expected_counts=None):
        os.write(1, b"vendor warning on stdout\n")
        return {
            "schema_version": sensor_status.SCHEMA_VERSION,
            "generated_at": "2026-06-19T00:00:00+00:00",
            "families": [],
            "total_connected": 0,
            "all_expected_connected": True,
        }

    monkeypatch.setattr(cli, "collect_sensor_status", noisy_collect_sensor_status)
    monkeypatch.setattr(sys, "argv", ["sensor_status.py", "--json"])

    assert cli.main() == 0
    captured = capfd.readouterr()
    assert json.loads(captured.out)["schema_version"] == sensor_status.SCHEMA_VERSION
    assert "vendor warning on stdout" not in captured.out
    assert "vendor warning on stdout" in captured.err


def test_parse_expected_counts_validates_sensor_type_and_count() -> None:
    counts = sensor_status.parse_expected_counts(
        ["realsense_d435=2", "oak_d_pro=none", "zed_2i=0"]
    )

    assert counts[SensorType.REALSENSE_D435] == 2
    assert counts[SensorType.OAK_D_PRO] is None
    assert counts[SensorType.ZED_2I] == 0
    assert sensor_status.parse_expected_counts([]) == {}

    try:
        sensor_status.parse_expected_counts(["unknown=1"])
    except ValueError as exc:
        assert "Unknown sensor type" in str(exc)
    else:
        raise AssertionError("unknown sensor type was accepted")

    try:
        sensor_status.parse_expected_counts(["zed_2i=-1"])
    except ValueError as exc:
        assert "cannot be negative" in str(exc)
    else:
        raise AssertionError("negative expected count was accepted")


def test_sensor_aliases_round_trip_and_merge_into_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    alias_path = tmp_path / "sensor_aliases.json"
    save_sensor_aliases(
        {
            sensor_alias_key(SensorType.REALSENSE_D435, "rs-1"): {
                "alias": "Wrist RealSense",
                "mounting_mode": "eye_in_hand",
                "inverted": True,
            }
        },
        alias_path,
    )
    aliases = load_sensor_aliases(alias_path)
    assert aliases["realsense_d435:rs-1"]["alias"] == "Wrist RealSense"

    monkeypatch.setattr(sensor_status, "sdk_module_available", lambda _: True)
    status = sensor_status.collect_sensor_status(
        aliases=aliases,
        discoverers={
            SensorType.REALSENSE_D435: lambda: [
                device(SensorType.REALSENSE_D435, "rs-1"),
            ],
            SensorType.OAK_D_PRO: lambda: [],
            SensorType.ZED_2I: lambda: [],
        },
    )

    device_payload = status["families"][0]["devices"][0]
    assert device_payload["alias"] == "Wrist RealSense"
    assert device_payload["effective_display_name"] == "Wrist RealSense"
    assert device_payload["mounting_mode"] == "eye_in_hand"
    assert device_payload["inverted"] is True


def test_corrupted_sensor_alias_file_reports_state(tmp_path: Path) -> None:
    alias_path = tmp_path / "sensor_aliases.json"
    alias_path.write_text("{not json")

    assert load_sensor_aliases(alias_path) == {}
    state = sensor_alias_file_state(alias_path)
    assert state["aliases"] == {}
    assert "JSONDecodeError" in state["error"]
