from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

from posetestbot.sensors import status as sensor_status
from posetestbot.sensors.contracts import SensorDeviceInfo, SensorType


def device(sensor_type: SensorType, device_id: str) -> SensorDeviceInfo:
    return SensorDeviceInfo(
        sensor_type=sensor_type,
        device_id=device_id,
        display_name=f"{sensor_type.value} {device_id}",
        metadata={"fixture": True},
    )


def test_collect_sensor_status_reports_expected_counts(monkeypatch) -> None:
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
    assert status["all_expected_connected"] is False

    families = {family["sensor_type"]: family for family in status["families"]}
    assert families["realsense_d435"]["meets_expected"] is True
    assert families["realsense_d435"]["expected_count"] == 3
    assert families["realsense_d435"]["devices"][0]["device_id"] == "rs-1"
    assert families["oak_d_pro"]["meets_expected"] is True
    assert families["zed_2i"]["sdk_available"] is True
    assert families["zed_2i"]["meets_expected"] is False


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
