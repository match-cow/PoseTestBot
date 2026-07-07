"""JSON-friendly sensor status snapshots for CLI/API/UI surfaces."""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, Mapping

from posetestbot.sensors import discovery
from posetestbot.sensors.contracts import SensorDeviceInfo, SensorType
from posetestbot.sensors.registry import SENSOR_ADAPTERS

SCHEMA_VERSION = "sensor_status.v1"

SENSOR_FAMILY_LABELS: Mapping[SensorType, str] = {
    sensor_type: adapter.display_name
    for sensor_type, adapter in SENSOR_ADAPTERS.items()
}

SENSOR_SDK_MODULES: Mapping[SensorType, str] = {
    sensor_type: adapter.sdk_module
    for sensor_type, adapter in SENSOR_ADAPTERS.items()
}

LAB_EXPECTED_SENSOR_COUNTS: Mapping[SensorType, int] = {
    SensorType.REALSENSE_D435: 3,
    SensorType.OAK_D_PRO: 1,
    SensorType.ZED_2I: 1,
}

DISCOVERERS: Mapping[SensorType, Callable[[], list[SensorDeviceInfo]]] = {
    SensorType.REALSENSE_D435: discovery.discover_realsense_d435,
    SensorType.OAK_D_PRO: discovery.discover_oak_d_pro,
    SensorType.ZED_2I: discovery.discover_zed_2i,
}


@dataclass(frozen=True)
class SensorFamilyStatus:
    sensor_type: SensorType
    display_name: str
    sdk_module: str
    sdk_available: bool
    expected_count: int | None
    connected_count: int
    meets_expected: bool | None
    devices: list[SensorDeviceInfo]
    error: str | None = None
    diagnostics: list[dict] | None = None

    def as_dict(self) -> dict:
        value = asdict(self)
        value["sensor_type"] = self.sensor_type.value
        value["devices"] = [sensor_device_to_dict(device) for device in self.devices]
        value["diagnostics"] = list(self.diagnostics or [])
        return value


def sensor_device_to_dict(device: SensorDeviceInfo) -> dict:
    return {
        "sensor_type": device.sensor_type.value,
        "device_id": device.device_id,
        "display_name": device.display_name,
        "connected": device.connected,
        "metadata": dict(device.metadata),
    }


def sdk_module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _sensor_family_diagnostics(
    *,
    sensor_type: SensorType,
    sdk_module: str,
    sdk_available: bool,
    expected_count: int | None,
    connected_count: int,
    error: str | None,
) -> list[dict]:
    diagnostics: list[dict] = []
    if error:
        error_lower = error.lower()
        hints = [
            "Run the status command on the lab host with camera USB devices visible.",
        ]
        if "udev" in error_lower or "usb" in error_lower:
            hints.append(
                "Check USB/udev access; containers usually need /dev/bus/usb mounted and a USB device cgroup rule."
            )
        diagnostics.append(
            {
                "code": "discovery_error",
                "severity": "error",
                "message": f"{SENSOR_FAMILY_LABELS[sensor_type]} discovery failed: {error}",
                "hints": hints,
            }
        )
    if not sdk_available:
        diagnostics.append(
            {
                "code": "sdk_unavailable",
                "severity": "warning",
                "message": f"Python SDK module {sdk_module!r} is not importable.",
                "hints": [
                    f"Install or activate the {sdk_module} Python SDK in the uv environment.",
                ],
            }
        )
    if expected_count is not None and connected_count < expected_count:
        hints = [
            "Verify the expected lab cameras are physically connected and powered.",
        ]
        if sensor_type in {SensorType.REALSENSE_D435, SensorType.OAK_D_PRO}:
            hints.append(
                "Verify USB permissions and udev rules for camera discovery."
            )
        if sensor_type == SensorType.ZED_2I:
            hints.append(
                "Verify the ZED SDK and pyzed.sl Python bindings are installed."
            )
        diagnostics.append(
            {
                "code": "expected_count_not_met",
                "severity": "warning",
                "message": (
                    f"Connected {connected_count} of expected {expected_count} "
                    f"{SENSOR_FAMILY_LABELS[sensor_type]} device(s)."
                ),
                "hints": hints,
            }
        )
    return diagnostics


def collect_sensor_family_status(
    sensor_type: SensorType,
    *,
    expected_count: int | None = None,
    discoverer: Callable[[], list[SensorDeviceInfo]] | None = None,
) -> SensorFamilyStatus:
    sdk_module = SENSOR_SDK_MODULES[sensor_type]
    sdk_available = sdk_module_available(sdk_module)
    discoverer = discoverer or DISCOVERERS[sensor_type]

    devices: list[SensorDeviceInfo] = []
    error = None
    try:
        devices = discoverer()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    connected_count = len(devices)
    meets_expected = None
    if expected_count is not None:
        meets_expected = connected_count >= expected_count
    diagnostics = _sensor_family_diagnostics(
        sensor_type=sensor_type,
        sdk_module=sdk_module,
        sdk_available=sdk_available,
        expected_count=expected_count,
        connected_count=connected_count,
        error=error,
    )

    return SensorFamilyStatus(
        sensor_type=sensor_type,
        display_name=SENSOR_FAMILY_LABELS[sensor_type],
        sdk_module=sdk_module,
        sdk_available=sdk_available,
        expected_count=expected_count,
        connected_count=connected_count,
        meets_expected=meets_expected,
        devices=devices,
        error=error,
        diagnostics=diagnostics,
    )


def collect_sensor_status(
    *,
    expected_counts: Mapping[SensorType, int | None] | None = None,
    discoverers: Mapping[SensorType, Callable[[], list[SensorDeviceInfo]]] | None = None,
) -> dict:
    expected_counts = expected_counts or LAB_EXPECTED_SENSOR_COUNTS
    discoverers = discoverers or DISCOVERERS
    families = []
    for sensor_type in SensorType:
        families.append(
            collect_sensor_family_status(
                sensor_type,
                expected_count=expected_counts.get(sensor_type),
                discoverer=discoverers.get(sensor_type),
            ).as_dict()
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "families": families,
        "total_connected": sum(family["connected_count"] for family in families),
        "all_expected_connected": all(
            family["meets_expected"] is not False for family in families
        ),
    }


def parse_expected_counts(values: list[str]) -> dict[SensorType, int | None]:
    expected_counts = dict(LAB_EXPECTED_SENSOR_COUNTS)
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected count must be SENSOR_TYPE=COUNT, got {value!r}")
        sensor_type_value, count_value = value.split("=", 1)
        try:
            sensor_type = SensorType(sensor_type_value)
        except ValueError as exc:
            valid = ", ".join(sensor.value for sensor in SensorType)
            raise ValueError(
                f"Unknown sensor type {sensor_type_value!r}; expected one of: {valid}"
            ) from exc

        if count_value.lower() in {"none", "unknown", "-"}:
            expected_counts[sensor_type] = None
            continue
        try:
            count = int(count_value)
        except ValueError as exc:
            raise ValueError(
                f"Expected count for {sensor_type.value} must be an integer"
            ) from exc
        if count < 0:
            raise ValueError(f"Expected count for {sensor_type.value} cannot be negative")
        expected_counts[sensor_type] = count
    return expected_counts
