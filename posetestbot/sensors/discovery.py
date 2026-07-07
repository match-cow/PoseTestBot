"""Best-effort discovery for connected RGB-D sensors."""

from __future__ import annotations

from posetestbot.sensors.contracts import SensorDeviceInfo, SensorType


def _depthai_device_id(device_info) -> str:
    for method_name in ("getMxId", "getDeviceId"):
        method = getattr(device_info, method_name, None)
        if callable(method):
            try:
                value = method()
            except Exception:
                continue
            if value:
                return str(value)
    for attribute_name in ("mxid", "deviceId", "name"):
        value = getattr(device_info, attribute_name, None)
        if value:
            return str(value)
    return "unknown"


def discover_realsense_d435() -> list[SensorDeviceInfo]:
    try:
        import pyrealsense2 as rs
    except ImportError:
        return []

    devices: list[SensorDeviceInfo] = []
    context = rs.context()
    for dev in context.query_devices():
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        product_line = dev.get_info(rs.camera_info.product_line)
        devices.append(
            SensorDeviceInfo(
                sensor_type=SensorType.REALSENSE_D435,
                device_id=serial,
                display_name=f"{name} {serial}",
                metadata={"product_line": product_line},
            )
        )
    return devices


def discover_oak_d_pro() -> list[SensorDeviceInfo]:
    try:
        import depthai as dai
    except ImportError:
        return []

    devices: list[SensorDeviceInfo] = []
    for dev in dai.Device.getAllAvailableDevices():
        mxid = _depthai_device_id(dev)
        devices.append(
            SensorDeviceInfo(
                sensor_type=SensorType.OAK_D_PRO,
                device_id=mxid,
                display_name=f"OAK-D Pro {mxid}",
                metadata={
                    "state": str(getattr(dev, "state", "unknown")),
                    "name": str(getattr(dev, "name", "")),
                    "platform": str(getattr(dev, "platform", "")),
                    "protocol": str(getattr(dev, "protocol", "")),
                },
            )
        )
    return devices


def discover_zed_2i() -> list[SensorDeviceInfo]:
    try:
        import pyzed.sl as sl
    except ImportError:
        return []

    devices: list[SensorDeviceInfo] = []
    for dev in sl.Camera.get_device_list():
        serial = str(getattr(dev, "serial_number", "unknown"))
        model = str(getattr(dev, "camera_model", "ZED 2i"))
        devices.append(
            SensorDeviceInfo(
                sensor_type=SensorType.ZED_2I,
                device_id=serial,
                display_name=f"{model} {serial}",
            )
        )
    return devices


def discover_all() -> list[SensorDeviceInfo]:
    return [
        *discover_realsense_d435(),
        *discover_oak_d_pro(),
        *discover_zed_2i(),
    ]
