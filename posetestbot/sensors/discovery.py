"""Best-effort discovery for connected RGB-D sensors."""

from __future__ import annotations

import re
import os
import subprocess
from pathlib import Path

from posetestbot.sensors.contracts import SensorDeviceInfo, SensorType


REALSENSE_D435_USB_PRODUCTS = {
    "0b07": "Intel RealSense D435",
    "0b3a": "Intel RealSense D435i",
}


def _udev_properties_for_node(path: Path) -> dict[str, str]:
    try:
        result = subprocess.run(
            ["udevadm", "info", "-q", "property", "-n", path.as_posix()],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, OSError):
        return {}
    properties = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            properties[key] = value
    return properties


def _video_node_metadata_by_serial() -> dict[str, dict[str, object]]:
    by_serial: dict[str, dict[str, object]] = {}
    for path in sorted(Path("/dev").glob("video*")):
        properties = _udev_properties_for_node(path)
        if properties.get("ID_VENDOR_ID") != "8086":
            continue
        product_id = str(properties.get("ID_MODEL_ID", "")).lower()
        if product_id not in REALSENSE_D435_USB_PRODUCTS:
            continue
        serial = properties.get("ID_SERIAL_SHORT")
        if not serial:
            continue
        node = path.as_posix()
        accessible = os.access(path, os.R_OK | os.W_OK)
        record = by_serial.setdefault(
            serial,
            {
                "video_nodes": [],
                "video_accessible": False,
                "video_permission_hint": None,
            },
        )
        record["video_nodes"].append(
            {
                "path": node,
                "interface": properties.get("ID_USB_INTERFACE_NUM"),
                "capabilities": properties.get("ID_V4L_CAPABILITIES"),
                "accessible": accessible,
            }
        )
        record["video_accessible"] = bool(record["video_accessible"] or accessible)

    for record in by_serial.values():
        if not record["video_accessible"]:
            nodes = ", ".join(
                str(node["path"]) for node in record["video_nodes"] if isinstance(node, dict)
            )
            record["video_permission_hint"] = (
                f"{nodes} are not accessible to this process; add the operator "
                "user to the video group or update udev ACLs/rules."
            )
    return by_serial


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


def _parse_realsense_lsusb_devices(text: str) -> list[SensorDeviceInfo]:
    devices: list[SensorDeviceInfo] = []
    current: dict[str, str] | None = None

    def append_current() -> None:
        if current is None:
            return
        product_id = current["product_id"].lower()
        fallback_name = REALSENSE_D435_USB_PRODUCTS.get(product_id, "Intel RealSense")
        serial = current.get("serial", "").strip()
        device_id = serial or f"usb-{current['bus']}-{current['device']}"
        name = current.get("product", "").strip() or current.get("name", "").strip()
        display_name = name or fallback_name
        devices.append(
            SensorDeviceInfo(
                sensor_type=SensorType.REALSENSE_D435,
                device_id=device_id,
                display_name=f"{display_name} {device_id}",
                metadata={
                    "product_line": "D400",
                    "product_id": product_id,
                    "bus": current["bus"],
                    "device": current["device"],
                    "serial_available": bool(serial),
                    "discovery": "lsusb_descriptor",
                },
            )
        )

    header_re = re.compile(
        r"^Bus\s+(?P<bus>\d+)\s+Device\s+(?P<device>\d+):\s+ID\s+"
        r"8086:(?P<product_id>[0-9a-fA-F]{4})\s+(?P<name>.+)$"
    )
    serial_re = re.compile(r"^\s*iSerial\s+\d+\s+(?P<serial>\S+)\s*$")
    product_re = re.compile(r"^\s*iProduct\s+\d+\s+(?P<product>.+?)\s*$")

    for line in text.splitlines():
        header = header_re.match(line)
        if header:
            append_current()
            product_id = header.group("product_id").lower()
            current = (
                header.groupdict()
                if product_id in REALSENSE_D435_USB_PRODUCTS
                else None
            )
            continue
        if current is None:
            continue
        serial = serial_re.match(line)
        if serial:
            current["serial"] = serial.group("serial")
            continue
        product = product_re.match(line)
        if product:
            current["product"] = product.group("product")

    append_current()
    return devices


def _discover_realsense_from_lsusb() -> list[SensorDeviceInfo]:
    devices: list[SensorDeviceInfo] = []
    video_metadata = _video_node_metadata_by_serial()
    for product_id in REALSENSE_D435_USB_PRODUCTS:
        try:
            result = subprocess.run(
                ["lsusb", "-v", "-d", f"8086:{product_id}"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, OSError):
            continue
        devices.extend(_parse_realsense_lsusb_devices(result.stdout))
    seen: set[str] = set()
    unique = []
    for device in devices:
        if device.device_id in seen:
            continue
        seen.add(device.device_id)
        metadata = dict(device.metadata)
        metadata.update(video_metadata.get(device.device_id, {}))
        unique.append(
            SensorDeviceInfo(
                sensor_type=device.sensor_type,
                device_id=device.device_id,
                display_name=device.display_name,
                connected=device.connected,
                metadata=metadata,
            )
        )
    return unique


def discover_realsense_d435() -> list[SensorDeviceInfo]:
    try:
        import pyrealsense2 as rs
    except ImportError:
        return _discover_realsense_from_lsusb()

    devices: list[SensorDeviceInfo] = []
    sdk_error = None
    try:
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
    except Exception as exc:
        sdk_error = f"{type(exc).__name__}: {exc}"

    fallback_devices = _discover_realsense_from_lsusb()
    known_ids = {device.device_id for device in devices}
    for device in fallback_devices:
        if device.device_id in known_ids:
            continue
        metadata = dict(device.metadata)
        if sdk_error:
            metadata["sdk_discovery_error"] = sdk_error
        devices.append(
            SensorDeviceInfo(
                sensor_type=device.sensor_type,
                device_id=device.device_id,
                display_name=device.display_name,
                connected=device.connected,
                metadata=metadata,
            )
        )
    if sdk_error and not devices:
        raise RuntimeError(sdk_error)
    return devices


def _depthai_device_is_local_usb(device_info) -> bool:
    protocol = str(getattr(device_info, "protocol", "") or "").upper()
    if not protocol:
        return True
    return "TCP" not in protocol and "IP" not in protocol


def discover_oak_d_pro() -> list[SensorDeviceInfo]:
    try:
        import depthai as dai
    except ImportError:
        return []

    devices: list[SensorDeviceInfo] = []
    for dev in dai.Device.getAllAvailableDevices():
        if not _depthai_device_is_local_usb(dev):
            continue
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
