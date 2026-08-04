"""V4L2 helpers for lightweight RGB sensor previews."""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

import cv2

from posetestbot.sensors.discovery import (
    _udev_properties_for_node,
    _video_metadata_for_physical_port,
    _video_node_metadata_by_serial,
)


COLOR_PIXEL_FORMATS = {"YUYV", "MJPG", "RGB3", "BGR3"}
NON_RGB_PIXEL_FORMATS = {"Z16", "GREY", "Y8", "Y8I", "Y10", "Y12", "UYVY"}


@dataclass(frozen=True)
class V4L2NodeCandidate:
    path: str
    interface: str | None = None
    capabilities: str | None = None
    usb_serial: str | None = None
    accessible: bool = True

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class V4L2PreviewSelection:
    path: str
    formats: tuple[str, ...]
    candidate: V4L2NodeCandidate
    score: int

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "formats": list(self.formats),
            "candidate": self.candidate.as_dict(),
            "score": self.score,
        }


@dataclass(frozen=True)
class V4L2IntegerControl:
    name: str
    minimum: int
    maximum: int
    step: int
    default: int
    value: int

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


def parse_v4l2_pixel_formats(text: str) -> tuple[str, ...]:
    formats = {
        match.group("format").upper()
        for match in re.finditer(
            r"(?:Pixel Format:\s+|\[\d+\]:\s+)'(?P<format>[^']+)'",
            text,
        )
    }
    return tuple(sorted(formats))


def parse_v4l2_integer_control(
    text: str,
    control_name: str,
) -> V4L2IntegerControl | None:
    """Parse one integer control from ``v4l2-ctl --list-ctrls`` output."""

    expected = control_name.strip()
    for line in text.splitlines():
        match = re.match(
            r"^\s*(?P<name>[a-zA-Z0-9_]+)\s+0x[0-9a-fA-F]+\s+\(int\)\s*:\s*(?P<values>.*)$",
            line,
        )
        if match is None or match.group("name") != expected:
            continue
        values = {
            key: int(value)
            for key, value in re.findall(
                r"\b(min|max|step|default|value)=(-?\d+)",
                match.group("values"),
            )
        }
        required = {"min", "max", "step", "default", "value"}
        if not required.issubset(values):
            raise ValueError(f"Malformed V4L2 integer control line: {line.strip()}")
        if values["min"] > values["max"] or values["step"] <= 0:
            raise ValueError(f"Invalid V4L2 integer control range: {line.strip()}")
        return V4L2IntegerControl(
            name=expected,
            minimum=values["min"],
            maximum=values["max"],
            step=values["step"],
            default=values["default"],
            value=values["value"],
        )
    return None


def read_v4l2_integer_control(
    path: str | Path,
    control_name: str,
) -> V4L2IntegerControl | None:
    result = subprocess.run(
        ["v4l2-ctl", "--list-ctrls", "-d", Path(path).as_posix()],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"Could not read V4L2 controls for {Path(path).as_posix()}: {message}"
        )
    return parse_v4l2_integer_control(result.stdout, control_name)


def list_v4l2_pixel_formats(path: str | Path) -> tuple[str, ...]:
    result = subprocess.run(
        ["v4l2-ctl", "--list-formats-ext", "-d", Path(path).as_posix()],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"Could not read V4L2 formats for {Path(path).as_posix()}: {message}"
        )
    return parse_v4l2_pixel_formats(result.stdout)


def _candidate_from_node(node: Mapping[str, object]) -> V4L2NodeCandidate | None:
    path = str(node.get("path") or "").strip()
    if not path:
        return None
    current_access = os.access(path, os.R_OK | os.W_OK)
    return V4L2NodeCandidate(
        path=path,
        interface=str(node.get("interface") or "") or None,
        capabilities=str(node.get("capabilities") or "") or None,
        accessible=bool(node.get("accessible", current_access) and current_access),
    )


def candidates_from_metadata(metadata: Mapping[str, object] | None) -> list[V4L2NodeCandidate]:
    if not isinstance(metadata, Mapping):
        return []
    candidates: list[V4L2NodeCandidate] = []
    for node in metadata.get("video_nodes", []):
        if not isinstance(node, Mapping):
            continue
        candidate = _candidate_from_node(node)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _candidates_from_video_metadata(
    video_metadata: Mapping[str, object],
    *,
    usb_serial: str | None = None,
) -> list[V4L2NodeCandidate]:
    candidates = []
    for node in video_metadata.get("video_nodes", []):
        if not isinstance(node, Mapping):
            continue
        candidate = _candidate_from_node(node)
        if candidate is None:
            continue
        candidates.append(
            V4L2NodeCandidate(
                path=candidate.path,
                interface=candidate.interface,
                capabilities=candidate.capabilities,
                usb_serial=usb_serial,
                accessible=candidate.accessible,
            )
        )
    return candidates


def _candidates_for_sdk_serial(device_id: str) -> list[V4L2NodeCandidate]:
    try:
        import pyrealsense2 as rs
    except ImportError:
        return []

    video_metadata_by_usb_serial = _video_node_metadata_by_serial()
    for dev in rs.context().query_devices():
        try:
            serial = dev.get_info(rs.camera_info.serial_number)
        except Exception:
            continue
        if serial != device_id:
            continue
        try:
            physical_port = dev.get_info(rs.camera_info.physical_port)
        except Exception:
            return []
        usb_serial, metadata = _video_metadata_for_physical_port(
            physical_port,
            video_metadata_by_usb_serial,
        )
        return _candidates_from_video_metadata(metadata, usb_serial=usb_serial)
    return []


def _candidates_for_usb_serial(device_id: str) -> list[V4L2NodeCandidate]:
    metadata = _video_node_metadata_by_serial().get(device_id, {})
    return _candidates_from_video_metadata(metadata, usb_serial=device_id)


def candidates_for_usb_id(
    vendor_id: str,
    product_id: str,
) -> list[V4L2NodeCandidate]:
    """Return V4L2 nodes belonging to one USB vendor/product pair."""

    expected_vendor = vendor_id.strip().lower()
    expected_product = product_id.strip().lower()
    candidates = []
    for path in sorted(
        Path("/dev").glob("video*"),
        key=lambda item: _numeric_video_index(item.as_posix()),
    ):
        properties = _udev_properties_for_node(path)
        if properties.get("ID_VENDOR_ID", "").lower() != expected_vendor:
            continue
        if properties.get("ID_MODEL_ID", "").lower() != expected_product:
            continue
        candidates.append(
            V4L2NodeCandidate(
                path=path.as_posix(),
                interface=properties.get("ID_USB_INTERFACE_NUM"),
                capabilities=properties.get("ID_V4L_CAPABILITIES"),
                usb_serial=properties.get("ID_SERIAL_SHORT"),
                accessible=os.access(path, os.R_OK | os.W_OK),
            )
        )
    return candidates


def _score_candidate(candidate: V4L2NodeCandidate, formats: tuple[str, ...]) -> int | None:
    if not candidate.accessible:
        return None
    format_set = set(formats)
    has_color_format = bool(format_set & COLOR_PIXEL_FORMATS)
    if not has_color_format and candidate.interface != "03":
        return None

    score = 0
    if has_color_format:
        score += 100
    if "YUYV" in format_set:
        score += 20
    if "MJPG" in format_set:
        score += 10
    if candidate.interface == "03":
        score += 50
    if candidate.capabilities and ":capture:" in candidate.capabilities:
        score += 10
    if format_set & NON_RGB_PIXEL_FORMATS:
        score -= 60
    if "Z16" in format_set:
        score -= 100
    return score


def select_best_rgb_node(
    candidates: list[V4L2NodeCandidate],
    *,
    format_reader: Callable[[str], tuple[str, ...]] = list_v4l2_pixel_formats,
) -> V4L2PreviewSelection:
    errors: list[str] = []
    scored: list[V4L2PreviewSelection] = []
    for candidate in candidates:
        try:
            formats = format_reader(candidate.path)
        except Exception as exc:
            errors.append(f"{candidate.path}: {type(exc).__name__}: {exc}")
            formats = ()
        score = _score_candidate(candidate, formats)
        if score is None:
            continue
        scored.append(
            V4L2PreviewSelection(
                path=candidate.path,
                formats=formats,
                candidate=candidate,
                score=score,
            )
        )
    if scored:
        return sorted(
            scored,
            key=lambda item: (item.score, _numeric_video_index(item.path)),
            reverse=True,
        )[0]
    if errors:
        raise RuntimeError("No RGB-capable V4L2 node found. " + "; ".join(errors))
    raise RuntimeError("No RGB-capable V4L2 node found.")


def _numeric_video_index(path: str) -> int:
    match = re.search(r"video(?P<index>\d+)$", path)
    return int(match.group("index")) if match else -1


def select_realsense_rgb_node(
    device_id: str,
    *,
    metadata: Mapping[str, object] | None = None,
    format_reader: Callable[[str], tuple[str, ...]] = list_v4l2_pixel_formats,
) -> V4L2PreviewSelection:
    candidates = candidates_from_metadata(metadata)
    if not candidates:
        candidates = _candidates_for_sdk_serial(device_id)
    if not candidates:
        candidates = _candidates_for_usb_serial(device_id)
    if not candidates:
        raise RuntimeError(f"No V4L2 nodes found for RealSense device {device_id}.")
    return select_best_rgb_node(candidates, format_reader=format_reader)


def select_usb_rgb_node(
    vendor_id: str,
    product_id: str,
    *,
    format_reader: Callable[[str], tuple[str, ...]] = list_v4l2_pixel_formats,
) -> V4L2PreviewSelection:
    candidates = candidates_for_usb_id(vendor_id, product_id)
    if not candidates:
        raise RuntimeError(
            f"No accessible V4L2 nodes found for USB camera {vendor_id}:{product_id}."
        )
    return select_best_rgb_node(candidates, format_reader=format_reader)


def open_v4l2_capture(
    path: str,
    *,
    width: int,
    height: int,
    fps: int,
    pixel_format: str = "YUYV",
):
    """Open a low-buffer V4L2 capture, releasing every failed attempt."""

    for attempt in range(3):
        capture = cv2.VideoCapture(path, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*pixel_format))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        capture.set(cv2.CAP_PROP_FPS, float(fps))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if capture.isOpened():
            return capture
        capture.release()
        if attempt < 2:
            time.sleep(0.5)
    raise RuntimeError(f"Could not open RGB preview node {path} after 3 attempts.")
