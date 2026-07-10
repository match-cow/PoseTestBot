from __future__ import annotations

from posetestbot.sensors.previews import build_preview_command
from posetestbot.sensors.v4l2_preview import (
    V4L2NodeCandidate,
    parse_v4l2_pixel_formats,
    select_best_rgb_node,
    select_usb_rgb_node,
)
from posetestbot.sensors import v4l2_preview


def test_parse_v4l2_pixel_formats_accepts_bracketed_and_pixel_format_forms() -> None:
    text = """
    [0]: 'YUYV' (YUYV 4:2:2)
    Pixel Format: 'MJPG'
    """

    assert parse_v4l2_pixel_formats(text) == ("MJPG", "YUYV")


def test_select_best_rgb_node_prefers_realsense_color_interface() -> None:
    candidates = [
        V4L2NodeCandidate("/dev/video0", interface="00", capabilities=":capture:"),
        V4L2NodeCandidate("/dev/video2", interface="00", capabilities=":capture:"),
        V4L2NodeCandidate("/dev/video4", interface="03", capabilities=":capture:"),
    ]
    formats = {
        "/dev/video0": ("Z16",),
        "/dev/video2": ("GREY", "UYVY"),
        "/dev/video4": ("YUYV",),
    }

    selection = select_best_rgb_node(
        candidates,
        format_reader=lambda path: formats[path],
    )

    assert selection.path == "/dev/video4"
    assert selection.formats == ("YUYV",)


def test_build_preview_command_targets_rgb_preview_worker(tmp_path) -> None:
    command = build_preview_command(
        preview_root=tmp_path / "preview",
        spec={"sensor_type": "realsense_d435", "device_id": "825412070181"},
        fps=6,
        width=640,
        height=480,
    )

    assert command[:4] == [
        "uv",
        "run",
        "python",
        "scripts/stream_sensor_rgb_preview.py",
    ]
    assert "--sensor-json" in command


def test_select_usb_rgb_node_uses_usb_matched_candidates(monkeypatch) -> None:
    candidates = [
        V4L2NodeCandidate("/dev/video18", interface="00", capabilities=":capture:"),
        V4L2NodeCandidate("/dev/video19", interface="00", capabilities=":metadata:"),
    ]
    monkeypatch.setattr(v4l2_preview, "candidates_for_usb_id", lambda *_args: candidates)

    selection = select_usb_rgb_node(
        "0c45",
        "2283",
        format_reader=lambda path: ("MJPG",) if path == "/dev/video18" else (),
    )

    assert selection.path == "/dev/video18"
    assert selection.formats == ("MJPG",)
