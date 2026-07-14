from __future__ import annotations

import importlib.util
import argparse
import json
from pathlib import Path

import numpy as np

from posetestbot.sensors.previews import build_preview_command
from posetestbot.sensors.v4l2_preview import (
    V4L2NodeCandidate,
    parse_v4l2_pixel_formats,
    select_best_rgb_node,
    select_usb_rgb_node,
)
from posetestbot.sensors import v4l2_preview


def load_preview_script():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "stream_sensor_rgb_preview.py"
    )
    spec = importlib.util.spec_from_file_location("stream_sensor_rgb_preview", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_open_capture_releases_failed_attempt_before_retry(monkeypatch) -> None:
    stream_sensor_rgb_preview = load_preview_script()

    class FakeCapture:
        def __init__(self, opened: bool):
            self.opened = opened
            self.released = False

        def set(self, *_args):
            return True

        def isOpened(self):
            return self.opened

        def release(self):
            self.released = True

    captures = [FakeCapture(False), FakeCapture(True)]
    monkeypatch.setattr(
        stream_sensor_rgb_preview.cv2,
        "VideoCapture",
        lambda *_args: captures.pop(0),
    )
    monkeypatch.setattr(stream_sensor_rgb_preview.time, "sleep", lambda _seconds: None)
    failed_capture = captures[0]

    opened = stream_sensor_rgb_preview._open_capture(
        "/dev/video18",
        width=320,
        height=240,
        fps=5,
        pixel_format="MJPG",
    )

    assert failed_capture.released is True
    assert opened.isOpened() is True


def test_oak_preview_writes_latest_frame_heartbeat_and_closes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    stream_sensor_rgb_preview = load_preview_script()

    class FakeOakStream:
        def __init__(self, **_kwargs):
            self.selected_source = {
                "kind": "depthai",
                "device_id": "18443010314F3B1300",
                "queue_blocking": False,
                "queue_max_size": 1,
            }
            self.calls = 0
            self.closed = False

        def try_get_frame(self):
            self.calls += 1
            if self.calls == 1:
                return np.ones((4, 6, 3), dtype=np.uint8) * 80
            stream_sensor_rgb_preview._STOP_REQUESTED = True
            return None

        def close(self):
            self.closed = True

    instances = []

    def open_stream(**kwargs):
        stream = FakeOakStream(**kwargs)
        instances.append(stream)
        return stream

    monkeypatch.setattr(
        stream_sensor_rgb_preview,
        "OAKDProPreviewStream",
        open_stream,
    )
    args = argparse.Namespace(
        preview_root=(tmp_path / "preview").as_posix(),
        sensor_json=json.dumps(
            {
                "sensor_type": "oak_d_pro",
                "device_id": "18443010314F3B1300",
            }
        ),
        fps=6,
        width=640,
        height=480,
        jpeg_quality=82,
    )

    assert stream_sensor_rgb_preview.run_preview(args) == 0

    status = json.loads((tmp_path / "preview" / "preview_status.json").read_text())
    assert status["status"] == "stopped"
    assert status["frame_count"] == 1
    assert status["heartbeat_at"]
    assert status["selected_node"]["queue_blocking"] is False
    assert instances[0].closed is True
