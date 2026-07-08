from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np

from posetestbot.io.artifacts import (
    CAMERA_JSON,
    DEPTH_DIR,
    FRAME_METADATA_JSONL,
    RGB_DIR,
)
from posetestbot.sensors.contracts import SensorType
from posetestbot.sensors.discovery import discover_realsense_d435
from posetestbot.sensors.realsense import capture_realsense_rgbd


class FakeIntrinsics:
    fx = 600.0
    fy = 601.0
    ppx = 320.0
    ppy = 240.0
    width = 1280
    height = 720


class FakeFrameProfile:
    intrinsics = FakeIntrinsics()

    def as_video_stream_profile(self):
        return self


class FakeFrame:
    def __init__(self, index: int, *, color: bool):
        self.index = index
        self.profile = FakeFrameProfile()
        self.color = color

    def get_data(self):
        if self.color:
            image = np.zeros((3, 4, 3), dtype=np.uint8)
            image[:, :, 0] = np.arange(12, dtype=np.uint8).reshape(3, 4)
            image[:, :, 1] = self.index
            return image
        return np.arange(12, dtype=np.uint16).reshape(3, 4) + self.index * 100

    def get_timestamp(self):
        return float(self.index * 10)

    def get_frame_number(self):
        return self.index


class FakeFrames:
    def __init__(self, index: int):
        self.index = index

    def get_depth_frame(self):
        return FakeFrame(self.index, color=False)

    def get_color_frame(self):
        return FakeFrame(self.index, color=True)


class FakeDepthSensor:
    def get_depth_scale(self):
        return 0.001


class FakeSensor:
    def __init__(self, name: str):
        self.name = name

    def get_info(self, key):
        if key == "name":
            return self.name
        return ""


class FakeDevice:
    def __init__(self, serial: str = "123"):
        self.serial = serial
        self.sensors = [FakeSensor("RGB Camera")]

    def get_info(self, key):
        values = {
            "serial_number": self.serial,
            "name": "Intel RealSense D435i",
            "product_line": "D400",
        }
        return values[key]

    def first_depth_sensor(self):
        return FakeDepthSensor()


class FakeProfile:
    def __init__(self, device: FakeDevice):
        self.device = device

    def get_device(self):
        return self.device


class FakeConfig:
    def __init__(self, device: FakeDevice):
        self.device = device
        self.enabled_device = None
        self.streams = []

    def enable_device(self, serial: str):
        self.enabled_device = serial

    def resolve(self, _pipeline_wrapper):
        return FakeProfile(self.device)

    def enable_stream(self, *args):
        self.streams.append(args)


class FakePipeline:
    def __init__(self, device: FakeDevice):
        self.device = device
        self.index = 0
        self.stopped = False

    def start(self, _config):
        return FakeProfile(self.device)

    def wait_for_frames(self):
        self.index += 1
        return FakeFrames(self.index)

    def stop(self):
        self.stopped = True


class FakeAlign:
    def __init__(self, _stream):
        pass

    def process(self, frames):
        return frames


class FakeRS:
    camera_info = SimpleNamespace(
        serial_number="serial_number",
        name="name",
        product_line="product_line",
    )
    stream = SimpleNamespace(depth="depth", color="color")
    format = SimpleNamespace(z16="z16", bgr8="bgr8")

    def __init__(self, serial: str = "123"):
        self.device = FakeDevice(serial)
        self.pipeline_instance = None
        self.config_instance = None

    def pipeline(self):
        self.pipeline_instance = FakePipeline(self.device)
        return self.pipeline_instance

    def config(self):
        self.config_instance = FakeConfig(self.device)
        return self.config_instance

    def pipeline_wrapper(self, pipeline):
        return pipeline

    def align(self, stream):
        return FakeAlign(stream)


class PreviewSpy:
    def __init__(self):
        self.imshow_calls = 0
        self.wait_key_calls = 0
        self.destroy_calls = 0

    def imshow(self, *_args):
        self.imshow_calls += 1

    def waitKey(self, *_args):
        self.wait_key_calls += 1
        return -1

    def destroyAllWindows(self):
        self.destroy_calls += 1


def test_capture_realsense_rgbd_writes_frames_without_preview(tmp_path) -> None:
    fake_rs = FakeRS("825412070181")
    preview = PreviewSpy()

    summary = capture_realsense_rgbd(
        tmp_path,
        device_id="825412070181",
        fps=6,
        max_frames=2,
        warmup_frames=1,
        preview=False,
        rs_module=fake_rs,
        cv2_module=preview,
    )

    assert summary["schema_version"] == "realsense_capture_summary.v1"
    assert summary["sensor_id"] == "825412070181"
    assert summary["product_line"] == "D400"
    assert summary["frame_count"] == 2
    assert summary["preview"] is False
    assert preview.imshow_calls == 0
    assert fake_rs.pipeline_instance.stopped is True
    assert fake_rs.config_instance.enabled_device == "825412070181"
    assert len(list((tmp_path / RGB_DIR).glob("*.png"))) == 2
    assert len(list((tmp_path / DEPTH_DIR).glob("*.png"))) == 2
    assert (tmp_path / CAMERA_JSON).is_file()
    records = [
        json.loads(line)
        for line in (tmp_path / FRAME_METADATA_JSONL).read_text().splitlines()
    ]
    assert [record["frame_index"] for record in records] == [0, 1]
    assert records[0]["sensor_id"] == "825412070181"
    assert records[0]["inverted"] is False
    assert records[0]["image_rotation_degrees"] == 0


def test_capture_realsense_rgbd_preview_is_optional(tmp_path) -> None:
    preview = PreviewSpy()

    summary = capture_realsense_rgbd(
        tmp_path,
        device_id="123",
        fps=6,
        max_frames=1,
        preview=True,
        rs_module=FakeRS("123"),
        cv2_module=preview,
    )

    assert summary["frame_count"] == 1
    assert summary["preview"] is True
    assert preview.imshow_calls == 1
    assert preview.wait_key_calls == 1
    assert preview.destroy_calls == 1


def test_capture_realsense_rgbd_inverted_rotates_frames_and_intrinsics(tmp_path) -> None:
    summary = capture_realsense_rgbd(
        tmp_path,
        device_id="123",
        fps=6,
        max_frames=1,
        preview=False,
        inverted=True,
        rs_module=FakeRS("123"),
    )

    assert summary["inverted"] is True
    assert summary["image_rotation_degrees"] == 180
    assert summary["orientation"] == "inverted"

    rgb_path = next((tmp_path / RGB_DIR).glob("*.png"))
    depth_path = next((tmp_path / DEPTH_DIR).glob("*.png"))
    written_rgb = cv2.imread(rgb_path.as_posix(), cv2.IMREAD_UNCHANGED)
    written_depth = cv2.imread(depth_path.as_posix(), cv2.IMREAD_UNCHANGED)
    expected_rgb = np.rot90(FakeFrame(1, color=True).get_data(), 2)
    expected_depth = np.rot90(FakeFrame(1, color=False).get_data(), 2)

    assert np.array_equal(written_rgb, expected_rgb)
    assert np.array_equal(written_depth, expected_depth)

    camera = json.loads((tmp_path / CAMERA_JSON).read_text())
    assert camera["cam_K"] == [
        600.0,
        0.0,
        959.0,
        0.0,
        601.0,
        479.0,
        0.0,
        0.0,
        1.0,
    ]
    records = [
        json.loads(line)
        for line in (tmp_path / FRAME_METADATA_JSONL).read_text().splitlines()
    ]
    assert records[0]["inverted"] is True
    assert records[0]["image_rotation_degrees"] == 180
    assert records[0]["orientation"] == "inverted"


def test_discover_realsense_d435_reads_mocked_sdk_devices(monkeypatch) -> None:
    class FakeDiscoveryDevice:
        def get_info(self, key):
            return {
                "serial_number": "rs-1",
                "name": "Intel RealSense D435",
                "product_line": "D400",
            }[key]

    fake_rs = SimpleNamespace(
        camera_info=FakeRS.camera_info,
        context=lambda: SimpleNamespace(query_devices=lambda: [FakeDiscoveryDevice()]),
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)

    devices = discover_realsense_d435()

    assert len(devices) == 1
    assert devices[0].sensor_type == SensorType.REALSENSE_D435
    assert devices[0].device_id == "rs-1"
    assert devices[0].metadata["product_line"] == "D400"
