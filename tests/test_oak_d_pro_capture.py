from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np

from posetestbot.io.artifacts import DEPTH_DIR, FRAME_METADATA_JSONL, RGB_DIR
from posetestbot.sensors.oak_d_pro import (
    OAKDProCaptureError,
    TIMESTAMP_SOURCE,
    camera_intrinsics_from_matrix,
    capture_oak_d_pro_rgbd,
    dai_timestamp_ns,
    estimate_capture_wall_timestamp_ns,
    rgb_depth_pair_is_usable,
    rgb_depth_timestamp_delta_ns,
    write_oak_d_pro_rgbd_frame,
)


class FakeDepthAIFrame:
    def __init__(
        self,
        *,
        timestamp_ns: int,
        device_timestamp_ns: int,
        sequence_num: int,
        rgb_image=None,
        depth_image=None,
    ):
        self.timestamp_ns = timestamp_ns
        self.device_timestamp_ns = device_timestamp_ns
        self.sequence_num = sequence_num
        self.rgb_image = rgb_image
        self.depth_image = depth_image

    def getTimestamp(self) -> timedelta:
        return timedelta(microseconds=self.timestamp_ns // 1_000)

    def getTimestampDevice(self) -> timedelta:
        return timedelta(microseconds=self.device_timestamp_ns // 1_000)

    def getSequenceNum(self) -> int:
        return self.sequence_num

    def getCvFrame(self):
        return self.rgb_image

    def getFrame(self):
        return self.depth_image


class FakeDaiNoDevice:
    __version__ = "3.7.1"

    class Device:
        def __init__(self):
            raise RuntimeError("No available devices")


def test_depthai_timestamp_helpers_use_host_synced_clock() -> None:
    rgb = FakeDepthAIFrame(
        timestamp_ns=4_900_123_000,
        device_timestamp_ns=9_000_000_000,
        sequence_num=7,
    )
    depth = FakeDepthAIFrame(
        timestamp_ns=4_902_123_000,
        device_timestamp_ns=9_002_000_000,
        sequence_num=8,
    )

    assert dai_timestamp_ns(rgb) == 4_900_123_000
    assert dai_timestamp_ns(rgb, device_clock=True) == 9_000_000_000
    assert rgb_depth_timestamp_delta_ns(rgb, depth) == 2_000_000
    assert rgb_depth_pair_is_usable(rgb, depth, max_delta_ns=2_000_000) is True
    assert rgb_depth_pair_is_usable(rgb, depth, max_delta_ns=1_999_999) is False

    host_wall_received_ns = 1_700_000_000_010_000_000
    assert (
        estimate_capture_wall_timestamp_ns(
            frame_depthai_timestamp_ns=4_900_123_000,
            depthai_now_ns=5_000_123_000,
            host_wall_received_timestamp_ns=host_wall_received_ns,
        )
        == host_wall_received_ns - 100_000_000
    )


def test_write_oak_d_pro_rgbd_frame_uses_capture_wall_timestamp(
    tmp_path: Path,
) -> None:
    rgb_image = np.zeros((3, 4, 3), dtype=np.uint8)
    rgb_image[:, :, 1] = 128
    depth_image = np.ones((3, 4), dtype=np.uint16) * 42
    rgb = FakeDepthAIFrame(
        timestamp_ns=4_900_123_000,
        device_timestamp_ns=9_000_000_000,
        sequence_num=7,
        rgb_image=rgb_image,
    )
    depth = FakeDepthAIFrame(
        timestamp_ns=4_901_123_000,
        device_timestamp_ns=9_001_000_000,
        sequence_num=8,
        depth_image=depth_image,
    )
    host_wall_received_ns = 1_700_000_000_010_000_000
    capture_wall_ns = host_wall_received_ns - 100_000_000

    metadata = write_oak_d_pro_rgbd_frame(
        tmp_path,
        rgb_packet=rgb,
        depth_packet=depth,
        sensor_id="mxid-1",
        frame_index=3,
        host_received_timestamp_ns=123,
        host_wall_received_timestamp_ns=host_wall_received_ns,
        depthai_now_ns=5_000_123_000,
    )

    assert metadata["frame_id"] == f"{capture_wall_ns // 1_000_000}.png"
    assert metadata["sensor_type"] == "oak_d_pro"
    assert metadata["sensor_id"] == "mxid-1"
    assert metadata["sensor_timestamp_ns"] == 4_900_123_000
    assert metadata["depth_sensor_timestamp_ns"] == 4_901_123_000
    assert metadata["device_timestamp_ns"] == 9_000_000_000
    assert metadata["depth_device_timestamp_ns"] == 9_001_000_000
    assert metadata["capture_wall_timestamp_ns"] == capture_wall_ns
    assert metadata["host_wall_timestamp_ns"] == capture_wall_ns
    assert metadata["host_wall_received_timestamp_ns"] == host_wall_received_ns
    assert metadata["host_received_timestamp_ns"] == 123
    assert metadata["rgb_sequence_num"] == 7
    assert metadata["depth_sequence_num"] == 8
    assert metadata["rgb_depth_timestamp_delta_ns"] == 1_000_000
    assert metadata["timestamp_source"] == TIMESTAMP_SOURCE

    rgb_path = tmp_path / RGB_DIR / metadata["frame_id"]
    depth_path = tmp_path / DEPTH_DIR / metadata["frame_id"]
    assert rgb_path.is_file()
    assert depth_path.is_file()
    assert cv2.imread(rgb_path.as_posix()).shape == (3, 4, 3)
    assert cv2.imread(depth_path.as_posix(), cv2.IMREAD_UNCHANGED).dtype == np.uint16
    assert json.loads((tmp_path / FRAME_METADATA_JSONL).read_text()) == metadata


def test_camera_intrinsics_from_matrix_validates_shape() -> None:
    intrinsics = camera_intrinsics_from_matrix(
        np.array([[100.0, 0.0, 2.0], [0.0, 101.0, 3.0], [0.0, 0.0, 1.0]]),
        width=4,
        height=3,
        depth_scale_to_mm=1.0,
    )

    assert intrinsics.width == 4
    assert intrinsics.height == 3
    assert intrinsics.cam_k == (
        100.0,
        0.0,
        2.0,
        0.0,
        101.0,
        3.0,
        0.0,
        0.0,
        1.0,
    )


def test_capture_wraps_depthai_device_open_failures(tmp_path: Path) -> None:
    try:
        capture_oak_d_pro_rgbd(tmp_path, max_frames=1, dai_module=FakeDaiNoDevice)
    except OAKDProCaptureError as exc:
        assert "Unable to open OAK-D Pro device" in str(exc)
        assert "No available devices" in str(exc)
    else:
        raise AssertionError("device-open failure was not wrapped")
