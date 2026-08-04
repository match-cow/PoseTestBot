from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from posetestbot.io.artifacts import (
    CAM_K,
    CAMERA_DATA_JSON,
    CAMERA_JSON,
    DEPTH_DIR,
    FRAME_METADATA_JSONL,
    RGB_DIR,
)
from posetestbot.calibration.intrinsics import (
    factory_intrinsic_profile,
    projection_is_opencv_compatible,
)
from posetestbot.sensors.frame_writer import write_legacy_camera_sidecars
from posetestbot.sensors.contracts import SensorType
from posetestbot.sensors.discovery import (
    _parse_realsense_lsusb_devices,
    discover_realsense_d435,
)
from posetestbot.sensors.realsense import (
    _intrinsics_for_orientation,
    RealSenseCaptureError,
    camera_intrinsics_from_realsense,
    capture_realsense_rgbd,
)


class FakeIntrinsics:
    fx = 600.0
    fy = 601.0
    ppx = 320.0
    ppy = 240.0
    width = 1280
    height = 720
    coeffs = (0.1, -0.02, 0.003, -0.004, 0.005)
    model = "distortion.brown_conrady"


class FakeDepthIntrinsics(FakeIntrinsics):
    fx = 111.0
    fy = 112.0
    coeffs = (0.9, 0.8, 0.7, 0.6, 0.5)
    model = "distortion.inverse_brown_conrady"


class FakeFrameProfile:
    def __init__(self, *, color: bool):
        self.intrinsics = FakeIntrinsics() if color else FakeDepthIntrinsics()

    def as_video_stream_profile(self):
        return self


class FakeFrame:
    def __init__(self, index: int, *, color: bool):
        self.index = index
        self.profile = FakeFrameProfile(color=color)
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
        self.started = False
        self.stopped = False

    def start(self, _config):
        self.started = True
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


class FakeOptionSensor:
    def __init__(
        self,
        name: str,
        *,
        supported_options: set[str],
        readback_overrides: dict[str, float] | None = None,
        depth_scale: float | None = None,
    ) -> None:
        self.name = name
        self.supported_options = set(supported_options)
        self.readback_overrides = dict(readback_overrides or {})
        self.depth_scale = depth_scale
        self.values: dict[str, float] = {}
        self.set_calls: list[tuple[str, float]] = []

    def get_info(self, key):
        if key == "name":
            return self.name
        return ""

    def supports(self, option) -> bool:
        return option in self.supported_options

    def set_option(self, option, value: float) -> None:
        if option not in self.supported_options:
            raise RuntimeError(f"unsupported option: {option}")
        self.set_calls.append((option, float(value)))
        self.values[option] = float(value)

    def get_option(self, option) -> float:
        if option not in self.supported_options:
            raise RuntimeError(f"unsupported option: {option}")
        return self.readback_overrides.get(option, self.values.get(option, 0.0))

    def get_depth_scale(self) -> float:
        if self.depth_scale is None:
            raise RuntimeError("not a depth sensor")
        return self.depth_scale


class FakeOptionDevice(FakeDevice):
    def __init__(
        self,
        *,
        serial: str,
        depth_options: set[str],
        color_options: set[str],
        depth_readback_overrides: dict[str, float] | None = None,
        color_readback_overrides: dict[str, float] | None = None,
    ) -> None:
        self.serial = serial
        self.depth_sensor = FakeOptionSensor(
            "Stereo Module",
            supported_options=depth_options,
            readback_overrides=depth_readback_overrides,
            depth_scale=0.001,
        )
        self.color_sensor = FakeOptionSensor(
            "RGB Camera",
            supported_options=color_options,
            readback_overrides=color_readback_overrides,
        )
        self.sensors = [self.depth_sensor, self.color_sensor]

    def first_depth_sensor(self):
        return self.depth_sensor


class FakeHardwareSyncRS(FakeRS):
    option = SimpleNamespace(
        inter_cam_sync_mode="inter_cam_sync_mode",
        global_time_enabled="global_time_enabled",
    )

    def __init__(
        self,
        serial: str = "123",
        *,
        depth_options: set[str] | None = None,
        color_options: set[str] | None = None,
        depth_readback_overrides: dict[str, float] | None = None,
        color_readback_overrides: dict[str, float] | None = None,
    ) -> None:
        self.device = FakeOptionDevice(
            serial=serial,
            depth_options=(
                {"inter_cam_sync_mode", "global_time_enabled"}
                if depth_options is None
                else depth_options
            ),
            color_options=(
                {"global_time_enabled"} if color_options is None else color_options
            ),
            depth_readback_overrides=depth_readback_overrides,
            color_readback_overrides=color_readback_overrides,
        )
        self.pipeline_instance = None
        self.config_instance = None


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
    camera_data = json.loads((tmp_path / CAMERA_DATA_JSON).read_text())
    assert camera_data["K"][0][0] == 600.0
    assert camera_data["distortion"] == [0.1, -0.02, 0.003, -0.004, 0.005]
    assert camera_data["distortion_model"] == "brown_conrady"
    assert camera_data["projection_source"] == "realsense_sdk_color_stream"
    assert len((tmp_path / "cam_K.txt").read_text().splitlines()) == 4
    records = [
        json.loads(line)
        for line in (tmp_path / FRAME_METADATA_JSONL).read_text().splitlines()
    ]
    assert [record["frame_index"] for record in records] == [0, 1]
    assert records[0]["sensor_id"] == "825412070181"
    assert records[0]["inverted"] is False
    assert records[0]["image_rotation_degrees"] == 0


@pytest.mark.parametrize(
    ("role", "expected_mode"),
    [("master", 1), ("subordinate", 2)],
)
def test_capture_realsense_configures_verified_depth_hardware_sync_and_provenance(
    tmp_path: Path,
    role: str,
    expected_mode: int,
) -> None:
    fake_rs = FakeHardwareSyncRS("825412070181")

    summary = capture_realsense_rgbd(
        tmp_path,
        device_id="825412070181",
        fps=30,
        max_frames=1,
        hardware_sync_role=role,
        hardware_sync_group_id="mixed-rig-01",
        hardware_sync_scope="depth_exposure",
        rs_module=fake_rs,
    )

    depth_sensor = fake_rs.device.depth_sensor
    color_sensor = fake_rs.device.color_sensor
    assert depth_sensor.set_calls == [
        ("global_time_enabled", 1.0),
        ("inter_cam_sync_mode", float(expected_mode)),
    ]
    assert color_sensor.set_calls == [("global_time_enabled", 1.0)]
    assert fake_rs.pipeline_instance.started is True

    record = json.loads((tmp_path / FRAME_METADATA_JSONL).read_text())
    assert record["capture_group_id"] == "mixed-rig-01"
    assert record["hardware_sync_role"] == role
    assert record["hardware_sync_scope"] == "depth_exposure"
    assert record["hardware_sync_transport"] == "realsense_inter_cam_sync"
    assert record["inter_cam_sync_mode_configured"] == expected_mode
    assert record["inter_cam_sync_mode_readback"] == expected_mode
    assert record["depth_sensor_timestamp_ns"] == 10_000_000
    assert record["depth_frame_number"] == 1
    assert "depth_timestamp_domain" in record
    assert record["global_time_enabled_evidence"] == [
        {
            "sensor_index": 0,
            "sensor_name": "Stereo Module",
            "configured": 1,
            "readback": 1,
            "is_depth_sensor": True,
        },
        {
            "sensor_index": 1,
            "sensor_name": "RGB Camera",
            "configured": 1,
            "readback": 1,
            "is_depth_sensor": False,
        },
    ]

    assert summary["hardware_sync_enabled"] is True
    assert summary["hardware_sync_transport"] == "realsense_inter_cam_sync"
    assert summary["capture_group_id"] == "mixed-rig-01"
    assert summary["hardware_sync_role"] == role
    assert summary["hardware_sync_scope"] == "depth_exposure"
    assert summary["hardware_sync_rgb_exposure_claimed"] is False
    assert summary["inter_cam_sync_mode_configured"] == expected_mode
    assert summary["inter_cam_sync_mode_readback"] == expected_mode
    assert (
        summary["global_time_enabled_evidence"]
        == record["global_time_enabled_evidence"]
    )


def test_capture_realsense_rejects_unsupported_hardware_sync_before_stream_start(
    tmp_path: Path,
) -> None:
    fake_rs = FakeHardwareSyncRS(
        depth_options={"global_time_enabled"},
    )

    with pytest.raises(RealSenseCaptureError, match="does not support"):
        capture_realsense_rgbd(
            tmp_path,
            max_frames=1,
            hardware_sync_role="master",
            hardware_sync_group_id="group-1",
            hardware_sync_scope="depth_exposure",
            rs_module=fake_rs,
        )

    assert fake_rs.pipeline_instance.started is False
    assert not (tmp_path / RGB_DIR).exists()


def test_capture_realsense_rejects_non_d435_identity_for_hardware_sync(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_rs = FakeHardwareSyncRS("825412070181")
    original_get_info = fake_rs.device.get_info

    def device_info(key):
        if key == "product_id":
            return "0B5C"
        if key == "name":
            return "Intel RealSense D455"
        return original_get_info(key)

    monkeypatch.setattr(
        fake_rs.camera_info,
        "product_id",
        "product_id",
        raising=False,
    )
    monkeypatch.setattr(fake_rs.device, "get_info", device_info)

    with pytest.raises(RealSenseCaptureError, match="verified RealSense D435/D435i"):
        capture_realsense_rgbd(
            tmp_path,
            max_frames=1,
            hardware_sync_role="master",
            hardware_sync_group_id="group-1",
            hardware_sync_scope="depth_exposure",
            rs_module=fake_rs,
        )

    assert fake_rs.pipeline_instance.started is False
    assert not (tmp_path / RGB_DIR).exists()


def test_capture_realsense_rejects_inter_cam_sync_readback_mismatch(
    tmp_path: Path,
) -> None:
    fake_rs = FakeHardwareSyncRS(
        depth_readback_overrides={"inter_cam_sync_mode": 0.0},
    )

    with pytest.raises(RealSenseCaptureError, match="readback mismatch"):
        capture_realsense_rgbd(
            tmp_path,
            max_frames=1,
            hardware_sync_role="subordinate",
            hardware_sync_group_id="group-1",
            hardware_sync_scope="depth_exposure",
            rs_module=fake_rs,
        )

    assert fake_rs.pipeline_instance.started is False
    assert not (tmp_path / RGB_DIR).exists()


def test_capture_realsense_resets_persisted_sync_mode_without_hardware_sync(
    tmp_path: Path,
) -> None:
    fake_rs = FakeHardwareSyncRS()

    summary = capture_realsense_rgbd(
        tmp_path,
        max_frames=1,
        rs_module=fake_rs,
    )

    record = json.loads((tmp_path / FRAME_METADATA_JSONL).read_text())
    assert summary["hardware_sync_enabled"] is False
    assert summary["inter_cam_sync_mode_configured"] is None
    assert summary["inter_cam_sync_mode_readback"] is None
    assert summary["inter_cam_sync_reset_evidence"] == {
        "sensor_name": "Stereo Module",
        "configured": 0,
        "readback": 0,
    }
    assert summary["global_time_enabled_evidence"] == []
    assert (
        record["global_time_enabled_evidence"]
        == summary["global_time_enabled_evidence"]
    )
    assert fake_rs.device.depth_sensor.set_calls == [
        ("inter_cam_sync_mode", 0.0)
    ]
    assert fake_rs.device.color_sensor.set_calls == []
    assert "capture_group_id" not in record
    assert "inter_cam_sync_mode_configured" not in record
    assert record["inter_cam_sync_reset_evidence"] == (
        summary["inter_cam_sync_reset_evidence"]
    )


def test_capture_realsense_rejects_failed_sync_reset_before_default_stream(
    tmp_path: Path,
) -> None:
    fake_rs = FakeHardwareSyncRS(
        depth_readback_overrides={"inter_cam_sync_mode": 2.0},
    )

    with pytest.raises(
        RealSenseCaptureError,
        match="inter_cam_sync_mode readback mismatch",
    ):
        capture_realsense_rgbd(
            tmp_path,
            max_frames=1,
            rs_module=fake_rs,
        )

    assert fake_rs.pipeline_instance.started is False
    assert not (tmp_path / RGB_DIR).exists()


def test_capture_realsense_rejects_global_time_readback_mismatch_before_stream(
    tmp_path: Path,
) -> None:
    fake_rs = FakeHardwareSyncRS(
        color_readback_overrides={"global_time_enabled": 0.0},
    )

    with pytest.raises(
        RealSenseCaptureError, match="global_time_enabled readback mismatch"
    ):
        capture_realsense_rgbd(
            tmp_path,
            max_frames=1,
            hardware_sync_role="master",
            hardware_sync_group_id="group-1",
            hardware_sync_scope="depth_exposure",
            rs_module=fake_rs,
        )

    assert fake_rs.pipeline_instance.started is False
    assert not (tmp_path / RGB_DIR).exists()


def test_capture_realsense_rejects_hardware_sync_without_global_time_evidence(
    tmp_path: Path,
) -> None:
    fake_rs = FakeHardwareSyncRS(
        depth_options={"inter_cam_sync_mode"},
        color_options=set(),
    )

    with pytest.raises(RealSenseCaptureError, match="global_time_enabled"):
        capture_realsense_rgbd(
            tmp_path,
            max_frames=1,
            hardware_sync_role="master",
            hardware_sync_group_id="group-1",
            hardware_sync_scope="depth_exposure",
            rs_module=fake_rs,
        )

    assert fake_rs.pipeline_instance.started is False
    assert fake_rs.device.depth_sensor.set_calls == []
    assert not (tmp_path / RGB_DIR).exists()


def test_capture_realsense_rejects_color_only_global_time_evidence_before_stream(
    tmp_path: Path,
) -> None:
    fake_rs = FakeHardwareSyncRS(
        depth_options={"inter_cam_sync_mode"},
        color_options={"global_time_enabled"},
    )

    with pytest.raises(
        RealSenseCaptureError,
        match="depth sensor does not expose global_time_enabled",
    ):
        capture_realsense_rgbd(
            tmp_path,
            max_frames=1,
            hardware_sync_role="master",
            hardware_sync_group_id="group-1",
            hardware_sync_scope="depth_exposure",
            rs_module=fake_rs,
        )

    assert fake_rs.pipeline_instance.started is False
    assert fake_rs.device.depth_sensor.set_calls == []
    assert fake_rs.device.color_sensor.set_calls == []
    assert not (tmp_path / RGB_DIR).exists()


def test_realsense_capture_cli_parses_hardware_sync_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "capture_realsense_720p.py"
    )
    spec = importlib.util.spec_from_file_location(
        "capture_realsense_720p_cli_test",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            script_path.as_posix(),
            "/tmp/run/realsense_123",
            "--hardware-sync-role",
            "subordinate",
            "--hardware-sync-group-id",
            "mixed-rig-01",
            "--hardware-sync-scope",
            "depth_exposure",
        ],
    )

    args = module.parse_args()

    assert args.hardware_sync_role == "subordinate"
    assert args.hardware_sync_group_id == "mixed-rig-01"
    assert args.hardware_sync_scope == "depth_exposure"


def test_host_received_timestamp_is_sampled_before_alignment(
    tmp_path,
    monkeypatch,
) -> None:
    state = {"phase": "created"}

    class PhasePipeline(FakePipeline):
        def wait_for_frames(self):
            frames = super().wait_for_frames()
            state["phase"] = "sdk_returned"
            return frames

    class PhaseAlign(FakeAlign):
        def process(self, frames):
            state["phase"] = "aligned"
            return super().process(frames)

    class PhaseRS(FakeRS):
        def pipeline(self):
            self.pipeline_instance = PhasePipeline(self.device)
            return self.pipeline_instance

        def align(self, stream):
            return PhaseAlign(stream)

    def monotonic_ns() -> int:
        assert state["phase"] == "sdk_returned"
        state["phase"] = "timestamped"
        return 123_456_789

    monkeypatch.setattr(
        "posetestbot.sensors.realsense.time.monotonic_ns",
        monotonic_ns,
    )

    capture_realsense_rgbd(
        tmp_path,
        max_frames=1,
        rs_module=PhaseRS(),
    )

    record = json.loads((tmp_path / FRAME_METADATA_JSONL).read_text())
    assert record["host_received_timestamp_ns"] == 123_456_789


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


def test_capture_realsense_rgbd_honors_graceful_stop_between_frames(tmp_path) -> None:
    fake_rs = FakeRS("123")

    summary = capture_realsense_rgbd(
        tmp_path,
        device_id="123",
        fps=6,
        max_frames=0,
        preview=False,
        stop_requested=lambda: fake_rs.pipeline_instance.index >= 2,
        rs_module=fake_rs,
    )

    assert summary["frame_count"] == 1
    assert fake_rs.pipeline_instance.stopped is True
    assert len(list((tmp_path / RGB_DIR).glob("*.png"))) == 1
    assert len(list((tmp_path / DEPTH_DIR).glob("*.png"))) == 1
    assert len((tmp_path / FRAME_METADATA_JSONL).read_text().splitlines()) == 1


def test_capture_realsense_rgbd_inverted_rotates_frames_and_intrinsics(
    tmp_path,
) -> None:
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
    assert camera["distortion"] == [0.1, -0.02, -0.003, 0.004, 0.005]
    assert camera["distortion_model"] == "brown_conrady"
    assert camera["projection_source"] == ("realsense_sdk_color_stream_rotated_180")
    records = [
        json.loads(line)
        for line in (tmp_path / FRAME_METADATA_JSONL).read_text().splitlines()
    ]
    assert records[0]["inverted"] is True
    assert records[0]["image_rotation_degrees"] == 180
    assert records[0]["orientation"] == "inverted"


def test_inverse_sdk_distortion_is_preserved_but_not_misapplied_to_opencv(
    tmp_path,
) -> None:
    sdk_intrinsics = SimpleNamespace(
        fx=600.0,
        fy=601.0,
        ppx=320.0,
        ppy=240.0,
        width=1280,
        height=720,
        coeffs=(0.1, -0.02, 0.003, -0.004, 0.005),
        model="distortion.inverse_brown_conrady",
    )
    native = camera_intrinsics_from_realsense(sdk_intrinsics, 1.0)
    inverted = _intrinsics_for_orientation(native, inverted=True)

    assert inverted.distortion_model == "inverse_brown_conrady"
    assert inverted.distortion == (0.1, -0.02, -0.003, 0.004, 0.005)
    write_legacy_camera_sidecars(tmp_path, inverted)

    assert len((tmp_path / CAM_K).read_text().splitlines()) == 3
    profile = factory_intrinsic_profile(tmp_path)
    assert profile["native"]["distortion"] == [
        0.1,
        -0.02,
        -0.003,
        0.004,
        0.005,
    ]
    assert profile["native"]["distortion_model"] == "inverse_brown_conrady"
    assert profile["source"]["opencv_projection_compatible"] is False
    assert profile["source"]["rectification_available"] is False
    assert profile["rectified"] is None
    assert projection_is_opencv_compatible(profile["native"]) is False


def test_exact_zero_inverse_distortion_is_model_invariant_for_opencv(
    tmp_path,
) -> None:
    sdk_intrinsics = SimpleNamespace(
        fx=600.0,
        fy=601.0,
        ppx=320.0,
        ppy=240.0,
        width=1280,
        height=720,
        coeffs=(0.0, -0.0, 0.0, 0.0, 0.0),
        model="distortion.inverse_brown_conrady",
    )
    native = camera_intrinsics_from_realsense(sdk_intrinsics, 1.0)
    write_legacy_camera_sidecars(tmp_path, native)

    profile = factory_intrinsic_profile(tmp_path)

    assert profile["native"]["distortion_model"] == "inverse_brown_conrady"
    assert profile["native"]["distortion"] == [0.0] * 5
    assert projection_is_opencv_compatible(profile["native"]) is True
    assert profile["source"]["opencv_projection_compatible"] is True
    assert profile["source"]["opencv_projection_compatibility_basis"] == (
        "exact_zero_distortion_is_model_invariant"
    )
    assert profile["source"]["rectification_available"] is True
    assert profile["source"]["rectification_unavailable_reason"] is None
    assert profile["rectified"] is not None
    assert profile["rectified"]["distortion"] == [0.0] * 5
    assert (
        projection_is_opencv_compatible(
            {
                "distortion_model": "kannala_brandt4",
                "distortion": [0.0] * 5,
            }
        )
        is False
    )


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
    monkeypatch.setattr(
        "posetestbot.sensors.discovery._video_node_metadata_by_serial",
        lambda: {},
    )
    monkeypatch.setattr(
        "posetestbot.sensors.discovery._discover_realsense_from_lsusb",
        lambda: [],
    )

    devices = discover_realsense_d435()

    assert len(devices) == 1
    assert devices[0].sensor_type == SensorType.REALSENSE_D435
    assert devices[0].device_id == "rs-1"
    assert devices[0].metadata["product_line"] == "D400"


def test_discover_realsense_d435_filters_other_realsense_models(monkeypatch) -> None:
    class FakeCameraInfo:
        serial_number = "serial_number"
        name = "name"
        product_line = "product_line"
        product_id = "product_id"

    class FakeDiscoveryDevice:
        def __init__(self, serial: str, name: str, product_id: str):
            self.values = {
                "serial_number": serial,
                "name": name,
                "product_line": "D400",
                "product_id": product_id,
            }

        def supports(self, _key):
            return True

        def get_info(self, key):
            return self.values[key]

    fake_rs = SimpleNamespace(
        camera_info=FakeCameraInfo,
        context=lambda: SimpleNamespace(
            query_devices=lambda: [
                FakeDiscoveryDevice("d435", "Intel RealSense D435", "0B07"),
                FakeDiscoveryDevice("d455", "Intel RealSense D455", "0B5C"),
            ]
        ),
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)
    monkeypatch.setattr(
        "posetestbot.sensors.discovery._video_node_metadata_by_serial",
        lambda: {},
    )
    monkeypatch.setattr(
        "posetestbot.sensors.discovery._discover_realsense_from_lsusb",
        lambda: [],
    )

    devices = discover_realsense_d435()

    assert [device.device_id for device in devices] == ["d435"]


def test_parse_realsense_lsusb_fallback_reads_d435_and_d435i() -> None:
    devices = _parse_realsense_lsusb_devices(
        """
Bus 003 Device 005: ID 8086:0b07 Intel Corp. RealSense D435
  iProduct                2 Intel(R) RealSense(TM) Depth Camera 435
  iSerial                 3 926223021865
Bus 003 Device 008: ID 8086:0b3a Intel Corp. Intel(R) RealSense(TM) Depth Camera 435i
  iProduct                2 Intel(R) RealSense(TM) Depth Camera 435i
  iSerial                 3 923322072633
"""
    )

    assert [device.device_id for device in devices] == [
        "926223021865",
        "923322072633",
    ]
    assert devices[0].metadata["product_id"] == "0b07"
    assert devices[1].metadata["product_id"] == "0b3a"
