"""Shared RGB-D frame and calibration contracts for sensor adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class SensorType(StrEnum):
    REALSENSE_D435 = "realsense_d435"
    OAK_D_PRO = "oak_d_pro"
    ZED_2I = "zed_2i"


class MountingMode(StrEnum):
    EYE_IN_HAND = "eye_in_hand"
    STATIC = "static"


@dataclass(frozen=True)
class CameraIntrinsics:
    """OpenCV-style pinhole intrinsics plus optional distortion."""

    cam_k: tuple[float, float, float, float, float, float, float, float, float]
    width: int
    height: int
    distortion: tuple[float, ...] = ()
    depth_scale_to_mm: float = 1.0
    distortion_model: str = "brown_conrady"
    projection_source: str | None = None

    def as_matrix_rows(self) -> tuple[tuple[float, float, float], ...]:
        return (
            (self.cam_k[0], self.cam_k[1], self.cam_k[2]),
            (self.cam_k[3], self.cam_k[4], self.cam_k[5]),
            (self.cam_k[6], self.cam_k[7], self.cam_k[8]),
        )


@dataclass(frozen=True)
class AlignedRgbdFrame:
    """Canonical frame object every capture adapter should eventually return."""

    sensor_id: str
    sensor_type: SensorType
    frame_index: int
    sensor_timestamp_ns: int | None
    host_received_timestamp_ns: int
    rgb_image: Any
    depth_image_aligned_to_rgb: Any
    intrinsics: CameraIntrinsics
    exposure_metadata: Mapping[str, Any] = field(default_factory=dict)
    camera_pose_hint: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SensorDeviceInfo:
    """Minimal discovery result for UI/status views."""

    sensor_type: SensorType
    device_id: str
    display_name: str
    connected: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
