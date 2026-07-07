"""RealSense D435/D435i capture helpers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from posetestbot.sensors.contracts import CameraIntrinsics, SensorType
from posetestbot.sensors.frame_writer import (
    ensure_legacy_rgbd_folders,
    write_legacy_camera_sidecars,
    write_legacy_rgbd_frame,
)


CAPTURE_SUMMARY_SCHEMA_VERSION = "realsense_capture_summary.v1"


class RealSenseCaptureError(RuntimeError):
    """Raised when a RealSense capture cannot be started or completed."""


def camera_intrinsics_from_realsense(
    intrinsics: Any,
    depth_scale_to_mm: float,
) -> CameraIntrinsics:
    return CameraIntrinsics(
        cam_k=(
            float(intrinsics.fx),
            0.0,
            float(intrinsics.ppx),
            0.0,
            float(intrinsics.fy),
            float(intrinsics.ppy),
            0.0,
            0.0,
            1.0,
        ),
        width=int(getattr(intrinsics, "width", 1280)),
        height=int(getattr(intrinsics, "height", 720)),
        depth_scale_to_mm=float(depth_scale_to_mm),
    )


def _import_realsense() -> Any:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RealSenseCaptureError(
            "pyrealsense2 is not importable in the current uv environment."
        ) from exc
    return rs


def _device_info(device: Any, rs: Any, key_name: str, default: str = "unknown") -> str:
    try:
        return str(device.get_info(getattr(rs.camera_info, key_name)))
    except Exception:
        return default


def _resolve_device(config: Any, pipeline: Any, rs: Any, device_id: str | None) -> Any:
    target = f" serial {device_id}" if device_id else ""
    try:
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        return pipeline_profile.get_device()
    except Exception as exc:
        raise RealSenseCaptureError(
            f"Unable to resolve RealSense device{target}: {type(exc).__name__}: {exc}"
        ) from exc


def _has_rgb_sensor(device: Any, rs: Any) -> bool:
    for sensor in getattr(device, "sensors", []):
        try:
            if sensor.get_info(rs.camera_info.name) == "RGB Camera":
                return True
        except Exception:
            continue
    return False


def _enable_streams(config: Any, rs: Any, *, fps: int, product_line: str) -> None:
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, fps)
    if product_line == "L500":
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, fps)
    else:
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, fps)


def _cv2_for_preview(cv2_module: Any | None) -> Any:
    if cv2_module is not None:
        return cv2_module
    try:
        import cv2
    except ImportError as exc:
        raise RealSenseCaptureError(
            "OpenCV is required for --preview but cv2 is not importable."
        ) from exc
    return cv2


def capture_realsense_rgbd(
    output_path: str | Path | None,
    *,
    device_id: str | None = None,
    fps: int = 30,
    max_frames: int = 0,
    warmup_frames: int = 0,
    preview: bool = False,
    record: bool = True,
    rs_module: Any | None = None,
    cv2_module: Any | None = None,
) -> dict[str, Any]:
    """Capture aligned RealSense RGB-D frames into the legacy folder contract."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    if max_frames < 0:
        raise ValueError("max_frames must be greater than or equal to 0")
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be greater than or equal to 0")
    if record and output_path is None:
        raise ValueError("output_path is required when record=True")

    output = Path(output_path) if output_path is not None else None
    rs = rs_module or _import_realsense()
    cv2_preview = _cv2_for_preview(cv2_module) if preview else None

    pipeline = rs.pipeline()
    config = rs.config()
    if device_id:
        try:
            config.enable_device(device_id)
        except Exception as exc:
            raise RealSenseCaptureError(
                f"Unable to select RealSense serial {device_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    device = _resolve_device(config, pipeline, rs, device_id)
    product_line = _device_info(device, rs, "product_line")
    resolved_serial = _device_info(device, rs, "serial_number", device_id or "default")
    device_name = _device_info(device, rs, "name", "RealSense")
    if not _has_rgb_sensor(device, rs):
        raise RealSenseCaptureError(
            f"RealSense device {resolved_serial} does not expose an RGB Camera sensor."
        )

    _enable_streams(config, rs, fps=fps, product_line=product_line)

    try:
        profile = pipeline.start(config)
    except Exception as exc:
        raise RealSenseCaptureError(
            f"Unable to start RealSense stream for {resolved_serial}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    metadata_records: list[dict[str, Any]] = []
    sidecar_paths: dict[str, str] = {}
    captured_frames = 0
    valid_frames_seen = 0
    sidecars_written = False
    started_at_ns = time.time_ns()
    last_frame_wall_timestamp_ns = 0

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale_to_mm = float(depth_sensor.get_depth_scale()) * 1000.0
    align = rs.align(rs.stream.color)

    if record and output is not None:
        ensure_legacy_rgbd_folders(output)

    try:
        while max_frames <= 0 or captured_frames < max_frames:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not color_frame:
                continue

            if valid_frames_seen < warmup_frames:
                valid_frames_seen += 1
                if cv2_preview is not None:
                    color_image = np.asanyarray(color_frame.get_data())
                    cv2_preview.imshow("RealSense Capture RGB aligned", color_image)
                    key = cv2_preview.waitKey(1)
                    if key & 0xFF == ord("q") or key == 27:
                        break
                continue

            intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            camera_intrinsics = camera_intrinsics_from_realsense(
                intrinsics,
                depth_scale_to_mm,
            )
            if record and output is not None and not sidecars_written:
                written = write_legacy_camera_sidecars(output, camera_intrinsics)
                sidecar_paths = {key: path.name for key, path in written.items()}
                sidecars_written = True

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            key = -1
            if cv2_preview is not None:
                cv2_preview.imshow("RealSense Capture RGB aligned", color_image)
                key = cv2_preview.waitKey(1)

            if record and output is not None:
                host_wall_timestamp_ns = time.time_ns()
                min_next_wall_timestamp_ns = last_frame_wall_timestamp_ns + 1_000_000
                if host_wall_timestamp_ns < min_next_wall_timestamp_ns:
                    host_wall_timestamp_ns = min_next_wall_timestamp_ns
                last_frame_wall_timestamp_ns = host_wall_timestamp_ns
                metadata = write_legacy_rgbd_frame(
                    output,
                    rgb_image=color_image,
                    depth_image=depth_image,
                    sensor_type=SensorType.REALSENSE_D435,
                    sensor_id=resolved_serial,
                    frame_index=captured_frames,
                    sensor_timestamp_ns=int(color_frame.get_timestamp() * 1_000_000),
                    depth_sensor_timestamp_ns=int(
                        aligned_depth_frame.get_timestamp() * 1_000_000
                    ),
                    host_received_timestamp_ns=time.monotonic_ns(),
                    host_wall_timestamp_ns=host_wall_timestamp_ns,
                    extra_metadata={
                        "color_frame_number": color_frame.get_frame_number(),
                        "depth_frame_number": aligned_depth_frame.get_frame_number(),
                        "product_line": product_line,
                    },
                )
                metadata_records.append(metadata)

            captured_frames += 1
            valid_frames_seen += 1
            if key & 0xFF == ord("q") or key == 27:
                break
    finally:
        try:
            pipeline.stop()
        finally:
            if cv2_preview is not None:
                cv2_preview.destroyAllWindows()

    first_metadata = metadata_records[0] if metadata_records else {}
    last_metadata = metadata_records[-1] if metadata_records else {}
    return {
        "schema_version": CAPTURE_SUMMARY_SCHEMA_VERSION,
        "status": "succeeded",
        "sensor_type": SensorType.REALSENSE_D435.value,
        "sensor_id": resolved_serial,
        "requested_device_id": device_id,
        "display_name": f"{device_name} {resolved_serial}".strip(),
        "product_line": product_line,
        "output_path": output.as_posix() if output is not None else None,
        "record": record,
        "preview": preview,
        "fps": fps,
        "max_frames": max_frames,
        "warmup_frames": warmup_frames,
        "frame_count": captured_frames,
        "sidecars": sidecar_paths,
        "first_frame_id": first_metadata.get("frame_id"),
        "last_frame_id": last_metadata.get("frame_id"),
        "first_sensor_timestamp_ns": first_metadata.get("sensor_timestamp_ns"),
        "last_sensor_timestamp_ns": last_metadata.get("sensor_timestamp_ns"),
        "started_at_ns": started_at_ns,
        "ended_at_ns": time.time_ns(),
    }
