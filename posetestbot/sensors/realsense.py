"""RealSense D435/D435i capture helpers."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from posetestbot.sensors.contracts import CameraIntrinsics, SensorType
from posetestbot.sensors.discovery import is_realsense_d435_identity
from posetestbot.sensors.frame_writer import (
    ensure_legacy_rgbd_folders,
    sync_frame_metadata,
    write_legacy_camera_sidecars,
    write_legacy_rgbd_frame,
)
from posetestbot.sensors.registry import (
    REALSENSE_HARDWARE_SYNC_TRANSPORT,
    validate_hardware_sync_request,
)


CAPTURE_SUMMARY_SCHEMA_VERSION = "realsense_capture_summary.v1"


class RealSenseCaptureError(RuntimeError):
    """Raised when a RealSense capture cannot be started or completed."""


def _distortion_model_name(value: Any) -> str:
    """Return the stable librealsense distortion name without importing its SDK."""

    if value is None:
        return "unknown"
    name = str(value).strip().lower().rsplit(".", 1)[-1]
    aliases = {
        "0": "none",
        "1": "modified_brown_conrady",
        "2": "inverse_brown_conrady",
        "3": "ftheta",
        "4": "brown_conrady",
        "5": "kannala_brandt4",
    }
    return aliases.get(name, name)


def _opencv_projection_compatible(model: str) -> bool:
    # Modified Brown-Conrady, like ordinary Brown-Conrady, is a forward
    # undistorted-to-distorted projection. Inverse Brown coefficients are not.
    return model in {"none", "brown_conrady", "modified_brown_conrady"}


def camera_intrinsics_from_realsense(
    intrinsics: Any,
    depth_scale_to_mm: float,
) -> CameraIntrinsics:
    raw_coefficients = getattr(intrinsics, "coeffs", ())
    distortion = tuple(
        float(value) for value in (() if raw_coefficients is None else raw_coefficients)
    )
    model = _distortion_model_name(getattr(intrinsics, "model", None))
    if model == "unknown" and not any(distortion):
        model = "none"
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
        distortion=distortion,
        depth_scale_to_mm=float(depth_scale_to_mm),
        distortion_model=model,
        projection_source="realsense_sdk_color_stream",
    )


def _rotate_180(image: Any) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(np.asanyarray(image), 2))


def _intrinsics_for_orientation(
    intrinsics: CameraIntrinsics,
    *,
    inverted: bool,
) -> CameraIntrinsics:
    if not inverted:
        return intrinsics

    cam_k = list(intrinsics.cam_k)
    cam_k[2] = float(intrinsics.width - 1) - float(cam_k[2])
    cam_k[5] = float(intrinsics.height - 1) - float(cam_k[5])
    distortion = list(intrinsics.distortion)
    if (
        intrinsics.distortion_model
        in {
            "brown_conrady",
            "modified_brown_conrady",
            "inverse_brown_conrady",
        }
        and len(distortion) >= 4
    ):
        # A 180-degree image rotation negates normalized x/y. Radial terms are
        # unchanged, while both tangential terms change sign. The inverse model
        # remains tagged inverse and is never passed to OpenCV as a forward model.
        distortion[2] = -float(distortion[2])
        distortion[3] = -float(distortion[3])
    return CameraIntrinsics(
        cam_k=(
            float(cam_k[0]),
            float(cam_k[1]),
            float(cam_k[2]),
            float(cam_k[3]),
            float(cam_k[4]),
            float(cam_k[5]),
            float(cam_k[6]),
            float(cam_k[7]),
            float(cam_k[8]),
        ),
        width=intrinsics.width,
        height=intrinsics.height,
        distortion=tuple(float(value) for value in distortion),
        depth_scale_to_mm=intrinsics.depth_scale_to_mm,
        distortion_model=intrinsics.distortion_model,
        projection_source=(
            f"{intrinsics.projection_source}_rotated_180"
            if intrinsics.projection_source
            else "rotated_180"
        ),
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


def _frame_timestamp_ns(frame: Any) -> int | None:
    try:
        return int(float(frame.get_timestamp()) * 1_000_000)
    except Exception:
        return None


def _frame_stem_from_sensor_timestamp_ns(timestamp_ns: int) -> str:
    return str(int(timestamp_ns) // 1_000_000)


def _timestamp_domain_name(frame: Any) -> str | None:
    try:
        domain = frame.get_frame_timestamp_domain()
    except Exception:
        return None

    name = getattr(domain, "name", None)
    if isinstance(name, str):
        return name
    if callable(name):
        try:
            return str(name())
        except Exception:
            pass

    value = str(domain)
    return value.rsplit(".", 1)[-1] if value else None


def _metadata_value(frame: Any, rs: Any, name: str) -> int | None:
    try:
        metadata_key = getattr(rs.frame_metadata_value, name)
    except Exception:
        return None
    try:
        if not frame.supports_frame_metadata(metadata_key):
            return None
        return int(frame.get_frame_metadata(metadata_key))
    except Exception:
        return None


def _realsense_timestamp_metadata(
    color_frame: Any,
    depth_frame: Any,
    rs: Any,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "color_timestamp_domain": _timestamp_domain_name(color_frame),
        "depth_timestamp_domain": _timestamp_domain_name(depth_frame),
    }
    for frame_name, frame in (
        ("color", color_frame),
        ("depth", depth_frame),
    ):
        for metadata_name in (
            "backend_timestamp",
            "frame_timestamp",
            "sensor_timestamp",
            "time_of_arrival",
        ):
            value = _metadata_value(frame, rs, metadata_name)
            if value is not None:
                metadata[f"{frame_name}_{metadata_name}"] = value
    return metadata


def _sensor_supports_option(sensor: Any, option: Any) -> bool:
    supports = getattr(sensor, "supports", None)
    if callable(supports):
        try:
            return bool(supports(option))
        except Exception:
            return False
    return callable(getattr(sensor, "set_option", None)) and callable(
        getattr(sensor, "get_option", None)
    )


def _sensor_name(sensor: Any, rs: Any, *, fallback: str) -> str:
    try:
        return str(sensor.get_info(rs.camera_info.name))
    except Exception:
        return fallback


def _set_integer_option_and_verify(
    sensor: Any,
    option: Any,
    configured: int,
    *,
    option_name: str,
    sensor_name: str,
) -> int:
    try:
        sensor.set_option(option, float(configured))
        raw_readback = float(sensor.get_option(option))
    except Exception as exc:
        raise RealSenseCaptureError(
            f"Unable to configure RealSense {option_name}={configured} on "
            f"{sensor_name}: {type(exc).__name__}: {exc}"
        ) from exc
    if not math.isfinite(raw_readback) or not math.isclose(
        raw_readback, float(configured), abs_tol=1e-6
    ):
        raise RealSenseCaptureError(
            f"RealSense {option_name} readback mismatch on {sensor_name}: "
            f"configured {configured}, read back {raw_readback}."
        )
    return int(round(raw_readback))


def _configure_global_time(
    device: Any,
    depth_sensor: Any,
    rs: Any,
) -> list[dict[str, Any]]:
    """Enable global timestamps and require verified depth-sensor readback."""

    option_namespace = getattr(rs, "option", None)
    option = getattr(option_namespace, "global_time_enabled", None)
    if option is None:
        raise RealSenseCaptureError(
            "RealSense SDK does not expose global_time_enabled; "
            "refusing hardware-synchronized capture."
        )

    sensors = list(getattr(device, "sensors", ()) or ())
    depth_sensor_index = next(
        (
            index
            for index, sensor in enumerate(sensors)
            if sensor is depth_sensor
        ),
        None,
    )
    if depth_sensor_index is None:
        sensors.append(depth_sensor)
        depth_sensor_index = len(sensors) - 1

    if not _sensor_supports_option(depth_sensor, option):
        raise RealSenseCaptureError(
            "RealSense depth sensor does not expose global_time_enabled; "
            "evidence from another sensor is insufficient for "
            "hardware-synchronized depth capture."
        )

    evidence: list[dict[str, Any]] = []
    ordered_sensors = [
        (depth_sensor_index, depth_sensor, True),
        *[
            (index, sensor, False)
            for index, sensor in enumerate(sensors)
            if index != depth_sensor_index
        ],
    ]
    for index, sensor, is_depth_sensor in ordered_sensors:
        if not _sensor_supports_option(sensor, option):
            continue
        name = _sensor_name(sensor, rs, fallback=f"sensor_{index}")
        readback = _set_integer_option_and_verify(
            sensor,
            option,
            1,
            option_name="global_time_enabled",
            sensor_name=name,
        )
        evidence.append(
            {
                "sensor_index": index,
                "sensor_name": name,
                "configured": 1,
                "readback": readback,
                "is_depth_sensor": is_depth_sensor,
            }
        )
    return evidence


def _configure_inter_cam_sync(
    depth_sensor: Any,
    rs: Any,
    *,
    hardware_sync_role: str,
) -> tuple[int, int]:
    option_namespace = getattr(rs, "option", None)
    option = getattr(option_namespace, "inter_cam_sync_mode", None)
    if option is None or not _sensor_supports_option(depth_sensor, option):
        raise RealSenseCaptureError(
            "RealSense depth sensor does not support inter_cam_sync_mode; "
            "refusing hardware-synchronized capture."
        )

    configured = 1 if hardware_sync_role == "master" else 2
    sensor_name = _sensor_name(depth_sensor, rs, fallback="depth sensor")
    readback = _set_integer_option_and_verify(
        depth_sensor,
        option,
        configured,
        option_name="inter_cam_sync_mode",
        sensor_name=sensor_name,
    )
    return configured, readback


def _reset_inter_cam_sync(depth_sensor: Any, rs: Any) -> dict[str, Any] | None:
    """Clear a master/slave mode that may persist across capture processes."""

    option_namespace = getattr(rs, "option", None)
    option = getattr(option_namespace, "inter_cam_sync_mode", None)
    if option is None or not _sensor_supports_option(depth_sensor, option):
        return None
    sensor_name = _sensor_name(depth_sensor, rs, fallback="depth sensor")
    readback = _set_integer_option_and_verify(
        depth_sensor,
        option,
        0,
        option_name="inter_cam_sync_mode",
        sensor_name=sensor_name,
    )
    return {
        "sensor_name": sensor_name,
        "configured": 0,
        "readback": readback,
    }


def capture_realsense_rgbd(
    output_path: str | Path | None,
    *,
    device_id: str | None = None,
    fps: int = 30,
    max_frames: int = 0,
    warmup_frames: int = 0,
    preview: bool = False,
    record: bool = True,
    inverted: bool = False,
    hardware_sync_role: str | None = None,
    hardware_sync_group_id: str | None = None,
    hardware_sync_scope: str | None = None,
    stop_requested: Callable[[], bool] | None = None,
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
    hardware_sync = validate_hardware_sync_request(
        sensor_type=SensorType.REALSENSE_D435,
        hardware_sync_role=hardware_sync_role,
        hardware_sync_group_id=hardware_sync_group_id,
        hardware_sync_scope=hardware_sync_scope,
    )

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
    product_id = _device_info(device, rs, "product_id", "")
    if hardware_sync is not None and not is_realsense_d435_identity(
        name=device_name,
        product_id=product_id or None,
    ):
        raise RealSenseCaptureError(
            "Hardware-synchronized capture is restricted to verified "
            f"RealSense D435/D435i devices; resolved {device_name!r} "
            f"(product_id={product_id or 'unavailable'!r})."
        )
    if not _has_rgb_sensor(device, rs):
        raise RealSenseCaptureError(
            f"RealSense device {resolved_serial} does not expose an RGB Camera sensor."
        )

    try:
        prestart_depth_sensor = device.first_depth_sensor()
    except Exception as exc:
        raise RealSenseCaptureError(
            f"Unable to access RealSense depth sensor for {resolved_serial}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    global_time_enabled_evidence: list[dict[str, Any]] = []
    inter_cam_sync_reset_evidence: dict[str, Any] | None = None
    inter_cam_sync_mode_configured: int | None = None
    inter_cam_sync_mode_readback: int | None = None
    if hardware_sync is not None:
        global_time_enabled_evidence = _configure_global_time(
            device,
            prestart_depth_sensor,
            rs,
        )
        (
            inter_cam_sync_mode_configured,
            inter_cam_sync_mode_readback,
        ) = _configure_inter_cam_sync(
            prestart_depth_sensor,
            rs,
            hardware_sync_role=hardware_sync["role"],
        )
    else:
        inter_cam_sync_reset_evidence = _reset_inter_cam_sync(
            prestart_depth_sensor,
            rs,
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
    image_rotation_degrees = 180 if inverted else 0
    orientation = "inverted" if inverted else "normal"

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale_to_mm = float(depth_sensor.get_depth_scale()) * 1000.0
    align = rs.align(rs.stream.color)

    if record and output is not None:
        ensure_legacy_rgbd_folders(output)

    try:
        while (max_frames <= 0 or captured_frames < max_frames) and not (
            stop_requested and stop_requested()
        ):
            frames = pipeline.wait_for_frames()
            # Capture host receipt at the first userspace point after the SDK
            # delivers the frameset.  Alignment, conversion, preview and disk
            # work must not become part of the robot/camera sync timestamp.
            host_received_timestamp_ns = time.monotonic_ns()
            if stop_requested and stop_requested():
                break
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not color_frame:
                continue

            if valid_frames_seen < warmup_frames:
                valid_frames_seen += 1
                if cv2_preview is not None:
                    color_image = np.asanyarray(color_frame.get_data())
                    if inverted:
                        color_image = _rotate_180(color_image)
                    cv2_preview.imshow("RealSense Capture RGB aligned", color_image)
                    key = cv2_preview.waitKey(1)
                    if key & 0xFF == ord("q") or key == 27:
                        break
                continue

            # PnP uses the RGB image, so provenance must come from the aligned
            # color profile rather than relying on the aligned-depth profile to
            # happen to expose equivalent calibration.
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            camera_intrinsics = camera_intrinsics_from_realsense(
                intrinsics,
                depth_scale_to_mm,
            )
            camera_intrinsics = _intrinsics_for_orientation(
                camera_intrinsics,
                inverted=inverted,
            )
            if record and output is not None and not sidecars_written:
                written = write_legacy_camera_sidecars(
                    output,
                    camera_intrinsics,
                    include_distortion_in_cam_k=(
                        bool(camera_intrinsics.distortion)
                        and _opencv_projection_compatible(
                            camera_intrinsics.distortion_model
                        )
                    ),
                )
                sidecar_paths = {key: path.name for key, path in written.items()}
                sidecars_written = True

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            if inverted:
                depth_image = _rotate_180(depth_image)
                color_image = _rotate_180(color_image)
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
                color_timestamp_ns = _frame_timestamp_ns(color_frame)
                depth_timestamp_ns = _frame_timestamp_ns(aligned_depth_frame)
                frame_stem = (
                    _frame_stem_from_sensor_timestamp_ns(color_timestamp_ns)
                    if color_timestamp_ns is not None
                    else None
                )
                timestamp_metadata = _realsense_timestamp_metadata(
                    color_frame,
                    aligned_depth_frame,
                    rs,
                )
                metadata = write_legacy_rgbd_frame(
                    output,
                    rgb_image=color_image,
                    depth_image=depth_image,
                    sensor_type=SensorType.REALSENSE_D435,
                    sensor_id=resolved_serial,
                    frame_index=captured_frames,
                    sensor_timestamp_ns=color_timestamp_ns,
                    depth_sensor_timestamp_ns=depth_timestamp_ns,
                    host_received_timestamp_ns=host_received_timestamp_ns,
                    host_wall_timestamp_ns=host_wall_timestamp_ns,
                    frame_stem=frame_stem,
                    extra_metadata={
                        "color_frame_number": color_frame.get_frame_number(),
                        "depth_frame_number": aligned_depth_frame.get_frame_number(),
                        "inverted": bool(inverted),
                        "image_rotation_degrees": image_rotation_degrees,
                        "orientation": orientation,
                        "product_line": product_line,
                        "product_id": product_id or None,
                        **timestamp_metadata,
                        **(
                            {
                                "capture_group_id": hardware_sync["group_id"],
                                "hardware_sync_role": hardware_sync["role"],
                                "hardware_sync_scope": hardware_sync["scope"],
                                "hardware_sync_transport": (
                                    REALSENSE_HARDWARE_SYNC_TRANSPORT
                                ),
                                "inter_cam_sync_mode_configured": (
                                    inter_cam_sync_mode_configured
                                ),
                                "inter_cam_sync_mode_readback": (
                                    inter_cam_sync_mode_readback
                                ),
                            }
                            if hardware_sync is not None
                            else {}
                        ),
                        "global_time_enabled_evidence": (global_time_enabled_evidence),
                        "inter_cam_sync_reset_evidence": (
                            inter_cam_sync_reset_evidence
                        ),
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
            try:
                if record and output is not None:
                    sync_frame_metadata(output)
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
        "product_id": product_id or None,
        "output_path": output.as_posix() if output is not None else None,
        "record": record,
        "preview": preview,
        "inverted": bool(inverted),
        "image_rotation_degrees": image_rotation_degrees,
        "orientation": orientation,
        "hardware_sync_enabled": hardware_sync is not None,
        "hardware_sync_transport": (
            REALSENSE_HARDWARE_SYNC_TRANSPORT if hardware_sync is not None else None
        ),
        "capture_group_id": (
            hardware_sync["group_id"] if hardware_sync is not None else None
        ),
        "hardware_sync_role": (
            hardware_sync["role"] if hardware_sync is not None else None
        ),
        "hardware_sync_scope": (
            hardware_sync["scope"] if hardware_sync is not None else None
        ),
        "hardware_sync_rgb_exposure_claimed": False,
        "inter_cam_sync_mode_configured": inter_cam_sync_mode_configured,
        "inter_cam_sync_mode_readback": inter_cam_sync_mode_readback,
        "inter_cam_sync_reset_evidence": inter_cam_sync_reset_evidence,
        "global_time_enabled_evidence": global_time_enabled_evidence,
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
