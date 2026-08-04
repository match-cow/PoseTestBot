#!/usr/bin/env python3
"""Stream one sensor RGB preview into a rolling JPEG file."""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path
from typing import Any, Mapping

import cv2

from posetestbot.io.atomic import atomic_write_json
from posetestbot.io.manifest import utc_now_iso
from posetestbot.pipeline.run_config import normalize_inverted, normalize_sensor_type
from posetestbot.sensors.oak_d_pro import OAKDProPreviewStream
from posetestbot.sensors.previews import (
    PREVIEW_IMAGE_NAME,
    PREVIEW_STATUS_NAME,
    PREVIEW_STOP_NAME,
    PREVIEW_STATUS_SCHEMA,
)
from posetestbot.sensors.v4l2_preview import (
    open_v4l2_capture,
    select_realsense_rgb_node,
)


_STOP_REQUESTED = False


def _handle_signal(_signum, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream sensor RGB preview JPEGs.")
    parser.add_argument("preview_root", help="Output folder for this preview stream.")
    parser.add_argument(
        "--sensor-json",
        required=True,
        help="JSON object with sensor_type, device_id, display_name, and metadata.",
    )
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    return parser.parse_args()


def _load_sensor_spec(value: str) -> dict[str, Any]:
    loaded = json.loads(value)
    if not isinstance(loaded, Mapping):
        raise ValueError("--sensor-json must be a JSON object")
    raw_sensor_type = str(loaded.get("sensor_type", "")).strip()
    sensor_type = normalize_sensor_type(raw_sensor_type).value
    device_id = str(loaded.get("device_id", "")).strip()
    if not device_id:
        raise ValueError("sensor_json device_id must not be empty")
    spec = dict(loaded)
    spec["sensor_type"] = sensor_type
    spec["device_id"] = device_id
    spec["inverted"] = normalize_inverted(spec.get("inverted", False))
    return spec


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_json(path, value)


def _atomic_jpeg(path: Path, frame, *, jpeg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.jpg")
    ok = cv2.imwrite(
        tmp.as_posix(),
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise OSError(f"Could not write preview JPEG: {path}")
    tmp.replace(path)


def _normalize_frame(frame):
    if frame is None:
        return None
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if len(frame.shape) == 3 and frame.shape[2] == 2:
        return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def _base_status(preview_root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    sensor_type = str(spec["sensor_type"])
    device_id = str(spec["device_id"])
    return {
        "schema_version": PREVIEW_STATUS_SCHEMA,
        "generated_at": utc_now_iso(),
        "preview_root": preview_root.as_posix(),
        "sensor_key": f"{sensor_type}:{device_id}",
        "sensor_type": sensor_type,
        "device_id": device_id,
        "display_name": spec.get("display_name"),
        "alias": spec.get("alias"),
        "effective_display_name": spec.get("effective_display_name")
        or spec.get("alias")
        or spec.get("display_name"),
        "mounting_mode": spec.get("mounting_mode"),
        "inverted": bool(spec.get("inverted", False)),
        "metadata": spec.get("metadata", {}),
        "status": "starting",
        "frame_count": 0,
        "heartbeat_at": utc_now_iso(),
        "last_frame_at": None,
        "latest_image": None,
        "selected_node": None,
        "error": None,
    }


def _open_capture(
    path: str,
    *,
    width: int,
    height: int,
    fps: int,
    pixel_format: str = "YUYV",
):
    return open_v4l2_capture(
        path,
        width=width,
        height=height,
        fps=fps,
        pixel_format=pixel_format,
    )


def run_preview(args: argparse.Namespace) -> int:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    preview_root = Path(args.preview_root)
    preview_root.mkdir(parents=True, exist_ok=True)
    status_path = preview_root / PREVIEW_STATUS_NAME
    image_path = preview_root / PREVIEW_IMAGE_NAME
    stop_path = preview_root / PREVIEW_STOP_NAME
    spec = _load_sensor_spec(args.sensor_json)
    status = _base_status(preview_root, spec)
    _atomic_json(status_path, status)

    oak_stream: OAKDProPreviewStream | None = None
    capture = None
    if spec["sensor_type"] == "realsense_d435":
        selection = select_realsense_rgb_node(
            str(spec["device_id"]),
            metadata=(
                spec.get("metadata")
                if isinstance(spec.get("metadata"), Mapping)
                else None
            ),
        )
        pixel_format = "YUYV"
        selected_source = selection.as_dict()
    elif spec["sensor_type"] == "oak_d_pro":
        oak_stream = OAKDProPreviewStream(
            device_id=str(spec["device_id"]),
            fps=max(1, int(args.fps)),
            width=max(1, int(args.width)),
            height=max(1, int(args.height)),
        )
        selected_source = oak_stream.selected_source
    else:
        status.update(
            {
                "status": "failed",
                "generated_at": utc_now_iso(),
                "error": f"Live RGB preview is not implemented for {spec['sensor_type']}.",
            }
        )
        _atomic_json(status_path, status)
        return 2
    try:
        status.update(
            {
                "status": "opening",
                "generated_at": utc_now_iso(),
                "heartbeat_at": utc_now_iso(),
                "selected_node": selected_source,
            }
        )
        _atomic_json(status_path, status)
        print(
            f"Streaming {status['sensor_key']} RGB preview from "
            f"{selected_source.get('path') or selected_source.get('device_id')}",
            flush=True,
        )

        if oak_stream is None:
            capture = _open_capture(
                selection.path,
                width=max(1, int(args.width)),
                height=max(1, int(args.height)),
                fps=max(1, int(args.fps)),
                pixel_format=pixel_format,
            )
    except BaseException:
        if capture is not None:
            capture.release()
        if oak_stream is not None:
            oak_stream.close()
        raise
    frame_count = 0
    interval_s = 1.0 / max(1, int(args.fps))
    opened_at = time.monotonic()
    last_heartbeat = 0.0
    last_frame_at = opened_at
    try:
        while not _STOP_REQUESTED and not stop_path.exists():
            start = time.monotonic()
            if oak_stream is not None:
                frame = oak_stream.try_get_frame()
                ok = frame is not None
            else:
                assert capture is not None
                ok, frame = capture.read()
            if not ok or frame is None:
                if time.monotonic() - last_frame_at > 5.0:
                    raise RuntimeError(
                        f"No RGB frames received from "
                        f"{selected_source.get('path') or selected_source.get('device_id')}."
                    )
                if time.monotonic() - last_heartbeat >= 1.0:
                    last_heartbeat = time.monotonic()
                    status.update(
                        generated_at=utc_now_iso(),
                        heartbeat_at=utc_now_iso(),
                    )
                    _atomic_json(status_path, status)
                time.sleep(min(0.05 if oak_stream is not None else 0.2, interval_s))
                continue
            last_frame_at = time.monotonic()
            frame = _normalize_frame(frame)
            if frame is None:
                continue
            if bool(spec.get("inverted", False)):
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            _atomic_jpeg(
                image_path,
                frame,
                jpeg_quality=min(100, max(1, int(args.jpeg_quality))),
            )
            frame_count += 1
            status.update(
                {
                    "status": "running",
                    "generated_at": utc_now_iso(),
                    "heartbeat_at": utc_now_iso(),
                    "last_frame_at": utc_now_iso(),
                    "frame_count": frame_count,
                    "latest_image": PREVIEW_IMAGE_NAME,
                    "error": None,
                }
            )
            _atomic_json(status_path, status)
            elapsed = time.monotonic() - start
            if elapsed < interval_s:
                time.sleep(interval_s - elapsed)
    finally:
        if capture is not None:
            capture.release()
        if oak_stream is not None:
            oak_stream.close()

    status.update(
        {
            "status": "stopped",
            "generated_at": utc_now_iso(),
            "heartbeat_at": utc_now_iso(),
            "frame_count": frame_count,
            "latest_image": PREVIEW_IMAGE_NAME if image_path.is_file() else None,
        }
    )
    _atomic_json(status_path, status)
    return 0


def main() -> int:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    args = parse_args()
    try:
        return run_preview(args)
    except Exception as exc:
        preview_root = Path(args.preview_root)
        preview_root.mkdir(parents=True, exist_ok=True)
        try:
            spec = _load_sensor_spec(args.sensor_json)
            status = _base_status(preview_root, spec)
        except Exception:
            status = {
                "schema_version": PREVIEW_STATUS_SCHEMA,
                "generated_at": utc_now_iso(),
                "preview_root": preview_root.as_posix(),
                "status": "failed",
            }
        status.update(
            {
                "status": "failed",
                "generated_at": utc_now_iso(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _atomic_json(preview_root / PREVIEW_STATUS_NAME, status)
        print(status["error"], flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
