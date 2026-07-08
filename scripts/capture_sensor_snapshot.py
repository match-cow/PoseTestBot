#!/usr/bin/env python3
"""Capture short RGB-D sensor snapshots for the web UI."""

from __future__ import annotations

import argparse
import getpass
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from posetestbot.io.artifacts import DEPTH_DIR, FRAME_METADATA_JSONL, RGB_DIR
from posetestbot.io.manifest import utc_now_iso
from posetestbot.pipeline.run_config import normalize_inverted, normalize_sensor_type
from posetestbot.sensors.registry import build_sensor_capture_command


MANIFEST_NAME = "sensor_snapshot_manifest.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one-frame RGB-D snapshots.")
    parser.add_argument("snapshot_root", help="Output folder for this snapshot batch.")
    parser.add_argument(
        "--sensor-json",
        action="append",
        default=[],
        help="JSON object with sensor_type, device_id, display_name, and UI defaults.",
    )
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--resolution", default="720p")
    parser.add_argument("--max-frames", type=int, default=1)
    return parser.parse_args()


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("._") or "sensor"


def _read_first_metadata(capture_root: Path) -> dict[str, Any] | None:
    path = capture_root / FRAME_METADATA_JSONL
    if not path.is_file():
        return None
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                value = json.loads(line)
                return value if isinstance(value, dict) else None
    return None


def _write_thumbnail(source: Path, target: Path, *, depth: bool = False) -> None:
    image = cv2.imread(source.as_posix(), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"Could not read image: {source}")
    if depth:
        depth_image = image.astype(np.float32)
        nonzero = depth_image[depth_image > 0]
        if nonzero.size:
            low = float(np.percentile(nonzero, 2))
            high = float(np.percentile(nonzero, 98))
            if high <= low:
                high = low + 1.0
            scaled = np.clip((depth_image - low) / (high - low), 0, 1)
        else:
            scaled = np.zeros_like(depth_image, dtype=np.float32)
        image = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    height, width = image.shape[:2]
    scale = min(1.0, 360.0 / max(width, height))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(target.as_posix(), image):
        raise OSError(f"Could not write thumbnail: {target}")


def _load_sensor_specs(values: list[str]) -> list[dict[str, Any]]:
    specs = []
    for value in values:
        loaded = json.loads(value)
        if not isinstance(loaded, Mapping):
            raise ValueError("--sensor-json must be a JSON object")
        sensor_type = normalize_sensor_type(str(loaded.get("sensor_type", ""))).value
        device_id = str(loaded.get("device_id", "")).strip()
        if not device_id:
            raise ValueError("sensor_json device_id must not be empty")
        spec = dict(loaded)
        spec["sensor_type"] = sensor_type
        spec["device_id"] = device_id
        spec["inverted"] = normalize_inverted(spec.get("inverted", False))
        specs.append(spec)
    return specs


def _capture_one(
    *,
    snapshot_root: Path,
    spec: Mapping[str, Any],
    fps: int,
    resolution: str,
    max_frames: int,
) -> dict[str, Any]:
    sensor_type = str(spec["sensor_type"])
    device_id = str(spec["device_id"])
    sensor_key = f"{sensor_type}:{device_id}"
    sensor_dir = snapshot_root / _safe_slug(f"{sensor_type}_{device_id}")
    capture_root = sensor_dir / "capture"
    record: dict[str, Any] = {
        "sensor_key": sensor_key,
        "sensor_type": sensor_type,
        "device_id": device_id,
        "display_name": spec.get("display_name"),
        "alias": spec.get("alias"),
        "effective_display_name": spec.get("effective_display_name")
        or spec.get("alias")
        or spec.get("display_name"),
        "mounting_mode": spec.get("mounting_mode"),
        "inverted": bool(spec.get("inverted", False)),
        "status": "running",
        "capture_root": capture_root.relative_to(snapshot_root).as_posix(),
        "metadata": spec.get("metadata", {}),
        "latest_frame_metadata": None,
        "rgb_thumbnail": None,
        "depth_thumbnail": None,
        "error": None,
        "diagnostics": [],
    }

    spec_metadata = spec.get("metadata", {})
    if not isinstance(spec_metadata, Mapping):
        spec_metadata = {}
    if (
        sensor_type == "realsense_d435"
        and spec_metadata.get("video_nodes")
        and not spec_metadata.get("video_accessible", True)
    ):
        nodes = [
            str(node.get("path"))
            for node in spec_metadata.get("video_nodes", [])
            if isinstance(node, Mapping) and node.get("path")
        ]
        user = getpass.getuser()
        record["status"] = "failed"
        record["error"] = (
            "RealSense preview/capture device nodes are not accessible to "
            f"user {user}: {', '.join(nodes)}."
        )
        record["diagnostics"].append(
            {
                "code": "video_permission_denied",
                "severity": "error",
                "message": record["error"],
                "hints": [
                    f"Add {user} to the video group and start a new login session.",
                    "Install/reload Intel RealSense udev rules so USB and V4L nodes are writable.",
                    "After permissions change, unplug/replug the RealSense cameras or reload udev.",
                ],
            }
        )
        return record

    command = build_sensor_capture_command(
        sensor_type=sensor_type,
        device_id=device_id,
        output_folder=capture_root.as_posix(),
        fps=fps,
        resolution=resolution,
        max_frames=max_frames,
        inverted=bool(spec.get("inverted", False)),
    )
    command.append("--print-json")
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=_repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    record["returncode"] = result.returncode
    if result.returncode != 0:
        record["status"] = "failed"
        output_tail = "\n".join(result.stdout.strip().splitlines()[-3:])
        record["error"] = (
            f"Capture command exited with status {result.returncode}."
            + (f" {output_tail}" if output_tail else "")
        )
        record["diagnostics"].append(
            {
                "code": "capture_command_failed",
                "severity": "error",
                "message": record["error"],
                "hints": [
                    "Check camera permissions and whether another process has the device open.",
                ],
            }
        )
        return record

    metadata = _read_first_metadata(capture_root)
    record["latest_frame_metadata"] = metadata
    if not metadata:
        record["status"] = "failed"
        record["error"] = f"Capture did not write {FRAME_METADATA_JSONL}."
        return record

    try:
        rgb_path = capture_root / str(metadata["rgb_path"])
        depth_path = capture_root / str(metadata["depth_path"])
        rgb_thumbnail = sensor_dir / "rgb_thumbnail.png"
        depth_thumbnail = sensor_dir / "depth_thumbnail.png"
        _write_thumbnail(rgb_path, rgb_thumbnail)
        _write_thumbnail(depth_path, depth_thumbnail, depth=True)
        record["rgb_thumbnail"] = rgb_thumbnail.relative_to(snapshot_root).as_posix()
        record["depth_thumbnail"] = depth_thumbnail.relative_to(snapshot_root).as_posix()
        record["rgb_path"] = rgb_path.relative_to(snapshot_root).as_posix()
        record["depth_path"] = depth_path.relative_to(snapshot_root).as_posix()
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    record["status"] = "succeeded"
    return record


def main() -> int:
    args = parse_args()
    snapshot_root = Path(args.snapshot_root)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    specs = _load_sensor_specs(args.sensor_json)
    if not specs:
        raise SystemExit("No sensors requested.")

    manifest = {
        "schema_version": "sensor_snapshot_manifest.v1",
        "generated_at": utc_now_iso(),
        "snapshot_root": snapshot_root.as_posix(),
        "sensors": [],
    }
    exit_code = 0
    for spec in specs:
        record = _capture_one(
            snapshot_root=snapshot_root,
            spec=spec,
            fps=args.fps,
            resolution=args.resolution,
            max_frames=args.max_frames,
        )
        manifest["sensors"].append(record)
        if record["status"] != "succeeded":
            exit_code = 1

    manifest["status"] = "succeeded" if exit_code == 0 else "failed"
    with open(snapshot_root / MANIFEST_NAME, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {snapshot_root / MANIFEST_NAME}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
