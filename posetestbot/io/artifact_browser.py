"""Acquisition-focused run artifact discovery and safe previews."""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from posetestbot.io.artifacts import (
    ARUCO_COVERAGE_REPORT,
    BLENDERPROC_RENDER_PLAN,
    BOP_COCO_ANNOTATIONS,
    BOP_DATASET_INFO,
    BOP_DIR,
    BOP_EXPORT_MANIFEST,
    BOP_FRAME_MAP_JSON,
    BOP_MULTIVIEW_TARGETS,
    BOP_TARGETS_BOP19,
    CALIBRATION_CANDIDATES,
    CALIBRATION_TARGET,
    CALIBRATION_OBSERVATIONS,
    CALIBRATION_PREFLIGHT_REPORT,
    CALIBRATION_PROFILES,
    CALIBRATION_PROFILES_FROM_OBSERVATIONS,
    CALIBRATION_PROFILES_SOLVED,
    CALIBRATION_SOLVER_REPORT,
    CALIBRATION_VALIDATION_REPORT,
    CAMERA_RECTIFICATION_REPORT,
    CAPTURE_EXECUTION_LOGS_DIR,
    CAPTURE_EXECUTION_PLAN,
    CAPTURE_EXECUTION_REPORT,
    CAPTURE_EXECUTION_STATUS,
    CAPTURE_PLAN,
    CAPTURE_PLAN_PREFLIGHT_REPORT,
    CAPTURE_REHEARSAL_REPORT,
    DATASET_MANIFEST,
    DEPTH_DIR,
    HARDWARE_STATUS_REPORT,
    INTRINSIC_CALIBRATION_PROFILES,
    MODELS_DIR,
    PIPELINE_SEQUENCE_PLAN,
    REALSENSE_CAPTURE_SMOKE_REPORT,
    RGB_DIR,
    REWRITE_GATE_REPORT,
    REWRITE_STATUS_REPORT,
    RUN_CONFIG,
    RUN_PREFLIGHT_REPORT,
    SYNC_QUALITY_REPORT,
)
from posetestbot.io.manifest import load_run_manifest
from posetestbot.pipeline.preflight import run_preflight_queue_summary
from posetestbot.pipeline.run_config import validate_run_config


TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


class ArtifactPathError(ValueError):
    """Raised when an artifact path is invalid or outside the run root."""


@dataclass(frozen=True)
class ArtifactRecord:
    key: str
    source: str
    path: str
    relative_path: str | None
    kind: str
    exists: bool
    preview_type: str
    size_bytes: int | None = None
    modified_at: str | None = None
    child_count: int | None = None
    summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["display_label"] = _artifact_display_label(self)
        return payload


def _artifact_display_label(record: ArtifactRecord) -> str:
    state = "ok" if record.exists else "missing"
    bits = [record.key, record.source, state]
    if isinstance(record.summary, Mapping):
        summary_type = record.summary.get("type")
        if summary_type:
            bits.append(str(summary_type))
    return " | ".join(bits)


def _run_root(run_root: str | Path) -> Path:
    return Path(run_root).resolve()


def _relative_to(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ArtifactPathError(f"Artifact path is outside run root: {path}") from exc


def resolve_artifact_path(run_root: str | Path, artifact_path: str | Path) -> Path:
    root = _run_root(run_root)
    path = Path(artifact_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    _relative_to(path, root)
    return path


def _safe_json(path: Path) -> object | None:
    if not path.is_file():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _preview_type(path: Path) -> str:
    if path.is_dir():
        return "directory"
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in TEXT_SUFFIXES:
        return "text"
    return "binary"


def _kind(path: Path) -> str:
    if path.is_dir():
        return "directory"
    return "file"


def _modified_at(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()


def _image_summary(path: Path) -> dict[str, Any] | None:
    image = cv2.imread(path.as_posix(), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    return {
        "type": "image",
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
        "dtype": str(image.dtype),
    }


def _status_json_summary(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": path.stem,
        "schema_version": value.get("schema_version"),
        "status": value.get("overall_status", value.get("status")),
    }


def _aruco_coverage_report_summary(
    path: Path,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _status_json_summary(path, value)
    status = summary["status"]
    sensors = value.get("sensors")
    summary.update(
        {
            "type": "aruco_coverage_report",
            "ready_for_calibration": status not in {"error", "failed", "blocked"},
            "blocker": None
            if status not in {"error", "failed", "blocked"}
            else "failed_aruco_coverage_report",
            "sensor_names": [
                str(sensor["sensor_name"])
                for sensor in sensors
                if isinstance(sensor, Mapping) and isinstance(sensor.get("sensor_name"), str)
            ]
            if isinstance(sensors, list)
            else [],
            "valid_pose_count": value.get("valid_pose_count"),
            "frame_count": value.get("frame_count"),
            "valid_pose_ratio": value.get("valid_pose_ratio"),
        }
    )
    return summary


def _run_config_summary(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    ready = True
    error = None
    try:
        validate_run_config(value)
    except Exception as exc:
        ready = False
        error = str(exc)
    summary: dict[str, Any] = {
        "type": "run_config",
        "schema_version": value.get("schema_version"),
        "ready_for_pipeline": ready,
        "sequence_id": (
            value.get("pipeline", {}).get("sequence_id")
            if isinstance(value.get("pipeline"), Mapping)
            else None
        ),
        "plan_only": (
            value.get("pipeline", {}).get("plan_only")
            if isinstance(value.get("pipeline"), Mapping)
            else None
        ),
    }
    if error:
        summary["error"] = error
    return summary


def _bop_export_summary(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    exports = value.get("exports")
    object_models = value.get("object_models")
    return {
        "type": "bop_export_manifest",
        "schema_version": value.get("schema_version"),
        "export_count": len(exports) if isinstance(exports, list) else 0,
        "model_count": len(object_models) if isinstance(object_models, list) else 0,
        "targets_path": value.get("targets_path"),
    }


def _bop_targets_summary(value: object) -> dict[str, Any]:
    return {
        "type": "bop_targets",
        "target_count": len(value) if isinstance(value, list) else 0,
    }


def _bop_scene_summary(path: Path) -> dict[str, Any] | None:
    scene_camera = _safe_json(path / "scene_camera.json")
    scene_gt = _safe_json(path / "scene_gt.json")
    if not isinstance(scene_camera, Mapping):
        return None
    annotation_count = 0
    if isinstance(scene_gt, Mapping):
        for annotations in scene_gt.values():
            if isinstance(annotations, list):
                annotation_count += len(annotations)
    return {
        "type": "bop_scene",
        "image_count": len(scene_camera),
        "rgb_count": len(list((path / RGB_DIR).glob("*.png")))
        if (path / RGB_DIR).is_dir()
        else 0,
        "depth_count": len(list((path / DEPTH_DIR).glob("*.png")))
        if (path / DEPTH_DIR).is_dir()
        else 0,
        "annotation_count": annotation_count,
        "has_scene_gt_info": (path / "scene_gt_info.json").is_file(),
        "has_mask": (path / "mask").is_dir(),
        "has_mask_visib": (path / "mask_visib").is_dir(),
    }


def _summary_for_path(path: Path, root: Path) -> dict[str, Any] | None:
    if path.is_dir():
        if (path / "scene_camera.json").is_file():
            return _bop_scene_summary(path)
        return None
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return _image_summary(path)
    value = _safe_json(path) if path.suffix.lower() == ".json" else None
    if not isinstance(value, Mapping):
        if path.name == BOP_TARGETS_BOP19:
            return _bop_targets_summary(_safe_json(path))
        return None
    if path.name == RUN_CONFIG:
        summary = _run_config_summary(path, value)
        if summary["ready_for_pipeline"]:
            summary["preflight_queue"] = run_preflight_queue_summary(root, value)
        return summary
    if path.name == BOP_EXPORT_MANIFEST:
        return _bop_export_summary(path, value)
    if path.name == ARUCO_COVERAGE_REPORT:
        return _aruco_coverage_report_summary(path, value)
    if path.name in {
        RUN_PREFLIGHT_REPORT,
        HARDWARE_STATUS_REPORT,
        CAPTURE_PLAN_PREFLIGHT_REPORT,
        CAPTURE_EXECUTION_PLAN,
        CAPTURE_EXECUTION_REPORT,
        CAPTURE_REHEARSAL_REPORT,
        REALSENSE_CAPTURE_SMOKE_REPORT,
        SYNC_QUALITY_REPORT,
        CALIBRATION_PREFLIGHT_REPORT,
        CALIBRATION_OBSERVATIONS,
        CALIBRATION_CANDIDATES,
        CALIBRATION_SOLVER_REPORT,
        CALIBRATION_VALIDATION_REPORT,
        CAMERA_RECTIFICATION_REPORT,
        REWRITE_GATE_REPORT,
        REWRITE_STATUS_REPORT,
    }:
        return _status_json_summary(path, value)
    return {
        "type": path.stem,
        "schema_version": value.get("schema_version"),
    }


def _record(
    *,
    root: Path,
    key: str,
    source: str,
    artifact_path: str | Path,
) -> ArtifactRecord:
    path = Path(artifact_path)
    if not path.is_absolute():
        path = root / path
    exists = path.exists()
    relative_path = _relative_to(path, root) if exists else None
    return ArtifactRecord(
        key=key,
        source=source,
        path=path.as_posix(),
        relative_path=relative_path,
        kind=_kind(path) if exists else "missing",
        exists=exists,
        preview_type=_preview_type(path) if exists else "missing",
        size_bytes=path.stat().st_size if exists and path.is_file() else None,
        modified_at=_modified_at(path),
        child_count=len(list(path.iterdir())) if exists and path.is_dir() else None,
        summary=_summary_for_path(path, root) if exists else None,
    )


def _known_run_artifacts(root: Path) -> Iterable[tuple[str, str, str]]:
    known = [
        DATASET_MANIFEST,
        RUN_CONFIG,
        RUN_PREFLIGHT_REPORT,
        HARDWARE_STATUS_REPORT,
        CAPTURE_PLAN,
        CAPTURE_PLAN_PREFLIGHT_REPORT,
        CAPTURE_EXECUTION_PLAN,
        CAPTURE_EXECUTION_STATUS,
        CAPTURE_EXECUTION_REPORT,
        CAPTURE_EXECUTION_LOGS_DIR,
        CAPTURE_REHEARSAL_REPORT,
        REALSENSE_CAPTURE_SMOKE_REPORT,
        SYNC_QUALITY_REPORT,
        ARUCO_COVERAGE_REPORT,
        CALIBRATION_TARGET,
        INTRINSIC_CALIBRATION_PROFILES,
        CAMERA_RECTIFICATION_REPORT,
        CALIBRATION_PROFILES,
        CALIBRATION_PREFLIGHT_REPORT,
        CALIBRATION_OBSERVATIONS,
        CALIBRATION_CANDIDATES,
        CALIBRATION_PROFILES_FROM_OBSERVATIONS,
        CALIBRATION_SOLVER_REPORT,
        CALIBRATION_PROFILES_SOLVED,
        CALIBRATION_VALIDATION_REPORT,
        BLENDERPROC_RENDER_PLAN,
        PIPELINE_SEQUENCE_PLAN,
        REWRITE_GATE_REPORT,
        REWRITE_STATUS_REPORT,
        f"{BOP_DIR}/{BOP_EXPORT_MANIFEST}",
        f"{BOP_DIR}/{BOP_DATASET_INFO}",
        f"{BOP_DIR}/{BOP_FRAME_MAP_JSON}",
        f"{BOP_DIR}/{MODELS_DIR}/models_info.json",
        f"{BOP_DIR}/{BOP_TARGETS_BOP19}",
        f"{BOP_DIR}/{BOP_MULTIVIEW_TARGETS}",
        f"{BOP_DIR}/{BOP_COCO_ANNOTATIONS}",
    ]
    return [
        (Path(path).name, "known", path)
        for path in known
        if (root / path).exists()
    ]


def _manifest_artifacts(root: Path) -> Iterable[tuple[str, str, str]]:
    try:
        manifest = load_run_manifest(root)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return []
    entries: list[tuple[str, str, str]] = []
    stages = manifest.get("stages")
    if not isinstance(stages, list):
        return entries
    for stage in stages:
        if not isinstance(stage, Mapping):
            continue
        stage_name = str(stage.get("name") or "stage")
        artifacts = stage.get("artifacts")
        if not isinstance(artifacts, Mapping):
            continue
        for key, value in artifacts.items():
            if isinstance(value, str):
                entries.append((str(key), f"manifest:{stage_name}", value))
    return entries


def _bop_export_artifacts(root: Path) -> Iterable[tuple[str, str, str]]:
    manifest = _safe_json(root / BOP_DIR / BOP_EXPORT_MANIFEST)
    if not isinstance(manifest, Mapping):
        return []
    artifact_base = (
        root / BOP_DIR
        if manifest.get("schema_version") == "bop_export_manifest.v2"
        else root
    )

    def artifact_value(value: str) -> str:
        path = Path(value)
        return (
            path.as_posix()
            if path.is_absolute()
            else (artifact_base / path).as_posix()
        )

    entries: list[tuple[str, str, str]] = []
    for key in (
        "targets_path",
        "multiview_targets_path",
        "coco_annotations_path",
        "frame_map_path",
        "dataset_info_path",
    ):
        value = manifest.get(key)
        if isinstance(value, str):
            entries.append((key, "bop_export", artifact_value(value)))
    for export in manifest.get("exports", []):
        if not isinstance(export, Mapping):
            continue
        sensor_name = str(export.get("sensor_name") or "sensor")
        scene_folder = export.get("scene_folder")
        if isinstance(scene_folder, str):
            entries.append(
                (
                    f"{sensor_name}:scene",
                    "bop_export.scene",
                    artifact_value(scene_folder),
                )
            )
        artifacts = export.get("artifacts")
        if isinstance(artifacts, Mapping):
            for key, value in artifacts.items():
                if isinstance(value, str):
                    entries.append(
                        (
                            f"{sensor_name}:{key}",
                            "bop_export.scene",
                            artifact_value(value),
                        )
                    )
    for model in manifest.get("object_models", []):
        if not isinstance(model, Mapping):
            continue
        object_name = str(model.get("object_name") or "object")
        bop_path = model.get("bop_path")
        if isinstance(bop_path, str):
            entries.append(
                (
                    f"{object_name}:model",
                    "bop_export.models",
                    artifact_value(bop_path),
                )
            )
    return entries


def collect_run_artifacts(run_root: str | Path) -> list[ArtifactRecord]:
    """Collect known, manifest-listed, and BOP export artifacts for a run root."""

    root = _run_root(run_root)
    if not root.exists():
        raise FileNotFoundError(f"Run root not found: {root}")

    records: list[ArtifactRecord] = []
    seen: set[tuple[str, str, str]] = set()
    for key, source, artifact_path in (
        *_known_run_artifacts(root),
        *_manifest_artifacts(root),
        *_bop_export_artifacts(root),
    ):
        path = Path(artifact_path)
        full_path = path if path.is_absolute() else root / path
        dedupe_key = (str(key), str(source), full_path.resolve().as_posix())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        records.append(
            _record(
                root=root,
                key=str(key),
                source=str(source),
                artifact_path=artifact_path,
            )
        )
    return sorted(
        records,
        key=lambda record: (
            record.source,
            record.key,
            record.relative_path or record.path,
        ),
    )


def _directory_listing(path: Path, *, limit: int) -> list[dict[str, Any]]:
    children = []
    for child in sorted(path.iterdir(), key=lambda item: item.name)[:limit]:
        stat = child.stat()
        children.append(
            {
                "name": child.name,
                "kind": "directory" if child.is_dir() else "file",
                "size_bytes": stat.st_size if child.is_file() else None,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
            }
        )
    return children


def preview_artifact(
    run_root: str | Path,
    artifact_path: str | Path,
    *,
    text_limit: int = 8000,
    child_limit: int = 200,
) -> dict[str, Any]:
    """Return a safe compact preview for an artifact path under a run root."""

    root = _run_root(run_root)
    path = resolve_artifact_path(root, artifact_path)
    if not path.exists():
        raise ArtifactPathError(f"Artifact path not found: {path}")
    preview_type = _preview_type(path)
    payload: dict[str, Any] = {
        "path": path.as_posix(),
        "relative_path": _relative_to(path, root),
        "preview_type": preview_type,
        "kind": _kind(path),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "summary": _summary_for_path(path, root),
    }
    if path.is_dir():
        payload["children"] = _directory_listing(path, limit=child_limit)
        payload["child_limit"] = child_limit
        return payload
    if preview_type == "image":
        payload["image"] = _image_summary(path)
        with open(path, "rb") as f:
            payload["base64"] = base64.b64encode(f.read()).decode("ascii")
        return payload
    if preview_type == "text":
        text = path.read_text(errors="replace")
        payload["text"] = text[:text_limit]
        payload["truncated"] = len(text) > text_limit
        payload["text_limit"] = text_limit
        return payload
    return payload


def _int_string(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _sorted_image_keys(*mappings: object) -> list[str]:
    keys: set[str] = set()
    for mapping in mappings:
        if isinstance(mapping, Mapping):
            keys.update(str(key) for key in mapping.keys())
    return sorted(keys, key=lambda item: (_int_string(item) is None, _int_string(item) or 0, item))


def _frame_file(root: Path, folder: Path, image_id: int) -> dict[str, Any]:
    path = folder / f"{image_id:06d}.png"
    return {
        "path": path.as_posix(),
        "relative_path": _relative_to(path, root) if path.exists() else None,
        "relative_name": f"{folder.name}/{path.name}",
        "exists": path.is_file(),
        "summary": _image_summary(path) if path.is_file() else None,
    }


def _mask_paths(folder: Path, image_id: int) -> list[Path]:
    if not folder.is_dir():
        return []
    return [
        path
        for path in sorted(folder.glob(f"{image_id:06d}_*.png"))
        if path.is_file()
    ]


def _mask_file_artifacts(root: Path, folder: Path, image_id: int) -> list[dict[str, Any]]:
    artifacts = []
    for path in _mask_paths(folder, image_id):
        artifacts.append(
            {
                "name": path.name,
                "path": path.as_posix(),
                "relative_path": _relative_to(path, root),
                "exists": True,
                "summary": _image_summary(path),
            }
        )
    return artifacts


def _bop_scene_lookup(run_root: Path, *, split: str | None = None) -> dict[int, dict[str, Any]]:
    manifest = _safe_json(run_root / BOP_DIR / BOP_EXPORT_MANIFEST)
    if not isinstance(manifest, Mapping):
        return {}
    scene_base = (
        run_root / BOP_DIR
        if manifest.get("schema_version") == "bop_export_manifest.v2"
        else run_root
    )
    scenes: dict[int, dict[str, Any]] = {}
    for export in manifest.get("exports", []):
        if not isinstance(export, Mapping):
            continue
        export_split = export.get("split")
        if split is not None and export_split != split:
            continue
        scene_id = _int_string(export.get("scene_id"))
        if scene_id is None:
            continue
        scene_folder = export.get("scene_folder")
        scene_info: dict[str, Any] = {
            "scene_id": scene_id,
            "sensor_name": export.get("sensor_name"),
            "split": export_split,
            "scene_folder": scene_folder,
        }
        if isinstance(scene_folder, str):
            scene_path = Path(scene_folder)
            if not scene_path.is_absolute():
                scene_path = scene_base / scene_path
            try:
                scene_info["relative_scene_folder"] = _relative_to(scene_path, run_root)
            except ArtifactPathError:
                scene_info["relative_scene_folder"] = None
        scenes[scene_id] = scene_info
    return scenes


def _scene_info_for_folder(run_root: Path, scene_folder: Path) -> dict[str, Any] | None:
    relative_scene_folder = _relative_to(scene_folder, run_root)
    for scene in _bop_scene_lookup(run_root).values():
        if scene.get("relative_scene_folder") == relative_scene_folder:
            return dict(scene)
    return None


def _frame_map_for_scene(
    run_root: Path, scene: Mapping[str, Any] | None
) -> Mapping[str, Any] | None:
    if scene is None:
        return None
    root_map = _safe_json(run_root / BOP_DIR / BOP_FRAME_MAP_JSON)
    if not isinstance(root_map, Mapping):
        return None
    scenes = root_map.get("scenes")
    if not isinstance(scenes, Mapping):
        return None
    scene_entry = scenes.get(str(scene.get("scene_id")))
    if not isinstance(scene_entry, Mapping):
        return None
    frames = scene_entry.get("frames")
    return frames if isinstance(frames, Mapping) else None


def bop_scene_detail(
    run_root: str | Path,
    scene_path: str | Path,
    *,
    frame_limit: int = 200,
) -> dict[str, Any]:
    """Return a safe frame-by-frame drill-down for one BOP scene folder."""

    if frame_limit < 1:
        raise ValueError("frame_limit must be at least 1")
    root = _run_root(run_root)
    scene_folder = resolve_artifact_path(root, scene_path)
    if not scene_folder.is_dir():
        raise FileNotFoundError(f"BOP scene folder not found: {scene_folder}")

    scene_camera = _safe_json(scene_folder / "scene_camera.json")
    scene_gt = _safe_json(scene_folder / "scene_gt.json")
    scene_gt_info = _safe_json(scene_folder / "scene_gt_info.json")
    scene_info = _scene_info_for_folder(root, scene_folder)
    frame_map = _frame_map_for_scene(root, scene_info)
    if frame_map is None:
        frame_map = _safe_json(scene_folder / BOP_FRAME_MAP_JSON)
    if not isinstance(scene_camera, Mapping):
        raise ValueError(f"Missing or invalid scene_camera.json in {scene_folder}")

    frames = []
    for image_key in _sorted_image_keys(scene_camera, scene_gt, scene_gt_info)[:frame_limit]:
        image_id = _int_string(image_key)
        if image_id is None:
            continue
        gt_annotations = scene_gt.get(image_key) if isinstance(scene_gt, Mapping) else []
        gt_info = scene_gt_info.get(image_key) if isinstance(scene_gt_info, Mapping) else None
        frames.append(
            {
                "image_key": image_key,
                "image_id": image_id,
                "rgb": _frame_file(root, scene_folder / RGB_DIR, image_id),
                "depth": _frame_file(root, scene_folder / DEPTH_DIR, image_id),
                "camera": scene_camera.get(image_key),
                "gt_count": len(gt_annotations) if isinstance(gt_annotations, list) else 0,
                "gt": gt_annotations if isinstance(gt_annotations, list) else None,
                "gt_info": gt_info,
                "mask_files": [path.name for path in _mask_paths(scene_folder / "mask", image_id)],
                "mask_artifacts": _mask_file_artifacts(root, scene_folder / "mask", image_id),
                "mask_visib_files": [
                    path.name for path in _mask_paths(scene_folder / "mask_visib", image_id)
                ],
                "mask_visib_artifacts": _mask_file_artifacts(
                    root,
                    scene_folder / "mask_visib",
                    image_id,
                ),
                "frame_map": frame_map.get(image_key) if isinstance(frame_map, Mapping) else None,
            }
        )
    return {
        "type": "bop_scene_detail",
        "scene_path": scene_folder.as_posix(),
        "relative_path": _relative_to(scene_folder, root),
        "summary": _bop_scene_summary(scene_folder) or {"type": "bop_scene"},
        "frame_count": len(_sorted_image_keys(scene_camera, scene_gt, scene_gt_info)),
        "frame_limit": frame_limit,
        "frames": frames,
        "files": {
            "scene_camera": (scene_folder / "scene_camera.json").is_file(),
            "scene_gt": (scene_folder / "scene_gt.json").is_file(),
            "scene_gt_info": (scene_folder / "scene_gt_info.json").is_file(),
            "frame_map": frame_map is not None,
            "rgb_dir": (scene_folder / RGB_DIR).is_dir(),
            "depth_dir": (scene_folder / DEPTH_DIR).is_dir(),
            "mask_dir": (scene_folder / "mask").is_dir(),
            "mask_visib_dir": (scene_folder / "mask_visib").is_dir(),
        },
    }


def bop_frame_detail(
    run_root: str | Path,
    scene_path: str | Path,
    *,
    image_id: int,
    result_path: str | Path | None = None,
    row_limit: int = 100,
) -> dict[str, Any]:
    """Return one BOP frame bundle for RGB/depth/mask/GT inspection."""

    del result_path, row_limit
    if image_id < 0:
        raise ValueError("image_id must be non-negative")
    root = _run_root(run_root)
    scene_folder = resolve_artifact_path(root, scene_path)
    if not scene_folder.is_dir():
        raise FileNotFoundError(f"BOP scene folder not found: {scene_folder}")
    scene_camera = _safe_json(scene_folder / "scene_camera.json")
    scene_gt = _safe_json(scene_folder / "scene_gt.json")
    scene_gt_info = _safe_json(scene_folder / "scene_gt_info.json")
    scene_info = _scene_info_for_folder(root, scene_folder)
    frame_map = _frame_map_for_scene(root, scene_info)
    if frame_map is None:
        frame_map = _safe_json(scene_folder / BOP_FRAME_MAP_JSON)
    if not isinstance(scene_camera, Mapping):
        raise ValueError(f"Missing or invalid scene_camera.json in {scene_folder}")
    image_key = str(image_id)
    gt_annotations = scene_gt.get(image_key) if isinstance(scene_gt, Mapping) else []
    gt_info = scene_gt_info.get(image_key) if isinstance(scene_gt_info, Mapping) else None
    return {
        "type": "bop_frame_detail",
        "scene_path": scene_folder.as_posix(),
        "relative_path": _relative_to(scene_folder, root),
        "scene": scene_info,
        "image_id": image_id,
        "image_key": image_key,
        "rgb": _frame_file(root, scene_folder / RGB_DIR, image_id),
        "depth": _frame_file(root, scene_folder / DEPTH_DIR, image_id),
        "camera": scene_camera.get(image_key),
        "gt_count": len(gt_annotations) if isinstance(gt_annotations, list) else 0,
        "gt": gt_annotations if isinstance(gt_annotations, list) else None,
        "gt_info": gt_info,
        "mask_artifacts": _mask_file_artifacts(root, scene_folder / "mask", image_id),
        "mask_visib_artifacts": _mask_file_artifacts(
            root,
            scene_folder / "mask_visib",
            image_id,
        ),
        "frame_map": frame_map.get(image_key) if isinstance(frame_map, Mapping) else None,
        "result": None,
    }


def _mask_to_bool(mask: np.ndarray, *, width: int, height: int) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def _apply_mask_overlay(
    image: np.ndarray,
    mask_path: Path,
    *,
    color: tuple[int, int, int],
    alpha: float,
) -> bool:
    mask = cv2.imread(mask_path.as_posix(), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return False
    height, width = image.shape[:2]
    active = _mask_to_bool(mask, width=width, height=height)
    if not np.any(active):
        return False
    overlay = np.zeros_like(image)
    overlay[:, :] = color
    image[active] = cv2.addWeighted(image[active], 1.0 - alpha, overlay[active], alpha, 0)
    return True


def _draw_gt_boxes(image: np.ndarray, gt_info: object, gt: object) -> int:
    rows = gt_info if isinstance(gt_info, list) and gt_info else gt
    if not isinstance(rows, list):
        return 0
    drawn = 0
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        bbox = row.get("bbox_visib", row.get("bbox_obj"))
        if not isinstance(bbox, list | tuple) or len(bbox) != 4:
            continue
        try:
            x, y, w, h = [int(round(float(value))) for value in bbox]
        except (TypeError, ValueError):
            continue
        color = (40, 220, 40)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            image,
            f"GT {index}",
            (x, max(12, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        drawn += 1
    return drawn


def render_bop_frame_overlay_png(
    run_root: str | Path,
    scene_path: str | Path,
    *,
    image_id: int,
    result_path: str | Path | None = None,
    row_limit: int = 20,
    include_masks: bool = True,
    include_gt: bool = True,
    include_results: bool = False,
) -> bytes:
    """Render a PNG overlay of masks and GT boxes on a BOP RGB frame."""

    del result_path, row_limit, include_results
    root = _run_root(run_root)
    scene_folder = resolve_artifact_path(root, scene_path)
    rgb_path = scene_folder / RGB_DIR / f"{image_id:06d}.png"
    if not rgb_path.is_file():
        raise FileNotFoundError(f"BOP RGB frame not found: {rgb_path}")
    image = cv2.imread(rgb_path.as_posix(), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read BOP RGB frame: {rgb_path}")

    if include_masks:
        colors = [
            (255, 80, 80),
            (80, 180, 255),
            (255, 200, 80),
            (200, 80, 255),
        ]
        for index, mask_path in enumerate(_mask_paths(scene_folder / "mask_visib", image_id)):
            _apply_mask_overlay(image, mask_path, color=colors[index % len(colors)], alpha=0.35)
        for index, mask_path in enumerate(_mask_paths(scene_folder / "mask", image_id)):
            _apply_mask_overlay(image, mask_path, color=colors[index % len(colors)], alpha=0.18)

    if include_gt:
        image_key = str(image_id)
        scene_gt = _safe_json(scene_folder / "scene_gt.json")
        scene_gt_info = _safe_json(scene_folder / "scene_gt_info.json")
        gt = scene_gt.get(image_key) if isinstance(scene_gt, Mapping) else None
        gt_info = scene_gt_info.get(image_key) if isinstance(scene_gt_info, Mapping) else None
        _draw_gt_boxes(image, gt_info, gt)

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode BOP frame overlay")
    return bytes(encoded)
