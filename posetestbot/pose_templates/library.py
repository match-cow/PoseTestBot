"""Exact full-pose previews and immutable pose-template bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from posetestbot.io.atomic import atomic_write_bytes, atomic_write_json
from posetestbot.pose_templates.adapter import (
    ADAPTER_VERSION,
    POSETEMPLATECREATOR_REVISION,
    load_posetemplatecreator_backend,
)
from posetestbot.pose_templates.catalog import (
    _sha256,
    default_working_data_root,
    get_catalog_object,
    load_catalog,
    utc_now_iso,
)
from posetestbot.pose_templates.transforms import matrix_from_xyz_rpy, transform_record


BUNDLE_SCHEMA_VERSION = "pose_template_bundle.v1"
PREVIEW_SCHEMA_VERSION = "pose_template_preview.v1"
LIBRARY_DIRECTORY = "pose_templates"
BUNDLE_MANIFEST = "pose_template_bundle.json"
TEMPLATE_PDF = "pose_template.pdf"
PREVIEW_JSON = "pose_template_preview.json"
ARCHIVE_STATE = "archive_state.json"
MAX_INSTANCES = 20
_LOCK = threading.RLock()


def default_template_library_root() -> Path:
    return default_working_data_root() / LIBRARY_DIRECTORY


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _uuid(value: Any, *, label: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label} must be a UUID") from exc


def _normalize_configuration(value: Mapping[str, Any]) -> dict[str, Any]:
    name = str(value.get("display_name", "")).strip()
    if not name or len(name) > 120:
        raise ValueError("Template display_name must contain 1 to 120 characters")
    description = value.get("description")
    if description is not None and len(str(description)) > 2000:
        raise ValueError("Template description must not exceed 2000 characters")
    page = value.get("page", {})
    if not isinstance(page, Mapping):
        raise ValueError("Template page must be an object")
    size = str(page.get("size", "A3"))
    orientation = str(page.get("orientation", "landscape"))
    if size not in {"A0", "A1", "A2", "A3", "A4"}:
        raise ValueError("Template page size must be A0, A1, A2, A3, or A4")
    if orientation not in {"portrait", "landscape"}:
        raise ValueError("Template orientation must be portrait or landscape")
    compensation = value.get("print_compensation", {})
    if not isinstance(compensation, Mapping):
        raise ValueError("print_compensation must be an object")
    scale_x = float(compensation.get("x_scale", 1.0))
    scale_y = float(compensation.get("y_scale", 1.0))
    if not np.isfinite([scale_x, scale_y]).all() or not (0.5 <= scale_x <= 1.5) or not (
        0.5 <= scale_y <= 1.5
    ):
        raise ValueError("Print compensation factors must be finite and between 0.5 and 1.5")
    instances = value.get("instances", [])
    if not isinstance(instances, list) or not instances:
        raise ValueError("Template must contain at least one object instance")
    if len(instances) > MAX_INSTANCES:
        raise ValueError(f"Template may contain at most {MAX_INSTANCES} instances")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(instances):
        if not isinstance(item, Mapping):
            raise ValueError(f"Template instance {index} must be an object")
        if not item.get("instance_uuid"):
            raise ValueError(f"Template instance {index} requires an immutable instance_uuid")
        instance_uuid = _uuid(item.get("instance_uuid"), label="instance_uuid")
        if instance_uuid in seen:
            raise ValueError("Template instance UUIDs must be unique")
        seen.add(instance_uuid)
        catalog_uuid = _uuid(item.get("catalog_uuid"), label="catalog_uuid")
        pose = item.get("pose", {})
        if not isinstance(pose, Mapping):
            raise ValueError(f"Template instance {index} pose must be an object")
        pose_value = {
            key: float(pose.get(key, 0.0))
            for key in ("x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg")
        }
        if not np.isfinite(list(pose_value.values())).all():
            raise ValueError(f"Template instance {index} pose must be finite")
        normalized.append(
            {
                "instance_uuid": instance_uuid,
                "catalog_uuid": catalog_uuid,
                "pose": pose_value,
            }
        )
    normalized.sort(key=lambda item: item["instance_uuid"])
    return {
        "display_name": name,
        "description": str(description).strip() if description else None,
        "page": {
            "size": size,
            "orientation": orientation,
            "origin_from_lower_left_mm": [15.0, 15.0],
            "page_static_elements_scaled": False,
        },
        "print_compensation": {"x_scale": scale_x, "y_scale": scale_y},
        "instances": normalized,
    }


def build_template_preview(
    configuration: Mapping[str, Any], *, catalog_root: str | Path | None = None
) -> dict[str, Any]:
    """Build exact full-pose slice geometry without committing a template."""
    config = _normalize_configuration(configuration)
    catalog = load_catalog(catalog_root)
    records = {item["catalog_uuid"]: item for item in catalog["objects"]}
    root = Path(catalog["catalog_root"])
    backend = load_posetemplatecreator_backend()
    scale_x = config["print_compensation"]["x_scale"]
    scale_y = config["print_compensation"]["y_scale"]
    preview_instances: list[dict[str, Any]] = []
    layout_objects: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for item in config["instances"]:
        record = records.get(item["catalog_uuid"])
        if record is None:
            raise ValueError(f"Unknown catalog object: {item['catalog_uuid']}")
        if record["state"] != "active":
            raise ValueError(f"Archived object cannot be added: {record['name']}")
        canonical_path = root / record["assets"]["canonical_ply"]["path"]
        pose = item["pose"]
        matrix = matrix_from_xyz_rpy(**pose)
        try:
            nominal = backend.posed_contours("canonical.ply", canonical_path.read_bytes(), matrix)
        except Exception as exc:
            detail = getattr(exc, "message", str(exc))
            code = getattr(exc, "code", "invalid_intersection")
            errors.append(
                {"instance_uuid": item["instance_uuid"], "code": code, "message": detail}
            )
            continue
        compensated = [
            [
                {"x_mm": point["x_mm"] * scale_x, "y_mm": point["y_mm"] * scale_y}
                for point in contour
            ]
            for contour in nominal
        ]
        instance = {
            **item,
            "catalog": {
                "catalog_uuid": record["catalog_uuid"],
                "obj_id": record["obj_id"],
                "name": record["name"],
                "canonical_ply_sha256": record["canonical_ply_sha256"],
                "texture_sha256": record.get("texture_sha256"),
            },
            "pose_template_from_object": transform_record(
                matrix, parent="pose_template", child=f"object:{item['instance_uuid']}"
            ),
            "nominal_contours": nominal,
            "compensated_contours": compensated,
            "nominal_geometry_sha256": _hash_json(nominal),
            "compensated_geometry_sha256": _hash_json(compensated),
        }
        preview_instances.append(instance)
        layout_objects.append(
            {
                "id": item["instance_uuid"],
                "name": record["name"],
                "source_filename": "canonical.ply",
                "source_sha256": record["canonical_ply_sha256"],
                "contours": [{"points": contour} for contour in compensated],
                # Geometry already includes the complete rigid pose and compensation.
                "pose": {"x_mm": 0.0, "y_mm": 0.0, "rotation_deg": 0.0},
            }
        )
    if errors:
        return {
            "schema_version": PREVIEW_SCHEMA_VERSION,
            "valid": False,
            "configuration": config,
            "configuration_sha256": _hash_json(config),
            "instances": preview_instances,
            "errors": errors,
            "source": {
                "revision": POSETEMPLATECREATOR_REVISION,
                "adapter_version": ADAPTER_VERSION,
            },
        }
    request = {
        "schema_version": "2.0",
        "template_name": config["display_name"],
        "paper_size": config["page"]["size"],
        "orientation": config["page"]["orientation"],
        "objects": layout_objects,
    }
    scene = backend.build_scene(request)
    validation = backend.scene.validation_from_scene(scene).model_dump(mode="json")
    issues = [
        {
            "instance_uuid": str(issue["object_id"]),
            "code": issue["code"],
            "message": issue["message"],
            "bounds": issue["bounds"],
        }
        for issue in validation["issues"]
    ]
    return {
        "schema_version": PREVIEW_SCHEMA_VERSION,
        "valid": bool(scene.valid),
        "configuration": config,
        "configuration_sha256": _hash_json(config),
        "page": validation["page"],
        "instances": preview_instances,
        "fit": {
            "valid": bool(scene.valid),
            "objects": validation["objects"],
            "issues": issues,
        },
        "errors": issues,
        "print_geometry_sha256": _hash_json(
            [item["compensated_contours"] for item in preview_instances]
        ),
        "nominal_geometry_sha256": _hash_json(
            [item["nominal_contours"] for item in preview_instances]
        ),
        "source": {
            "revision": POSETEMPLATECREATOR_REVISION,
            "adapter_version": ADAPTER_VERSION,
        },
        "_layout_request": request,
    }


def generate_template_bundle(
    configuration: Mapping[str, Any],
    *,
    catalog_root: str | Path | None = None,
    library_root: str | Path | None = None,
    template_uuid: str | None = None,
    cloned_from: str | None = None,
) -> dict[str, Any]:
    preview = build_template_preview(configuration, catalog_root=catalog_root)
    if not preview["valid"]:
        raise ValueError("Pose template is invalid: " + "; ".join(item["message"] for item in preview["errors"]))
    opaque_id = _uuid(template_uuid or uuid.uuid4(), label="template_uuid")
    library = Path(library_root or default_template_library_root())
    destination = library / opaque_id
    if destination.exists():
        raise ValueError(f"Pose template already exists: {opaque_id}")
    stage = library / f".{opaque_id}.{uuid.uuid4().hex}.tmp"
    stage.mkdir(parents=True, exist_ok=False)
    try:
        catalog = load_catalog(catalog_root)
        catalog_base = Path(catalog["catalog_root"])
        snapshot_root = stage / "assets"
        file_records: dict[str, Any] = {}
        snapshot_instances: list[dict[str, Any]] = []
        for item in preview["instances"]:
            record = get_catalog_object(item["catalog_uuid"], catalog_root=catalog_base)
            instance_dir = snapshot_root / item["instance_uuid"]
            instance_dir.mkdir(parents=True)
            canonical_source = catalog_base / record["assets"]["canonical_ply"]["path"]
            canonical_target = instance_dir / "canonical.ply"
            shutil.copyfile(canonical_source, canonical_target)
            files = {
                "canonical_ply": {
                    "path": canonical_target.relative_to(stage).as_posix(),
                    "sha256": _sha256(canonical_target),
                    "size_bytes": canonical_target.stat().st_size,
                }
            }
            texture_record = record["assets"].get("texture")
            if texture_record:
                texture_target = instance_dir / "texture.png"
                shutil.copyfile(catalog_base / texture_record["path"], texture_target)
                files["texture"] = {
                    "path": texture_target.relative_to(stage).as_posix(),
                    "sha256": _sha256(texture_target),
                    "size_bytes": texture_target.stat().st_size,
                }
            file_records[item["instance_uuid"]] = files
            snapshot_instances.append({**item, "assets": files})
        serializable_preview = {key: value for key, value in preview.items() if not key.startswith("_")}
        atomic_write_json(stage / PREVIEW_JSON, serializable_preview)
        backend = load_posetemplatecreator_backend()
        scene = backend.build_scene(preview["_layout_request"])
        atomic_write_bytes(stage / TEMPLATE_PDF, backend.render_pdf(scene))
        created = utc_now_iso()
        catalog_snapshot = {
            "schema_version": catalog["schema_version"],
            "version": catalog["version"],
            "objects": [item["catalog"] for item in snapshot_instances],
        }
        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "template_uuid": opaque_id,
            "display_name": preview["configuration"]["display_name"],
            "description": preview["configuration"]["description"],
            "created_at": created,
            "updated_at": created,
            "cloned_from": _uuid(cloned_from, label="cloned_from") if cloned_from else None,
            "page": preview["configuration"]["page"],
            "layout": {
                "nominal": {"units": "mm", "frame": "pose_template"},
                "compensated": preview["configuration"]["print_compensation"],
            },
            "print_compensation": preview["configuration"]["print_compensation"],
            "instances": snapshot_instances,
            "catalog_snapshot": catalog_snapshot,
            "configuration": preview["configuration"],
            "hashes": {
                "catalog": _hash_json(catalog_snapshot),
                "configuration": preview["configuration_sha256"],
                "nominal_geometry": preview["nominal_geometry_sha256"],
                "compensated_geometry": preview["print_geometry_sha256"],
                "pdf": _sha256(stage / TEMPLATE_PDF),
                "preview": _sha256(stage / PREVIEW_JSON),
                "assets": _hash_json(file_records),
            },
            "files": {
                "pdf": TEMPLATE_PDF,
                "preview": PREVIEW_JSON,
                "assets": file_records,
            },
            "source": {
                "name": "PoseTemplateCreator",
                "revision": POSETEMPLATECREATOR_REVISION,
                "adapter_version": ADAPTER_VERSION,
            },
        }
        manifest["bundle_sha256"] = _hash_json(manifest)
        atomic_write_json(stage / BUNDLE_MANIFEST, manifest)
        atomic_write_json(
            stage / ARCHIVE_STATE,
            {"schema_version": "pose_template_archive_state.v1", "state": "active", "updated_at": created},
        )
        validate_template_bundle(stage, library_root=library, allow_staging=True)
        library.mkdir(parents=True, exist_ok=True)
        os.replace(stage, destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return validate_template_bundle(destination, library_root=library)


def validate_template_bundle(
    bundle_path: str | Path,
    *,
    library_root: str | Path | None = None,
    allow_staging: bool = False,
) -> dict[str, Any]:
    bundle_dir = Path(bundle_path)
    library = Path(library_root or bundle_dir.parent).resolve()
    resolved = bundle_dir.resolve()
    try:
        resolved.relative_to(library)
    except ValueError as exc:
        raise ValueError("Pose-template bundle escapes library") from exc
    if bundle_dir.is_symlink() or not bundle_dir.is_dir():
        raise FileNotFoundError(f"Pose-template bundle does not exist: {bundle_dir}")
    with open(bundle_dir / BUNDLE_MANIFEST, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"Bundle schema must be {BUNDLE_SCHEMA_VERSION}")
    template_uuid = _uuid(manifest.get("template_uuid"), label="template_uuid")
    if not allow_staging and bundle_dir.name != template_uuid:
        raise ValueError("Bundle directory does not match template_uuid")
    expected_bundle_hash = manifest.get("bundle_sha256")
    unhashed = {key: value for key, value in manifest.items() if key != "bundle_sha256"}
    if _hash_json(unhashed) != expected_bundle_hash:
        raise ValueError("Pose-template bundle manifest hash mismatch")
    if manifest.get("source", {}).get("revision") != POSETEMPLATECREATOR_REVISION:
        raise ValueError("Pose-template bundle has the wrong upstream revision")
    files = manifest.get("files", {})
    for name, digest_key in (("pdf", "pdf"), ("preview", "preview")):
        relative = Path(str(files.get(name, "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Bundle file path must be relative")
        path = bundle_dir / relative
        if path.is_symlink() or not path.is_file() or _sha256(path) != manifest["hashes"][digest_key]:
            raise ValueError(f"Bundle {name} is missing or was modified")
    for instance_files in files.get("assets", {}).values():
        for record in instance_files.values():
            relative = Path(record["path"])
            path = bundle_dir / relative
            if relative.is_absolute() or ".." in relative.parts or path.is_symlink():
                raise ValueError("Bundle asset path is unsafe")
            if not path.is_file() or path.stat().st_size != int(record["size_bytes"]):
                raise ValueError("Bundle asset is missing or has the wrong size")
            if _sha256(path) != record["sha256"]:
                raise ValueError("Bundle asset hash mismatch")
    with open(bundle_dir / ARCHIVE_STATE, "r", encoding="utf-8") as handle:
        archive = json.load(handle)
    if archive.get("state") not in {"active", "archived"}:
        raise ValueError("Pose-template archive state is invalid")
    return {**manifest, "archive": archive, "bundle_path": resolved.as_posix()}


def list_template_bundles(library_root: str | Path | None = None) -> list[dict[str, Any]]:
    library = Path(library_root or default_template_library_root())
    if not library.is_dir():
        return []
    bundles = []
    for child in sorted(library.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            try:
                bundles.append(validate_template_bundle(child, library_root=library))
            except (OSError, ValueError):
                continue
    return bundles


def set_template_archive_state(
    template_uuid: str, *, state: str, library_root: str | Path | None = None
) -> dict[str, Any]:
    if state not in {"active", "archived"}:
        raise ValueError("Template state must be active or archived")
    library = Path(library_root or default_template_library_root())
    bundle = validate_template_bundle(library / _uuid(template_uuid, label="template_uuid"), library_root=library)
    with _LOCK:
        atomic_write_json(
            Path(bundle["bundle_path"]) / ARCHIVE_STATE,
            {"schema_version": "pose_template_archive_state.v1", "state": state, "updated_at": utc_now_iso()},
        )
    return validate_template_bundle(bundle["bundle_path"], library_root=library)


def clone_template_configuration(
    template_uuid: str, *, library_root: str | Path | None = None
) -> dict[str, Any]:
    library = Path(library_root or default_template_library_root())
    bundle = validate_template_bundle(library / _uuid(template_uuid, label="template_uuid"), library_root=library)
    configuration = json.loads(json.dumps(bundle["configuration"]))
    configuration["display_name"] = f"{configuration['display_name']} (copy)"
    for item in configuration["instances"]:
        item["instance_uuid"] = str(uuid.uuid4())
    return configuration
