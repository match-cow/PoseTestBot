"""Global transactional, archive-only managed object catalog."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from posetestbot.io.atomic import atomic_write_bytes, atomic_write_json
from posetestbot.pose_templates.adapter import load_posetemplatecreator_backend


SCHEMA_VERSION = "object_catalog.v1"
CATALOG_MANIFEST = "object_catalog.json"
CATALOG_DIRECTORY = "object_catalog"
STAGING_DIRECTORY = "object_catalog_staging"
_LOCK = threading.RLock()


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def default_working_data_root() -> Path:
    configured = os.environ.get("POSETESTBOT_WORKING_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    app_root = os.environ.get("POSETESTBOT_APP_ROOT")
    root = Path(app_root).expanduser().resolve() if app_root else Path(__file__).resolve().parents[2]
    return root / "working_data"


def default_catalog_root() -> Path:
    return default_working_data_root() / CATALOG_DIRECTORY


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _empty_catalog() -> dict[str, Any]:
    now = utc_now_iso()
    return {
        "schema_version": SCHEMA_VERSION,
        "version": 1,
        "created_at": now,
        "updated_at": now,
        "next_obj_id": 1,
        "objects": [],
    }


def _validate_uuid(value: Any, *, label: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{label} must be a UUID") from exc


def _contained(path: Path, root: Path, *, must_exist: bool = True) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve(strict=must_exist)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Catalog asset escapes managed root: {path}") from exc
    if path.is_symlink():
        raise ValueError(f"Catalog assets must not be symlinks: {path}")
    return resolved


def _asset_record(path: Path, root: Path, *, media_type: str) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    return {
        "path": relative,
        "media_type": media_type,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _validate_asset_record(record: Mapping[str, Any], root: Path) -> Path:
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Catalog asset path must be catalog-relative")
    path = root / relative
    _contained(path, root)
    if not path.is_file():
        raise FileNotFoundError(f"Catalog asset is missing: {path}")
    if path.stat().st_size != int(record.get("size_bytes", -1)):
        raise ValueError(f"Catalog asset size mismatch: {relative}")
    if _sha256(path) != record.get("sha256"):
        raise ValueError(f"Catalog asset hash mismatch: {relative}")
    return path


def load_catalog(catalog_root: str | Path | None = None, *, verify_assets: bool = True) -> dict[str, Any]:
    root = Path(catalog_root or default_catalog_root())
    manifest_path = root / CATALOG_MANIFEST
    if not manifest_path.exists():
        return {**_empty_catalog(), "catalog_root": root.resolve().as_posix()}
    _contained(manifest_path, root)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Object catalog schema must be {SCHEMA_VERSION}")
    records = value.get("objects")
    if not isinstance(records, list):
        raise ValueError("Object catalog objects must be a list")
    uuids: set[str] = set()
    ids: set[int] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Object catalog records must be objects")
        opaque_id = _validate_uuid(record.get("catalog_uuid"), label="catalog_uuid")
        obj_id = int(record.get("obj_id", 0))
        if obj_id <= 0 or opaque_id in uuids or obj_id in ids:
            raise ValueError("Catalog UUIDs and positive obj_id values must be unique")
        uuids.add(opaque_id)
        ids.add(obj_id)
        if record.get("state") not in {"active", "archived"}:
            raise ValueError("Catalog object state must be active or archived")
        assets = record.get("assets")
        if not isinstance(assets, Mapping) or "source" not in assets or "canonical_ply" not in assets:
            raise ValueError("Catalog object must retain source and canonical PLY assets")
        if verify_assets:
            for asset in assets.values():
                if not isinstance(asset, Mapping):
                    raise ValueError("Catalog asset record must be an object")
                _validate_asset_record(asset, root)
    result = dict(value)
    result["catalog_root"] = root.resolve().as_posix()
    return result


def _commit_catalog(value: dict[str, Any], root: Path) -> None:
    value = {key: item for key, item in value.items() if key != "catalog_root"}
    value["version"] = int(value.get("version", 0)) + 1
    value["updated_at"] = utc_now_iso()
    root.mkdir(parents=True, exist_ok=True)
    revisions = root / "revisions"
    revisions.mkdir(exist_ok=True)
    atomic_write_json(root / CATALOG_MANIFEST, value)
    atomic_write_json(revisions / f"{value['version']:08d}.json", value)


def _validate_texture(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Texture upload must be a regular staged file")
    if path.suffix.lower() != ".png":
        raise ValueError("Texture must be a single PNG file")
    data = path.read_bytes()
    if len(data) > 50 * 1024 * 1024:
        raise ValueError("Texture exceeds the 50 MiB file limit")
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError("Texture content is not PNG")
            image.verify()
    except (OSError, SyntaxError) as exc:
        raise ValueError("Texture content is not a valid PNG") from exc
    return data


def import_catalog_object(
    *,
    name: str,
    cad_path: str | Path,
    description: str | None = None,
    texture_path: str | Path | None = None,
    catalog_root: str | Path | None = None,
    catalog_uuid: str | None = None,
    obj_id: int | None = None,
    import_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Inspect staged assets and atomically add one immutable asset snapshot."""
    display_name = str(name).strip()
    if not display_name or len(display_name) > 120:
        raise ValueError("Object name must contain 1 to 120 characters")
    if description is not None and len(str(description)) > 2000:
        raise ValueError("Object description must not exceed 2000 characters")
    source_path = Path(cad_path)
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError("CAD upload must be a regular staged file")
    source_data = source_path.read_bytes()
    backend = load_posetemplatecreator_backend()
    safe_name = backend.safe_filename(source_path.name)
    source_format = backend.file_format(safe_name)
    canonical, extraction = backend.canonical_ply(safe_name, source_data)
    texture_data = _validate_texture(Path(texture_path)) if texture_path is not None else None
    if len(source_data) + (len(texture_data) if texture_data else 0) > 100 * 1024 * 1024:
        raise ValueError("Upload batch exceeds the 100 MiB limit")

    root = Path(catalog_root or default_catalog_root())
    staging_root = root.parent / STAGING_DIRECTORY
    staging_root.mkdir(parents=True, exist_ok=True)
    opaque_id = _validate_uuid(catalog_uuid or uuid.uuid4(), label="catalog_uuid")
    stage = staging_root / f"{opaque_id}.{uuid.uuid4().hex}.tmp"
    destination = root / "objects" / opaque_id
    stage.mkdir(parents=False, exist_ok=False)
    moved = False
    try:
        atomic_write_bytes(stage / safe_name, source_data)
        atomic_write_bytes(stage / "canonical.ply", canonical)
        if texture_data is not None:
            atomic_write_bytes(stage / "texture.png", texture_data)
        with _LOCK:
            catalog = load_catalog(root)
            if any(item["catalog_uuid"] == opaque_id for item in catalog["objects"]):
                raise ValueError(f"Catalog UUID already exists: {opaque_id}")
            assigned_id = int(obj_id) if obj_id is not None else int(catalog["next_obj_id"])
            if assigned_id <= 0 or any(int(item["obj_id"]) == assigned_id for item in catalog["objects"]):
                raise ValueError(f"BOP obj_id is not available: {assigned_id}")
            if destination.exists():
                raise ValueError(f"Catalog asset directory already exists: {opaque_id}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage, destination)
            moved = True
            now = utc_now_iso()
            assets = {
                "source": _asset_record(
                    destination / safe_name, root, media_type="application/octet-stream"
                ),
                "canonical_ply": _asset_record(
                    destination / "canonical.ply", root, media_type="application/octet-stream"
                ),
            }
            if texture_data is not None:
                assets["texture"] = _asset_record(
                    destination / "texture.png", root, media_type="image/png"
                )
            record = {
                "catalog_uuid": opaque_id,
                "obj_id": assigned_id,
                "name": display_name,
                "description": str(description).strip() if description else None,
                "source_filename": safe_name,
                "source_format": source_format,
                "source_sha256": _sha256_bytes(source_data),
                "canonical_ply_sha256": _sha256_bytes(canonical),
                "texture_sha256": _sha256_bytes(texture_data) if texture_data else None,
                "assets": assets,
                "extraction": extraction,
                "created_at": now,
                "updated_at": now,
                "state": "active",
                "import_provenance": dict(import_provenance or {}),
            }
            catalog["objects"].append(record)
            catalog["objects"].sort(key=lambda item: int(item["obj_id"]))
            catalog["next_obj_id"] = max(int(catalog["next_obj_id"]), assigned_id + 1)
            _commit_catalog(catalog, root)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        if moved:
            shutil.rmtree(destination, ignore_errors=True)
        raise
    return get_catalog_object(opaque_id, catalog_root=root)


def get_catalog_object(catalog_uuid: str, *, catalog_root: str | Path | None = None) -> dict[str, Any]:
    opaque_id = _validate_uuid(catalog_uuid, label="catalog_uuid")
    catalog = load_catalog(catalog_root)
    for item in catalog["objects"]:
        if item["catalog_uuid"] == opaque_id:
            return {**item, "catalog_root": catalog["catalog_root"]}
    raise KeyError(f"Unknown catalog object: {opaque_id}")

def set_catalog_object_state(
    catalog_uuid: str, *, state: str, catalog_root: str | Path | None = None
) -> dict[str, Any]:
    if state not in {"active", "archived"}:
        raise ValueError("Catalog state must be active or archived")
    root = Path(catalog_root or default_catalog_root())
    opaque_id = _validate_uuid(catalog_uuid, label="catalog_uuid")
    with _LOCK:
        catalog = load_catalog(root)
        for item in catalog["objects"]:
            if item["catalog_uuid"] == opaque_id:
                if item["state"] != state:
                    item["state"] = state
                    item["updated_at"] = utc_now_iso()
                    _commit_catalog(catalog, root)
                return get_catalog_object(opaque_id, catalog_root=root)
    raise KeyError(f"Unknown catalog object: {opaque_id}")
