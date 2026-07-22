from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from posetestbot.pose_templates import catalog as catalog_module
from posetestbot.pose_templates.catalog import (
    catalog_export_manifest,
    delete_catalog_object,
    import_catalog_metadata,
    import_catalog_object,
    load_catalog,
    normalize_catalog_metadata,
    set_catalog_object_state,
    update_catalog_object_metadata,
)


class FakeMeshBackend:
    constants = SimpleNamespace(MAX_UPLOAD_BYTES=50 * 1024 * 1024)

    @staticmethod
    def safe_filename(filename: str | None) -> str:
        value = Path(str(filename or "")).name
        if not value:
            raise ValueError("A filename is required")
        return value

    @staticmethod
    def file_format(filename: str) -> str:
        extension = Path(filename).suffix.lower().lstrip(".")
        if extension not in {"ply", "stl", "obj"}:
            raise ValueError("Unsupported CAD format")
        return extension

    def canonical_ply(self, filename: str, data: bytes) -> tuple[bytes, dict]:
        self.file_format(filename)
        return (
            b"ply\nformat ascii 1.0\ncomment recovery test\nend_header\n",
            {
                "vertices": 8,
                "faces": 12,
                "bounds_mm": [[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]],
                "watertight": True,
            },
        )


@pytest.fixture(autouse=True)
def fake_mesh_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        catalog_module,
        "load_posetemplatecreator_backend",
        lambda: FakeMeshBackend(),
    )


def add_workpiece(root: Path, source: Path, *, name: str) -> dict:
    source.write_bytes(b"solid recovery fixture\nendsolid recovery fixture\n")
    return import_catalog_object(name=name, cad_path=source, catalog_root=root)


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    [
        ("name", None, "Object name must be a string"),
        ("name", 17, "Object name must be a string"),
        ("name", {"label": "fixture"}, "Object name must be a string"),
        ("tags", [None], "tags value must be a string"),
        ("tags", [17], "tags value must be a string"),
        ("tags", [{"label": "inspection"}], "tags value must be a string"),
        ("groups", [None], "groups value must be a string"),
        ("groups", [False], "groups value must be a string"),
        ("groups", [{"label": "bench-a"}], "groups value must be a string"),
    ],
    ids=[
        "null-name",
        "numeric-name",
        "object-name",
        "null-tag",
        "numeric-tag",
        "object-tag",
        "null-group",
        "boolean-group",
        "object-group",
    ],
)
def test_name_and_classification_values_require_actual_strings(
    field: str,
    invalid_value: object,
    message: str,
) -> None:
    metadata = {
        "name": "Recovery fixture",
        "alias": None,
        "description": None,
        "tags": ["inspection"],
        "groups": ["bench-a"],
        "attributes": {},
    }
    metadata[field] = invalid_value

    with pytest.raises(ValueError, match=re.escape(message)):
        normalize_catalog_metadata(metadata)


def test_catalog_export_remains_available_when_an_asset_is_corrupt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "object_catalog"
    intact = add_workpiece(root, tmp_path / "intact.stl", name="Intact")
    corrupt = add_workpiece(root, tmp_path / "corrupt.stl", name="Corrupt")
    source = root / corrupt["assets"]["source"]["path"]
    source.write_bytes(b"x" * source.stat().st_size)

    with pytest.raises(ValueError, match="hash mismatch"):
        load_catalog(root)

    exported = catalog_export_manifest(root)

    assert exported["schema_version"] == "object_catalog.v1"
    assert "catalog_root" not in exported
    assert [item["catalog_uuid"] for item in exported["objects"]] == [
        intact["catalog_uuid"],
        corrupt["catalog_uuid"],
    ]


def test_metadata_import_updates_intact_records_and_skips_damaged_assets(
    tmp_path: Path,
) -> None:
    root = tmp_path / "object_catalog"
    intact = add_workpiece(root, tmp_path / "intact.stl", name="Intact original")
    corrupt = add_workpiece(root, tmp_path / "corrupt.stl", name="Corrupt original")
    missing = add_workpiece(root, tmp_path / "missing.stl", name="Missing original")
    portable = copy.deepcopy(catalog_export_manifest(root))
    imported_names = {
        intact["catalog_uuid"]: "Intact imported",
        corrupt["catalog_uuid"]: "Corrupt imported",
        missing["catalog_uuid"]: "Missing imported",
    }
    for item in portable["objects"]:
        item["name"] = imported_names[item["catalog_uuid"]]
        item["tags"] = ["imported"]

    corrupt_source = root / corrupt["assets"]["source"]["path"]
    corrupt_source.write_bytes(b"x" * corrupt_source.stat().st_size)
    missing_canonical = root / missing["assets"]["canonical_ply"]["path"]
    missing_canonical.unlink()

    result = import_catalog_metadata(portable, catalog_root=root)
    records = {
        item["catalog_uuid"]: item
        for item in load_catalog(root, verify_assets=False)["objects"]
    }

    assert result["updated"] == [intact["catalog_uuid"]]
    assert result["unchanged"] == []
    assert result["skipped_missing_assets"] == [
        corrupt["catalog_uuid"],
        missing["catalog_uuid"],
    ]
    assert records[intact["catalog_uuid"]]["name"] == "Intact imported"
    assert records[intact["catalog_uuid"]]["tags"] == ["imported"]
    assert records[corrupt["catalog_uuid"]]["name"] == "Corrupt original"
    assert records[corrupt["catalog_uuid"]]["tags"] == []
    assert records[missing["catalog_uuid"]]["name"] == "Missing original"
    assert records[missing["catalog_uuid"]]["tags"] == []


def test_legacy_v1_defaults_load_and_persist_on_next_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "object_catalog"
    record = add_workpiece(root, tmp_path / "legacy.stl", name="Legacy")
    manifest_path = root / "object_catalog.json"
    manifest = json.loads(manifest_path.read_text())
    legacy = manifest["objects"][0]
    for field in ("alias", "tags", "groups", "attributes"):
        legacy.pop(field)
    for field in ("geometry_revision", "source_to_mm_scale", "geometry_revisions"):
        legacy.pop(field)
    manifest.pop("tombstones")
    manifest_path.write_text(json.dumps(manifest))
    canonical_path = root / record["assets"]["canonical_ply"]["path"]
    canonical_bytes = canonical_path.read_bytes()

    loaded = load_catalog(root)
    exported = catalog_export_manifest(root)

    loaded_record = loaded["objects"][0]
    assert loaded["schema_version"] == "object_catalog.v1"
    assert loaded["tombstones"] == []
    assert loaded_record["catalog_uuid"] == record["catalog_uuid"]
    assert loaded_record["alias"] is None
    assert loaded_record["tags"] == []
    assert loaded_record["groups"] == []
    assert loaded_record["attributes"] == {}
    assert loaded_record["geometry_revision"] == 1
    assert loaded_record["source_to_mm_scale"] == 1.0
    assert (
        loaded_record["geometry_revisions"][0]["canonical_ply_sha256"]
        == record["canonical_ply_sha256"]
    )
    assert exported["objects"][0]["geometry_revisions"][0]["revision"] == 1
    update_catalog_object_metadata(
        record["catalog_uuid"], {"alias": "Migrated"}, catalog_root=root
    )
    persisted = json.loads(manifest_path.read_text())["objects"][0]
    assert persisted["geometry_revision"] == 1
    assert persisted["geometry_revisions"][0]["canonical_ply"]["path"] == record[
        "assets"
    ]["canonical_ply"]["path"]
    assert canonical_path.read_bytes() == canonical_bytes


def test_delete_records_cleanup_failure_and_retries_from_tombstone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "object_catalog"
    record = add_workpiece(root, tmp_path / "delete.stl", name="Delete me")
    object_root = root / "objects" / record["catalog_uuid"]
    set_catalog_object_state(record["catalog_uuid"], state="archived", catalog_root=root)
    original_rmtree = catalog_module.shutil.rmtree
    attempts = 0

    def fail_once(path: str | Path, *args: object, **kwargs: object) -> None:
        nonlocal attempts
        if Path(path) == object_root and attempts == 0:
            attempts += 1
            raise OSError("injected cleanup failure")
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(catalog_module.shutil, "rmtree", fail_once)
    first = delete_catalog_object(
        record["catalog_uuid"],
        catalog_root=root,
        template_library_root=tmp_path / "pose_templates",
    )

    assert first["status"] == "deleted_cleanup_pending"
    assert first["asset_cleanup"]["status"] == "pending"
    assert "injected cleanup failure" in first["asset_cleanup"]["last_error"]
    assert object_root.is_dir()
    assert load_catalog(root, verify_assets=False)["objects"] == []

    second = delete_catalog_object(
        record["catalog_uuid"],
        catalog_root=root,
        template_library_root=tmp_path / "pose_templates",
    )

    assert second["status"] == "deleted"
    assert second["already_deleted"] is True
    assert second["asset_cleanup"]["status"] == "complete"
    assert not object_root.exists()
