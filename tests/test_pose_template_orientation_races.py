from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import trimesh

from posetestbot.pose_templates import orientations as orientations_module
from posetestbot.pose_templates.catalog import (
    delete_catalog_object,
    import_catalog_object,
    set_catalog_object_state,
    update_catalog_object_metadata,
)
from posetestbot.pose_templates.library import generate_template_bundle


def test_orientation_worker_cannot_recreate_permanently_deleted_workpiece(
    tmp_path: Path, monkeypatch: Any
) -> None:
    cad = tmp_path / "box.stl"
    cad.write_bytes(trimesh.creation.box(extents=(20, 10, 8)).export(file_type="stl"))
    catalog = tmp_path / "object_catalog"
    library = tmp_path / "pose_templates"
    record = import_catalog_object(name="Box", cad_path=cad, catalog_root=catalog)

    analyzed = threading.Event()
    allow_publication = threading.Event()
    original_build = orientations_module.build_orientation_analysis

    def pause_after_analysis(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_build(*args, **kwargs)
        analyzed.set()
        if not allow_publication.wait(timeout=10):
            raise TimeoutError("test did not release orientation publication")
        return result

    monkeypatch.setattr(
        orientations_module, "build_orientation_analysis", pause_after_analysis
    )
    errors: list[BaseException] = []

    def analyze() -> None:
        try:
            orientations_module.analyze_catalog_orientations(
                record["catalog_uuid"], catalog_root=catalog
            )
        except BaseException as exc:  # pragma: no cover - surfaced by assertions
            errors.append(exc)

    worker = threading.Thread(target=analyze)
    worker.start()
    assert analyzed.wait(timeout=10)

    set_catalog_object_state(
        record["catalog_uuid"], state="archived", catalog_root=catalog
    )
    delete_catalog_object(
        record["catalog_uuid"],
        catalog_root=catalog,
        template_library_root=library,
    )
    assert not (catalog / "objects" / record["catalog_uuid"]).exists()

    allow_publication.set()
    worker.join(timeout=10)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], KeyError)
    assert not (catalog / "objects" / record["catalog_uuid"]).exists()


def test_bundle_generation_reenters_catalog_lock_for_missing_orientation_cache(
    tmp_path: Path,
) -> None:
    cad = tmp_path / "box.stl"
    cad.write_bytes(trimesh.creation.box(extents=(20, 10, 8)).export(file_type="stl"))
    catalog = tmp_path / "object_catalog"
    record = import_catalog_object(name="Box", cad_path=cad, catalog_root=catalog)
    analysis = orientations_module.analyze_catalog_orientations(
        record["catalog_uuid"], catalog_root=catalog
    )
    cache = orientations_module.orientation_cache_path(
        record["catalog_uuid"], catalog_root=catalog
    )
    cache.unlink()

    results: list[dict[str, Any]] = []
    errors: list[BaseException] = []

    def generate() -> None:
        try:
            results.append(
                generate_template_bundle(
                    {
                        "display_name": "Reentrant catalogue lock",
                        "instances": [
                            {
                                "instance_uuid": (
                                    "11111111-1111-4111-8111-111111111111"
                                ),
                                "catalog_uuid": record["catalog_uuid"],
                                "orientation_id": analysis["orientations"][0][
                                    "orientation_id"
                                ],
                                "pose": {
                                    "x_mm": 40,
                                    "y_mm": 40,
                                    "rotation_deg": 0,
                                },
                            }
                        ],
                    },
                    catalog_root=catalog,
                    library_root=tmp_path / "pose_templates",
                )
            )
        except BaseException as exc:  # pragma: no cover - surfaced by assertions
            errors.append(exc)

    worker = threading.Thread(target=generate, daemon=True)
    worker.start()
    worker.join(timeout=15)

    assert not worker.is_alive(), "nested catalogue mutation lock deadlocked"
    assert errors == []
    assert len(results) == 1
    assert cache.is_file()


def test_orientation_card_uses_separate_bounded_cache_without_full_analysis(
    tmp_path: Path, monkeypatch: Any
) -> None:
    cad = tmp_path / "box.stl"
    cad.write_bytes(trimesh.creation.box(extents=(20, 10, 8)).export(file_type="stl"))
    catalog = tmp_path / "object_catalog"
    record = import_catalog_object(name="Box", cad_path=cad, catalog_root=catalog)
    analysis = orientations_module.analyze_catalog_orientations(
        record["catalog_uuid"], catalog_root=catalog
    )
    canonical = catalog / record["assets"]["canonical_ply"]["path"]
    thumbnail_path = canonical.with_name(
        orientations_module.ORIENTATION_THUMBNAIL_FILENAME
    )
    assert (
        thumbnail_path.stat().st_size
        <= orientations_module.ORIENTATION_THUMBNAIL_MAX_BYTES
    )

    def unexpected_full_analysis(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("orientation card loaded the full contour analysis")

    monkeypatch.setattr(
        orientations_module,
        "load_catalog_orientation_analysis",
        unexpected_full_analysis,
    )
    update_catalog_object_metadata(
        record["catalog_uuid"],
        {
            "name": "😀" * 120,
            "alias": "alias" * 24,
            "tags": [f"tag-{index:02d}-" + "x" * 73 for index in range(64)],
            "groups": [f"group-{index:02d}-" + "y" * 71 for index in range(64)],
        },
        catalog_root=catalog,
    )
    thumbnail = orientations_module.load_catalog_orientation_thumbnail(
        record["catalog_uuid"], catalog_root=catalog
    )

    assert thumbnail["catalog"] == {
        "obj_id": record["obj_id"],
        "name": "😀" * 120,
    }
    assert thumbnail["orientation"]["orientation_id"] == analysis["orientations"][0][
        "orientation_id"
    ]
    assert "contours" not in json.dumps(thumbnail)
    assert (
        len(orientations_module._canonical_json(thumbnail) + b"\n")
        <= orientations_module.ORIENTATION_THUMBNAIL_MAX_BYTES
    )
