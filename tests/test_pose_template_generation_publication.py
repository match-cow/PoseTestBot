from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import trimesh

from posetestbot.pose_templates import library as library_module
from posetestbot.pose_templates.catalog import (
    import_catalog_object,
    set_catalog_object_state,
)
from posetestbot.pose_templates.library import generate_template_bundle
from posetestbot.pose_templates.orientations import analyze_catalog_orientations


def test_heavy_generation_does_not_hold_catalog_lock_and_stale_stage_is_rejected(
    tmp_path: Path, monkeypatch: Any
) -> None:
    cad = tmp_path / "box.stl"
    cad.write_bytes(trimesh.creation.box(extents=(20, 10, 8)).export(file_type="stl"))
    catalog = tmp_path / "object_catalog"
    library = tmp_path / "pose_templates"
    record = import_catalog_object(name="Box", cad_path=cad, catalog_root=catalog)
    analysis = analyze_catalog_orientations(
        record["catalog_uuid"], catalog_root=catalog
    )
    staged = threading.Event()
    allow_publication = threading.Event()
    original_thumbnail = library_module.build_template_thumbnail

    def pause_staging(*args: Any, **kwargs: Any) -> dict[str, Any]:
        value = original_thumbnail(*args, **kwargs)
        staged.set()
        if not allow_publication.wait(timeout=10):
            raise TimeoutError("test did not release template publication")
        return value

    monkeypatch.setattr(library_module, "build_template_thumbnail", pause_staging)
    generation_errors: list[BaseException] = []
    archive_errors: list[BaseException] = []

    def generate() -> None:
        try:
            generate_template_bundle(
                {
                    "display_name": "Concurrent publication",
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
                library_root=library,
            )
        except BaseException as exc:  # pragma: no cover - surfaced by assertions
            generation_errors.append(exc)

    def archive() -> None:
        try:
            set_catalog_object_state(
                record["catalog_uuid"], state="archived", catalog_root=catalog
            )
        except BaseException as exc:  # pragma: no cover - surfaced by assertions
            archive_errors.append(exc)

    generation_thread = threading.Thread(target=generate, daemon=True)
    generation_thread.start()
    assert staged.wait(timeout=10)
    archive_thread = threading.Thread(target=archive)
    archive_thread.start()
    archive_thread.join(timeout=1)
    catalog_was_not_blocked = not archive_thread.is_alive()
    allow_publication.set()
    archive_thread.join(timeout=10)
    generation_thread.join(timeout=10)

    assert catalog_was_not_blocked, "heavy template generation held catalogue lock"
    assert archive_errors == []
    assert not generation_thread.is_alive()
    assert len(generation_errors) == 1
    assert "archived object" in str(generation_errors[0])
    assert not [path for path in library.iterdir() if not path.name.startswith(".")]
