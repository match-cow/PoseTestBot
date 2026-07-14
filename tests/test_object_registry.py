from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from posetestbot.objects.registry import load_object_registry
from posetestbot.pipeline.run_config import (
    create_run_config,
    load_run_config,
    sequence_plan_from_run_config,
    write_run_config,
)


def transform(x: float = 0.0) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def registry_fixture(tmp_path: Path) -> Path:
    folder = tmp_path / "objects"
    folder.mkdir()
    (folder / "objects.json").write_text(
        json.dumps({"zebra": transform(40), "alpha": transform(10), "middle": transform(20)})
    )
    for name in ("zebra", "alpha", "middle"):
        (folder / f"{name}.ply").write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    (folder / "alpha.png").write_bytes(b"png")
    return folder


def test_registry_inverts_transform_and_keeps_stable_subset_ids(tmp_path: Path) -> None:
    folder = registry_fixture(tmp_path)
    registry = load_object_registry(folder)

    assert registry.id_mapping == {"alpha": 1, "middle": 2, "zebra": 3}
    subset = registry.selected_entries(["zebra", "alpha"])
    assert [(entry.name, entry.obj_id) for entry in subset] == [("alpha", 1), ("zebra", 3)]
    np.testing.assert_allclose(registry.by_name["alpha"].object_to_template[:3, 3], [-10, 0, 0])
    assert registry.by_name["alpha"].texture_path == folder / "alpha.png"
    assert registry.selected_entries([]) == ()


def test_registry_reports_invalid_rigid_transform_and_symlink_escape(tmp_path: Path) -> None:
    folder = tmp_path / "objects"
    folder.mkdir()
    outside = tmp_path / "outside.ply"
    outside.write_text("ply\n")
    (folder / "escape.ply").symlink_to(outside)
    (folder / "objects.json").write_text(json.dumps({"escape": [[2, 0, 0, 0]] * 4}))

    entry = load_object_registry(folder).entries[0]

    assert entry.valid is False
    assert any("escapes" in error for error in entry.errors)
    assert any("bottom row" in error or "orthonormal" in error for error in entry.errors)


def test_run_config_snapshots_selection_legacy_fallback_and_sequence_injection(tmp_path: Path) -> None:
    folder = registry_fixture(tmp_path)
    run_root = tmp_path / "run"
    config = create_run_config(
        run_root=run_root,
        object_folder=folder.as_posix(),
        selected_objects=["zebra"],
        sequence_id="sync_to_bop_dry_run",
    )
    assert config.selected_objects == ("zebra",)
    plan = sequence_plan_from_run_config(config.to_dict())
    prepare = next(step for step in plan.steps if step.stage_id == "blenderproc_prepare")
    export = next(step for step in plan.steps if step.stage_id == "bop_export")
    assert prepare.options["object_name"] == ["zebra"]
    assert export.options["object_name"] == ["zebra"]

    write_run_config(run_root, config)
    raw = json.loads((run_root / "run_config.json").read_text())
    raw.pop("selected_objects")
    (run_root / "run_config.json").write_text(json.dumps(raw))
    loaded = load_run_config(run_root / "run_config.json")
    assert loaded["selected_objects"] == ["alpha", "middle", "zebra"]
    assert any(item["code"] == "legacy_object_selection_inferred" for item in loaded["warnings"])

    objectless = create_run_config(
        run_root=tmp_path / "empty",
        object_folder=folder.as_posix(),
        selected_objects=[],
        sequence_id="sync_to_bop_dry_run",
    )
    empty_plan = sequence_plan_from_run_config(objectless.to_dict())
    assert all(
        step.options.get("objectless") is True
        for step in empty_plan.steps
        if step.stage_id in {"blenderproc_prepare", "blenderproc_render", "bop_export"}
    )


def test_registry_rejects_unknown_explicit_selection(tmp_path: Path) -> None:
    registry = load_object_registry(registry_fixture(tmp_path))
    with pytest.raises(ValueError, match="Unknown selected object"):
        registry.validate_selection(["missing"])
