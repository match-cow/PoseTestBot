"""Run-owned immutable template selection and per-instance GT resolution."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from posetestbot.io.artifacts import (
    BLENDERPROC_RENDER_PLAN,
    BOP_DIR,
    MASKS_DIR,
    OBJECT_INSTANCES,
    POSE_TEMPLATE_SELECTION,
    PROCESSED_DIR,
)
from posetestbot.io.atomic import atomic_write_json
from posetestbot.pose_templates.catalog import utc_now_iso
from posetestbot.pose_templates.library import (
    default_template_library_root,
    validate_template_bundle,
)
from posetestbot.pose_templates.transforms import (
    matrix_from_record,
    transform_record,
    validate_rigid_matrix,
)


SELECTION_SCHEMA_VERSION = "pose_template_selection.v1"
OBJECT_INSTANCES_SCHEMA_VERSION = "object_instances.v1"
SELECTION_DIRECTORY = "pose_template_selection"


class PoseTemplateSelectionConflict(RuntimeError):
    def __init__(self, message: str, *, blockers: list[str]):
        super().__init__(message)
        self.blockers = blockers


def replacement_blockers(run_root: str | Path) -> list[str]:
    root = Path(run_root)
    candidates = [
        root / OBJECT_INSTANCES,
        root / BLENDERPROC_RENDER_PLAN,
        root / "blenderproc_output",
        root / MASKS_DIR,
        root / BOP_DIR,
        root / PROCESSED_DIR / "blenderproc",
    ]
    for tree_name in ("synchronized", "rectified"):
        tree = root / PROCESSED_DIR / tree_name
        if not tree.is_dir():
            continue
        for sensor in tree.iterdir():
            if not sensor.is_dir():
                continue
            candidates.extend((sensor / "blenderproc", sensor / MASKS_DIR))
    return sorted(
        {path.relative_to(root).as_posix() for path in candidates if path.exists()}
    )


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def select_pose_template(
    run_root: str | Path,
    template_uuid: str,
    *,
    placement: Mapping[str, Any],
    confirmed: bool,
    operator: str,
    library_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {root}")
    operator_name = str(operator).strip()
    if not operator_name:
        raise ValueError("Selection operator provenance is required")
    placement_matrix = matrix_from_record(placement, label="template placement")
    library = Path(library_root or default_template_library_root())
    source = validate_template_bundle(library / str(uuid.UUID(template_uuid)), library_root=library)
    if source["archive"]["state"] != "active":
        raise ValueError("Archived pose templates cannot be selected for a new run")
    current_path = root / POSE_TEMPLATE_SELECTION
    if current_path.exists():
        current = load_pose_template_selection(root)
        same = (
            current["template_uuid"] == source["template_uuid"]
            and np.allclose(
                np.asarray(current["template_base_from_pose_template"]["matrix"]),
                placement_matrix,
                atol=1e-10,
            )
            and bool(current["placement_confirmed"]) == bool(confirmed)
        )
        if same:
            return current
        blockers = replacement_blockers(root)
        if blockers:
            raise PoseTemplateSelectionConflict(
                "Pose-template selection cannot be replaced after dependent artifacts exist.",
                blockers=blockers,
            )
    selection_root = root / PROCESSED_DIR / SELECTION_DIRECTORY
    stage = selection_root.parent / f".{SELECTION_DIRECTORY}.{uuid.uuid4().hex}.tmp"
    if stage.exists():
        raise FileExistsError(stage)
    stage.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source["bundle_path"], stage)
    snapshot_bundle = validate_template_bundle(stage, library_root=stage.parent, allow_staging=True)
    resolved = []
    for item in snapshot_bundle["instances"]:
        nominal = validate_rigid_matrix(
            item["pose_template_from_object"]["matrix"], label="pose_template_from_object"
        )
        final = placement_matrix @ nominal
        resolved.append(
            {
                "instance_uuid": item["instance_uuid"],
                "catalog_uuid": item["catalog"]["catalog_uuid"],
                "obj_id": int(item["catalog"]["obj_id"]),
                "name": item["catalog"]["name"],
                "pose_template_from_object": item["pose_template_from_object"],
                "template_base_from_object": transform_record(
                    final,
                    parent="template_base",
                    child=f"object:{item['instance_uuid']}",
                ),
                "assets": item["assets"],
            }
        )
    if selection_root.exists():
        shutil.rmtree(selection_root)
    os.replace(stage, selection_root)
    selected_at = utc_now_iso()
    selection = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "template_uuid": source["template_uuid"],
        "bundle_sha256": source["bundle_sha256"],
        "configuration_sha256": source["hashes"]["configuration"],
        "bundle_snapshot": (Path(PROCESSED_DIR) / SELECTION_DIRECTORY).as_posix(),
        "bundle_snapshot_sha256": _tree_hash(selection_root),
        "template_base_from_pose_template": transform_record(
            placement_matrix, parent="template_base", child="pose_template"
        ),
        "placement_confirmed": bool(confirmed),
        "instances": resolved,
        "selected_at": selected_at,
        "operator": operator_name,
        "source": source["source"],
        "print_compensation": source["print_compensation"],
        "catalog_snapshot": source["catalog_snapshot"],
    }
    atomic_write_json(current_path, selection)
    config_path = root / "run_config.json"
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        config["schema_version"] = "run_config.v2"
        config["dataset_mode"] = "pose_template"
        config["selected_objects"] = []
        config["pose_template"] = {
            "template_uuid": source["template_uuid"],
            "selection_artifact": POSE_TEMPLATE_SELECTION,
            "bundle_sha256": source["bundle_sha256"],
            "placement_confirmed": bool(confirmed),
        }
        from posetestbot.pipeline.run_config import validate_run_config

        validate_run_config(config)
        atomic_write_json(config_path, config)
    return load_pose_template_selection(root)


def load_pose_template_selection(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root).resolve()
    path = root / POSE_TEMPLATE_SELECTION
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping) or value.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ValueError(f"Selection schema must be {SELECTION_SCHEMA_VERSION}")
    snapshot_relative = Path(str(value.get("bundle_snapshot", "")))
    if snapshot_relative.is_absolute() or ".." in snapshot_relative.parts:
        raise ValueError("Selection bundle snapshot must be run-relative")
    snapshot = root / snapshot_relative
    bundle = validate_template_bundle(snapshot, library_root=snapshot.parent, allow_staging=True)
    if bundle["template_uuid"] != value.get("template_uuid"):
        raise ValueError("Selection snapshot template UUID mismatch")
    if bundle["bundle_sha256"] != value.get("bundle_sha256"):
        raise ValueError("Selection snapshot bundle hash mismatch")
    # archive_state is intentionally mutable library metadata and does not affect
    # a selected run's immutable content provenance.
    if _tree_hash(snapshot) != value.get("bundle_snapshot_sha256"):
        raise ValueError("Selection bundle snapshot hash mismatch")
    placement = matrix_from_record(value["template_base_from_pose_template"])
    if len(value.get("instances", [])) != len(bundle["instances"]):
        raise ValueError("Selection resolved instance count mismatch")
    for selected, source in zip(value["instances"], bundle["instances"], strict=True):
        nominal = matrix_from_record(source["pose_template_from_object"])
        expected = placement @ nominal
        actual = matrix_from_record(selected["template_base_from_object"])
        if selected["instance_uuid"] != source["instance_uuid"] or not np.allclose(
            actual, expected, atol=1e-8
        ):
            raise ValueError("Selection resolved instance transform mismatch")
    return dict(value)


def prepare_object_instances(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root).resolve()
    selection = load_pose_template_selection(root)
    if not selection.get("placement_confirmed"):
        raise ValueError("Pose-template placement must be explicitly confirmed")
    snapshot = root / selection["bundle_snapshot"]
    objects = []
    for item in selection["instances"]:
        files = item["assets"]
        canonical = snapshot / files["canonical_ply"]["path"]
        texture = snapshot / files["texture"]["path"] if "texture" in files else None
        objects.append(
            {
                "instance_uuid": item["instance_uuid"],
                "catalog_uuid": item["catalog_uuid"],
                "obj_id": item["obj_id"],
                "name": item["name"],
                "canonical_ply": canonical.relative_to(root).as_posix(),
                "canonical_ply_sha256": files["canonical_ply"]["sha256"],
                "texture": texture.relative_to(root).as_posix() if texture else None,
                "texture_sha256": files.get("texture", {}).get("sha256"),
                "pose_template_from_object": item["pose_template_from_object"],
                "template_base_from_object": item["template_base_from_object"],
            }
        )
    artifact = {
        "schema_version": OBJECT_INSTANCES_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "template_uuid": selection["template_uuid"],
        "bundle_sha256": selection["bundle_sha256"],
        "selection_sha256": hashlib.sha256(
            (root / POSE_TEMPLATE_SELECTION).read_bytes()
        ).hexdigest(),
        "instances": objects,
        "provenance": {
            "source": selection["source"],
            "operator": selection["operator"],
            "selected_at": selection["selected_at"],
        },
    }
    atomic_write_json(root / OBJECT_INSTANCES, artifact)
    return artifact
