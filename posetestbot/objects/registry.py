"""Validated, stable-ID object model registry.

``objects.json`` stores the historical template-to-object matrices.  All new
consumers use this service's explicit object-to-template transform instead of
open-coding the required inversion.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt

SAFE_OBJECT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _contained_file(folder: Path, path: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(folder.resolve(strict=True))
    except (FileNotFoundError, OSError, ValueError):
        return False
    return path.is_file()


@dataclass(frozen=True)
class ObjectRegistryEntry:
    name: str
    obj_id: int
    valid: bool
    errors: tuple[str, ...]
    model_path: Path
    texture_path: Path | None
    object_to_template: np.ndarray | None

    def to_dict(self, *, selected: bool = False, asset_prefix: str | None = None) -> dict:
        transform = None
        if self.object_to_template is not None:
            quaternion = pr.quaternion_from_matrix(self.object_to_template[:3, :3])
            transform = {
                "semantics": "entity_to_parent",
                "parent_frame": "template_base",
                "translation_mm": self.object_to_template[:3, 3].tolist(),
                "rotation_quaternion_wxyz": quaternion.tolist(),
            }
        model_url = texture_url = None
        if asset_prefix and self.valid:
            model_url = f"{asset_prefix}/{self.name}/mesh"
            if self.texture_path is not None:
                texture_url = f"{asset_prefix}/{self.name}/texture"
        return {
            "name": self.name,
            "obj_id": self.obj_id,
            "valid": self.valid,
            "errors": list(self.errors),
            "selected": selected,
            "model_filename": self.model_path.name,
            "texture_filename": self.texture_path.name if self.texture_path else None,
            "model_url": model_url,
            "texture_url": texture_url,
            "transform": transform,
        }


@dataclass(frozen=True)
class ObjectRegistry:
    folder: Path
    entries: tuple[ObjectRegistryEntry, ...]
    source_sha256: str

    @property
    def by_name(self) -> dict[str, ObjectRegistryEntry]:
        return {entry.name: entry for entry in self.entries}

    @property
    def valid_names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.entries if entry.valid)

    @property
    def id_mapping(self) -> dict[str, int]:
        return {entry.name: entry.obj_id for entry in self.entries}

    def validate_selection(self, names: Iterable[str]) -> tuple[str, ...]:
        selected = tuple(dict.fromkeys(str(name) for name in names))
        known = self.by_name
        unknown = [name for name in selected if name not in known]
        invalid = [name for name in selected if name in known and not known[name].valid]
        if unknown:
            raise ValueError("Unknown selected object(s): " + ", ".join(sorted(unknown)))
        if invalid:
            raise ValueError("Invalid selected object(s): " + ", ".join(sorted(invalid)))
        return selected

    def selected_entries(self, names: Iterable[str]) -> tuple[ObjectRegistryEntry, ...]:
        selected = set(self.validate_selection(names))
        return tuple(entry for entry in self.entries if entry.name in selected)

    def provenance(self) -> dict:
        return {
            "schema_version": "object_registry.v1",
            "source": (self.folder / "objects.json").as_posix(),
            "source_sha256": self.source_sha256,
            "entry_count": len(self.entries),
            "valid_count": sum(entry.valid for entry in self.entries),
            "invalid_count": sum(not entry.valid for entry in self.entries),
            "stable_id_mapping": self.id_mapping,
            "transform_convention": "objects_json_template_to_object_inverted_to_object_to_template",
        }


def _rigid_transform(value: object, *, name: str) -> tuple[np.ndarray | None, list[str]]:
    errors: list[str] = []
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None, [f"Object transform for {name!r} must be a finite 4x4 matrix"]
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        return None, [f"Object transform for {name!r} must be a finite 4x4 matrix"]
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        errors.append("Transform bottom row must be [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5
    ):
        errors.append("Transform rotation must be right-handed and orthonormal")
    if errors:
        return None, errors
    return pt.invert_transform(matrix, check=True), []


def load_object_registry(folder: str | Path) -> ObjectRegistry:
    folder = Path(folder)
    registry_path = folder / "objects.json"
    if not registry_path.is_file() or registry_path.is_symlink():
        raise FileNotFoundError(f"Missing object registry: {registry_path}")
    raw = registry_path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid object registry JSON {registry_path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"Object registry must be a JSON object: {registry_path}")

    entries: list[ObjectRegistryEntry] = []
    # Stable IDs are assigned before selection and include every registry key.
    for obj_id, raw_name in enumerate(sorted(value, key=str), start=1):
        name = str(raw_name)
        errors: list[str] = []
        if not SAFE_OBJECT_NAME.fullmatch(name) or name in {".", ".."}:
            errors.append("Object name must be one safe filename component")
        model_path = folder / f"{name}.ply"
        if not _contained_file(folder, model_path):
            errors.append("PLY model is missing or escapes the registry folder")
        texture_candidate = folder / f"{name}.png"
        texture_path = texture_candidate if _contained_file(folder, texture_candidate) else None
        transform, transform_errors = _rigid_transform(value[raw_name], name=name)
        errors.extend(transform_errors)
        entries.append(
            ObjectRegistryEntry(
                name=name,
                obj_id=obj_id,
                valid=not errors,
                errors=tuple(errors),
                model_path=model_path,
                texture_path=texture_path,
                object_to_template=transform,
            )
        )
    return ObjectRegistry(
        folder=folder,
        entries=tuple(entries),
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )
