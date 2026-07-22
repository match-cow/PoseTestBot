"""Private, verified access to the pinned PoseTemplateCreator backend.

Only the small source modules needed for mesh safety, layout, and rendering are
loaded.  In particular, the upstream FastAPI application is never imported.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


POSETEMPLATECREATOR_REVISION = "97ddb9b7b756912deb8c2d2d6dde186b461e5d9d"
LEGACY_POSETEMPLATECREATOR_REVISION = "450747bfee0e50b76f72ab38e1d0d04643124e02"
POSETEMPLATECREATOR_RELATIVE_PATH = Path("third_party/PoseTemplateCreator")
ADAPTER_VERSION = "posetestbot_posetemplatecreator_adapter.v2"
_PRIVATE_PACKAGE = f"_posetestbot_posetemplatecreator_{POSETEMPLATECREATOR_REVISION[:12]}"
_MODULES = ("constants", "models", "mesh", "scene", "render")
_REQUIRED_FILES = tuple(Path("backend") / f"{name}.py" for name in _MODULES)
_LOAD_LOCK = threading.RLock()
_BACKEND_LOCK = threading.RLock()
_CACHE: dict[Path, "PoseTemplateCreatorBackend"] = {}


class PoseTemplateCreatorUnavailable(RuntimeError):
    """The optional pinned source checkout cannot be used safely."""


@dataclass(frozen=True)
class PoseTemplateCreatorBackend:
    checkout: Path
    revision: str
    constants: types.ModuleType
    models: types.ModuleType
    mesh: types.ModuleType
    scene: types.ModuleType
    render: types.ModuleType

    def safe_filename(self, filename: str | None) -> str:
        return str(self.mesh.safe_filename(filename))

    def file_format(self, filename: str) -> str:
        return str(self.mesh.file_format(filename))

    def load_mesh(self, filename: str, data: bytes):
        if len(data) > int(self.constants.MAX_UPLOAD_BYTES):
            raise ValueError(
                f"CAD file exceeds the {self.constants.MAX_UPLOAD_BYTES} byte limit"
            )
        safe_name = self.safe_filename(filename)
        extension = self.file_format(safe_name)
        with _BACKEND_LOCK:
            return self.mesh._load_mesh(data, extension)

    def canonical_ply(self, filename: str, data: bytes) -> tuple[bytes, dict[str, Any]]:
        mesh = self.load_mesh(filename, data)
        exported = mesh.export(file_type="ply", encoding="binary_little_endian")
        if isinstance(exported, str):
            exported = exported.encode("utf-8")
        payload = bytes(exported)
        metadata = {
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "bounds_mm": np.asarray(mesh.bounds, dtype=float).tolist(),
            "watertight": bool(mesh.is_watertight),
        }
        return payload, metadata

    def provenance(self) -> dict[str, Any]:
        """Return the exact implementation versions behind derived geometry."""

        dependencies: dict[str, str] = {}
        for distribution in ("numpy", "scipy", "trimesh", "networkx"):
            try:
                dependencies[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                dependencies[distribution] = "unavailable"
        return {
            "adapter_version": ADAPTER_VERSION,
            "upstream_name": "PoseTemplateCreator",
            "upstream_revision": self.revision,
            "dependencies": dependencies,
        }

    def orientation_artifacts(self, filename: str, data: bytes) -> dict[str, Any]:
        """Extract deterministic stable orientations and one bounded source mesh.

        The public adapter result intentionally uses compact PoseTestBot field
        names while retaining the upstream matrices and contours without
        modification.  The source mesh is expressed in the catalogue model's
        millimetre coordinate frame; each ``source_to_placed`` rigid transform
        grounds that source mesh on the corresponding stable base.
        """

        safe_name = self.safe_filename(filename)
        source_sha256 = hashlib.sha256(data).hexdigest()
        with _BACKEND_LOCK:
            extraction = self.mesh.extract_orientations_with_preview(safe_name, data)
        orientations: list[dict[str, Any]] = []
        for footprint in extraction.orientations:
            value = footprint.model_dump(mode="json")
            if value["source_sha256"] != source_sha256:
                raise PoseTemplateCreatorUnavailable(
                    "PoseTemplateCreator returned orientation geometry for a different source"
                )
            orientations.append(
                {
                    "label": value["orientation_label"],
                    "probability": value["orientation_probability"],
                    "source_to_placed": value["source_to_placed"],
                    "slice_z_mm": value["slice_z_mm"],
                    "contours": value["contours"],
                }
            )
        return {
            "source_filename": safe_name,
            "source_sha256": source_sha256,
            "orientations": orientations,
            "preview_mesh": extraction.preview_mesh.model_dump(mode="json"),
            "provenance": self.provenance(),
        }

    # A readable alias for callers that do not persist the returned artifacts.
    analyze_orientations = orientation_artifacts

    def posed_contours(self, filename: str, data: bytes, matrix: np.ndarray) -> list[list[dict[str, float]]]:
        transform = np.asarray(matrix, dtype=float)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("Object pose must be a finite 4x4 matrix")
        mesh = self.load_mesh(filename, data).copy()
        mesh.apply_transform(transform)
        with _BACKEND_LOCK:
            contours = self.mesh._closed_contours(mesh)
        return [
            [
                {"x_mm": float(point.x_mm), "y_mm": float(point.y_mm)}
                for point in contour.points
            ]
            for contour in contours
        ]

    def build_scene(self, request: dict[str, Any]):
        validated = self.models.LayoutRequestV2.model_validate(request)
        with _BACKEND_LOCK:
            return self.scene.build_scene(validated)

    def render_pdf(self, scene: Any) -> bytes:
        with _BACKEND_LOCK:
            return bytes(self.render.render_pdf(scene))


def default_posetemplatecreator_checkout() -> Path:
    configured = os.environ.get("POSETESTBOT_APP_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        source = Path(__file__).resolve().parents[2]
        root = source if (source / "pyproject.toml").is_file() else Path.cwd()
    return root / POSETEMPLATECREATOR_RELATIVE_PATH


def _git(checkout: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", checkout.as_posix(), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PoseTemplateCreatorUnavailable(
            f"Unable to inspect PoseTemplateCreator checkout at {checkout}: {exc}"
        ) from exc
    return result.stdout.strip()


def verify_posetemplatecreator_checkout(checkout: str | Path | None = None) -> dict[str, Any]:
    root = Path(checkout or default_posetemplatecreator_checkout()).resolve()
    result: dict[str, Any] = {
        "schema_version": "posetemplatecreator_source_status.v1",
        "status": "missing",
        "available": False,
        "checkout": root.as_posix(),
        "required_revision": POSETEMPLATECREATOR_REVISION,
        "revision": None,
        "clean": None,
        "missing_files": [],
        "reason": None,
        "adapter_version": ADAPTER_VERSION,
    }
    if not root.is_dir() or not (root / ".git").exists():
        result["reason"] = (
            "PoseTemplateCreator is missing. Run 'git submodule update --init "
            "third_party/PoseTemplateCreator' or 'bash scripts/install.sh "
            "--with-posetemplatecreator'."
        )
        return result
    missing = [item.as_posix() for item in _REQUIRED_FILES if not (root / item).is_file()]
    result["missing_files"] = missing
    if missing:
        result["reason"] = "Required backend files are missing: " + ", ".join(missing)
        return result
    try:
        revision = _git(root, "rev-parse", "HEAD")
        dirty = _git(root, "status", "--porcelain", "--untracked-files=all")
    except PoseTemplateCreatorUnavailable as exc:
        result["reason"] = str(exc)
        return result
    result["revision"] = revision
    result["clean"] = not bool(dirty)
    if revision != POSETEMPLATECREATOR_REVISION:
        result["status"] = "revision_mismatch"
        result["reason"] = (
            f"PoseTemplateCreator revision mismatch: found {revision}, "
            f"required {POSETEMPLATECREATOR_REVISION}."
        )
        return result
    if dirty:
        result["status"] = "dirty"
        result["reason"] = "PoseTemplateCreator checkout has local modifications."
        return result
    result["status"] = "available"
    result["available"] = True
    return result


def _load_module(package_name: str, backend_dir: Path, name: str) -> types.ModuleType:
    full_name = f"{package_name}.{name}"
    spec = importlib.util.spec_from_file_location(full_name, backend_dir / f"{name}.py")
    if spec is None or spec.loader is None:
        raise PoseTemplateCreatorUnavailable(f"Unable to load upstream module {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def load_posetemplatecreator_backend(
    checkout: str | Path | None = None,
) -> PoseTemplateCreatorBackend:
    root = Path(checkout or default_posetemplatecreator_checkout()).resolve()
    with _LOAD_LOCK:
        status = verify_posetemplatecreator_checkout(root)
        if not status["available"]:
            _CACHE.pop(root, None)
            raise PoseTemplateCreatorUnavailable(str(status["reason"]))
        cached = _CACHE.get(root)
        if cached is not None:
            return cached
        suffix = hashlib.sha256(root.as_posix().encode()).hexdigest()[:10]
        package_name = f"{_PRIVATE_PACKAGE}_{suffix}"
        backend_dir = root / "backend"
        package = types.ModuleType(package_name)
        package.__file__ = (backend_dir / "__init__.py").as_posix()
        package.__package__ = package_name
        package.__path__ = [backend_dir.as_posix()]
        sys.modules[package_name] = package
        loaded: dict[str, types.ModuleType] = {}
        # Upstream v2 uses absolute ``backend.*`` imports. Provide those aliases
        # only while loading, then remove them so PoseTestBot never exposes or
        # collides with the upstream web application's package name.
        sentinel = object()
        prior_aliases: dict[str, object] = {"backend": sys.modules.get("backend", sentinel)}
        sys.modules["backend"] = package
        try:
            for name in _MODULES:
                loaded[name] = _load_module(package_name, backend_dir, name)
                alias = f"backend.{name}"
                prior_aliases[alias] = sys.modules.get(alias, sentinel)
                sys.modules[alias] = loaded[name]
            backend = PoseTemplateCreatorBackend(
                checkout=root,
                revision=POSETEMPLATECREATOR_REVISION,
                **loaded,
            )
            required = (
                (backend.mesh, "_load_mesh"),
                (backend.mesh, "_closed_contours"),
                (backend.mesh, "extract_orientations_with_preview"),
                (backend.scene, "build_scene"),
                (backend.render, "render_pdf"),
            )
            absent = [name for module, name in required if not hasattr(module, name)]
            if absent:
                raise PoseTemplateCreatorUnavailable(
                    "Pinned backend lacks required capabilities: " + ", ".join(absent)
                )
        except Exception as exc:
            for name in tuple(sys.modules):
                if name == package_name or name.startswith(package_name + "."):
                    sys.modules.pop(name, None)
            if isinstance(exc, PoseTemplateCreatorUnavailable):
                raise
            raise PoseTemplateCreatorUnavailable(
                f"Unable to initialize pinned PoseTemplateCreator backend: {exc}"
            ) from exc
        finally:
            for alias, previous in prior_aliases.items():
                if previous is sentinel:
                    sys.modules.pop(alias, None)
                else:
                    sys.modules[alias] = previous  # type: ignore[assignment]
        _CACHE[root] = backend
        return backend


def posetemplatecreator_status(checkout: str | Path | None = None) -> dict[str, Any]:
    status = verify_posetemplatecreator_checkout(checkout)
    if status["available"]:
        try:
            backend = load_posetemplatecreator_backend(checkout)
        except PoseTemplateCreatorUnavailable as exc:
            status.update(status="unavailable", available=False, reason=str(exc))
        else:
            status["capabilities"] = {
                "formats": list(backend.constants.SUPPORTED_FORMATS),
                "page_sizes_mm": dict(backend.constants.PAGE_SIZES_MM),
                "limits": {
                    "cad_bytes": int(backend.constants.MAX_UPLOAD_BYTES),
                    "batch_bytes": int(backend.constants.MAX_BATCH_BYTES),
                    "faces": int(backend.constants.MAX_FACES),
                    "contour_vertices": int(backend.constants.MAX_CONTOUR_VERTICES),
                    "instances": int(backend.constants.MAX_OBJECTS),
                    "orientations_per_object": int(
                        backend.constants.MAX_ORIENTATIONS
                    ),
                    "preview_vertices": int(
                        backend.constants.MAX_PREVIEW_VERTICES
                    ),
                    "preview_faces": int(backend.constants.MAX_PREVIEW_FACES),
                },
                "coordinate_convention": (
                    "millimetres; source_to_placed maps catalogue-model coordinates "
                    "to a grounded stable orientation; planar template pose is applied last"
                ),
            }
    return status
