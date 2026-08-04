"""Lazy, source-checkout-only access to the pinned PoseGridGen renderer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


POSEGRIDGEN_REVISION = "9e6975901fe096bf65f7b7b599d7b82461d2e67c"
POSEGRIDGEN_COMPATIBLE_BUNDLE_REVISIONS = frozenset(
    {
        "ad152e369e8d2746d0cf66cb1455f2371b0ec0f0",
        POSEGRIDGEN_REVISION,
    }
)
POSEGRIDGEN_RELATIVE_PATH = Path("third_party/PoseGridGen")
_PRIVATE_PACKAGE = f"_posetestbot_posegridgen_{POSEGRIDGEN_REVISION[:12]}"
_BACKEND_MODULES = ("models", "errors", "fit", "scene", "render")
_REQUIRED_FILES = tuple(Path("backend") / f"{name}.py" for name in _BACKEND_MODULES)
_RENDER_LOCK = threading.RLock()
_LOAD_LOCK = threading.RLock()
_CACHE: dict[Path, "PoseGridGenBackend"] = {}


class PoseGridGenUnavailable(RuntimeError):
    """The optional pinned generator checkout cannot be used safely."""


@dataclass(frozen=True)
class PoseGridGenBackend:
    checkout: Path
    revision: str
    models: types.ModuleType
    errors: types.ModuleType
    fit: types.ModuleType
    scene: types.ModuleType
    render: types.ModuleType

    def validate_request(self, value: Mapping[str, Any]):
        return self.models.GenerateRequest.model_validate(dict(value))

    def canonical_request(self, request: Any) -> tuple[str, bytes]:
        raw = json.dumps(
            request.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(raw).hexdigest(), raw


def default_posegridgen_checkout() -> Path:
    configured = os.environ.get("POSETESTBOT_APP_ROOT")
    if configured:
        app_root = Path(configured).expanduser().resolve()
    else:
        source_root = Path(__file__).resolve().parents[2]
        app_root = source_root if (source_root / "pyproject.toml").is_file() else Path.cwd()
    return app_root / POSEGRIDGEN_RELATIVE_PATH


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
        raise PoseGridGenUnavailable(
            f"Unable to inspect PoseGridGen checkout at {checkout}: {exc}"
        ) from exc
    return result.stdout.strip()


def verify_posegridgen_checkout(
    checkout: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(checkout or default_posegridgen_checkout()).resolve()
    status: dict[str, Any] = {
        "available": False,
        "source_checkout": False,
        "checkout": root.as_posix(),
        "required_revision": POSEGRIDGEN_REVISION,
        "revision": None,
        "clean": None,
        "missing_files": [],
        "reason": None,
    }
    if not root.is_dir() or not (root / ".git").exists():
        status["reason"] = (
            "PoseGridGen source checkout is missing. Initialize it with "
            "'git submodule update --init third_party/PoseGridGen' or run "
            "'bash scripts/install.sh --with-posegridgen'."
        )
        return status
    status["source_checkout"] = True
    missing = [path.as_posix() for path in _REQUIRED_FILES if not (root / path).is_file()]
    status["missing_files"] = missing
    if missing:
        status["reason"] = "PoseGridGen backend files are missing: " + ", ".join(missing)
        return status
    try:
        revision = _git(root, "rev-parse", "HEAD")
        dirty = _git(root, "status", "--porcelain", "--untracked-files=all")
    except PoseGridGenUnavailable as exc:
        status["reason"] = str(exc)
        return status
    status["revision"] = revision
    status["clean"] = not bool(dirty)
    if revision != POSEGRIDGEN_REVISION:
        status["reason"] = (
            f"PoseGridGen revision mismatch: found {revision}, "
            f"required {POSEGRIDGEN_REVISION}."
        )
        return status
    if dirty:
        status["reason"] = "PoseGridGen checkout has local modifications or untracked files."
        return status
    status["available"] = True
    return status


def _load_module(package_name: str, backend_dir: Path, name: str) -> types.ModuleType:
    full_name = f"{package_name}.{name}"
    path = backend_dir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(full_name, path)
    if spec is None or spec.loader is None:
        raise PoseGridGenUnavailable(f"Unable to load PoseGridGen backend module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(full_name, None)
        raise
    return module


def _verify_capabilities(backend: PoseGridGenBackend) -> None:
    import cv2
    import numpy as np

    required = (
        (backend.models, "GenerateRequest"),
        (backend.errors, "FitError"),
        (backend.fit, "fit_request"),
        (backend.scene, "build_scene"),
        (backend.render, "render_png"),
        (backend.render, "render_pdf"),
        (backend.render, "manifest"),
    )
    missing = [f"{module.__name__}.{name}" for module, name in required if not hasattr(module, name)]
    if missing:
        raise PoseGridGenUnavailable(
            "Pinned PoseGridGen renderer is missing required capabilities: " + ", ".join(missing)
        )
    request = backend.models.GenerateRequest()
    scene = backend.scene.build_scene(request)
    if request.board.type != "aruco" or not scene.features:
        raise PoseGridGenUnavailable("Pinned PoseGridGen renderer defaults are incompatible.")
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        cv2.aruco.Board(
            [np.zeros((4, 3), dtype=np.float32)],
            dictionary,
            np.asarray([0], dtype=np.int32),
        )
    except (AttributeError, cv2.error) as exc:
        raise PoseGridGenUnavailable(
            "Installed opencv-python lacks the required cv2.aruco.Board APIs."
        ) from exc


def load_posegridgen_backend(
    checkout: str | Path | None = None,
) -> PoseGridGenBackend:
    root = Path(checkout or default_posegridgen_checkout()).resolve()
    with _LOAD_LOCK:
        cached = _CACHE.get(root)
        if cached is not None:
            status = verify_posegridgen_checkout(root)
            if status["available"]:
                return cached
            _CACHE.pop(root, None)
        status = verify_posegridgen_checkout(root)
        if not status["available"]:
            raise PoseGridGenUnavailable(str(status["reason"]))

        package_name = _PRIVATE_PACKAGE
        if root != default_posegridgen_checkout().resolve():
            package_name = f"{_PRIVATE_PACKAGE}_{hashlib.sha256(root.as_posix().encode()).hexdigest()[:12]}"
        backend_dir = root / "backend"
        package = types.ModuleType(package_name)
        package.__file__ = (backend_dir / "__init__.py").as_posix()
        package.__package__ = package_name
        package.__path__ = [backend_dir.as_posix()]
        sys.modules[package_name] = package
        loaded: dict[str, types.ModuleType] = {}
        try:
            for name in ("models", "errors", "scene", "fit", "render"):
                loaded[name] = _load_module(package_name, backend_dir, name)
            backend = PoseGridGenBackend(
                checkout=root,
                revision=POSEGRIDGEN_REVISION,
                **loaded,
            )
            _verify_capabilities(backend)
        except Exception as exc:
            for name in tuple(sys.modules):
                if name == package_name or name.startswith(f"{package_name}."):
                    sys.modules.pop(name, None)
            if isinstance(exc, PoseGridGenUnavailable):
                raise
            raise PoseGridGenUnavailable(
                f"Unable to initialize pinned PoseGridGen backend: {exc}"
            ) from exc
        _CACHE[root] = backend
        return backend


def posegridgen_status(checkout: str | Path | None = None) -> dict[str, Any]:
    status = verify_posegridgen_checkout(checkout)
    if not status["available"]:
        status["renderer_compatible"] = False
        return status
    try:
        load_posegridgen_backend(checkout)
    except PoseGridGenUnavailable as exc:
        status["available"] = False
        status["renderer_compatible"] = False
        status["reason"] = str(exc)
    else:
        status["renderer_compatible"] = True
    return status


def posegridgen_capabilities(checkout: str | Path | None = None) -> dict[str, Any]:
    backend = load_posegridgen_backend(checkout)
    constants = sys.modules[f"{backend.models.__package__}.constants"]
    defaults = backend.models.GenerateRequest()
    return {
        "schema_version": "posegridgen_capabilities.v1",
        "generator_schema_version": "2.0",
        "generator_revision": backend.revision,
        "paper_sizes_mm": constants.PAPER_SIZES_MM,
        "dictionaries": constants.DICTIONARIES,
        "limits": {
            "grid": 100,
            "physical_mm": 200,
            "preview_max_pixels": 1600,
            "page_edge_clearance_mm": constants.EDGE_CLEARANCE_MM,
        },
        "board_types": ["aruco"],
        "defaults": defaults.model_dump(mode="json"),
        "board_defaults": {"aruco": defaults.board.model_dump(mode="json")},
    }


def build_posegridgen_scene(
    value: Mapping[str, Any], checkout: str | Path | None = None
) -> tuple[PoseGridGenBackend, str, Any]:
    backend = load_posegridgen_backend(checkout)
    request = backend.validate_request(value)
    configuration_hash, _raw = backend.canonical_request(request)
    with _RENDER_LOCK:
        scene = backend.scene.build_scene(request)
    return backend, configuration_hash, scene


def fit_posegridgen_request(
    value: Mapping[str, Any], checkout: str | Path | None = None
) -> dict[str, Any]:
    backend = load_posegridgen_backend(checkout)
    request = backend.validate_request(value)
    with _RENDER_LOCK:
        result = backend.fit.fit_request(request)
    return result.model_dump(mode="json")


def render_posegridgen_preview(
    value: Mapping[str, Any], checkout: str | Path | None = None
) -> tuple[bytes, str]:
    backend, configuration_hash, scene = build_posegridgen_scene(value, checkout)
    with _RENDER_LOCK:
        payload = backend.render.render_png(scene)
    return payload, configuration_hash


def render_posegridgen_bundle(
    value: Mapping[str, Any], checkout: str | Path | None = None
) -> tuple[bytes, bytes, str]:
    """Build one scene and derive its canonical source manifest and PDF."""

    backend = load_posegridgen_backend(checkout)
    request = backend.validate_request(value)
    configuration_hash, _raw = backend.canonical_request(request)
    with _RENDER_LOCK:
        scene = backend.scene.build_scene(request)
        source = backend.render.manifest(scene, configuration_hash)
        pdf = backend.render.render_pdf(scene)
    return source, pdf, configuration_hash
