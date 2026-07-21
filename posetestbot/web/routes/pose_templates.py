"""Managed object, immutable template, and per-run Ground Truth APIs."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request, send_file

from posetestbot.io.atomic import atomic_write_json
from posetestbot.jobs.runner import ResourceBusyError
from posetestbot.pose_templates.adapter import (
    PoseTemplateCreatorUnavailable,
    load_posetemplatecreator_backend,
    posetemplatecreator_status,
)
from posetestbot.pose_templates.catalog import (
    default_catalog_root,
    default_working_data_root,
    get_catalog_object,
    load_catalog,
    set_catalog_object_state,
)
from posetestbot.pose_templates.library import (
    BUNDLE_MANIFEST,
    TEMPLATE_PDF,
    clone_template_configuration,
    default_template_library_root,
    list_template_bundles,
    set_template_archive_state,
    validate_template_bundle,
)
from posetestbot.pose_templates.selection import (
    PoseTemplateSelectionConflict,
    load_pose_template_selection,
    replacement_blockers,
)
from posetestbot.web.legacy import job_runner
from posetestbot.web.paths import APP_ROOT
from posetestbot.web.security import resolve_web_run_root


pose_templates_bp = Blueprint("pose_templates", __name__)
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_UPLOAD_BATCH_BYTES = 100 * 1024 * 1024
REQUEST_ROOT = default_working_data_root() / "jobs" / "pose_template_requests"


def _json() -> dict[str, Any]:
    if request.content_length is not None and request.content_length > MAX_JSON_BYTES:
        raise ValueError("Pose-template JSON request exceeds 2 MiB")
    value = request.get_json(silent=True)
    if not isinstance(value, dict):
        raise ValueError("A JSON object is required")
    return value


def _write_request(kind: str, value: dict[str, Any]) -> tuple[str, Path]:
    request_id = uuid.uuid4().hex
    folder = REQUEST_ROOT / kind / request_id
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / "request.json"
    atomic_write_json(path, value)
    return request_id, path


def _submit(
    *, name: str, script: str, request_path: Path, request_id: str, resources: list[str]
):
    job = job_runner.submit(
        name=name,
        command=["uv", "run", "python", script, "--request", request_path.as_posix()],
        cwd=APP_ROOT,
        resources=resources,
        parameters={"request_id": request_id, "request_path": request_path.as_posix()},
    )
    return jsonify({"job": job.to_dict(), "job_id": job.id, "request_id": request_id}), 202


def _error(exc: Exception):
    if isinstance(exc, PoseTemplateSelectionConflict):
        return jsonify({"output": str(exc), "blockers": exc.blockers}), 409
    if isinstance(exc, ResourceBusyError):
        return jsonify({"output": str(exc)}), 409
    if isinstance(exc, (KeyError, FileNotFoundError)):
        return jsonify({"output": str(exc)}), 404
    if isinstance(exc, PoseTemplateCreatorUnavailable):
        return jsonify({"output": str(exc)}), 409
    code = getattr(exc, "code", None)
    if code:
        return jsonify({"errors": [{"code": code, "message": getattr(exc, "message", str(exc))}]}), 422
    return jsonify({"output": str(exc)}), 400


@pose_templates_bp.get("/pose-templates/status")
def source_status():
    return jsonify(posetemplatecreator_status())


@pose_templates_bp.get("/pose-templates/catalog")
def catalog_list():
    try:
        return jsonify(load_catalog())
    except (OSError, ValueError) as exc:
        return _error(exc)


@pose_templates_bp.get("/pose-templates/catalog/<catalog_uuid>")
def catalog_detail(catalog_uuid: str):
    try:
        return jsonify(get_catalog_object(catalog_uuid))
    except (KeyError, OSError, ValueError) as exc:
        return _error(exc)


@pose_templates_bp.post("/pose-templates/catalog/upload")
def catalog_upload():
    folder: Path | None = None
    try:
        if request.content_length is not None and request.content_length > MAX_UPLOAD_BATCH_BYTES + 1024 * 1024:
            raise ValueError("Upload exceeds the 100 MiB batch limit")
        cad = request.files.get("cad")
        textures = request.files.getlist("texture")
        if cad is None or not cad.filename:
            raise ValueError("One CAD file is required")
        if len(textures) > 1:
            raise ValueError("At most one PNG texture is supported")
        backend = load_posetemplatecreator_backend()
        safe_name = backend.safe_filename(cad.filename)
        backend.file_format(safe_name)
        request_id = uuid.uuid4().hex
        folder = REQUEST_ROOT / "catalog_upload" / request_id
        folder.mkdir(parents=True, exist_ok=False)
        cad_path = folder / safe_name
        cad.save(cad_path)
        if cad_path.stat().st_size > int(backend.constants.MAX_UPLOAD_BYTES):
            raise ValueError("CAD upload exceeds the 50 MiB file limit")
        texture_path = None
        if textures and textures[0].filename:
            texture_name = backend.safe_filename(textures[0].filename)
            if Path(texture_name).suffix.lower() != ".png":
                raise ValueError("Texture must be PNG")
            texture_path = folder / "texture.png"
            textures[0].save(texture_path)
        total = cad_path.stat().st_size + (texture_path.stat().st_size if texture_path else 0)
        if total > MAX_UPLOAD_BATCH_BYTES:
            raise ValueError("Upload exceeds the 100 MiB batch limit")
        value = {
            "name": request.form.get("name") or Path(safe_name).stem,
            "description": request.form.get("description") or None,
            "cad_path": cad_path.as_posix(),
            "texture_path": texture_path.as_posix() if texture_path else None,
            "catalog_root": default_catalog_root().as_posix(),
        }
        request_path = folder / "request.json"
        atomic_write_json(request_path, value)
        return _submit(
            name="object_catalog_import",
            script="scripts/run_object_catalog_import.py",
            request_path=request_path,
            request_id=request_id,
            resources=["cpu", "disk_io"],
        )
    except Exception as exc:
        if folder is not None:
            shutil.rmtree(folder, ignore_errors=True)
        return _error(exc)


@pose_templates_bp.post("/pose-templates/catalog/<catalog_uuid>/<action>")
def catalog_state(catalog_uuid: str, action: str):
    try:
        if action not in {"archive", "restore"}:
            raise KeyError("Unknown catalog action")
        return jsonify(
            set_catalog_object_state(
                catalog_uuid, state="archived" if action == "archive" else "active"
            )
        )
    except Exception as exc:
        return _error(exc)


@pose_templates_bp.get("/pose-templates/catalog/<catalog_uuid>/assets/<kind>")
def catalog_asset(catalog_uuid: str, kind: str):
    try:
        item = get_catalog_object(catalog_uuid)
        if kind not in item["assets"]:
            raise KeyError("Unknown catalog asset")
        record = item["assets"][kind]
        path = Path(item["catalog_root"]) / record["path"]
        return send_file(
            path,
            mimetype=record["media_type"],
            as_attachment=True,
            download_name=path.name,
            conditional=True,
        )
    except Exception as exc:
        return _error(exc)


@pose_templates_bp.post("/pose-templates/preview")
@pose_templates_bp.post("/pose-templates/validate")
def preview():
    try:
        value = _json()
        configuration = value.get("configuration")
        if not isinstance(configuration, dict):
            raise ValueError("configuration must be an object")
        request_kind = "validate" if request.path.endswith("/validate") else "preview"
        request_id, request_path = _write_request(request_kind, {"configuration": configuration})
        output = request_path.parent / "preview.json"
        job = job_runner.submit(
            name=("pose_template_validation" if request_kind == "validate" else "pose_template_preview"),
            command=[
                "uv", "run", "python", "scripts/run_pose_template_preview.py",
                "--request", request_path.as_posix(), "--output", output.as_posix(),
            ],
            cwd=APP_ROOT,
            resources=["cpu", "disk_io"],
            parameters={"request_id": request_id, "result": output.as_posix()},
        )
        return jsonify({"job": job.to_dict(), "job_id": job.id, "request_id": request_id}), 202
    except Exception as exc:
        return _error(exc)


@pose_templates_bp.get("/pose-templates/preview/<request_id>")
@pose_templates_bp.get("/pose-templates/validate/<request_id>")
def preview_result(request_id: str):
    try:
        if not request_id.isalnum() or len(request_id) != 32:
            raise ValueError("Invalid preview request ID")
        request_kind = "validate" if "/validate/" in request.path else "preview"
        path = REQUEST_ROOT / request_kind / request_id / "preview.json"
        with open(path, "r", encoding="utf-8") as handle:
            return jsonify(json.load(handle))
    except Exception as exc:
        return _error(exc)


@pose_templates_bp.post("/pose-templates/generate")
def generate():
    try:
        value = _json()
        configuration = value.get("configuration")
        if not isinstance(configuration, dict):
            raise ValueError("configuration must be an object")
        request_value = {"configuration": configuration, "cloned_from": value.get("cloned_from")}
        request_id, request_path = _write_request("generate", request_value)
        return _submit(
            name="pose_template_generate",
            script="scripts/run_pose_template_generate.py",
            request_path=request_path,
            request_id=request_id,
            resources=["cpu", "disk_io"],
        )
    except Exception as exc:
        return _error(exc)


@pose_templates_bp.get("/pose-templates/library")
def library_list():
    return jsonify(
        {"schema_version": "pose_template_library.v1", "templates": list_template_bundles()}
    )


@pose_templates_bp.get("/pose-templates/library/<template_uuid>")
def library_detail(template_uuid: str):
    try:
        return jsonify(
            validate_template_bundle(
                default_template_library_root() / template_uuid,
                library_root=default_template_library_root(),
            )
        )
    except Exception as exc:
        return _error(exc)


@pose_templates_bp.post("/pose-templates/library/<template_uuid>/<action>")
def library_action(template_uuid: str, action: str):
    try:
        if action in {"archive", "restore"}:
            return jsonify(
                set_template_archive_state(
                    template_uuid, state="archived" if action == "archive" else "active"
                )
            )
        if action == "clone":
            configuration = clone_template_configuration(template_uuid)
            value = _json() if request.content_length else {}
            if value.get("display_name"):
                configuration["display_name"] = str(value["display_name"])
            request_id, request_path = _write_request(
                "generate", {"configuration": configuration, "cloned_from": template_uuid}
            )
            return _submit(
                name="pose_template_clone",
                script="scripts/run_pose_template_generate.py",
                request_path=request_path,
                request_id=request_id,
                resources=["cpu", "disk_io"],
            )
        raise KeyError("Unknown template action")
    except Exception as exc:
        return _error(exc)


@pose_templates_bp.get("/pose-templates/library/<template_uuid>/download/<kind>")
def library_download(template_uuid: str, kind: str):
    try:
        bundle = validate_template_bundle(
            default_template_library_root() / template_uuid,
            library_root=default_template_library_root(),
        )
        names = {"pdf": TEMPLATE_PDF, "manifest": BUNDLE_MANIFEST}
        if kind not in names:
            raise KeyError("Unknown template download")
        path = Path(bundle["bundle_path"]) / names[kind]
        return send_file(path, as_attachment=True, download_name=path.name, conditional=True)
    except Exception as exc:
        return _error(exc)


@pose_templates_bp.get("/pose-templates/runs/selection")
def run_selection_detail():
    try:
        run_root = resolve_web_run_root(request.args.get("run_root"))
        try:
            selection = load_pose_template_selection(run_root)
        except FileNotFoundError:
            selection = None
        return jsonify(
            {
                "schema_version": "pose_template_run_status.v1",
                "run_root": run_root.as_posix(),
                "selection": selection,
                "replacement_blockers": replacement_blockers(run_root),
                "ready": bool(selection and selection.get("placement_confirmed")),
            }
        )
    except Exception as exc:
        return _error(exc)


@pose_templates_bp.post("/pose-templates/runs/selection")
@pose_templates_bp.post("/pose-templates/runs/placement")
def run_selection_update():
    try:
        value = _json()
        run_root = resolve_web_run_root(value.get("run_root"))
        placement = value.get("placement")
        if not isinstance(placement, dict):
            raise ValueError("placement must be a transform object")
        request_value = {
            "run_root": run_root.as_posix(),
            "template_uuid": value.get("template_uuid"),
            "placement": placement,
            "confirmed": bool(value.get("confirmed", False)),
            "operator": value.get("operator"),
        }
        request_id, request_path = _write_request("select", request_value)
        return _submit(
            name="pose_template_select",
            script="scripts/run_pose_template_select.py",
            request_path=request_path,
            request_id=request_id,
            resources=["disk_io"],
        )
    except Exception as exc:
        return _error(exc)
