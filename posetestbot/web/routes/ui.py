"""Bootstrap and run-discovery endpoints for the operator console."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request, send_file

from posetestbot.config import DEFAULT_ROBOT_PORT, LAB_ROBOT_IP
from posetestbot.io.artifacts import RUN_CONFIG
from posetestbot.pipeline.run_config import load_run_config_for_run_root
from posetestbot.cell.scene import build_cell_scene, cell_timeline_page
from posetestbot.pose_templates.selection import load_pose_template_selection
from posetestbot.web.security import (
    DEFAULT_RUN_ROOT,
    resolve_web_run_root,
    web_run_roots,
)


ui_bp = Blueprint("ui", __name__)


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def default_web_run_root() -> Path:
    """Return the initial run folder advertised to the browser."""

    configured = os.environ.get("POSETESTBOT_WEB_DEFAULT_RUN_ROOT")
    if configured:
        return resolve_web_run_root(configured)
    candidate = DEFAULT_RUN_ROOT / "web_run"
    try:
        return resolve_web_run_root(candidate)
    except ValueError:
        return resolve_web_run_root(web_run_roots()[0] / "web_run")


def _modified_at(path: Path) -> tuple[float, str]:
    candidates = [path]
    config_path = path / RUN_CONFIG
    if config_path.is_file() and not config_path.is_symlink():
        candidates.append(config_path)
    timestamp = max(candidate.stat().st_mtime for candidate in candidates)
    value = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return timestamp, value.isoformat().replace("+00:00", "Z")


def _run_record(path: Path) -> dict[str, Any]:
    sequence = None
    plan_only = None
    config_valid = False
    config_error = None
    try:
        config = load_run_config_for_run_root(path)
        pipeline = config.get("pipeline", {})
        if isinstance(pipeline, dict):
            raw_sequence = pipeline.get("sequence_id")
            sequence = str(raw_sequence) if raw_sequence is not None else None
            raw_plan_only = pipeline.get("plan_only")
            plan_only = raw_plan_only if isinstance(raw_plan_only, bool) else None
        config_valid = True
    except (FileNotFoundError, OSError, ValueError) as exc:
        config_error = str(exc)

    sort_timestamp, modified_at = _modified_at(path)
    return {
        "path": path.as_posix(),
        "name": path.name,
        "sequence": sequence,
        "plan_only": plan_only,
        "config_valid": config_valid,
        "config_error": config_error,
        "modified_at": modified_at,
        "_sort_timestamp": sort_timestamp,
    }


def discover_web_runs() -> list[dict[str, Any]]:
    """List direct, contained run directories without following symlinks."""

    records: dict[str, dict[str, Any]] = {}
    for allowed_root in web_run_roots():
        if not allowed_root.is_dir():
            continue
        for candidate in allowed_root.iterdir():
            try:
                if candidate.name == "calibration_targets":
                    continue
                if candidate.is_symlink() or not candidate.is_dir():
                    continue
                resolved = candidate.resolve()
                if not _is_below(resolved, allowed_root):
                    continue
                record = _run_record(resolved)
            except OSError:
                # Allowed roots such as /tmp may contain service-private folders
                # that are intentionally not traversable by the web process.
                continue
            records[resolved.as_posix()] = record

    ordered = sorted(
        records.values(),
        key=lambda item: (-item["_sort_timestamp"], item["path"]),
    )
    for item in ordered:
        item.pop("_sort_timestamp", None)
    return ordered


@ui_bp.get("/ui/bootstrap")
def ui_bootstrap():
    return jsonify(
        {
            "schema_version": "web_bootstrap.v1",
            "brand": {
                "name": "PoseTestBot",
                "logo_url": "/assets/cow_light.png",
                "logo_urls": {
                    "light": "/assets/cow_light.png",
                    "dark": "/assets/cow_dark.png",
                },
                "favicon_url": "/assets/cow_favicon.png",
            },
            "robot": {
                "ip": LAB_ROBOT_IP,
                "port": DEFAULT_ROBOT_PORT,
            },
            "default_run_root": default_web_run_root().as_posix(),
            "allowed_run_roots": [root.as_posix() for root in web_run_roots()],
        }
    )


@ui_bp.get("/ui/runs")
def ui_runs():
    return jsonify(
        {
            "schema_version": "web_run_index.v1",
            "runs": discover_web_runs(),
        }
    )


def _requested_run_root() -> Path:
    value = request.args.get("run_root")
    if not value:
        raise ValueError("run_root is required")
    return resolve_web_run_root(value)


@ui_bp.get("/ui/cell-scene")
def ui_cell_scene():
    try:
        return jsonify(build_cell_scene(_requested_run_root()))
    except FileNotFoundError as exc:
        return jsonify({"output": str(exc)}), 404
    except (OSError, ValueError) as exc:
        return jsonify({"output": str(exc)}), 400


@ui_bp.get("/ui/cell-scene/timeline")
def ui_cell_timeline():
    timeline_id = request.args.get("timeline_id")
    if not timeline_id:
        return jsonify({"output": "timeline_id is required"}), 400
    try:
        payload = cell_timeline_page(
            _requested_run_root(),
            timeline_id,
            offset=int(request.args.get("offset", "0")),
            limit=int(request.args.get("limit", str(2_000))),
        )
        return jsonify(payload)
    except KeyError as exc:
        return jsonify({"output": str(exc)}), 404
    except FileNotFoundError as exc:
        return jsonify({"output": str(exc)}), 404
    except (OSError, ValueError) as exc:
        return jsonify({"output": str(exc)}), 400


@ui_bp.get("/ui/cell-pose-template-assets/<instance_uuid>/<asset_kind>")
def ui_cell_pose_template_asset(instance_uuid: str, asset_kind: str):
    try:
        run_root = _requested_run_root()
        selection = load_pose_template_selection(run_root)
        item = next(
            (
                entry
                for entry in selection["instances"]
                if entry["instance_uuid"] == instance_uuid
            ),
            None,
        )
        if item is None:
            return jsonify({"output": "Unknown selected template instance"}), 404
        key = "canonical_ply" if asset_kind == "mesh" else "texture"
        if key not in item["assets"]:
            return jsonify({"output": "Unknown or unavailable instance asset"}), 404
        snapshot = run_root / selection["bundle_snapshot"]
        path = snapshot / item["assets"][key]["path"]
        path.resolve(strict=True).relative_to(snapshot.resolve())
        return send_file(
            path,
            mimetype="image/png" if key == "texture" else "application/octet-stream",
            conditional=True,
            max_age=3600,
        )
    except FileNotFoundError as exc:
        return jsonify({"output": str(exc)}), 404
    except (OSError, ValueError) as exc:
        return jsonify({"output": str(exc)}), 400
