"""Thin browser-safe proxy for the external cluster controller."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

from flask import Blueprint, jsonify, request

from posetestbot.bop.evaluation import (
    import_external_bop_result,
    inspect_dataset,
    public_dataset_descriptor,
)
from posetestbot.cluster.client import ClusterClientError, new_idempotency_key
from posetestbot.jobs.runner import ResourceBusyError, TERMINAL_STATUSES
from posetestbot.run_folders import (
    resolve_destination_root,
    resolve_direct_run_folder,
    validate_expected_identity,
)
from posetestbot.web.runtime import get_cluster_client, get_job_runner, get_web_runtime
from posetestbot.web.security import resolve_web_run_root, web_run_roots


cluster_bp = Blueprint("cluster", __name__)
CONTROLLER_ID_RE = re.compile(
    r"^(?:archive|job|pose|restore)-[0-9a-f]{8}-[0-9a-f]{4}-"
    r"[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
SUCCESS_STATES = {"succeeded", "succeeded-with-warning"}
PUBLIC_RUNTIME_FIELDS = {
    "runtime_id",
    "foundationpose_revision",
    "bop_toolkit_revision",
    "sif_sha256",
    "weights_sha256",
    "foundationpose_license",
    "foundationpose_license_sha256",
    "qualified",
}
PUBLIC_PROFILE_FIELDS = {
    "profile_id",
    "enabled",
    "partition",
    "gres",
    "cpus",
    "memory",
    "walltime",
    "max_targets",
}
CONTROLLER_PATH_RE = re.compile(r"(?<![A-Za-z0-9_.-])/(?:[^\s'\"<>|,;)\]}]+)")
CONTROLLER_BEARER_RE = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
CONTROLLER_SECRET_LINE_RE = re.compile(
    r"(?im)^.*\b(?:authorization|api[_ -]?token|password|private key|secret)\b\s*[:=].*$"
)


def _json_object() -> dict[str, Any]:
    value = request.get_json(silent=True)
    if not isinstance(value, dict):
        raise ValueError("A JSON object is required")
    return value


def _require_id(value: Any, *, prefix: str | None = None) -> str:
    if not isinstance(value, str) or CONTROLLER_ID_RE.fullmatch(value) is None:
        raise ValueError("Controller identifier is invalid")
    if prefix is not None and not value.startswith(f"{prefix}-"):
        raise ValueError("Controller identifier has the wrong kind")
    return value


def _error(exc: Exception):
    if isinstance(exc, ClusterClientError):
        return jsonify({"output": _public_controller_text(exc)}), exc.status
    if isinstance(exc, ResourceBusyError | FileExistsError | RuntimeError):
        return jsonify({"output": str(exc)}), 409
    if isinstance(exc, FileNotFoundError | KeyError):
        return jsonify({"output": str(exc)}), 404
    if isinstance(exc, PermissionError):
        return jsonify({"output": str(exc)}), 403
    return jsonify({"output": str(exc)}), 400


def _settings():
    return get_web_runtime().settings


def _require_cluster_enabled() -> None:
    if not _settings().cluster_enabled:
        raise PermissionError("Cluster integration is disabled")


def _public_controller_text(value: Any) -> str | None:
    if value is None:
        return None
    text = CONTROLLER_SECRET_LINE_RE.sub("[redacted controller detail]", str(value))
    text = CONTROLLER_BEARER_RE.sub("Bearer [redacted]", text)
    return CONTROLLER_PATH_RE.sub("[controller path]", text)


def _selected_mapping(value: Any, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {field: value[field] for field in fields if field in value}


def _public_job(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("The controller returned an invalid job")
    job_id = _require_id(value.get("job_id"))
    payload = _selected_mapping(
        value.get("payload"),
        {"run_root", "dataset_alias", "dataset_sha256", "profile_id", "operator"},
    )
    result = _selected_mapping(
        value.get("result"),
        {
            "filename",
            "sha256",
            "dataset_sha256",
            "estimate_count",
            "failure_count",
        },
    )
    return {
        "schema_version": "posetestbot_cluster_job.v1",
        "job_id": job_id,
        "kind": value.get("kind"),
        "state": value.get("state"),
        "status": value.get("status", value.get("state")),
        "created_at": value.get("created_at"),
        "updated_at": value.get("updated_at"),
        "slurm_job_id": value.get("slurm_job_id"),
        "payload": payload,
        "result": result or None,
        "error": _public_controller_text(value.get("error")),
        "log_available": value.get("log_available") is True,
        "cancel_requested": value.get("cancel_requested") is True,
        "terminal": value.get("terminal") is True,
    }


def _public_job_response(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("The controller returned an invalid response")
    return {"job": _public_job(value.get("job"))}


def _public_archive(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("The controller returned an invalid archive")
    return {
        "schema_version": "posetestbot_cluster_archive.v1",
        "archive_id": _require_id(value.get("archive_id"), prefix="archive"),
        "job_id": value.get("job_id"),
        "state": value.get("state"),
        "status": value.get("status", value.get("state")),
        "source_run_root": value.get("source_run_root"),
        "source_identity": _selected_mapping(
            value.get("source_identity"), {"device", "inode"}
        ),
        "created_at": value.get("created_at"),
        "updated_at": value.get("updated_at"),
        "archive_sha256": value.get("archive_sha256"),
        "operator": value.get("operator"),
        "verified": value.get("verified") is True,
    }


def _controller_status() -> dict[str, Any]:
    settings = _settings()
    integration = {
        "enabled": settings.cluster_enabled,
        "controller_configured": get_web_runtime().cluster_client is not None,
    }
    if not settings.cluster_enabled:
        return {
            "schema_version": "posetestbot_cluster_status_proxy.v1",
            "ready": False,
            "available": False,
            "integration": integration,
            "blockers": [
                {
                    "code": "cluster_disabled",
                    "message": "Cluster integration is disabled on this workstation.",
                }
            ],
        }
    if get_web_runtime().cluster_client is None:
        return {
            "schema_version": "posetestbot_cluster_status_proxy.v1",
            "ready": False,
            "available": False,
            "integration": integration,
            "blockers": [
                {
                    "code": "controller_not_configured",
                    "message": "The loopback cluster controller token is not configured.",
                }
            ],
        }
    try:
        status = get_cluster_client().status()
    except ClusterClientError as exc:
        return {
            "schema_version": "posetestbot_cluster_status_proxy.v1",
            "ready": False,
            "available": False,
            "integration": integration,
            "blockers": [{"code": "controller_unavailable", "message": str(exc)}],
        }
    blockers = []
    for index, item in enumerate(status.get("blockers") or []):
        if isinstance(item, Mapping):
            blockers.append(
                {
                    "code": str(item.get("code") or f"controller_blocker_{index + 1}"),
                    "message": _public_controller_text(
                        item.get("message") or "Cluster controller is not ready."
                    ),
                }
            )
        else:
            blockers.append(
                {
                    "code": f"controller_blocker_{index + 1}",
                    "message": _public_controller_text(item),
                }
            )
    profiles = [
        _selected_mapping(item, PUBLIC_PROFILE_FIELDS)
        for item in status.get("profiles", [])
        if isinstance(item, Mapping)
    ]
    runtime = _selected_mapping(status.get("runtime"), PUBLIC_RUNTIME_FIELDS)
    features = _selected_mapping(
        status.get("features"), {"pose_estimation", "archive_read", "archive_mutation"}
    )
    raw_feature_blockers = status.get("feature_blockers")
    feature_blockers = {
        key: [
            _public_controller_text(message) or "Cluster feature is not ready."
            for message in messages
        ]
        for key, messages in (
            raw_feature_blockers.items()
            if isinstance(raw_feature_blockers, Mapping)
            else []
        )
        if key in {"estimation", "archive"} and isinstance(messages, list)
    }
    return {
        "schema_version": "posetestbot_cluster_status_proxy.v1",
        "ready": status.get("ready") is True,
        "available": True,
        "mode": status.get("mode"),
        "features": features,
        "feature_blockers": feature_blockers,
        "runtime": runtime,
        "profiles": profiles,
        "integration": integration,
        "blockers": blockers,
    }


def _load_bop_manifest(run_root: Path) -> Mapping[str, Any]:
    path = run_root / "bop" / "bop_export_manifest.json"
    if path.is_symlink() or not path.is_file():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError("BOP export manifest must be a JSON object")
    return value


def _build_pose_setup(run_root: Path) -> dict[str, Any]:
    # Readiness stays request-bounded: the companion hashes every staged file
    # in its background worker before submission, while this identity binds the
    # existing BOP metadata and semantic content without synchronously reading
    # every depth image in a Flask request.
    dataset = inspect_dataset(run_root)
    manifest = _load_bop_manifest(run_root)
    status = _controller_status()
    blockers = [
        {"code": f"dataset_{index + 1}", "message": str(message)}
        for index, message in enumerate(dataset.get("blockers", []))
    ]
    if manifest.get("annotation_mode") != "pose_and_masks":
        blockers.append(
            {
                "code": "pose_and_masks_required",
                "message": (
                    "FoundationPose v1 requires a complete BOP v5 pose_and_masks "
                    "export with visible GT instance masks."
                ),
            }
        )
    if dataset.get("split") != "test":
        blockers.append(
            {
                "code": "test_split_required",
                "message": "FoundationPose v1 requires the exported test split.",
            }
        )
    capabilities = manifest.get("capabilities")
    if (
        not isinstance(capabilities, Mapping)
        or capabilities.get("gt_masks_visible") is not True
    ):
        blockers.append(
            {
                "code": "visible_masks_missing",
                "message": "The BOP export does not declare complete visible GT masks.",
            }
        )
    if not status.get("available") or not status.get("ready"):
        blockers.extend(status.get("blockers") or [])
        if status.get("available") and not status.get("ready"):
            blockers.append(
                {
                    "code": "cluster_connection_not_ready",
                    "message": "The controller cannot currently verify both LUIS hosts.",
                }
            )
    features = (
        status.get("features") if isinstance(status.get("features"), Mapping) else {}
    )
    if status.get("available") and features.get("pose_estimation") is not True:
        remote = status.get("feature_blockers")
        messages = remote.get("estimation", []) if isinstance(remote, Mapping) else []
        blockers.extend(
            {"code": "controller_estimation_blocked", "message": str(message)}
            for message in messages
        )
    if not _settings().cluster_enabled:
        blockers.append(
            {
                "code": "cluster_disabled",
                "message": "Pose-estimation submission is disabled on this workstation.",
            }
        )
    profiles = (
        status.get("profiles") if isinstance(status.get("profiles"), list) else []
    )
    enabled_profiles = [
        profile
        for profile in profiles
        if isinstance(profile, Mapping) and profile.get("enabled") is True
    ]
    if not enabled_profiles:
        blockers.append(
            {
                "code": "no_qualified_profile",
                "message": "No server-owned GPU resource profile is qualified and enabled.",
            }
        )
    unique_blockers = list(
        {
            (str(item.get("code")), str(item.get("message"))): {
                "code": str(item.get("code")),
                "message": str(item.get("message")),
            }
            for item in blockers
            if isinstance(item, Mapping)
        }.values()
    )
    return {
        "schema_version": "cluster_pose_estimation_setup.v1",
        "run_root": run_root.as_posix(),
        "dataset": public_dataset_descriptor(dataset),
        "annotation_mode": manifest.get("annotation_mode"),
        "oracle_mask_contract": "bop_mask_visib_gt_instance.v1",
        "score_contract": "constant_1.0_no_detection_confidence",
        "execution_contract": "independent_register_per_target_no_tracking.v1",
        "controller": status,
        "runtime": status.get("runtime") if status.get("available") else None,
        "profiles": profiles,
        "enabled_profiles": enabled_profiles,
        "ready": not unique_blockers,
        "blockers": unique_blockers,
        "warnings": [
            {
                "code": "oracle_gt_masks",
                "message": (
                    "Every estimate is conditioned on a BOP GT-visible instance mask; "
                    "this is pose estimation, not detection or segmentation."
                ),
            }
        ],
    }


def _all_local_jobs():
    return get_job_runner().list(include_services=True)


def _assert_no_active_run_jobs(run_root: Path) -> None:
    active: list[str] = []
    for job in _all_local_jobs():
        if (
            job.status in TERMINAL_STATUSES
            or job.scope_kind != "run"
            or not job.run_root
        ):
            continue
        try:
            same = Path(job.run_root).resolve() == run_root.resolve()
        except OSError:
            same = job.run_root == run_root.as_posix()
        if same:
            active.append(job.id)
    if active:
        raise ResourceBusyError(
            "Run folder has active background work: " + ", ".join(sorted(active))
        )


@cluster_bp.get("/cluster/status")
def cluster_status():
    return jsonify(_controller_status())


@cluster_bp.get("/cluster/pose-estimation/setup")
def cluster_pose_setup():
    try:
        run_root = resolve_web_run_root(request.args.get("run_root"))
        return jsonify(_build_pose_setup(run_root))
    except Exception as exc:
        return _error(exc)


@cluster_bp.post("/cluster/pose-estimation/jobs")
def submit_cluster_pose_job():
    try:
        _require_cluster_enabled()
        value = _json_object()
        run_root = resolve_web_run_root(value.get("run_root"))
        setup = _build_pose_setup(run_root)
        if not setup["ready"]:
            raise RuntimeError(
                "Pose estimation is blocked: "
                + " ".join(item["message"] for item in setup["blockers"])
            )
        profile_id = value.get("profile_id")
        enabled_ids = {
            item.get("profile_id")
            for item in setup["enabled_profiles"]
            if isinstance(item, Mapping)
        }
        if profile_id not in enabled_ids:
            raise ValueError("Selected resource profile is not enabled")
        operator = value.get("operator")
        if not isinstance(operator, str) or not operator.strip():
            raise ValueError("operator is required")
        dataset = inspect_dataset(run_root)
        response = get_cluster_client().create_pose_job(
            {
                "run_root": run_root.as_posix(),
                "dataset_alias": dataset["dataset_alias"],
                "dataset_sha256": dataset["dataset_sha256"],
                "profile_id": profile_id,
                "operator": operator.strip(),
            },
            idempotency_key=new_idempotency_key("pose-submit"),
        )
        return jsonify(_public_job_response(response)), 202
    except Exception as exc:
        return _error(exc)


@cluster_bp.get("/cluster/jobs")
def list_cluster_jobs():
    try:
        _require_cluster_enabled()
        limit = request.args.get("limit", default=50, type=int)
        if limit is None or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        response = get_cluster_client().pose_jobs(
            limit=limit,
            before=request.args.get("before"),
            state=request.args.get("state"),
        )
        jobs = response.get("jobs") if isinstance(response, Mapping) else None
        if not isinstance(jobs, list):
            raise RuntimeError("The controller returned an invalid job list")
        return jsonify(
            {
                "jobs": [_public_job(job) for job in jobs],
                "next_cursor": response.get("next_cursor"),
            }
        )
    except Exception as exc:
        return _error(exc)


@cluster_bp.get("/cluster/jobs/<job_id>")
def get_cluster_job(job_id: str):
    try:
        _require_cluster_enabled()
        _require_id(job_id)
        response = _public_job_response(get_cluster_client().job(job_id))
        if request.args.get("include_log") in {"1", "true", "yes"}:
            response["log"] = _public_controller_text(
                get_cluster_client().job_log(job_id)
            )
        return jsonify(response)
    except Exception as exc:
        return _error(exc)


@cluster_bp.post("/cluster/jobs/<job_id>/cancel")
def cancel_cluster_job(job_id: str):
    try:
        _require_cluster_enabled()
        _require_id(job_id)
        return (
            jsonify(
                _public_job_response(
                    get_cluster_client().cancel_job(
                        job_id, idempotency_key=new_idempotency_key("job-cancel")
                    )
                )
            ),
            202,
        )
    except Exception as exc:
        return _error(exc)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@cluster_bp.post("/cluster/jobs/<job_id>/import-result")
def import_cluster_result(job_id: str):
    import_root: Path | None = None
    try:
        _require_cluster_enabled()
        _require_id(job_id, prefix="pose")
        value = _json_object()
        run_root = resolve_web_run_root(value.get("run_root"))
        response = get_cluster_client().job(job_id)
        job = response.get("job")
        if not isinstance(job, Mapping) or job.get("state") not in SUCCESS_STATES:
            raise RuntimeError(
                "The cluster pose job has no successful result to import"
            )
        payload = job.get("payload")
        result = job.get("result")
        if not isinstance(payload, Mapping) or not isinstance(result, Mapping):
            raise RuntimeError("The cluster job is missing immutable result evidence")
        if payload.get("run_root") != run_root.as_posix():
            raise ValueError("The cluster job belongs to a different run")
        expected_dataset = result.get("dataset_sha256")
        if not isinstance(expected_dataset, str):
            raise RuntimeError("The cluster result has no staged dataset digest")
        import_root = (
            run_root
            / "processed"
            / "bop_evaluation"
            / ".cluster-imports"
            / uuid.uuid4().hex
        )
        import_root.mkdir(parents=True, exist_ok=False)
        filename = result.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise RuntimeError("The controller returned an invalid result filename")
        result_path = get_cluster_client().download_artifact(
            job_id, "result.csv", import_root / filename
        )
        provenance_path = get_cluster_client().download_artifact(
            job_id,
            "provenance.json",
            import_root / "provenance.json",
            max_bytes=8 * 1024 * 1024,
        )
        if _sha256(result_path) != result.get("sha256") or _sha256(
            provenance_path
        ) != result.get("provenance_sha256"):
            raise RuntimeError(
                "Downloaded controller artifacts failed integrity checks"
            )
        provenance = json.loads(provenance_path.read_text())
        if not isinstance(provenance, Mapping):
            raise ValueError("Controller provenance must be a JSON object")
        registered, created = import_external_bop_result(
            run_root,
            result_path,
            external_job_id=job_id,
            expected_dataset_sha256=expected_dataset,
            source_provenance_sha256=result["provenance_sha256"],
            controller_provenance=provenance,
        )
        result_id = registered["result_id"]
        return (
            jsonify(
                {
                    "result": registered,
                    "created": created,
                    "evaluation_url": f"/bop-evaluation?result_id={result_id}",
                    "download_url": (
                        f"/bop/evaluation/results/{result_id}/download"
                        f"?run_root={run_root.as_posix()}"
                    ),
                }
            ),
            201 if created else 200,
        )
    except Exception as exc:
        return _error(exc)
    finally:
        if import_root is not None:
            shutil.rmtree(import_root, ignore_errors=True)


@cluster_bp.get("/cluster/archives")
def list_cluster_archives():
    try:
        _require_cluster_enabled()
        response = get_cluster_client().archives()
        archives = response.get("archives") if isinstance(response, Mapping) else None
        if not isinstance(archives, list):
            raise RuntimeError("The controller returned an invalid archive list")
        return jsonify(
            {
                "archives": [_public_archive(archive) for archive in archives],
                "integration": {"enabled": _settings().cluster_enabled},
            }
        )
    except Exception as exc:
        return _error(exc)


@cluster_bp.post("/cluster/archives")
def create_cluster_archive():
    try:
        _require_cluster_enabled()
        value = _json_object()
        run_root = resolve_direct_run_folder(
            resolve_web_run_root(value.get("run_root")),
            allowed_roots=web_run_roots(),
        )
        expected = value.get("expected_identity")
        validate_expected_identity(run_root, expected)
        _assert_no_active_run_jobs(run_root)
        operator = value.get("operator")
        if not isinstance(operator, str) or not operator.strip():
            raise ValueError("operator is required")
        response = get_cluster_client().create_archive(
            {
                "run_root": run_root.as_posix(),
                "operator": operator.strip(),
            },
            idempotency_key=new_idempotency_key("archive-copy"),
        )
        if not isinstance(response, Mapping):
            raise RuntimeError("The controller returned an invalid response")
        return jsonify({"archive": _public_archive(response.get("archive"))}), 202
    except Exception as exc:
        return _error(exc)


@cluster_bp.post("/cluster/archives/<archive_id>/restore")
def restore_cluster_archive(archive_id: str):
    try:
        _require_cluster_enabled()
        _require_id(archive_id, prefix="archive")
        value = _json_object()
        destination_root = resolve_destination_root(
            value.get("destination_root"), allowed_roots=web_run_roots()
        )
        destination_name = value.get("destination_name")
        if destination_name is not None and (
            not isinstance(destination_name, str)
            or Path(destination_name).name != destination_name
            or destination_name in {".", ".."}
        ):
            raise ValueError("destination_name must be one folder name")
        operator = value.get("operator")
        if not isinstance(operator, str) or not operator.strip():
            raise ValueError("operator is required")
        response = get_cluster_client().restore_archive(
            archive_id,
            {
                "destination_root": destination_root.as_posix(),
                "destination_name": destination_name,
                "operator": operator.strip(),
            },
            idempotency_key=new_idempotency_key("archive-restore"),
        )
        return jsonify(_public_job_response(response)), 202
    except Exception as exc:
        return _error(exc)
