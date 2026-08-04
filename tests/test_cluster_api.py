from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from posetestbot.bop.evaluation import inspect_dataset, list_results
from posetestbot.cluster.client import ClusterClientError
from posetestbot.web.app import create_app
from posetestbot.web.runtime import WebRuntime, WebSettings
from tests.test_bop_evaluation import make_tiny_evaluation_run, write_result_csv


class FakeRunner:
    def __init__(self, root: Path):
        self.job_root = root

    def list(self, *, include_services: bool = True):
        return []


def _status() -> dict[str, Any]:
    return {
        "schema_version": "posetestbot_cluster_status.v1",
        "ready": True,
        "connection": {"ready": True},
        "features": {
            "archive_read": True,
            "archive_mutation": True,
            "pose_estimation": True,
        },
        "feature_blockers": {"archive": [], "estimation": []},
        "runtime": {
            "runtime_id": "foundationpose-a1b694b8",
            "foundationpose_revision": "a1b694b83e633c2cb6115b9063d940a687759392",
            "bop_toolkit_revision": "cea62d651c7e395b2e1962b9749e4e89693c6ac4",
            "sif_sha256": "1" * 64,
            "weights_sha256": "2" * 64,
            "weights_files_sha256": "4" * 64,
            "qualification_manifest_sha256": "5" * 64,
            "foundationpose_license": "NVIDIA Source Code License",
            "foundationpose_license_sha256": "3" * 64,
            "qualified": True,
            "ready": True,
        },
        "profiles": [
            {
                "profile_id": "smoke",
                "enabled": True,
                "partition": "gpu",
                "gres": "gpu:1",
                "cpus": 4,
                "memory": "24G",
                "walltime": "00:20:00",
                "max_targets": 2,
            }
        ],
    }


class FakeController:
    def __init__(self):
        self.pose_payload: dict[str, Any] | None = None
        self.pose_key: str | None = None
        self.job_value: dict[str, Any] | None = None
        self.result_source: Path | None = None
        self.provenance_source: Path | None = None
        self.archive_value: dict[str, Any] | None = None
        self.archive_payload: dict[str, Any] | None = None
        self.archive_key: str | None = None
        self.restore_payload: dict[str, Any] | None = None
        self.restore_key: str | None = None
        self.cancel_key: str | None = None

    def status(self):
        return _status()

    def create_pose_job(self, payload, *, idempotency_key: str):
        self.pose_payload = dict(payload)
        self.pose_key = idempotency_key
        job_id = f"pose-{uuid.UUID('12345678-1234-4234-9234-123456789abc')}"
        return {
            "job": {
                "schema_version": "posetestbot_cluster_job.v1",
                "job_id": job_id,
                "state": "preparing",
                "status": "preparing",
                "payload": dict(payload),
            }
        }

    def pose_jobs(self, **_kwargs):
        return {"jobs": [self.job_value] if self.job_value else [], "next_cursor": None}

    def job(self, _job_id: str):
        if self.job_value is None:
            raise KeyError("missing fixture job")
        return {"job": self.job_value}

    def download_artifact(
        self, _job_id, artifact, destination, *, max_bytes=128 * 1024 * 1024
    ):
        source = (
            self.result_source if artifact == "result.csv" else self.provenance_source
        )
        assert source is not None
        assert source.stat().st_size <= max_bytes
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return destination

    def archives(self, *, verify_archive_id=None):
        if verify_archive_id is not None and self.archive_value is None:
            raise KeyError("missing fixture archive")
        return {
            "archives": [self.archive_value] if self.archive_value else [],
            "verified_archive": self.archive_value if verify_archive_id else None,
        }

    def create_archive(self, payload, *, idempotency_key: str):
        self.archive_payload = dict(payload)
        self.archive_key = idempotency_key
        return {"archive": self.archive_value}

    def restore_archive(self, _archive_id, payload, *, idempotency_key: str):
        self.restore_payload = dict(payload)
        self.restore_key = idempotency_key
        return {"job": self.job_value}

    def cancel_job(self, _job_id, *, idempotency_key: str):
        self.cancel_key = idempotency_key
        return {"job": self.job_value}

    def job_log(self, _job_id):
        return "controller log\nremote /secret/work\nAuthorization: Bearer fixture\n"


class OfflineController(FakeController):
    def status(self):
        raise ClusterClientError("Cluster controller is unavailable")


class BlockedController(FakeController):
    def status(self):
        status = _status()
        status["ready"] = False
        status["connection"] = {"ready": False}
        status["blockers"] = [
            "PROJECT quota cannot currently be verified.",
            {
                "code": "login_host_unavailable",
                "message": "The LUIS login host did not answer.",
            },
        ]
        return status


def _pose_ready_run(root: Path) -> Path:
    run = make_tiny_evaluation_run(root, name="pose-ready")
    scene = run / "bop" / "test" / "000001"
    (scene / "mask_visib").mkdir()
    assert cv2.imwrite(
        (scene / "mask_visib" / "000000_000000.png").as_posix(),
        np.full((8, 8), 255, dtype=np.uint8),
    )
    manifest_path = run / "bop" / "bop_export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["annotation_mode"] = "pose_and_masks"
    manifest["capabilities"].update(
        {"gt_masks_full": True, "gt_masks_visible": True, "gt_visibility_info": True}
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return run


def _app(tmp_path: Path, controller, *, enabled: bool = True):
    runs_root = tmp_path / "runs"
    runs_root.mkdir(exist_ok=True)
    settings = WebSettings(
        host="127.0.0.1",
        port=5000,
        debug=False,
        job_root=tmp_path / "jobs",
        cluster_url="http://127.0.0.1:8765",
        cluster_token="x" * 32,
        cluster_enabled=enabled,
    )
    runner = FakeRunner(settings.job_root)
    runtime = WebRuntime(settings, runner, controller)
    return create_app(runtime=runtime), runs_root


def test_pose_setup_submission_is_server_revalidated_and_loopback_proxied(
    tmp_path: Path, monkeypatch
) -> None:
    controller = FakeController()
    app, runs_root = _app(tmp_path, controller)
    run = _pose_ready_run(runs_root)
    (run / "run_config.json").write_text("{}\n")
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", runs_root.as_posix())
    client = app.test_client()

    setup = client.get(
        "/cluster/pose-estimation/setup", query_string={"run_root": run.as_posix()}
    )
    assert setup.status_code == 200
    assert setup.get_json()["ready"] is True
    assert setup.get_json()["annotation_mode"] == "pose_and_masks"
    assert setup.get_json()["oracle_mask_contract"] == "bop_mask_visib_gt_instance.v1"

    submitted = client.post(
        "/cluster/pose-estimation/jobs",
        json={
            "run_root": run.as_posix(),
            "profile_id": "smoke",
            "operator": "Fixture Operator",
            "dataset_sha256": "caller-must-not-control-this",
        },
    )
    assert submitted.status_code == 202
    dataset = inspect_dataset(run, include_depth_content=True)
    assert controller.pose_payload == {
        "run_root": run.as_posix(),
        "dataset_alias": dataset["dataset_alias"],
        "dataset_sha256": dataset["dataset_sha256"],
        "profile_id": "smoke",
        "operator": "Fixture Operator",
    }
    assert controller.pose_key is not None
    assert controller.pose_key.startswith("pose-submit:")


def test_pose_setup_exposes_controller_outage_and_containment_blockers(
    tmp_path: Path, monkeypatch
) -> None:
    app, runs_root = _app(tmp_path, OfflineController())
    run = _pose_ready_run(runs_root)
    outside = _pose_ready_run(tmp_path / "outside")
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", runs_root.as_posix())
    client = app.test_client()

    setup = client.get(
        "/cluster/pose-estimation/setup", query_string={"run_root": run.as_posix()}
    )
    assert setup.status_code == 200
    assert setup.get_json()["ready"] is False
    assert any(
        item["code"] == "controller_unavailable"
        for item in setup.get_json()["blockers"]
    )

    escaped = client.get(
        "/cluster/pose-estimation/setup",
        query_string={"run_root": outside.as_posix()},
    )
    assert escaped.status_code == 400
    assert "allowed root" in escaped.get_json()["output"]


def test_pose_setup_preserves_structured_controller_readiness_blockers(
    tmp_path: Path, monkeypatch
) -> None:
    app, runs_root = _app(tmp_path, BlockedController())
    run = _pose_ready_run(runs_root)
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", runs_root.as_posix())
    client = app.test_client()

    response = client.get(
        "/cluster/pose-estimation/setup", query_string={"run_root": run.as_posix()}
    )

    assert response.status_code == 200
    setup = response.get_json()
    assert setup["ready"] is False
    assert {(item["code"], item["message"]) for item in setup["blockers"]} >= {
        (
            "controller_blocker_1",
            "PROJECT quota cannot currently be verified.",
        ),
        ("login_host_unavailable", "The LUIS login host did not answer."),
    }


def _successful_external_job(
    controller: FakeController, run: Path, tmp_path: Path
) -> str:
    dataset = inspect_dataset(run)
    job_id = "pose-12345678-1234-4234-9234-123456789abc"
    result = write_result_csv(
        tmp_path / f"foundationpose_{dataset['dataset_alias']}-test_{job_id}.csv"
    )
    result_hash = hashlib.sha256(result.read_bytes()).hexdigest()
    provenance = {
        "schema_version": "posetestbot_cluster_collected_result.v1",
        "job_id": job_id,
        "method": "foundationpose",
        "dataset_sha256": dataset["dataset_sha256"],
        "oracle_mask_contract": "bop_mask_visib_gt_instance.v1",
        "score_contract": "constant_1.0_no_detection_confidence",
        "execution_contract": "independent_register_per_target_no_tracking.v1",
        "units": {
            "bop_model": "millimetres",
            "bop_depth": "millimetres",
            "foundationpose": "metres",
            "result_translation": "millimetres",
        },
        "runtime": _status()["runtime"],
        "input_manifest_sha256": "3" * 64,
        "input_hashes": {"rgb": "4" * 64, "depth": "5" * 64},
        "bop_content_sha256": "6" * 64,
        "output_hashes": {result.name: result_hash},
        "project_copy": {
            "state": "verified",
            "artifact_sha256": {result.name: result_hash},
        },
        "estimate_count": 1,
        "failure_count": 0,
        "collected_at": "2026-08-04T12:00:00+00:00",
        "remote_work_dir": f"/secret/project/results/{job_id}",
        "scheduler": {"command": "sbatch --secret=/secret/token"},
        "external_job": {
            "provider": "posetestbot-cluster",
            "job_id": job_id,
            "slurm_job_id": "81234",
        },
        "result": {
            "filename": result.name,
            "sha256": result_hash,
            "size_bytes": result.stat().st_size,
        },
    }
    provenance_path = tmp_path / "controller-provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    controller.result_source = result
    controller.provenance_source = provenance_path
    controller.job_value = {
        "schema_version": "posetestbot_cluster_job.v1",
        "job_id": job_id,
        "kind": "pose-estimation",
        "state": "succeeded",
        "status": "succeeded",
        "payload": {"run_root": run.as_posix()},
        "result": {
            "filename": result.name,
            "sha256": result_hash,
            "provenance_sha256": hashlib.sha256(
                provenance_path.read_bytes()
            ).hexdigest(),
            "dataset_sha256": dataset["dataset_sha256"],
            "estimate_count": 1,
            "failure_count": 0,
        },
        "terminal": True,
    }
    return job_id


def test_external_result_import_is_idempotent_and_historical_download_survives_drift(
    tmp_path: Path, monkeypatch
) -> None:
    controller = FakeController()
    app, runs_root = _app(tmp_path, controller)
    run = _pose_ready_run(runs_root)
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", runs_root.as_posix())
    job_id = _successful_external_job(controller, run, tmp_path)
    client = app.test_client()

    first = client.post(
        f"/cluster/jobs/{job_id}/import-result", json={"run_root": run.as_posix()}
    )
    second = client.post(
        f"/cluster/jobs/{job_id}/import-result", json={"run_root": run.as_posix()}
    )
    assert first.status_code == 201
    assert second.status_code == 200
    assert (
        first.get_json()["result"]["result_id"]
        == second.get_json()["result"]["result_id"]
    )
    assert second.get_json()["created"] is False
    records = list_results(run)
    assert len(records) == 1
    assert records[0]["source_kind"] == "external_controller"
    assert records[0]["external_job"]["slurm_job_id"] == "81234"
    stored = json.loads((run / records[0]["controller_provenance_path"]).read_text())
    assert stored["schema_version"] == "posetestbot_external_result_provenance.v1"
    assert "project_copy" not in stored and "scheduler" not in stored
    assert "/secret" not in json.dumps(stored)

    manifest = run / "bop" / "bop_export_manifest.json"
    manifest.write_text(manifest.read_text() + " ")
    result_id = records[0]["result_id"]
    download = client.get(
        f"/bop/evaluation/results/{result_id}/download",
        query_string={"run_root": run.as_posix()},
    )
    assert download.status_code == 200
    assert hashlib.sha256(download.data).hexdigest() == records[0]["sha256"]


def test_external_result_import_refuses_dataset_drift_without_losing_remote_result(
    tmp_path: Path, monkeypatch
) -> None:
    controller = FakeController()
    app, runs_root = _app(tmp_path, controller)
    run = _pose_ready_run(runs_root)
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", runs_root.as_posix())
    job_id = _successful_external_job(controller, run, tmp_path)
    manifest = run / "bop" / "bop_export_manifest.json"
    manifest.write_text(manifest.read_text() + " ")

    response = app.test_client().post(
        f"/cluster/jobs/{job_id}/import-result", json={"run_root": run.as_posix()}
    )
    assert response.status_code == 409
    assert "changed after this cluster job was staged" in response.get_json()["output"]
    assert list_results(run) == []
    assert controller.result_source is not None and controller.result_source.is_file()


def test_cluster_jobs_logs_cancel_and_archive_copy_restore_use_server_keys(
    tmp_path: Path, monkeypatch
) -> None:
    controller = FakeController()
    app, runs_root = _app(tmp_path, controller)
    run = _pose_ready_run(runs_root)
    (run / "run_config.json").write_text("{}\n")
    identity = {"device": run.stat().st_dev, "inode": run.stat().st_ino}
    archive_id = "archive-12345678-1234-4234-9234-123456789abc"
    job_id = "pose-12345678-1234-4234-9234-123456789abc"
    controller.archive_value = {
        "archive_id": archive_id,
        "state": "succeeded",
        "status": "succeeded",
        "source_run_root": run.as_posix(),
        "source_identity": identity,
        "verified": True,
        "remote_path": "/secret/project/archive",
    }
    controller.job_value = {
        "job_id": job_id,
        "kind": "pose-estimation",
        "state": "running",
        "status": "running",
        "payload": {"run_root": run.as_posix(), "remote_path": "/secret/work"},
        "error": "remote failure at /secret/work",
        "log_available": True,
        "terminal": False,
    }
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", runs_root.as_posix())
    client = app.test_client()

    created = client.post(
        "/cluster/archives",
        headers={"Idempotency-Key": "browser-controlled"},
        json={
            "run_root": run.as_posix(),
            "expected_identity": identity,
            "operator": "Fixture Operator",
        },
    )
    assert created.status_code == 202, created.get_json()
    assert controller.archive_payload == {
        "run_root": run.as_posix(),
        "operator": "Fixture Operator",
    }
    assert controller.archive_key is not None
    assert controller.archive_key.startswith("archive-copy:")
    assert controller.archive_key != "browser-controlled"
    assert "remote_path" not in created.get_json()["archive"]

    listed = client.get("/cluster/archives")
    assert listed.status_code == 200
    assert listed.get_json()["integration"] == {"enabled": True}
    assert listed.get_json()["archives"][0]["archive_id"] == archive_id

    restored = client.post(
        f"/cluster/archives/{archive_id}/restore",
        headers={"Idempotency-Key": "browser-controlled"},
        json={
            "destination_root": runs_root.as_posix(),
            "destination_name": "restored-run",
            "operator": "Fixture Operator",
        },
    )
    assert restored.status_code == 202
    assert controller.restore_payload == {
        "destination_root": runs_root.as_posix(),
        "destination_name": "restored-run",
        "operator": "Fixture Operator",
    }
    assert controller.restore_key is not None
    assert controller.restore_key.startswith("archive-restore:")

    job = client.get(f"/cluster/jobs/{job_id}", query_string={"include_log": "1"})
    assert job.status_code == 200
    assert job.get_json()["log"] == (
        "controller log\nremote [controller path]\n[redacted controller detail]\n"
    )
    assert job.get_json()["job"]["error"] == "remote failure at [controller path]"
    assert "remote_path" not in job.get_json()["job"]["payload"]

    canceled = client.post(
        f"/cluster/jobs/{job_id}/cancel",
        headers={"Idempotency-Key": "browser-controlled"},
    )
    assert canceled.status_code == 202
    assert controller.cancel_key is not None
    assert controller.cancel_key.startswith("job-cancel:")
