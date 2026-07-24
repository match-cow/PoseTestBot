from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import trimesh

from posetestbot.jobs.runner import ResourceBusyError
from posetestbot.pipeline.run_config import create_run_config, write_run_config
from posetestbot.pose_templates.catalog import import_catalog_object
from posetestbot.pose_templates.library import generate_template_bundle
from posetestbot.pose_templates.orientations import (
    ORIENTATION_THUMBNAIL_MAX_BYTES,
    analyze_catalog_orientations,
)
from posetestbot.web.app import create_app
from posetestbot.web.routes import pose_templates as routes


@dataclass
class FakeJob:
    id: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": "pose_template",
            "command": [],
            "cwd": None,
            "status": "queued",
            "created_at": "2026-07-20T00:00:00+00:00",
            "started_at": None,
            "ended_at": None,
            "returncode": None,
            "message": None,
            "tail": [],
            "resources": [],
            "parameters": {},
            "log_path": "log.txt",
            "visibility": "operator",
        }


class FakeRunner:
    def __init__(self) -> None:
        self.submissions: list[dict] = []

    def submit(self, **kwargs):
        self.submissions.append(kwargs)
        return FakeJob(f"posejob{len(self.submissions)}")


def mesh_bytes() -> bytes:
    return bytes(trimesh.creation.box(extents=(20, 10, 10)).export(file_type="stl"))


def test_pose_template_api_queues_heavy_work_and_serves_immutable_assets(
    tmp_path: Path, monkeypatch
) -> None:
    working = tmp_path / "working"
    runs = tmp_path / "runs"
    run = runs / "run"
    run.mkdir(parents=True)
    write_run_config(run, create_run_config(run_root=run, dataset_mode="pose_template"))
    monkeypatch.setenv("POSETESTBOT_WORKING_DATA_ROOT", working.as_posix())
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", runs.as_posix())
    monkeypatch.setenv("POSETESTBOT_WEB_INPUT_ROOTS", tmp_path.as_posix())
    monkeypatch.setattr(routes, "REQUEST_ROOT", working / "jobs" / "pose_template_requests")
    runner = FakeRunner()
    monkeypatch.setattr(routes, "job_runner", runner)

    cad = tmp_path / "box.stl"
    cad.write_bytes(mesh_bytes())
    record = import_catalog_object(
        name="Box", cad_path=cad, catalog_root=working / "object_catalog"
    )
    configuration = {
        "display_name": "API template",
        "instances": [
            {
                "catalog_uuid": record["catalog_uuid"],
                "instance_uuid": "11111111-1111-4111-8111-111111111111",
                "pose": {"x_mm": 40, "y_mm": 40},
            }
        ],
    }
    bundle = generate_template_bundle(
        configuration,
        catalog_root=working / "object_catalog",
        library_root=working / "pose_templates",
    )
    client = create_app().test_client()

    assert client.get("/pose-templates/status").get_json()["available"] is True
    listing = client.get("/pose-templates/catalog").get_json()
    assert listing["objects"][0]["catalog_uuid"] == record["catalog_uuid"]
    assert client.get(
        f"/pose-templates/catalog/{record['catalog_uuid']}/assets/canonical_ply"
    ).status_code == 200
    library_response = client.get("/pose-templates/library")
    assert library_response.get_json()["templates"][0]["template_uuid"] == bundle[
        "template_uuid"
    ]
    assert library_response.get_json()["templates"][0]["instance_count"] == 1
    assert library_response.get_json()["templates"][0]["thumbnail"]["stored"] is True
    assert b'"nominal_contours"' not in library_response.data
    assert b'"compensated_contours"' not in library_response.data
    assert b'"preview_meshes"' not in library_response.data
    assert client.get(
        f"/pose-templates/library/{bundle['template_uuid']}/download/pdf"
    ).data.startswith(b"%PDF")
    stored_preview = client.get(
        f"/pose-templates/library/{bundle['template_uuid']}/preview"
    )
    assert stored_preview.status_code == 200
    assert stored_preview.get_json()["schema_version"] == "pose_template_preview.v1"
    stored_thumbnail = client.get(
        f"/pose-templates/library/{bundle['template_uuid']}/thumbnail"
    )
    assert stored_thumbnail.status_code == 200
    assert stored_thumbnail.get_json()["schema_version"] == "pose_template_thumbnail.v1"
    assert stored_thumbnail.get_json()["template_uuid"] == bundle["template_uuid"]
    assert b'"preview_meshes"' not in stored_thumbnail.data
    immutable_mesh = client.get(
        f"/pose-templates/library/{bundle['template_uuid']}/assets/"
        "11111111-1111-4111-8111-111111111111/canonical_ply"
    )
    assert immutable_mesh.status_code == 200

    missing_orientations = client.get(
        f"/pose-templates/workpieces/{record['catalog_uuid']}/orientations"
    )
    assert missing_orientations.status_code == 404
    assert missing_orientations.get_json()["analysis_required"] is True
    orientation_job = client.post(
        f"/pose-templates/workpieces/{record['catalog_uuid']}/orientations"
    )
    assert orientation_job.status_code == 202
    assert runner.submissions[-1]["name"] == "pose_template_orientation_analysis"
    assert runner.submissions[-1]["command"][3] == (
        "scripts/run_pose_template_orientation_analysis.py"
    )
    assert f"workpiece_catalog:{record['catalog_uuid']}" in runner.submissions[-1][
        "resources"
    ]
    analyze_catalog_orientations(
        record["catalog_uuid"], catalog_root=working / "object_catalog"
    )
    thumbnail = client.get(
        f"/pose-templates/workpieces/{record['catalog_uuid']}/orientation-thumbnail"
    )
    assert thumbnail.status_code == 200
    assert (
        thumbnail.get_json()["schema_version"]
        == "pose_template_orientation_thumbnail.v1"
    )
    assert b'"contours"' not in thumbnail.data
    assert len(thumbnail.data) <= ORIENTATION_THUMBNAIL_MAX_BYTES

    upload = client.post(
        "/pose-templates/catalog/upload",
        data={"name": "Queued", "cad": (io.BytesIO(mesh_bytes()), "queued.STL")},
        content_type="multipart/form-data",
    )
    assert upload.status_code == 202
    assert runner.submissions[-1]["resources"] == ["cpu", "disk_io"]
    assert runner.submissions[-1]["command"][3] == "scripts/run_object_catalog_import.py"
    assert client.post("/pose-templates/catalog/legacy-import", json={}).status_code == 405

    preview = client.post(
        "/pose-templates/preview", json={"configuration": configuration}
    )
    assert preview.status_code == 202
    assert runner.submissions[-1]["name"] == "pose_template_preview"
    assert runner.submissions[-1]["resources"] == ["cpu", "disk_io"]
    preview_result = Path(runner.submissions[-1]["parameters"]["result"])
    preview_result.write_text('{"schema_version":"pose_template_preview.v1"}')
    consumed = client.get(
        f"/pose-templates/preview/{preview.get_json()['request_id']}"
    )
    assert consumed.status_code == 200
    assert consumed.get_json()["schema_version"] == "pose_template_preview.v1"
    assert not preview_result.parent.exists()
    assert client.get(
        f"/pose-templates/preview/{preview.get_json()['request_id']}"
    ).status_code == 404

    validation = client.post(
        "/pose-templates/validate", json={"configuration": configuration}
    )
    assert validation.status_code == 202
    assert runner.submissions[-1]["name"] == "pose_template_validation"

    generated = client.post(
        "/pose-templates/generate", json={"configuration": configuration}
    )
    assert generated.status_code == 202
    assert runner.submissions[-1]["name"] == "pose_template_generate"

    cloned = client.post(
        f"/pose-templates/library/{bundle['template_uuid']}/clone", json={}
    )
    assert cloned.status_code == 202
    assert runner.submissions[-1]["name"] == "pose_template_clone"

    selected = client.post(
        "/pose-templates/runs/selection",
        json={
            "run_root": run.as_posix(),
            "template_uuid": bundle["template_uuid"],
            "placement": {"matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
            "confirmed": True,
            "operator": "pytest",
        },
    )
    assert selected.status_code == 202
    assert runner.submissions[-1]["resources"] == ["disk_io"]
    assert runner.submissions[-1]["command"][3] == "scripts/run_pose_template_select.py"

    library_endpoint = f"/pose-templates/library/{bundle['template_uuid']}"
    unconfirmed_delete = client.delete(library_endpoint, json={})
    assert unconfirmed_delete.status_code == 400
    assert "confirm must be true" in unconfirmed_delete.get_json()["output"]
    deleted = client.delete(library_endpoint, json={"confirm": True})
    assert deleted.status_code == 202
    assert deleted.get_json()["schema_version"] == "pose_template_library_delete.v1"
    assert deleted.get_json()["status"] == "deleted_cleanup_pending"
    assert deleted.get_json()["job_id"] == "posejob8"
    assert runner.submissions[-1]["name"] == "pose_template_delete_cleanup"
    assert runner.submissions[-1]["command"][3:] == [
        "scripts/run_pose_template_delete_cleanup.py",
        "--template-uuid",
        bundle["template_uuid"],
    ]
    assert runner.submissions[-1]["resources"] == [
        "disk_io",
        f"pose_template_library:{bundle['template_uuid']}",
    ]
    assert client.get(library_endpoint).status_code == 404
    assert client.get("/pose-templates/library").get_json()["templates"] == []


def test_pose_template_api_remains_browsable_when_source_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    working = tmp_path / "working"
    monkeypatch.setenv("POSETESTBOT_WORKING_DATA_ROOT", working.as_posix())
    monkeypatch.setattr(
        routes,
        "posetemplatecreator_status",
        lambda: {
            "status": "missing",
            "available": False,
            "reason": "initialize the submodule",
        },
    )
    client = create_app().test_client()
    assert client.get("/pose-templates/status").get_json()["status"] == "missing"
    assert client.get("/pose-templates/catalog").status_code == 200
    assert client.get("/pose-templates/library").status_code == 200


def test_pose_template_delete_reports_pending_after_cleanup_queue_conflict(
    monkeypatch,
) -> None:
    template_uuid = "22222222-2222-4222-8222-222222222222"
    pending = {
        "schema_version": "pose_template_library_delete.v1",
        "template_uuid": template_uuid,
        "status": "deleted_cleanup_pending",
        "asset_cleanup": {
            "status": "pending",
            "path": f"{template_uuid}.assets",
            "last_error": None,
        },
    }

    class BusyRunner:
        def submit(self, **_kwargs):
            raise ResourceBusyError("Requested resources are busy: disk_io")

    monkeypatch.setattr(routes, "job_runner", BusyRunner())
    monkeypatch.setattr(
        routes,
        "delete_template_bundle",
        lambda _template_uuid, cleanup_assets: pending,
    )
    monkeypatch.setattr(
        routes,
        "record_template_cleanup_submission_failure",
        lambda _template_uuid, error: {
            **pending,
            "asset_cleanup": {
                **pending["asset_cleanup"],
                "last_error": str(error),
            },
        },
    )

    response = create_app().test_client().delete(
        f"/pose-templates/library/{template_uuid}",
        json={"confirm": True},
    )

    assert response.status_code == 200
    assert response.get_json()["status"] == "deleted_cleanup_pending"
    assert "resources are busy" in response.get_json()["cleanup_job_error"]
