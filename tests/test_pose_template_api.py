from __future__ import annotations

from dataclasses import dataclass

from posetestbot.jobs.runner import ResourceBusyError
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


def test_workpieces_owns_catalogue_routes_and_pose_templates_keeps_library(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("POSETESTBOT_WORKING_DATA_ROOT", tmp_path.as_posix())
    client = create_app().test_client()

    assert client.get("/workpieces/catalog").status_code == 200
    assert client.get("/pose-templates/library").status_code == 200
    assert client.get("/pose-templates/catalog").status_code == 404
    assert client.post("/pose-templates/catalog/upload").status_code == 404


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

    response = (
        create_app()
        .test_client()
        .delete(
            f"/pose-templates/library/{template_uuid}",
            json={"confirm": True},
        )
    )

    assert response.status_code == 200
    assert response.get_json()["status"] == "deleted_cleanup_pending"
    assert "resources are busy" in response.get_json()["cleanup_job_error"]
