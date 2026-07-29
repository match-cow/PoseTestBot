from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from posetestbot.jobs.runner import (
    FAILED,
    SERVICE_VISIBILITY,
    LocalJobRunner,
)
from posetestbot.web.app import create_app


def _terminal_job(
    root: Path,
    *,
    job_id: str,
    created_at: str,
    status: str = "succeeded",
    scope_kind: str = "global",
    run_root: str | None = None,
    name: str | None = None,
) -> None:
    folder = root / job_id
    folder.mkdir(parents=True)
    (folder / "log.txt").write_text(f"log for {job_id}")
    (folder / "job.json").write_text(
        json.dumps(
            {
                "id": job_id,
                "name": name or job_id,
                "command": [sys.executable, "-c", "pass"],
                "cwd": None,
                "status": status,
                "created_at": created_at,
                "ended_at": created_at,
                "log_path": (folder / "log.txt").as_posix(),
                "resources": [],
                "parameters": {"command_provenance": job_id},
                "scope_kind": scope_kind,
                "run_root": run_root,
            }
        )
    )


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("POSETESTBOT_WEB_RUN_ROOTS", tmp_path.as_posix())
    job_root = tmp_path / "jobs"
    active_run = (tmp_path / "active-run").resolve().as_posix()
    other_run = (tmp_path / "other-run").resolve().as_posix()
    _terminal_job(
        job_root,
        job_id="global-old",
        created_at="2026-07-01T00:00:00+00:00",
    )
    _terminal_job(
        job_root,
        job_id="library-job",
        created_at="2026-07-02T00:00:00+00:00",
        scope_kind="library",
        name="Template authoring",
    )
    _terminal_job(
        job_root,
        job_id="other-failed",
        created_at="2026-07-03T00:00:00+00:00",
        status=FAILED,
        scope_kind="run",
        run_root=other_run,
        name="Other run failure",
    )
    _terminal_job(
        job_root,
        job_id="active-finished",
        created_at="2026-07-04T00:00:00+00:00",
        scope_kind="run",
        run_root=active_run,
    )
    runner = LocalJobRunner(job_root)
    app = create_app(job_runner=runner)
    app.config.update(TESTING=True)
    return app.test_client(), runner, active_run, other_run


def test_jobs_api_paginates_terminal_history_and_keeps_active_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, runner, active_run, _other_run = _client(tmp_path, monkeypatch)
    active = runner.submit(
        name="active-run-work",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        resources=["disk_io"],
        scope_kind="run",
        run_root=active_run,
    )
    try:
        response = client.get("/jobs?limit=1")
        assert response.status_code == 200
        value = response.get_json()
        assert [job["id"] for job in value["jobs"]] == [
            active.id,
            "active-finished",
        ]
        assert value["total"] == 5
        assert value["limit"] == 1
        assert value["status_counts"]["succeeded"] == 3
        assert value["status_counts"]["failed"] == 1
        assert value["status_counts"][value["jobs"][0]["status"]] == 1
        assert value["resources"] == {"disk_io": active.id}
        assert value["next_cursor"]

        next_page = client.get(
            "/jobs",
            query_string={"limit": 1, "cursor": value["next_cursor"]},
        ).get_json()
        assert [job["id"] for job in next_page["jobs"]] == ["other-failed"]
        assert active.id not in {job["id"] for job in next_page["jobs"]}
    finally:
        runner.cancel(active.id)
        runner.wait(active.id, timeout=5)


def test_jobs_api_filters_search_status_scope_and_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _runner, active_run, other_run = _client(tmp_path, monkeypatch)

    library = client.get(
        "/jobs",
        query_string={"scope_kind": "library", "search": "template"},
    ).get_json()
    assert [job["id"] for job in library["jobs"]] == ["library-job"]
    assert library["total"] == 1
    assert library["jobs"][0]["scope_kind"] == "library"
    assert library["jobs"][0]["run_root"] is None

    failed = client.get(
        "/jobs",
        query_string={
            "scope": "run",
            "run_root": other_run,
            "status": "failed",
        },
    ).get_json()
    assert [job["id"] for job in failed["jobs"]] == ["other-failed"]
    assert failed["jobs"][0]["run_root"] == other_run

    selected = client.get(
        "/jobs",
        query_string={"scope_kind": "run", "run_root": active_run},
    ).get_json()
    assert [job["id"] for job in selected["jobs"]] == ["active-finished"]


def test_jobs_api_validates_limits_and_cursor_filter_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _runner, _active_run, _other_run = _client(tmp_path, monkeypatch)

    assert client.get("/jobs?limit=101").status_code == 400
    assert client.get("/jobs?limit=not-a-number").status_code == 400
    first = client.get("/jobs?limit=1").get_json()
    mismatch = client.get(
        "/jobs",
        query_string={
            "limit": 1,
            "cursor": first["next_cursor"],
            "scope_kind": "library",
        },
    )
    assert mismatch.status_code == 400
    assert "current filters" in mismatch.get_json()["output"]


def test_exact_job_log_and_cancel_routes_remain_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, runner, _active_run, _other_run = _client(tmp_path, monkeypatch)
    exact = client.get("/jobs/library-job")
    assert exact.status_code == 200
    assert exact.get_json()["job"]["id"] == "library-job"
    assert client.get("/jobs/library-job/log").get_data(as_text=True) == (
        "log for library-job"
    )

    active = runner.submit(
        name="cancel-me",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        scope_kind="global",
    )
    canceled = client.post(f"/jobs/{active.id}/cancel")
    assert canceled.status_code == 200
    assert canceled.get_json()["job"]["status"] in {"canceling", "canceled"}
    assert runner.wait(active.id, timeout=5).status == "canceled"
    assert client.get("/jobs/does-not-exist").status_code == 404


def test_jobs_api_refuses_cancel_for_committed_storage_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, runner, _active_run, _other_run = _client(tmp_path, monkeypatch)
    active = runner.submit(
        name="run-folder-delete",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        parameters={"cancelable": False, "run_folder_operation": "delete"},
        scope_kind="global",
    )
    try:
        refused = client.post(f"/jobs/{active.id}/cancel")
        assert refused.status_code == 409
        assert "cannot be canceled safely" in refused.get_json()["output"]
        assert runner.get(active.id).status in {"queued", "running"}
    finally:
        runner.cancel(active.id)
        runner.wait(active.id, timeout=5)


def test_jobs_api_hides_service_jobs_and_resources_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, runner, _active_run, _other_run = _client(tmp_path, monkeypatch)
    service = runner.submit(
        name="service",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        resources=["monitoring_camera:test"],
        scope_kind="global",
        visibility=SERVICE_VISIBILITY,
    )
    try:
        public = client.get("/jobs").get_json()
        assert service.id not in {job["id"] for job in public["jobs"]}
        assert "monitoring_camera:test" not in public["resources"]

        internal = client.get("/jobs?include_services=true").get_json()
        assert service.id in {job["id"] for job in internal["jobs"]}
        assert internal["resources"]["monitoring_camera:test"] == service.id
    finally:
        runner.cancel(service.id)
        runner.wait(service.id, timeout=5)
