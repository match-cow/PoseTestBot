from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from posetestbot.web.app import create_app
from posetestbot.web.routes import pose_templates as legacy_routes
from posetestbot.web.routes import workpieces as routes


class RecordingRunner:
    def __init__(self) -> None:
        self.submissions: list[dict] = []

    def submit(self, **kwargs):
        self.submissions.append(kwargs)
        raise AssertionError("an oversized upload must not be queued")


@pytest.mark.parametrize(
    "environ_overrides",
    [
        {},
        {"CONTENT_LENGTH": "", "wsgi.input_terminated": True},
    ],
    ids=["declared-content-length", "unknown-content-length"],
)
def test_workpiece_upload_enforces_streamed_request_limit_before_queueing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environ_overrides: dict[str, object],
) -> None:
    request_root = tmp_path / "workpiece_catalog_requests"
    runner = RecordingRunner()
    monkeypatch.setattr(routes, "MAX_UPLOAD_BATCH_BYTES", 1)
    monkeypatch.setattr(routes, "REQUEST_ROOT", request_root)
    monkeypatch.setattr(routes, "job_runner", runner)
    client = create_app().test_client()

    response = client.post(
        "/workpieces/catalog/upload",
        data={
            "cad": (
                io.BytesIO(b"x" * (1024 * 1024 + 64 * 1024)),
                "oversized.stl",
            )
        },
        content_type="multipart/form-data",
        environ_overrides=environ_overrides,
    )

    assert response.status_code == 413
    assert "size limit" in response.get_json()["output"]
    assert runner.submissions == []
    assert not request_root.exists()


@pytest.mark.parametrize(
    "environ_overrides",
    [
        {},
        {"CONTENT_LENGTH": "", "wsgi.input_terminated": True},
    ],
    ids=["declared-content-length", "unknown-content-length"],
)
def test_legacy_catalog_upload_enforces_the_same_streamed_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environ_overrides: dict[str, object],
) -> None:
    runner = RecordingRunner()
    monkeypatch.setattr(legacy_routes, "MAX_UPLOAD_BATCH_BYTES", 1)
    monkeypatch.setattr(
        legacy_routes, "WORKPIECE_REQUEST_ROOT", tmp_path / "legacy-requests"
    )
    monkeypatch.setattr(legacy_routes, "job_runner", runner)
    client = create_app().test_client()

    response = client.post(
        "/pose-templates/catalog/upload",
        data={
            "cad": (
                io.BytesIO(b"x" * (1024 * 1024 + 64 * 1024)),
                "oversized.stl",
            )
        },
        content_type="multipart/form-data",
        environ_overrides=environ_overrides,
    )

    assert response.status_code == 413
    assert "100 MiB" in response.get_json()["output"]
    assert runner.submissions == []
    assert not (tmp_path / "legacy-requests").exists()


@pytest.mark.parametrize(
    "path",
    [
        "/workpieces/catalog/11111111-1111-4111-8111-111111111111",
        "/pose-templates/preview",
    ],
)
@pytest.mark.parametrize(
    "environ_overrides",
    [
        {},
        {"CONTENT_LENGTH": "", "wsgi.input_terminated": True},
    ],
    ids=["declared-content-length", "unknown-content-length"],
)
def test_catalog_and_template_json_limits_apply_before_security_parsing(
    path: str,
    environ_overrides: dict[str, object],
) -> None:
    client = create_app().test_client()
    payload = json.dumps({"padding": "x" * (2 * 1024 * 1024)})

    response = client.open(
        path,
        method="PATCH" if path.startswith("/workpieces/") else "POST",
        data=payload,
        content_type="application/json",
        environ_overrides=environ_overrides,
    )

    assert response.status_code == 413
    assert "2 MiB" in response.get_json()["output"]
