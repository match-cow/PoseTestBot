from __future__ import annotations

from posetestbot.runtime import status as runtime_status


def test_collect_runtime_status_reports_acquisition_runtimes(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_status,
        "module_available",
        lambda module: module == "pyzed.sl",
    )

    status = runtime_status.collect_runtime_status(
        which=lambda executable: f"/usr/bin/{executable}",
    )

    assert status["schema_version"] == runtime_status.SCHEMA_VERSION
    assert status["all_available"] is True
    runtimes = {item["runtime_id"]: item for item in status["runtimes"]}
    assert sorted(runtimes) == ["blenderproc", "zed_sdk_python"]
    assert runtimes["blenderproc"]["category"] == "renderer"
    assert runtimes["zed_sdk_python"]["category"] == "camera_sdk"


def test_collect_runtime_status_reports_missing_acquisition_prerequisites(
    monkeypatch,
) -> None:
    monkeypatch.setattr(runtime_status, "module_available", lambda _: False)

    status = runtime_status.collect_runtime_status(
        which=lambda _: None,
    )

    assert status["all_available"] is False
    assert status["available_count"] == 0
    runtimes = {item["runtime_id"]: item for item in status["runtimes"]}
    assert runtimes["blenderproc"]["checks"][0]["name"] == "executable:blenderproc"
    assert runtimes["blenderproc"]["checks"][0]["ok"] is False
    assert runtimes["zed_sdk_python"]["checks"][0]["name"] == "module:pyzed.sl"
    assert runtimes["zed_sdk_python"]["checks"][0]["ok"] is False
