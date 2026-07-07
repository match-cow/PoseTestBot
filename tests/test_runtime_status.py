from __future__ import annotations

from pathlib import Path

from posetestbot.runtime import status as runtime_status


def test_collect_runtime_status_reports_available_and_missing_runtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    foundationpose_root = tmp_path / "FoundationPose"
    foundationpose_root.mkdir()
    (foundationpose_root / "run_demo.py").write_text("# demo\n")
    (foundationpose_root / "run_demo_no_tracking.py").write_text("# demo\n")
    bop_toolkit_root = tmp_path / "bop_toolkit"
    (bop_toolkit_root / "scripts").mkdir(parents=True)
    (bop_toolkit_root / "scripts" / "eval_bop19_pose.py").write_text("# eval\n")
    megapose_wrapper = tmp_path / "megapose_wrapper.py"
    megapose_wrapper.write_text("# megapose\n")
    sam6d_wrapper = tmp_path / "sam6d_wrapper.py"
    sam6d_wrapper.write_text("# sam6d\n")

    monkeypatch.setattr(
        runtime_status,
        "module_available",
        lambda module: module == "pyzed.sl",
    )

    status = runtime_status.collect_runtime_status(
        env={
            "FOUNDATIONPOSE_ROOT": foundationpose_root.as_posix(),
            "BOP_TOOLKIT_ROOT": bop_toolkit_root.as_posix(),
            "MEGAPOSE_WRAPPER": megapose_wrapper.as_posix(),
            "SAM6D_WRAPPER": sam6d_wrapper.as_posix(),
        },
        cwd=tmp_path,
        home=tmp_path,
        which=lambda executable: f"/usr/bin/{executable}",
    )

    assert status["schema_version"] == runtime_status.SCHEMA_VERSION
    assert status["all_available"] is True
    assert status["available_count"] == status["runtime_count"]
    runtimes = {item["runtime_id"]: item for item in status["runtimes"]}
    assert runtimes["blenderproc"]["available"] is True
    assert runtimes["foundationpose"]["available"] is True
    assert runtimes["megapose"]["available"] is True
    assert runtimes["sam6d"]["available"] is True
    assert runtimes["bop_toolkit"]["available"] is True
    assert runtimes["zed_sdk_python"]["available"] is True


def test_collect_runtime_status_reports_missing_prerequisites(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runtime_status, "module_available", lambda _: False)

    status = runtime_status.collect_runtime_status(
        env={},
        cwd=tmp_path,
        home=tmp_path,
        which=lambda _: None,
    )

    assert status["all_available"] is False
    assert status["available_count"] == 0
    runtimes = {item["runtime_id"]: item for item in status["runtimes"]}
    foundationpose_checks = {
        check["name"]: check
        for check in runtimes["foundationpose"]["checks"]
    }
    assert foundationpose_checks["executable:docker"]["ok"] is False
    assert foundationpose_checks["checkout"]["ok"] is False
    assert runtimes["megapose"]["checks"][0]["name"] == "wrapper_script"
    assert runtimes["megapose"]["checks"][0]["ok"] is False
    assert runtimes["sam6d"]["checks"][0]["name"] == "wrapper_script"
    assert runtimes["sam6d"]["checks"][0]["ok"] is False
    assert runtimes["bop_toolkit"]["available"] is False
    assert runtimes["zed_sdk_python"]["checks"][0]["name"] == "module:pyzed.sl"
