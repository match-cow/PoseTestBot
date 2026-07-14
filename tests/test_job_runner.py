from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from posetestbot.jobs.runner import (
    CANCELED,
    CANCELING,
    FAILED,
    QUEUED,
    SUCCEEDED,
    LocalJobRunner,
    ResourceBusyError,
)


def test_local_job_runner_captures_successful_command(tmp_path: Path) -> None:
    runner = LocalJobRunner(tmp_path / "jobs")

    job = runner.submit(
        name="echo",
        command=[sys.executable, "-c", "print('hello from job')"],
        parameters={"purpose": "unit-test"},
    )
    finished = runner.wait(job.id, timeout=5)

    assert finished.status == SUCCEEDED
    assert finished.returncode == 0
    assert finished.parameters == {"purpose": "unit-test"}
    assert "hello from job" in runner.log_text(job.id)
    assert (tmp_path / "jobs" / job.id / "job.json").is_file()


def test_local_job_runner_records_failed_command(tmp_path: Path) -> None:
    runner = LocalJobRunner(tmp_path / "jobs")

    job = runner.submit(
        name="fail",
        command=[sys.executable, "-c", "print('nope'); raise SystemExit(7)"],
    )
    finished = runner.wait(job.id, timeout=5)

    assert finished.status == FAILED
    assert finished.returncode == 7
    assert "nope" in runner.log_text(job.id)


def test_local_job_runner_can_cancel_running_command(tmp_path: Path) -> None:
    runner = LocalJobRunner(tmp_path / "jobs")

    job = runner.submit(
        name="sleep",
        command=[sys.executable, "-c", "import time; time.sleep(10)"],
    )
    deadline = time.time() + 5
    while runner.get(job.id).started_at is None and time.time() < deadline:
        time.sleep(0.01)

    canceled = runner.cancel(job.id)
    finished = runner.wait(job.id, timeout=5)

    assert canceled.status in {CANCELING, CANCELED}
    assert finished.status == CANCELED


def test_local_job_runner_cancels_child_process_group(tmp_path: Path) -> None:
    marker = tmp_path / "child_survived.txt"
    runner = LocalJobRunner(tmp_path / "jobs")
    script = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', "
        f"\"import pathlib, time; time.sleep(1); pathlib.Path({str(marker)!r}).write_text('alive')\"]); "
        "time.sleep(10)"
    )

    job = runner.submit(name="parent", command=[sys.executable, "-c", script])
    deadline = time.time() + 5
    while runner.get(job.id).started_at is None and time.time() < deadline:
        time.sleep(0.01)

    runner.cancel(job.id)
    finished = runner.wait(job.id, timeout=5)
    time.sleep(1.2)

    assert finished.status == CANCELED
    assert not marker.exists()


def test_local_job_runner_reloads_persisted_history(tmp_path: Path) -> None:
    runner = LocalJobRunner(tmp_path / "jobs")

    job = runner.submit(
        name="echo",
        command=[sys.executable, "-c", "print('persisted')"],
        resources=["camera"],
    )
    finished = runner.wait(job.id, timeout=5)
    reloaded = LocalJobRunner(tmp_path / "jobs")

    loaded = reloaded.get(finished.id)
    assert loaded.status == SUCCEEDED
    assert loaded.resources == ["camera"]
    assert "persisted" in reloaded.log_text(finished.id)


def test_local_job_runner_marks_interrupted_jobs_failed_on_reload(tmp_path: Path) -> None:
    job_root = tmp_path / "jobs"
    job_dir = job_root / "orphaned"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "id": "orphaned",
                "name": "sleep",
                "command": [sys.executable, "-c", "import time; time.sleep(10)"],
                "cwd": None,
                "status": QUEUED,
                "created_at": "2026-06-16T00:00:00+00:00",
                "log_path": (job_dir / "log.txt").as_posix(),
                "resources": ["robot"],
                "parameters": {"capture": True},
            }
        )
    )

    reloaded = LocalJobRunner(job_root)
    loaded = reloaded.get("orphaned")

    assert loaded.status == FAILED
    assert loaded.message == "Job runner restarted before this job completed."
    assert loaded.parameters == {"capture": True}


def test_local_job_runner_stops_verified_orphaned_process_group_on_reload(
    tmp_path: Path,
) -> None:
    if os.name == "nt":
        pytest.skip("Process-group recovery uses Linux process metadata")

    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    process_start_time = LocalJobRunner._read_process_start_time(process.pid)
    if process_start_time is None:
        process.kill()
        process.wait()
        pytest.skip("Linux /proc process start metadata is unavailable")

    job_root = tmp_path / "jobs"
    job_dir = job_root / "orphaned"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "id": "orphaned",
                "name": "sleep",
                "command": [sys.executable, "-c", "import time; time.sleep(30)"],
                "cwd": None,
                "status": "running",
                "created_at": "2026-07-10T00:00:00+00:00",
                "log_path": (job_dir / "log.txt").as_posix(),
                "process_pid": process.pid,
                "process_group_id": os.getpgid(process.pid),
                "process_start_time": process_start_time,
                "runner_pid": 999_999_999,
                "runner_start_time": 1,
            }
        )
    )

    try:
        reloaded = LocalJobRunner(job_root)
        loaded = reloaded.get("orphaned")
        process.wait(timeout=5)

        assert loaded.status == FAILED
        assert "orphaned process group was stopped" in loaded.message
        assert process.returncode == -signal.SIGTERM
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_local_job_runner_stops_verified_orphan_from_legacy_terminal_job(
    tmp_path: Path,
) -> None:
    if os.name == "nt":
        pytest.skip("Process-group recovery uses Linux process metadata")

    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    process_start_time = LocalJobRunner._read_process_start_time(process.pid)
    if process_start_time is None:
        process.kill()
        process.wait()
        pytest.skip("Linux /proc process start metadata is unavailable")

    job_root = tmp_path / "jobs"
    job_dir = job_root / "legacy-terminal-orphan"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "id": "legacy-terminal-orphan",
                "name": "sensor-preview:test",
                "command": [sys.executable, "-c", "import time; time.sleep(30)"],
                "cwd": None,
                "status": FAILED,
                "created_at": "2026-07-10T00:00:00+00:00",
                "ended_at": "2026-07-10T00:01:00+00:00",
                "log_path": (job_dir / "log.txt").as_posix(),
                "message": "Job runner restarted before this job completed.",
                "process_pid": process.pid,
                "process_group_id": os.getpgid(process.pid),
                "process_start_time": process_start_time,
                "runner_pid": 999_999_999,
                "runner_start_time": 1,
            }
        )
    )

    try:
        reloaded = LocalJobRunner(job_root)
        loaded = reloaded.get("legacy-terminal-orphan")
        process.wait(timeout=5)

        assert loaded.status == FAILED
        assert loaded.message == "Job runner restarted before this job completed."
        assert "verified orphaned process group" in loaded.tail[-1]
        assert process.returncode == -signal.SIGTERM
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_shutdown_cancels_only_jobs_owned_by_this_runner(tmp_path: Path) -> None:
    runner = LocalJobRunner(tmp_path / "jobs")
    job = runner.submit(
        name="sleep",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
    )
    deadline = time.time() + 5
    while job.id not in runner._processes and time.time() < deadline:
        time.sleep(0.01)

    runner.shutdown(timeout=5)

    assert runner.wait(job.id, timeout=5).status == CANCELED


def test_local_job_runner_rejects_busy_resources(tmp_path: Path) -> None:
    runner = LocalJobRunner(tmp_path / "jobs")

    job = runner.submit(
        name="sleep",
        command=[sys.executable, "-c", "import time; time.sleep(10)"],
        resources=["robot"],
    )
    try:
        with pytest.raises(ResourceBusyError, match="robot held by job"):
            runner.submit(
                name="other",
                command=[sys.executable, "-c", "print('blocked')"],
                resources=["robot"],
            )
        assert runner.resource_holders()["robot"] == job.id
    finally:
        runner.cancel(job.id)


def test_local_job_runner_applies_hierarchical_resource_conflicts(
    tmp_path: Path,
) -> None:
    runner = LocalJobRunner(tmp_path / "jobs")
    preview = runner.submit(
        name="preview",
        command=[sys.executable, "-c", "import time; time.sleep(10)"],
        resources=["camera:realsense_d435:123"],
    )
    try:
        with pytest.raises(ResourceBusyError, match="camera conflicts with"):
            runner.submit(
                name="capture",
                command=[sys.executable, "-c", "print('blocked')"],
                resources=["camera"],
            )
        other = runner.submit(
            name="other-preview",
            command=[sys.executable, "-c", "print('allowed')"],
            resources=["camera:realsense_d435:456"],
        )
        assert runner.wait(other.id, timeout=5).status == SUCCEEDED
    finally:
        runner.cancel(preview.id)
        runner.wait(preview.id, timeout=5)


def test_canceling_job_retains_resource_until_process_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = LocalJobRunner(tmp_path / "jobs")
    job = runner.submit(
        name="sleep",
        command=[sys.executable, "-c", "import time; time.sleep(10)"],
        resources=["camera"],
    )
    deadline = time.time() + 5
    while job.id not in runner._processes and time.time() < deadline:
        time.sleep(0.01)
    process = runner._processes[job.id]
    terminate = runner._terminate_process_group
    monkeypatch.setattr(runner, "_terminate_process_group", lambda _process: None)

    canceled = runner.cancel(job.id)

    assert canceled.status == CANCELING
    assert runner.resource_holders()["camera"] == job.id
    with pytest.raises(ResourceBusyError):
        runner.submit(
            name="blocked",
            command=[sys.executable, "-c", "pass"],
            resources=["camera:realsense_d435:123"],
        )

    terminate(process)
    assert runner.wait(job.id, timeout=5).status == CANCELED
